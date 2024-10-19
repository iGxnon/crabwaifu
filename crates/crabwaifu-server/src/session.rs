use std::io;
use std::sync::Arc;
use std::time::Duration;

use crabml::cpu::CpuTensor;
use crabml_llama2::llama2::Llama2Runner;
use crabwaifu_common::network::{Rx, Tx};
use crabwaifu_common::proto::chat::Message;
use crabwaifu_common::proto::{chat, Packet};
use tokio::sync::{watch, Notify};

use crate::templ::{ChatReplyIterator, ChatTemplate};

pub struct Session<T, R> {
    tx: T,
    rx: R,
    flush_notify: Arc<Notify>,
    close_notify: Arc<Notify>,
    llama_runner: Llama2Runner<CpuTensor<'static>>,
    chat_templ: ChatTemplate,
    default_steps: usize,
}

impl<T: Tx, R: Rx> Session<T, R> {
    pub fn new(
        tx: T,
        rx: R,
        flush_notify: Arc<Notify>,
        close_notify: Arc<Notify>,
        llama_runner: Llama2Runner<CpuTensor<'static>>,
        default_steps: usize,
    ) -> Self {
        Self {
            tx,
            rx,
            flush_notify,
            close_notify,
            chat_templ: ChatTemplate::heuristic_guess(&llama_runner),
            llama_runner,
            default_steps,
        }
    }

    fn generate<'a>(
        runner: &'a mut Llama2Runner<CpuTensor<'static>>,
        prompt: &str,
        stop_mark: &str,
        steps: usize,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<String>> + 'a> {
        log::debug!("completion prompt: `{prompt}`");
        let bos = runner.kv_cache_len() == 0;
        let (pos, _prev_token, token) = runner.prefill(prompt, bos, false)?;
        let inner = runner
            .generate(pos, token, Some(steps))
            .map(|item| anyhow::Ok(item?));
        Ok(ChatReplyIterator::new(inner, vec![stop_mark.to_string()]))
    }

    fn oneshot_completion(&mut self, messages: &[Message], steps: usize) -> anyhow::Result<String> {
        let prompt = self.chat_templ.format(messages);
        let stop_mark = self.chat_templ.stop_mark();
        let content = Self::generate(&mut self.llama_runner, &prompt, stop_mark, steps)?
            .collect::<Result<Vec<_>, _>>()?
            .concat();
        Ok(content)
    }

    #[inline]
    async fn handle_pack(&mut self, pack: Packet) {
        match pack {
            Packet::ChatRequest(request) => {
                log::info!("got ChatRequest");
                let res: anyhow::Result<()> = try {
                    let content = self.oneshot_completion(
                        &request.messages,
                        request.steps.unwrap_or(self.default_steps),
                    )?;
                    self.tx
                        .send_pack(chat::Response {
                            message: Message {
                                role: chat::Role::Assistant,
                                content,
                            },
                        })
                        .await?;
                };
                if let Err(err) = res {
                    log::error!("error: {err}");
                    return;
                }
                // try flush as we completed a whole response
                self.flush_notify.notify_one();
            }
            Packet::ChatStreamRequest(request) => {
                log::info!("got ChatStreamRequest");
                let res: anyhow::Result<()> = try {
                    let mut iter = Self::generate(
                        &mut self.llama_runner,
                        &self.chat_templ.format_prompt(&request.prompt),
                        self.chat_templ.stop_mark(),
                        self.default_steps,
                    )?;
                    for ele in &mut iter {
                        let token = ele?;
                        self.tx
                            .send_pack(chat::StreamResponse {
                                partial: token,
                                eos: false,
                            })
                            .await?;
                    }
                };
                let res = if let Err(err) = res {
                    self.tx.send_pack(chat::StreamResponse {
                        partial: err.to_string(),
                        eos: true,
                    })
                } else {
                    self.tx.send_pack(chat::StreamResponse {
                        partial: String::new(),
                        eos: true,
                    })
                }
                .await;
                if let Err(err) = res {
                    log::error!("error: {err}");
                }
            }
            _ => {
                log::warn!("got unexpected packet on server {pack:?}");
            }
        }
    }

    pub async fn run(mut self, mut shutdown: watch::Receiver<&'static str>) {
        let signal = shutdown.changed();
        tokio::pin!(signal);
        loop {
            tokio::select! {
                res = self.rx.recv_pack() => match res {
                    Ok(pack) => self.handle_pack(pack).await,
                    Err(err) => {
                        if err.kind() == io::ErrorKind::ConnectionAborted {
                            log::info!("connection closed by remote");
                            break;
                        }
                        log::error!("error in recv packets {err}")
                    },
                },
                _ = &mut signal => {
                    break;
                }
            }
        }
        // notify that we'll gonna to shutdown
        self.close_notify.notify_one();
        // wait for flusher response in 10s
        if tokio::time::timeout(Duration::from_secs(10), self.close_notify.notified())
            .await
            .is_err()
        {
            log::warn!("cannot shutdown session background task `flusher` in 10s, skip it");
        }
        // TODO: wait all background task of this session to be done and return
    }
}
