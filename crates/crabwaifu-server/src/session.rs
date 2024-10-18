use std::sync::Arc;

use crabml::cpu::CpuTensor;
use crabml_llama2::llama2::Llama2Runner;
use crabwaifu_common::network::{Rx, Tx};
use crabwaifu_common::proto::chat::Message;
use crabwaifu_common::proto::{chat, Packet};
use tokio::sync::{watch, Notify};

use crate::templ::ChatTemplate;

pub struct Session<T, R> {
    tx: T,
    rx: R,
    flush_notify: Arc<Notify>,
    close_notify: Arc<Notify>,
    llama_runner: Llama2Runner<CpuTensor<'static>>,
    chat_templ: ChatTemplate,
    default_steps: usize,
}

impl<T, R> Drop for Session<T, R> {
    fn drop(&mut self) {
        // notify the flusher
        self.close_notify.notify_one();
    }
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

    async fn completion(&mut self, messages: &[Message], steps: usize) -> anyhow::Result<String> {
        log::info!("processing completion...");
        let prompt = self.chat_templ.format(messages);
        log::trace!("completion prompt: `{prompt}`");
        let content = self
            .llama_runner
            .prefill_and_generate(&prompt, steps)?
            .collect::<Result<Vec<_>, _>>()?
            .concat();
        let stop_mark = self.chat_templ.stop_mark();
        let mut need_stop_mark = false;
        let trimmed = content
            .split_once(stop_mark)
            .map(|(valid, _)| {
                need_stop_mark = true;
                valid.to_string()
            })
            .unwrap_or(content);
        if need_stop_mark {
            log::trace!("appending stop mark {stop_mark}");
            self.llama_runner.prefill(stop_mark, false, false)?;
        }
        Ok(trimmed)
    }

    #[inline]
    async fn handle_pack(&mut self, pack: Packet) {
        match pack {
            Packet::ChatRequest(request) => {
                let res: anyhow::Result<()> = try {
                    let content = self
                        .completion(
                            &request.messages,
                            request.steps.unwrap_or(self.default_steps),
                        )
                        .await?;
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
                        log::error!("error in recv packets {err}")
                    },
                },
                _ = &mut signal => {
                    // TODO: wait all background task of this session to be done and return
                    break;
                }
            }
        }
    }
}
