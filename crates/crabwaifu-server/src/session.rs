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

// Safety: we ensure that llama_runner does not reside in multiple threads
unsafe impl<T: Send, R: Send> Send for Session<T, R> {}

impl<T, R> Drop for Session<T, R> {
    fn drop(&mut self) {
        // close flusher on drop
        self.close_notify.notify_waiters();
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

    async fn completion(&mut self, messages: &[Message], steps: usize) -> String {
        let prompt = self.chat_templ.format(messages);
        self.llama_runner
            .prefill_and_generate(&prompt, steps)
            .expect("prefill error")
            .collect::<Result<Vec<_>, _>>()
            .expect("generate error")
            .concat()
    }

    #[inline]
    async fn dispatch_pack(&mut self, pack: Packet) {
        match pack {
            Packet::ChatRequest(request) => {
                let content = self
                    .completion(
                        &request.messages,
                        request.steps.unwrap_or(self.default_steps),
                    )
                    .await;
                if let Err(err) = self
                    .tx
                    .send_pack(chat::Response {
                        message: Message {
                            role: chat::Role::Assistant,
                            content,
                        },
                    })
                    .await
                {
                    log::error!("send response error: {err}");
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
                    Ok(pack) => self.dispatch_pack(pack).await,
                    Err(err) => {
                        log::error!("error in recv packets {err}")
                    },
                },
                _ = &mut signal => {
                    // TODO: wait all task of this session to be done and return
                    break;
                }
            }
        }
    }
}
