use std::sync::Arc;
use std::time::Duration;
use std::{cmp, io, mem};

use crabml::cpu::CpuTensor;
use crabml_llama2::llama2::Llama2Runner;
use crabwaifu_common::network::{Rx, Tx};
use crabwaifu_common::proto::chat::Message;
use crabwaifu_common::proto::{bench, chat, Packet};
use crabwaifu_common::utils::TimeoutWrapper;
use tokio::sync::{watch, Notify};
use tokio::task::JoinHandle;

use crate::templ::{ChatReplyIterator, ChatTemplate};

const CONSERVATIVE_HEAD_SIZE: usize = 17;

pub struct Session<T, R> {
    tx: T,
    rx: R,
    flush_notify: Arc<Notify>,
    close_notify: Arc<Notify>,
    llama_runner: Llama2Runner<CpuTensor<'static>>,
    chat_templ: ChatTemplate,
    default_steps: usize,
    mtu: usize,
    flush_task: JoinHandle<()>,
}

impl<T, R> Drop for Session<T, R> {
    fn drop(&mut self) {
        self.flush_task.abort();
    }
}

impl<T: Tx, R: Rx> Session<T, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tx: T,
        rx: R,
        flush_notify: Arc<Notify>,
        close_notify: Arc<Notify>,
        llama_runner: Llama2Runner<CpuTensor<'static>>,
        default_steps: usize,
        mtu: usize,
        flusher_task: JoinHandle<()>,
    ) -> Self {
        Self {
            tx,
            rx,
            flush_notify,
            close_notify,
            chat_templ: ChatTemplate::heuristic_guess(&llama_runner),
            llama_runner,
            default_steps,
            mtu,
            flush_task: flusher_task,
        }
    }

    fn generate<'a>(
        runner: &'a mut Llama2Runner<CpuTensor<'static>>,
        prompt: &str,
        stop_mark: &str,
        steps: usize,
        has_stop_mark: &'a mut bool,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<String>> + 'a> {
        log::debug!("completing prompt: `{prompt}`");
        let bos = runner.kv_cache_len() == 0;
        let (pos, _prev_token, token) = runner.prefill(prompt, bos, false)?;
        let inner = runner
            .generate(pos, token, Some(steps))
            .map(|item| anyhow::Ok(item?));
        Ok(ChatReplyIterator::new(
            inner,
            vec![stop_mark.to_string()],
            has_stop_mark,
        ))
    }

    fn oneshot_completion(&mut self, messages: &[Message], steps: usize) -> anyhow::Result<String> {
        let prompt = self.chat_templ.format(messages);
        let stop_mark = self.chat_templ.stop_mark();
        let mut has_stop_mark = false;
        let content = Self::generate(
            &mut self.llama_runner,
            &prompt,
            stop_mark,
            steps,
            &mut has_stop_mark,
        )?
        .collect::<Result<Vec<_>, _>>()?
        .concat();
        if !has_stop_mark {
            log::debug!("appended stop mark: {stop_mark}");
            self.llama_runner.prefill(stop_mark, false, false)?;
        }
        Ok(content)
    }

    #[inline]
    async fn handle_chat(&mut self, request: chat::Request) {
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

    #[inline]
    async fn handle_chat_stream(&mut self, request: chat::StreamRequest) {
        let res: anyhow::Result<()> = try {
            let mut has_stop_mark = false;
            let stop_mark = self.chat_templ.stop_mark();
            let mut iter = Self::generate(
                &mut self.llama_runner,
                &self.chat_templ.format_prompt(&request.prompt),
                stop_mark,
                self.default_steps,
                &mut has_stop_mark,
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
            drop(iter);
            if !has_stop_mark {
                log::debug!("appended stop mark: {stop_mark}");
                self.llama_runner.prefill(stop_mark, false, false)?;
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

    #[inline]
    async fn handle_bench_unreliable(&mut self, request: bench::UnreliableRequest) {
        let max_len = self.mtu - CONSERVATIVE_HEAD_SIZE;
        let mut data = vec![0; request.data_len];
        let mut res = if request.data_len > max_len {
            try {
                let parts = request.data_len.div_ceil(max_len);
                for _ in 0..parts {
                    let mut data_partial = data.split_off(cmp::min(max_len, data.len()));
                    mem::swap(&mut data, &mut data_partial);
                    self.tx
                        .send_pack(bench::UnreliableResponse { data_partial })
                        .await?
                }
                debug_assert!(data.is_empty(), "data remains");
            }
        } else {
            self.tx
                .send_pack(bench::UnreliableResponse { data_partial: data })
                .await
        };
        if let Err(err) = res {
            log::error!("transfer data: {err}");
        }
        // a reliable packet sent as EOF
        res = self
            .tx
            .send_pack(bench::UnreliableRequest { data_len: 0 })
            .await;
        if let Err(err) = res {
            log::error!("finish data: {err}");
        }
    }

    #[inline]
    async fn handle_bench_commutative(&mut self, request: bench::CommutativeRequest) {
        let per_len = request.batch_size;
        let parts = request.data_len.div_ceil(per_len);
        let mut data = vec![0; request.data_len];
        let res: anyhow::Result<()> = try {
            for _ in 0..parts {
                let mut data_partial = data.split_off(cmp::min(per_len, data.len()));
                mem::swap(&mut data, &mut data_partial);
                self.tx
                    .send_pack(bench::CommutativeResponse { data_partial })
                    .await?;
            }
        };
        if let Err(err) = res {
            log::error!("transfer data: {err}");
        }
    }

    #[inline]
    async fn handle_bench_ordered(&mut self, request: bench::OrderedRequest) {
        let per_len = request.batch_size;
        let parts = request.data_len.div_ceil(per_len);
        let mut data = vec![0; request.data_len];
        let res: anyhow::Result<()> = try {
            for index in 0..parts {
                let mut data_partial = data.split_off(cmp::min(per_len, data.len()));
                mem::swap(&mut data, &mut data_partial);
                self.tx
                    .send_pack(bench::OrderedResponse {
                        data_partial,
                        index,
                    })
                    .await?;
            }
        };
        if let Err(err) = res {
            log::error!("transfer data: {err}");
        }
    }

    #[inline]
    async fn handle_pack(&mut self, pack: Packet) {
        match pack {
            Packet::ChatRequest(request) => {
                log::info!("got ChatRequest");
                self.handle_chat(request).await;
            }
            Packet::ChatStreamRequest(request) => {
                log::info!("got ChatStreamRequest");
                self.handle_chat_stream(request).await;
            }
            Packet::BenchUnreliableRequest(request) => {
                log::info!("got BenchUnreliableRequest");
                self.handle_bench_unreliable(request).await;
            }
            Packet::BenchCommutativeRequest(request) => {
                log::info!("got BenchCommutativeRequest");
                self.handle_bench_commutative(request).await;
            }
            Packet::BenchOrderedRequest(request) => {
                log::info!("got BenchOrderedRequest");
                self.handle_bench_ordered(request).await;
            }
            _ => {
                log::warn!("got unexpected packet on server {pack:?}");
            }
        }
    }

    pub async fn run(mut self, mut shutdown: watch::Receiver<&'static str>) {
        let signal = shutdown.changed();
        tokio::pin!(signal);
        let wait = loop {
            tokio::select! {
                res = self.rx.recv_pack() => match res {
                    Ok(pack) => self.handle_pack(pack).await,
                    Err(err) => {
                        if err.kind() == io::ErrorKind::ConnectionAborted {
                            log::info!("connection closed by remote");
                            break false;
                        }
                        log::error!("error in recv packets {err}")
                    },
                },
                _ = &mut signal => {
                    break true;
                }
            }
        };
        let wait_timeout = Duration::from_secs(10);
        // notify that we'll gonna to shutdown
        self.close_notify.notify_one();
        // wait for flusher closing
        let _ = self.close_notify.notified().timeout(wait_timeout).await;
        // if we shutdown the server, then wait for the last 2MSL
        if wait {
            log::info!("wait 2MSL...");
            let _ = self.close_notify.notified().timeout(wait_timeout).await;
        }
        log::info!("session shutdown");
        // TODO: wait all background task of this session to be done and return
    }
}
