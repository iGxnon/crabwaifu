use std::sync::Arc;
use std::time::Duration;
use std::{cmp, io, mem};

use crabml::cpu::CpuTensor;
use crabml_llama2::llama2::Llama2Runner;
use crabwaifu_common::network::{Rx, Tx};
use crabwaifu_common::proto::chat::Message;
use crabwaifu_common::proto::{bench, chat, realtime, user, Packet};
use crabwaifu_common::utils::TimeoutWrapper;
use tokio::sync::{oneshot, watch, Notify};
use tokio::task::JoinHandle;
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

use crate::db;
use crate::templ::{ChatReplyIterator, ChatTemplate};

const CONSERVATIVE_HEAD_SIZE: usize = 17;

struct UserContext {
    user_id: i32,
    context: Vec<Message>,
    init_len: usize,
}

pub struct Session<T, R> {
    tx: T,
    rx: R,
    flush_notify: Arc<Notify>,
    llama_runner: Llama2Runner<CpuTensor<'static>>,
    chat_templ: ChatTemplate,
    default_steps: usize,
    whisper_runner: WhisperState,
    language: String,
    close_notify: Option<oneshot::Sender<bool>>,
    flusher_task: Option<JoinHandle<()>>,
    user_context: Option<UserContext>,
    audio_buffer: Vec<f32>,
}

impl<T: Tx, R: Rx> Session<T, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tx: T,
        rx: R,
        flush_notify: Arc<Notify>,
        close_notify: oneshot::Sender<bool>,
        llama_runner: Llama2Runner<CpuTensor<'static>>,
        default_steps: usize,
        whisper_runner: WhisperState,
        language: String,
        flusher_task: JoinHandle<()>,
    ) -> Self {
        Self {
            tx,
            rx,
            flush_notify,
            chat_templ: ChatTemplate::heuristic_guess(&llama_runner),
            llama_runner,
            default_steps,
            whisper_runner,
            language,
            close_notify: Some(close_notify),
            flusher_task: Some(flusher_task),
            user_context: None,
            audio_buffer: Vec::new(),
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
                        role: chat::Role::User,
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
            if let Some(context) = &mut self.user_context {
                // save context in memory
                context.context.push(Message {
                    role: chat::Role::User,
                    content: request.prompt,
                });
            }
            let mut response = String::new();
            for ele in &mut iter {
                let token = ele?;
                response.push_str(&token);
                self.tx
                    .send_pack(chat::StreamResponse {
                        partial: token,
                        eos: false,
                    })
                    .await?;
                // blocking iterator, random yield to other tasks
                random_yield().await;
            }
            drop(iter);
            log::debug!("chat response: {response}");
            if let Some(context) = &mut self.user_context {
                // save context in memory
                context.context.push(Message {
                    role: chat::Role::Assistant,
                    content: response,
                });
            }
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
        let max_len = request.mtu - CONSERVATIVE_HEAD_SIZE;
        let parts = request.data_len.div_ceil(max_len);
        let mut data = vec![0; request.data_len];
        let mut res = try {
            for _ in 0..parts {
                let mut data_partial = data.split_off(cmp::min(max_len, data.len()));
                mem::swap(&mut data, &mut data_partial);
                self.tx
                    .send_pack(bench::UnreliableResponse { data_partial })
                    .await?;
                random_yield().await;
            }
            debug_assert!(data.is_empty(), "data remains");
        };
        if let Err(err) = res {
            log::error!("transfer data: {err}");
        }
        // a reliable packet sent as EOF
        res = self
            .tx
            .send_pack(bench::UnreliableRequest {
                data_len: 0,
                mtu: 0,
            })
            .await;
        if let Err(err) = res {
            log::error!("error: {err}");
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
                random_yield().await;
            }
            debug_assert!(data.is_empty(), "data remains");
        };
        if let Err(err) = res {
            log::error!("error: {err}");
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
                random_yield().await;
            }
            debug_assert!(data.is_empty(), "data remains");
        };
        if let Err(err) = res {
            log::error!("error: {err}");
        }
    }

    #[inline]
    async fn handle_login(&mut self, request: user::LoginRequest) {
        let username = request.username;
        let pwd_raw = request.password;
        let res: anyhow::Result<()> = try {
            let res: Result<(String, i32), sqlx::Error> =
                sqlx::query_as("SELECT pass_hash, id FROM users WHERE username = $1")
                    .bind(username.clone())
                    .fetch_one(db::pool())
                    .await;
            let (pwd_hash, user_id) = match res {
                Ok(res) => res,
                Err(err) => {
                    self.tx
                        .send_pack(user::LoginResponse {
                            success: false,
                            message: if err.to_string().contains("no rows") {
                                "User not found".to_string()
                            } else {
                                format!("Database error: {err}")
                            },
                            context: Vec::new(),
                        })
                        .await?;
                    return;
                }
            };
            if db::pbkdf2().verify(pwd_raw, pwd_hash) {
                // query user context
                let (mut context_str,): (String,) =
                    sqlx::query_as("SELECT context FROM history WHERE user_id = $1")
                        .bind(user_id)
                        .fetch_one(db::pool())
                        .await
                        .unwrap_or_default(); // may not exists
                if context_str.is_empty() {
                    context_str.push_str("[]");
                }
                let context: Vec<Message> = serde_json::from_str(&context_str)?;
                if !context.is_empty() {
                    // load user context
                    let _ignore = self.oneshot_completion(&context, self.default_steps)?;
                }
                // return to user
                self.tx
                    .send_pack(user::LoginResponse {
                        success: true,
                        message: String::new(),
                        context: context.clone(),
                    })
                    .await?;
                // save user context to memory
                self.user_context = Some(UserContext {
                    init_len: context.len(),
                    user_id,
                    context,
                });
            } else {
                self.tx
                    .send_pack(user::LoginResponse {
                        success: false,
                        message: "Password incorrect".to_string(),
                        context: Vec::new(),
                    })
                    .await?;
            }
        };
        if let Err(err) = res {
            log::error!("error: {err}");
        }
    }

    #[inline]
    async fn handle_register(&mut self, request: user::RegisterRequest) {
        let username = request.username;
        let pwd_raw = request.password;
        let res: anyhow::Result<()> = try {
            let failed = sqlx::query("INSERT INTO users (username, pass_hash) VALUES ($1, $2)")
                .bind(username)
                .bind(db::pbkdf2().key(pwd_raw))
                .execute(db::pool())
                .await
                .is_err();
            if failed {
                self.tx
                    .send_pack(user::RegisterResponse {
                        success: false,
                        message: "User already exists".to_string(),
                    })
                    .await?
            } else {
                self.tx
                    .send_pack(user::RegisterResponse {
                        success: true,
                        message: String::new(),
                    })
                    .await?
            }
        };
        if let Err(err) = res {
            log::error!("error: {err}");
        }
    }

    #[inline]
    async fn handle_realtime_audio(&mut self, request: realtime::RealtimeAudioChunk) {
        self.audio_buffer.extend(request.data);
        if !request.eos {
            return;
        };
        log::info!("got RealtimeAudioChunk({})", self.audio_buffer.len());
        let audio_buffer = self.audio_buffer.split_off(0);

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // and set the language to translate to to english
        params.set_language(Some(&self.language));
        // we also explicitly disable anything that prints to stdout
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let res: anyhow::Result<()> = try {
            self.whisper_runner.full(params, &audio_buffer[..])?;
            let num_segments = self.whisper_runner.full_n_segments()?;
            let mut content = String::new();
            for i in 0..num_segments {
                let segment = self.whisper_runner.full_get_segment_text(i)?;
                let start_timestamp = self.whisper_runner.full_get_segment_t0(i)?;
                let end_timestamp = self.whisper_runner.full_get_segment_t1(i)?;
                log::info!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
                content.push_str(&segment);
            }
            self.tx
                .send_pack(chat::Response {
                    message: Message {
                        role: chat::Role::User,
                        content: content.clone(),
                    },
                })
                .await?;
            self.flush_notify.notify_one();
            tokio::task::yield_now().await;
            self.handle_chat_stream(chat::StreamRequest { prompt: content })
                .await;
        };

        if let Err(err) = res {
            log::error!("error: {err}");
        }
    }

    #[inline]
    async fn handle_user_cleanup(&mut self, _: user::CleanupRequest) {
        let res: anyhow::Result<()> = try {
            let Some(context) = &self.user_context else {
                self.tx
                    .send_pack(user::CleanupResponse {
                        success: true,
                        message: "User not logged in".to_string(),
                    })
                    .await?;
                return;
            };
            let user_id = context.user_id;
            self.user_context = Some(UserContext {
                user_id,
                context: Vec::new(),
                init_len: 0,
            });
            self.tx
                .send_pack(user::CleanupResponse {
                    success: true,
                    message: String::new(),
                })
                .await?;
        };
        if let Err(err) = res {
            log::error!("error: {err}");
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
            Packet::UserRegisterRequest(request) => {
                log::info!("got UserRegisterRequest");
                self.handle_register(request).await;
            }
            Packet::UserLoginRequest(request) => {
                log::info!("got UserLoginRequest");
                self.handle_login(request).await;
            }
            Packet::UserCleanupRequest(request) => {
                log::info!("got UserCleanupRequest");
                self.handle_user_cleanup(request).await;
            }
            Packet::RealtimeAudioChunk(request) => {
                self.handle_realtime_audio(request).await;
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
        // save user context
        if let Some(context) = &self.user_context
            && (context.init_len != context.context.len() || context.context.is_empty())
        {
            let context_str = serde_json::to_string(&context.context).unwrap_or_default();
            let res = sqlx::query("INSERT INTO history (user_id, context) VALUES ($1, $2) ON CONFLICT (user_id) DO UPDATE SET context = $2")
                .bind(context.user_id)
                .bind(context_str)
                .execute(db::pool())
                .await;
            if let Err(err) = res {
                log::error!("error saving user context {err}");
            }
        }

        let wait_timeout = Duration::from_secs(10);
        self.close_notify.take().unwrap().send(wait).unwrap();
        let _ = self
            .flusher_task
            .take()
            .unwrap()
            .timeout(wait_timeout)
            .await;
        log::info!("session shutdown");
    }
}

async fn random_yield() {
    if rand::random_bool(0.5) {
        tokio::task::yield_now().await;
    }
}
