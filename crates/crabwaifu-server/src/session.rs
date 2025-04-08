use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::{cmp, io, mem};

use crabml::cpu::CpuTensor;
use crabml_llama2::llama2::Llama2Runner;
use crabwaifu_common::network::{Rx, Tx};
use crabwaifu_common::proto::chat::Message;
use crabwaifu_common::proto::{bench, chat, realtime, user, Packet};
use crabwaifu_common::utils::TimeoutWrapper;
use kokoros::tts::koko::TTSKoko;
use tokio::sync::{oneshot, watch, Notify};
use tokio::task::JoinHandle;
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

use crate::db;
use crate::templ::{ChatTemplate, ReplyIterator};

const CONSERVATIVE_HEAD_SIZE: usize = 17;
const DEFAULT_STEPS: usize = 512;

pub struct Session<T, R> {
    tx: T,
    rx: R,
    flush_notify: Arc<Notify>,
    llm_states: HashMap<String, LLMState>,
    whisper_state: WhisperState,
    tts_state: TTState,
    close_notify: Option<oneshot::Sender<bool>>,
    flusher_task: Option<JoinHandle<()>>,
    audio_buffer: Vec<f32>,
}

pub struct LLMState {
    runner: Llama2Runner<CpuTensor<'static>>,
    chat_templ: ChatTemplate,
}

impl LLMState {
    pub fn new(runner: Llama2Runner<CpuTensor<'static>>) -> Self {
        Self {
            chat_templ: ChatTemplate::heuristic_guess(&runner),
            runner,
        }
    }

    fn generate<'a>(
        &'a mut self,
        prompt: &str,
        steps: usize,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<String>> + 'a> {
        let stop_mark = self.chat_templ.stop_mark().to_owned();
        let bos = self.runner.kv_cache_len() == 0;
        let (pos, _prev_token, token) =
            self.runner
                .prefill(&self.chat_templ.format_prompt(prompt), bos, false)?;
        let inner = self
            .runner
            .generate(pos, token, Some(steps))
            .map(|item| anyhow::Ok(item?));
        let iter = ReplyIterator::new(inner, stop_mark);
        Ok(iter)
    }

    fn oneshot_completion(&mut self, messages: &[Message], steps: usize) -> anyhow::Result<String> {
        self.runner.reset_kv_cache();
        let prompt = self.chat_templ.format(messages);
        let stop_mark = self.chat_templ.stop_mark().to_owned();
        let inner = self
            .runner
            .prefill_and_generate(&prompt, steps)?
            .map(|item| anyhow::Ok(item?));
        let iter = ReplyIterator::new(inner, stop_mark);
        let content = iter.collect::<Result<Vec<_>, _>>()?.concat();
        Ok(content)
    }
}

pub struct TTState {
    tts_koko: TTSKoko,
    style: String,
    speed: f32,
}

impl TTState {
    pub fn new(tts_koko: TTSKoko, style: String, speed: f32) -> Self {
        Self {
            tts_koko,
            style,
            speed,
        }
    }

    /// Generate raw audio from text
    fn tts_raw_audio(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let res = self
            .tts_koko
            .tts_raw_audio(text, "en-us", &self.style, self.speed, None)
            .map_err(|_| anyhow::anyhow!("tts error"))?;
        Ok(res)
    }
}

struct SentenceMatcher {
    buffer: String,
}

impl SentenceMatcher {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    fn match_sentence(&mut self, text: &str) -> Option<String> {
        self.buffer.push_str(text);
        for comma in [
            ". ", ", ", "! ", "? ", ": ", ".\n", ",\n", "!\n", "?\n", ":\n",
        ]
        .iter()
        {
            if let Some(pos) = self.buffer.find(comma) {
                let mut sentence = self.buffer.split_off(pos + comma.len());
                mem::swap(&mut self.buffer, &mut sentence);
                // remove sign
                sentence = sentence
                    .chars()
                    .skip_while(|c| c.is_ascii_punctuation())
                    .collect();
                return Some(sentence);
            }
        }
        None
    }

    fn remaining(&self) -> &str {
        &self.buffer
    }
}

impl<T: Tx, R: Rx> Session<T, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tx: T,
        rx: R,
        flush_notify: Arc<Notify>,
        close_notify: oneshot::Sender<bool>,
        llm_states: HashMap<String, LLMState>,
        whisper_state: WhisperState,
        tts_state: TTState,
        flusher_task: JoinHandle<()>,
    ) -> Self {
        Self {
            tx,
            rx,
            flush_notify,
            llm_states,
            whisper_state,
            tts_state,
            close_notify: Some(close_notify),
            flusher_task: Some(flusher_task),
            audio_buffer: Vec::new(),
        }
    }

    #[inline]
    async fn handle_chat(&mut self, request: chat::Request) {
        let res: anyhow::Result<()> = try {
            let message = self
                .llm_states
                .get_mut(&request.model)
                .ok_or(anyhow::anyhow!("no such model"))?
                .oneshot_completion(&request.messages, request.steps.unwrap_or(DEFAULT_STEPS))?;
            self.tx.send_pack(chat::Response { message }).await?;
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
            let mut iter = self
                .llm_states
                .get_mut(&request.model)
                .ok_or(anyhow::anyhow!("no such model"))?
                .generate(&request.prompt, DEFAULT_STEPS)?;
            let mut matcher = SentenceMatcher::new();
            let mut text = String::new();
            for ele in &mut iter {
                let token = ele?;
                text.push_str(&token);
                if let Some(sentence) = matcher.match_sentence(&token) {
                    log::debug!("got sentence: {}", sentence);
                }
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

            log::debug!("eos: {}", matcher.remaining());
            if request.voice {
                let audio = self.tts_state.tts_raw_audio(&text)?;
                let (_stream, stream_handle) = rodio::OutputStream::try_default().unwrap();
                let sink = rodio::Sink::try_new(&stream_handle).unwrap();
                let source = rodio::buffer::SamplesBuffer::new(1, 24000, audio);
                // Play the audio
                sink.append(source);
                sink.sleep_until_end();
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
            let (pwd_hash, _user_id) = match res {
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
                        })
                        .await?;
                    return;
                }
            };
            if db::pbkdf2().verify(pwd_raw, pwd_hash) {
                // return to user
                self.tx
                    .send_pack(user::LoginResponse {
                        success: true,
                        message: String::new(),
                    })
                    .await?;
            } else {
                self.tx
                    .send_pack(user::LoginResponse {
                        success: false,
                        message: "Password incorrect".to_string(),
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
        let Some((model, voice)) = request.eos else {
            return;
        };
        log::info!("got RealtimeAudioChunk({})", self.audio_buffer.len());
        let audio_buffer = self.audio_buffer.split_off(0);

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // we also explicitly disable anything that prints to stdout
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let res: anyhow::Result<()> = try {
            self.whisper_state.full(params, &audio_buffer[..])?;
            let num_segments = self.whisper_state.full_n_segments()?;
            let mut message = String::new();
            for i in 0..num_segments {
                let segment = self.whisper_state.full_get_segment_text(i)?;
                let start_timestamp = self.whisper_state.full_get_segment_t0(i)?;
                let end_timestamp = self.whisper_state.full_get_segment_t1(i)?;
                log::info!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
                message.push_str(&segment);
            }
            self.tx
                .send_pack(chat::Response {
                    message: message.clone(),
                })
                .await?;
            self.flush_notify.notify_one();
            tokio::task::yield_now().await;
            self.handle_chat_stream(chat::StreamRequest {
                model,
                prompt: message,
                voice,
            })
            .await;
        };

        if let Err(err) = res {
            log::error!("error: {err}");
        }
    }

    #[inline]
    async fn handle_user_cleanup(&mut self, _: user::CleanupRequest) {
        let res: anyhow::Result<()> = try {
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
