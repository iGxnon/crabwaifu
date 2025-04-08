use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{cmp, mem};

use anyhow::{anyhow, bail};
use crabwaifu_common::network::{spawn_flush_task, tcp_split, Rx, Tx};
use crabwaifu_common::proto::chat::{self, Message};
use crabwaifu_common::proto::{bench, realtime, user, Packet};
use crabwaifu_common::utils::TimeoutWrapper;
use futures::Stream;
use histogram::AtomicHistogram;
use indicatif::ProgressBar;
use raknet_rs::client::{self, ConnectTo};
use rubato::{FftFixedInOut, Resampler};
use tokio::net::{self, TcpSocket};
use tokio::sync::{oneshot, Notify};
use tokio::task::JoinHandle;

const CONSERVATIVE_HEAD_SIZE: usize = 17;

pub struct Client<T, R> {
    tx: T,
    rx: R,
    flush_notify: Arc<Notify>,
    close_notify: Option<oneshot::Sender<bool>>,
    flush_task: Option<JoinHandle<()>>,
    is_raknet: bool,
    mtu: u16,
}

pub async fn tcp_connect_to(
    addr: SocketAddr,
    mtu: u16,
) -> anyhow::Result<Client<impl Tx, impl Rx>> {
    let socket = TcpSocket::new_v4()?;
    let stream = socket.connect(addr).await?;
    let (rx, writer) = tcp_split(stream);
    let (tx, flush_notify, close_notify, flush_task) = spawn_flush_task(writer);
    let rx = Box::pin(rx);
    let client = Client {
        tx,
        rx,
        flush_notify,
        close_notify: Some(close_notify),
        flush_task: Some(flush_task),
        is_raknet: false,
        mtu,
    };
    Ok(client)
}

pub async fn raknet_connect_to(
    addr: SocketAddr,
    config: client::Config,
    mtu: u16,
) -> anyhow::Result<Client<impl Tx, impl Rx>> {
    let (rx, writer) = net::UdpSocket::bind("0.0.0.0:0")
        .await?
        .connect_to(addr, config)
        .await?;
    let (tx, flush_notify, close_notify, flush_task) = spawn_flush_task(writer);
    let rx = Box::pin(rx);
    let client = Client {
        tx,
        rx,
        flush_notify,
        close_notify: Some(close_notify),
        flush_task: Some(flush_task),
        is_raknet: true,
        mtu,
    };
    Ok(client)
}

impl<T: Tx, R: Rx> Client<T, R> {
    pub async fn login(&mut self, username: String, password: String) -> anyhow::Result<()> {
        self.tx
            .send_pack(user::LoginRequest { username, password })
            .await?;
        self.flush_notify.notify_one();
        let pack = self.rx.recv_pack().await.unwrap();
        if let Packet::UserLoginResponse(resp) = pack {
            if resp.success {
                Ok(())
            } else {
                bail!(resp.message);
            }
        } else {
            bail!("response interrupt");
        }
    }

    pub async fn register(&mut self, username: String, password: String) -> anyhow::Result<()> {
        self.tx
            .send_pack(user::RegisterRequest { username, password })
            .await?;
        self.flush_notify.notify_one();
        let pack = self.rx.recv_pack().await.unwrap();
        if let Packet::UserRegisterResponse(resp) = pack {
            if resp.success {
                Ok(())
            } else {
                bail!(resp.message);
            }
        } else {
            bail!("response interrupt");
        }
    }

    pub async fn clear_session(&mut self, model: String) -> anyhow::Result<()> {
        self.tx.send_pack(user::CleanupRequest { model }).await?;
        self.flush_notify.notify_one();
        let pack = self.rx.recv_pack().await.unwrap();
        if let Packet::UserCleanupResponse(resp) = pack {
            if resp.success {
                Ok(())
            } else {
                bail!(resp.message);
            }
        } else {
            bail!("response interrupt");
        }
    }

    pub async fn oneshot(
        &mut self,
        model: String,
        messages: Vec<Message>,
        steps: Option<usize>,
    ) -> anyhow::Result<String> {
        self.tx
            .send_pack(chat::Request {
                model,
                messages,
                steps,
            })
            .await?;
        self.flush_notify.notify_one();
        let pack = self.rx.recv_pack().await?;
        if let Packet::ChatResponse(resp) = pack {
            Ok(resp.message.content)
        } else {
            bail!("response interrupt")
        }
    }

    pub async fn stream(
        &mut self,
        model: String,
        prompt: String,
    ) -> anyhow::Result<impl Stream<Item = anyhow::Result<String>> + '_> {
        self.tx
            .send_pack(chat::StreamRequest { model, prompt })
            .await?;
        self.flush_notify.notify_one();
        let stream = {
            #[futures_async_stream::stream]
            async move {
                loop {
                    let res = self.rx.recv_pack().await;
                    match res {
                        Ok(pack) => {
                            if let Packet::ChatStreamResponse(resp) = pack {
                                if resp.eos {
                                    if !resp.partial.is_empty() {
                                        println!("response error: {}", resp.partial);
                                    }
                                    return;
                                }
                                yield Ok(resp.partial);
                            } else {
                                yield Err(anyhow!("response interrupt"));
                            }
                        }
                        Err(err) => yield Err(err.into()),
                    }
                }
            }
        };
        Ok(stream)
    }

    // first message is user prompt
    pub async fn audio_stream(
        &mut self,
        model: String,
        chunk_raw: Vec<f32>,
        sample_rate: usize,
    ) -> anyhow::Result<impl Stream<Item = anyhow::Result<String>> + '_> {
        // Resample the audio buffer to 16000Hz
        let chunk_size = sample_rate / 100;
        let mut resampler = FftFixedInOut::<f32>::new(sample_rate, 16000, chunk_size, 1)?;
        let mut chunks = Vec::new();
        for chunk in chunk_raw.chunks(chunk_size) {
            // Process in smaller chunks to avoid data loss
            let resampled_chunk = resampler.process(&[chunk], None)?;
            chunks.extend(&resampled_chunk[0]);
        }

        let per_chunk = (self.mtu as usize - CONSERVATIVE_HEAD_SIZE) / mem::size_of::<f32>();
        loop {
            let mut data = chunks.split_off(cmp::min(chunks.len(), per_chunk));
            mem::swap(&mut data, &mut chunks);
            if chunks.is_empty() {
                self.tx
                    .send_pack_with_reliability(
                        realtime::RealtimeAudioChunk {
                            data,
                            eos: Some(model),
                        },
                        raknet_rs::Reliability::Reliable, // TODO: use reliable sequenced
                    )
                    .await?;
                break;
            } else {
                self.tx
                    .send_pack(realtime::RealtimeAudioChunk { data, eos: None })
                    .await?;
            }
        }
        debug_assert!(chunks.is_empty(), "data remains");

        let stream = {
            #[futures_async_stream::stream]
            async move {
                loop {
                    let res = self.rx.recv_pack().await;
                    match res {
                        Ok(pack) => match pack {
                            Packet::ChatResponse(resp) => {
                                yield Ok(resp.message.content);
                            }
                            Packet::ChatStreamResponse(resp) => {
                                if resp.eos {
                                    if !resp.partial.is_empty() {
                                        println!("response error: {}", resp.partial);
                                    }
                                    return;
                                }
                                yield Ok(resp.partial);
                            }
                            _ => {
                                yield Err(anyhow!("response interrupt"));
                            }
                        },
                        Err(err) => yield Err(err.into()),
                    }
                }
            }
        };

        Ok(stream)
    }

    pub async fn bench_unreliable(
        &mut self,
        received: usize,
        delay_histogram: &AtomicHistogram,
    ) -> anyhow::Result<String> {
        let start_at = Instant::now();
        let mut last_recv_at: Option<Instant> = None;
        let mut actual_received = 0;
        self.tx
            .send_pack(bench::UnreliableRequest {
                data_len: received,
                mtu: self.mtu as usize,
            })
            .await?;
        let bar = ProgressBar::new(received as u64);
        loop {
            let pack = self.rx.recv_pack().await?;
            if let Some(last_recv_at) = last_recv_at {
                delay_histogram.increment(last_recv_at.elapsed().as_nanos() as u64)?;
            }
            last_recv_at = Some(Instant::now());
            match pack {
                Packet::BenchUnreliableRequest(_) => break, // EOF
                Packet::BenchUnreliableResponse(res) => {
                    actual_received += res.data_partial.len();
                    bar.inc(res.data_partial.len() as u64);
                }
                _ => bail!("interrupt pack {pack:?}"),
            }
        }
        bar.finish_and_clear();
        let dur = start_at.elapsed();
        Ok(format!(
            "expect received\t{}\nactual received\t{}\nlost ratio\t{}\ncost\t{:?}\nrecv rate\t{}",
            bytes(received),
            bytes(actual_received),
            1.0 - actual_received as f64 / received as f64,
            dur,
            rate(actual_received, dur)
        ))
    }

    pub async fn bench_commutative(
        &mut self,
        received: usize,
        batch_size: usize,
        delay_histogram: &AtomicHistogram,
    ) -> anyhow::Result<String> {
        let start_at = Instant::now();
        let mut last_recv_at: Option<Instant> = None;
        self.tx
            .send_pack(bench::CommutativeRequest {
                data_len: received,
                batch_size,
            })
            .await?;
        let mut total_received = 0;
        let bar = ProgressBar::new(received as u64);
        loop {
            let pack = self.rx.recv_pack().await?;
            if let Some(last_recv_at) = last_recv_at {
                delay_histogram.increment(last_recv_at.elapsed().as_nanos() as u64)?;
            }
            last_recv_at = Some(Instant::now());
            if let Packet::BenchCommutativeResponse(res) = pack {
                total_received += res.data_partial.len();
                bar.inc(res.data_partial.len() as u64);
            } else {
                bail!("interrupt pack {pack:?}")
            }
            if total_received == received {
                break;
            }
        }
        bar.finish_and_clear();
        let dur = start_at.elapsed();
        Ok(format!(
            "received\t{}\nbatch size\t{}\ncost\t{:?}\nrecv rate\t{}",
            bytes(received),
            bytes(batch_size),
            dur,
            rate(total_received, dur)
        ))
    }

    pub async fn bench_ordered(
        &mut self,
        received: usize,
        batch_size: usize,
        delay_histogram: &AtomicHistogram,
    ) -> anyhow::Result<String> {
        let start_at = Instant::now();
        let mut last_recv_at: Option<Instant> = None;
        self.tx
            .send_pack(bench::OrderedRequest {
                data_len: received,
                batch_size,
            })
            .await?;
        let mut total_received = 0;
        let mut expect_index = 0;
        let bar = ProgressBar::new(received as u64);
        loop {
            let pack = self.rx.recv_pack().await?;
            if let Some(last_recv_at) = last_recv_at {
                delay_histogram.increment(last_recv_at.elapsed().as_nanos() as u64)?;
            }
            last_recv_at = Some(Instant::now());
            if let Packet::BenchOrderedResponse(res) = pack {
                total_received += res.data_partial.len();
                bar.inc(res.data_partial.len() as u64);
                assert_eq!(res.index, expect_index); // ordered
                expect_index += 1;
            } else {
                bail!("interrupt pack {pack:?}")
            }
            if total_received == received {
                break;
            }
        }
        bar.finish_and_clear();
        let dur = start_at.elapsed();
        Ok(format!(
            "received\t{}\nbatch size\t{}\ncost\t{:?}\nrecv rate\t{}",
            bytes(received),
            bytes(batch_size),
            dur,
            rate(total_received, dur)
        ))
    }

    pub async fn finish(&mut self) {
        let shutdown_timeout = Duration::from_secs(5);
        self.close_notify
            .take()
            .unwrap()
            .send(self.is_raknet) // TCP handle 2MSL in the kernel, not here
            .unwrap();
        let _ = self
            .flush_task
            .take()
            .unwrap()
            .timeout(shutdown_timeout)
            .await;
    }
}

fn rate(bytes: usize, duration: Duration) -> String {
    let per_second = Duration::from_secs(1).as_nanos() as f64 / duration.as_nanos() as f64;
    let bits_per_second = (bytes as f64 * 8.0 * per_second) as u64;

    use humansize::{format_size, DECIMAL};
    let value = format_size(bits_per_second, DECIMAL.space_after_value(false));
    let value = value.trim_end_matches('B');

    format!("{value}bps")
}

fn bytes(value: usize) -> String {
    use humansize::{format_size, DECIMAL};
    format_size(value, DECIMAL.space_after_value(false))
}
