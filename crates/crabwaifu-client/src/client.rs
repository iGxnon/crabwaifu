use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail};
use crabwaifu_common::network::{spawn_flush_task, tcp_split, Rx, Tx};
use crabwaifu_common::proto::chat::{self, Message};
use crabwaifu_common::proto::{bench, Packet};
use crabwaifu_common::utils::TimeoutWrapper;
use futures::Stream;
use histogram::AtomicHistogram;
use raknet_rs::client::{self, ConnectTo};
use tokio::net::{self, TcpSocket};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

pub struct Client<T, R> {
    tx: T,
    rx: R,
    flush_notify: Arc<Notify>,
    close_notify: Arc<Notify>,
    flush_task: JoinHandle<()>,
    is_raknet: bool,
}

impl<T, R> Drop for Client<T, R> {
    fn drop(&mut self) {
        self.flush_task.abort();
    }
}

pub async fn tcp_connect_to(addr: SocketAddr) -> anyhow::Result<Client<impl Tx, impl Rx>> {
    let socket = TcpSocket::new_v4()?;
    let stream = socket.connect(addr).await?;
    let (rx, writer) = tcp_split(stream);
    let (tx, flush_notify, close_notify, flush_task) = spawn_flush_task(writer);
    let rx = Box::pin(rx);
    let client = Client {
        tx,
        rx,
        flush_notify,
        close_notify,
        flush_task,
        is_raknet: false,
    };
    Ok(client)
}

pub async fn raknet_connect_to(
    addr: SocketAddr,
    config: client::Config,
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
        close_notify,
        flush_task,
        is_raknet: true,
    };
    Ok(client)
}

impl<T: Tx, R: Rx> Client<T, R> {
    pub async fn oneshot(
        &mut self,
        messages: Vec<Message>,
        steps: Option<usize>,
    ) -> anyhow::Result<String> {
        self.tx.send_pack(chat::Request { messages, steps }).await?;
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
        prompt: String,
    ) -> anyhow::Result<impl Stream<Item = anyhow::Result<String>> + '_> {
        self.tx.send_pack(chat::StreamRequest { prompt }).await?;
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

    pub async fn bench_unreliable(
        &mut self,
        received: usize,
        mtu: usize,
        delay_histogram: &AtomicHistogram,
    ) -> anyhow::Result<String> {
        let start_at = Instant::now();
        let mut last_recv_at = start_at;
        let mut actual_received = 0;
        self.tx
            .send_pack(bench::UnreliableRequest {
                data_len: received,
                mtu,
            })
            .await?;
        loop {
            let pack = self.rx.recv_pack().await?;
            let delay = last_recv_at.elapsed().as_micros();
            delay_histogram.increment(delay as u64)?;
            last_recv_at = Instant::now();
            match pack {
                Packet::BenchUnreliableRequest(_) => break, // EOF
                Packet::BenchUnreliableResponse(res) => {
                    actual_received += res.data_partial.len();
                }
                _ => bail!("interrupt pack {pack:?}"),
            }
        }
        let dur = start_at.elapsed();
        Ok(format!(
            "expect received {}\nactual received {}\nlost ratio {}\ncost {}ms\nrecv rate {}",
            bytes(received),
            bytes(actual_received),
            1.0 - actual_received as f64 / received as f64,
            dur.as_millis_f64(),
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
        let mut last_recv_at = start_at;
        self.tx
            .send_pack(bench::CommutativeRequest {
                data_len: received,
                batch_size,
            })
            .await?;
        let mut total_received = 0;
        loop {
            let pack = self.rx.recv_pack().await?;
            let delay = last_recv_at.elapsed().as_micros();
            delay_histogram.increment(delay as u64)?;
            last_recv_at = Instant::now();
            if let Packet::BenchCommutativeResponse(res) = pack {
                total_received += res.data_partial.len();
            } else {
                bail!("interrupt pack {pack:?}")
            }
            if total_received == received {
                break;
            }
        }
        let dur = start_at.elapsed();
        Ok(format!(
            "received {}\ncost {}ms\nrecv rate {}",
            bytes(received),
            dur.as_millis_f64(),
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
        let mut last_recv_at = start_at;
        self.tx
            .send_pack(bench::OrderedRequest {
                data_len: received,
                batch_size,
            })
            .await?;
        let mut total_received = 0;
        let mut expect_index = 0;
        loop {
            let pack = self.rx.recv_pack().await?;
            let delay = last_recv_at.elapsed().as_micros();
            delay_histogram.increment(delay as u64)?;
            last_recv_at = Instant::now();
            if let Packet::BenchOrderedResponse(res) = pack {
                total_received += res.data_partial.len();
                assert_eq!(res.index, expect_index); // ordered
                expect_index += 1;
            } else {
                bail!("interrupt pack {pack:?}")
            }
            if total_received == received {
                break;
            }
        }
        let dur = start_at.elapsed();
        Ok(format!(
            "received {}\ncost {}ms\nrecv rate {}",
            bytes(received),
            dur.as_millis_f64(),
            rate(total_received, dur)
        ))
    }

    pub async fn finish(&self) {
        self.close_notify.notify_one();
        let shutdown_timeout = Duration::from_secs(5);
        // wait for flusher closing
        let _ = self.close_notify.notified().timeout(shutdown_timeout).await;
        if self.is_raknet {
            // TCP handle 2MSL in the kernel, not here
            // wait for flusher shutdown
            eprintln!("wait 2MSL...");
            let _ = self.close_notify.notified().timeout(shutdown_timeout).await;
        }
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
