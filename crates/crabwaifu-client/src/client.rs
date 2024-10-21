use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail};
use crabwaifu_common::network::{spawn_flush_task, tcp_split, Rx, Tx};
use crabwaifu_common::proto::chat::{self, Message};
use crabwaifu_common::proto::Packet;
use crabwaifu_common::utils::TimeoutWrapper;
use futures::Stream;
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

    pub async fn finish(&self) {
        self.close_notify.notify_one();
        let shutdown_timeout = Duration::from_secs(5);
        // wait for flusher closing
        let _ = self.close_notify.notified().timeout(shutdown_timeout).await;
        if self.is_raknet {
            // wait for flusher shutdown
            eprintln!("wait 2MSL...");
            let _ = self.close_notify.notified().timeout(shutdown_timeout).await;
        }
    }
}
