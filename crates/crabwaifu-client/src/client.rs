use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{anyhow, bail};
use crabwaifu_common::network::{spawn_flush_task, Rx, Tx};
use crabwaifu_common::proto::chat::{self, Message};
use crabwaifu_common::proto::Packet;
use futures::Stream;
use raknet_rs::client::{self, ConnectTo};
use tokio::net;
use tokio::sync::Notify;

pub struct Client<T, R> {
    tx: T,
    rx: R,
    flush_notify: Arc<Notify>,
    close_notify: Arc<Notify>,
}

pub async fn connect_to(
    addr: SocketAddr,
    config: client::Config,
) -> anyhow::Result<Client<impl Tx, impl Rx>> {
    let (rx, writer) = net::UdpSocket::bind("0.0.0.0:0")
        .await?
        .connect_to(addr, config)
        .await?;
    let (tx, flush_notify, close_notify) = spawn_flush_task(writer);
    let rx = Box::pin(rx);

    let client = Client {
        tx,
        rx,
        flush_notify,
        close_notify,
    };
    Ok(client)
}

impl<T, R> Drop for Client<T, R> {
    fn drop(&mut self) {
        self.close_notify.notify_one();
    }
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
            bail!("no response")
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
                                yield Err(anyhow!("response irrupt!"));
                            }
                        }
                        Err(err) => yield Err(err.into()),
                    }
                }
            }
        };
        Ok(stream)
    }
}
