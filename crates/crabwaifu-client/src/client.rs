use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use crabwaifu_common::network::{spawn_flush_task, Rx, Tx};
use crabwaifu_common::proto::chat::{self, Message};
use raknet_rs::client::{self, ConnectTo};
use tokio::net;

pub async fn connect_to(addr: SocketAddr, config: client::Config) -> io::Result<()> {
    let (rx, writer) = net::UdpSocket::bind("0.0.0.0:0")
        .await?
        .connect_to(addr, config)
        .await?;
    let (tx, flush_notify, close_notify) = spawn_flush_task(writer);

    // TODO: use tx and rx to construct a client

    println!("send message");

    tx.send_pack(chat::Request {
        messages: vec![Message {
            role: chat::Role::User,
            content: "alice wonderful land".to_string(),
        }],
        steps: None,
    })
    .await
    .unwrap();

    flush_notify.notify_one();

    println!("wait for reply...");

    let pack = Box::pin(rx).recv_pack().await.unwrap();
    println!("{:?}", pack);

    tokio::time::sleep(Duration::from_secs(5)).await;

    Ok(())
}
