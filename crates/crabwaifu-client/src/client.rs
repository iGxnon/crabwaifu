use std::io::{self, Write};
use std::net::SocketAddr;
use std::time::Duration;

use crabwaifu_common::network::{spawn_flush_task, Rx, Tx};
use crabwaifu_common::proto::{self, chat};
use raknet_rs::client::{self, ConnectTo};
use tokio::net;

pub async fn connect_to(addr: SocketAddr, config: client::Config) -> io::Result<()> {
    let (rx, writer) = net::UdpSocket::bind("0.0.0.0:0")
        .await?
        .connect_to(addr, config)
        .await?;
    let (tx, flush_notify, close_notify) = spawn_flush_task(writer);
    let mut rx = Box::pin(rx);

    // TODO: use tx and rx to construct a client

    println!("send message");
    tx.send_pack(chat::StreamRequest {
        prompt: "请问 1 + 1 等于多少？".to_string(),
    })
    .await
    .unwrap();
    flush_notify.notify_one();

    println!("wait for reply...");
    println!("me: 请问 1 + 1 等于多少？");
    std::io::stdout().flush().unwrap();
    while let Ok(pack) = rx.recv_pack().await {
        if let proto::Packet::ChatStreamResponse(resp) = pack {
            std::io::stdout().flush().unwrap();
            print!("{}", resp.partial);
            if resp.eos {
                println!();
            }
        }
    }

    println!("connection closed, wait for return in 10s");
    close_notify.notify_one();
    if (tokio::time::timeout(Duration::from_secs(10), close_notify.notified()).await).is_err() {
        println!("error for waiting return");
    }

    Ok(())
}
