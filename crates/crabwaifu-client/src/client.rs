use std::io;

use crabwaifu_common::network::spawn_flush_task;
use raknet_rs::client::{self, ConnectTo};
use tokio::net;

async fn connect_to(
    addr: impl std::net::ToSocketAddrs + 'static,
    config: client::Config,
) -> io::Result<()> {
    let (rx, writer) = net::UdpSocket::bind("0.0.0.0:0")
        .await?
        .connect_to(addr, config)
        .await?;
    let (tx, flush_notify, close_notify) = spawn_flush_task(writer);
    tokio::pin!(rx);

    // TODO: use tx and rx to construct a client

    Ok(())
}
