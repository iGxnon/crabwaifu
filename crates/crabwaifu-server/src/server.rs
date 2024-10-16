use std::io;

use crabwaifu_common::network::spawn_flush_task;
use futures::StreamExt;
use raknet_rs::server::{self, MakeIncoming};
use tokio::net;

use crate::session;

async fn serve_on(addr: impl net::ToSocketAddrs, config: server::Config) -> io::Result<()> {
    let mut incoming = net::UdpSocket::bind(addr).await?.make_incoming(config);

    tokio::spawn(async move {
        loop {
            let (rx, writer) = incoming
                .next()
                .await
                .expect("incoming task should not terminate");
            let (tx, flush_notify, close_notify) = spawn_flush_task(writer);

            // TODO: start session

            let session = session::Session::new(tx, Box::pin(rx));
        }
    });

    Ok(())
}
