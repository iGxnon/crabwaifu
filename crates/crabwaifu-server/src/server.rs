use std::io;

use crabwaifu_common::network::spawn_flush_task;
use futures::StreamExt;
use raknet_rs::server::MakeIncoming;
use tokio::net;
use tokio::sync::watch;

use crate::config::Config;
use crate::session;

pub async fn serve(config: Config) -> io::Result<()> {
    let mut incoming = net::UdpSocket::bind(config.listen_addr)
        .await?
        .make_incoming(config.raknet);

    tokio::spawn(async move {
        let (shutdown, watcher) = watch::channel("running");
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        loop {
            tokio::select! {
                Some((rx, writer)) = incoming.next() => {
                    let (tx, flush_notify, close_notify) = spawn_flush_task(writer);
                    let session = session::Session::new(tx, Box::pin(rx), flush_notify, close_notify);
                    tokio::spawn(session.run(watcher.clone()));
                }
                _ = &mut ctrl_c => {
                    // The root watcher
                    if shutdown.receiver_count() == 1 {
                        break;
                    }
                    shutdown.send("shutdown").unwrap();
                    let ctrl_c_c = tokio::signal::ctrl_c();
                    tokio::select! {
                        _ = shutdown.closed() => {
                            break;
                        }
                        _ = ctrl_c_c => {
                            log::warn!("force shutdown");
                            return;
                        }
                    }
                }
            }
        }
        log::info!("shutdown gracefully")
    });

    Ok(())
}
