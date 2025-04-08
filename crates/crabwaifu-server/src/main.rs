use std::env::args;

use crabwaifu_server::config::{Config, Network};
use crabwaifu_server::server::{make_raknet_incoming, make_tcp_incoming, serve};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let config: Config = args()
        .nth(1)
        .map(std::fs::read_to_string)
        .transpose()
        .unwrap()
        .as_deref()
        .map(toml::from_str)
        .transpose()
        .unwrap()
        .unwrap();
    log::debug!(
        "server configuration: \n{}",
        serde_json::to_string_pretty(&config).unwrap()
    );

    match config.network {
        Network::Raknet => {
            println!("server is listening on raknet://{}", config.listen_addr);
            let incoming = make_raknet_incoming(config.listen_addr, config.raknet.clone()).await?;
            serve(incoming, config).await
        }
        Network::TCP => {
            println!("server is listening on tcp://{}", config.listen_addr);
            let incoming = make_tcp_incoming(config.listen_addr, config.tcp.clone()).await?;
            serve(incoming, config).await
        }
        Network::Both => {
            let c1 = config.clone();
            let raknet = async move {
                let incoming = make_raknet_incoming(config.listen_addr, c1.raknet.clone()).await?;
                serve(incoming, c1).await
            };
            let c2 = config.clone();
            let tcp = async move {
                let incoming = make_tcp_incoming(config.listen_addr, c2.tcp.clone()).await?;
                serve(incoming, c2).await
            };
            let rak_handle = tokio::spawn(raknet);
            let tcp_handle = tokio::spawn(tcp);
            rak_handle.await??;
            tcp_handle.await??;
            Ok(())
        }
    }
}
