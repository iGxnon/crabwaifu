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
        .unwrap_or_default();
    log::debug!(
        "server configuration: \n{}",
        serde_json::to_string_pretty(&config).unwrap()
    );

    match config.network {
        Network::Raknet => {
            println!("server is listening on raknet://{}", config.listen_addr);
            let incoming = make_raknet_incoming(config.listen_addr, config.raknet).await?;
            serve(incoming, config.llama).await
        }
        Network::TCP => {
            println!("server is listening on tcp://{}", config.listen_addr);
            let incoming = make_tcp_incoming(config.listen_addr, config.tcp).await?;
            serve(incoming, config.llama).await
        }
    }
}
