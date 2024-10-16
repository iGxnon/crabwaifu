use std::env::args;

use crabwaifu_server::config::Config;
use crabwaifu_server::server::serve;

#[tokio::main]
async fn main() {
    env_logger::init();

    let config: Config = args().nth(1)
        .map(std::fs::read_to_string)
        .transpose()
        .unwrap()
        .as_deref()
        .map(toml::from_str)
        .transpose()
        .unwrap()
        .unwrap_or_default();
    log::trace!(
        "server configuration: \n{}",
        serde_json::to_string_pretty(&config).unwrap()
    );

    if let Err(err) = serve(config).await {
        log::error!("failed to start server {err}");
    }
}
