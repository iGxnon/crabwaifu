use std::net::SocketAddr;

use clap::Parser;
use crabwaifu_client::client::connect_to;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    addr: SocketAddr,
}

#[tokio::main]
async fn main() {
    // let args = Args::parse();
    connect_to(
        "127.0.0.1:8808".parse().unwrap(),
        raknet_rs::client::Config::default(),
    )
    .await
    .unwrap();
}
