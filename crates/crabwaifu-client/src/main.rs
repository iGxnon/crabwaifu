use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use crabwaifu_client::cli;
use crabwaifu_client::client::connect_to;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8808")]
    endpoint: SocketAddr,
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Cli,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut client = connect_to(args.endpoint, raknet_rs::client::Config::default()).await?;
    match args.command {
        Command::Cli => cli::run(&mut client, args.verbose).await,
    }
}
