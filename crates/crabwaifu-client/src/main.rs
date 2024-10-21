use std::fmt;
use std::net::SocketAddr;

use clap::{Parser, Subcommand, ValueEnum};
use crabwaifu_client::cli;
use crabwaifu_client::client::{raknet_connect_to, tcp_connect_to, Client};
use crabwaifu_common::network::{Rx, Tx};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'E', long, default_value = "127.0.0.1:8808")]
    endpoint: SocketAddr,
    #[arg(short = 'N', long, default_value_t = Network::Raknet)]
    network: Network,
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
    #[clap(subcommand)]
    command: Command,
}

#[derive(Clone, Debug, ValueEnum)]
enum Network {
    Raknet,
    Tcp,
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Network::Raknet => write!(f, "raknet"),
            Network::Tcp => write!(f, "tcp"),
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    Cli,
    Bench,
}

async fn run(client: &mut Client<impl Tx, impl Rx>, args: Args) -> anyhow::Result<()> {
    match args.command {
        Command::Cli => cli::run(client, args.verbose).await,
        Command::Bench => todo!(),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.network {
        Network::Raknet => {
            let mut client =
                raknet_connect_to(args.endpoint, raknet_rs::client::Config::default()).await?;
            if let Err(err) = run(&mut client, args).await {
                eprintln!("error: {err}");
            }
            client.finish().await;
        }
        Network::Tcp => {
            let mut client = tcp_connect_to(args.endpoint).await?;
            if let Err(err) = run(&mut client, args).await {
                eprintln!("error: {err}");
            }
            client.finish().await;
        }
    }
    Ok(())
}
