use std::fmt;
use std::net::SocketAddr;

use clap::{Parser, Subcommand, ValueEnum};
use crabwaifu_client::client::{raknet_connect_to, tcp_connect_to, Client};
use crabwaifu_client::{bench, cli, ui};
use crabwaifu_common::network::{Rx, Tx};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'E', long, default_value = "127.0.0.1:8808")]
    endpoint: SocketAddr,
    #[arg(short = 'N', long, default_value_t = Network::Raknet)]
    network: Network,
    /// Only works in raknet
    #[arg(long, default_value_t = 1480)]
    mtu: u16,
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
    Cli {
        #[arg(short, long, default_value_t = false)]
        verbose: bool,
    },
    Ui,
    Bench {
        /// unreliable|commutative|ordered
        #[arg(short, long)]
        suite: bench::Suite,
        /// Total data wished to be received from server
        #[arg(long)]
        receive: usize,
        /// Data batch size expected to be divided into parts, ignored when use unreliable bench
        /// suite
        #[arg(long, default_value_t = 1460)]
        batch_size: usize,
        /// Brief output
        #[arg(long, default_value_t = false)]
        brief: bool,
    },
}

async fn run(client: &mut Client<impl Tx, impl Rx>, args: Args) -> anyhow::Result<()> {
    match args.command {
        Command::Cli { verbose } => cli::run(client, verbose).await,
        Command::Ui => ui::run_ui(client).await,
        Command::Bench {
            suite,
            receive,
            batch_size: parts,
            brief,
        } => bench::run(client, suite, receive, parts, args.mtu, brief).await,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.network {
        Network::Raknet => {
            let mut client = raknet_connect_to(
                args.endpoint,
                raknet_rs::client::Config::default()
                    .mtu(args.mtu)
                    .max_channels(255),
            )
            .await?;
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
