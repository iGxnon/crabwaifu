use std::fmt;

use clap::ValueEnum;
use crabwaifu_common::network::{Rx, Tx};

use crate::client::Client;

#[derive(Clone, Debug, ValueEnum)]
pub enum Suite {
    Unreliable,
    Commutative,
    Ordered,
}

impl fmt::Display for Suite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Suite::Unreliable => write!(f, "unreliable"),
            Suite::Commutative => write!(f, "commutative"),
            Suite::Ordered => write!(f, "ordered"),
        }
    }
}

pub async fn run(
    client: &mut Client<impl Tx, impl Rx>,
    suite: Suite,
    received: usize,
    batch_size: usize,
) -> anyhow::Result<()> {
    match suite {
        Suite::Unreliable => {
            println!("=== Unreliable Connection Benchmark ===");
            client.bench_unreliable().await;
        }
        Suite::Commutative => {
            println!("=== Commutative Connection Benchmark ===");
            client.bench_commutative().await;
        }
        Suite::Ordered => {
            println!("=== Ordered Connection Benchmark ===");
            client.bench_ordered().await;
        }
    }

    Ok(())
}
