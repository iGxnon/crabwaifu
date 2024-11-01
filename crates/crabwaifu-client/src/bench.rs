use std::fmt;

use clap::ValueEnum;
use crabwaifu_common::network::{Rx, Tx};
use histogram::{AtomicHistogram, Histogram};

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
    mtu: u16,
    brief: bool,
) -> anyhow::Result<()> {
    let histogram = AtomicHistogram::new(7, 64)?;
    println!("=== BENCHMARK START ===");
    let output = match suite {
        Suite::Unreliable => {
            client
                .bench_unreliable(received, mtu as usize, &histogram)
                .await
        }
        Suite::Commutative => {
            client
                .bench_commutative(received, batch_size, &histogram)
                .await
        }
        Suite::Ordered => client.bench_ordered(received, batch_size, &histogram).await,
    }?;
    println!("{}", output);
    print_histogram(histogram.load(), brief);
    println!("=== BENCHMARK ENDED ===");
    Ok(())
}

fn print_histogram(histogram: Histogram, brief: bool) {
    println!("delay histogram:");
    let cols = termsize::get().unwrap().cols;
    let sum: u64 = histogram.as_slice().iter().sum();
    for bucket in histogram.into_iter() {
        if bucket.count() == 0 {
            continue;
        }
        let range = bucket.range();
        let prefix = format!(
            "{:05} ~ {:05} us | {} \t",
            range.start(),
            range.end(),
            bucket.count()
        );
        let cols = cols as usize - prefix.len();
        let cols = (bucket.count() as f64 / sum as f64 * cols as f64) as usize;
        if brief && cols == 0 {
            continue;
        }
        println!("{prefix}{}", "⬛".repeat(cols));
    }
}
