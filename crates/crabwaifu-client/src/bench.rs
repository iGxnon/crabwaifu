use std::time::Duration;
use std::{cmp, fmt};

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
    brief: bool,
) -> anyhow::Result<()> {
    let histogram = AtomicHistogram::new(7, 64)?;
    println!("=== BENCHMARK START ===");
    let output = match suite {
        Suite::Unreliable => client.bench_unreliable(received, &histogram).await,
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
    let cols = termsize::get()
        .map(|size| size.cols)
        .unwrap_or(80)
        .clamp(80, 160);
    let sum: u64 = histogram.as_slice().iter().sum();
    let count_digits = digits(
        histogram
            .as_slice()
            .iter()
            .max()
            .copied()
            .unwrap_or_default(),
    );
    let mut min_delay = u64::MAX;
    let mut max_delay = 0;
    let histogram_c = histogram.clone();
    for bucket in histogram.into_iter() {
        if bucket.count() == 0 {
            continue;
        }
        let range = bucket.range();
        max_delay = cmp::max(max_delay, *range.end());
        min_delay = cmp::min(min_delay, *range.start());
        let prefix = format!(
            "{:09} ~ {:09} ns | {} ",
            range.start(),
            range.end(),
            pad_digits(bucket.count(), count_digits)
        );
        let cols = cols as usize - prefix.len();
        let cols = (bucket.count() as f64 / sum as f64 * cols as f64) as usize;
        if brief && cols == 0 {
            continue;
        }
        println!("{prefix}{}", "█".repeat(cols));
    }
    let p50 = histogram_c.percentile(50.0).unwrap().unwrap().start();
    let p95 = histogram_c.percentile(95.0).unwrap().unwrap().start();
    let p999 = histogram_c.percentile(99.9).unwrap().unwrap().start();
    let p9999 = histogram_c.percentile(99.99).unwrap().unwrap().start();
    println!(
        "delay distribution:\nmax\t{:?}\nmin\t{:?}\np50\t{:?}\np95\t{:?}\np99.9\t{:?}\np99.99\t{:?}",
        Duration::from_nanos(max_delay),
        Duration::from_nanos(min_delay),
        Duration::from_nanos(p50),
        Duration::from_nanos(p95),
        Duration::from_nanos(p999),
        Duration::from_nanos(p9999),
    );
}

fn digits(mut num: u64) -> usize {
    let mut ret = 0;
    while num > 0 {
        ret += 1;
        num /= 10;
    }
    ret
}

fn pad_digits(num: u64, digit: usize) -> String {
    let mut ret = num.to_string();
    while ret.len() < digit {
        ret.insert(0, ' ');
    }
    ret
}
