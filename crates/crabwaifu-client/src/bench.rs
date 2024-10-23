use crabwaifu_common::network::{Rx, Tx};

use crate::client::Client;

pub async fn run(client: &mut Client<impl Tx, impl Rx>) -> anyhow::Result<()> {
    println!("=== Unreliable Connection Benchmark ===");
    client.bench_unreliable().await;

    println!("=== Commutative Connection Benchmark ===");
    client.bench_commutative().await;

    println!("=== Ordered Connection Benchmark ===");
    client.bench_ordered().await;
    Ok(())
}
