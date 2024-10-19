use std::io::Write;

use crabwaifu_common::network::{Rx, Tx};
use futures::StreamExt;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tokio::time::Instant;

use crate::client::Client;

pub async fn run(client: &mut Client<impl Tx, impl Rx>, verbose: bool) -> anyhow::Result<()> {
    let mut rl = DefaultEditor::new()?;
    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(line) => {
                if line.is_empty() {
                    continue;
                }
                rl.add_history_entry(line.as_str())?;
                let start_at = Instant::now();
                let mut tokens = 0;
                let reply = client.stream(line).await?;
                tokio::pin!(reply);
                while let Some(ele) = reply.next().await {
                    tokens += 1;
                    let token = ele?;
                    print!("{token}");
                    std::io::stdout().flush().unwrap();
                }
                println!();
                let elapsed: std::time::Duration = start_at.elapsed();
                let tokens_per_second = tokens as f64 / elapsed.as_secs_f64();
                if verbose {
                    println!(
                        "\x1b[32m generated {} tokens, total {}ms, {} tokens/s \x1b[0m",
                        tokens,
                        elapsed.as_millis(),
                        tokens_per_second
                    );
                }
            }
            Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    Ok(())
}
