use std::io::Write;
use std::time::Duration;

use crabwaifu_common::network::{Rx, Tx};
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
                let mut first_token_recv = None;
                let mut last_token_recv: Option<Instant> = None;
                let mut total_delay = Duration::from_secs(0);
                let mut tokens = 0;
                let reply = client.stream(line).await?;
                #[futures_async_stream::for_await]
                for ele in reply {
                    tokens += 1;
                    if tokens == 1 {
                        let elapsed = start_at.elapsed();
                        first_token_recv = Some(elapsed);
                        total_delay = elapsed;
                    }
                    if let Some(last_recv) = last_token_recv {
                        total_delay += last_recv.elapsed();
                    }
                    last_token_recv = Some(Instant::now());
                    let token = ele?;
                    print!("{token}");
                    std::io::stdout().flush().unwrap();
                }
                println!();
                let elapsed: Duration = start_at.elapsed();
                let tokens_per_second = tokens as f64 / elapsed.as_secs_f64();
                let average_delay = total_delay.as_millis_f64() / tokens as f64;
                if verbose {
                    println!(
                        "\x1b[32mgenerated {} tokens, total {}ms, {} tokens/s, first token received in {}ms, average token delay {}ms \x1b[0m",
                        tokens,
                        elapsed.as_millis(),
                        tokens_per_second,
                        first_token_recv.unwrap().as_millis(),
                        average_delay,
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
