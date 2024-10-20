use core::task::ContextBuilder;
use std::sync::Arc;
use std::time::Duration;
use std::{cmp, io};

use futures::SinkExt;
use raknet_rs::opts::FlushStrategy;
use raknet_rs::Message;
use tokio::sync::{mpsc, Notify};
use tokio::time;

use super::PinWriter;
use crate::utils::TimeoutWrapper;

// Flusher can automatically adjust the flush delay, below is the range of adjustment
const MIN_FLUSH_DELAY_US: u64 = 2_000; // 2ms -- The fastest time for the server to process a small piece of data.
const MAX_FLUSH_DELAY_US: u64 = 512_000; // 512ms -- The maximum acceptable RTO

// Default options
const DEFAULT_BUF_SIZE: usize = 1024;
const DEFAULT_CLOSE_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_2MSL: Duration = Duration::from_secs(6);

/// Spawn the flush task and return the handles of it
pub fn spawn_flush_task(
    writer: impl PinWriter,
) -> (mpsc::Sender<Message>, Arc<Notify>, Arc<Notify>) {
    let (mut flusher, tx) = Flusher::new(writer, DEFAULT_BUF_SIZE);
    let flush_notify = Arc::new(Notify::new());
    let close_notify = Arc::new(Notify::new());

    let flush_notify_clone = flush_notify.clone();
    let close_notify_clone = close_notify.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = flusher.wait() => {
                    if let Err(err) = flusher.flush().await {
                        log::error!("unexpected flush error {err}, terminating flush task gracefully");
                        break
                    }
                }
                _ = flush_notify_clone.notified() => {
                    if let Err(err) = flusher.flush().await {
                        log::error!("unexpected flush error {err}, terminating flush task gracefully");
                        break
                    }
                }
                _ = close_notify_clone.notified() => break,
            }
        }
        if let Err(err) = flusher.close().await {
            log::error!("unexpected closing error {err}");
            // notify the waker that we have closed the flusher
            close_notify_clone.notify_one();
            return;
        }
        // notify the waker that we have closed the flusher
        close_notify_clone.notify_one();

        log::info!("flusher closed, wait 2MSL...");
        let last_2msl = tokio::time::sleep(DEFAULT_2MSL);
        tokio::pin!(last_2msl);
        loop {
            tokio::select! {
                _ = flusher.wait() => {
                    if let Err(err) = flusher.flush().await {
                        log::error!("unexpected flush error {err}, terminating flush task gracefully");
                        break
                    }
                }
                _ = &mut last_2msl => break,
            }
        }

        // notify twice the waker that we have shutdown the flusher
        close_notify_clone.notify_one();
    });

    (tx, flush_notify, close_notify)
}

// TODO: add metrics
/// A naive auto balanced flush controller for each connection
struct Flusher<W: PinWriter> {
    writer: W,
    next_flush: Option<time::Instant>,
    buffer: mpsc::Receiver<Message>,
    delay_us: u64,
}

impl<W: PinWriter> Flusher<W> {
    fn new(writer: W, size: usize) -> (Self, mpsc::Sender<Message>) {
        let (tx, rx) = mpsc::channel(size);
        let flusher = Self {
            writer,
            next_flush: None,
            buffer: rx,
            delay_us: MIN_FLUSH_DELAY_US,
        };
        (flusher, tx)
    }

    #[inline(always)]
    async fn wait(&self) {
        if let Some(next_flush) = self.next_flush {
            time::sleep_until(next_flush).await;
        }
    }

    #[inline(always)]
    async fn flush(&mut self) -> io::Result<()> {
        // non-blocking recv
        while let Ok(msg) = self.buffer.try_recv() {
            self.writer.feed(msg).await?;
        }

        // Flush with strategy
        let mut strategy = FlushStrategy::new(true, true, true);
        std::future::poll_fn(|cx| {
            let mut cx = ContextBuilder::from(cx).ext(&mut strategy).build();
            self.writer.poll_flush_unpin(&mut cx)
        })
        .await?;

        // A naive exponential algorithm
        if strategy.flushed_ack() + strategy.flushed_nack() + strategy.flushed_pack() > 0 {
            self.delay_us = cmp::max(self.delay_us / 2, MIN_FLUSH_DELAY_US);
        } else {
            self.delay_us = cmp::min(self.delay_us * 2, MAX_FLUSH_DELAY_US);
        }
        self.next_flush = Some(time::Instant::now() + time::Duration::from_micros(self.delay_us));

        Ok(())
    }

    /// Close the writer with timeout
    #[inline(always)]
    async fn close(&mut self) -> io::Result<()> {
        self.writer.close().timeout(DEFAULT_CLOSE_TIMEOUT).await?
    }
}
