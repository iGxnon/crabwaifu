use std::io;
use std::sync::Arc;
use std::time::Duration;

use futures::SinkExt;
use raknet_rs::Message;
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;
use tokio::time;

use super::PinWriter;
use crate::utils::TimeoutWrapper;

// Default options
const DEFAULT_FLUSH_DELAY: Duration = Duration::from_millis(10); // 10ms -- The fastest time for the server to process a small piece of data.
const DEFAULT_BUF_SIZE: usize = 1; // Do not buffer too many messages
const DEFAULT_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

/// Spawn the flush task and return the handles of it
pub fn spawn_flush_task(
    writer: impl PinWriter,
) -> (
    mpsc::Sender<Message>,
    Arc<Notify>,
    Arc<Notify>,
    JoinHandle<()>,
) {
    let mut ticker = time::interval(DEFAULT_FLUSH_DELAY);
    ticker.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

    let (mut flusher, tx) = Flusher::new(writer, DEFAULT_BUF_SIZE, ticker);
    let flush_notify = Arc::new(Notify::new());
    let close_notify = Arc::new(Notify::new());

    let flush_notify_clone = flush_notify.clone();
    let close_notify_clone = close_notify.clone();
    let handle = tokio::spawn(async move {
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

        log::info!("destroying flusher...");
        if let Err(err) = flusher.close().await {
            log::error!("unexpected closing error {err}");
            close_notify_clone.notify_one();
            return;
        }

        // notify the waker that we have closed the flusher
        log::info!("flusher closed");
        close_notify_clone.notify_one();

        // Both peers will wait for 2MSL, the difference is that the passive closing side can abort
        // this task in advance.
        if let Err(err) = flusher.last_2msl().await {
            log::error!("unexpected shutdown error {err}");
            close_notify_clone.notify_one();
            return;
        }

        // notify again the waker that we have shutdown the flusher
        log::info!("flusher shutdown");
        close_notify_clone.notify_one();
    });

    (tx, flush_notify, close_notify, handle)
}

// TODO: add metrics
/// A naive auto balanced flush controller for each connection
struct Flusher<W: PinWriter> {
    writer: W,
    buffer: mpsc::Receiver<Message>,
    ticker: time::Interval,
}

impl<W: PinWriter> Flusher<W> {
    fn new(writer: W, size: usize, ticker: time::Interval) -> (Self, mpsc::Sender<Message>) {
        let (tx, rx) = mpsc::channel(size);
        let flusher = Self {
            writer,
            buffer: rx,
            ticker,
        };
        (flusher, tx)
    }

    #[inline(always)]
    async fn wait(&mut self) {
        self.ticker.tick().await;
    }

    #[inline(always)]
    async fn flush(&mut self) -> io::Result<()> {
        // non-blocking recv
        while let Ok(msg) = self.buffer.try_recv() {
            self.writer.feed(msg).await?;
        }
        self.writer.flush().await?;
        Ok(())
    }

    /// Close the writer with timeout
    #[inline(always)]
    async fn close(&mut self) -> io::Result<()> {
        self.writer.close().timeout(DEFAULT_CLOSE_TIMEOUT).await?
    }

    #[inline(always)]
    async fn last_2msl(&mut self) -> io::Result<()> {
        let task = async {
            loop {
                self.wait().await;
                self.flush().await?;
            }
        }
        .timeout(2 * DEFAULT_CLOSE_TIMEOUT);
        match task.await {
            Ok(res) => res,
            Err(_) => Ok(()),
        }
    }
}
