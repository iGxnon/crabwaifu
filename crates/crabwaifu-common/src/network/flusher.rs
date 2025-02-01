use std::io;
use std::sync::Arc;
use std::time::Duration;

use futures::SinkExt;
use raknet_rs::Message;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{mpsc, oneshot, Notify};
use tokio::task::JoinHandle;
use tokio::time;

use super::PinWriter;
use crate::utils::TimeoutWrapper;

const DEFAULT_FLUSH_DELAY: Duration = Duration::from_millis(1); // The maximum delay accepted by peer or the minimum delay of processing a piece of data
const DEFAULT_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

/// Spawn the flush task and return the handles of it
pub fn spawn_flush_task(
    writer: impl PinWriter,
) -> (
    UnboundedSender<Message>,
    Arc<Notify>,
    oneshot::Sender<bool>,
    JoinHandle<()>,
) {
    let mut ticker = time::interval(DEFAULT_FLUSH_DELAY);
    ticker.set_missed_tick_behavior(time::MissedTickBehavior::Delay);

    let (mut flusher, tx) = Flusher::new(writer, ticker);
    let flush_notify = Arc::new(Notify::new());
    let (close_tx, mut close_rx) = oneshot::channel();

    let flush_notify_clone = flush_notify.clone();
    let handle = tokio::spawn(async move {
        let wait_2msl = loop {
            tokio::select! {
                _ = flusher.wait() => {
                    if let Err(err) = flusher.flush().await {
                        log::error!("unexpected flush error {err}, terminating flush task");
                        break false;
                    }
                }
                _ = flush_notify_clone.notified() => {
                    if let Err(err) = flusher.flush().await {
                        log::error!("unexpected flush error {err}, terminating flush task");
                        break false;
                    }
                }
                res = &mut close_rx => break res.expect("channel never close"),
            }
        };

        log::info!("destroying flusher...");
        if let Err(err) = flusher.close().await {
            log::error!("unexpected closing error {err}");
            return;
        }
        log::info!("flusher closed");
        
        if !wait_2msl {
            return;
        }
        log::info!("wait 2MSL...");
        if let Err(err) = flusher.last_2msl().await {
            log::error!("unexpected shutdown error {err}");
        }
    });

    (tx, flush_notify, close_tx, handle)
}

// TODO: add metrics
/// A naive auto balanced flush controller for each connection
struct Flusher<W: PinWriter> {
    writer: W,
    buffer: UnboundedReceiver<Message>,
    ticker: time::Interval,
}

impl<W: PinWriter> Flusher<W> {
    fn new(writer: W, ticker: time::Interval) -> (Self, UnboundedSender<Message>) {
        let (tx, rx) = mpsc::unbounded_channel();
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
