use std::io;
use std::sync::Arc;
use std::time::Duration;

use futures::SinkExt;
use raknet_rs::Message;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::{oneshot, Notify};
use tokio::task::JoinHandle;
use tokio::time;

use super::PinWriter;
use crate::utils::TimeoutWrapper;

/// Spawn the flush task and return the handles of it
pub fn spawn_flush_task(
    writer: impl PinWriter,
) -> (
    Sender<Message>,
    Arc<Notify>,
    oneshot::Sender<bool>,
    JoinHandle<()>,
) {
    let (mut flusher, flush_notify, tx) = Flusher::new(writer);
    let (close_tx, mut close_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let wait_2msl = tokio::select! {
            Ok(wait) = &mut close_rx => wait,
            Err(err) = flusher.run() => {
                log::error!("unexpected flush error {err}, terminating flush task");
                false
            }
        };

        log::info!("closing flusher...");
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
    ticker: time::Interval,
    notifier: Arc<Notify>,
    rx: Receiver<Message>,
}

impl<W: PinWriter> Flusher<W> {
    // The maximum delay accepted by peer or the minimum delay of processing a piece of data
    const DEFAULT_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);
    const DEFAULT_FLUSH_DELAY: Duration = Duration::from_millis(1);
    const MAX_FLUSH_PENDING: usize = 1;

    fn new(writer: W) -> (Self, Arc<Notify>, Sender<Message>) {
        let mut ticker = time::interval(Self::DEFAULT_FLUSH_DELAY);
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
        let notifier = Arc::new(Notify::new());
        let (tx, rx) = mpsc::channel(Self::MAX_FLUSH_PENDING);
        let me = Self {
            writer,
            ticker,
            notifier: notifier.clone(),
            rx,
        };
        (me, notifier, tx)
    }

    async fn run(&mut self) -> io::Result<()> {
        loop {
            tokio::select! {
                _ = self.ticker.tick() => {
                    self.writer.flush().await?;
                }
                _ = self.notifier.notified() => {
                    self.writer.flush().await?;
                }
                Some(msg) = self.rx.recv() => {
                    self.writer.feed(msg).await?;
                }
            }
        }
    }

    /// Close the writer with timeout
    async fn close(&mut self) -> io::Result<()> {
        self.writer
            .close()
            .timeout(Self::DEFAULT_CLOSE_TIMEOUT)
            .await?
    }

    async fn last_2msl(&mut self) -> io::Result<()> {
        match self.run().timeout(2 * Self::DEFAULT_CLOSE_TIMEOUT).await {
            Ok(res) => res,
            Err(_) => Ok(()),
        }
    }
}
