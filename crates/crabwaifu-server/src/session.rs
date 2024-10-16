use std::sync::Arc;

use crabwaifu_common::network::{Rx, Tx};
use crabwaifu_common::proto::Packet;
use tokio::sync::{watch, Notify};

pub struct Session<T, R> {
    tx: T,
    rx: R,
    flush_notify: Arc<Notify>,
    close_notify: Arc<Notify>,
}

impl<T, R> Drop for Session<T, R> {
    fn drop(&mut self) {
        // close flusher on drop
        self.close_notify.notify_waiters();
    }
}

impl<T: Tx, R: Rx<Packet>> Session<T, R> {
    pub fn new(tx: T, rx: R, flush_notify: Arc<Notify>, close_notify: Arc<Notify>) -> Self {
        Self {
            tx,
            rx,
            flush_notify,
            close_notify,
        }
    }

    #[inline]
    fn dispatch_pack(&self, pack: Packet) {}

    pub async fn run(mut self, mut shutdown: watch::Receiver<&'static str>) {
        let signal = shutdown.changed();
        tokio::pin!(signal);
        loop {
            tokio::select! {
                res = self.rx.recv_pack() => match res {
                    Ok(pack) => self.dispatch_pack(pack),
                    Err(err) => {
                        log::error!("error in recv packets {err}")
                    },
                },
                _ = &mut signal => {
                    // TODO: wait all task of this session to be done and return
                }
            }
        }
    }
}
