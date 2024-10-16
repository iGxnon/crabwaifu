use crabwaifu_common::network::{Rx, Tx};
use crabwaifu_common::proto::Packet;

pub struct Session<T, R> {
    tx: T,
    rx: R,
}

impl<T: Tx<Packet>, R: Rx<Packet>> Session<T, R> {
    pub fn new(tx: T, rx: R) -> Self {
        Self { tx, rx }
    }
}
