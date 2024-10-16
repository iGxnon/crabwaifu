use std::future::Future;
use std::io;

use bytes::{BufMut, Bytes, BytesMut};
use futures::{Sink, Stream, StreamExt};
use raknet_rs::{Message, Reliability};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::mpsc;

mod flusher;

pub use flusher::spawn_flush_task;

// I am the loyal fan of static dispatch(
// Ensure unpin here cz i do not want to write too many projections
pub trait UnpinWriter = Sink<Message, Error = io::Error> + Unpin + Send + Sync + 'static;
pub trait UnpinReader = Stream<Item = Bytes> + Unpin + Send + Sync + 'static;

pub trait Packet: Serialize + DeserializeOwned {
    const GUESS_CAP: usize;
    const RELIABILITY: Reliability;
    const ORDER_CHANNEL: u8;
}

/// Default implementation for random packets
impl<P> Packet for P
where
    P: Serialize + DeserializeOwned,
{
    const GUESS_CAP: usize = 0;
    const ORDER_CHANNEL: u8 = 0;
    const RELIABILITY: Reliability = Reliability::ReliableOrdered;
}

pub trait Tx<P: Packet> {
    /// It can be sent frequently to a target in multiple threads, so it is a shared reference
    fn send_raw(&self, msg: Message) -> impl Future<Output = io::Result<()>> + Send + Sync;

    /// Send a packet with reliability and order channel specified in Pack
    fn send_pack(&self, pack: P) -> impl Future<Output = io::Result<()>> {
        async move {
            let mut writer = BytesMut::with_capacity(P::GUESS_CAP).writer();
            bincode::serialize_into(&mut writer, &pack)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
            let buf = writer.into_inner();
            self.send_raw(Message::new(P::RELIABILITY, P::ORDER_CHANNEL, buf.freeze()))
                .await
        }
    }
}

pub trait Rx<P: Packet> {
    /// Receiving happens only in one place, so this is an exclusive reference (mutable reference)
    fn recv_raw(&mut self) -> impl Future<Output = io::Result<Bytes>> + Send + Sync;

    fn recv_pack(&mut self) -> impl Future<Output = io::Result<P>> {
        async move {
            let raw = self.recv_raw().await?;
            let pack: P = bincode::deserialize(&raw)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
            Ok(pack)
        }
    }
}

// something underneath Tx and Rx

impl<P: Packet> Tx<P> for mpsc::Sender<Message> {
    async fn send_raw(&self, msg: Message) -> io::Result<()> {
        self.send(msg)
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "flusher closed"))
    }
}

impl<P: Packet, R: UnpinReader> Rx<P> for R {
    async fn recv_raw(&mut self) -> io::Result<Bytes> {
        self.next()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "connection closed"))
    }
}
