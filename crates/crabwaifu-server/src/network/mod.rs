use std::io;
use std::pin::Pin;

use bytes::{BufMut, Bytes, BytesMut};
use futures::{Sink, Stream, StreamExt};
use raknet_rs::{Message, Reliability};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::mpsc;

mod flusher;

// I am not a loyal fan of static dispatch
type PinWriter = Pin<Box<dyn Sink<Message, Error = io::Error> + Send + Sync + 'static>>;
type PinReader = Pin<Box<dyn Stream<Item = Bytes> + Send + Sync + 'static>>;
type BoxTx<P> = Box<dyn Tx<P> + Send + Sync + 'static>;
type BoxRx<P> = Box<dyn Rx<P> + Send + Sync + 'static>;

trait Packet: Serialize + DeserializeOwned {
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

trait Tx<P: Packet> {
    /// It can be sent frequently to a target in multiple threads, so it is a shared reference
    async fn send_raw(&self, msg: Message) -> io::Result<()>;

    /// Send a packet with reliability and order channel specified in Pack
    async fn send_pack(&self, pack: P) -> io::Result<()> {
        let mut writer = BytesMut::with_capacity(P::GUESS_CAP).writer();
        bincode::serialize_into(&mut writer, &pack)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        let buf = writer.into_inner();
        self.send_raw(Message::new(P::RELIABILITY, P::ORDER_CHANNEL, buf.freeze()))
            .await?;
        Ok(())
    }
}

trait Rx<P: Packet> {
    /// Receiving happens only in one place, so this is an exclusive reference (mutable reference)
    async fn recv_raw(&mut self) -> io::Result<Bytes>;

    async fn recv_pack(&mut self) -> io::Result<P> {
        let raw = self.recv_raw().await?;
        let pack: P = bincode::deserialize(&raw)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        Ok(pack)
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

impl<P: Packet> Rx<P> for PinReader {
    async fn recv_raw(&mut self) -> io::Result<Bytes> {
        self.next()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "connection closed"))
    }
}
