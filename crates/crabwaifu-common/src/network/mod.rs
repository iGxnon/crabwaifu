use std::future::Future;
use std::io;
use std::io::Write;

use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures::{Sink, Stream, StreamExt};
use raknet_rs::{Message, Reliability};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::mpsc;

mod flusher;

pub use flusher::spawn_flush_task;

use crate::proto::{chat, Packet, PacketID};

// I am the loyal fan of static dispatch(
// Ensure unpin here cz i do not want to write too many projections
pub trait PinWriter = Sink<Message, Error = io::Error> + Unpin + Send + Sync + 'static;
pub trait PinReader = Stream<Item = Bytes> + Unpin + Send + Sync + 'static;

pub trait Pack: Serialize + DeserializeOwned + Send + Sync + 'static {
    const ID: PacketID;
    const RELIABILITY: Reliability;
    const ORDER_CHANNEL: u8;
}

/// Default implementation for random packets
impl<P> Pack for P
where
    P: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    // ID must be override
    default const ID: PacketID = PacketID::InvalidPack;
    default const ORDER_CHANNEL: u8 = 0;
    default const RELIABILITY: Reliability = Reliability::ReliableOrdered;
}

// Sending packets within many threads, so this is a shared reference
pub trait Tx: Send + Sync {
    fn send_raw(&self, msg: Message) -> impl Future<Output = io::Result<()>> + Send + Sync;

    /// Send a packet with reliability and order channel specified in P
    fn send_pack<P: Pack>(&self, pack: P) -> impl Future<Output = io::Result<()>> + Send {
        debug_assert!(
            !matches!(P::ID, PacketID::InvalidPack),
            "please send a valid packet"
        );

        async move {
            let cap = bincode::serialized_size(&pack).unwrap_or_default() as usize + 1;
            let mut writer = BytesMut::with_capacity(cap).writer();
            writer
                .write_all(&[P::ID as u8])
                .expect("failed to write ID into buffer");
            bincode::serialize_into(&mut writer, &pack)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
            let buf = writer.into_inner();
            self.send_raw(Message::new(P::RELIABILITY, P::ORDER_CHANNEL, buf.freeze()))
                .await
        }
    }
}

/// Receiving packets happens only in one place, so this is an exclusive reference (mutable
/// reference)
pub trait Rx {
    fn recv_raw(&mut self) -> impl Future<Output = io::Result<Bytes>> + Send + Sync;

    fn recv_pack(&mut self) -> impl Future<Output = io::Result<Packet>> {
        async move {
            macro_rules! deserialize {
                ($from:ty, $to:expr, $var:expr) => {
                    $to(bincode::deserialize::<$from>(&$var)
                        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?)
                };
            }

            let mut raw = self.recv_raw().await?;
            let id = PacketID::from_u8(raw.get_u8());
            let pack = match id {
                PacketID::InvalidPack => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid packet id",
                    ))
                }
                PacketID::ChatRequest => {
                    deserialize!(chat::Request, Packet::ChatRequest, raw)
                }
                PacketID::ChatResponse => {
                    deserialize!(chat::Response, Packet::ChatResponse, raw)
                }
            };
            Ok(pack)
        }
    }
}

// something underneath Tx and Rx

impl Tx for mpsc::Sender<Message> {
    async fn send_raw(&self, msg: Message) -> io::Result<()> {
        self.send(msg)
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "flusher closed"))
    }
}

impl<R: PinReader> Rx for R {
    async fn recv_raw(&mut self) -> io::Result<Bytes> {
        self.next()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "connection closed"))
    }
}
