use std::future::Future;
use std::io;
use std::io::Write;

use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures::{Sink, Stream, StreamExt};
use raknet_rs::{Message, Reliability};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::mpsc::{self, UnboundedSender};

mod flusher;
mod tcp;

pub use flusher::spawn_flush_task;
pub use tcp::{tcp_split, TcpListenerStream};

use crate::proto::{bench, chat, Packet, PacketID};

// I am the loyal fan of static dispatch(
// Ensure unpin here cz i do not want to write too many projections
pub trait PinWriter = Sink<Message, Error = io::Error> + Unpin + Send + Sync + 'static;
pub trait PinReader = Stream<Item = Bytes> + Unpin + Send + Sync + 'static;

pub trait Pack: Serialize + DeserializeOwned + Send + Sync + 'static {
    const ID: PacketID;
    const RELIABILITY: Reliability;
    const ORDER_CHANNEL: u8;
}

// Sending packets within many threads, so this is a shared reference
pub trait Tx: Send + Sync + 'static {
    fn send_raw(&self, msg: Message) -> impl Future<Output = io::Result<()>> + Send + Sync;

    /// Send a packet with reliability and order channel specified in P
    fn send_pack<P: Pack>(&self, pack: P) -> impl Future<Output = io::Result<()>> + Send + Sync {
        debug_assert!(
            !matches!(P::ID, PacketID::InvalidPack),
            "please send a valid packet"
        );

        async move {
            let cap = bincode::serialized_size(&pack).unwrap_or_default() as usize + 2;
            let mut writer = BytesMut::with_capacity(cap).writer();
            writer.write_all(&[0xfe, P::ID as u8])?;
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
pub trait Rx: Send + Sync + 'static {
    fn recv_raw(&mut self) -> impl Future<Output = io::Result<Bytes>> + Send + Sync;

    fn recv_pack(&mut self) -> impl Future<Output = io::Result<Packet>> + Send + Sync {
        async move {
            macro_rules! deserialize {
                ($from:ty, $to:expr, $var:expr) => {
                    $to(bincode::deserialize::<$from>(&$var)
                        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?)
                };
            }

            let mut raw = self.recv_raw().await?;
            if raw.get_u8() != 0xfe {
                return Err(io::Error::new(
                    io::ErrorKind::NetworkDown,
                    "network unavailable",
                ));
            }
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
                PacketID::ChatStreamRequest => {
                    deserialize!(chat::StreamRequest, Packet::ChatStreamRequest, raw)
                }
                PacketID::ChatStreamResponse => {
                    deserialize!(chat::StreamResponse, Packet::ChatStreamResponse, raw)
                }
                PacketID::BenchUnreliableRequest => {
                    deserialize!(
                        bench::UnreliableRequest,
                        Packet::BenchUnreliableRequest,
                        raw
                    )
                }
                PacketID::BenchUnreliableResponse => {
                    deserialize!(
                        bench::UnreliableResponse,
                        Packet::BenchUnreliableResponse,
                        raw
                    )
                }
                PacketID::BenchCommutativeRequest => {
                    deserialize!(
                        bench::CommutativeRequest,
                        Packet::BenchCommutativeRequest,
                        raw
                    )
                }
                PacketID::BenchCommutativeResponse => {
                    deserialize!(
                        bench::CommutativeResponse,
                        Packet::BenchCommutativeResponse,
                        raw
                    )
                }
                PacketID::BenchOrderedRequest => {
                    deserialize!(bench::OrderedRequest, Packet::BenchOrderedRequest, raw)
                }
                PacketID::BenchOrderedResponse => {
                    deserialize!(bench::OrderedResponse, Packet::BenchOrderedResponse, raw)
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

impl Tx for UnboundedSender<Message> {
    async fn send_raw(&self, msg: Message) -> io::Result<()> {
        self.send(msg)
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "flusher closed"))
    }
}

impl<R: PinReader> Rx for R {
    async fn recv_raw(&mut self) -> io::Result<Bytes> {
        self.next()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::ConnectionAborted, "connection closed"))
    }
}
