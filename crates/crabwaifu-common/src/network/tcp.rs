use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::BytesMut;
use futures::{SinkExt, Stream, StreamExt};
use raknet_rs::Message;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::codec::LengthDelimitedCodec;

use super::{PinReader, PinWriter};

pub fn tcp_split(stream: TcpStream) -> (impl PinReader, impl PinWriter) {
    let (reader, writer) = stream.into_split();
    let r = LengthDelimitedCodec::builder()
        .new_read(reader)
        .filter_map(|v| async move { v.ok().map(BytesMut::freeze) });
    let w = LengthDelimitedCodec::builder()
        .new_write(writer)
        .with(|msg: Message| async move { Ok(msg.data) });
    (Box::pin(r), Box::pin(w))
}

/// A wrapper around [`TcpListener`] that implements [`Stream`].
///
/// [`TcpListener`]: struct@tokio::net::TcpListener
/// [`Stream`]: trait@crate::Stream
#[derive(Debug)]
#[cfg_attr(docsrs, doc(cfg(feature = "net")))]
pub struct TcpListenerStream {
    inner: TcpListener,
}

impl TcpListenerStream {
    /// Create a new `TcpListenerStream`.
    pub fn new(listener: TcpListener) -> Self {
        Self { inner: listener }
    }
}

impl Stream for TcpListenerStream {
    type Item = io::Result<TcpStream>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<io::Result<TcpStream>>> {
        match self.inner.poll_accept(cx) {
            Poll::Ready(Ok((stream, _))) => Poll::Ready(Some(Ok(stream))),
            Poll::Ready(Err(err)) => Poll::Ready(Some(Err(err))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl AsRef<TcpListener> for TcpListenerStream {
    fn as_ref(&self) -> &TcpListener {
        &self.inner
    }
}

impl AsMut<TcpListener> for TcpListenerStream {
    fn as_mut(&mut self) -> &mut TcpListener {
        &mut self.inner
    }
}
