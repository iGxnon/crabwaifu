use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::{FutureExt, Sink, Stream};
use raknet_rs::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};

use super::{PinReader, PinWriter, WriterInfo};

pub fn tcp_split(stream: TcpStream, flush: bool) -> (impl PinReader, impl PinWriter) {
    let (reader, writer) = stream.into_split();
    (tcp_reader(reader), tcp_writer(writer, flush))
}

fn tcp_reader(mut reader: OwnedReadHalf) -> impl PinReader {
    let stream = {
        #[futures_async_stream::stream]
        async move {
            loop {
                match reader.read_u32().await {
                    Ok(size) => {
                        let mut buffer = vec![0; size as usize];
                        if let Err(err) = reader.read_exact(&mut buffer).await {
                            log::warn!("tcp reader error: {err:?}");
                            return;
                        }
                        yield Bytes::from(buffer);
                    }
                    Err(err) => {
                        log::warn!("tcp reader error: {err:?}");
                        return;
                    }
                }
            }
        }
    };
    Box::pin(stream)
}

fn tcp_writer(writer: OwnedWriteHalf, flush: bool) -> impl PinWriter {
    TcpWriter {
        inner: writer,
        flush,
    }
}

struct TcpWriter {
    inner: OwnedWriteHalf,
    flush: bool,
}

impl Sink<Message> for TcpWriter {
    type Error = io::Error;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let fut = self.inner.writable(); // cancellable
        tokio::pin!(fut);
        fut.poll_unpin(cx)
    }

    fn start_send(self: Pin<&mut Self>, msg: Message) -> Result<(), Self::Error> {
        // len(4B) + data
        let size = msg.get_data().len();
        self.inner.try_write(&(size as u32).to_be_bytes())?;
        let written = self.inner.try_write(msg.get_data())?;
        if written != size {
            Err(io::Error::new(io::ErrorKind::Other, "tcp buffer full"))
        } else {
            Ok(())
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        if self.flush {
            let fut = self.inner.flush(); // cancellable
            tokio::pin!(fut);
            fut.poll_unpin(cx)
        } else {
            // let tcp stack to decide when to flush
            Poll::Ready(Ok(()))
        }
    }

    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let fut = self.inner.shutdown(); // cancellable
        tokio::pin!(fut);
        fut.poll_unpin(cx)
    }
}

impl WriterInfo for TcpWriter {
    fn mtu(&self) -> usize {
        // no mtu
        0
    }
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
