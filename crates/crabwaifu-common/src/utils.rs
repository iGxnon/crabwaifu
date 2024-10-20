use std::future::Future;
use std::time::Duration;

use tokio::time::Instant;

pub trait TimeoutWrapper: Sized {
    fn timeout(self, dur: Duration) -> tokio::time::Timeout<Self>;

    fn timeout_at(self, at: Instant) -> tokio::time::Timeout<Self>;
}

impl<F> TimeoutWrapper for F
where
    F: Future,
{
    fn timeout(self, dur: Duration) -> tokio::time::Timeout<Self> {
        tokio::time::timeout(dur, self)
    }

    fn timeout_at(self, at: Instant) -> tokio::time::Timeout<Self> {
        tokio::time::timeout_at(at, self)
    }
}
