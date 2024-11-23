#![feature(context_ext)]
#![feature(local_waker)]
#![feature(trait_alias)]
#![feature(proc_macro_hygiene)]
#![feature(coroutines)]
#![feature(stmt_expr_attributes)]
#![feature(io_error_more)]

/// Prototypes in network transmission
pub mod proto;

/// Common network part
pub mod network;

pub mod utils;
