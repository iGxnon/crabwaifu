use serde::{Deserialize, Serialize};

// Enumeration containing all packets for receiving and dispatching.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub enum Packet {
    Ping,
    Pong,
}

/// Chat prototypes
pub mod chat {

}

/// Voice prototypes (Realtime)
pub mod voice {

}
