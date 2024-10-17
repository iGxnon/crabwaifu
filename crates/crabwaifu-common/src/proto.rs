// Enumeration containing all packets for receiving and dispatching, not for ser/de.
#[derive(Debug, Clone)]
pub enum Packet {
    ChatRequest(chat::Request),
    ChatResponse(chat::Response),
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum PacketID {
    InvalidPack = 0,
    ChatRequest = 1,
    ChatResponse = 2,
}

impl PacketID {
    pub fn from_u8(id: u8) -> Self {
        match id {
            1 => PacketID::ChatRequest,
            2 => PacketID::ChatResponse,
            _ => PacketID::InvalidPack,
        }
    }
}

/// Chat completion prototypes
/// These APIs should be stateless
pub mod chat {
    use raknet_rs::Reliability;
    use serde::{Deserialize, Serialize};

    use super::PacketID;
    use crate::network::Pack;

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct Request {
        pub messages: Vec<Message>,
        // i.e max_tokens
        pub steps: Option<usize>,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct Message {
        pub role: Role,
        // only support string content
        pub content: String,
    }

    #[derive(Debug, Clone, Copy, Deserialize, Serialize)]
    pub enum Role {
        User,
        System,
        Assistant,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct Response {
        pub message: Message,
    }

    impl Request {
        pub fn validate(&self) -> bool {
            let system_prompt_count = self
                .messages
                .iter()
                .filter(|msg| matches!(msg.role, Role::System))
                .count();
            let steps = self.steps.unwrap_or_default();
            !self.messages.is_empty() && system_prompt_count <= 1 && steps > 0
        }
    }

    // chat::Request should be in ordered and reliable, it occupied the order channel 0
    impl Pack for Request {
        const ID: PacketID = PacketID::ChatRequest;
        const ORDER_CHANNEL: u8 = 0;
        const RELIABILITY: Reliability = Reliability::ReliableOrdered;
    }

    // chat::Response is as same as chat::Request
    impl Pack for Response {
        const ID: PacketID = PacketID::ChatResponse;
        const ORDER_CHANNEL: u8 = 0;
        const RELIABILITY: Reliability = Reliability::ReliableOrdered;
    }
}

/// Realtime prototypes
pub mod realtime {}
