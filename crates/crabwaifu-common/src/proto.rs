use raknet_rs::{Priority, Reliability};
use serde::{Deserialize, Serialize};

use crate::network::Pack;

// Enumeration containing all packets for receiving and dispatching, not for ser/de.
#[derive(Debug, Clone)]
pub enum Packet {
    ChatRequest(chat::Request),
    ChatResponse(chat::Response),
    ChatStreamRequest(chat::StreamRequest),
    ChatStreamResponse(chat::StreamResponse),
    UserLoginRequest(user::LoginRequest),
    UserLoginResponse(user::LoginResponse),
    UserRegisterRequest(user::RegisterRequest),
    UserRegisterResponse(user::RegisterResponse),
    UserCleanupRequest(user::CleanupRequest),
    UserCleanupResponse(user::CleanupResponse),
    RealtimeAudioChunk(realtime::RealtimeAudioChunk),

    BenchUnreliableRequest(bench::UnreliableRequest),
    BenchUnreliableResponse(bench::UnreliableResponse),
    BenchCommutativeRequest(bench::CommutativeRequest),
    BenchCommutativeResponse(bench::CommutativeResponse),
    BenchOrderedRequest(bench::OrderedRequest),
    BenchOrderedResponse(bench::OrderedResponse),
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum PacketID {
    InvalidPack = 0,
    ChatRequest = 1,
    ChatResponse = 2,
    ChatStreamRequest = 3,
    ChatStreamResponse = 4,
    UserLoginRequest = 5,
    UserLoginResponse = 6,
    UserRegisterRequest = 7,
    UserRegisterResponse = 8,
    UserCleanupRequest = 9,
    UserCleanupResponse = 10,
    RealtimeAudioChunk = 11,

    BenchUnreliableRequest = 200,
    BenchUnreliableResponse = 201,
    BenchCommutativeRequest = 202,
    BenchCommutativeResponse = 203,
    BenchOrderedRequest = 204,
    BenchOrderedResponse = 205,
}

impl PacketID {
    pub fn from_u8(id: u8) -> Self {
        match id {
            1 => PacketID::ChatRequest,
            2 => PacketID::ChatResponse,
            3 => PacketID::ChatStreamRequest,
            4 => PacketID::ChatStreamResponse,
            5 => PacketID::UserLoginRequest,
            6 => PacketID::UserLoginResponse,
            7 => PacketID::UserRegisterRequest,
            8 => PacketID::UserRegisterResponse,
            9 => PacketID::UserCleanupRequest,
            10 => PacketID::UserCleanupResponse,
            11 => PacketID::RealtimeAudioChunk,

            200 => PacketID::BenchUnreliableRequest,
            201 => PacketID::BenchUnreliableResponse,
            202 => PacketID::BenchCommutativeRequest,
            203 => PacketID::BenchCommutativeResponse,
            204 => PacketID::BenchOrderedRequest,
            205 => PacketID::BenchOrderedResponse,
            _ => PacketID::InvalidPack,
        }
    }
}

/// Chat completion prototypes
pub mod chat {
    use super::*;

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

    /// Oneshot chat request
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct Request {
        pub messages: Vec<Message>,
        // i.e max_tokens
        pub steps: Option<usize>,
    }

    /// Oneshot chat response
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

    // chat::Request should be reliable (may not be ordered)
    impl Pack for Request {
        const ID: PacketID = PacketID::ChatRequest;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    // chat::Response is as same as chat::Request
    impl Pack for Response {
        const ID: PacketID = PacketID::ChatResponse;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    /// Stateful stream request
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct StreamRequest {
        pub prompt: String,
    }

    /// Stateful stream response
    /// 1 StreamRequest : N StreamResponse
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct StreamResponse {
        pub partial: String,
        // when it is true, then the `partial` represents the error message,
        // empty `partial`  means there is no error
        pub eos: bool,
    }

    // it should be reliable and ordered
    impl Pack for StreamRequest {
        const ID: PacketID = PacketID::ChatStreamRequest;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::ReliableOrdered;
    }

    impl Pack for StreamResponse {
        const ID: PacketID = PacketID::ChatStreamResponse;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::ReliableOrdered;
    }
}

pub mod user {
    use super::chat::Message;
    use super::*;

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct LoginRequest {
        pub username: String,
        pub password: String,
    }

    impl Pack for LoginRequest {
        const ID: PacketID = PacketID::UserLoginRequest;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct LoginResponse {
        pub success: bool,
        pub message: String,
        pub context: Vec<Message>,
    }

    impl Pack for LoginResponse {
        const ID: PacketID = PacketID::UserLoginResponse;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct RegisterRequest {
        pub username: String,
        pub password: String,
    }

    impl Pack for RegisterRequest {
        const ID: PacketID = PacketID::UserRegisterRequest;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct RegisterResponse {
        pub success: bool,
        pub message: String,
    }

    impl Pack for RegisterResponse {
        const ID: PacketID = PacketID::UserRegisterResponse;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct CleanupRequest;

    impl Pack for CleanupRequest {
        const ID: PacketID = PacketID::UserCleanupRequest;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct CleanupResponse {
        pub success: bool,
        pub message: String,
    }

    impl Pack for CleanupResponse {
        const ID: PacketID = PacketID::UserCleanupResponse;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }
}

/// Realtime prototypes
pub mod realtime {
    use super::*;

    /// Message sent from client to server containing audio data chunk.
    /// N RealtimeAudioChunk : 1 chat::Response (user prompt) + N chat::StreamResponse
    /// (assistant response)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RealtimeAudioChunk {
        /// Raw audio data bytes.
        pub data: Vec<f32>,
        /// Flag indicating if this is the last chunk for the stream.
        pub is_final: bool,
    }

    impl Pack for RealtimeAudioChunk {
        const ID: PacketID = PacketID::RealtimeAudioChunk;
        // audio channel
        const ORDER_CHANNEL: u8 = 1;
        const PRIORITY: Priority = Priority::Medium;
        // sequenced required here to prevent old audio chunk proceeding
        const RELIABILITY: Reliability = Reliability::UnreliableSequenced;
    }
}

/// Benchmark prototypes
/// TODO: sequenced connection
pub mod bench {
    use super::*;

    /// The client instructs the server to return how much data, and at the same time, the server
    /// will treat this data packet as the end.
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct UnreliableRequest {
        pub data_len: usize,
        pub mtu: usize,
    }

    /// 1 UnreliableRequest : N UnreliableResponse (each of them does not exceed MTU)
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct UnreliableResponse {
        pub data_partial: Vec<u8>,
    }

    impl Pack for UnreliableRequest {
        const ID: PacketID = PacketID::BenchUnreliableRequest;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable; // ensure it is received by peer
    }

    impl Pack for UnreliableResponse {
        const ID: PacketID = PacketID::BenchUnreliableResponse;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Unreliable;
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct CommutativeRequest {
        pub data_len: usize,
        pub batch_size: usize,
    }

    /// 1 CommutativeRequest : N CommutativeResponse (each of them does not depend each other)
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct CommutativeResponse {
        pub data_partial: Vec<u8>,
    }

    impl Pack for CommutativeRequest {
        const ID: PacketID = PacketID::BenchCommutativeRequest;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    impl Pack for CommutativeResponse {
        const ID: PacketID = PacketID::BenchCommutativeResponse;
        const ORDER_CHANNEL: u8 = 0;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct OrderedRequest {
        pub data_len: usize,
        pub batch_size: usize,
    }

    /// 1 OrderedRequest : N OrderedResponse (server will divide it)
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct OrderedResponse {
        pub data_partial: Vec<u8>,
        pub index: usize, // used to check order
    }

    impl Pack for OrderedRequest {
        const ID: PacketID = PacketID::BenchOrderedRequest;
        const ORDER_CHANNEL: u8 = 200;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::Reliable;
    }

    impl Pack for OrderedResponse {
        const ID: PacketID = PacketID::BenchOrderedResponse;
        const ORDER_CHANNEL: u8 = 200;
        const PRIORITY: Priority = Priority::Medium;
        const RELIABILITY: Reliability = Reliability::ReliableOrdered;
    }
}
