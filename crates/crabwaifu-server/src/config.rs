use std::net::SocketAddr;

use raknet_rs::server;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: Vec<CrabLLMConfig>,
    pub whisper: WhisperConfig,
    pub kokoro: KokoroConfig,

    pub listen_addr: SocketAddr,
    pub network: Network,
    pub raknet: server::Config,
    pub tcp: TCPConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Network {
    Raknet,
    TCP,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCPConfig {
    pub ttl: u32,
    pub nodelay: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrabLLMConfig {
    /// The name of this model
    pub name: String,
    /// The checkpoint file to load (gguf format)
    pub model: String,
    /// The probability of sampling from the top-p.
    pub probability: f32,
    pub temperature: f32,
    pub threads: usize,
    /// The mmap memory lock option, reducing the swap overhead
    pub mlock: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroConfig {
    pub model: String,
    pub voice: String,
    pub style: String,
    pub speed: f32,
}
