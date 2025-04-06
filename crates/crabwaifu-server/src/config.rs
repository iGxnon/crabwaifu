use std::net::SocketAddr;

use raknet_rs::server;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llama: CrabLlamaConfig,
    pub whisper: WhisperConfig,
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
pub struct CrabLlamaConfig {
    /// The checkpoint file to load (gguf format)
    pub model: String,
    /// The number of tokens to generate
    pub steps: usize,
    /// The probability of sampling from the top-p.
    pub probability: f32,
    pub temperature: f32,
    pub threads: usize,
    /// The mmap memory lock option, reducing the swap overhead
    pub mlock: bool,
    /// Use f16 kv cache to reduce memory usage
    pub f16_kv_cache: bool,
    /// A session maximum context length, in which will be terminated forcibly if exceeded.
    pub max_context_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    pub model: String,
    pub language: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llama: CrabLlamaConfig {
                model: "./llama.gguf".to_string(),
                steps: 300,
                probability: 0.9,
                temperature: 1.0,
                threads: num_cpus::get(),
                mlock: false,
                f16_kv_cache: true,
                max_context_length: 4096,
            },
            whisper: WhisperConfig {
                model: "./whisper-tiny-ggml-tiny.bin".to_string(),
                language: "en".to_string(),
            },
            listen_addr: "0.0.0.0:8808".parse().unwrap(),
            network: Network::Raknet,
            raknet: Default::default(),
            tcp: TCPConfig {
                ttl: 30,
                nodelay: true,
            },
        }
    }
}
