use std::net::SocketAddr;

use raknet_rs::server;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llama: CrabLlamaConfig,
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
