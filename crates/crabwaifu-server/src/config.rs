use std::net::SocketAddr;

use raknet_rs::server;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub llama: CrabLlamaConfig,
    pub raknet: server::Config,
    pub listen_addr: SocketAddr,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CrabLlamaConfig {
    /// The checkpoint file to load (gguf format)
    pub model: String,
    /// The number of tokens to generate
    pub steps: usize,
    /// The probability of sampling from the top-p.
    pub probability: f32,
    /// The system prompt
    pub prompt: Option<String>,
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
                prompt: None,
                temperature: 1.0,
                threads: num_cpus::get(),
                mlock: false,
            },
            raknet: Default::default(),
            listen_addr: "0.0.0.0:8808".parse().unwrap(),
        }
    }
}
