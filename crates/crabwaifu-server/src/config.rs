use std::net::SocketAddr;

use raknet_rs::server;

pub struct Config {
    pub llama: CrabLlamaConfig,
    pub raknet: server::Config,
    pub listen_addr: SocketAddr,
}

pub struct CrabLlamaConfig {
    /// The checkpoint file to load (gguf format)
    model: String,
    /// The number of tokens to generate
    steps: usize,
    /// The probability of sampling from the top-p.
    probability: f32,
    /// The system prompt
    prompt: Option<String>,
    temperature: f32,
    threads: usize,
    mlock: bool,
}
