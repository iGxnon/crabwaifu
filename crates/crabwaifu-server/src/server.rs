use std::cmp;
use std::collections::HashMap;
use std::net::SocketAddr;

use crabml::gguf::{GGUFFile, GGUFFileLoader, GGUFMetadataValueType};
use crabml_llama2::llama2::Llama2Runner;
use crabml_llama2::model::CpuLlamaModelLoader;
use crabml_llama2::CpuLlamaModel;
use crabwaifu_common::network::{
    spawn_flush_task, tcp_split, PinReader, PinWriter, TcpListenerStream,
};
use futures::{Stream, StreamExt};
use kokoros::tts::koko::TTSKoko;
use raknet_rs::server::MakeIncoming;
use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::watch;
use whisper_rs::{WhisperContext, WhisperContextParameters};

use crate::config::{self, Config, TCPConfig};
use crate::session::{self, LLMState, TTState};

fn setup_llm_models(
    configs: &[config::CrabLLMConfig],
) -> anyhow::Result<HashMap<String, CpuLlamaModel<'static>>> {
    let mut ret = HashMap::new();
    for config in configs {
        let gl = GGUFFileLoader::new(&config.model, config.mlock)?;
        let gl: &'static GGUFFileLoader = Box::leak(Box::new(gl));
        let gf: &'static GGUFFile<'static> = Box::leak(Box::new(gl.open()?));
        dump_gguf_metadata(gf);
        let model_cpu = CpuLlamaModelLoader::new()
            .with_thread_num(config.threads)
            .with_temperature(config.temperature)
            .with_probability(config.probability)
            .load(gf)?;
        ret.insert(config.name.clone(), model_cpu);
    }
    Ok(ret)
}

fn build_llm_states(models: &HashMap<String, CpuLlamaModel<'static>>) -> HashMap<String, LLMState> {
    let mut ret = HashMap::new();
    for (name, model) in models {
        let runner = Llama2Runner::new(model, cmp::max(4096, model.conf.seq_len), true)
            .expect("llama runner cannot be initialized");
        ret.insert(name.clone(), LLMState::new(runner));
    }
    ret
}

fn setup_whisper_model(config: &config::WhisperConfig) -> anyhow::Result<WhisperContext> {
    let ctx = WhisperContext::new_with_params(&config.model, WhisperContextParameters::default())?;
    Ok(ctx)
}

fn dump_gguf_metadata(gf: &GGUFFile) {
    log::debug!("dump model:");
    for (key, value) in gf.metadata().as_hashmap() {
        if value.typ() != GGUFMetadataValueType::Array {
            log::debug!("{}: {:?}", key, value);
        }
    }
    for tensor in gf.tensor_infos() {
        log::debug!(
            "- {} \t\t\t {} \t {:?}",
            tensor.name(),
            tensor.typ(),
            tensor.dimensions()
        );
    }
}

pub async fn make_tcp_incoming(
    listen_addr: SocketAddr,
    config: TCPConfig,
) -> anyhow::Result<impl Stream<Item = (impl PinReader, impl PinWriter)>> {
    let listener = TcpListener::bind(listen_addr).await?;
    listener.set_ttl(config.ttl)?;
    let stream = TcpListenerStream::new(listener).map(move |res| {
        let stream = res.expect("failed to init tcp stream");
        log::debug!(
            "accept new tcp connection, default tcp no_delay: {}",
            stream.nodelay().expect("cannot get nodelay")
        );
        stream
            .set_nodelay(config.nodelay)
            .expect("cannot set nodelay");
        tcp_split(stream)
    });
    Ok(stream)
}

pub async fn make_raknet_incoming(
    listen_addr: SocketAddr,
    config: raknet_rs::server::Config,
) -> anyhow::Result<impl Stream<Item = (impl PinReader, impl PinWriter)>> {
    let stream = UdpSocket::bind(listen_addr)
        .await?
        .make_incoming(config)
        .map(|(reader, writer)| (Box::pin(reader), writer));
    Ok(stream)
}

pub async fn serve(
    incoming: impl Stream<Item = (impl PinReader, impl PinWriter)>,
    config: Config,
) -> anyhow::Result<()> {
    let llm_models = setup_llm_models(&config.llm)?;
    let whisper_model = setup_whisper_model(&config.whisper)?;
    let tts_model = TTSKoko::new(&config.kokoro.model, &config.kokoro.voice).await;
    let (shutdown, watcher) = watch::channel("running");
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    tokio::pin!(incoming);

    loop {
        tokio::select! {
            Some((rx, writer)) = incoming.next() => {
                let (tx, flush_notify, close_notify, task) = spawn_flush_task(writer);
                let llm_runners = build_llm_states(&llm_models);
                let whisper_runner = whisper_model.create_state()
                    .expect("whisper runner cannot be initialized");
                let tts_runner = TTState::new(tts_model.clone(), config.kokoro.style.clone(), config.kokoro.speed);
                let session = session::Session::new(
                    tx,
                    rx,
                    flush_notify,
                    close_notify,
                    llm_runners,
                    whisper_runner,
                    tts_runner,
                    task,
                );
                tokio::task::spawn(session.run(watcher.clone()));
            }
            _ = &mut ctrl_c => {
                eprintln!(" received ctrl-c signal, press again to force shutdown");
                drop(watcher);
                if shutdown.receiver_count() == 0 {
                    break;
                }
                shutdown.send("shutdown").unwrap();
                let ctrl_c_c = tokio::signal::ctrl_c();
                tokio::select! {
                    _ = shutdown.closed() => {
                        break;
                    }
                    _ = ctrl_c_c => {
                        log::warn!("force shutdown");
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}
