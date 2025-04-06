use std::cmp;
use std::net::SocketAddr;

use crabml::gguf::{GGUFFile, GGUFFileLoader, GGUFMetadataValueType};
use crabml_llama2::llama2::Llama2Runner;
use crabml_llama2::model::CpuLlamaModelLoader;
use crabml_llama2::CpuLlamaModel;
use crabwaifu_common::network::{
    spawn_flush_task, tcp_split, PinReader, PinWriter, TcpListenerStream,
};
use futures::{Stream, StreamExt};
use raknet_rs::server::MakeIncoming;
use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::watch;
use whisper_rs::{WhisperContext, WhisperContextParameters};

use crate::config::{self, Config, TCPConfig};
use crate::session;

fn setup_llama_model(config: &config::CrabLlamaConfig) -> anyhow::Result<CpuLlamaModel<'static>> {
    let gl = GGUFFileLoader::new(&config.model, config.mlock)?;
    let gl: &'static GGUFFileLoader = Box::leak(Box::new(gl));
    let gf: &'static GGUFFile<'static> = Box::leak(Box::new(gl.open()?));
    dump_gguf_metadata(gf);

    let model_cpu = CpuLlamaModelLoader::new()
        .with_thread_num(config.threads)
        .with_temperature(config.temperature)
        .with_probability(config.probability)
        .load(gf)?;

    Ok(model_cpu)
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
    let llama_model = setup_llama_model(&config.llama)?;
    let whisper_model = setup_whisper_model(&config.whisper)?;

    let seq_len = cmp::max(llama_model.conf.seq_len, config.llama.max_context_length);
    let default_steps = config.llama.steps;
    let f16_kv_cache = config.llama.f16_kv_cache;
    let language = config.whisper.language;

    let (shutdown, watcher) = watch::channel("running");
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    tokio::pin!(incoming);

    loop {
        tokio::select! {
            Some((rx, writer)) = incoming.next() => {
                let (tx, flush_notify, close_notify, task) = spawn_flush_task(writer);
                let llama_runner = Llama2Runner::new(&llama_model, seq_len, f16_kv_cache)
                    .expect("llama runner cannot be initialized");
                let whisper_runner = whisper_model.create_state()
                    .expect("whisper runner cannot be initialized");

                let session = session::Session::new(
                    tx,
                    rx,
                    flush_notify,
                    close_notify,
                    llama_runner,
                    default_steps,
                    whisper_runner,
                    language.clone(),
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
