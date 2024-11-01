use std::cmp;
use std::net::SocketAddr;

use crabml::gguf::{GGUFFile, GGUFFileLoader, GGUFMetadataValueType};
use crabml_llama2::llama2::Llama2Runner;
use crabml_llama2::model::{CpuLlamaModelLoader, LlamaConfig};
use crabml_llama2::CpuLlamaModel;
use crabwaifu_common::network::{
    spawn_flush_task, tcp_split, PinReader, PinWriter, TcpListenerStream,
};
use futures::{Stream, StreamExt};
use raknet_rs::server::MakeIncoming;
use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::watch;

use crate::config::{self, CrabLlamaConfig, TCPConfig};
use crate::session;

fn setup_llama_model(
    config: &config::CrabLlamaConfig,
) -> anyhow::Result<(CpuLlamaModel<'static>, LlamaConfig)> {
    let gl = GGUFFileLoader::new(&config.model, config.mlock)?;
    let gl: &'static GGUFFileLoader = Box::leak(Box::new(gl));
    let gf: &'static GGUFFile<'static> = Box::leak(Box::new(gl.open()?));
    dump_gguf_metadata(gf);

    let model_cpu = CpuLlamaModelLoader::new()
        .with_thread_num(config.threads)
        .with_temperature(config.temperature)
        .with_probability(config.probability)
        .load(gf)?;
    let conf = model_cpu.conf.clone();

    Ok((model_cpu, conf))
}

fn dump_gguf_metadata(gf: &GGUFFile) {
    log::trace!("dump model:");
    for (key, value) in gf.metadata().as_hashmap() {
        if value.typ() != GGUFMetadataValueType::Array {
            log::trace!("{}: {:?}", key, value);
        }
    }
    for tensor in gf.tensor_infos() {
        log::trace!(
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
    llama_config: CrabLlamaConfig,
) -> anyhow::Result<()> {
    let (llama_model, conf) = setup_llama_model(&llama_config)?;

    let seq_len = cmp::max(conf.seq_len, llama_config.max_context_length);
    let default_steps = llama_config.steps;
    let f16_kv_cache = llama_config.f16_kv_cache;

    let (shutdown, watcher) = watch::channel("running");
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    tokio::pin!(incoming);

    loop {
        tokio::select! {
            Some((rx, writer)) = incoming.next() => {
                let (tx, flush_notify, close_notify, task) = spawn_flush_task(writer);
                let runner = Llama2Runner::new(&llama_model, seq_len, f16_kv_cache)
                    .expect("llama runner cannot be initialized");
                let session = session::Session::new(
                    tx,
                    rx,
                    flush_notify,
                    close_notify,
                    runner,
                    default_steps,
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
