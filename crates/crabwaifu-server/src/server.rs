use crabml::gguf::{GGUFFile, GGUFFileLoader, GGUFMetadataValueType};
use crabml_llama2::llama2::Llama2Runner;
use crabml_llama2::model::{CpuLlamaModelLoader, LlamaConfig};
use crabml_llama2::CpuLlamaModel;
use crabwaifu_common::network::spawn_flush_task;
use futures::StreamExt;
use raknet_rs::server::MakeIncoming;
use tokio::net;
use tokio::sync::watch;

use crate::config::{self, Config};
use crate::session;

fn setup_llama_model(
    config: config::CrabLlamaConfig,
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

pub async fn serve(config: Config) -> anyhow::Result<()> {
    let default_steps = config.llama.steps;
    let (llama_model, conf) = setup_llama_model(config.llama)?;
    println!("server listens on {}", config.listen_addr);
    let mut incoming = net::UdpSocket::bind(config.listen_addr)
        .await?
        .make_incoming(config.raknet);
    let (shutdown, watcher) = watch::channel("running");
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    loop {
        tokio::select! {
            Some((rx, writer)) = incoming.next() => {
                let (tx, flush_notify, close_notify) = spawn_flush_task(writer);
                let runner = Llama2Runner::new(&llama_model, conf.seq_len, true)
                    .expect("llama runner cannot be initialized");
                let session = session::Session::new(tx, Box::pin(rx), flush_notify, close_notify, runner, default_steps);
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
                        return Ok(());
                    }
                }
            }
        }
    }
    Ok(())
}
