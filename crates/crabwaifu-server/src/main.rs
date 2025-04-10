use std::collections::{HashMap, HashSet};
use std::env::args;

use crabwaifu_server::config::{self, Config, Network};
use crabwaifu_server::db;
use crabwaifu_server::server::{make_raknet_incoming, make_tcp_incoming, serve};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let config: Config = args()
        .nth(1)
        .map(std::fs::read_to_string)
        .transpose()
        .unwrap()
        .as_deref()
        .map(toml::from_str)
        .transpose()
        .unwrap()
        .unwrap();
    log::debug!(
        "server configuration: \n{}",
        serde_json::to_string_pretty(&config).unwrap()
    );

    let model_mapping = refresh_db_models(&config.llm).await?;

    match config.network {
        Network::Raknet => {
            println!("server is listening on raknet://{}", config.listen_addr);
            let incoming = make_raknet_incoming(config.listen_addr, config.raknet.clone()).await?;
            serve(incoming, config, &model_mapping).await
        }
        Network::TCP => {
            println!("server is listening on tcp://{}", config.listen_addr);
            let incoming = make_tcp_incoming(config.listen_addr, config.tcp.clone()).await?;
            serve(incoming, config, &model_mapping).await
        }
        Network::Both => {
            let c1 = config.clone();
            let mapping = model_mapping.clone();
            let raknet = async move {
                let incoming = make_raknet_incoming(config.listen_addr, c1.raknet.clone()).await?;
                serve(incoming, c1, &mapping).await
            };
            let c2 = config.clone();
            let mapping = model_mapping.clone();
            let tcp = async move {
                let incoming = make_tcp_incoming(config.listen_addr, c2.tcp.clone()).await?;
                serve(incoming, c2, &mapping).await
            };
            let rak_handle = tokio::spawn(raknet);
            let tcp_handle = tokio::spawn(tcp);
            rak_handle.await??;
            tcp_handle.await??;
            Ok(())
        }
    }
}

async fn refresh_db_models(
    configs: &[config::CrabLLMConfig],
) -> anyhow::Result<HashMap<String, i32>> {
    let loaded_models = configs
        .iter()
        .map(|c| c.name.to_owned())
        .collect::<HashSet<_>>();
    let models: Vec<(i32, String)> = sqlx::query_as("SELECT id, name FROM models")
        .fetch_all(db::pool())
        .await?;
    let mut models = models
        .into_iter()
        .fold(HashMap::new(), |mut map, (id, name)| {
            map.insert(name, id);
            map
        });
    let db_models = models.keys().cloned().collect::<HashSet<_>>();
    for remove in db_models.difference(&loaded_models) {
        sqlx::query("DELETE FROM models WHERE id = $1")
            .bind(models[remove])
            .execute(db::pool())
            .await?;
        log::info!("remove model {remove} from database");
        models.remove(remove);
    }
    for add in loaded_models.difference(&db_models) {
        let (id,): (i32,) = sqlx::query_as("INSERT INTO models (name) VALUES ($1) RETURNING id")
            .bind(add)
            .fetch_one(db::pool())
            .await?;
        log::info!("load model {add} into database");
        models.insert(add.to_owned(), id);
    }
    for (name, id) in &models {
        log::debug!("[DB] model: {name} => id: {id}")
    }
    Ok(models)
}
