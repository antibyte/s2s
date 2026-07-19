//! Raw PCM WebSocket mode (16 kHz mono i16 LE), like HF `--mode websocket`.

use crate::audio::pcm::i16_to_bytes_le;
use crate::config::Config;
use crate::messages::{Control, QueueItem};
use crate::pipeline::{spawn_pipeline, PipelineHandles};
use anyhow::Result;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::WebSocketUpgrade;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

pub async fn run_websocket_server(cfg: Config) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse()?;
    let cfg = Arc::new(cfg);

    let app = Router::new()
        .route("/", get(ws_upgrade))
        .route("/ws", get(ws_upgrade))
        .layer(CorsLayer::permissive())
        .with_state(cfg.clone());

    info!("WebSocket PCM server listening on ws://{addr}  (16 kHz mono i16 LE)");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    axum::extract::State(cfg): axum::extract::State<Arc<Config>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, cfg))
}

async fn handle_socket(socket: WebSocket, cfg: Arc<Config>) {
    info!("WebSocket client connected");
    let handles = spawn_pipeline((*cfg).clone());
    let PipelineHandles {
        audio_in_tx,
        mut audio_out_rx,
        should_listen: _,
        join,
    } = handles;

    let (sink, mut stream) = socket.split();
    let sink = Arc::new(Mutex::new(sink));

    let out_sink = sink.clone();
    let play_task = tokio::spawn(async move {
        while let Some(chunk) = audio_out_rx.recv().await {
            if chunk.response_done && chunk.pcm_i16.is_empty() {
                // Optional marker: empty binary frame or skip.
                continue;
            }
            let bytes = i16_to_bytes_le(&chunk.pcm_i16);
            let mut s = out_sink.lock().await;
            if s.send(Message::Binary(bytes.into())).await.is_err() {
                break;
            }
        }
    });

    while let Some(msg) = stream.next().await {
        match msg {
            Ok(Message::Binary(data)) => {
                if audio_in_tx
                    .send(QueueItem::Data(data.to_vec()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Ok(Message::Text(t)) => {
                if t == "END" || t == "session_end" {
                    let _ = audio_in_tx
                        .send(QueueItem::Control(Control::SessionEnd))
                        .await;
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(_) => {}
            Err(e) => {
                error!("WebSocket error: {e}");
                break;
            }
        }
    }

    let _ = audio_in_tx.send(QueueItem::end()).await;
    play_task.abort();
    for h in join {
        h.abort();
    }
    info!("WebSocket client disconnected");
}
