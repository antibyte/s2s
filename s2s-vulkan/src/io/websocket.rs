//! Raw PCM WebSocket mode plus the local speech-lab control API.

use crate::audio::pcm::i16_to_bytes_le;
use crate::benchmark::{BenchmarkRating, BenchmarkRequest, BenchmarkService};
use crate::config::Config;
use crate::gpu::GpuReport;
use crate::lab::{ActivateStackRequest, LabController};
use crate::messages::{Control, PipelineEvent, QueueItem};
use crate::pipeline::{spawn_pipeline_shared, PipelineHandles};
use crate::registry::{BackendCatalog, HardwareProfile};
use crate::runtime::{self, SharedRuntime, StackSelection, StackStatus};
use anyhow::Result;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, Query, State, WebSocketUpgrade};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

#[derive(Clone)]
struct AppState {
    runtime: SharedRuntime,
    lab: LabController,
    benchmarks: BenchmarkService,
}

pub async fn run_websocket_server(cfg: Config, gpu_report: GpuReport) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse()?;
    let runtime = runtime::runtime_from(cfg);
    let catalog = BackendCatalog::load()?;
    let hardware = HardwareProfile::from_gpu_report(&gpu_report);
    let lab = LabController::new(catalog, hardware, runtime.clone())?;
    lab.reconcile_active_stack().await;
    lab.start_llm_monitor();
    let benchmarks = BenchmarkService::new(runtime.clone(), serde_json::to_value(&gpu_report)?)?;
    let state = AppState {
        runtime,
        lab,
        benchmarks,
    };

    let app = Router::new()
        .route("/", get(ws_upgrade))
        .route("/ws", get(ws_upgrade))
        .route("/api/v1/catalog", get(get_catalog))
        .route("/api/v1/stack", get(get_stack).put(put_stack))
        .route(
            "/api/v1/models/{backend_id}/download",
            axum::routing::post(post_model_download).delete(delete_model_download),
        )
        .route(
            "/api/v1/models/{backend_id}",
            axum::routing::delete(delete_model),
        )
        .route("/api/v1/benchmarks", axum::routing::post(post_benchmark))
        .route("/api/v1/benchmarks/{id}", get(get_benchmark))
        .route(
            "/api/v1/benchmarks/{id}/audio/{sample_index}",
            get(get_benchmark_audio),
        )
        .route(
            "/api/v1/benchmarks/{id}/ratings",
            axum::routing::post(post_benchmark_rating),
        )
        .layer(CorsLayer::permissive())
        .with_state(state);

    info!("Speech lab listening on http://{addr} (WebSocket PCM + catalog/stack/benchmark API)");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn get_catalog(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.lab.catalog().await)
}

async fn post_model_download(
    State(state): State<AppState>,
    Path(backend_id): Path<String>,
) -> Response {
    match state.lab.start_model_download(&backend_id).await {
        Ok(result) => (StatusCode::ACCEPTED, Json(result)).into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("{error:#}") })),
        )
            .into_response(),
    }
}

async fn delete_model_download(
    State(state): State<AppState>,
    Path(backend_id): Path<String>,
) -> Response {
    match state.lab.cancel_model_download(&backend_id).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": format!("{error:#}") })),
        )
            .into_response(),
    }
}

async fn delete_model(State(state): State<AppState>, Path(backend_id): Path<String>) -> Response {
    match state.lab.delete_model(&backend_id).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": format!("{error:#}") })),
        )
            .into_response(),
    }
}

async fn get_stack(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.lab.status().await)
}

async fn put_stack(
    State(state): State<AppState>,
    Json(request): Json<ActivateStackRequest>,
) -> impl IntoResponse {
    Json(state.lab.activate(request).await)
}

async fn post_benchmark(
    State(state): State<AppState>,
    Json(request): Json<BenchmarkRequest>,
) -> Response {
    match state.benchmarks.start(request).await {
        Ok(run) => (StatusCode::ACCEPTED, Json(run)).into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("{error:#}") })),
        )
            .into_response(),
    }
}

#[derive(Debug, Default, Deserialize)]
struct BenchmarkQuery {
    #[serde(default)]
    format: String,
}

async fn get_benchmark(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<BenchmarkQuery>,
) -> Response {
    match state.benchmarks.get(&id).await {
        Ok(Some(run)) if query.format.eq_ignore_ascii_case("csv") => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/csv; charset=utf-8")],
            run.to_csv(),
        )
            .into_response(),
        Ok(Some(run)) => Json(run).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "benchmark not found" })),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("{error:#}") })),
        )
            .into_response(),
    }
}

async fn get_benchmark_audio(
    State(state): State<AppState>,
    Path((id, sample_index)): Path<(String, usize)>,
) -> Response {
    match state.benchmarks.audio(&id, sample_index).await {
        Ok(Some(bytes)) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "audio/wav"),
                (header::CACHE_CONTROL, "private, max-age=3600"),
            ],
            bytes,
        )
            .into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("{error:#}") })),
        )
            .into_response(),
    }
}

async fn post_benchmark_rating(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(rating): Json<BenchmarkRating>,
) -> Response {
    match state.benchmarks.rate(&id, rating).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("{error:#}") })),
        )
            .into_response(),
    }
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

#[derive(Debug, Deserialize)]
struct ClientMsg {
    #[serde(default)]
    r#type: String,
    #[serde(flatten)]
    stack: StackSelection,
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    info!("WebSocket client connected");
    // Session lifecycle: cancel idle unload and warm parked models if needed.
    state.lab.session_connected().await;

    let handles = spawn_pipeline_shared(state.runtime.clone());
    let PipelineHandles {
        audio_in_tx,
        mut audio_out_rx,
        mut event_rx,
        should_listen: _,
        runtime: _,
        join,
    } = handles;

    let (sink, mut stream) = socket.split();
    let sink = Arc::new(Mutex::new(sink));

    send_stack_event(&sink, runtime_status(&state.runtime).await, "connected").await;

    let out_sink = sink.clone();
    let play_task = tokio::spawn(async move {
        while let Some(chunk) = audio_out_rx.recv().await {
            if chunk.response_done && chunk.pcm_i16.is_empty() {
                continue;
            }
            let bytes = i16_to_bytes_le(&chunk.pcm_i16);
            let mut socket = out_sink.lock().await;
            if socket.send(Message::Binary(bytes.into())).await.is_err() {
                break;
            }
        }
    });

    let event_sink = sink.clone();
    let event_task = tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            let Ok(json) = serde_json::to_string(&event) else {
                continue;
            };
            let mut socket = event_sink.lock().await;
            if socket.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    });

    let lab_event_sink = sink.clone();
    let mut lab_events = state.lab.subscribe();
    let lab_event_task = tokio::spawn(async move {
        loop {
            match lab_events.recv().await {
                Ok(event) => {
                    let Ok(json) = serde_json::to_string(&event) else {
                        continue;
                    };
                    let mut socket = lab_event_sink.lock().await;
                    if socket.send(Message::Text(json.into())).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    while let Some(message) = stream.next().await {
        match message {
            Ok(Message::Binary(data)) => {
                if audio_in_tx
                    .send(QueueItem::Data(data.to_vec()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Ok(Message::Text(text)) => {
                let text = text.to_string();
                if text == "END" || text == "session_end" {
                    let _ = audio_in_tx
                        .send(QueueItem::Control(Control::SessionEnd))
                        .await;
                    continue;
                }
                let Ok(mut client_message) = serde_json::from_str::<ClientMsg>(&text) else {
                    continue;
                };
                let message_type = client_message.r#type.to_ascii_lowercase();
                if message_type == "set_stack" || message_type == "stack" {
                    let request = ActivateStackRequest {
                        asr_id: client_message.stack.asr.take(),
                        tts_id: client_message.stack.tts.take(),
                        llm_id: client_message.stack.llm.take(),
                        voice: None,
                    };
                    let lab_status = state.lab.activate(request).await;

                    // Keep advanced legacy overrides compatible, but catalog
                    // ids now own ASR/TTS/LLM lifecycle and routing.
                    let (mut status, _) =
                        runtime::apply_stack(&state.runtime, client_message.stack).await;
                    if !lab_status.ok {
                        status.ok = false;
                        status.message = lab_status.message;
                    }
                    send_stack_event(&sink, status, "stack").await;
                } else if message_type == "get_stack" {
                    send_stack_event(&sink, runtime_status(&state.runtime).await, "status").await;
                } else if message_type == "session_end" || message_type == "end" {
                    let _ = audio_in_tx
                        .send(QueueItem::Control(Control::SessionEnd))
                        .await;
                } else {
                    warn!("Unknown client control type: {}", client_message.r#type);
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(_) => {}
            Err(error) => {
                error!("WebSocket error: {error}");
                break;
            }
        }
    }

    let _ = audio_in_tx.send(QueueItem::end()).await;
    play_task.abort();
    event_task.abort();
    lab_event_task.abort();
    for handle in join {
        handle.abort();
    }
    // After the last client leaves, schedule unload of model containers (2 min default).
    state.lab.session_disconnected().await;
    info!("WebSocket client disconnected");
}

async fn runtime_status(runtime: &SharedRuntime) -> StackStatus {
    let guard = runtime.read().await;
    runtime::status_of(&guard)
}

async fn send_stack_event<S>(sink: &Arc<Mutex<S>>, status: StackStatus, default_message: &str)
where
    S: futures_util::Sink<Message> + Unpin,
{
    let event = PipelineEvent::Stack {
        asr: status.asr,
        tts: status.tts,
        llm: status.llm,
        whisper_url: status.whisper_url,
        llm_base_url: status.llm_base_url,
        model_name: status.model_name,
        tts_backend: status.tts_backend,
        tts_url: status.tts_url,
        voice: status.voice,
        language: status.language,
        ok: status.ok,
        message: if status.message.is_empty() {
            default_message.into()
        } else {
            status.message
        },
    };
    if let Ok(json) = serde_json::to_string(&event) {
        let mut socket = sink.lock().await;
        let _ = socket.send(Message::Text(json.into())).await;
    }
}
