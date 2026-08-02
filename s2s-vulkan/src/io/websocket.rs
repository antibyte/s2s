//! Raw PCM WebSocket mode plus the local speech-lab control API.

use crate::audio::pcm::i16_to_bytes_le;
use crate::benchmark::{BenchmarkRating, BenchmarkRequest, BenchmarkService};
use crate::config::Config;
use crate::gateway::{GatewaySpeechRequest, GatewayVoiceNotActive};
use crate::gpu::GpuReport;
use crate::lab::{ActivateStackRequest, LabController};
use crate::messages::{Control, PipelineEvent, QueueItem};
use crate::pipeline::{spawn_pipeline_shared, PipelineHandles};
use crate::registry::{BackendCatalog, HardwareProfile};
use crate::runtime::{self, SharedRuntime, StackSelection, StackStatus};
use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, Query, Request, State, WebSocketUpgrade};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

const MAX_ASR_WAV_BYTES: usize = 8 * 1024 * 1024;
const SPEECH_LAB_CONTRACT_VERSION: &str = "speech-lab/v1";
const MAX_ASR_REQUEST_BYTES: usize = MAX_ASR_WAV_BYTES + 64 * 1024;

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
        .route("/health", get(get_health))
        .route("/ready", get(get_ready))
        // Stable ASR/TTS gateway for AuraGo (paths fixed across stack switches).
        .route("/v1/audio/transcriptions", post(post_gateway_transcribe))
        .route("/api/v1/asr", post(post_gateway_transcribe))
        .route("/v1/audio/speech", post(post_gateway_speech))
        .route("/api/v1/tts", post(post_gateway_speech))
        .route("/api/v1/catalog", get(get_catalog))
        .route("/api/v1/capability", get(get_capability))
        .route("/api/v1/suggestions", get(get_suggestions))
        .route("/api/v1/stack", get(get_stack).put(put_stack))
        .route(
            "/api/v1/models/{backend_id}/download",
            post(post_model_download).delete(delete_model_download),
        )
        .route(
            "/api/v1/models/{backend_id}",
            axum::routing::delete(delete_model),
        )
        .route("/api/v1/benchmarks", post(post_benchmark))
        .route("/api/v1/benchmarks/{id}", get(get_benchmark))
        .route(
            "/api/v1/benchmarks/{id}/audio/{sample_index}",
            get(get_benchmark_audio),
        )
        .route(
            "/api/v1/benchmarks/{id}/ratings",
            post(post_benchmark_rating),
        )
        .layer(CorsLayer::permissive())
        .with_state(state);

    info!(
        "Speech lab listening on http://{addr} (gateway ASR/TTS + catalog/capability/suggestions/stack)"
    );
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn get_health() -> impl IntoResponse {
    let mut health = LabController::health_ok();
    if let Some(object) = health.as_object_mut() {
        object.insert(
            "contract_version".into(),
            SPEECH_LAB_CONTRACT_VERSION.into(),
        );
        object.insert(
            "bundle_version".into(),
            std::env::var("S2S_BUNDLE_VERSION")
                .unwrap_or_else(|_| "unknown".into())
                .into(),
        );
    }
    Json(health)
}

async fn get_ready(State(state): State<AppState>) -> Response {
    let ready = state.lab.gateway_ready().await;
    let status = readiness_http_status(ready.ready);
    let mut value =
        serde_json::to_value(ready).unwrap_or_else(|_| serde_json::json!({"ready": false}));
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "contract_version".into(),
            SPEECH_LAB_CONTRACT_VERSION.into(),
        );
        object.insert(
            "bundle_version".into(),
            std::env::var("S2S_BUNDLE_VERSION")
                .unwrap_or_else(|_| "unknown".into())
                .into(),
        );
    }
    (status, Json(value)).into_response()
}

fn readiness_http_status(ready: bool) -> StatusCode {
    if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn post_gateway_transcribe(State(state): State<AppState>, req: Request) -> Response {
    let language_header = req
        .headers()
        .get("x-language")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let body = match axum::body::to_bytes(req.into_body(), MAX_ASR_REQUEST_BYTES).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(serde_json::json!({ "error": format!("ASR request exceeds 8 MiB: {error}") })),
            )
                .into_response();
        }
    };

    let (wav, language_form) = match extract_wav_and_language(&content_type, &body).await {
        Ok(value) => value,
        Err(error) => {
            let message = format!("{error:#}");
            let status = if message.contains("exceeds 8 MiB") {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return (status, Json(serde_json::json!({ "error": message }))).into_response();
        }
    };
    let language = language_form.or(language_header);

    match state.lab.gateway_transcribe(wav, language.as_deref()).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({ "error": format!("{error:#}") })),
        )
            .into_response(),
    }
}

async fn post_gateway_speech(
    State(state): State<AppState>,
    Json(request): Json<GatewaySpeechRequest>,
) -> Response {
    if let Err(error) = request.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("{error:#}") })),
        )
            .into_response();
    }
    match state.lab.gateway_speech(request).await {
        Ok(audio) => {
            let mut response = Response::new(Body::from(audio.bytes));
            *response.status_mut() = StatusCode::OK;
            if let Ok(value) = HeaderValue::from_str(&audio.content_type) {
                response.headers_mut().insert(header::CONTENT_TYPE, value);
            }
            if let Ok(value) = HeaderValue::from_str(&audio.tts_id) {
                response.headers_mut().insert("x-s2s-tts-id", value);
            }
            if let Ok(value) = HeaderValue::from_str(&audio.voice) {
                response.headers_mut().insert("x-s2s-voice", value);
            }
            response
        }
        Err(error) => {
            let message = format!("{error:#}");
            if let Some(voice_error) = error.downcast_ref::<GatewayVoiceNotActive>() {
                return (
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "code": "voice_not_active",
                        "error": message,
                        "active_voice": voice_error.active,
                    })),
                )
                    .into_response();
            }
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({ "error": message })),
            )
                .into_response()
        }
    }
}

async fn extract_wav_and_language(
    content_type: &str,
    body: &[u8],
) -> Result<(Vec<u8>, Option<String>), anyhow::Error> {
    let ctype = content_type.to_ascii_lowercase();
    if ctype.contains("multipart/") {
        let boundary = multer::parse_boundary(content_type)
            .map_err(|e| anyhow::anyhow!("multipart boundary: {e}"))?;
        let mut multipart = multer::Multipart::new(
            futures_util::stream::once(async move {
                Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(body))
            }),
            boundary,
        );
        let mut file_bytes = None;
        let mut language = None;
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| anyhow::anyhow!("multipart field: {e}"))?
        {
            let name = field.name().unwrap_or("").to_string();
            let field_content_type = field.content_type().map(ToString::to_string);
            let data = field
                .bytes()
                .await
                .map_err(|e| anyhow::anyhow!("multipart bytes: {e}"))?;
            match name.as_str() {
                "file" | "audio" | "data" => {
                    let Some(field_content_type) = field_content_type else {
                        return Err(anyhow::anyhow!(
                            "multipart audio field requires Content-Type audio/wav"
                        ));
                    };
                    if !is_wav_content_type(&field_content_type) {
                        return Err(anyhow::anyhow!(
                            "unsupported multipart audio Content-Type {field_content_type}; expected audio/wav"
                        ));
                    }
                    file_bytes = Some(data.to_vec());
                }
                "language" | "lang" => {
                    language = Some(String::from_utf8_lossy(&data).trim().to_string());
                }
                _ => {}
            }
        }
        let wav =
            file_bytes.ok_or_else(|| anyhow::anyhow!("multipart form missing file/audio field"))?;
        validate_pcm_wav(&wav)?;
        return Ok((wav, language.filter(|s| !s.is_empty())));
    }

    if !is_wav_content_type(content_type) {
        return Err(anyhow::anyhow!(
            "unsupported Content-Type {content_type}; expected audio/wav or multipart/form-data"
        ));
    }
    if body.is_empty() {
        return Err(anyhow::anyhow!(
            "empty body; send multipart file= or raw audio/wav"
        ));
    }
    validate_pcm_wav(body)?;
    Ok((body.to_vec(), None))
}

fn is_wav_content_type(value: &str) -> bool {
    matches!(
        value
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "audio/wav" | "audio/x-wav"
    )
}

fn validate_pcm_wav(bytes: &[u8]) -> Result<()> {
    if bytes.len() > MAX_ASR_WAV_BYTES {
        anyhow::bail!("PCM-WAV exceeds 8 MiB");
    }
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("invalid RIFF/WAVE audio");
    }

    let mut reader =
        hound::WavReader::new(std::io::Cursor::new(bytes)).context("invalid PCM-WAV structure")?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int {
        anyhow::bail!("compressed or floating-point WAV is unsupported; expected PCM-WAV");
    }
    if spec.channels == 0
        || spec.sample_rate == 0
        || !matches!(spec.bits_per_sample, 8 | 16 | 24 | 32)
    {
        anyhow::bail!("invalid PCM-WAV format");
    }
    if reader.duration() == 0 {
        anyhow::bail!("PCM-WAV contains no audio samples");
    }
    for sample in reader.samples::<i32>() {
        sample.context("invalid PCM-WAV sample data")?;
    }
    Ok(())
}

async fn get_catalog(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.lab.catalog().await)
}

async fn get_capability(State(state): State<AppState>) -> impl IntoResponse {
    let hardware = state.lab.catalog().await.hardware;
    let mut value = serde_json::to_value(hardware).unwrap_or_else(|_| serde_json::json!({}));
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "contract_version".into(),
            SPEECH_LAB_CONTRACT_VERSION.into(),
        );
        object.insert(
            "bundle_version".into(),
            std::env::var("S2S_BUNDLE_VERSION")
                .unwrap_or_else(|_| "unknown".into())
                .into(),
        );
    }
    Json(value)
}

async fn get_suggestions(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let catalog = state.lab.catalog().await;
    Json(build_gateway_suggestions(&catalog, &params))
}

#[derive(Debug, Serialize)]
struct GatewaySuggestedPair {
    asr_id: String,
    tts_id: String,
    asr_name: String,
    tts_name: String,
    score: f32,
    reason: String,
    vram_gb: f32,
}

#[derive(Debug, Serialize)]
struct GatewaySuggestions {
    capability: HardwareProfile,
    suggested_pairs: Vec<GatewaySuggestedPair>,
    scoring: &'static str,
    note: &'static str,
}

fn build_gateway_suggestions(
    catalog: &crate::lab::LabCatalogResponse,
    params: &HashMap<String, String>,
) -> GatewaySuggestions {
    let language = params
        .get("language")
        .or_else(|| params.get("lang"))
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty() && value != "auto");
    let stable_only = params
        .get("stable_only")
        .or_else(|| params.get("stable"))
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true);
    let budget = params
        .get("max_vram_gb")
        .or_else(|| params.get("max_vram"))
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(8.0);
    let limit = params
        .get("limit")
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(1, 32);

    let eligible = |status: &&crate::registry::CatalogBackendStatus,
                    stage: crate::registry::BackendStage| {
        status.backend.stage == stage
            && status.available
            && status.selected_variant.as_ref().is_some_and(|variant| {
                (!stable_only || variant.stable)
                    && language.as_deref().is_none_or(|wanted| {
                        status.backend.languages.is_empty()
                            || status.backend.languages.iter().any(|item| {
                                let item = item.to_ascii_lowercase();
                                item == "auto"
                                    || item == wanted
                                    || item.starts_with(&format!("{wanted}-"))
                            })
                    })
            })
    };
    let asr: Vec<_> = catalog
        .backends
        .iter()
        .filter(|status| eligible(status, crate::registry::BackendStage::Asr))
        .collect();
    let tts: Vec<_> = catalog
        .backends
        .iter()
        .filter(|status| eligible(status, crate::registry::BackendStage::Tts))
        .collect();

    let mut suggested_pairs = Vec::new();
    for asr_status in asr {
        for tts_status in &tts {
            let vram_gb = (asr_status.backend.resources.vram_gb
                + tts_status.backend.resources.vram_gb)
                .max(0.0);
            let installed_bonus = u8::from(asr_status.installed) + u8::from(tts_status.installed);
            let budget_score = if vram_gb <= budget { 0.25 } else { -0.35 };
            let score = (0.55 + budget_score + f32::from(installed_bonus) * 0.08).clamp(0.0, 0.99);
            suggested_pairs.push(GatewaySuggestedPair {
                asr_id: asr_status.backend.id.clone(),
                tts_id: tts_status.backend.id.clone(),
                asr_name: asr_status.backend.name.clone(),
                tts_name: tts_status.backend.name.clone(),
                score: (score * 100.0).round() / 100.0,
                reason: if vram_gb <= budget {
                    "stable catalog pair within the requested VRAM budget".into()
                } else {
                    "stable catalog pair above the requested VRAM budget".into()
                },
                vram_gb: (vram_gb * 100.0).round() / 100.0,
            });
        }
    }
    suggested_pairs.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.vram_gb
                    .partial_cmp(&right.vram_gb)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.asr_id.cmp(&right.asr_id))
            .then_with(|| left.tts_id.cmp(&right.tts_id))
    });
    suggested_pairs.truncate(limit);

    GatewaySuggestions {
        capability: catalog.hardware.clone(),
        suggested_pairs,
        scoring: "heuristic_v1",
        note: "Predictions from catalog metadata and capacity budget; not runtime benchmarks.",
    }
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
) -> Response {
    let result = state.lab.activate(request).await;
    let status = stack_http_status(result.ok);
    (status, Json(result)).into_response()
}

fn stack_http_status(ok: bool) -> StatusCode {
    if ok {
        StatusCode::OK
    } else {
        StatusCode::CONFLICT
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pcm_wav() -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut bytes), spec).unwrap();
            writer.write_sample::<i16>(0).unwrap();
            writer.write_sample::<i16>(512).unwrap();
            writer.finalize().unwrap();
        }
        bytes
    }

    #[tokio::test]
    async fn raw_asr_accepts_only_valid_pcm_wav() {
        let wav = pcm_wav();
        let (actual, language) = extract_wav_and_language("audio/wav", &wav).await.unwrap();
        assert_eq!(actual, wav);
        assert_eq!(language, None);

        assert!(extract_wav_and_language("audio/webm", &wav).await.is_err());
        assert!(extract_wav_and_language("audio/wav", b"RIFFbrokenWAVE")
            .await
            .is_err());
        for compressed in [
            b"ID3compressed-mp3".as_slice(),
            b"OggScompressed-ogg".as_slice(),
        ] {
            assert!(extract_wav_and_language("audio/wav", compressed)
                .await
                .is_err());
        }

        let mut floating = wav;
        floating[20..22].copy_from_slice(&3_u16.to_le_bytes());
        assert!(extract_wav_and_language("audio/wav", &floating)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn multipart_asr_checks_part_type_and_language() {
        let boundary = "aurago-speech-lab-test";
        let wav = pcm_wav();
        let mut body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nde\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .into_bytes();
        body.extend_from_slice(&wav);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let (actual, language) =
            extract_wav_and_language(&format!("multipart/form-data; boundary={boundary}"), &body)
                .await
                .unwrap();
        assert_eq!(actual, wav);
        assert_eq!(language.as_deref(), Some("de"));

        let mut wrong_type = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.webm\"\r\nContent-Type: audio/webm\r\n\r\n"
        )
        .into_bytes();
        wrong_type.extend_from_slice(&wav);
        wrong_type.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        assert!(extract_wav_and_language(
            &format!("multipart/form-data; boundary={boundary}"),
            &wrong_type,
        )
        .await
        .is_err());
    }

    #[test]
    fn asr_rejects_oversized_wav() {
        let mut wav = pcm_wav();
        wav.resize(MAX_ASR_WAV_BYTES + 1, 0);
        assert!(validate_pcm_wav(&wav)
            .unwrap_err()
            .to_string()
            .contains("exceeds 8 MiB"));
    }

    #[test]
    fn failed_stack_activation_is_http_conflict() {
        assert_eq!(stack_http_status(true), StatusCode::OK);
        assert_eq!(stack_http_status(false), StatusCode::CONFLICT);
    }

    #[test]
    fn readiness_uses_service_unavailable_until_both_components_are_ready() {
        assert_eq!(readiness_http_status(true), StatusCode::OK);
        assert_eq!(
            readiness_http_status(false),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
