//! Text-to-speech backends: Supertonic (ONNX/CPU), HTTP, Piper, system.
//!
//! Supertonic is in-process Rust + ONNX Runtime (not GGML/Vulkan).
//! Qwen3 Vulkan remains available via `--tts http` / external qwentts server.

mod supertonic;
#[allow(dead_code, unused_imports, clippy::all)]
mod supertonic_helper;

use crate::audio::pcm::{decode_wav, f32_to_i16, resample_f32};
use crate::config::{Config, TtsBackend};
use crate::messages::{AudioOut, Control, LlmChunk, PipelineEvent, QueueItem, TurnId};
use crate::runtime::SharedRuntime;
use anyhow::{anyhow, Context, Result};
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use serde::Deserialize;
use std::net::SocketAddr;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use supertonic::SupertonicEngine;

#[derive(Clone)]
struct SupertonicServerState {
    engine: Arc<SupertonicEngine>,
    cfg: Config,
}

#[derive(Debug, Deserialize)]
struct SpeechRequest {
    #[serde(default, alias = "text")]
    input: String,
    #[serde(default)]
    language: String,
    #[serde(default)]
    voice: String,
    #[serde(default = "default_response_format")]
    response_format: String,
}

fn default_response_format() -> String {
    "pcm".into()
}

pub async fn run_supertonic_server(cfg: Config) -> Result<()> {
    let address: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse()?;
    let engine = tokio::task::spawn_blocking({
        let cfg = cfg.clone();
        move || SupertonicEngine::load(&cfg)
    })
    .await
    .map_err(|error| anyhow!("Supertonic loader join: {error}"))??;
    let state = SupertonicServerState {
        engine: Arc::new(engine),
        cfg,
    };
    let app = Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route("/v1/models", get(supertonic_models))
        .route("/v1/audio/speech", post(supertonic_speech))
        .with_state(state);
    info!("Supertonic TTS sidecar listening on http://{address}");
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn supertonic_models() -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{ "id": "supertonic", "object": "model" }]
    }))
}

async fn supertonic_speech(
    State(state): State<SupertonicServerState>,
    Json(request): Json<SpeechRequest>,
) -> Response<Body> {
    if request.input.trim().is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "input must not be empty");
    }
    let cfg = state.cfg.clone();
    if !request.voice.trim().is_empty() && request.voice != cfg.supertonic_voice {
        // A sidecar loads one style at a time. Rejecting an unloaded voice is
        // preferable to silently synthesizing with the wrong speaker.
        return json_error(
            StatusCode::BAD_REQUEST,
            &format!(
                "voice '{}' is not loaded; active voice is '{}'",
                request.voice, cfg.supertonic_voice
            ),
        );
    }
    let language = if request.language.trim().is_empty() {
        cfg.resolve_tts_language(None)
    } else {
        request.language
    };
    let engine = state.engine.clone();
    let text = request.input;
    let synthesis =
        tokio::task::spawn_blocking(move || engine.synthesize_blocking(&text, Some(&language)))
            .await;
    let (pcm, sample_rate) = match synthesis {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                &format!("Supertonic synthesis failed: {error:#}"),
            )
        }
        Err(error) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Supertonic synthesis task failed: {error}"),
            )
        }
    };
    if request.response_format.eq_ignore_ascii_case("wav") {
        let wav = match crate::audio::pcm::encode_wav_f32(
            &crate::audio::pcm::i16_to_f32(&pcm),
            sample_rate,
        ) {
            Ok(wav) => wav,
            Err(error) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("WAV encoding failed: {error:#}"),
                )
            }
        };
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "audio/wav")
            .body(Body::from(wav))
            .unwrap();
    }
    let bytes = crate::audio::pcm::i16_to_bytes_le(&pcm);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/pcm")
        .header("x-sample-rate", sample_rate)
        .body(Body::from(bytes))
        .unwrap()
}

fn json_error(status: StatusCode, message: &str) -> Response<Body> {
    let body = serde_json::json!({
        "error": { "message": message, "type": "invalid_request_error" }
    });
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

pub async fn run_tts(
    runtime: SharedRuntime,
    mut llm_in: mpsc::Receiver<QueueItem<LlmChunk>>,
    audio_out: mpsc::Sender<AudioOut>,
    should_listen: Arc<std::sync::atomic::AtomicBool>,
    event_tx: mpsc::Sender<PipelineEvent>,
) {
    {
        let mut rt = runtime.write().await;
        let resolved = resolve_tts_backend(&rt.cfg);
        if resolved != rt.cfg.tts {
            info!(
                "TTS auto-selected backend: {:?} → {:?}",
                rt.cfg.tts, resolved
            );
            rt.cfg.tts = resolved;
        }
        info!(
            "TTS handler started (backend={:?}, tts_language={}) (hot-swap)",
            rt.cfg.tts, rt.cfg.tts_language
        );
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .expect("http client");

    // Lazy-loaded Supertonic engine (loaded on first use / after hot-swap to it).
    let mut supertonic: Option<Arc<SupertonicEngine>> = None;

    while let Some(item) = llm_in.recv().await {
        match item {
            QueueItem::Control(Control::PipelineEnd) => break,
            QueueItem::Control(Control::SessionEnd) => {
                should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                continue;
            }
            QueueItem::Data(chunk) => {
                let mut cfg = runtime.read().await.cfg.clone();
                // Resolve Auto on the fly (hot-swap may set Auto).
                let resolved = resolve_tts_backend(&cfg);
                if resolved != cfg.tts {
                    cfg.tts = resolved;
                }

                if chunk.is_final {
                    let turn_total_ms = chunk.speech_end_at.elapsed().as_secs_f64() * 1000.0;
                    let _ = audio_out
                        .send(AudioOut {
                            pcm_i16: Vec::new(),
                            sample_rate: cfg.tts_sample_rate,
                            turn: chunk.turn.clone(),
                            response_done: true,
                        })
                        .await;
                    let _ = event_tx
                        .send(PipelineEvent::ResponseDone {
                            turn: PipelineEvent::turn_id(&chunk.turn),
                        })
                        .await;
                    let _ = event_tx
                        .send(PipelineEvent::Metrics {
                            stage: "turn".into(),
                            turn: PipelineEvent::turn_id(&chunk.turn),
                            values: serde_json::json!({
                                "speech_end_to_response_complete_ms": turn_total_ms,
                            }),
                        })
                        .await;
                    should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                    info!("TTS: response done — listening again");
                    continue;
                }
                if chunk.text.trim().is_empty() {
                    continue;
                }

                // Ensure Supertonic is loaded when selected.
                if matches!(cfg.tts, TtsBackend::Supertonic) && supertonic.is_none() {
                    // Hot-swap from Qwen may leave voice=Aiden — map to a real style.
                    let mut load_cfg = cfg.clone();
                    let v = load_cfg.supertonic_voice.trim().to_ascii_uppercase();
                    if !matches!(
                        v.as_str(),
                        "M1" | "M2" | "M3" | "M4" | "M5" | "F1" | "F2" | "F3" | "F4" | "F5"
                    ) {
                        warn!(
                            "Supertonic voice '{}' invalid — using M1",
                            load_cfg.supertonic_voice
                        );
                        load_cfg.supertonic_voice = "M1".into();
                        runtime.write().await.cfg.supertonic_voice = "M1".into();
                        cfg.supertonic_voice = "M1".into();
                    }
                    match SupertonicEngine::load(&load_cfg) {
                        Ok(e) => {
                            info!(
                                "Supertonic engine loaded (hot-swap, voice={})",
                                load_cfg.supertonic_voice
                            );
                            supertonic = Some(Arc::new(e));
                        }
                        Err(e) => {
                            error!(
                                "Supertonic load failed: {e:#} — NOT using system/SAPI; \
                                 try --supertonic-model-dir or switch TTS back to Qwen"
                            );
                            let _ = event_tx
                                .send(PipelineEvent::Error {
                                    stage: "tts".into(),
                                    message: format!("Supertonic load failed: {e:#}"),
                                })
                                .await;
                            should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        }
                    }
                }

                let tts_lang = cfg.resolve_tts_language(chunk.language.as_deref());
                // qwentts wants full names (german), Supertonic wants ISO (de).
                let http_lang = Config::qwen_tts_language(&tts_lang);
                let t0 = Instant::now();

                // HTTP/Qwen: buffer full sentence PCM then play (RTF may be >1).
                if matches!(cfg.tts, TtsBackend::Http) {
                    match stream_http(
                        &client,
                        &cfg,
                        &chunk.text,
                        Some(http_lang.as_str()),
                        chunk.turn.clone(),
                        &audio_out,
                    )
                    .await
                    {
                        Ok(stats) => {
                            let audio_s = stats.samples as f64 / cfg.tts_sample_rate as f64;
                            let _ = event_tx
                                .send(PipelineEvent::Metrics {
                                    stage: "tts".into(),
                                    turn: PipelineEvent::turn_id(&chunk.turn),
                                    values: serde_json::json!({
                                        "time_to_first_pcm_ms": stats.first_audio_ms,
                                        "speech_end_to_first_audio_ms":
                                            t0.saturating_duration_since(chunk.speech_end_at)
                                                .as_secs_f64() * 1000.0
                                                + stats.first_audio_ms,
                                        "total_ms": stats.total_ms,
                                        "audio_duration_ms": audio_s * 1000.0,
                                        "rtf": stats.total_ms / 1000.0 / audio_s.max(f64::EPSILON),
                                    }),
                                })
                                .await;
                            info!(
                                "TTS[{}/{}] stream: {} samples @ {} Hz in {:.0} ms for \"{}\"",
                                tts_lang,
                                http_lang,
                                stats.samples,
                                cfg.tts_sample_rate,
                                t0.elapsed().as_secs_f64() * 1000.0,
                                chunk.text
                            );
                        }
                        Err(e) => {
                            error!("TTS stream failed: {e:#}");
                            should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                } else {
                    match synthesize(
                        &client,
                        &cfg,
                        supertonic.as_ref(),
                        &chunk.text,
                        Some(tts_lang.as_str()),
                    )
                    .await
                    {
                        Ok((pcm, sr)) => {
                            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            let audio_s = pcm.len() as f64 / sr.max(1) as f64;
                            let _ = event_tx
                                .send(PipelineEvent::Metrics {
                                    stage: "tts".into(),
                                    turn: PipelineEvent::turn_id(&chunk.turn),
                                    values: serde_json::json!({
                                        "time_to_first_pcm_ms": elapsed_ms,
                                        "speech_end_to_first_audio_ms":
                                            chunk.speech_end_at.elapsed().as_secs_f64() * 1000.0,
                                        "total_ms": elapsed_ms,
                                        "audio_duration_ms": audio_s * 1000.0,
                                        "rtf": elapsed_ms / 1000.0 / audio_s.max(f64::EPSILON),
                                    }),
                                })
                                .await;
                            info!(
                                "TTS[{}]: {} samples @ {} Hz in {:.0} ms for \"{}\"",
                                tts_lang,
                                pcm.len(),
                                sr,
                                t0.elapsed().as_secs_f64() * 1000.0,
                                chunk.text
                            );
                            if audio_out
                                .send(AudioOut {
                                    pcm_i16: pcm,
                                    sample_rate: sr,
                                    turn: chunk.turn,
                                    response_done: false,
                                })
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(e) => {
                            error!("TTS failed: {e:#}");
                            should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }
}

/// Pick concrete backend for `--tts auto`.
pub fn resolve_tts_backend(cfg: &Config) -> TtsBackend {
    match cfg.tts {
        TtsBackend::Auto => {
            if supertonic::is_available(cfg) {
                TtsBackend::Supertonic
            } else if cfg.piper_model.is_some() {
                TtsBackend::Piper
            } else {
                // Prefer HTTP if a local TTS server is configured, else system.
                TtsBackend::System
            }
        }
        other => other,
    }
}

async fn synthesize(
    client: &reqwest::Client,
    cfg: &Config,
    supertonic: Option<&Arc<SupertonicEngine>>,
    text: &str,
    language: Option<&str>,
) -> Result<(Vec<i16>, u32)> {
    match cfg.tts {
        TtsBackend::Http => synthesize_http(client, cfg, text, language).await,
        TtsBackend::Piper => synthesize_piper(cfg, text).await,
        TtsBackend::System => synthesize_system(cfg, text).await,
        TtsBackend::Supertonic => {
            let eng = supertonic
                .ok_or_else(|| anyhow!("Supertonic engine not loaded"))?
                .clone();
            let text = text.to_string();
            let language = language.map(|s| s.to_string());
            tokio::task::spawn_blocking(move || eng.synthesize_blocking(&text, language.as_deref()))
                .await
                .map_err(|e| anyhow!("supertonic join: {e}"))?
        }
        TtsBackend::Auto => unreachable!("auto resolved before synthesize"),
    }
}

/// Prefer WAV for Kokoro / Higgs (clean headers / correct rate); PCM for Qwen-style servers.
fn http_wants_pcm(cfg: &Config) -> bool {
    let m = cfg.tts_model.to_ascii_lowercase();
    let u = cfg.tts_url.to_ascii_lowercase();
    // Kokoro wrapper always supports wav; pcm optional.
    if m.contains("kokoro") || u.contains("8084") || u.contains("kokoro") {
        return false;
    }
    // Higgs SGLang/vLLM-Omni default is WAV; non-stream PCM is less reliable across stacks.
    if m.contains("higgs") || u.contains("higgs") || u.contains("8086") || u.contains("tts-higgs")
    {
        return false;
    }
    // CrispASR VibeVoice defaults to WAV (24 kHz mono).
    if m.contains("vibevoice")
        || u.contains("vibevoice")
        || u.contains("8089")
        || u.contains("tts-vibevoice")
    {
        return false;
    }
    // Default: request raw PCM (qwentts benefits).
    true
}

#[derive(Debug, Clone, Copy)]
struct TtsStreamStats {
    samples: usize,
    first_audio_ms: f64,
    total_ms: f64,
}

/// Pull OpenAI-style / Qwen / Kokoro audio (`response_format=pcm|wav`).
/// Raw PCM is decoded and resampled progressively; WAV remains buffered.
async fn stream_http(
    client: &reqwest::Client,
    cfg: &Config,
    text: &str,
    language: Option<&str>,
    turn: Option<TurnId>,
    audio_out: &mpsc::Sender<AudioOut>,
) -> Result<TtsStreamStats> {
    let url = cfg.tts_url.clone();
    let lang = language.unwrap_or("en");
    let target_sr = cfg.tts_sample_rate;
    let model = if cfg.tts_model.trim().is_empty() {
        "tts"
    } else {
        cfg.tts_model.as_str()
    };
    let want_pcm = http_wants_pcm(cfg);
    let fmt = if want_pcm { "pcm" } else { "wav" };
    let mut body = serde_json::json!({
        "model": model,
        "input": text,
        "text": text,
        "language": lang,
        "lang": lang,
        "voice": cfg.supertonic_voice,
        "response_format": fmt,
    });
    if http_is_qwen(cfg) {
        body["max_new_tokens"] = serde_json::Value::from(qwen_frame_limit(cfg));
    }
    if http_is_higgs(cfg) {
        // Higgs multi-codebook steps (not Qwen codec frames). Cookbook default ≈ 1024.
        body["max_new_tokens"] = serde_json::Value::from(1024u32);
        body["temperature"] = serde_json::json!(0.8);
        body["top_k"] = serde_json::json!(50);
    }

    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header(
            "Accept",
            if want_pcm {
                "audio/pcm, audio/wav, */*"
            } else {
                "audio/wav, audio/pcm, */*"
            },
        )
        .json(&body);
    if !cfg.tts_api_key.is_empty() {
        req = req.bearer_auth(&cfg.tts_api_key);
    }

    let t0 = Instant::now();
    let resp = req.send().await.with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let t = resp.text().await.unwrap_or_default();
        return Err(anyhow!("TTS HTTP {status}: {t}"));
    }

    let ctype = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let header_sr = resp
        .headers()
        .get("x-sample-rate")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());

    // Some servers ignore response_format and still return WAV — fall back.
    if ctype.contains("wav") || ctype.contains("wave") || !want_pcm {
        let bytes = resp.bytes().await?;
        // If body looks like RIFF/WAV, decode; else try raw PCM.
        if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" {
            let (pcm, sr) = pcm_from_audio_bytes(&bytes, target_sr)?;
            let n = pcm.len();
            info!(
                "TTS WAV ready in {:.0} ms ({} samples @ {} Hz, model={})",
                t0.elapsed().as_secs_f64() * 1000.0,
                n,
                sr,
                model
            );
            send_pcm_continuous(audio_out, pcm, sr, turn).await?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            return Ok(TtsStreamStats {
                samples: n,
                first_audio_ms: elapsed_ms,
                total_ms: elapsed_ms,
            });
        }
        // Fall through to PCM path with these bytes.
        return finish_pcm_body(
            &bytes, cfg, target_sr, header_sr, t0, turn, audio_out, model,
        )
        .await;
    }

    stream_pcm_body(resp, cfg, target_sr, header_sr, t0, turn, audio_out, model).await
}

fn http_is_qwen(cfg: &Config) -> bool {
    let model = cfg.tts_model.to_ascii_lowercase();
    let url = cfg.tts_url.to_ascii_lowercase();
    model.contains("qwen")
        || url.contains("qwen")
        || url.contains(":8083/")
        || url.ends_with(":8083")
}

fn http_is_higgs(cfg: &Config) -> bool {
    let model = cfg.tts_model.to_ascii_lowercase();
    let url = cfg.tts_url.to_ascii_lowercase();
    model.contains("higgs")
        || url.contains("higgs")
        || url.contains("tts-higgs")
        || url.contains(":8086/")
        || url.ends_with(":8086")
}

fn qwen_frame_limit(cfg: &Config) -> u32 {
    cfg.tts_http_max_new_tokens.clamp(1, 256)
}

async fn stream_pcm_body(
    resp: reqwest::Response,
    cfg: &Config,
    target_sr: u32,
    header_sr: Option<u32>,
    t0: Instant,
    turn: Option<TurnId>,
    audio_out: &mpsc::Sender<AudioOut>,
    model: &str,
) -> Result<TtsStreamStats> {
    let native_sr = header_sr
        .filter(|sample_rate| *sample_rate >= 8000 && *sample_rate <= 96_000)
        .unwrap_or(cfg.tts_native_sample_rate.max(8000));
    let chunk_samples =
        ((target_sr as u64).saturating_mul(cfg.tts_stream_chunk_ms.clamp(40, 2000) as u64) / 1000)
            .max(1) as usize;

    let mut stream = resp.bytes_stream();
    let mut prefix = Vec::new();
    while prefix.len() < 12 {
        match stream.next().await {
            Some(chunk) => prefix.extend_from_slice(&chunk.context("TTS PCM prefix")?),
            None => break,
        }
    }
    if prefix.len() >= 4 && &prefix[..4] == b"RIFF" {
        while let Some(chunk) = stream.next().await {
            prefix.extend_from_slice(&chunk.context("TTS WAV body")?);
        }
        let (pcm, sample_rate) = pcm_from_audio_bytes(&prefix, target_sr)?;
        let count = pcm.len();
        send_pcm_continuous(audio_out, pcm, sample_rate, turn).await?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        return Ok(TtsStreamStats {
            samples: count,
            first_audio_ms: elapsed_ms,
            total_ms: elapsed_ms,
        });
    }

    let mut decoder = StreamingPcmResampler::new(native_sr, target_sr);
    let mut pending = Vec::with_capacity(chunk_samples * 2);
    let mut total_samples = 0usize;
    let mut first_audio_ms = None;

    let first = decoder.push_bytes(&prefix)?;
    pending.extend(first);
    emit_stream_chunks(
        audio_out,
        &mut pending,
        chunk_samples,
        target_sr,
        &turn,
        &mut total_samples,
        t0,
        &mut first_audio_ms,
        model,
    )
    .await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("TTS PCM body")?;
        pending.extend(decoder.push_bytes(&chunk)?);
        emit_stream_chunks(
            audio_out,
            &mut pending,
            chunk_samples,
            target_sr,
            &turn,
            &mut total_samples,
            t0,
            &mut first_audio_ms,
            model,
        )
        .await?;
    }
    pending.extend(decoder.finish()?);
    if !pending.is_empty() {
        total_samples += pending.len();
        audio_out
            .send(AudioOut {
                pcm_i16: std::mem::take(&mut pending),
                sample_rate: target_sr,
                turn,
                response_done: false,
            })
            .await
            .map_err(|_| anyhow!("audio_out closed"))?;
        if first_audio_ms.is_none() {
            first_audio_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
            info!(
                "TTS first PCM in {:.0} ms (model={})",
                t0.elapsed().as_secs_f64() * 1000.0,
                model
            );
        }
    }
    if total_samples == 0 {
        return Err(anyhow!("TTS PCM stream produced no samples"));
    }
    let wall_s = t0.elapsed().as_secs_f64();
    let audio_s = total_samples as f64 / target_sr as f64;
    info!(
        "TTS PCM complete in {:.0} ms ({:.2}s audio, RTF {:.2}, {} samples @ {} Hz, model={})",
        wall_s * 1000.0,
        audio_s,
        wall_s / audio_s.max(f64::EPSILON),
        total_samples,
        target_sr,
        model
    );
    Ok(TtsStreamStats {
        samples: total_samples,
        first_audio_ms: first_audio_ms.unwrap_or(wall_s * 1000.0),
        total_ms: wall_s * 1000.0,
    })
}

#[allow(clippy::too_many_arguments)]
async fn emit_stream_chunks(
    audio_out: &mpsc::Sender<AudioOut>,
    pending: &mut Vec<i16>,
    chunk_samples: usize,
    sample_rate: u32,
    turn: &Option<TurnId>,
    total_samples: &mut usize,
    started: Instant,
    first_audio_ms: &mut Option<f64>,
    model: &str,
) -> Result<()> {
    while pending.len() >= chunk_samples {
        let remainder = pending.split_off(chunk_samples);
        let samples = std::mem::replace(pending, remainder);
        *total_samples += samples.len();
        audio_out
            .send(AudioOut {
                pcm_i16: samples,
                sample_rate,
                turn: turn.clone(),
                response_done: false,
            })
            .await
            .map_err(|_| anyhow!("audio_out closed"))?;
        if first_audio_ms.is_none() {
            *first_audio_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            info!(
                "TTS first PCM in {:.0} ms ({} ms playout buffer, model={})",
                started.elapsed().as_secs_f64() * 1000.0,
                chunk_samples as u64 * 1000 / sample_rate.max(1) as u64,
                model
            );
        }
    }
    Ok(())
}

/// Chunk-boundary invariant s16le decoder with a phase-stable linear resampler.
struct StreamingPcmResampler {
    from_sr: u32,
    to_sr: u32,
    source: Vec<i16>,
    produced: usize,
    odd_byte: Option<u8>,
}

impl StreamingPcmResampler {
    fn new(from_sr: u32, to_sr: u32) -> Self {
        Self {
            from_sr: from_sr.max(1),
            to_sr: to_sr.max(1),
            source: Vec::new(),
            produced: 0,
            odd_byte: None,
        }
    }

    fn push_bytes(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        let mut offset = 0;
        if let Some(low) = self.odd_byte.take() {
            let Some(&high) = bytes.first() else {
                self.odd_byte = Some(low);
                return Ok(Vec::new());
            };
            self.source.push(i16::from_le_bytes([low, high]));
            offset = 1;
        }
        let complete = &bytes[offset..];
        for pair in complete.chunks_exact(2) {
            self.source.push(i16::from_le_bytes([pair[0], pair[1]]));
        }
        if complete.len() % 2 != 0 {
            self.odd_byte = complete.last().copied();
        }
        Ok(self.produce(false))
    }

    fn finish(&mut self) -> Result<Vec<i16>> {
        if self.odd_byte.is_some() {
            return Err(anyhow!("TTS PCM stream ended with an odd byte"));
        }
        Ok(self.produce(true))
    }

    fn produce(&mut self, final_chunk: bool) -> Vec<i16> {
        if self.source.is_empty() {
            return Vec::new();
        }
        if self.from_sr == self.to_sr {
            let output = self.source[self.produced..].to_vec();
            self.produced = self.source.len();
            return output;
        }

        let final_target =
            (self.source.len() as u64 * self.to_sr as u64 / self.from_sr as u64) as usize;
        let mut output = Vec::new();
        while self.produced < final_target {
            let source_position = self.produced as f64 * self.from_sr as f64 / self.to_sr as f64;
            let lower = source_position.floor() as usize;
            if !final_chunk && lower + 1 >= self.source.len() {
                break;
            }
            let upper = (lower + 1).min(self.source.len() - 1);
            let fraction = (source_position - lower as f64) as f32;
            let a = self.source[lower.min(self.source.len() - 1)] as f32;
            let b = self.source[upper] as f32;
            output.push(
                (a + (b - a) * fraction)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16,
            );
            self.produced += 1;
        }
        output
    }
}

async fn finish_pcm_body(
    bytes: &[u8],
    cfg: &Config,
    target_sr: u32,
    header_sr: Option<u32>,
    t0: Instant,
    turn: Option<TurnId>,
    audio_out: &mpsc::Sender<AudioOut>,
    model: &str,
) -> Result<TtsStreamStats> {
    if bytes.len() < 2 {
        return Err(anyhow!("TTS PCM stream produced no samples"));
    }
    if bytes.len() % 2 != 0 {
        return Err(anyhow!("TTS PCM odd length {}", bytes.len()));
    }

    let mut src: Vec<i16> = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        src.push(i16::from_le_bytes([c[0], c[1]]));
    }

    let native_sr = header_sr
        .filter(|s| *s >= 8000 && *s <= 96_000)
        .unwrap_or(cfg.tts_native_sample_rate.max(8000));

    let out = if target_sr == native_sr {
        src
    } else {
        // One-shot linear resample (same quality as hop path, continuous).
        resample_i16_linear(&src, native_sr, target_sr)
    };
    let n = out.len();
    let audio_s = n as f64 / target_sr as f64;
    let wall_s = t0.elapsed().as_secs_f64();
    info!(
        "TTS PCM ready in {:.0} ms ({:.2}s audio, RTF {:.2}, {} samples @ {} Hz, model={})",
        wall_s * 1000.0,
        audio_s,
        if audio_s > 0.0 { wall_s / audio_s } else { 0.0 },
        n,
        target_sr,
        model
    );
    send_pcm_continuous(audio_out, out, target_sr, turn).await?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(TtsStreamStats {
        samples: n,
        first_audio_ms: elapsed_ms,
        total_ms: elapsed_ms,
    })
}

/// Push PCM to the playout path without intentional gaps.
/// Large utterances are sliced only for WS frame size — all slices are sent
/// back-to-back so the client can schedule a continuous timeline.
async fn send_pcm_continuous(
    audio_out: &mpsc::Sender<AudioOut>,
    pcm: Vec<i16>,
    sample_rate: u32,
    turn: Option<TurnId>,
) -> Result<()> {
    if pcm.is_empty() {
        return Ok(());
    }
    // ~2 s per frame at 16 kHz — keeps messages modest without playout gaps.
    let max_slice = (sample_rate as usize).saturating_mul(2).max(8000);
    if pcm.len() <= max_slice {
        audio_out
            .send(AudioOut {
                pcm_i16: pcm,
                sample_rate,
                turn,
                response_done: false,
            })
            .await
            .map_err(|_| anyhow!("audio_out closed"))?;
        return Ok(());
    }
    let mut offset = 0;
    while offset < pcm.len() {
        let end = (offset + max_slice).min(pcm.len());
        let slice = pcm[offset..end].to_vec();
        offset = end;
        audio_out
            .send(AudioOut {
                pcm_i16: slice,
                sample_rate,
                turn: turn.clone(),
                response_done: false,
            })
            .await
            .map_err(|_| anyhow!("audio_out closed"))?;
    }
    Ok(())
}

fn resample_i16_linear(input: &[i16], from_sr: u32, to_sr: u32) -> Vec<i16> {
    if input.is_empty() || from_sr == to_sr {
        return input.to_vec();
    }
    let n_out = ((input.len() as u64) * to_sr as u64 / from_sr as u64) as usize;
    if n_out == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n_out);
    let ratio = from_sr as f64 / to_sr as f64;
    for i in 0..n_out {
        let src_pos = i as f64 * ratio;
        let i0 = src_pos.floor() as usize;
        let frac = (src_pos - i0 as f64) as f32;
        let s0 = input[i0.min(input.len() - 1)] as f32;
        let s1 = input[(i0 + 1).min(input.len() - 1)] as f32;
        let v = s0 + (s1 - s0) * frac;
        out.push(v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    out
}

async fn synthesize_http(
    client: &reqwest::Client,
    cfg: &Config,
    text: &str,
    language: Option<&str>,
) -> Result<(Vec<i16>, u32)> {
    let url = cfg.tts_url.clone();
    let lang = language.unwrap_or("en");
    let model = if cfg.tts_model.trim().is_empty() {
        "tts"
    } else {
        cfg.tts_model.as_str()
    };
    // Common keys used by OpenAI-style TTS, Qwen wrappers, Kokoro, Higgs, supertonic serve.
    let mut body = serde_json::json!({
        "model": model,
        "input": text,
        "text": text,
        "language": lang,
        "lang": lang,
        "voice": cfg.supertonic_voice,
        "response_format": "wav",
    });
    if http_is_higgs(cfg) {
        body["max_new_tokens"] = serde_json::Value::from(1024u32);
        body["temperature"] = serde_json::json!(0.8);
        body["top_k"] = serde_json::json!(50);
    }

    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body);
    if !cfg.tts_api_key.is_empty() {
        req = req.bearer_auth(&cfg.tts_api_key);
    }

    let resp = req.send().await.with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let t = resp.text().await.unwrap_or_default();
        return Err(anyhow!("TTS HTTP {status}: {t}"));
    }

    let bytes = resp.bytes().await?;
    pcm_from_audio_bytes(&bytes, cfg.tts_sample_rate)
}

async fn synthesize_piper(cfg: &Config, text: &str) -> Result<(Vec<i16>, u32)> {
    let Some(model) = cfg.piper_model.as_ref() else {
        return Err(anyhow!("--piper-model is required for --tts piper"));
    };
    let mut child = Command::new(&cfg.piper_bin)
        .arg("--model")
        .arg(model)
        .arg("--output_raw")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn {:?}", cfg.piper_bin))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(text.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        drop(stdin);
    }

    let output = child.wait_with_output().await?;
    if !output.status.success() {
        return Err(anyhow!(
            "piper failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let mut pcm = Vec::with_capacity(output.stdout.len() / 2);
    for c in output.stdout.chunks_exact(2) {
        pcm.push(i16::from_le_bytes([c[0], c[1]]));
    }
    Ok((pcm, 22050))
}

#[cfg(windows)]
async fn synthesize_system(cfg: &Config, text: &str) -> Result<(Vec<i16>, u32)> {
    let dir = std::env::temp_dir();
    let wav_path = dir.join(format!("s2s_tts_{}.wav", uuid::Uuid::new_v4()));
    let wav_str = wav_path.to_string_lossy().replace('\'', "''");
    let text_escaped = text.replace('\'', "''");

    let ps = format!(
        r#"
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile('{wav}')
$s.Speak('{text}')
$s.Dispose()
"#,
        wav = wav_str,
        text = text_escaped
    );

    let status = Command::new("powershell")
        .args(["-NoProfile", "-Command", &ps])
        .status()
        .await
        .context("powershell SAPI")?;

    if !status.success() {
        return Err(anyhow!("SAPI TTS failed"));
    }

    let bytes = tokio::fs::read(&wav_path).await?;
    let _ = tokio::fs::remove_file(&wav_path).await;
    let (f32s, sr) = decode_wav(&bytes)?;
    let target = cfg.tts_sample_rate;
    let f32s = if sr == target {
        f32s
    } else {
        resample_f32(&f32s, sr, target)?
    };
    Ok((f32_to_i16(&f32s), target))
}

#[cfg(not(windows))]
async fn synthesize_system(cfg: &Config, text: &str) -> Result<(Vec<i16>, u32)> {
    let output = Command::new("espeak-ng")
        .args(["-v", "en", "--stdout", text])
        .output()
        .await
        .context("espeak-ng (install espeak-ng or use --tts http/piper/supertonic)")?;
    if !output.status.success() {
        return Err(anyhow!(
            "espeak-ng failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let (f32s, sr) = decode_wav(&output.stdout)?;
    let target = cfg.tts_sample_rate;
    let f32s = if sr == target {
        f32s
    } else {
        resample_f32(&f32s, sr, target)?
    };
    Ok((f32_to_i16(&f32s), target))
}

/// Decode TTS audio bytes to mono i16 at `target_sr`.
///
/// WAV responses (e.g. Qwen tts-server at 24 kHz) are resampled to `target_sr`
/// so WebSocket/web playback at 16 kHz does not sound like a slow record.
fn pcm_from_audio_bytes(bytes: &[u8], target_sr: u32) -> Result<(Vec<i16>, u32)> {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" {
        let (f32s, sr) = decode_wav(bytes)?;
        let f32s = if sr == target_sr || f32s.is_empty() {
            f32s
        } else {
            info!(
                "TTS WAV resample {sr} → {target_sr} Hz ({} samples)",
                f32s.len()
            );
            resample_f32(&f32s, sr, target_sr)?
        };
        return Ok((f32_to_i16(&f32s), target_sr));
    }
    if bytes.len() % 2 != 0 {
        return Err(anyhow!("odd-length raw PCM from TTS"));
    }
    warn!("TTS response is not WAV — treating as s16le mono @ {target_sr} Hz");
    let mut pcm = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        pcm.push(i16::from_le_bytes([c[0], c[1]]));
    }
    Ok((pcm, target_sr))
}

pub async fn health_check(cfg: &Config) -> bool {
    let backend = resolve_tts_backend(cfg);
    match backend {
        TtsBackend::Http => {
            let client = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(2))
                .build()
            {
                Ok(c) => c,
                Err(_) => return false,
            };
            match client.get(&cfg.tts_url).send().await {
                Ok(_) => true,
                Err(e) => !e.is_connect(),
            }
        }
        TtsBackend::Piper => cfg.piper_bin.exists() || which_ok("piper"),
        TtsBackend::System => {
            #[cfg(windows)]
            {
                true
            }
            #[cfg(not(windows))]
            {
                which_ok("espeak-ng")
            }
        }
        TtsBackend::Supertonic => supertonic::is_available(cfg),
        TtsBackend::Auto => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{header, Response};
    use axum::routing::post;
    use axum::Router;
    use bytes::Bytes;
    use clap::Parser;
    use futures_util::stream;

    fn as_bytes(samples: &[i16]) -> Vec<u8> {
        samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect()
    }

    #[test]
    fn streaming_resampler_is_chunk_boundary_invariant() {
        let input: Vec<i16> = (0..2400)
            .map(|index| {
                let phase = index as f32 * 0.031;
                (phase.sin() * 20_000.0) as i16
            })
            .collect();
        let expected = resample_i16_linear(&input, 24_000, 16_000);
        let bytes = as_bytes(&input);
        let mut decoder = StreamingPcmResampler::new(24_000, 16_000);
        let mut actual = Vec::new();
        let mut offset = 0usize;
        let pattern = [1usize, 7, 2, 31, 5, 128, 3, 511];
        let mut pattern_index = 0usize;
        while offset < bytes.len() {
            let end = (offset + pattern[pattern_index % pattern.len()]).min(bytes.len());
            actual.extend(decoder.push_bytes(&bytes[offset..end]).unwrap());
            offset = end;
            pattern_index += 1;
        }
        actual.extend(decoder.finish().unwrap());
        assert_eq!(actual, expected);
    }

    #[test]
    fn streaming_decoder_preserves_split_odd_byte() {
        let input = [i16::MIN, -123, 0, 456, i16::MAX];
        let bytes = as_bytes(&input);
        let mut decoder = StreamingPcmResampler::new(16_000, 16_000);
        let mut output = Vec::new();
        for byte in bytes {
            output.extend(decoder.push_bytes(&[byte]).unwrap());
        }
        output.extend(decoder.finish().unwrap());
        assert_eq!(output, input);
    }

    #[test]
    fn streaming_decoder_rejects_trailing_odd_byte() {
        let mut decoder = StreamingPcmResampler::new(24_000, 16_000);
        decoder.push_bytes(&[0x01]).unwrap();
        assert!(decoder.finish().is_err());
    }

    #[test]
    fn qwen_frame_limit_never_allows_legacy_2048_frames() {
        let mut cfg = Config::try_parse_from(["s2s-vulkan", "--skip-health"]).unwrap();
        cfg.tts_http_max_new_tokens = 2048;
        assert_eq!(qwen_frame_limit(&cfg), 256);
        cfg.tts_http_max_new_tokens = 0;
        assert_eq!(qwen_frame_limit(&cfg), 1);
    }

    #[tokio::test]
    async fn first_audio_arrives_before_http_response_completes() {
        async fn pcm() -> Response<Body> {
            let block = as_bytes(&vec![1200i16; 1920]);
            let chunks = stream::unfold(0usize, move |index| {
                let block = block.clone();
                async move {
                    if index >= 3 {
                        return None;
                    }
                    if index == 1 {
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    } else if index == 2 {
                        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
                    }
                    Some((Ok::<Bytes, std::io::Error>(Bytes::from(block)), index + 1))
                }
            });
            Response::builder()
                .header(header::CONTENT_TYPE, "audio/pcm")
                .header("x-sample-rate", "24000")
                .body(Body::from_stream(chunks))
                .unwrap()
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/speech", post(pcm)))
                .await
                .unwrap();
        });

        let mut cfg = Config::try_parse_from(["s2s-vulkan", "--skip-health"]).unwrap();
        cfg.tts_url = format!("http://{address}/speech");
        cfg.tts_model = "qwen-test".into();
        cfg.tts_stream_chunk_ms = 160;
        let client = reqwest::Client::new();
        let (audio_tx, mut audio_rx) = mpsc::channel(8);
        let synthesis = tokio::spawn(async move {
            stream_http(&client, &cfg, "Hallo.", Some("german"), None, &audio_tx).await
        });

        let first = tokio::time::timeout(std::time::Duration::from_millis(250), audio_rx.recv())
            .await
            .expect("first PCM should arrive while response is open")
            .expect("audio channel");
        assert_eq!(first.sample_rate, 16_000);
        assert_eq!(first.pcm_i16.len(), 2560);
        assert!(!synthesis.is_finished());
        assert!(synthesis.await.unwrap().unwrap().samples > first.pcm_i16.len());
        server.abort();
    }
}

fn which_ok(name: &str) -> bool {
    std::process::Command::new(name)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success() || s.code().is_some())
        .unwrap_or(false)
}
