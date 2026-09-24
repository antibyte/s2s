//! Speech-to-text via whisper.cpp HTTP server (Vulkan-capable build).

use crate::audio::pcm::encode_wav_f32;
use crate::config::{Config, SttApi};
use crate::gateway::{read_response_bounded, MAX_GATEWAY_JSON_BYTES};
use crate::messages::{Control, PipelineEvent, QueueItem, Transcription, VadAudio};
use crate::runtime::SharedRuntime;
use anyhow::{anyhow, Context, Result};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

pub async fn run_stt(
    runtime: SharedRuntime,
    mut vad_in: mpsc::Receiver<QueueItem<VadAudio>>,
    stt_out: mpsc::Sender<QueueItem<Transcription>>,
    should_listen: std::sync::Arc<std::sync::atomic::AtomicBool>,
    event_tx: mpsc::Sender<PipelineEvent>,
) {
    {
        let rt = runtime.read().await;
        info!("STT handler started → {} (hot-swap)", rt.cfg.whisper_url);
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .expect("http client");

    while let Some(item) = vad_in.recv().await {
        match item {
            QueueItem::Control(Control::PipelineEnd) => {
                let _ = stt_out.send(QueueItem::end()).await;
                break;
            }
            QueueItem::Control(Control::SessionEnd) => {
                should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                continue;
            }
            QueueItem::Data(audio) => {
                if audio.mode != crate::messages::VadMode::Final {
                    // Progressive mode reserved for future live captions.
                    continue;
                }
                let dur_ms = audio.samples.len() as f32 / audio.sample_rate.max(1) as f32 * 1000.0;
                // Sub-~400 ms clips are mostly clicks/noise; whisper invents garbage.
                // (Was 700 ms — too aggressive after VAD silence-trim.)
                if dur_ms < 400.0 {
                    warn!("STT skip too-short segment ({dur_ms:.0} ms) — keep speaking");
                    should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                let cfg = runtime.read().await.cfg.clone();
                let t0 = Instant::now();
                match transcribe(&client, &cfg, &audio).await {
                    Ok(text) => {
                        let text = text.trim().to_string();
                        if text.is_empty() {
                            warn!("STT returned empty transcript — reopening mic");
                            let _ = event_tx
                                .send(PipelineEvent::Error {
                                    stage: "stt".into(),
                                    message: "empty transcript".into(),
                                })
                                .await;
                            should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        }
                        info!(
                            "STT ({:.0} ms): \"{text}\"",
                            t0.elapsed().as_secs_f64() * 1000.0
                        );
                        let lang = if cfg.language == "auto" {
                            None
                        } else {
                            Some(cfg.language.clone())
                        };
                        let _ = event_tx
                            .send(PipelineEvent::FinalTranscript {
                                text: text.clone(),
                                turn: PipelineEvent::turn_id(&audio.turn),
                                language: lang.clone(),
                            })
                            .await;
                        let _ = event_tx
                            .send(PipelineEvent::Metrics {
                                stage: "asr".into(),
                                turn: PipelineEvent::turn_id(&audio.turn),
                                values: serde_json::json!({
                                    "speech_end_to_final_ms":
                                        audio.created_at.elapsed().as_secs_f64() * 1000.0,
                                    "audio_duration_ms": dur_ms,
                                }),
                            })
                            .await;
                        let msg = Transcription {
                            text,
                            language: lang,
                            turn: audio.turn,
                            partial: false,
                            speech_end_at: audio.created_at,
                        };
                        if stt_out.send(QueueItem::Data(msg)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("STT failed: {e:#}");
                        let _ = event_tx
                            .send(PipelineEvent::Error {
                                stage: "stt".into(),
                                message: format!("{e:#}"),
                            })
                            .await;
                        // Don't leave the pipeline half-deaf after a backend outage.
                        should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        }
    }
}

async fn transcribe(client: &reqwest::Client, cfg: &Config, audio: &VadAudio) -> Result<String> {
    let wav = encode_wav_f32(&audio.samples, audio.sample_rate)?;
    transcribe_wav_bytes(client, cfg, &wav).await
}

/// Transcribe a WAV byte buffer via the selected ASR server API.
/// Used by the AuraGo gateway and the pipeline STT handler.
pub(crate) async fn transcribe_wav_bytes(
    client: &reqwest::Client,
    cfg: &Config,
    wav: &[u8],
) -> Result<String> {
    let openai = cfg.stt_api == SttApi::Openai;
    let path = if openai {
        "/v1/audio/transcriptions"
    } else {
        "/inference"
    };
    let url = format!("{}{path}", cfg.whisper_url.trim_end_matches('/'));

    // whisper-server multipart fields (ggml-org/whisper.cpp examples/server).
    let file_part = Part::bytes(wav.to_vec())
        .file_name("audio.wav")
        .mime_str("audio/wav")?;

    let mut form = Form::new()
        .part("file", file_part)
        .text("response_format", "json");
    if openai {
        form = form.text("model", cfg.stt_model.clone());
    } else {
        form = form.text("temperature", cfg.stt_temperature.to_string());
        // Prefer no timestamps for the whisper-server dialect.
        form = form.text("no_timestamps", "true");
    }

    if cfg.language != "auto" {
        form = form.text("language", cfg.language.clone());
    }

    let resp = client
        .post(&url)
        .multipart(form)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = read_response_bounded(resp, MAX_GATEWAY_JSON_BYTES)
            .await
            .map(|body| String::from_utf8_lossy(&body).into_owned())
            .unwrap_or_else(|error| format!("{error:#}"));
        return Err(anyhow!("ASR server {status}: {body}"));
    }

    let body = read_response_bounded(resp, MAX_GATEWAY_JSON_BYTES).await?;
    let body = String::from_utf8(body).context("ASR server response is not UTF-8")?;
    let text = parse_whisper_response(&body)?;
    if openai && cfg.stt_model.eq_ignore_ascii_case("confucius4-r2t2") {
        if let Some((prefix, transcript)) = text.split_once("<asr_text>") {
            let prefix = prefix.trim();
            if prefix.is_empty() || prefix.to_ascii_lowercase().starts_with("language ") {
                return Ok(transcript.trim().to_string());
            }
        }
    }
    Ok(text)
}

#[derive(Debug, Deserialize)]
struct WhisperJson {
    text: Option<String>,
    // some builds nest under "transcription"
    transcription: Option<String>,
}

fn parse_whisper_response(body: &str) -> Result<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    // Plain text response.
    if !trimmed.starts_with('{') {
        return Ok(trimmed.to_string());
    }
    let parsed: WhisperJson =
        serde_json::from_str(trimmed).with_context(|| format!("parse whisper JSON: {trimmed}"))?;
    Ok(parsed
        .text
        .or(parsed.transcription)
        .unwrap_or_default()
        .trim()
        .to_string())
}

/// Health probe for whisper-server / CrispASR HTTP.
pub async fn health_check(base: &str) -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok();
    let Some(client) = client else { return false };
    let base = base.trim_end_matches('/');
    // CrispASR exposes /health; whisper.cpp often only answers on /.
    for path in ["/health", "/"] {
        let url = format!("{base}{path}");
        match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => return true,
            Ok(_) => continue,
            Err(_) => continue,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::Multipart, routing::post, Json, Router};
    use clap::Parser;

    #[tokio::test]
    async fn transcription_uses_selected_multipart_api() {
        async fn openai(mut form: Multipart) -> Json<serde_json::Value> {
            let mut model = String::new();
            let mut file = Vec::new();
            while let Some(field) = form.next_field().await.unwrap() {
                match field.name().unwrap_or_default() {
                    "model" => model = field.text().await.unwrap(),
                    "file" => file = field.bytes().await.unwrap().to_vec(),
                    "no_timestamps" => panic!("Whisper field sent to OpenAI ASR"),
                    _ => {}
                }
            }
            assert_eq!(model, "confucius4-r2t2");
            assert_eq!(file, b"RIFF");
            Json(serde_json::json!({"text": "language German<asr_text>Guten Morgen"}))
        }

        async fn whisper(mut form: Multipart) -> Json<serde_json::Value> {
            let mut timestamps = String::new();
            while let Some(field) = form.next_field().await.unwrap() {
                match field.name().unwrap_or_default() {
                    "no_timestamps" => timestamps = field.text().await.unwrap(),
                    "model" => panic!("OpenAI model field sent to Whisper ASR"),
                    _ => {}
                }
            }
            assert_eq!(timestamps, "true");
            Json(serde_json::json!({"text": "Hallo"}))
        }

        let app = Router::new()
            .route("/v1/audio/transcriptions", post(openai))
            .route("/inference", post(whisper));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        let mut cfg = Config::parse_from(["s2s-vulkan"]);
        cfg.whisper_url = format!("http://{address}");
        cfg.stt_api = SttApi::Openai;
        assert_eq!(
            transcribe_wav_bytes(&client, &cfg, b"RIFF").await.unwrap(),
            "Guten Morgen"
        );
        cfg.stt_api = SttApi::Whisper;
        assert_eq!(
            transcribe_wav_bytes(&client, &cfg, b"RIFF").await.unwrap(),
            "Hallo"
        );
        server.abort();
    }
}
