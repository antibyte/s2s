//! Speech-to-text via whisper.cpp HTTP server (Vulkan-capable build).

use crate::audio::pcm::encode_wav_f32;
use crate::config::Config;
use crate::messages::{Control, QueueItem, Transcription, VadAudio};
use anyhow::{anyhow, Context, Result};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

pub async fn run_stt(
    cfg: Config,
    mut vad_in: mpsc::Receiver<QueueItem<VadAudio>>,
    stt_out: mpsc::Sender<QueueItem<Transcription>>,
    should_listen: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    info!("STT handler started → {}", cfg.whisper_url);
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
                match transcribe(&client, &cfg, &audio).await {
                    Ok(text) => {
                        let text = text.trim().to_string();
                        if text.is_empty() {
                            warn!("STT returned empty transcript — reopening mic");
                            should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        }
                        info!("STT: \"{text}\"");
                        let msg = Transcription {
                            text,
                            language: if cfg.language == "auto" {
                                None
                            } else {
                                Some(cfg.language.clone())
                            },
                            turn: audio.turn,
                            partial: false,
                        };
                        if stt_out.send(QueueItem::Data(msg)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("STT failed: {e:#}");
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
    let url = format!(
        "{}/inference",
        cfg.whisper_url.trim_end_matches('/')
    );

    // whisper-server multipart fields (ggml-org/whisper.cpp examples/server).
    let file_part = Part::bytes(wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")?;

    let mut form = Form::new()
        .part("file", file_part)
        .text("temperature", cfg.stt_temperature.to_string())
        .text("response_format", "json");

    if cfg.language != "auto" {
        form = form.text("language", cfg.language.clone());
    }

    // Prefer no timestamps for lower latency / simpler parse.
    form = form.text("no_timestamps", "true");

    let resp = client
        .post(&url)
        .multipart(form)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("whisper-server {status}: {body}"));
    }

    let body = resp.text().await?;
    parse_whisper_response(&body)
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
    let parsed: WhisperJson = serde_json::from_str(trimmed)
        .with_context(|| format!("parse whisper JSON: {trimmed}"))?;
    Ok(parsed
        .text
        .or(parsed.transcription)
        .unwrap_or_default()
        .trim()
        .to_string())
}

/// Health probe for whisper-server.
pub async fn health_check(base: &str) -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok();
    let Some(client) = client else { return false };
    let url = format!("{}/", base.trim_end_matches('/'));
    match client.get(&url).send().await {
        Ok(r) => r.status().is_success() || r.status().as_u16() == 404,
        Err(_) => {
            // Some builds only expose /inference — try OPTIONS/GET on inference is useless;
            // treat connection refusal as down, anything else as up-ish.
            false
        }
    }
}
