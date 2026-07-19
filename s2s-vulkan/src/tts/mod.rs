//! Text-to-speech backends: Supertonic (ONNX/CPU), HTTP, Piper, system.
//!
//! Supertonic is in-process Rust + ONNX Runtime (not GGML/Vulkan).
//! Qwen3 Vulkan remains available via `--tts http` / external qwentts server.

mod supertonic;
#[allow(dead_code, unused_imports, clippy::all)]
mod supertonic_helper;

use crate::audio::pcm::{decode_wav, f32_to_i16, resample_f32};
use crate::config::{Config, TtsBackend};
use crate::messages::{AudioOut, Control, LlmChunk, QueueItem};
use anyhow::{anyhow, Context, Result};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use supertonic::SupertonicEngine;

pub async fn run_tts(
    mut cfg: Config,
    mut llm_in: mpsc::Receiver<QueueItem<LlmChunk>>,
    audio_out: mpsc::Sender<AudioOut>,
    should_listen: Arc<std::sync::atomic::AtomicBool>,
) {
    let resolved = resolve_tts_backend(&cfg);
    if resolved != cfg.tts {
        info!("TTS auto-selected backend: {:?} → {:?}", cfg.tts, resolved);
        cfg.tts = resolved;
    }
    info!("TTS handler started (backend={:?})", cfg.tts);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .expect("http client");

    // Load Supertonic once; share across requests via Mutex inside the engine.
    let supertonic: Option<Arc<SupertonicEngine>> = if matches!(cfg.tts, TtsBackend::Supertonic) {
        match SupertonicEngine::load(&cfg) {
            Ok(e) => Some(Arc::new(e)),
            Err(e) => {
                error!("Supertonic load failed: {e:#}");
                warn!("Falling back to system TTS");
                cfg.tts = TtsBackend::System;
                None
            }
        }
    } else {
        None
    };

    while let Some(item) = llm_in.recv().await {
        match item {
            QueueItem::Control(Control::PipelineEnd) => break,
            QueueItem::Control(Control::SessionEnd) => {
                should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                continue;
            }
            QueueItem::Data(chunk) => {
                if chunk.is_final {
                    let _ = audio_out
                        .send(AudioOut {
                            pcm_i16: Vec::new(),
                            sample_rate: cfg.tts_sample_rate,
                            turn: chunk.turn.clone(),
                            response_done: true,
                        })
                        .await;
                    should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                    info!("TTS: response done — listening again");
                    continue;
                }
                if chunk.text.trim().is_empty() {
                    continue;
                }

                let synth = synthesize(
                    &client,
                    &cfg,
                    supertonic.as_ref(),
                    &chunk.text,
                    chunk.language.as_deref(),
                )
                .await;

                match synth {
                    Ok((pcm, sr)) => {
                        info!(
                            "TTS: {} samples @ {} Hz for \"{}\"",
                            pcm.len(),
                            sr,
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
            tokio::task::spawn_blocking(move || {
                eng.synthesize_blocking(&text, language.as_deref())
            })
            .await
            .map_err(|e| anyhow!("supertonic join: {e}"))?
        }
        TtsBackend::Auto => unreachable!("auto resolved before synthesize"),
    }
}

async fn synthesize_http(
    client: &reqwest::Client,
    cfg: &Config,
    text: &str,
    language: Option<&str>,
) -> Result<(Vec<i16>, u32)> {
    let url = cfg.tts_url.clone();
    let body = serde_json::json!({
        "model": "tts",
        "input": text,
        "text": text,
        "language": language.unwrap_or("en"),
        "voice": cfg.supertonic_voice,
        "response_format": "wav",
    });

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

fn pcm_from_audio_bytes(bytes: &[u8], fallback_sr: u32) -> Result<(Vec<i16>, u32)> {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" {
        let (f32s, sr) = decode_wav(bytes)?;
        return Ok((f32_to_i16(&f32s), sr));
    }
    if bytes.len() % 2 != 0 {
        return Err(anyhow!("odd-length raw PCM from TTS"));
    }
    warn!("TTS response is not WAV — treating as s16le mono @ {fallback_sr} Hz");
    let mut pcm = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        pcm.push(i16::from_le_bytes([c[0], c[1]]));
    }
    Ok((pcm, fallback_sr))
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

fn which_ok(name: &str) -> bool {
    std::process::Command::new(name)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success() || s.code().is_some())
        .unwrap_or(false)
}
