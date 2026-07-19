//! Text-to-speech backends: HTTP (qwentts wrapper / OpenAI-style), Piper, system.

use crate::audio::pcm::{decode_wav, f32_to_i16, resample_f32};
use crate::config::{Config, TtsBackend};
use crate::messages::{AudioOut, Control, LlmChunk, QueueItem};
use anyhow::{anyhow, Context, Result};
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

pub async fn run_tts(
    cfg: Config,
    mut llm_in: mpsc::Receiver<QueueItem<LlmChunk>>,
    audio_out: mpsc::Sender<AudioOut>,
    should_listen: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    info!("TTS handler started (backend={:?})", cfg.tts);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .expect("http client");

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
                    // Re-open mic after full response.
                    should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                    info!("TTS: response done — listening again");
                    continue;
                }
                if chunk.text.trim().is_empty() {
                    continue;
                }
                match synthesize(&client, &cfg, &chunk.text, chunk.language.as_deref()).await {
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
                    Err(e) => error!("TTS failed: {e:#}"),
                }
            }
        }
    }
}

async fn synthesize(
    client: &reqwest::Client,
    cfg: &Config,
    text: &str,
    language: Option<&str>,
) -> Result<(Vec<i16>, u32)> {
    match cfg.tts {
        TtsBackend::Http => synthesize_http(client, cfg, text, language).await,
        TtsBackend::Piper => synthesize_piper(cfg, text).await,
        TtsBackend::System => synthesize_system(cfg, text).await,
    }
}

/// Flexible HTTP TTS:
/// 1) OpenAI-style: POST {model, input, voice} → audio bytes (mp3/wav)
/// 2) Simple: POST {text, language} → wav
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
        "voice": "default",
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

    // Piper --output_raw is usually s16le mono at voice native rate (often 22050).
    let mut pcm = Vec::with_capacity(output.stdout.len() / 2);
    for c in output.stdout.chunks_exact(2) {
        pcm.push(i16::from_le_bytes([c[0], c[1]]));
    }
    Ok((pcm, 22050))
}

/// Best-effort system TTS for bring-up without neural models.
#[cfg(windows)]
async fn synthesize_system(cfg: &Config, text: &str) -> Result<(Vec<i16>, u32)> {
    // PowerShell SAPI → temporary WAV.
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
    // espeak-ng → wav stdout
    let output = Command::new("espeak-ng")
        .args(["-v", "en", "--stdout", text])
        .output()
        .await
        .context("espeak-ng (install espeak-ng or use --tts http/piper)")?;
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
    // Assume raw s16le mono.
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
    match cfg.tts {
        TtsBackend::Http => {
            let client = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(2))
                .build()
            {
                Ok(c) => c,
                Err(_) => return false,
            };
            // HEAD or GET may 404; connection success is enough signal for optional TTS.
            match client.get(&cfg.tts_url).send().await {
                Ok(_) => true,
                Err(e) => {
                    // Connection refused → down; other errors might still work for POST.
                    !e.is_connect()
                }
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
