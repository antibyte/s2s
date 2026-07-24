//! OpenAI-compatible LLM client (llama-server with Vulkan, vLLM, cloud, …).

use crate::config::Config;
use crate::messages::{Control, LlmChunk, PipelineEvent, QueueItem, Transcription};
use crate::runtime::SharedRuntime;
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

pub async fn run_llm(
    runtime: SharedRuntime,
    mut stt_in: mpsc::Receiver<QueueItem<Transcription>>,
    llm_out: mpsc::Sender<QueueItem<LlmChunk>>,
    should_listen: std::sync::Arc<std::sync::atomic::AtomicBool>,
    event_tx: mpsc::Sender<PipelineEvent>,
) {
    {
        let rt = runtime.read().await;
        info!(
            "LLM handler started → {} model={} (hot-swap)",
            rt.cfg.llm_base_url, rt.cfg.model_name
        );
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .expect("http client");

    let mut history: Vec<ChatMessage> = Vec::new();
    {
        let rt = runtime.read().await;
        history.push(ChatMessage {
            role: "system".into(),
            content: rt.cfg.system_prompt.clone(),
        });
    }

    while let Some(item) = stt_in.recv().await {
        match item {
            QueueItem::Control(Control::PipelineEnd) => {
                let _ = llm_out.send(QueueItem::end()).await;
                break;
            }
            QueueItem::Control(Control::SessionEnd) => {
                history.clear();
                let rt = runtime.read().await;
                history.push(ChatMessage {
                    role: "system".into(),
                    content: rt.cfg.system_prompt.clone(),
                });
                should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                continue;
            }
            QueueItem::Data(tr) => {
                if tr.partial || tr.text.trim().is_empty() {
                    should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                let cfg = runtime.read().await.cfg.clone();
                history.push(ChatMessage {
                    role: "user".into(),
                    content: tr.text.clone(),
                });
                // Keep history bounded (system + last N turns).
                trim_history(&mut history, cfg.chat_size);

                let llm_started = Instant::now();
                let result = if cfg.llm_stream {
                    stream_completion(&client, &cfg, &history, &tr, &llm_out, &event_tx).await
                } else {
                    complete_once(&client, &cfg, &history, &tr, &llm_out, &event_tx).await
                };

                match result {
                    Ok(assistant) => {
                        if !assistant.is_empty() {
                            let total_s = llm_started.elapsed().as_secs_f64();
                            let estimated_tokens =
                                assistant.split_whitespace().count().max(1) as f64;
                            let _ = event_tx
                                .send(PipelineEvent::Metrics {
                                    stage: "llm".into(),
                                    turn: PipelineEvent::turn_id(&tr.turn),
                                    values: serde_json::json!({
                                        "total_ms": total_s * 1000.0,
                                        "tokens_per_second": estimated_tokens / total_s.max(f64::EPSILON),
                                        "completion_tokens_estimated": estimated_tokens as u64,
                                    }),
                                })
                                .await;
                            let _ = event_tx
                                .send(PipelineEvent::LlmFull {
                                    text: assistant.clone(),
                                    turn: PipelineEvent::turn_id(&tr.turn),
                                })
                                .await;
                            history.push(ChatMessage {
                                role: "assistant".into(),
                                content: assistant,
                            });
                        } else {
                            // No TTS will fire — reopen listening.
                            should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    Err(e) => {
                        error!("LLM failed: {e:#} — using local echo fallback");
                        let _ = event_tx
                            .send(PipelineEvent::Error {
                                stage: "llm".into(),
                                message: format!("{e:#}"),
                            })
                            .await;
                        // Keep the voice loop alive without an external LLM.
                        let fallback = format!(
                            "Ich habe verstanden: {}. (LLM nicht erreichbar.)",
                            tr.text.trim()
                        );
                        let _ = event_tx
                            .send(PipelineEvent::LlmFull {
                                text: fallback.clone(),
                                turn: PipelineEvent::turn_id(&tr.turn),
                            })
                            .await;
                        if emit_text_chunks(&llm_out, &fallback, &tr).await.is_err() {
                            should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }
}

async fn emit_text_chunks(
    out: &mpsc::Sender<QueueItem<LlmChunk>>,
    text: &str,
    tr: &Transcription,
) -> Result<()> {
    for sentence in split_sentences(text) {
        out.send(QueueItem::Data(LlmChunk {
            text: sentence,
            language: tr.language.clone(),
            turn: tr.turn.clone(),
            is_final: false,
            speech_end_at: tr.speech_end_at,
        }))
        .await
        .map_err(|_| anyhow!("llm_out closed"))?;
    }
    out.send(QueueItem::Data(LlmChunk {
        text: String::new(),
        language: tr.language.clone(),
        turn: tr.turn.clone(),
        is_final: true,
        speech_end_at: tr.speech_end_at,
    }))
    .await
    .map_err(|_| anyhow!("llm_out closed"))?;
    Ok(())
}

fn trim_history(history: &mut Vec<ChatMessage>, chat_size: usize) {
    // Keep system message + last `chat_size` messages.
    if history.len() <= chat_size + 1 {
        return;
    }
    let system = history.first().cloned();
    let keep_from = history.len() - chat_size;
    let mut rest: Vec<_> = history.drain(keep_from..).collect();
    history.clear();
    if let Some(s) = system {
        if s.role == "system" {
            history.push(s);
        }
    }
    history.append(&mut rest);
}

fn chat_body(cfg: &Config, history: &[ChatMessage], stream: bool) -> Value {
    let mut body = serde_json::json!({
        "model": cfg.model_name,
        "messages": history,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": stream,
    });
    // Ollama cloud / thinking models: without this, `content` is often empty
    // and only `reasoning` is filled (unusable for TTS).
    if cfg.llm_no_think {
        body["reasoning_effort"] = Value::String("none".into());
    }
    // Ollama: keep weights warm between turns (default 5m) to avoid cold-load
    // latency. Set --llm-keep-alive 0 to free VRAM after each reply.
    // Cloud models and non-Ollama servers ignore unknown fields.
    if cfg.llm_base_url.contains("11434") {
        if let Some(v) = ollama_keep_alive_value(&cfg.llm_keep_alive) {
            body["keep_alive"] = v;
        }
    }
    body
}

/// Parse CLI/env keep_alive into a JSON value Ollama accepts.
/// - empty → omit (Ollama default)
/// - integer string (`0`, `300`, `-1`) → number
/// - duration (`5m`, `30s`, `1h`) → string
fn ollama_keep_alive_value(raw: &str) -> Option<Value> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(n) = s.parse::<i64>() {
        return Some(Value::Number(n.into()));
    }
    Some(Value::String(s.to_string()))
}

/// Extract speakable text from OpenAI-style message objects.
fn message_text(msg: &Value) -> String {
    let content = msg
        .get("content")
        .and_then(|c| {
            if let Some(s) = c.as_str() {
                Some(s.to_string())
            } else if let Some(arr) = c.as_array() {
                // multimodal content parts
                let joined: String = arr
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("");
                Some(joined)
            } else {
                None
            }
        })
        .unwrap_or_default();
    let content = content.trim().to_string();
    if !content.is_empty() {
        return content;
    }
    // Last-resort: some providers only fill reasoning; take the last non-empty line.
    if let Some(r) = msg.get("reasoning").and_then(|x| x.as_str()) {
        let last = r
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('*') && !l.starts_with('#'))
            .last()
            .unwrap_or("")
            .trim()
            .trim_matches('"')
            .to_string();
        if !last.is_empty() && last.len() < 200 {
            return last;
        }
    }
    String::new()
}

fn delta_text(delta: &Value) -> String {
    if let Some(s) = delta.get("content").and_then(|c| c.as_str()) {
        if !s.is_empty() {
            return s.to_string();
        }
    }
    // Ignore reasoning deltas for voice (too noisy / not final answer).
    String::new()
}

async fn complete_once(
    client: &reqwest::Client,
    cfg: &Config,
    history: &[ChatMessage],
    tr: &Transcription,
    out: &mpsc::Sender<QueueItem<LlmChunk>>,
    event_tx: &mpsc::Sender<PipelineEvent>,
) -> Result<String> {
    let url = format!(
        "{}/chat/completions",
        cfg.llm_base_url.trim_end_matches('/')
    );
    let body = chat_body(cfg, history, false);

    let mut req = client.post(&url).json(&body);
    if !cfg.llm_api_key.is_empty() {
        req = req.bearer_auth(&cfg.llm_api_key);
    }

    let request_started = Instant::now();
    let resp = req.send().await.with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let t = resp.text().await.unwrap_or_default();
        return Err(anyhow!("LLM {status}: {t}"));
    }

    let v: Value = resp.json().await?;
    let text = message_text(&v["choices"][0]["message"]);
    info!("LLM: \"{text}\"");
    if text.is_empty() {
        return Err(anyhow!(
            "LLM returned empty content (thinking-only response?). \
             Try --llm-no-think true or a non-thinking model."
        ));
    }
    let _ = event_tx
        .send(PipelineEvent::Metrics {
            stage: "llm".into(),
            turn: PipelineEvent::turn_id(&tr.turn),
            values: serde_json::json!({
                "time_to_first_token_ms": request_started.elapsed().as_secs_f64() * 1000.0,
            }),
        })
        .await;

    for sentence in split_sentences(&text) {
        let _ = event_tx
            .send(PipelineEvent::LlmChunk {
                text: sentence.clone(),
                turn: PipelineEvent::turn_id(&tr.turn),
            })
            .await;
        out.send(QueueItem::Data(LlmChunk {
            text: sentence,
            language: tr.language.clone(),
            turn: tr.turn.clone(),
            is_final: false,
            speech_end_at: tr.speech_end_at,
        }))
        .await
        .ok();
    }
    out.send(QueueItem::Data(LlmChunk {
        text: String::new(),
        language: tr.language.clone(),
        turn: tr.turn.clone(),
        is_final: true,
        speech_end_at: tr.speech_end_at,
    }))
    .await
    .ok();

    Ok(text)
}

async fn stream_completion(
    client: &reqwest::Client,
    cfg: &Config,
    history: &[ChatMessage],
    tr: &Transcription,
    out: &mpsc::Sender<QueueItem<LlmChunk>>,
    event_tx: &mpsc::Sender<PipelineEvent>,
) -> Result<String> {
    let url = format!(
        "{}/chat/completions",
        cfg.llm_base_url.trim_end_matches('/')
    );
    let body = chat_body(cfg, history, true);

    let mut req = client.post(&url).json(&body);
    if !cfg.llm_api_key.is_empty() {
        req = req.bearer_auth(&cfg.llm_api_key);
    }

    let request_started = Instant::now();
    let resp = req.send().await.with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let t = resp.text().await.unwrap_or_default();
        return Err(anyhow!("LLM stream {status}: {t}"));
    }

    let t_stream = Instant::now();
    let mut stream = resp.bytes_stream();
    let mut full = String::new();
    let mut sentence_buf = String::new();
    let mut line_buf = String::new();
    let mut first_sentence = true;
    let mut first_token_reported = false;

    while let Some(item) = stream.next().await {
        let chunk = item?;
        line_buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = line_buf.find('\n') {
            let line = line_buf[..pos].trim_end_matches('\r').to_string();
            line_buf = line_buf[pos + 1..].to_string();

            let line = line.trim();
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            let data = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
            if data == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                debug!("skip non-json SSE: {data}");
                continue;
            };
            let delta = delta_text(&v["choices"][0]["delta"]);
            if delta.is_empty() {
                continue;
            }
            if !first_token_reported {
                first_token_reported = true;
                let _ = event_tx
                    .send(PipelineEvent::Metrics {
                        stage: "llm".into(),
                        turn: PipelineEvent::turn_id(&tr.turn),
                        values: serde_json::json!({
                            "time_to_first_token_ms":
                                request_started.elapsed().as_secs_f64() * 1000.0,
                        }),
                    })
                    .await;
            }
            full.push_str(&delta);
            sentence_buf.push_str(&delta);

            while let Some(sentence) = pop_sentence(&mut sentence_buf) {
                if first_sentence {
                    info!(
                        "LLM first sentence in {:.0} ms: \"{sentence}\"",
                        t_stream.elapsed().as_secs_f64() * 1000.0
                    );
                    first_sentence = false;
                } else {
                    info!("LLM chunk: \"{sentence}\"");
                }
                let _ = event_tx
                    .send(PipelineEvent::LlmChunk {
                        text: sentence.clone(),
                        turn: PipelineEvent::turn_id(&tr.turn),
                    })
                    .await;
                if out
                    .send(QueueItem::Data(LlmChunk {
                        text: sentence,
                        language: tr.language.clone(),
                        turn: tr.turn.clone(),
                        is_final: false,
                        speech_end_at: tr.speech_end_at,
                    }))
                    .await
                    .is_err()
                {
                    return Ok(full);
                }
            }
        }
    }

    let rest = sentence_buf.trim().to_string();
    if !rest.is_empty() {
        info!("LLM chunk: \"{rest}\"");
        out.send(QueueItem::Data(LlmChunk {
            text: rest,
            language: tr.language.clone(),
            turn: tr.turn.clone(),
            is_final: false,
            speech_end_at: tr.speech_end_at,
        }))
        .await
        .ok();
    }

    out.send(QueueItem::Data(LlmChunk {
        text: String::new(),
        language: tr.language.clone(),
        turn: tr.turn.clone(),
        is_final: true,
        speech_end_at: tr.speech_end_at,
    }))
    .await
    .ok();

    let full = full.trim().to_string();
    info!("LLM full: \"{full}\"");
    if full.is_empty() {
        return Err(anyhow!(
            "LLM stream produced empty content (thinking-only?). \
             Try --llm-no-think true or --llm-stream false."
        ));
    }
    Ok(full)
}

fn pop_sentence(buf: &mut String) -> Option<String> {
    let break_at = {
        let bytes = buf.as_bytes();
        let mut found = None;
        for (i, &b) in bytes.iter().enumerate() {
            if matches!(b, b'.' | b'!' | b'?' | b';' | b'\n') {
                let next_ok = bytes
                    .get(i + 1)
                    .map(|c| c.is_ascii_whitespace())
                    .unwrap_or(true);
                if next_ok && i + 1 >= 12 {
                    found = Some(i + 1);
                    break;
                }
            }
        }
        found
    };

    if let Some(end) = break_at {
        let sentence = buf[..end].trim().to_string();
        let rest = buf[end..].trim_start().to_string();
        *buf = rest;
        if !sentence.is_empty() {
            return Some(sentence);
        }
    }

    // Flush long buffer without punctuation (UTF-8 safe: never slice mid-char).
    if buf.len() > 120 {
        let end = floor_char_boundary(buf, 80);
        if let Some(pos) = buf[..end].rfind(' ') {
            let sentence = buf[..pos].trim().to_string();
            let rest = buf[pos..].trim_start().to_string();
            *buf = rest;
            if !sentence.is_empty() {
                return Some(sentence);
            }
        }
    }
    None
}

/// Largest index `<= i` that is a char boundary in `s`.
fn floor_char_boundary(s: &str, i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    let mut idx = i;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut buf = text.to_string();
    let mut out = Vec::new();
    while let Some(s) = pop_sentence(&mut buf) {
        out.push(s);
    }
    let rest = buf.trim().to_string();
    if !rest.is_empty() {
        out.push(rest);
    }
    if out.is_empty() && !text.trim().is_empty() {
        out.push(text.trim().to_string());
    }
    out
}

pub async fn health_check(base: &str) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let url = format!("{}/models", base.trim_end_matches('/'));
    match client.get(&url).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}
