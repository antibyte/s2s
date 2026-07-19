//! OpenAI-compatible LLM client (llama-server with Vulkan, vLLM, cloud, …).

use crate::config::Config;
use crate::messages::{Control, LlmChunk, QueueItem, Transcription};
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

pub async fn run_llm(
    cfg: Config,
    mut stt_in: mpsc::Receiver<QueueItem<Transcription>>,
    llm_out: mpsc::Sender<QueueItem<LlmChunk>>,
    should_listen: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    info!(
        "LLM handler started → {} model={}",
        cfg.llm_base_url, cfg.model_name
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .expect("http client");

    let mut history: Vec<ChatMessage> = Vec::new();
    history.push(ChatMessage {
        role: "system".into(),
        content: cfg.system_prompt.clone(),
    });

    while let Some(item) = stt_in.recv().await {
        match item {
            QueueItem::Control(Control::PipelineEnd) => {
                let _ = llm_out.send(QueueItem::end()).await;
                break;
            }
            QueueItem::Control(Control::SessionEnd) => {
                history.clear();
                history.push(ChatMessage {
                    role: "system".into(),
                    content: cfg.system_prompt.clone(),
                });
                should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                continue;
            }
            QueueItem::Data(tr) => {
                if tr.partial || tr.text.trim().is_empty() {
                    should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                history.push(ChatMessage {
                    role: "user".into(),
                    content: tr.text.clone(),
                });
                // Keep history bounded (system + last N turns).
                trim_history(&mut history, cfg.chat_size);

                let result = if cfg.llm_stream {
                    stream_completion(&client, &cfg, &history, &tr, &llm_out).await
                } else {
                    complete_once(&client, &cfg, &history, &tr, &llm_out).await
                };

                match result {
                    Ok(assistant) => {
                        if !assistant.is_empty() {
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
                        // Keep the voice loop alive without an external LLM.
                        let fallback = format!(
                            "I heard you say: {}. (LLM backend unreachable — start Ollama or llama-server.)",
                            tr.text.trim()
                        );
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
        }))
        .await
        .map_err(|_| anyhow!("llm_out closed"))?;
    }
    out.send(QueueItem::Data(LlmChunk {
        text: String::new(),
        language: tr.language.clone(),
        turn: tr.turn.clone(),
        is_final: true,
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

async fn complete_once(
    client: &reqwest::Client,
    cfg: &Config,
    history: &[ChatMessage],
    tr: &Transcription,
    out: &mpsc::Sender<QueueItem<LlmChunk>>,
) -> Result<String> {
    let url = format!(
        "{}/chat/completions",
        cfg.llm_base_url.trim_end_matches('/')
    );
    let body = serde_json::json!({
        "model": cfg.model_name,
        "messages": history,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": false,
    });

    let mut req = client.post(&url).json(&body);
    if !cfg.llm_api_key.is_empty() {
        req = req.bearer_auth(&cfg.llm_api_key);
    }

    let resp = req.send().await.with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let t = resp.text().await.unwrap_or_default();
        return Err(anyhow!("LLM {status}: {t}"));
    }

    let v: Value = resp.json().await?;
    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();
    info!("LLM: \"{text}\"");

    for sentence in split_sentences(&text) {
        out.send(QueueItem::Data(LlmChunk {
            text: sentence,
            language: tr.language.clone(),
            turn: tr.turn.clone(),
            is_final: false,
        }))
        .await
        .ok();
    }
    out.send(QueueItem::Data(LlmChunk {
        text: String::new(),
        language: tr.language.clone(),
        turn: tr.turn.clone(),
        is_final: true,
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
) -> Result<String> {
    let url = format!(
        "{}/chat/completions",
        cfg.llm_base_url.trim_end_matches('/')
    );
    let body = serde_json::json!({
        "model": cfg.model_name,
        "messages": history,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": true,
    });

    let mut req = client.post(&url).json(&body);
    if !cfg.llm_api_key.is_empty() {
        req = req.bearer_auth(&cfg.llm_api_key);
    }

    let resp = req.send().await.with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let t = resp.text().await.unwrap_or_default();
        return Err(anyhow!("LLM stream {status}: {t}"));
    }

    let mut stream = resp.bytes_stream();
    let mut full = String::new();
    let mut sentence_buf = String::new();
    let mut line_buf = String::new();

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
            let delta = v["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap_or("");
            if delta.is_empty() {
                continue;
            }
            full.push_str(delta);
            sentence_buf.push_str(delta);

            while let Some(sentence) = pop_sentence(&mut sentence_buf) {
                info!("LLM chunk: \"{sentence}\"");
                if out
                    .send(QueueItem::Data(LlmChunk {
                        text: sentence,
                        language: tr.language.clone(),
                        turn: tr.turn.clone(),
                        is_final: false,
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
        }))
        .await
        .ok();
    }

    out.send(QueueItem::Data(LlmChunk {
        text: String::new(),
        language: tr.language.clone(),
        turn: tr.turn.clone(),
        is_final: true,
    }))
    .await
    .ok();

    info!("LLM full: \"{}\"", full.trim());
    Ok(full.trim().to_string())
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

    // Flush long buffer without punctuation.
    if buf.len() > 120 {
        if let Some(pos) = buf[..80].rfind(' ') {
            let sentence = buf[..pos].trim().to_string();
            let rest = buf[pos + 1..].to_string();
            *buf = rest;
            if !sentence.is_empty() {
                return Some(sentence);
            }
        }
    }
    None
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
