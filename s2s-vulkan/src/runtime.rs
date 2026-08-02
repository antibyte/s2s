//! Hot-swappable runtime stack (ASR / LLM / TTS) shared across pipeline handlers.
//!
//! The lab WebSocket sends `set_stack` JSON; handlers read `SharedConfig` each turn.

use crate::config::{Config, TtsBackend};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Notify, RwLock};
use tracing::{info, warn};

pub type SharedConfig = Arc<RwLock<Config>>;

pub fn shared_from(cfg: Config) -> SharedConfig {
    Arc::new(RwLock::new(cfg))
}

/// UI / client stack selection (ids match web/app.js catalogs).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StackSelection {
    #[serde(default)]
    pub asr: Option<String>,
    #[serde(default)]
    pub tts: Option<String>,
    #[serde(default)]
    pub llm: Option<String>,
    /// Optional free-form overrides (advanced).
    #[serde(default)]
    pub whisper_url: Option<String>,
    #[serde(default)]
    pub llm_base_url: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub tts_url: Option<String>,
    #[serde(default)]
    pub tts_backend: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StackStatus {
    pub asr: String,
    pub tts: String,
    pub llm: String,
    pub whisper_url: String,
    pub llm_base_url: String,
    pub model_name: String,
    pub tts_backend: String,
    pub tts_url: String,
    pub voice: String,
    pub language: String,
    pub ok: bool,
    pub message: String,
}

/// Labels stored on Config via env-style side channel fields we re-use loosely:
/// we keep selection ids in the system by encoding into unused optional path —
/// instead keep a small extension map on the SharedConfig wrapper.
#[derive(Debug, Clone)]
pub struct RuntimeState {
    pub cfg: Config,
    pub asr_id: String,
    pub tts_id: String,
    pub llm_id: String,
    pub turns: Arc<TurnCoordinator>,
}

pub type SharedRuntime = Arc<RwLock<RuntimeState>>;

pub fn runtime_from(cfg: Config) -> SharedRuntime {
    let asr_id = infer_asr_id(&cfg);
    let tts_id = infer_tts_id(&cfg);
    let llm_id = infer_llm_id(&cfg);
    Arc::new(RwLock::new(RuntimeState {
        cfg,
        asr_id,
        tts_id,
        llm_id,
        turns: Arc::new(TurnCoordinator::default()),
    }))
}

#[derive(Debug, Default)]
pub struct TurnCoordinator {
    pause_depth: AtomicUsize,
    active: AtomicUsize,
    idle: Notify,
}

impl TurnCoordinator {
    pub fn pause(self: &Arc<Self>) -> TurnPauseGuard {
        self.pause_depth.fetch_add(1, Ordering::AcqRel);
        TurnPauseGuard {
            coordinator: self.clone(),
        }
    }

    pub fn try_acquire(self: &Arc<Self>) -> Option<Arc<TurnLease>> {
        if self.pause_depth.load(Ordering::Acquire) != 0 {
            return None;
        }
        self.active.fetch_add(1, Ordering::AcqRel);
        if self.pause_depth.load(Ordering::Acquire) != 0 {
            self.release_turn();
            return None;
        }
        Some(Arc::new(TurnLease {
            coordinator: Arc::downgrade(self),
        }))
    }

    pub fn is_idle(&self) -> bool {
        self.active.load(Ordering::Acquire) == 0
    }

    pub async fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self.is_idle() {
                return true;
            }
            let notified = self.idle.notified();
            if self.is_idle() {
                return true;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return self.is_idle();
            }
        }
    }

    fn release_turn(&self) {
        let previous = self.active.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "turn coordinator underflow");
        if previous == 1 {
            self.idle.notify_waiters();
        }
    }
}

#[derive(Debug)]
pub struct TurnLease {
    coordinator: std::sync::Weak<TurnCoordinator>,
}

impl Drop for TurnLease {
    fn drop(&mut self) {
        if let Some(coordinator) = self.coordinator.upgrade() {
            coordinator.release_turn();
        }
    }
}

#[derive(Debug)]
pub struct TurnPauseGuard {
    coordinator: Arc<TurnCoordinator>,
}

impl Drop for TurnPauseGuard {
    fn drop(&mut self) {
        let previous = self.coordinator.pause_depth.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "turn pause depth underflow");
    }
}

fn infer_asr_id(cfg: &Config) -> String {
    let url = cfg.whisper_url.to_ascii_lowercase();
    if url.contains("voxtral") || url.contains(":8087") {
        "voxtral-mini-4b-realtime".into()
    } else if url.contains("parakeet") {
        "parakeet-tdt-0.6b-v3".into()
    } else if url.contains("whisper-tiny") {
        "fw-tiny".into()
    } else if url.contains("whisper-small") {
        "fw-small".into()
    } else if url.contains("whisper-base") || url.contains("8082") {
        "fw-base".into()
    } else {
        "custom-asr".into()
    }
}

fn infer_tts_id(cfg: &Config) -> String {
    match cfg.tts {
        TtsBackend::Supertonic | TtsBackend::Auto => "supertonic".into(),
        TtsBackend::Http => {
            let u = cfg.tts_url.to_ascii_lowercase();
            let m = cfg.tts_model.to_ascii_lowercase();
            if u.contains("8091") || m.contains("xtts") || u.contains("xtts") {
                "xtts-v2".into()
            } else if u.contains("8084") || m.contains("kokoro") || u.contains("kokoro") {
                "kokoro".into()
            } else if u.contains("vibevoice")
                || m.contains("vibevoice")
                || u.contains("8089")
                || u.contains("tts-vibevoice")
            {
                "vibevoice-realtime-0.5b".into()
            } else if u.contains("higgs")
                || m.contains("higgs")
                || u.contains("8086")
                || u.contains("tts-higgs")
            {
                "higgs-tts-3-4b".into()
            } else if u.contains("supertonic") || m.contains("supertonic") {
                "supertonic".into()
            } else if u.contains("qwen") || m.contains("qwen") || u.contains("8083") {
                "qwen3-tts-0.6b".into()
            } else {
                "http".into()
            }
        }
        TtsBackend::Piper => "piper".into(),
        TtsBackend::System => "system".into(),
    }
}

fn infer_llm_id(cfg: &Config) -> String {
    if cfg
        .llm_base_url
        .to_ascii_lowercase()
        .contains("llama-fallback")
    {
        return "local-fallback".into();
    }
    let m = cfg.model_name.to_ascii_lowercase();
    if m.contains("qwen3.5:cloud") || m.contains("qwen3.5") {
        "external-openai".into()
    } else if m.contains("llama3.2") || m.contains("llama3.2:1b") {
        "llama-1b".into()
    } else if m.contains("granite") {
        "granite-2b".into()
    } else {
        "external-openai".into()
    }
}

pub fn status_of(rt: &RuntimeState) -> StackStatus {
    StackStatus {
        asr: rt.asr_id.clone(),
        tts: rt.tts_id.clone(),
        llm: rt.llm_id.clone(),
        whisper_url: rt.cfg.whisper_url.clone(),
        llm_base_url: rt.cfg.llm_base_url.clone(),
        model_name: rt.cfg.model_name.clone(),
        tts_backend: format!("{:?}", rt.cfg.tts).to_ascii_lowercase(),
        tts_url: rt.cfg.tts_url.clone(),
        voice: rt.cfg.supertonic_voice.clone(),
        language: rt.cfg.language.clone(),
        ok: true,
        message: "ok".into(),
    }
}

/// Apply a stack selection to runtime state. Returns human-readable notes.
pub async fn apply_stack(rt: &SharedRuntime, sel: StackSelection) -> (StackStatus, Vec<String>) {
    let mut notes = Vec::new();
    let mut ok = true;
    let mut guard = rt.write().await;

    if let Some(ref id) = sel.asr {
        if let Some(n) = apply_asr(&mut guard, id) {
            notes.push(n);
        }
    }
    if let Some(ref id) = sel.tts {
        if let Some(n) = apply_tts(&mut guard, id) {
            notes.push(n);
        }
    }
    if let Some(ref id) = sel.llm {
        if let Some(n) = apply_llm(&mut guard, id) {
            notes.push(n);
        }
    }

    // Free-form overrides last.
    if let Some(u) = sel.whisper_url {
        guard.cfg.whisper_url = u;
        notes.push("whisper_url override".into());
    }
    if let Some(u) = sel.llm_base_url {
        guard.cfg.llm_base_url = u;
        notes.push("llm_base_url override".into());
    }
    if let Some(m) = sel.model_name {
        guard.cfg.model_name = m;
        notes.push("model_name override".into());
    }
    if let Some(u) = sel.tts_url {
        guard.cfg.tts_url = u;
        notes.push("tts_url override".into());
    }
    if let Some(b) = sel.tts_backend {
        if let Some(tb) = parse_tts_backend(&b) {
            guard.cfg.tts = tb;
            notes.push(format!("tts_backend={b}"));
        }
    }
    if let Some(v) = sel.voice {
        guard.cfg.supertonic_voice = v;
        notes.push("voice override".into());
    }
    if let Some(l) = sel.language {
        guard.cfg.language = l.clone();
        guard.cfg.tts_language = l;
        notes.push("language override".into());
    }

    // Snapshot for side-effects outside the write lock.
    let tts_backend = guard.cfg.tts;
    let tts_url = guard.cfg.tts_url.clone();
    let need_asr_reload = sel
        .asr
        .as_ref()
        .and_then(|id| asr_whisper_model(id).map(|m| (guard.cfg.whisper_url.clone(), m)));

    drop(guard);

    // Side-effect: ask faster-whisper to reload model if ASR id maps to one.
    if let Some((url, model)) = need_asr_reload {
        match reload_whisper_model(&url, model).await {
            Ok(msg) => notes.push(msg),
            Err(e) => {
                warn!("ASR model reload failed: {e:#}");
                notes.push(format!("ASR reload warn: {e:#}"));
            }
        }
    }

    // Probe HTTP TTS after switching to Qwen/HTTP — silent failure is the main
    // lab footgun (hot-swap succeeds, speech then Connect refused on :8083).
    if matches!(tts_backend, TtsBackend::Http) {
        match probe_http_tts(&tts_url).await {
            Ok(()) => notes.push(format!("TTS HTTP reachable ({tts_url})")),
            Err(e) => {
                ok = false;
                let hint = format!(
                    "TTS HTTP nicht erreichbar ({tts_url}): {e}. \
                     Starte Qwen TTS: scripts/start_qwen_sycl.ps1 \
                     (tts-server auf 127.0.0.1:8083). Bis dahin keine Stimme."
                );
                warn!("{hint}");
                notes.push(hint);
            }
        }
    }

    let guard = rt.read().await;
    let mut st = status_of(&guard);
    st.ok = ok;
    st.message = if notes.is_empty() {
        "no changes".into()
    } else {
        notes.join("; ")
    };
    info!("Hot-swap applied (ok={ok}): {}", st.message);
    (st, notes)
}

/// Cheap reachability probe for HTTP TTS (Qwen tts-server).
async fn probe_http_tts(tts_url: &str) -> Result<(), String> {
    let host_port = tts_url
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split('/')
        .next()
        .unwrap_or("")
        .to_string();
    if host_port.is_empty() {
        return Err("empty tts_url".into());
    }

    // 1) TCP connect (fast fail when nothing listens).
    match tokio::time::timeout(
        std::time::Duration::from_millis(800),
        tokio::net::TcpStream::connect(&host_port),
    )
    .await
    {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => {
            return Err(format!("tcp {host_port}: {e}"));
        }
        Err(_) => {
            return Err(format!("tcp timeout {host_port}"));
        }
    }

    // 2) Optional HTTP GET on /health or root — accept any response (incl. 404).
    let base = if let Some(idx) = tts_url.find("/v1/") {
        &tts_url[..idx]
    } else {
        tts_url.trim_end_matches('/')
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1200))
        .build()
        .map_err(|e| e.to_string())?;
    for path in ["/health", "/v1/models", "/"] {
        let url = format!("{base}{path}");
        match client.get(&url).send().await {
            Ok(_) => return Ok(()),
            Err(e) if e.is_connect() => return Err(format!("http connect {url}: {e}")),
            Err(_) => continue,
        }
    }
    // TCP worked; treat as ok even if no GET path answered.
    Ok(())
}

fn parse_tts_backend(s: &str) -> Option<TtsBackend> {
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(TtsBackend::Auto),
        "supertonic" => Some(TtsBackend::Supertonic),
        "http" => Some(TtsBackend::Http),
        "piper" => Some(TtsBackend::Piper),
        "system" => Some(TtsBackend::System),
        _ => None,
    }
}

fn asr_whisper_model(id: &str) -> Option<&'static str> {
    match id {
        "fw-tiny" => Some("tiny"),
        "fw-base" => Some("base"),
        "fw-small" => Some("small"),
        "fw-medium" => Some("medium"),
        _ => None,
    }
}

fn apply_asr(rt: &mut RuntimeState, id: &str) -> Option<String> {
    rt.asr_id = id.to_string();
    match id {
        "fw-tiny" | "fw-base" | "fw-small" | "fw-medium" => {
            // Same host STT process; model reloaded via /reload.
            if rt.cfg.whisper_url.is_empty() {
                rt.cfg.whisper_url = "http://127.0.0.1:8082".into();
            }
            Some(format!("ASR → faster-whisper {id}"))
        }
        "parakeet-tdt-0.6b-v3" | "parakeet" => {
            rt.cfg.whisper_url = "http://127.0.0.1:8082".into();
            Some(format!(
                "ASR → Parakeet HTTP ({id}) @ {}",
                rt.cfg.whisper_url
            ))
        }
        "voxtral-mini-4b-realtime" | "voxtral" => {
            rt.cfg.whisper_url = "http://127.0.0.1:8087".into();
            Some(format!(
                "ASR → Voxtral Mini 4B Realtime @ {}",
                rt.cfg.whisper_url
            ))
        }
        "wcpp-base" | "wcpp-small" => {
            // whisper.cpp server — typically same URL; no model hot-reload from here.
            rt.cfg.whisper_url = "http://127.0.0.1:8082".into();
            Some(format!(
                "ASR → whisper.cpp id={id} (ensure server serves that model)"
            ))
        }
        other => {
            warn!("Unknown ASR id {other}");
            Some(format!("ASR id '{other}' recorded (no mapping)"))
        }
    }
}

fn is_supertonic_voice(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_uppercase().as_str(),
        "M1" | "M2" | "M3" | "M4" | "M5" | "F1" | "F2" | "F3" | "F4" | "F5"
    )
}

fn is_kokoro_voice(name: &str) -> bool {
    let n = name.trim().to_ascii_lowercase();
    // Kokoro voice ids look like af_bella, bm_george, zf_xiaobei, …
    n.contains('_') && n.len() >= 4 && n.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
}

fn apply_tts(rt: &mut RuntimeState, id: &str) -> Option<String> {
    rt.tts_id = id.to_string();
    match id {
        "supertonic" => {
            rt.cfg.tts = TtsBackend::Supertonic;
            rt.cfg.tts_model = "tts".into();
            // Qwen speakers (Aiden, …) / Kokoro voices are invalid for Supertonic styles (M1–F5).
            if !is_supertonic_voice(&rt.cfg.supertonic_voice) {
                rt.cfg.supertonic_voice = "M1".into();
            }
            Some(format!(
                "TTS → Supertonic ONNX (voice={})",
                rt.cfg.supertonic_voice
            ))
        }
        "qwen-sycl" | "qwen-vulkan" | "qwen" | "http" => {
            rt.cfg.tts = TtsBackend::Http;
            rt.cfg.tts_url = "http://127.0.0.1:8083/v1/audio/speech".into();
            rt.cfg.tts_model = "tts".into();
            rt.cfg.tts_native_sample_rate = 24_000;
            // Qwen custom-voice default speaker (not a Supertonic style id)
            if is_supertonic_voice(&rt.cfg.supertonic_voice)
                || is_kokoro_voice(&rt.cfg.supertonic_voice)
                || rt.cfg.supertonic_voice.is_empty()
            {
                rt.cfg.supertonic_voice = "Aiden".into();
            }
            Some(format!("TTS → HTTP Qwen ({id}) @ {}", rt.cfg.tts_url))
        }
        "xtts-v2" | "xtts" => {
            rt.cfg.tts = TtsBackend::Http;
            rt.cfg.tts_url = "http://127.0.0.1:8091/v1/audio/speech".into();
            rt.cfg.tts_model = "coqui/XTTS-v2".into();
            rt.cfg.tts_native_sample_rate = 24_000;
            rt.cfg.supertonic_voice = "de_sample".into();
            Some(format!(
                "TTS → XTTS-v2 ONNX (voice={}) @ {}",
                rt.cfg.supertonic_voice, rt.cfg.tts_url
            ))
        }
        "kokoro" => {
            rt.cfg.tts = TtsBackend::Http;
            rt.cfg.tts_url = "http://127.0.0.1:8084/v1/audio/speech".into();
            rt.cfg.tts_model = "kokoro".into();
            rt.cfg.tts_native_sample_rate = 24_000;
            // Prefer a real Kokoro voice id; map away Supertonic/Qwen leftovers.
            if is_supertonic_voice(&rt.cfg.supertonic_voice)
                || !is_kokoro_voice(&rt.cfg.supertonic_voice)
                || rt.cfg.supertonic_voice.is_empty()
            {
                rt.cfg.supertonic_voice = "af_bella".into();
            }
            Some(format!(
                "TTS → Kokoro HTTP (voice={}) @ {}",
                rt.cfg.supertonic_voice, rt.cfg.tts_url
            ))
        }
        "vibevoice-realtime-0.5b" | "vibevoice" => {
            rt.cfg.tts = TtsBackend::Http;
            rt.cfg.tts_url = "http://127.0.0.1:8089/v1/audio/speech".into();
            rt.cfg.tts_model = "vibevoice-realtime-0.5b".into();
            rt.cfg.tts_native_sample_rate = 24_000;
            if is_supertonic_voice(&rt.cfg.supertonic_voice)
                || is_kokoro_voice(&rt.cfg.supertonic_voice)
                || rt.cfg.supertonic_voice.eq_ignore_ascii_case("Aiden")
                || rt.cfg.supertonic_voice.eq_ignore_ascii_case("default")
                || rt.cfg.supertonic_voice.is_empty()
            {
                rt.cfg.supertonic_voice = "emma".into();
            }
            Some(format!(
                "TTS → VibeVoice Realtime 0.5B (voice={}) @ {}",
                rt.cfg.supertonic_voice, rt.cfg.tts_url
            ))
        }
        "higgs-tts-3-4b" | "higgs" => {
            rt.cfg.tts = TtsBackend::Http;
            rt.cfg.tts_url = "http://127.0.0.1:8086/v1/audio/speech".into();
            rt.cfg.tts_model = "bosonai/higgs-tts-3-4b".into();
            rt.cfg.tts_native_sample_rate = 24_000;
            // Higgs uses "default" (or reference-audio cloning); clear engine-specific leftovers.
            if is_supertonic_voice(&rt.cfg.supertonic_voice)
                || is_kokoro_voice(&rt.cfg.supertonic_voice)
                || rt.cfg.supertonic_voice.eq_ignore_ascii_case("Aiden")
                || rt.cfg.supertonic_voice.is_empty()
            {
                rt.cfg.supertonic_voice = "default".into();
            }
            Some(format!(
                "TTS → Higgs TTS 3 HTTP (voice={}) @ {}",
                rt.cfg.supertonic_voice, rt.cfg.tts_url
            ))
        }
        "system" => {
            rt.cfg.tts = TtsBackend::System;
            Some("TTS → system (SAPI/espeak)".into())
        }
        "piper" => {
            rt.cfg.tts = TtsBackend::Piper;
            Some("TTS → Piper (needs --piper-model)".into())
        }
        other => {
            warn!("Unknown TTS id {other}");
            Some(format!("TTS id '{other}' recorded (no mapping)"))
        }
    }
}

fn apply_llm(rt: &mut RuntimeState, id: &str) -> Option<String> {
    rt.llm_id = id.to_string();
    match id {
        "qwen-cloud" | "qwen3.5:cloud" => {
            rt.cfg.llm_base_url = "http://127.0.0.1:11434/v1".into();
            rt.cfg.model_name = "qwen3.5:cloud".into();
            rt.cfg.llm_no_think = true;
            Some("LLM → Ollama qwen3.5:cloud".into())
        }
        "llama-1b" | "llama3.2:1b" => {
            rt.cfg.llm_base_url = "http://127.0.0.1:11434/v1".into();
            rt.cfg.model_name = "llama3.2:1b".into();
            rt.cfg.llm_no_think = true;
            Some("LLM → Ollama llama3.2:1b".into())
        }
        "granite-2b" => {
            // Prefer llama.cpp OpenAI endpoint if present; still set model name.
            if !rt.cfg.llm_base_url.contains("11434") {
                // keep existing non-ollama url
            } else {
                // If currently on Ollama, try classic llama-server port
                rt.cfg.llm_base_url = "http://127.0.0.1:8081/v1".into();
            }
            rt.cfg.model_name = "granite-3.3-2b-instruct".into();
            Some(format!(
                "LLM → granite-3.3-2b-instruct @ {}",
                rt.cfg.llm_base_url
            ))
        }
        "qwen-1.5b" => {
            if rt.cfg.llm_base_url.contains("11434") {
                rt.cfg.llm_base_url = "http://127.0.0.1:8081/v1".into();
            }
            rt.cfg.model_name = "qwen2.5-1.5b-instruct".into();
            Some(format!("LLM → qwen2.5-1.5b @ {}", rt.cfg.llm_base_url))
        }
        other => {
            // Treat as raw Ollama model name if it looks like one
            if other.contains(':') {
                rt.cfg.llm_base_url = "http://127.0.0.1:11434/v1".into();
                rt.cfg.model_name = other.to_string();
                return Some(format!("LLM → Ollama {other}"));
            }
            warn!("Unknown LLM id {other}");
            Some(format!("LLM id '{other}' recorded (no mapping)"))
        }
    }
}

pub(crate) async fn reload_whisper_model(base_url: &str, model: &str) -> anyhow::Result<String> {
    let url = format!("{}/reload", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()?;
    let body = serde_json::json!({ "model": model });
    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let t = resp.text().await.unwrap_or_default();
        // Older whisper servers without /reload — soft fail
        if status.as_u16() == 404 {
            return Ok(format!(
                "ASR model '{model}' requested (server has no /reload — restart STT manually)"
            ));
        }
        anyhow::bail!("reload {status}: {t}");
    }
    let v: serde_json::Value = resp.json().await.unwrap_or_default();
    Ok(format!(
        "ASR reloaded model={} ({})",
        model,
        v.get("model").and_then(|x| x.as_str()).unwrap_or(model)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[tokio::test]
    async fn turn_pause_rejects_new_turns_and_waits_for_existing_turn() {
        let turns = Arc::new(TurnCoordinator::default());
        let lease = turns.try_acquire().expect("first turn");
        let pause = turns.pause();
        assert!(turns.try_acquire().is_none());
        assert!(
            !turns.wait_idle(Duration::from_millis(5)).await,
            "existing turn must keep the coordinator busy"
        );
        drop(lease);
        assert!(turns.wait_idle(Duration::from_millis(50)).await);
        drop(pause);
        assert!(turns.try_acquire().is_some());
    }

    #[test]
    fn xtts_http_endpoint_and_model_are_inferred() {
        let mut config = Config::parse_from(["s2s-vulkan"]);
        config.tts = TtsBackend::Http;
        config.tts_url = "http://host.docker.internal:8091/v1/audio/speech".into();
        config.tts_model = "coqui/XTTS-v2".into();
        assert_eq!(infer_tts_id(&config), "xtts-v2");
    }

    #[tokio::test]
    async fn xtts_runtime_mapping_uses_24khz_and_default_voice() {
        let runtime = runtime_from(Config::parse_from(["s2s-vulkan"]));
        let mut state = runtime.write().await;
        let message = apply_tts(&mut state, "xtts-v2").unwrap();
        assert!(message.contains("XTTS-v2"));
        assert_eq!(state.cfg.tts_url, "http://127.0.0.1:8091/v1/audio/speech");
        assert_eq!(state.cfg.tts_model, "coqui/XTTS-v2");
        assert_eq!(state.cfg.tts_native_sample_rate, 24_000);
        assert_eq!(state.cfg.supertonic_voice, "de_sample");
    }
}
