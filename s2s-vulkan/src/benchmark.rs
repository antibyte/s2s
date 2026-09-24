//! Persistent, repeatable benchmark runs for the active speech-lab stack.

use crate::runtime::SharedRuntime;
use crate::stt;
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::warn;
use uuid::Uuid;

const DEFAULT_TTS_PROMPTS: &[&str] = &[
    "Guten Morgen, wie kann ich dir heute helfen?",
    "Die schnelle Antwort ist oft besser als eine lange Erklärung.",
    "Bitte prüfe Mikrofon, Netzwerk und Grafikkarte.",
    "Heute ist ein guter Tag für einen Sprachtest.",
    "Dieses Beispiel enthält Zahlen: zwölf, vierundzwanzig und zweihundert.",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkRequest {
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub prompts: Vec<String>,
}

fn default_kind() -> String {
    "tts".into()
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkRun {
    pub id: String,
    pub kind: String,
    pub status: String,
    pub created_at: i64,
    pub completed_at: Option<i64>,
    pub stack: Value,
    pub hardware: Value,
    pub request: Value,
    pub result: Option<Value>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkAccepted {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
struct TtsSampleResult {
    prompt: String,
    audio_url: String,
    ttfa_ms: f64,
    total_ms: f64,
    audio_seconds: f64,
    rtf: f64,
    bytes: usize,
    memory_mib: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BenchmarkRating {
    pub sample_index: usize,
    pub score: u8,
}

#[derive(Clone)]
pub struct BenchmarkService {
    store: BenchmarkStore,
    runtime: SharedRuntime,
    hardware: Value,
    client: reqwest::Client,
    audio_root: Arc<PathBuf>,
}

impl BenchmarkService {
    pub fn new(runtime: SharedRuntime, hardware: Value) -> Result<Self> {
        let path = benchmark_db_path();
        let audio_root = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("benchmarks");
        std::fs::create_dir_all(&audio_root).with_context(|| {
            format!("create benchmark audio directory {}", audio_root.display())
        })?;
        let store = BenchmarkStore::new(path)?;
        Ok(Self {
            store,
            runtime,
            hardware,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(180))
                .build()?,
            audio_root: Arc::new(audio_root),
        })
    }

    pub async fn start(&self, mut request: BenchmarkRequest) -> Result<BenchmarkAccepted> {
        request.kind = request.kind.trim().to_ascii_lowercase();
        if !matches!(request.kind.as_str(), "tts" | "asr" | "llm") {
            return Err(anyhow!(
                "benchmark kind '{}' is not implemented; supported: tts, asr, llm",
                request.kind
            ));
        }
        if request.prompts.is_empty() {
            request.prompts = DEFAULT_TTS_PROMPTS
                .iter()
                .map(|prompt| prompt.to_string())
                .collect();
        }
        if request.prompts.len() > 100
            || request
                .prompts
                .iter()
                .any(|prompt| prompt.is_empty() || prompt.len() > 1000)
        {
            return Err(anyhow!(
                "prompts must contain 1..100 entries of 1..1000 bytes"
            ));
        }

        let id = Uuid::new_v4().to_string();
        let stack = {
            let runtime = self.runtime.read().await;
            serde_json::to_value(crate::runtime::status_of(&runtime))?
        };
        self.store
            .create(
                id.clone(),
                request.kind.clone(),
                stack,
                self.hardware.clone(),
                serde_json::to_value(&request)?,
            )
            .await?;

        let service = self.clone();
        let run_id = id.clone();
        tokio::spawn(async move {
            let result = match request.kind.as_str() {
                "asr" => service.run_asr(&run_id, request).await,
                "llm" => service.run_llm(request).await,
                _ => service.run_tts(&run_id, request).await,
            };
            match result {
                Ok(value) => {
                    if let Err(error) = service.store.complete(run_id.clone(), value).await {
                        warn!("Unable to persist benchmark completion: {error:#}");
                    }
                }
                Err(error) => {
                    if let Err(store_error) = service
                        .store
                        .fail(run_id.clone(), format!("{error:#}"))
                        .await
                    {
                        warn!("Unable to persist benchmark failure: {store_error:#}");
                    }
                }
            }
        });
        Ok(BenchmarkAccepted {
            id,
            status: "running".into(),
        })
    }

    pub async fn get(&self, id: &str) -> Result<Option<BenchmarkRun>> {
        self.store.get(id.to_string()).await
    }

    pub async fn audio(&self, id: &str, sample_index: usize) -> Result<Option<Vec<u8>>> {
        validate_run_id(id)?;
        if sample_index >= 100 || self.store.get(id.to_string()).await?.is_none() {
            return Ok(None);
        }
        let path = self.audio_path(id, sample_index);
        match tokio::fs::read(path).await {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    pub async fn rate(&self, id: &str, rating: BenchmarkRating) -> Result<()> {
        validate_run_id(id)?;
        if rating.sample_index >= 100 || !(1..=5).contains(&rating.score) {
            return Err(anyhow!("sample_index must be 0..99 and score must be 1..5"));
        }
        let run = self
            .store
            .get(id.to_string())
            .await?
            .ok_or_else(|| anyhow!("benchmark not found"))?;
        let sample_count = run
            .result
            .as_ref()
            .and_then(|result| result["samples"].as_array())
            .map_or(0, Vec::len);
        if rating.sample_index >= sample_count {
            return Err(anyhow!("benchmark sample does not exist"));
        }
        self.store
            .rate(id.to_string(), rating.sample_index as u32, rating.score)
            .await
    }

    fn audio_path(&self, id: &str, sample_index: usize) -> PathBuf {
        self.audio_root
            .join(id)
            .join(format!("{sample_index:03}.wav"))
    }

    async fn run_tts(&self, run_id: &str, request: BenchmarkRequest) -> Result<Value> {
        validate_run_id(run_id)?;
        let cfg = self.runtime.read().await.cfg.clone();
        if !matches!(cfg.tts, crate::config::TtsBackend::Http) {
            return Err(anyhow!(
                "active TTS is not an HTTP sidecar; select a catalog TTS backend first"
            ));
        }
        let run_audio_dir = self.audio_root.join(run_id);
        tokio::fs::create_dir_all(&run_audio_dir).await?;
        let mut samples = Vec::with_capacity(request.prompts.len());
        for (sample_index, prompt) in request.prompts.into_iter().enumerate() {
            let mut body = serde_json::json!({
                "model": cfg.tts_model,
                "input": prompt,
                "voice": cfg.supertonic_voice,
                "language": cfg.resolve_tts_language(None),
                "response_format": "pcm"
            });
            let model_l = cfg.tts_model.to_ascii_lowercase();
            let url_l = cfg.tts_url.to_ascii_lowercase();
            if model_l.contains("qwen") || url_l.contains(":8083") {
                body["max_new_tokens"] = Value::from(cfg.tts_http_max_new_tokens.clamp(1, 256));
            }
            if model_l.contains("higgs") || url_l.contains("higgs") || url_l.contains(":8086") {
                body["max_new_tokens"] = Value::from(1024u32);
                body["temperature"] = Value::from(0.8);
                body["top_k"] = Value::from(50);
            }
            let started = Instant::now();
            let mut http = self.client.post(&cfg.tts_url).json(&body);
            if !cfg.tts_api_key.is_empty() {
                http = http.bearer_auth(&cfg.tts_api_key);
            }
            let response = http
                .send()
                .await
                .with_context(|| format!("benchmark POST {}", cfg.tts_url))?;
            if !response.status().is_success() {
                return Err(anyhow!("TTS benchmark HTTP {}", response.status()));
            }
            let sample_rate = response
                .headers()
                .get("x-sample-rate")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(cfg.tts_native_sample_rate)
                .max(1);
            let mut stream = response.bytes_stream();
            let mut first_byte_at = None;
            let mut pcm = Vec::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                if !chunk.is_empty() && first_byte_at.is_none() {
                    first_byte_at = Some(started.elapsed());
                }
                pcm.extend_from_slice(&chunk);
            }
            if pcm.is_empty() {
                return Err(anyhow!("TTS benchmark returned no audio"));
            }
            if pcm.len() % 2 != 0 {
                return Err(anyhow!("TTS benchmark returned an odd PCM byte count"));
            }
            let elapsed = started.elapsed();
            let bytes = pcm.len();
            let audio_seconds = bytes as f64 / 2.0 / sample_rate as f64;
            let wav = pcm16_wav(&pcm, sample_rate)?;
            let audio_path = self.audio_path(run_id, sample_index);
            let part_path = audio_path.with_extension("wav.part");
            tokio::fs::write(&part_path, wav).await?;
            tokio::fs::rename(&part_path, &audio_path).await?;
            let memory_mib = self.managed_memory_mib(&cfg.tts_url).await;
            samples.push(TtsSampleResult {
                prompt,
                audio_url: format!("/api/v1/benchmarks/{run_id}/audio/{sample_index}"),
                ttfa_ms: first_byte_at.unwrap_or(elapsed).as_secs_f64() * 1000.0,
                total_ms: elapsed.as_secs_f64() * 1000.0,
                audio_seconds,
                rtf: elapsed.as_secs_f64() / audio_seconds.max(f64::EPSILON),
                bytes,
                memory_mib,
            });
        }
        let ttfa: Vec<f64> = samples.iter().map(|sample| sample.ttfa_ms).collect();
        let rtf: Vec<f64> = samples.iter().map(|sample| sample.rtf).collect();
        let memory: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample.memory_mib)
            .collect();
        let memory_max_mib = (!memory.is_empty()).then(|| percentile(&memory, 1.0));
        Ok(serde_json::json!({
            "summary": {
                "samples": samples.len(),
                "ttfa_median_ms": percentile(&ttfa, 0.5),
                "ttfa_p95_ms": percentile(&ttfa, 0.95),
                "rtf_median": percentile(&rtf, 0.5),
                "rtf_p95": percentile(&rtf, 0.95),
                "memory_max_mib": memory_max_mib
            },
            "samples": samples
        }))
    }

    async fn run_asr(&self, run_id: &str, request: BenchmarkRequest) -> Result<Value> {
        validate_run_id(run_id)?;
        let cfg = self.runtime.read().await.cfg.clone();
        if !matches!(cfg.tts, crate::config::TtsBackend::Http) {
            return Err(anyhow!(
                "ASR benchmark needs the active HTTP TTS sidecar to create repeatable samples"
            ));
        }
        let run_audio_dir = self.audio_root.join(run_id);
        tokio::fs::create_dir_all(&run_audio_dir).await?;
        let mut samples = Vec::with_capacity(request.prompts.len());
        for (sample_index, prompt) in request.prompts.into_iter().enumerate() {
            let mut tts_body = serde_json::json!({
                "model": cfg.tts_model,
                "input": prompt,
                "voice": cfg.supertonic_voice,
                "language": cfg.resolve_tts_language(None),
                "response_format": "wav"
            });
            let model_l = cfg.tts_model.to_ascii_lowercase();
            if model_l.contains("qwen") {
                tts_body["max_new_tokens"] = Value::from(cfg.tts_http_max_new_tokens.clamp(1, 256));
            }
            if model_l.contains("higgs") {
                tts_body["max_new_tokens"] = Value::from(1024u32);
                tts_body["temperature"] = Value::from(0.8);
                tts_body["top_k"] = Value::from(50);
            }
            let mut tts_request = self.client.post(&cfg.tts_url).json(&tts_body);
            if !cfg.tts_api_key.is_empty() {
                tts_request = tts_request.bearer_auth(&cfg.tts_api_key);
            }
            let tts_response = tts_request.send().await?.error_for_status()?;
            let sample_rate = tts_response
                .headers()
                .get("x-sample-rate")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(cfg.tts_native_sample_rate)
                .max(1);
            let audio = tts_response.bytes().await?;
            let wav = if audio.starts_with(b"RIFF") {
                audio.to_vec()
            } else {
                pcm16_wav(&audio, sample_rate)?
            };
            let audio_path = self.audio_path(run_id, sample_index);
            let part_path = audio_path.with_extension("wav.part");
            tokio::fs::write(&part_path, &wav).await?;
            tokio::fs::rename(&part_path, &audio_path).await?;

            let started = Instant::now();
            let transcript = stt::transcribe_wav_bytes(&self.client, &cfg, &wav)
                .await
                .context("ASR benchmark transcription")?;
            let asr_ms = started.elapsed().as_secs_f64() * 1000.0;
            let memory_mib = self.managed_memory_mib(&cfg.whisper_url).await;
            samples.push(serde_json::json!({
                "prompt": prompt,
                "transcript": transcript,
                "audio_url": format!("/api/v1/benchmarks/{run_id}/audio/{sample_index}"),
                "asr_ms": asr_ms,
                "wer": word_error_rate(&prompt, &transcript),
                "cer": character_error_rate(&prompt, &transcript),
                "memory_mib": memory_mib
            }));
        }
        let latency: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample["asr_ms"].as_f64())
            .collect();
        let wer: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample["wer"].as_f64())
            .collect();
        let cer: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample["cer"].as_f64())
            .collect();
        let memory: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample["memory_mib"].as_f64())
            .collect();
        let memory_max_mib = (!memory.is_empty()).then(|| percentile(&memory, 1.0));
        Ok(serde_json::json!({
            "summary": {
                "samples": samples.len(),
                "asr_median_ms": percentile(&latency, 0.5),
                "asr_p95_ms": percentile(&latency, 0.95),
                "wer_mean": mean(&wer),
                "cer_mean": mean(&cer),
                "memory_max_mib": memory_max_mib
            },
            "samples": samples
        }))
    }

    async fn run_llm(&self, request: BenchmarkRequest) -> Result<Value> {
        let cfg = self.runtime.read().await.cfg.clone();
        let url = format!(
            "{}/chat/completions",
            cfg.llm_base_url.trim_end_matches('/')
        );
        let mut samples = Vec::with_capacity(request.prompts.len());
        for prompt in request.prompts {
            let body = serde_json::json!({
                "model": cfg.model_name,
                "messages": [
                    {"role": "system", "content": cfg.llm_system_prompt(None)},
                    {"role": "user", "content": prompt}
                ],
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "stream": true
            });
            let mut request = self.client.post(&url).json(&body);
            if !cfg.llm_api_key.is_empty() {
                request = request.bearer_auth(&cfg.llm_api_key);
            }
            let started = Instant::now();
            let response = request
                .send()
                .await
                .with_context(|| format!("LLM benchmark POST {url}"))?
                .error_for_status()?;
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut completion = String::new();
            let mut first_token = None;
            while let Some(chunk) = stream.next().await {
                buffer.push_str(&String::from_utf8_lossy(&chunk?));
                while let Some(newline) = buffer.find('\n') {
                    let line = buffer[..newline].trim().to_string();
                    buffer = buffer[newline + 1..].to_string();
                    let data = line.strip_prefix("data:").map(str::trim).unwrap_or(&line);
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    let Ok(value) = serde_json::from_str::<Value>(data) else {
                        continue;
                    };
                    if let Some(text) = value["choices"][0]["delta"]["content"].as_str() {
                        if !text.is_empty() {
                            first_token.get_or_insert_with(|| started.elapsed());
                            completion.push_str(text);
                        }
                    }
                }
            }
            let elapsed = started.elapsed();
            let estimated_tokens = completion.split_whitespace().count().max(1);
            let memory_mib = self.managed_memory_mib(&cfg.llm_base_url).await;
            samples.push(serde_json::json!({
                "prompt": prompt,
                "completion": completion,
                "ttft_ms": first_token.unwrap_or(elapsed).as_secs_f64() * 1000.0,
                "total_ms": elapsed.as_secs_f64() * 1000.0,
                "tokens_estimated": estimated_tokens,
                "tokens_per_second":
                    estimated_tokens as f64 / elapsed.as_secs_f64().max(f64::EPSILON),
                "memory_mib": memory_mib
            }));
        }
        let ttft: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample["ttft_ms"].as_f64())
            .collect();
        let total: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample["total_ms"].as_f64())
            .collect();
        let throughput: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample["tokens_per_second"].as_f64())
            .collect();
        let memory: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample["memory_mib"].as_f64())
            .collect();
        let memory_max_mib = (!memory.is_empty()).then(|| percentile(&memory, 1.0));
        Ok(serde_json::json!({
            "summary": {
                "samples": samples.len(),
                "ttft_median_ms": percentile(&ttft, 0.5),
                "ttft_p95_ms": percentile(&ttft, 0.95),
                "total_median_ms": percentile(&total, 0.5),
                "tokens_per_second_median": percentile(&throughput, 0.5),
                "memory_max_mib": memory_max_mib
            },
            "samples": samples
        }))
    }

    async fn managed_memory_mib(&self, endpoint: &str) -> Option<f64> {
        let proxy = std::env::var("S2S_DOCKER_PROXY_URL").ok()?;
        let container = managed_container_from_endpoint(endpoint)?;
        let inspect_url = format!(
            "{}/containers/{container}/json",
            proxy.trim_end_matches('/')
        );
        let inspect: Value = self
            .client
            .get(inspect_url)
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json()
            .await
            .ok()?;
        if inspect["Config"]["Labels"]["s2s.lab.managed"].as_str() != Some("true") {
            return None;
        }
        let stats_url = format!(
            "{}/containers/{container}/stats?stream=false",
            proxy.trim_end_matches('/')
        );
        let stats: Value = self
            .client
            .get(stats_url)
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json()
            .await
            .ok()?;
        let usage = stats["memory_stats"]["usage"].as_u64()?;
        let inactive_file = stats["memory_stats"]["stats"]["inactive_file"]
            .as_u64()
            .or_else(|| stats["memory_stats"]["stats"]["total_inactive_file"].as_u64())
            .unwrap_or_default();
        Some(usage.saturating_sub(inactive_file) as f64 / (1024.0 * 1024.0))
    }
}

#[derive(Clone)]
struct BenchmarkStore {
    path: Arc<PathBuf>,
}

impl BenchmarkStore {
    fn new(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create benchmark directory {}", parent.display()))?;
        }
        let connection = Connection::open(&path)
            .with_context(|| format!("open benchmark database {}", path.display()))?;
        connection.execute_batch(
            "
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                completed_at INTEGER,
                stack_json TEXT NOT NULL,
                hardware_json TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS benchmark_ratings (
                run_id TEXT NOT NULL,
                sample_index INTEGER NOT NULL,
                score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5),
                created_at INTEGER NOT NULL,
                PRIMARY KEY(run_id, sample_index),
                FOREIGN KEY(run_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE
            );
            ",
        )?;
        Ok(Self {
            path: Arc::new(path),
        })
    }

    async fn create(
        &self,
        id: String,
        kind: String,
        stack: Value,
        hardware: Value,
        request: Value,
    ) -> Result<()> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let connection = Connection::open(path.as_ref())?;
            connection.execute(
                "INSERT INTO benchmark_runs
                 (id, kind, status, created_at, stack_json, hardware_json, request_json)
                 VALUES (?1, ?2, 'running', ?3, ?4, ?5, ?6)",
                params![
                    id,
                    kind,
                    unix_timestamp(),
                    stack.to_string(),
                    hardware.to_string(),
                    request.to_string()
                ],
            )?;
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(())
    }

    async fn complete(&self, id: String, result: Value) -> Result<()> {
        self.finish(id, "completed", Some(result), None).await
    }

    async fn fail(&self, id: String, error: String) -> Result<()> {
        self.finish(id, "failed", None, Some(error)).await
    }

    async fn rate(&self, id: String, sample_index: u32, score: u8) -> Result<()> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let connection = Connection::open(path.as_ref())?;
            connection.execute(
                "INSERT INTO benchmark_ratings(run_id, sample_index, score, created_at)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(run_id, sample_index)
                 DO UPDATE SET score=excluded.score, created_at=excluded.created_at",
                params![id, sample_index, score, unix_timestamp()],
            )?;
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(())
    }

    async fn finish(
        &self,
        id: String,
        status: &'static str,
        result: Option<Value>,
        error: Option<String>,
    ) -> Result<()> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let connection = Connection::open(path.as_ref())?;
            connection.execute(
                "UPDATE benchmark_runs
                 SET status=?2, completed_at=?3, result_json=?4, error=?5
                 WHERE id=?1",
                params![
                    id,
                    status,
                    unix_timestamp(),
                    result.map(|value| value.to_string()),
                    error
                ],
            )?;
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(())
    }

    async fn get(&self, id: String) -> Result<Option<BenchmarkRun>> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let connection = Connection::open(path.as_ref())?;
            let mut statement = connection.prepare(
                "SELECT id, kind, status, created_at, completed_at,
                        stack_json, hardware_json, request_json, result_json, error
                 FROM benchmark_runs WHERE id=?1",
            )?;
            let mut rows = statement.query(params![id])?;
            let Some(row) = rows.next()? else {
                return Ok(None);
            };
            let parse = |raw: String| -> rusqlite::Result<Value> {
                serde_json::from_str(&raw).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        raw.len(),
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })
            };
            let result_raw: Option<String> = row.get(8)?;
            Ok(Some(BenchmarkRun {
                id: row.get(0)?,
                kind: row.get(1)?,
                status: row.get(2)?,
                created_at: row.get(3)?,
                completed_at: row.get(4)?,
                stack: parse(row.get(5)?)?,
                hardware: parse(row.get(6)?)?,
                request: parse(row.get(7)?)?,
                result: result_raw.map(parse).transpose()?,
                error: row.get(9)?,
            }))
        })
        .await?
    }
}

impl BenchmarkRun {
    pub fn to_csv(&self) -> String {
        let mut output = String::from(
            "run_id,kind,status,prompt,transcript,completion,ttfa_ms,ttft_ms,asr_ms,total_ms,audio_seconds,rtf,wer,cer,tokens_per_second,bytes,memory_mib\n",
        );
        let samples = self
            .result
            .as_ref()
            .and_then(|value| value["samples"].as_array());
        if let Some(samples) = samples {
            for sample in samples {
                output.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    csv(&self.id),
                    csv(&self.kind),
                    csv(&self.status),
                    csv(sample["prompt"].as_str().unwrap_or("")),
                    csv(sample["transcript"].as_str().unwrap_or("")),
                    csv(sample["completion"].as_str().unwrap_or("")),
                    sample["ttfa_ms"].as_f64().unwrap_or_default(),
                    sample["ttft_ms"].as_f64().unwrap_or_default(),
                    sample["asr_ms"].as_f64().unwrap_or_default(),
                    sample["total_ms"].as_f64().unwrap_or_default(),
                    sample["audio_seconds"].as_f64().unwrap_or_default(),
                    sample["rtf"].as_f64().unwrap_or_default(),
                    sample["wer"].as_f64().unwrap_or_default(),
                    sample["cer"].as_f64().unwrap_or_default(),
                    sample["tokens_per_second"].as_f64().unwrap_or_default(),
                    sample["bytes"].as_u64().unwrap_or_default(),
                    sample["memory_mib"].as_f64().unwrap_or_default()
                ));
            }
        }
        output
    }
}

fn managed_container_from_endpoint(endpoint: &str) -> Option<String> {
    let remainder = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))?;
    let authority = remainder.split('/').next()?;
    let host = authority.rsplit('@').next()?.split(':').next()?;
    let allowed = host == "supertonic"
        || host == "llama-fallback"
        || host.starts_with("whisper-")
        || host.starts_with("tts-")
        || host.starts_with("parakeet-")
        || host.starts_with("voxtral-");
    if !allowed
        || !host
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return None;
    }
    Some(format!("s2s-{host}"))
}

fn benchmark_db_path() -> PathBuf {
    if let Ok(path) = std::env::var("S2S_LAB_DB") {
        return PathBuf::from(path);
    }
    if crate::gpu::running_in_container() {
        PathBuf::from("/data/lab.db")
    } else {
        Path::new("target").join("s2s-lab.db")
    }
}

fn validate_run_id(id: &str) -> Result<()> {
    let parsed = Uuid::parse_str(id).context("invalid benchmark id")?;
    if parsed.to_string() != id.to_ascii_lowercase() {
        return Err(anyhow!("benchmark id must use canonical UUID form"));
    }
    Ok(())
}

fn pcm16_wav(pcm: &[u8], sample_rate: u32) -> Result<Vec<u8>> {
    let data_len = u32::try_from(pcm.len()).context("PCM sample is too large for WAV")?;
    let riff_len = data_len
        .checked_add(36)
        .ok_or_else(|| anyhow!("PCM sample is too large for WAV"))?;
    let byte_rate = sample_rate
        .checked_mul(2)
        .ok_or_else(|| anyhow!("invalid WAV sample rate"))?;
    let mut wav = Vec::with_capacity(pcm.len() + 44);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_len.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    Ok(wav)
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let index = ((sorted.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn normalize_text(value: &str) -> Vec<String> {
    value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_lowercase())
        .collect()
}

fn edit_distance<T: Eq>(reference: &[T], hypothesis: &[T]) -> usize {
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0usize; hypothesis.len() + 1];
    for (reference_index, reference_item) in reference.iter().enumerate() {
        current[0] = reference_index + 1;
        for (hypothesis_index, hypothesis_item) in hypothesis.iter().enumerate() {
            let substitution =
                previous[hypothesis_index] + usize::from(reference_item != hypothesis_item);
            current[hypothesis_index + 1] = substitution
                .min(previous[hypothesis_index + 1] + 1)
                .min(current[hypothesis_index] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[hypothesis.len()]
}

fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize_text(reference);
    let hypothesis = normalize_text(hypothesis);
    edit_distance(&reference, &hypothesis) as f64 / reference.len().max(1) as f64
}

fn character_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let reference: Vec<char> = normalize_text(reference).join("").chars().collect();
    let hypothesis: Vec<char> = normalize_text(hypothesis).join("").chars().collect();
    edit_distance(&reference, &hypothesis) as f64 / reference.len().max(1) as f64
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn csv(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_is_stable() {
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.5), 3.0);
        assert_eq!(percentile(&[], 0.95), 0.0);
    }

    #[test]
    fn csv_escapes_quotes() {
        assert_eq!(csv("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn wav_header_contains_pcm_payload() {
        let wav = pcm16_wav(&[1, 2, 3, 4], 24_000).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[40..44], &4u32.to_le_bytes());
        assert_eq!(&wav[44..], &[1, 2, 3, 4]);
    }

    #[test]
    fn error_rates_normalize_case_and_punctuation() {
        assert_eq!(word_error_rate("Hallo, Welt!", "hallo welt"), 0.0);
        assert_eq!(character_error_rate("Grüße!", "grüße"), 0.0);
        assert_eq!(word_error_rate("eins zwei", "eins drei"), 0.5);
    }

    #[test]
    fn memory_stats_only_accept_registry_sidecar_hosts() {
        assert_eq!(
            managed_container_from_endpoint("http://whisper-tiny:8082/inference").as_deref(),
            Some("s2s-whisper-tiny")
        );
        assert_eq!(
            managed_container_from_endpoint("http://tts-qwen-sycl-aot:8083/v1/audio/speech")
                .as_deref(),
            Some("s2s-tts-qwen-sycl-aot")
        );
        assert_eq!(
            managed_container_from_endpoint("http://host.docker.internal:11434/v1"),
            None
        );
        assert_eq!(
            managed_container_from_endpoint("http://supertonic.evil:8083/v1/audio/speech"),
            None
        );
    }
}
