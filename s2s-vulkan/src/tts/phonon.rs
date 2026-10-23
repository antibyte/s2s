//! Offline Phonon CPU sidecar. The response body owns the native process so
//! disconnecting a client drops, kills and cleans up its synthesis immediately.

use super::{json_error, SpeechRequest};
use crate::config::Config;
use anyhow::{bail, Context, Result};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, Response, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::Semaphore;

pub(crate) const MODEL_ID: &str = "phonon-40m-multilingual";
pub(crate) const DEFAULT_VOICE: &str = "2A2fFf8SkDEJ8nzx";
const MAX_WAV_BYTES: u64 = 16 << 20;
const SAMPLE_RATE: u32 = 24_000;

#[derive(Deserialize)]
struct ResourceFile {
    name: String,
    size: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct Voice {
    id: String,
    name: String,
    filename: String,
    size: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    runtime_version: String,
    runtime: ResourceFile,
    model_files: Vec<ResourceFile>,
    voices: Vec<Voice>,
}

static MANIFEST: LazyLock<Manifest> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../config/phonon.json"))
        .expect("invalid embedded Phonon manifest")
});

pub(crate) fn has_voice(id: &str) -> bool {
    MANIFEST.voices.iter().any(|voice| voice.id == id)
}

struct Runtime {
    root: PathBuf,
    binary: PathBuf,
    language: String,
    // shortcut: serialize CPU inference; add a bounded pool only if measured demand requires it.
    slot: Semaphore,
}

impl Runtime {
    async fn load(root: PathBuf, language: String) -> Result<Self> {
        #[cfg(target_arch = "x86_64")]
        if !std::is_x86_feature_detected!("avx2")
            || !std::is_x86_feature_detected!("fma")
            || !std::is_x86_feature_detected!("f16c")
        {
            bail!("Phonon v0.4.0 requires an x86-64-v3 CPU (AVX2, FMA, F16C)");
        }
        let verified_root = root.clone();
        tokio::task::spawn_blocking(move || verify_resources(&verified_root)).await??;
        let binary = root.join(&MANIFEST.runtime.name);
        let version = tokio::time::timeout(
            Duration::from_secs(5),
            Command::new(&binary)
                .arg("--version")
                .env_clear()
                .env("HF_HUB_OFFLINE", "1")
                .kill_on_drop(true)
                .output(),
        )
        .await
        .context("Phonon version check timed out")??;
        if !version.status.success()
            || !String::from_utf8_lossy(&version.stdout).contains(&MANIFEST.runtime_version)
        {
            bail!("incompatible Phonon runtime");
        }
        Ok(Self {
            root,
            binary,
            language,
            slot: Semaphore::new(1),
        })
    }

    async fn synthesize(&self, text: &str, language: &str, voice_id: &str) -> Result<Vec<u8>> {
        let _permit = self.slot.acquire().await?;
        let voice = MANIFEST
            .voices
            .iter()
            .find(|voice| voice.id == voice_id)
            .context("unknown Phonon voice")?;
        let work = PrivateOutput::create()?;
        let output = work.0.join("speech.wav");
        let mut child = Command::new(&self.binary)
            .arg("--model")
            .arg(self.root.join("model"))
            .args(["--quant", "q8", "--device", "cpu", "--lang", language])
            .arg("--voice")
            .arg(self.root.join(&voice.filename))
            .arg("--voice-catalog")
            .arg(self.root.join("voices.jsonl"))
            .args(["--threads", "2", "-o"])
            .arg(&output)
            .arg("--")
            .arg(text)
            .current_dir(&work.0)
            .env_clear()
            .env("HF_HUB_OFFLINE", "1")
            .env("OMP_NUM_THREADS", "2")
            .env("LANG", "C.UTF-8")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("start native Phonon")?;
        let status = loop {
            tokio::select! {
                status = child.wait() => break status.context("wait for native Phonon")?,
                _ = tokio::time::sleep(Duration::from_millis(50)) => {
                    if tokio::fs::symlink_metadata(&output).await.is_ok_and(|file| file.len() > MAX_WAV_BYTES) {
                        bail!("Phonon output exceeds 16 MiB");
                    }
                }
            }
        };
        if !status.success() {
            bail!("native Phonon failed ({status})");
        }
        let metadata = tokio::fs::symlink_metadata(&output)
            .await
            .context("missing Phonon output")?;
        if !metadata.is_file() || metadata.len() < 44 || metadata.len() > MAX_WAV_BYTES {
            bail!("invalid Phonon output file");
        }
        Ok(tokio::fs::read(output).await?)
    }
}

struct PrivateOutput(PathBuf);

impl PrivateOutput {
    fn create() -> Result<Self> {
        let path = std::env::temp_dir().join(format!("s2s-phonon-{}", uuid::Uuid::new_v4()));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&path)?;
        Ok(Self(path))
    }
}

impl Drop for PrivateOutput {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn verify_file(path: &Path, size: u64, sha256: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("missing resource {}", path.display()))?;
    if !metadata.is_file() || metadata.len() != size {
        bail!("invalid resource size: {}", path.display());
    }
    let mut source = std::fs::File::open(path)?.take(size + 1);
    let mut digest = Sha256::new();
    let mut buffer = [0; 65_536];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    if hex::encode(digest.finalize()) != sha256 {
        bail!("Phonon resource checksum mismatch: {}", path.display());
    }
    Ok(())
}

fn verify_resources(root: &Path) -> Result<()> {
    verify_file(
        &root.join(&MANIFEST.runtime.name),
        MANIFEST.runtime.size,
        &MANIFEST.runtime.sha256,
    )?;
    for file in &MANIFEST.model_files {
        verify_file(
            &root.join("model").join(&file.name),
            file.size,
            &file.sha256,
        )?;
    }
    for voice in &MANIFEST.voices {
        verify_file(&root.join(&voice.filename), voice.size, &voice.sha256)?;
    }
    Ok(())
}

fn language(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    let base = normalized.split('-').next()?;
    matches!(base, "de" | "en" | "fr" | "es" | "pt").then(|| base.to_string())
}

fn wav_pcm(wav: &[u8]) -> Result<Vec<u8>> {
    if wav.len() as u64 > MAX_WAV_BYTES {
        bail!("Phonon WAV exceeds output limit");
    }
    let mut reader = hound::WavReader::new(std::io::Cursor::new(wav))?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_rate != SAMPLE_RATE
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        bail!("Phonon requires mono PCM16 WAV at 24 kHz");
    }
    let samples = reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if samples.is_empty() {
        bail!("empty Phonon WAV");
    }
    Ok(crate::audio::pcm::i16_to_bytes_le(&samples))
}

async fn speech(
    State(runtime): State<Arc<Runtime>>,
    Json(request): Json<SpeechRequest>,
) -> Response<Body> {
    if request.input.trim().is_empty() || request.input.chars().count() > 2000 {
        return json_error(StatusCode::BAD_REQUEST, "Phonon requires 1–2000 characters");
    }
    let Some(language) = language(if request.language.trim().is_empty() {
        &runtime.language
    } else {
        &request.language
    }) else {
        return json_error(StatusCode::BAD_REQUEST, "unsupported Phonon language");
    };
    let voice = if request.voice.is_empty() {
        DEFAULT_VOICE
    } else {
        &request.voice
    };
    if !has_voice(voice) {
        return json_error(StatusCode::BAD_REQUEST, "unknown Phonon voice ID");
    }
    let format = request.response_format.to_ascii_lowercase();
    if !matches!(format.as_str(), "wav" | "pcm") {
        return json_error(StatusCode::BAD_REQUEST, "Phonon supports wav or pcm");
    }
    let voice = voice.to_string();
    let pcm = format == "pcm";
    // Keeping inference inside the body stream propagates disconnect/cancellation
    // to Child::kill_on_drop, the private output guard and the semaphore permit.
    let audio = stream::once(async move {
        let result = tokio::time::timeout(Duration::from_secs(120), async {
            let wav = runtime
                .synthesize(&request.input, &language, &voice)
                .await?;
            let samples = wav_pcm(&wav)?;
            Ok::<_, anyhow::Error>(if pcm { samples } else { wav })
        })
        .await
        .context("Phonon synthesis timed out")
        .and_then(|result| result);
        result.map(bytes::Bytes::from).map_err(|error| {
            tracing::warn!("Phonon synthesis failed: {error:#}");
            std::io::Error::other("Phonon synthesis failed")
        })
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            if pcm { "audio/pcm" } else { "audio/wav" },
        )
        .header("x-sample-rate", SAMPLE_RATE)
        .body(Body::from_stream(audio))
        .unwrap()
}

pub(crate) async fn run_server(cfg: Config) -> Result<()> {
    let runtime = Arc::new(
        Runtime::load(
            cfg.phonon_resource_dir.clone(),
            cfg.resolve_tts_language(None),
        )
        .await?,
    );
    let app = Router::new()
        .route("/health", get(|| async { Json(serde_json::json!({"ready": true, "model": MODEL_ID, "runtime": MANIFEST.runtime_version})) }))
        .route("/v1/models", get(|| async { Json(serde_json::json!({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})) }))
        .route("/v1/audio/speech", post(speech))
        .layer(DefaultBodyLimit::max(16 * 1024))
        .with_state(runtime);
    let listener = tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port)).await?;
    tracing::info!("Phonon CPU sidecar listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voices_and_language_are_allowlisted() {
        assert_eq!(MANIFEST.voices.len(), 10);
        assert_eq!(
            MANIFEST
                .voices
                .iter()
                .find(|v| v.id == DEFAULT_VOICE)
                .unwrap()
                .name,
            "Nele"
        );
        assert!(has_voice(DEFAULT_VOICE));
        assert!(!has_voice("../../somewhere"));
        assert_eq!(language("de-AT").as_deref(), Some("de"));
        assert_eq!(language("pt_BR").as_deref(), Some("pt"));
        assert!(language("ja").is_none());
    }

    #[test]
    fn wav_requires_nonempty_mono_pcm16_at_24khz() {
        let wav = crate::audio::pcm::encode_wav_f32(&[0.1, -0.2], SAMPLE_RATE).unwrap();
        assert_eq!(wav_pcm(&wav).unwrap().len(), 4);
        assert!(wav_pcm(b"not a wave").is_err());
        assert!(wav_pcm(&crate::audio::pcm::encode_wav_f32(&[0.1], 16000).unwrap()).is_err());
        assert!(wav_pcm(&crate::audio::pcm::encode_wav_f32(&[], SAMPLE_RATE).unwrap()).is_err());
        assert!(wav_pcm(&wav[..wav.len() - 1]).is_err());
    }

    #[test]
    fn corrupt_resources_are_rejected() {
        let temporary = PrivateOutput::create().unwrap();
        let file = temporary.0.join("asset");
        std::fs::write(&file, b"abc").unwrap();
        assert!(verify_file(&file, 3, &hex::encode(Sha256::digest(b"abc"))).is_ok());
        assert!(verify_file(&file, 2, &hex::encode(Sha256::digest(b"abc"))).is_err());
        assert!(verify_file(&file, 3, &hex::encode(Sha256::digest(b"bad"))).is_err());
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn native_process_failure_releases_the_synthesis_slot() {
        let temporary = PrivateOutput::create().unwrap();
        let runtime = Runtime {
            root: temporary.0.clone(),
            binary: "/bin/false".into(),
            language: "de".into(),
            slot: Semaphore::new(1),
        };
        for _ in 0..2 {
            assert!(runtime
                .synthesize("test", "de", DEFAULT_VOICE)
                .await
                .unwrap_err()
                .to_string()
                .contains("native Phonon failed"));
            assert_eq!(runtime.slot.available_permits(), 1);
        }
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn cancelling_synthesis_kills_child_and_removes_private_output() {
        use std::os::unix::fs::PermissionsExt;
        let temporary = PrivateOutput::create().unwrap();
        let binary = temporary.0.join("fake-ptts");
        let trace = temporary.0.join("trace");
        std::fs::write(
            &binary,
            format!(
                "#!/bin/sh\nprintf '%s\\n%s\\n' \"$$\" \"$PWD\" > '{}'\nexec /bin/sleep 30\n",
                trace.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
        let runtime = Arc::new(Runtime {
            root: temporary.0.clone(),
            binary,
            language: "de".into(),
            slot: Semaphore::new(1),
        });
        let worker =
            tokio::spawn(async move { runtime.synthesize("test", "de", DEFAULT_VOICE).await });
        let trace_text = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if let Ok(text) = tokio::fs::read_to_string(&trace).await {
                    if text.lines().count() == 2 {
                        break text;
                    }
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let mut lines = trace_text.lines();
        let pid = lines.next().unwrap();
        let output_dir = PathBuf::from(lines.next().unwrap());
        worker.abort();
        assert!(worker.await.unwrap_err().is_cancelled());
        assert!(!output_dir.exists());
        tokio::time::timeout(Duration::from_secs(3), async {
            while Path::new(&format!("/proc/{pid}")).exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("native child must be reaped after cancellation");
    }
}
