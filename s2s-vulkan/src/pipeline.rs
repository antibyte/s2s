//! Orchestrates the handler chain with tokio channels.
//!
//! ```text
//! audio_in  → VAD → STT → LLM → TTS → audio_out
//! ```

use crate::config::Config;
use crate::llm;
use crate::messages::{AudioOut, LlmChunk, PipelineEvent, QueueItem, Transcription, VadAudio};
use crate::runtime::{self, SharedRuntime};
use crate::stt;
use crate::tts;
use crate::vad;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::info;

pub struct PipelineHandles {
    pub audio_in_tx: mpsc::Sender<QueueItem<Vec<u8>>>,
    pub audio_out_rx: mpsc::Receiver<AudioOut>,
    /// UI events (transcripts, LLM text) for the lab WebSocket.
    pub event_rx: mpsc::Receiver<PipelineEvent>,
    pub should_listen: Arc<AtomicBool>,
    /// Hot-swappable stack (ASR / LLM / TTS).
    pub runtime: SharedRuntime,
    pub join: Vec<tokio::task::JoinHandle<()>>,
}

pub fn spawn_pipeline(cfg: Config) -> PipelineHandles {
    let runtime = runtime::runtime_from(cfg);
    spawn_pipeline_shared(runtime)
}

pub fn spawn_pipeline_shared(runtime: SharedRuntime) -> PipelineHandles {
    let turns = runtime
        .try_read()
        .expect("runtime must be readable while spawning a pipeline")
        .turns
        .clone();
    let (audio_in_tx, audio_in_rx) = mpsc::channel::<QueueItem<Vec<u8>>>(64);
    let (vad_tx, vad_rx) = mpsc::channel::<QueueItem<VadAudio>>(8);
    let (stt_tx, stt_rx) = mpsc::channel::<QueueItem<Transcription>>(8);
    let (llm_tx, llm_rx) = mpsc::channel::<QueueItem<LlmChunk>>(16);
    let (audio_out_tx, audio_out_rx) = mpsc::channel::<AudioOut>(32);
    let (event_tx, event_rx) = mpsc::channel::<PipelineEvent>(32);

    let should_listen = Arc::new(AtomicBool::new(true));

    let mut join = Vec::new();

    {
        let runtime = runtime.clone();
        let should_listen = should_listen.clone();
        let turns = turns.clone();
        join.push(tokio::spawn(async move {
            // VAD uses a snapshot at start (thresholds rarely hot-swapped).
            let cfg = runtime.read().await.cfg.clone();
            vad::run_vad(cfg, audio_in_rx, vad_tx, should_listen, turns).await;
        }));
    }
    {
        let runtime = runtime.clone();
        let should_listen = should_listen.clone();
        let event_tx = event_tx.clone();
        join.push(tokio::spawn(async move {
            stt::run_stt(runtime, vad_rx, stt_tx, should_listen, event_tx).await;
        }));
    }
    {
        let runtime = runtime.clone();
        let should_listen = should_listen.clone();
        let event_tx = event_tx.clone();
        join.push(tokio::spawn(async move {
            llm::run_llm(runtime, stt_rx, llm_tx, should_listen, event_tx).await;
        }));
    }
    {
        let runtime = runtime.clone();
        let should_listen = should_listen.clone();
        let event_tx = event_tx.clone();
        join.push(tokio::spawn(async move {
            tts::run_tts(runtime, llm_rx, audio_out_tx, should_listen, event_tx).await;
        }));
    }

    info!("Pipeline spawned: VAD → STT → LLM → TTS (hot-swap ready)");

    PipelineHandles {
        audio_in_tx,
        audio_out_rx,
        event_rx,
        should_listen,
        runtime,
        join,
    }
}

pub async fn check_backends(cfg: &Config) {
    info!("Checking backends…");
    let w = stt::health_check(&cfg.whisper_url).await;
    let l = llm::health_check(&cfg.llm_base_url).await;
    let t = tts::health_check(cfg).await;
    info!(
        "  whisper-server ({}): {}",
        cfg.whisper_url,
        if w { "OK" } else { "UNREACHABLE" }
    );
    info!(
        "  llm          ({}): {}",
        cfg.llm_base_url,
        if l { "OK" } else { "UNREACHABLE" }
    );
    info!(
        "  tts          ({:?}): {}",
        cfg.tts,
        if t { "OK" } else { "UNREACHABLE / optional" }
    );
    if !w {
        tracing::warn!(
            "whisper-server not reachable — start whisper.cpp with Vulkan, e.g.\n  \
             whisper-server -m models/ggml-small.bin --host 127.0.0.1 --port 8082"
        );
    }
    if !l {
        tracing::warn!(
            "LLM server not reachable — start llama-server with Vulkan, e.g.\n  \
             llama-server -m model.gguf -ngl 999 --port 8081"
        );
    }
}
