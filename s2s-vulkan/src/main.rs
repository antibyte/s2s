//! s2s-vulkan — HuggingFace speech-to-speech style pipeline in Rust,
//! with heavy inference on Vulkan-capable GGML servers.

mod audio;
mod config;
mod gpu;
mod io;
mod llm;
mod messages;
mod pipeline;
mod stt;
mod tts;
mod vad;

use crate::audio::list_devices;
use crate::config::{Config, Mode, TtsBackend};
use crate::pipeline::{check_backends, spawn_pipeline, PipelineHandles};
use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::parse_args();

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cfg.log_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    // GPU detection early so env is set before any backend talk / child processes.
    let gpu_report = gpu::detect(cfg.gpu);
    if cfg.list_gpus {
        println!("{}", gpu::report_json(&gpu_report));
        return Ok(());
    }
    gpu_report.apply_env();
    gpu_report.log();

    if cfg.list_devices {
        list_devices()?;
        return Ok(());
    }

    print_banner(&cfg, &gpu_report);

    if !cfg.skip_health {
        check_backends(&cfg).await;
    }

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            info!("Ctrl-C received");
            stop.store(true, Ordering::Relaxed);
        });
    }

    match cfg.mode {
        Mode::Local => run_local(cfg, stop).await?,
        Mode::Websocket => io::run_websocket_server(cfg).await?,
        Mode::Realtime => io::run_realtime_server(cfg).await?,
    }

    Ok(())
}

async fn run_local(cfg: Config, stop: Arc<AtomicBool>) -> Result<()> {
    info!("Mode: local (mic → pipeline → speakers). Speak after backends are ready.");
    if matches!(cfg.tts, TtsBackend::Http) {
        info!("TTS HTTP endpoint: {}", cfg.tts_url);
    }
    if gpu::running_in_container() {
        tracing::warn!(
            "Local audio mode inside a container usually has no mic/speakers. \
             Prefer --mode websocket or --mode realtime with host port mapping."
        );
    }

    let PipelineHandles {
        audio_in_tx,
        audio_out_rx,
        should_listen: _,
        join,
    } = spawn_pipeline(cfg.clone());

    let sample_rate = cfg.sample_rate;
    let input_device = cfg.input_device.clone();
    let output_device = cfg.output_device.clone();
    let io_stop = stop.clone();

    // cpal streams are !Send — keep them on a dedicated OS thread.
    let io_thread = std::thread::spawn(move || {
        if let Err(e) = audio::run_local_io(
            sample_rate,
            input_device,
            output_device,
            audio_in_tx,
            audio_out_rx,
            io_stop,
        ) {
            tracing::error!("local audio I/O error: {e:#}");
        }
    });

    while !stop.load(Ordering::Relaxed) {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    info!("Shutting down…");
    stop.store(true, Ordering::Relaxed);
    let _ = io_thread.join();
    for h in join {
        h.abort();
    }
    Ok(())
}

fn print_banner(cfg: &Config, gpu: &gpu::GpuReport) {
    info!("╔══════════════════════════════════════════════════╗");
    info!("║  s2s-vulkan  —  VAD → STT → LLM → TTS            ║");
    info!("╚══════════════════════════════════════════════════╝");
    info!(
        "mode={}  sample_rate={}  host={}:{}",
        format!("{:?}", cfg.mode).to_lowercase(),
        cfg.sample_rate,
        cfg.host,
        cfg.port
    );
    info!(
        "GPU  {}  kind={}  GGML_BACKEND={}",
        gpu.selected.name,
        gpu.selected.kind.as_str(),
        gpu.selected.ggml_backend
    );
    info!("STT  whisper-server  {}", cfg.whisper_url);
    info!("LLM  {}  model={}", cfg.llm_base_url, cfg.model_name);
    info!(
        "TTS  {:?}  {}",
        cfg.tts,
        match cfg.tts {
            TtsBackend::Auto => "auto (supertonic→system)",
            TtsBackend::Supertonic => "supertonic-onnx-cpu",
            TtsBackend::Http => cfg.tts_url.as_str(),
            TtsBackend::Piper => "piper",
            TtsBackend::System => "system",
        }
    );
}
