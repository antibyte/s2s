//! CLI / runtime configuration.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum Mode {
    /// Microphone in, speakers out (cpal).
    Local,
    /// Raw 16 kHz mono i16 PCM over WebSocket.
    Websocket,
    /// OpenAI Realtime-compatible subset at /v1/realtime (audio append + audio delta).
    Realtime,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum VadBackend {
    /// Lightweight energy + hangover VAD (no model download).
    #[default]
    Energy,
    /// Silero VAD v5 ONNX via `ort` — not linked by default; falls back to energy
    /// unless you pass a model path and build with the optional feature later.
    Silero,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum TtsBackend {
    /// HTTP TTS server (e.g. a thin wrapper around qwentts.cpp / piper).
    #[default]
    Http,
    /// Local Piper binary (`piper` on PATH or --piper_bin).
    Piper,
    /// Windows SAPI / `espeak-ng` fallback for bring-up without neural TTS.
    System,
}

/// Preferred accelerator. `auto` probes the host/container and sets `GGML_BACKEND`.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum GpuPreference {
    /// Detect Vulkan → CUDA → CPU (also honors `GGML_BACKEND` / `S2S_GPU`).
    #[default]
    Auto,
    /// Force Vulkan path (`GGML_BACKEND=Vulkan0` when available).
    Vulkan,
    /// Force CUDA path (`GGML_BACKEND=CUDA0` when available).
    Cuda,
    /// Force CPU.
    Cpu,
}

#[derive(Debug, Parser, Clone)]
#[command(
    name = "s2s-vulkan",
    about = "Voice agent: VAD → STT → LLM → TTS with Vulkan-capable GGML backends",
    long_about = "Rust reimplementation of the huggingface/speech-to-speech pipeline shape.\n\
Heavy inference is delegated to external GGML servers (whisper.cpp, llama.cpp) and TTS\n\
backends that can use Vulkan. See README for Vulkan build instructions."
)]
pub struct Config {
    /// Run mode.
    #[arg(long, value_enum, default_value_t = Mode::Local)]
    pub mode: Mode,

    /// Log level (error, warn, info, debug, trace).
    #[arg(long, default_value = "info", env = "RUST_LOG")]
    pub log_level: String,

    // ── VAD ──────────────────────────────────────────────────────────
    #[arg(long, value_enum, default_value_t = VadBackend::Energy)]
    pub vad: VadBackend,

    /// Speech probability / energy threshold (0..1 for energy VAD).
    #[arg(long, default_value_t = 0.55)]
    pub thresh: f32,

    #[arg(long, default_value_t = 384)]
    pub min_speech_ms: u64,

    #[arg(long, default_value_t = 400)]
    pub min_silence_ms: u64,

    #[arg(long, default_value_t = 30)]
    pub speech_pad_ms: u64,

    // ── STT (whisper.cpp server, ideally Vulkan-built) ───────────────
    /// whisper-server base URL (POST /inference).
    #[arg(long, default_value = "http://127.0.0.1:8082", env = "S2S_WHISPER_URL")]
    pub whisper_url: String,

    /// Whisper language (`auto`, `en`, `de`, …).
    #[arg(long, default_value = "auto")]
    pub language: String,

    /// whisper-server temperature.
    #[arg(long, default_value_t = 0.0)]
    pub stt_temperature: f32,

    // ── LLM (llama-server / OpenAI-compatible, ideally Vulkan) ───────
    #[arg(long, default_value = "http://127.0.0.1:8081/v1", env = "S2S_LLM_URL")]
    pub llm_base_url: String,

    #[arg(long, default_value = "", env = "S2S_LLM_API_KEY")]
    pub llm_api_key: String,

    #[arg(long, default_value = "local-model", env = "S2S_LLM_MODEL")]
    pub model_name: String,

    #[arg(long, default_value = "You are a helpful voice assistant. Keep replies concise and conversational.")]
    pub system_prompt: String,

    #[arg(long, default_value_t = 30)]
    pub chat_size: usize,

    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    #[arg(long, default_value_t = 256)]
    pub max_tokens: u32,

    /// Stream LLM tokens and speak sentence-by-sentence.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub llm_stream: bool,

    // ── TTS ──────────────────────────────────────────────────────────
    #[arg(long, value_enum, default_value_t = TtsBackend::Http)]
    pub tts: TtsBackend,

    /// HTTP TTS endpoint. Expected: POST JSON {text, language?} → WAV or raw PCM.
    #[arg(long, default_value = "http://127.0.0.1:8083/v1/audio/speech", env = "S2S_TTS_URL")]
    pub tts_url: String,

    #[arg(long, default_value = "")]
    pub tts_api_key: String,

    /// Piper executable path (when --tts piper).
    #[arg(long, default_value = "piper")]
    pub piper_bin: PathBuf,

    /// Piper voice model (.onnx). Optional unless `--tts piper`.
    #[arg(long, default_value = None)]
    pub piper_model: Option<PathBuf>,

    /// TTS output sample rate (used when synthesizing / resampling for playback).
    #[arg(long, default_value_t = 24000)]
    pub tts_sample_rate: u32,

    // ── Audio / IO ───────────────────────────────────────────────────
    #[arg(long, default_value_t = 16000)]
    pub sample_rate: u32,

    /// Input device name substring (empty = default).
    #[arg(long, default_value = "")]
    pub input_device: String,

    /// Output device name substring (empty = default).
    #[arg(long, default_value = "")]
    pub output_device: String,

    /// WebSocket / Realtime bind host.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// WebSocket / Realtime bind port.
    #[arg(long, default_value_t = 8765)]
    pub port: u16,

    /// List audio devices and exit.
    #[arg(long, default_value_t = false)]
    pub list_devices: bool,

    /// List detected GPUs (JSON) and exit.
    #[arg(long, default_value_t = false)]
    pub list_gpus: bool,

    /// Accelerator preference. In Docker this selects `GGML_BACKEND` for child backends.
    #[arg(long, value_enum, default_value_t = GpuPreference::Auto, env = "S2S_GPU")]
    pub gpu: GpuPreference,

    /// Skip waiting for backend health checks at startup.
    #[arg(long, default_value_t = false)]
    pub skip_health: bool,

    /// Bind to 0.0.0.0 automatically when running in a container (default: true).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, env = "S2S_AUTO_BIND")]
    pub auto_bind_container: bool,
}

impl Config {
    pub fn parse_args() -> Self {
        let mut cfg = Self::parse();
        cfg.apply_container_defaults();
        cfg
    }

    /// Docker-friendly defaults: listen on all interfaces so host port-maps work.
    pub fn apply_container_defaults(&mut self) {
        if !self.auto_bind_container {
            return;
        }
        if crate::gpu::running_in_container() && self.host == "127.0.0.1" {
            self.host = "0.0.0.0".into();
        }
        // Local mic mode is rarely available in containers — prefer websocket if still local
        // only when S2S_FORCE_LOCAL is unset. Leave mode alone; entrypoint sets --mode.
    }
}
