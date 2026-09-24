//! CLI / runtime configuration.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful voice assistant. Always reply in the same language the user speaks. If the user speaks German, reply only in German. Keep replies short and conversational (1-3 sentences).";
const DEFAULT_GERMAN_SYSTEM_PROMPT: &str = "Du bist ein deutschsprachiger Sprachassistent. Du verstehst Deutsch und beantwortest jede Frage direkt, freundlich und ausschließlich auf Deutsch. Deine Antworten sind kurz und natürlich.";

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum Mode {
    /// Microphone in, speakers out (cpal).
    Local,
    /// Raw 16 kHz mono i16 PCM over WebSocket.
    Websocket,
    /// Speech Lab controller API plus raw PCM WebSocket transport.
    Lab,
    /// OpenAI Realtime-compatible subset at /v1/realtime (audio append + audio delta).
    Realtime,
    /// OpenAI-compatible Supertonic HTTP sidecar.
    TtsServer,
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
    /// Prefer Supertonic if models present, else system/piper.
    #[default]
    Auto,
    /// In-process Supertonic 3 (ONNX Runtime, CPU — not GGML/Vulkan).
    Supertonic,
    /// HTTP TTS (OpenAI-style or qwentts/supertonic serve).
    Http,
    /// Local Piper binary (`piper` on PATH or --piper_bin).
    Piper,
    /// Windows SAPI / `espeak-ng` fallback for bring-up without neural TTS.
    System,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum SupertonicProvider {
    /// ONNX Runtime CPU execution provider.
    #[default]
    Cpu,
    /// ONNX Runtime WebGPU execution provider using Dawn's Vulkan backend.
    WebgpuVulkan,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum SttApi {
    #[default]
    Whisper,
    Openai,
}

/// Preferred accelerator. `auto` probes the host/container and sets `GGML_BACKEND`.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum GpuPreference {
    /// Smart detect (honors `GGML_BACKEND` / `S2S_GPU`):
    /// NVIDIA → CUDA; Intel Arc → SYCL if available else Vulkan (experimental);
    /// AMD → Vulkan; else CPU.
    #[default]
    Auto,
    /// Force Vulkan path (`GGML_BACKEND=Vulkan0` when available).
    Vulkan,
    /// Force CUDA path (`GGML_BACKEND=CUDA0` when available).
    Cuda,
    /// Force oneAPI SYCL path (`GGML_BACKEND=SYCL0` when available).
    /// Requires a SYCL-built GGML binary (qwentts/llama/whisper) and oneAPI/Level Zero.
    Sycl,
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
    /// Lower = more sensitive (more false starts). Raise if short noise triggers STT.
    #[arg(long, default_value_t = 0.62)]
    pub thresh: f32,

    /// Minimum continuous speech before a turn can end (ms). Short clips → Whisper hallucinations.
    #[arg(long, default_value_t = 900)]
    pub min_speech_ms: u64,

    /// Silence hangover after speech before finalizing the segment (ms).
    #[arg(long, default_value_t = 700)]
    pub min_silence_ms: u64,

    #[arg(long, default_value_t = 30)]
    pub speech_pad_ms: u64,

    // ── STT (whisper.cpp server, ideally Vulkan-built) ───────────────
    /// whisper-server base URL (POST /inference).
    #[arg(long, default_value = "http://127.0.0.1:8082", env = "S2S_WHISPER_URL")]
    pub whisper_url: String,

    /// Multipart ASR API served at the active endpoint.
    #[arg(long, value_enum, default_value_t = SttApi::Whisper, env = "S2S_STT_API")]
    pub stt_api: SttApi,

    /// Model alias for OpenAI-compatible transcription requests.
    #[arg(long, default_value = "confucius4-r2t2", env = "S2S_STT_MODEL")]
    pub stt_model: String,

    /// STT language hint (`auto`, `en`, `de`, …). Also default fallback for TTS when
    /// `--tts-language` is `auto` and the turn has no detected language.
    /// Default `de` for the German lab (faster STT than `auto` language detect).
    #[arg(long, default_value = "de", env = "S2S_LANGUAGE")]
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

    /// System prompt for the LLM. Default asks for short spoken replies in the
    /// user's language (important: cloud models otherwise default to English).
    #[arg(
        long,
        default_value = DEFAULT_SYSTEM_PROMPT,
        env = "S2S_SYSTEM_PROMPT"
    )]
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

    /// Disable chain-of-thought on Ollama/OpenAI-compatible "thinking" models
    /// (sends `reasoning_effort=none`). Required for cloud models that return
    /// empty `content` and only fill `reasoning` (e.g. glm-*:cloud, qwen*:cloud).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, env = "S2S_LLM_NO_THINK")]
    pub llm_no_think: bool,

    /// Ollama `keep_alive` for local models (seconds as number string, duration
    /// like `5m`, or `0` / `-1` / empty). Default keeps weights warm between turns
    /// to avoid cold-load latency. Only applied when `llm_base_url` hits :11434.
    #[arg(long, default_value = "5m", env = "S2S_LLM_KEEP_ALIVE")]
    pub llm_keep_alive: String,

    // ── TTS ──────────────────────────────────────────────────────────
    /// TTS engine: auto | supertonic | http | piper | system
    #[arg(long, value_enum, default_value_t = TtsBackend::Auto, env = "S2S_TTS")]
    pub tts: TtsBackend,

    /// TTS language for engines that need it (Supertonic, many HTTP TTS APIs).
    /// `auto` = use per-turn STT language if set, else `--language`, else `en`.
    /// Supertonic also accepts `na` (language-agnostic). Examples: `de`, `en`, `fr`.
    #[arg(long, default_value = "auto", env = "S2S_TTS_LANGUAGE")]
    pub tts_language: String,

    /// HTTP TTS endpoint. Expected: POST JSON {text, language?} → WAV or raw PCM.
    #[arg(
        long,
        default_value = "http://127.0.0.1:8083/v1/audio/speech",
        env = "S2S_TTS_URL"
    )]
    pub tts_url: String,

    /// OpenAI-style `model` field for HTTP TTS (e.g. `tts`, `kokoro`, qwen alias).
    #[arg(long, default_value = "tts", env = "S2S_TTS_MODEL")]
    pub tts_model: String,

    /// Native sample rate assumed for raw PCM HTTP responses (Qwen/Kokoro = 24000).
    /// WAV responses always use the rate embedded in the file.
    #[arg(long, default_value_t = 24000, env = "S2S_TTS_NATIVE_SR")]
    pub tts_native_sample_rate: u32,

    /// Maximum Qwen codec frames per request (256 = about 20.5 seconds).
    #[arg(
        long,
        default_value_t = 256,
        env = "S2S_QWEN_MAX_NEW_TOKENS",
        value_parser = clap::value_parser!(u32).range(1..=256)
    )]
    pub tts_http_max_new_tokens: u32,

    /// Target PCM chunk duration emitted to clients while HTTP TTS is streaming.
    #[arg(long, default_value_t = 160, env = "S2S_TTS_STREAM_CHUNK_MS")]
    pub tts_stream_chunk_ms: u32,

    #[arg(long, default_value = "", env = "S2S_TTS_API_KEY")]
    pub tts_api_key: String,

    /// Piper executable path (when --tts piper).
    #[arg(long, default_value = "piper")]
    pub piper_bin: PathBuf,

    /// Piper voice model (.onnx). Optional unless `--tts piper`.
    #[arg(long, default_value = None)]
    pub piper_model: Option<PathBuf>,

    /// Directory containing Supertonic ONNX assets (`duration_predictor.onnx`, …).
    #[arg(long, default_value = None, env = "S2S_SUPERTONIC_MODEL_DIR")]
    pub supertonic_model_dir: Option<PathBuf>,

    /// Preset voice name (e.g. M1, F1) or path to style JSON.
    #[arg(long, default_value = "M1", env = "S2S_SUPERTONIC_VOICE")]
    pub supertonic_voice: String,

    /// Explicit path to a voice style JSON (overrides --supertonic-voice lookup).
    #[arg(long, default_value = None)]
    pub supertonic_voice_path: Option<PathBuf>,

    /// Denoising steps (quality vs speed; 5–12 typical, default 8).
    #[arg(long, default_value_t = 8)]
    pub supertonic_steps: usize,

    /// Speech speed factor (0.7–2.0).
    #[arg(long, default_value_t = 1.05)]
    pub supertonic_speed: f32,

    /// Intra-op threads for ONNX (0 = runtime default).
    #[arg(long, default_value_t = 0)]
    pub supertonic_threads: usize,

    /// ONNX execution provider for all four Supertonic sessions.
    #[arg(
        long,
        value_enum,
        default_value_t = SupertonicProvider::Cpu,
        env = "S2S_SUPERTONIC_PROVIDER"
    )]
    pub supertonic_provider: SupertonicProvider,

    /// TTS output sample rate after synthesis (Supertonic native 44100; web UI often 16000).
    #[arg(long, default_value_t = 16000)]
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

    /// Use the tested German prompt for the default German lab without changing
    /// an explicitly configured system prompt or other conversation languages.
    pub fn llm_system_prompt(&self, detected: Option<&str>) -> String {
        let language = detected
            .map(str::trim)
            .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("auto"))
            .unwrap_or(self.language.as_str());
        if self.system_prompt == DEFAULT_SYSTEM_PROMPT
            && matches!(
                language.to_ascii_lowercase().as_str(),
                "de" | "deu" | "ger" | "deutsch" | "german"
            )
        {
            return DEFAULT_GERMAN_SYSTEM_PROMPT.to_string();
        }
        self.system_prompt.clone()
    }

    /// Resolve language code for TTS engines that require one (Supertonic, HTTP, …).
    ///
    /// Priority:
    /// 1. `--tts-language` if not `auto` / empty
    /// 2. per-turn STT `detected` (if provided)
    /// 3. `--language` if not `auto`
    /// 4. `en`
    ///
    /// Returns a lowercase code; use [`Self::qwen_tts_language`] when talking to
    /// qwentts (needs full names like `german`, not ISO `de`).
    pub fn resolve_tts_language(&self, detected: Option<&str>) -> String {
        let explicit = self.tts_language.trim();
        if !explicit.is_empty() && !explicit.eq_ignore_ascii_case("auto") {
            return explicit.to_ascii_lowercase();
        }
        if let Some(d) = detected {
            let d = d.trim();
            if !d.is_empty() && !d.eq_ignore_ascii_case("auto") {
                return d.to_ascii_lowercase();
            }
        }
        let stt = self.language.trim();
        if !stt.is_empty() && !stt.eq_ignore_ascii_case("auto") {
            return stt.to_ascii_lowercase();
        }
        "en".into()
    }

    /// Map ISO / short codes to qwentts.cpp language labels stored in the GGUF
    /// (`chinese`, `english`, `german`, …). Unknown values pass through lowercased.
    pub fn qwen_tts_language(code: &str) -> String {
        let c = code.trim().to_ascii_lowercase();
        match c.as_str() {
            "de" | "deu" | "ger" | "deutsch" => "german".into(),
            "en" | "eng" => "english".into(),
            "zh" | "cn" | "cmn" | "zh-cn" | "zh-tw" => "chinese".into(),
            "fr" | "fra" | "fre" => "french".into(),
            "es" | "spa" => "spanish".into(),
            "it" | "ita" => "italian".into(),
            "pt" | "por" => "portuguese".into(),
            "ja" | "jpn" => "japanese".into(),
            "ko" | "kor" => "korean".into(),
            "ru" | "rus" => "russian".into(),
            "auto" | "" => "auto".into(),
            other => other.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supertonic_provider_defaults_to_cpu_and_accepts_webgpu_vulkan() {
        let cpu = Config::try_parse_from(["s2s-vulkan"]).unwrap();
        assert_eq!(cpu.supertonic_provider, SupertonicProvider::Cpu);

        let vulkan =
            Config::try_parse_from(["s2s-vulkan", "--supertonic-provider", "webgpu-vulkan"])
                .unwrap();
        assert_eq!(vulkan.supertonic_provider, SupertonicProvider::WebgpuVulkan);
    }

    #[test]
    fn german_lab_prompt_preserves_other_languages_and_custom_prompts() {
        let default = Config::try_parse_from(["s2s-vulkan"]).unwrap();
        assert_eq!(
            default.llm_system_prompt(Some("de")),
            DEFAULT_GERMAN_SYSTEM_PROMPT
        );
        assert_eq!(default.llm_system_prompt(Some("en")), DEFAULT_SYSTEM_PROMPT);

        let english = Config::try_parse_from(["s2s-vulkan", "--language", "en"]).unwrap();
        assert_eq!(english.llm_system_prompt(None), DEFAULT_SYSTEM_PROMPT);

        let custom = Config::try_parse_from([
            "s2s-vulkan",
            "--system-prompt",
            "Answer in your configured language.",
        ])
        .unwrap();
        assert_eq!(custom.llm_system_prompt(Some("de")), custom.system_prompt);
    }
}
