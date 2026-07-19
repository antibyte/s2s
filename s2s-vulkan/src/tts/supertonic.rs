//! In-process Supertonic 3 TTS via ONNX Runtime (CPU).
//!
//! Not GGML/Vulkan — same audio contract as Piper/System/HTTP:
//! returns mono PCM `Vec<i16>` + sample rate.
//!
//! Models: Hugging Face `Supertone/supertonic-3` (onnx/ + voice_styles/).
//! Inference runs on a blocking thread pool (`spawn_blocking`).

use crate::audio::pcm::{f32_to_i16, resample_f32};
use crate::config::Config;
use crate::tts::supertonic_helper::{
    load_text_to_speech, load_voice_style, Style, TextToSpeech,
};
use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

/// Native sample rate of Supertonic open-weight assets.
#[allow(dead_code)]
pub const NATIVE_SAMPLE_RATE: u32 = 44100;

#[derive(Clone)]
pub struct SupertonicEngine {
    inner: Arc<Mutex<TextToSpeech>>,
    style: Arc<Style>,
    steps: usize,
    speed: f32,
    default_lang: String,
    target_sr: u32,
}

impl SupertonicEngine {
    pub fn load(cfg: &Config) -> Result<Self> {
        let onnx_dir = resolve_onnx_dir(cfg)?;
        let voice_path = resolve_voice_path(cfg, &onnx_dir)?;

        info!(
            "Supertonic: loading ONNX from {} (CPU), voice={}",
            onnx_dir.display(),
            voice_path.display()
        );

        let tts = load_text_to_speech(onnx_dir.to_str().unwrap_or("."), false)
            .with_context(|| format!("load Supertonic ONNX from {}", onnx_dir.display()))?;

        let style = load_voice_style(&[voice_path.to_string_lossy().into_owned()], true)
            .with_context(|| format!("load voice style {}", voice_path.display()))?;

        let default_lang = if cfg.language == "auto" {
            "en".into()
        } else {
            cfg.language.clone()
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(tts)),
            style: Arc::new(style),
            steps: cfg.supertonic_steps.max(1),
            speed: cfg.supertonic_speed.clamp(0.7, 2.0),
            default_lang,
            target_sr: cfg.tts_sample_rate,
        })
    }

    /// Blocking synthesis (call from `spawn_blocking`).
    pub fn synthesize_blocking(&self, text: &str, language: Option<&str>) -> Result<(Vec<i16>, u32)> {
        let lang = language
            .filter(|l| !l.is_empty() && *l != "auto")
            .unwrap_or(self.default_lang.as_str());

        let mut tts = self.inner.lock();
        let (wav_f32, _dur) = tts
            .call(
                text,
                lang,
                self.style.as_ref(),
                self.steps,
                self.speed,
                0.25,
            )
            .context("supertonic synthesize")?;

        let native_sr = tts.sample_rate as u32;
        let pcm = if self.target_sr != 0 && self.target_sr != native_sr {
            let resampled = resample_f32(&wav_f32, native_sr, self.target_sr)?;
            (f32_to_i16(&resampled), self.target_sr)
        } else {
            (f32_to_i16(&wav_f32), native_sr)
        };
        Ok(pcm)
    }
}

pub fn is_available(cfg: &Config) -> bool {
    resolve_onnx_dir(cfg).is_ok()
}

fn resolve_onnx_dir(cfg: &Config) -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = [
        cfg.supertonic_model_dir.clone(),
        Some(PathBuf::from("models/supertonic/onnx")),
        Some(PathBuf::from("assets/onnx")),
        std::env::var_os("S2S_SUPERTONIC_MODEL_DIR").map(PathBuf::from),
    ]
    .into_iter()
    .flatten()
    .collect();

    for dir in candidates {
        if onnx_dir_complete(&dir) {
            return Ok(dir);
        }
    }
    Err(anyhow!(
        "Supertonic ONNX models not found. Download with:\n  \
         python scripts/download_supertonic.py\n  \
         or set --supertonic-model-dir to the `onnx` folder of Supertone/supertonic-3"
    ))
}

fn onnx_dir_complete(dir: &Path) -> bool {
    dir.is_dir()
        && dir.join("duration_predictor.onnx").is_file()
        && dir.join("text_encoder.onnx").is_file()
        && dir.join("vector_estimator.onnx").is_file()
        && dir.join("vocoder.onnx").is_file()
        && dir.join("unicode_indexer.json").is_file()
}

fn resolve_voice_path(cfg: &Config, onnx_dir: &Path) -> Result<PathBuf> {
    let name = cfg.supertonic_voice.trim();
    let file = if name.ends_with(".json") {
        name.to_string()
    } else {
        format!("{name}.json")
    };

    let candidates = [
        cfg.supertonic_voice_path.clone(),
        Some(onnx_dir.join("../voice_styles").join(&file)),
        Some(onnx_dir.parent().unwrap_or(onnx_dir).join("voice_styles").join(&file)),
        Some(PathBuf::from("models/supertonic/voice_styles").join(&file)),
        Some(PathBuf::from("assets/voice_styles").join(&file)),
    ];

    for p in candidates.into_iter().flatten() {
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(anyhow!(
        "Supertonic voice style '{file}' not found (tried models/supertonic/voice_styles/)"
    ))
}
