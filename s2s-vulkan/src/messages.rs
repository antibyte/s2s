//! Pipeline messages — typed items flowing between VAD → STT → LLM → TTS.
//! Mirrors huggingface/speech-to-speech `pipeline/messages.py` (simplified).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static TURN_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct TurnId {
    pub id: String,
    #[allow(dead_code)]
    pub revision: u32,
    #[allow(dead_code)]
    lease: Option<std::sync::Arc<crate::runtime::TurnLease>>,
}

impl TurnId {
    pub fn next() -> Self {
        let n = TURN_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        Self {
            id: format!("turn_{n}"),
            revision: 0,
            lease: None,
        }
    }

    pub fn attach_lease(&mut self, lease: std::sync::Arc<crate::runtime::TurnLease>) {
        self.lease = Some(lease);
    }
}

/// PCM audio segment from VAD (f32 mono, typically 16 kHz).
#[derive(Debug, Clone)]
pub struct VadAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub mode: VadMode,
    pub turn: Option<TurnId>,
    #[allow(dead_code)]
    pub created_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadMode {
    #[allow(dead_code)]
    Progressive,
    Final,
}

/// Final (or partial) transcription from STT.
#[derive(Debug, Clone)]
pub struct Transcription {
    pub text: String,
    pub language: Option<String>,
    pub turn: Option<TurnId>,
    pub partial: bool,
    pub speech_end_at: Instant,
}

/// One streamed sentence/chunk from the LLM, ready for TTS.
#[derive(Debug, Clone)]
pub struct LlmChunk {
    pub text: String,
    pub language: Option<String>,
    pub turn: Option<TurnId>,
    pub is_final: bool,
    pub speech_end_at: Instant,
}

/// Synthesized PCM audio for playback / client (i16 mono).
#[derive(Debug, Clone)]
pub struct AudioOut {
    pub pcm_i16: Vec<i16>,
    pub sample_rate: u32,
    #[allow(dead_code)]
    pub turn: Option<TurnId>,
    pub response_done: bool,
}

/// Side-channel UI/debug events (JSON text frames on the lab WebSocket).
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PipelineEvent {
    #[allow(dead_code)]
    SpeechStarted {
        #[serde(skip_serializing_if = "Option::is_none")]
        turn: Option<String>,
    },
    #[allow(dead_code)]
    SpeechStopped {
        #[serde(skip_serializing_if = "Option::is_none")]
        turn: Option<String>,
        duration_ms: u64,
    },
    PartialTranscript {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        turn: Option<String>,
    },
    FinalTranscript {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        turn: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        language: Option<String>,
    },
    LlmChunk {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        turn: Option<String>,
    },
    LlmFull {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        turn: Option<String>,
    },
    ResponseDone {
        #[serde(skip_serializing_if = "Option::is_none")]
        turn: Option<String>,
    },
    Metrics {
        stage: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        turn: Option<String>,
        values: serde_json::Value,
    },
    /// Current lab stack after connect / hot-swap.
    Stack {
        asr: String,
        tts: String,
        llm: String,
        whisper_url: String,
        llm_base_url: String,
        model_name: String,
        tts_backend: String,
        tts_url: String,
        voice: String,
        language: String,
        ok: bool,
        message: String,
    },
    Error {
        stage: String,
        message: String,
    },
}

impl PipelineEvent {
    pub fn turn_id(t: &Option<TurnId>) -> Option<String> {
        t.as_ref().map(|x| x.id.clone())
    }
}

/// Control / lifecycle messages on any queue.
#[derive(Debug, Clone)]
pub enum Control {
    /// Soft reset of per-session state (keep handlers alive).
    SessionEnd,
    /// Hard stop — drain and exit handler task.
    PipelineEnd,
}

#[derive(Debug, Clone)]
pub enum QueueItem<T> {
    Data(T),
    Control(Control),
}

impl<T> QueueItem<T> {
    pub fn end() -> Self {
        Self::Control(Control::PipelineEnd)
    }
}
