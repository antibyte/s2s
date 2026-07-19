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
}

impl TurnId {
    pub fn next() -> Self {
        let n = TURN_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        Self {
            id: format!("turn_{n}"),
            revision: 0,
        }
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
}

/// One streamed sentence/chunk from the LLM, ready for TTS.
#[derive(Debug, Clone)]
pub struct LlmChunk {
    pub text: String,
    pub language: Option<String>,
    pub turn: Option<TurnId>,
    pub is_final: bool,
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

/// Side-channel UI/debug events (reserved for realtime captions / UI).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum PipelineEvent {
    SpeechStarted { turn: Option<TurnId> },
    SpeechStopped { turn: Option<TurnId>, duration_ms: u64 },
    PartialTranscript { text: String, turn: Option<TurnId> },
    FinalTranscript { text: String, turn: Option<TurnId> },
    LlmToken { text: String },
    ResponseDone { turn: Option<TurnId> },
    Error { stage: String, message: String },
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
