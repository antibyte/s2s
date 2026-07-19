//! Voice Activity Detection — energy-based (default), Silero-compatible API shape.

use crate::audio::pcm::{bytes_to_i16_le, i16_to_f32, rms};
use crate::config::{Config, VadBackend};
use crate::messages::{Control, QueueItem, TurnId, VadAudio, VadMode};
use tokio::sync::mpsc;
use tracing::{debug, info};

/// Frame size for energy VAD analysis (~32 ms at 16 kHz).
const FRAME_MS: u64 = 32;

pub async fn run_vad(
    cfg: Config,
    mut audio_in: mpsc::Receiver<QueueItem<Vec<u8>>>,
    vad_out: mpsc::Sender<QueueItem<VadAudio>>,
    should_listen: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    info!(
        "VAD handler started (backend={:?}, thresh={}, min_speech={}ms, min_silence={}ms)",
        cfg.vad, cfg.thresh, cfg.min_speech_ms, cfg.min_silence_ms
    );

    let mut state = EnergyVad::new(&cfg);

    while let Some(item) = audio_in.recv().await {
        match item {
            QueueItem::Control(Control::PipelineEnd) => {
                let _ = vad_out.send(QueueItem::end()).await;
                break;
            }
            QueueItem::Control(Control::SessionEnd) => {
                state.reset();
                should_listen.store(true, std::sync::atomic::Ordering::Relaxed);
                continue;
            }
            QueueItem::Data(pcm_bytes) => {
                if !should_listen.load(std::sync::atomic::Ordering::Relaxed) {
                    continue;
                }
                let i16s = bytes_to_i16_le(&pcm_bytes);
                let f32s = i16_to_f32(&i16s);
                if let Some(seg) = state.push(&f32s, cfg.vad) {
                    info!(
                        "VAD: speech segment {:.0} ms (turn={})",
                        seg.samples.len() as f32 / seg.sample_rate as f32 * 1000.0,
                        seg.turn.as_ref().map(|t| t.id.as_str()).unwrap_or("-")
                    );
                    // Stop listening until TTS finishes (turn-based, like HF normal mode).
                    should_listen.store(false, std::sync::atomic::Ordering::Relaxed);
                    if vad_out.send(QueueItem::Data(seg)).await.is_err() {
                        break;
                    }
                }
            }
        }
    }
    debug!("VAD handler stopped");
}

struct EnergyVad {
    sample_rate: u32,
    thresh: f32,
    min_speech_samples: usize,
    min_silence_samples: usize,
    pad_samples: usize,
    frame_samples: usize,

    triggered: bool,
    speech_buf: Vec<f32>,
    silence_run: usize,
    speech_run: usize,
    pre_roll: Vec<f32>,
}

impl EnergyVad {
    fn new(cfg: &Config) -> Self {
        let sr = cfg.sample_rate;
        Self {
            sample_rate: sr,
            thresh: cfg.thresh * 0.05, // map 0..1 UI thresh to RMS-ish scale
            min_speech_samples: ms_to_samples(cfg.min_speech_ms, sr),
            min_silence_samples: ms_to_samples(cfg.min_silence_ms, sr),
            pad_samples: ms_to_samples(cfg.speech_pad_ms, sr),
            frame_samples: ms_to_samples(FRAME_MS, sr).max(1),
            triggered: false,
            speech_buf: Vec::new(),
            silence_run: 0,
            speech_run: 0,
            pre_roll: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.triggered = false;
        self.speech_buf.clear();
        self.silence_run = 0;
        self.speech_run = 0;
        self.pre_roll.clear();
    }

    fn push(&mut self, chunk: &[f32], _backend: VadBackend) -> Option<VadAudio> {
        // Process frame-wise for stable energy estimates.
        let mut offset = 0;
        let mut finished: Option<VadAudio> = None;

        while offset < chunk.len() {
            let end = (offset + self.frame_samples).min(chunk.len());
            let frame = &chunk[offset..end];
            offset = end;

            let level = rms(frame);
            let is_speech = level >= self.thresh;

            if !self.triggered {
                // Keep a short pre-roll pad.
                self.pre_roll.extend_from_slice(frame);
                if self.pre_roll.len() > self.pad_samples * 3 {
                    let excess = self.pre_roll.len() - self.pad_samples * 3;
                    self.pre_roll.drain(..excess);
                }

                if is_speech {
                    self.speech_run += frame.len();
                    if self.speech_run >= self.min_speech_samples / 4 {
                        self.triggered = true;
                        self.speech_buf.clear();
                        self.speech_buf.extend_from_slice(&self.pre_roll);
                        self.speech_buf.extend_from_slice(frame);
                        self.silence_run = 0;
                        debug!("VAD triggered (rms={level:.4})");
                    }
                } else {
                    self.speech_run = 0;
                }
            } else {
                self.speech_buf.extend_from_slice(frame);
                if is_speech {
                    self.silence_run = 0;
                    self.speech_run += frame.len();
                } else {
                    self.silence_run += frame.len();
                    if self.silence_run >= self.min_silence_samples
                        && self.speech_buf.len() >= self.min_speech_samples
                    {
                        // Trim trailing silence but keep pad.
                        let trim = self.silence_run.saturating_sub(self.pad_samples);
                        if trim > 0 && trim < self.speech_buf.len() {
                            self.speech_buf.truncate(self.speech_buf.len() - trim);
                        }
                        let samples = std::mem::take(&mut self.speech_buf);
                        self.triggered = false;
                        self.silence_run = 0;
                        self.speech_run = 0;
                        self.pre_roll.clear();
                        finished = Some(VadAudio {
                            samples,
                            sample_rate: self.sample_rate,
                            mode: VadMode::Final,
                            turn: Some(TurnId::next()),
                            created_at: std::time::Instant::now(),
                        });
                        break;
                    }
                }
            }
        }

        finished
    }
}

fn ms_to_samples(ms: u64, sr: u32) -> usize {
    (ms as u64 * sr as u64 / 1000) as usize
}
