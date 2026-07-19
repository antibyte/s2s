//! Local microphone capture + speaker playback via cpal.
//!
//! cpal streams are !Send on some hosts, so all stream lifetime stays on one OS thread.

use crate::audio::pcm::{f32_to_i16, i16_to_bytes_le, i16_to_f32, resample_f32};
use crate::messages::{AudioOut, Control, QueueItem};
use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("Host: {:?}", host.id());
    println!("\nInput devices:");
    for d in host.input_devices()? {
        let name = d.name().unwrap_or_else(|_| "<unknown>".into());
        let default = host
            .default_input_device()
            .and_then(|x| x.name().ok())
            .map(|n| n == name)
            .unwrap_or(false);
        println!("  {}{}", name, if default { "  (default)" } else { "" });
    }
    println!("\nOutput devices:");
    for d in host.output_devices()? {
        let name = d.name().unwrap_or_else(|_| "<unknown>".into());
        let default = host
            .default_output_device()
            .and_then(|x| x.name().ok())
            .map(|n| n == name)
            .unwrap_or(false);
        println!("  {}{}", name, if default { "  (default)" } else { "" });
    }
    Ok(())
}

fn find_device(devices: impl Iterator<Item = Device>, needle: &str, kind: &str) -> Result<Device> {
    if needle.is_empty() {
        return devices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no {kind} device found"));
    }
    let lower = needle.to_lowercase();
    for d in devices {
        if let Ok(name) = d.name() {
            if name.to_lowercase().contains(&lower) {
                return Ok(d);
            }
        }
    }
    Err(anyhow!("no {kind} device matching '{needle}'"))
}

/// Capture mic → `audio_tx` as raw i16 LE PCM chunks at `target_sr`.
/// Playback from `audio_rx` (AudioOut) on the matched output device.
///
/// Runs entirely on the calling OS thread (must not be a Tokio worker that expects Send futures
/// holding cpal streams). Prefer calling via `std::thread::spawn`.
pub fn run_local_io(
    target_sr: u32,
    input_needle: String,
    output_needle: String,
    audio_tx: mpsc::Sender<QueueItem<Vec<u8>>>,
    mut audio_rx: mpsc::Receiver<AudioOut>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let host = cpal::default_host();

    let input_dev = if input_needle.is_empty() {
        host.default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?
    } else {
        find_device(host.input_devices()?, &input_needle, "input")?
    };
    let output_dev = if output_needle.is_empty() {
        host.default_output_device()
            .ok_or_else(|| anyhow!("no default output device"))?
    } else {
        find_device(host.output_devices()?, &output_needle, "output")?
    };

    info!("Input device:  {}", input_dev.name()?);
    info!("Output device: {}", output_dev.name()?);

    // ── Capture ──────────────────────────────────────────────────────
    let in_config = input_dev
        .default_input_config()
        .context("default input config")?;
    let in_sr = in_config.sample_rate().0;
    let in_channels = in_config.channels() as usize;
    let in_format = in_config.sample_format();
    let in_stream_config: StreamConfig = in_config.clone().into();

    info!(
        "Capture: {} Hz, {} ch, {:?}",
        in_sr, in_channels, in_format
    );

    // Bridge cpal callback → OS thread → tokio mpsc via blocking_send.
    let (raw_tx, raw_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(32);
    let stop_cap = stop.clone();
    let audio_tx_cap = audio_tx.clone();

    let capture_thread = thread::spawn(move || {
        while !stop_cap.load(Ordering::Relaxed) {
            match raw_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(mono_native) => {
                    let resampled = if in_sr == target_sr {
                        mono_native
                    } else {
                        match resample_f32(&mono_native, in_sr, target_sr) {
                            Ok(v) => v,
                            Err(e) => {
                                warn!("resample capture failed: {e}");
                                continue;
                            }
                        }
                    };
                    let pcm = i16_to_bytes_le(&f32_to_i16(&resampled));
                    if audio_tx_cap.blocking_send(QueueItem::Data(pcm)).is_err() {
                        break;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        let _ = audio_tx_cap.blocking_send(QueueItem::Control(Control::PipelineEnd));
    });

    let stream_in = match in_format {
        SampleFormat::F32 => build_input_stream_f32(&input_dev, &in_stream_config, in_channels, raw_tx)?,
        SampleFormat::I16 => build_input_stream_i16(&input_dev, &in_stream_config, in_channels, raw_tx)?,
        other => return Err(anyhow!("unsupported input sample format: {other:?}")),
    };
    stream_in.play()?;

    // ── Playback ─────────────────────────────────────────────────────
    let out_config = output_dev
        .default_output_config()
        .context("default output config")?;
    let out_sr = out_config.sample_rate().0;
    let out_channels = out_config.channels() as usize;
    let out_format = out_config.sample_format();
    let out_stream_config: StreamConfig = out_config.clone().into();

    info!(
        "Playback: {} Hz, {} ch, {:?}",
        out_sr, out_channels, out_format
    );

    let play_buf = Arc::new(parking_lot::Mutex::new(Vec::<f32>::new()));
    let play_buf_cb = play_buf.clone();

    let stream_out = match out_format {
        SampleFormat::F32 => {
            let err_fn = |e| error!("output stream error: {e}");
            output_dev.build_output_stream(
                &out_stream_config,
                move |data: &mut [f32], _| {
                    let mut buf = play_buf_cb.lock();
                    let frames = data.len() / out_channels;
                    for frame in 0..frames {
                        let sample = if buf.is_empty() { 0.0 } else { buf.remove(0) };
                        for ch in 0..out_channels {
                            data[frame * out_channels + ch] = sample;
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let err_fn = |e| error!("output stream error: {e}");
            let play_buf_cb = play_buf.clone();
            output_dev.build_output_stream(
                &out_stream_config,
                move |data: &mut [i16], _| {
                    let mut buf = play_buf_cb.lock();
                    let frames = data.len() / out_channels;
                    for frame in 0..frames {
                        let sample = if buf.is_empty() { 0.0 } else { buf.remove(0) };
                        let s = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                        for ch in 0..out_channels {
                            data[frame * out_channels + ch] = s;
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        other => return Err(anyhow!("unsupported output sample format: {other:?}")),
    };
    stream_out.play()?;

    // Drain TTS audio on this thread via blocking_recv.
    while !stop.load(Ordering::Relaxed) {
        match audio_rx.try_recv() {
            Ok(chunk) => {
                if chunk.response_done && chunk.pcm_i16.is_empty() {
                    continue;
                }
                let f32s = i16_to_f32(&chunk.pcm_i16);
                let resampled = if chunk.sample_rate == out_sr {
                    f32s
                } else {
                    match resample_f32(&f32s, chunk.sample_rate, out_sr) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("resample playback failed: {e}");
                            continue;
                        }
                    }
                };
                play_buf.lock().extend(resampled);
            }
            Err(mpsc::error::TryRecvError::Empty) => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(mpsc::error::TryRecvError::Disconnected) => break,
        }
    }

    drop(stream_in);
    drop(stream_out);
    let _ = capture_thread.join();
    Ok(())
}

fn build_input_stream_f32(
    device: &Device,
    config: &StreamConfig,
    channels: usize,
    tx: std::sync::mpsc::SyncSender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let err_fn = |e| error!("input stream error: {e}");
    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _| {
            let mono = downmix_f32(data, channels);
            let _ = tx.try_send(mono);
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

fn build_input_stream_i16(
    device: &Device,
    config: &StreamConfig,
    channels: usize,
    tx: std::sync::mpsc::SyncSender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let err_fn = |e| error!("input stream error: {e}");
    let stream = device.build_input_stream(
        config,
        move |data: &[i16], _| {
            let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
            let mono = downmix_f32(&f, channels);
            let _ = tx.try_send(mono);
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

fn downmix_f32(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    let frames = data.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += data[i * channels + c];
        }
        mono.push(acc / channels as f32);
    }
    mono
}
