use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::io::Cursor;

pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| s as f32 / 32768.0).collect()
}

pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| {
            let c = s.clamp(-1.0, 1.0);
            (c * 32767.0) as i16
        })
        .collect()
}

pub fn bytes_to_i16_le(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

pub fn i16_to_bytes_le(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Encode mono f32 samples as 16-bit PCM WAV in memory.
pub fn encode_wav_f32(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)?;
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(v)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// Decode a WAV blob to mono f32 + sample rate.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples_f32: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            let raw: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
            i16_to_f32(&raw?)
        }
        (SampleFormat::Int, 32) => {
            let raw: Result<Vec<i32>, _> = reader.samples::<i32>().collect();
            raw?.into_iter()
                .map(|s| s as f32 / i32::MAX as f32)
                .collect()
        }
        (SampleFormat::Float, 32) => {
            let raw: Result<Vec<f32>, _> = reader.samples::<f32>().collect();
            raw?
        }
        _ => {
            return Err(anyhow!(
                "unsupported WAV format: {:?} {}-bit",
                spec.sample_format,
                spec.bits_per_sample
            ))
        }
    };

    if channels == 1 {
        return Ok((samples_f32, sr));
    }

    // Downmix to mono.
    let frames = samples_f32.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += samples_f32[i * channels + c];
        }
        mono.push(acc / channels as f32);
    }
    Ok((mono, sr))
}

pub fn resample_f32(input: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    if from_sr == to_sr || input.is_empty() {
        return Ok(input.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 64,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        to_sr as f64 / from_sr as f64,
        2.0,
        params,
        input.len(),
        1,
    )?;

    let waves_in = vec![input.to_vec()];
    let waves_out = resampler.process(&waves_in, None)?;
    Ok(waves_out.into_iter().next().unwrap_or_default())
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}
