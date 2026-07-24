/**
 * AudioWorklet: mic → 16 kHz mono i16 frames (40 ms) for s2s WebSocket PCM.
 * Runs off the main thread (unlike deprecated ScriptProcessor).
 */
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || 16000;
    this.frameSamples = opts.frameSamples || 640; // 40 ms @ 16 kHz
    this._leftover = new Float32Array(0);
    this._armed = true;
    this.port.onmessage = (ev) => {
      const msg = ev.data || {};
      if (msg.type === "arm") this._armed = !!msg.value;
    };
  }

  process(inputs) {
    if (!this._armed) return true;
    const input = inputs[0] && inputs[0][0];
    if (!input || input.length === 0) return true;

    // Level (RMS) for UI
    let sum = 0;
    for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
    const level = Math.sqrt(sum / input.length);

    // Downsample native context rate → 16 kHz (point sample; cheap + stable)
    const fromRate = sampleRate;
    let down;
    if (fromRate === this.targetRate) {
      down = input;
    } else {
      const ratio = fromRate / this.targetRate;
      const newLen = Math.floor(input.length / ratio);
      down = new Float32Array(newLen);
      for (let i = 0; i < newLen; i++) {
        down[i] = input[Math.floor(i * ratio)];
      }
    }

    const merged = new Float32Array(this._leftover.length + down.length);
    merged.set(this._leftover);
    merged.set(down, this._leftover.length);

    let offset = 0;
    const frames = [];
    while (offset + this.frameSamples <= merged.length) {
      const slice = merged.subarray(offset, offset + this.frameSamples);
      const pcm = new Int16Array(this.frameSamples);
      for (let i = 0; i < this.frameSamples; i++) {
        const s = Math.max(-1, Math.min(1, slice[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      frames.push(pcm.buffer);
      offset += this.frameSamples;
    }
    this._leftover = merged.subarray(offset);

    if (frames.length) {
      this.port.postMessage({ type: "frames", frames, level }, frames);
    } else if (level > 0.001) {
      this.port.postMessage({ type: "level", level });
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
