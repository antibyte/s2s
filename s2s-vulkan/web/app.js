/**
 * s2s lab — browser client for s2s-vulkan raw WebSocket PCM mode.
 * Mic → 16 kHz mono i16 LE binary frames → backend
 * Backend → binary PCM → AudioContext playback + reactive visuals
 */

const SAMPLE_RATE = 16000;
const FRAME_MS = 40;
const FRAME_SAMPLES = (SAMPLE_RATE * FRAME_MS) / 1000;

const $ = (id) => document.getElementById(id);

const els = {
  canvas: $("fx"),
  mic: $("mic"),
  micLabel: $("mic-label"),
  hint: $("hint"),
  wsUrl: $("ws-url"),
  talkMode: $("talk-mode"),
  btnConnect: $("btn-connect"),
  btnDisconnect: $("btn-disconnect"),
  pillConn: $("pill-conn"),
  pillMode: $("pill-mode"),
  barMic: $("bar-mic"),
  barOut: $("bar-out"),
  log: $("log"),
  fps: $("fps"),
};

// ── State ───────────────────────────────────────────────────────────
const state = {
  ws: null,
  connected: false,
  talking: false,
  speaking: false, // backend audio playing
  micLevel: 0,
  outLevel: 0,
  audioCtx: null,
  mediaStream: null,
  processor: null,
  source: null,
  playTime: 0,
  playNodes: 0,
  talkMode: "hold",
};

const params = new URLSearchParams(location.search);
function defaultWsUrl() {
  if (params.get("ws")) return params.get("ws");
  const saved = localStorage.getItem("s2s.ws");
  if (saved) return saved;
  // Served from the optional web container: same-origin nginx proxies /ws → s2s.
  if (location.port === "8088" || location.pathname === "/" && !location.port) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws`;
  }
  const host =
    location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "127.0.0.1"
      : location.hostname;
  return `ws://${host}:8765`;
}
const defaultWs = defaultWsUrl();

els.wsUrl.value = defaultWs;
els.talkMode.value = localStorage.getItem("s2s.talkMode") || "hold";
state.talkMode = els.talkMode.value;

function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  els.log.textContent = `${line}\n${els.log.textContent}`.slice(0, 4000);
  console.log(msg);
}

function setPill(el, text, dataState) {
  el.textContent = text;
  el.dataset.state = dataState;
}

function setCssLevels() {
  const l = Math.max(state.micLevel, state.outLevel * 0.9);
  document.documentElement.style.setProperty("--level", l.toFixed(3));
  document.documentElement.style.setProperty("--hotness", state.talking ? "1" : "0");
  els.barMic.style.width = `${Math.min(100, state.micLevel * 120)}%`;
  els.barOut.style.width = `${Math.min(100, state.outLevel * 120)}%`;
}

// ── Visual engine ───────────────────────────────────────────────────
const vis = {
  particles: [],
  rings: [],
  t: 0,
  last: performance.now(),
  frames: 0,
  fpsT: 0,
};

function initParticles() {
  const n = 120;
  vis.particles = Array.from({ length: n }, (_, i) => ({
    a: (i / n) * Math.PI * 2,
    r: 0.18 + Math.random() * 0.55,
    s: 0.15 + Math.random() * 0.9,
    size: 0.8 + Math.random() * 2.2,
    hue: Math.random() < 0.33 ? 190 : Math.random() < 0.5 ? 265 : 330,
  }));
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = window.innerWidth;
  const h = window.innerHeight;
  els.canvas.width = Math.floor(w * dpr);
  els.canvas.height = Math.floor(h * dpr);
  els.canvas.style.width = `${w}px`;
  els.canvas.style.height = `${h}px`;
  const ctx = els.canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

let ctx = resizeCanvas();
window.addEventListener("resize", () => {
  ctx = resizeCanvas();
});
initParticles();

function drawFrame(now) {
  const dt = Math.min(0.05, (now - vis.last) / 1000);
  vis.last = now;
  vis.t += dt;
  vis.frames++;
  if (now - vis.fpsT > 500) {
    els.fps.textContent = `${Math.round((vis.frames * 1000) / (now - vis.fpsT))} fps`;
    vis.frames = 0;
    vis.fpsT = now;
  }

  const w = window.innerWidth;
  const h = window.innerHeight;
  const cx = w * 0.5;
  const cy = h * 0.48;
  const energy = Math.max(state.micLevel, state.outLevel);
  const boost = state.talking ? 1.35 : state.speaking ? 1.15 : 0.85;

  // soft trail
  ctx.fillStyle = "rgba(5, 6, 12, 0.22)";
  ctx.fillRect(0, 0, w, h);

  // ambient nebula
  const g = ctx.createRadialGradient(cx, cy, 20, cx, cy, Math.max(w, h) * 0.55);
  g.addColorStop(0, `rgba(92, 225, 255, ${0.03 + energy * 0.12})`);
  g.addColorStop(0.35, `rgba(167, 139, 250, ${0.04 + energy * 0.08})`);
  g.addColorStop(1, "rgba(5,6,12,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  // waveform ring
  const baseR = Math.min(w, h) * 0.16;
  const segs = 96;
  ctx.beginPath();
  for (let i = 0; i <= segs; i++) {
    const t = (i / segs) * Math.PI * 2;
    const wobble =
      Math.sin(t * 5 + vis.t * 3.2) * 6 +
      Math.sin(t * 11 - vis.t * 4.5) * 3 +
      energy * 40 * Math.sin(t * 3 + vis.t * 6) * boost;
    const r = baseR + wobble + energy * 28;
    const x = cx + Math.cos(t) * r;
    const y = cy + Math.sin(t) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.strokeStyle = state.talking
    ? `rgba(255, 77, 141, ${0.35 + energy * 0.5})`
    : `rgba(92, 225, 255, ${0.25 + energy * 0.45})`;
  ctx.lineWidth = 1.5 + energy * 2.5;
  ctx.shadowColor = state.talking ? "#ff4d8d" : "#5ce1ff";
  ctx.shadowBlur = 12 + energy * 30;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // secondary ghost ring
  ctx.beginPath();
  for (let i = 0; i <= segs; i++) {
    const t = (i / segs) * Math.PI * 2 + 0.2;
    const wobble = Math.cos(t * 4 - vis.t * 2.5) * (8 + energy * 20);
    const r = baseR * 1.35 + wobble;
    const x = cx + Math.cos(t) * r;
    const y = cy + Math.sin(t) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.strokeStyle = `rgba(167, 139, 250, ${0.12 + energy * 0.25})`;
  ctx.lineWidth = 1;
  ctx.stroke();

  // particles
  for (const p of vis.particles) {
    p.a += dt * p.s * (0.25 + energy * 1.8) * (state.talking ? 1.6 : 1);
    const rr = p.r * Math.min(w, h) * (0.55 + energy * 0.35);
    const x = cx + Math.cos(p.a) * rr;
    const y = cy + Math.sin(p.a * 0.97) * rr * 0.92;
    const alpha = 0.15 + energy * 0.55 + (state.speaking ? 0.15 : 0);
    ctx.beginPath();
    ctx.fillStyle = `hsla(${p.hue}, 90%, 70%, ${alpha})`;
    ctx.arc(x, y, p.size * (0.7 + energy * 1.8), 0, Math.PI * 2);
    ctx.fill();
  }

  // center bloom when speaking out
  if (state.speaking || energy > 0.05) {
    const bloom = ctx.createRadialGradient(cx, cy, 0, cx, cy, baseR * (0.9 + energy));
    bloom.addColorStop(0, `rgba(255,255,255,${0.04 + energy * 0.08})`);
    bloom.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = bloom;
    ctx.beginPath();
    ctx.arc(cx, cy, baseR * 1.2, 0, Math.PI * 2);
    ctx.fill();
  }

  setCssLevels();
  // decay visual levels smoothly when idle
  state.micLevel *= 0.92;
  state.outLevel *= 0.9;
  if (state.outLevel < 0.02 && state.playNodes === 0) {
    state.speaking = false;
    els.mic.classList.toggle("speaking", false);
    if (!state.talking && state.connected) {
      setPill(els.pillMode, "idle", "idle");
    }
  }

  requestAnimationFrame(drawFrame);
}
requestAnimationFrame(drawFrame);

// ── Audio ───────────────────────────────────────────────────────────
async function ensureAudio() {
  if (!state.audioCtx) {
    state.audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  }
  if (state.audioCtx.state === "suspended") {
    await state.audioCtx.resume();
  }
  return state.audioCtx;
}

function rmsF32(buf) {
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / Math.max(1, buf.length));
}

function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function downsample(float32, fromRate, toRate) {
  if (fromRate === toRate) return float32;
  const ratio = fromRate / toRate;
  const newLen = Math.floor(float32.length / ratio);
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const idx = Math.floor(i * ratio);
    result[i] = float32[idx];
  }
  return result;
}

async function startCapture() {
  const actx = await ensureAudio();
  if (!state.mediaStream) {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
  }
  if (state.source) return;

  state.source = actx.createMediaStreamSource(state.mediaStream);
  // ScriptProcessor is deprecated but widely supported; AudioWorklet needs HTTPS module load.
  const bufferSize = 2048;
  state.processor = actx.createScriptProcessor(bufferSize, 1, 1);
  let leftover = new Float32Array(0);

  state.processor.onaudioprocess = (e) => {
    if (!state.talking || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const level = rmsF32(input);
    state.micLevel = Math.min(1, level * 4.5);

    const down = downsample(input, actx.sampleRate, SAMPLE_RATE);
    const merged = new Float32Array(leftover.length + down.length);
    merged.set(leftover);
    merged.set(down, leftover.length);

    let offset = 0;
    while (offset + FRAME_SAMPLES <= merged.length) {
      const slice = merged.subarray(offset, offset + FRAME_SAMPLES);
      const pcm = floatTo16BitPCM(slice);
      state.ws.send(pcm.buffer);
      offset += FRAME_SAMPLES;
    }
    leftover = merged.subarray(offset);
  };

  state.source.connect(state.processor);
  state.processor.connect(actx.destination); // required for some browsers; near-silent if gain 0
  // mute local loopback
  const mute = actx.createGain();
  mute.gain.value = 0;
  state.processor.disconnect();
  state.source.connect(state.processor);
  state.processor.connect(mute);
  mute.connect(actx.destination);

  log("Microphone capture started");
}

function stopCaptureTracks() {
  // keep stream for reuse; only stop processor path
  if (state.processor) {
    try {
      state.processor.disconnect();
    } catch (_) {}
    state.processor.onaudioprocess = null;
    state.processor = null;
  }
  if (state.source) {
    try {
      state.source.disconnect();
    } catch (_) {}
    state.source = null;
  }
}

function playPcmI16(arrayBuffer) {
  if (!state.audioCtx) return;
  const actx = state.audioCtx;
  const i16 = new Int16Array(arrayBuffer);
  if (!i16.length) return;

  let sum = 0;
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) {
    f32[i] = i16[i] / 32768;
    sum += f32[i] * f32[i];
  }
  state.outLevel = Math.min(1, Math.sqrt(sum / i16.length) * 4);
  state.speaking = true;
  els.mic.classList.add("speaking");
  setPill(els.pillMode, "speaking", "speak");

  const buf = actx.createBuffer(1, f32.length, SAMPLE_RATE);
  buf.copyToChannel(f32, 0);
  const src = actx.createBufferSource();
  src.buffer = buf;
  const gain = actx.createGain();
  gain.gain.value = 1;
  src.connect(gain);
  gain.connect(actx.destination);

  const now = actx.currentTime;
  if (state.playTime < now) state.playTime = now + 0.02;
  src.start(state.playTime);
  state.playTime += buf.duration;
  state.playNodes++;
  src.onended = () => {
    state.playNodes = Math.max(0, state.playNodes - 1);
  };
}

// ── WebSocket ───────────────────────────────────────────────────────
function connect() {
  const url = els.wsUrl.value.trim();
  if (!url) return;
  localStorage.setItem("s2s.ws", url);
  if (state.ws) {
    try {
      state.ws.close();
    } catch (_) {}
  }

  log(`Connecting ${url} …`);
  setPill(els.pillConn, "connecting", "idle");
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onopen = async () => {
    state.connected = true;
    setPill(els.pillConn, "online", "on");
    els.btnConnect.disabled = true;
    els.btnDisconnect.disabled = false;
    els.hint.textContent = "Connected. Hold the orb and speak.";
    log("WebSocket open");
    try {
      await ensureAudio();
    } catch (e) {
      log(`AudioContext: ${e.message || e}`);
    }
  };

  ws.onclose = () => {
    state.connected = false;
    state.talking = false;
    els.mic.classList.remove("active");
    setPill(els.pillConn, "offline", "off");
    setPill(els.pillMode, "idle", "idle");
    els.btnConnect.disabled = false;
    els.btnDisconnect.disabled = true;
    els.hint.textContent = "Disconnected. Connect to test the backend.";
    els.micLabel.textContent = "hold";
    stopCaptureTracks();
    log("WebSocket closed");
  };

  ws.onerror = () => log("WebSocket error");

  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      playPcmI16(ev.data);
    } else if (typeof ev.data === "string") {
      log(`← ${ev.data.slice(0, 120)}`);
    }
  };
}

function disconnect() {
  if (state.ws) state.ws.close();
  state.ws = null;
  stopTalking();
}

// ── Talk control ────────────────────────────────────────────────────
async function startTalking() {
  if (!state.connected || state.talking) return;
  try {
    await ensureAudio();
    await startCapture();
  } catch (e) {
    log(`Mic error: ${e.message || e}`);
    els.hint.textContent = "Microphone permission required.";
    return;
  }
  state.talking = true;
  els.mic.classList.add("active");
  els.micLabel.textContent = "live";
  setPill(els.pillMode, "listening", "live");
  els.hint.textContent = "Listening… release to end your turn.";
}

function stopTalking() {
  if (!state.talking) return;
  state.talking = false;
  els.mic.classList.remove("active");
  els.micLabel.textContent = state.talkMode === "toggle" ? "tap" : "hold";
  if (state.connected && !state.speaking) {
    setPill(els.pillMode, "idle", "idle");
    els.hint.textContent = "Processing… wait for the reply.";
  }
  // keep processor for next press; capture stays warm
}

// pointer events on big button
els.mic.addEventListener("pointerdown", async (e) => {
  e.preventDefault();
  els.mic.setPointerCapture(e.pointerId);
  if (!state.connected) {
    connect();
    els.hint.textContent = "Connecting… hold again once online.";
    return;
  }
  if (state.talkMode === "toggle") {
    if (state.talking) stopTalking();
    else await startTalking();
  } else {
    await startTalking();
  }
});

els.mic.addEventListener("pointerup", (e) => {
  if (state.talkMode === "hold") stopTalking();
});
els.mic.addEventListener("pointercancel", () => {
  if (state.talkMode === "hold") stopTalking();
});
els.mic.addEventListener("pointerleave", (e) => {
  // only end hold if primary button released path; ignore hover leave while captured
});

els.btnConnect.addEventListener("click", () => connect());
els.btnDisconnect.addEventListener("click", () => disconnect());
els.talkMode.addEventListener("change", () => {
  state.talkMode = els.talkMode.value;
  localStorage.setItem("s2s.talkMode", state.talkMode);
  els.micLabel.textContent = state.talkMode === "toggle" ? "tap" : "hold";
  if (state.talking) stopTalking();
});

// keyboard: space = PTT
window.addEventListener("keydown", async (e) => {
  if (e.code === "Space" && !e.repeat && e.target === document.body) {
    e.preventDefault();
    if (state.talkMode === "toggle") {
      if (state.talking) stopTalking();
      else await startTalking();
    } else await startTalking();
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space" && state.talkMode === "hold") {
    e.preventDefault();
    stopTalking();
  }
});

els.micLabel.textContent = state.talkMode === "toggle" ? "tap" : "hold";
log("s2s lab ready");
els.hint.textContent = "Set WebSocket URL, connect, then use the orb.";
