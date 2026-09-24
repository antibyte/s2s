/**
 * s2s lab — browser client for s2s-vulkan raw WebSocket PCM mode.
 * Talk first, Lab second: orb + captions primary; stack settings in drawer.
 */

const SAMPLE_RATE = 16000;
// The same build runs at / locally and below AuraGo's /speech-lab/ proxy.
const LAB_BASE_PATH = new URL('.', import.meta.url).pathname;
const labPath = (path) => LAB_BASE_PATH + String(path || '').replace(/^\/+/, '');
const FRAME_MS = 40;
const FRAME_SAMPLES = (SAMPLE_RATE * FRAME_MS) / 1000;
const VRAM_BUDGET_GB = 8;

/** Auto-reconnect: exponential backoff (ms), then cap. Keeps trying while wantConnected. */
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 20000;
const RECONNECT_JITTER = 0.25;

const $ = (id) => document.getElementById(id);

/** Central UI copy (DE). */
const copy = {
  offline: "Offline",
  connecting: "Verbinden…",
  ready: "Bereit",
  listening: "Zuhören",
  processing: "Verarbeiten",
  speaking: "Antwort",
  error: "Fehler",
  hintOffline: "Verbinden, dann Orb halten und sprechen.",
  hintConnecting: "WebSocket wird aufgebaut…",
  hintReadyHold: "Orb halten und sprechen. Leertaste geht auch.",
  hintReadyToggle: "Orb tippen zum Start/Stop. Leertaste geht auch.",
  hintListeningHold: "Zuhören… loslassen beendet den Zug.",
  hintListeningToggle: "Zuhören… nochmal tippen beendet den Zug.",
  hintProcessing: "Verarbeiten… Antwort kommt gleich.",
  hintSpeaking: "Bot spricht…",
  hintMicDenied: "Mikrofon-Zugriff nötig (Browser-Berechtigung).",
  hintDisconnected: "Getrennt. Erneut verbinden zum Testen.",
  hintReconnecting: (sec, n) =>
    `Verbindung weg — erneuter Versuch in ${sec}s (Nr. ${n})…`,
  hintReconnectNow: (n) => `Verbinde erneut… (Versuch ${n})`,
  hintStackSwap: "Stack wird umgeschaltet…",
  hintStackLive: (a, l, t) => `Stack live: ${a} → ${l} → ${t}`,
  hintStackErr: (m) => `Stack-Fehler: ${m || "?"}`,
  hintAsr: (t) => `Erkannt: „${(t || "").slice(0, 80)}"`,
  hintErr: (stage, m) => `Fehler (${stage || "pipeline"}): ${m || ""}`,
  toastConnected: "Verbunden",
  toastReconnected: "Wieder verbunden",
  toastDisconnected: "Getrennt",
  toastReconnectGiveUp: "Reconnect gestoppt",
  toastStack: "Stack aktualisiert",
  micHold: "halten",
  micToggle: "tippen",
  micLive: "live",
  micConnect: "connect",
  liveAsrEmpty: "Noch nichts gesagt",
  liveLlmEmpty: "Antwort erscheint hier",
  liveLlmWait: "…",
};

const els = {
  canvas: $("fx"),
  mic: $("mic"),
  micLabel: $("mic-label"),
  hint: $("hint"),
  wsUrl: $("ws-url"),
  talkMode: $("talk-mode"),
  btnConnect: $("btn-connect"),
  btnDisconnect: $("btn-disconnect"),
  btnConnectLab: $("btn-connect-lab"),
  btnDisconnectLab: $("btn-disconnect-lab"),
  btnLab: $("btn-lab"),
  btnLabClose: $("btn-lab-close"),
  labDrawer: $("lab-drawer"),
  labScrim: $("lab-scrim"),
  labBadge: $("lab-badge"),
  statusChip: $("status-chip"),
  barMic: $("bar-mic"),
  barOut: $("bar-out"),
  log: $("log"),
  fps: $("fps"),
  asrChoices: $("asr-choices"),
  ttsChoices: $("tts-choices"),
  voiceField: $("voice-field"),
  voiceSelect: $("voice-select"),
  voiceHint: $("voice-hint"),
  llmChoices: $("llm-choices"),
  asrBadge: $("asr-badge"),
  ttsBadge: $("tts-badge"),
  llmBadge: $("llm-badge"),
  stackBudget: $("stack-budget"),
  stackHint: $("stack-hint"),
  budgetFill: $("budget-fill"),
  presetChips: $("preset-chips"),
  pipeAsrName: $("pipe-asr-name"),
  pipeAsrMeta: $("pipe-asr-meta"),
  pipeLlmName: $("pipe-llm-name"),
  pipeLlmMeta: $("pipe-llm-meta"),
  pipeTtsName: $("pipe-tts-name"),
  pipeTtsMeta: $("pipe-tts-meta"),
  pipeline: $("pipeline"),
  liveAsrT: $("live-asr-t"),
  liveLlmT: $("live-llm-t"),
  coach: $("coach"),
  btnCoachDismiss: $("btn-coach-dismiss"),
  toast: $("toast"),
  btnLogClear: $("btn-log-clear"),
  btnBenchmark: $("btn-benchmark"),
  benchmarkKind: $("benchmark-kind"),
  benchmarkPrompts: $("benchmark-prompts"),
  benchmarkStatus: $("benchmark-status"),
  benchmarkResults: $("benchmark-results"),
  benchmarkHead: $("benchmark-head"),
  benchmarkTableWrap: $("benchmark-table-wrap"),
  benchmarkExport: $("benchmark-export"),
  modelDialog: $("model-dialog"),
  modelDialogTitle: $("model-dialog-title"),
  modelDialogCopy: $("model-dialog-copy"),
  modelDialogSize: $("model-dialog-size"),
  modelDialogHf: $("model-dialog-hf"),
  modelDialogHfLink: $("model-dialog-hf-link"),
  modelDialogHfToken: $("model-dialog-hf-token"),
  modelDialogHfRemember: $("model-dialog-hf-remember"),
  modelProgress: $("model-progress"),
  modelProgressFill: $("model-progress-fill"),
  modelProgressLabel: $("model-progress-label"),
  modelDialogError: $("model-dialog-error"),
  modelDialogCancel: $("model-dialog-cancel"),
  modelDialogConfirm: $("model-dialog-confirm"),
  capabilityLine: $("capability-line"),
};

// ── Catalogs ────────────────────────────────────────────────────────

/** @typedef {{ cpu: number, gpu: number, vram: number }} ResourceStars */
/** @typedef {{ show: boolean, cpu: boolean, nvidia: boolean, intel: boolean, amd: boolean, vulkan: boolean }} GpuSupport */
/** @typedef {{ id: string, label: string }} LanguageLabel */
/** @typedef {{ id: string, stage: string, name: string, tag: string, desc: string, vramGb: number, stars: ResourceStars, gpuSupport: GpuSupport, languageLabels: LanguageLabel[], meta: string, available?: boolean, compatible: boolean, incompatibilityLabel: string, activatable: boolean, managedRuntime: boolean, variantId: string, installed: boolean, bundled: boolean, downloadState: string, downloadSizeBytes: number, downloadedBytes: number, imageDownloadSizeBytes: number, deletable: boolean, downloadError: string, defaultVoice: string, voices: string[], voiceMode: "request"|"restart"|"fixed", licenses: string[], accessUrl: string, authRequired?: boolean, hfTokenConfigured?: boolean, runtimeState: string, runtimeReason: string, env?: Record<string,string>, note?: string }} EngineOpt */

/** Model choices and presets are populated exclusively from /api/v1/catalog. */
/** @type {EngineOpt[]} */
let ASR_OPTIONS = [];
/** @type {EngineOpt[]} */
let TTS_OPTIONS = [];
/** @type {EngineOpt[]} */
let LLM_OPTIONS = [];
let PRESETS = {};
/** @type {Record<string, any> | null} */
let CAPABILITY = null;
let CATALOG_REVISION = "";
let INITIAL_STACK_READ = false;
/** @type {Record<string, any> | null} */
let SUGGESTIONS = null;
/** Heuristic recommendations from /api/v1/suggestions (not benchmarks). */
const RECOMMENDED = {
  asr: new Set(),
  tts: new Set(),
  presets: new Set(),
};

function suggestionLanguage() {
  const raw =
    localStorage.getItem("s2s.lab.suggestLang") ||
    localStorage.getItem("s2s.lab.language") ||
    "de";
  return String(raw).trim().toLowerCase() || "de";
}

function applySuggestionsPayload(payload) {
  SUGGESTIONS = payload && typeof payload === "object" ? payload : null;
  CAPABILITY = SUGGESTIONS?.capability || CAPABILITY;
  RECOMMENDED.asr.clear();
  RECOMMENDED.tts.clear();
  RECOMMENDED.presets.clear();
  for (const item of SUGGESTIONS?.suggested_presets || []) {
    if (item?.id) RECOMMENDED.presets.add(item.id);
    if (item?.asr_id) RECOMMENDED.asr.add(item.asr_id);
    if (item?.tts_id) RECOMMENDED.tts.add(item.tts_id);
  }
  for (const item of SUGGESTIONS?.suggested_pairs || []) {
    if (item?.asr_id) RECOMMENDED.asr.add(item.asr_id);
    if (item?.tts_id) RECOMMENDED.tts.add(item.tts_id);
  }
}

function renderCapabilityLine() {
  if (!els.capabilityLine) return;
  const cap = CAPABILITY || SUGGESTIONS?.capability;
  if (!cap) {
    els.capabilityLine.hidden = true;
    els.capabilityLine.textContent = "";
    return;
  }
  const tier = cap.tier || "cpu-light";
  const device = cap.device_name || "host";
  const acc = Array.isArray(cap.accelerators) ? cap.accelerators.join("+") : "cpu";
  const ram =
    cap.ram_available_gb != null
      ? ` · RAM ~${Number(cap.ram_available_gb).toFixed(1)} GB frei`
      : "";
  const budget =
    SUGGESTIONS?.budget_vram_gb != null
      ? ` · Budget ~${Number(SUGGESTIONS.budget_vram_gb).toFixed(1)} GB VRAM`
      : "";
  const host = cap.host_agent_online ? " · Host-Agent online" : "";
  els.capabilityLine.hidden = false;
  els.capabilityLine.innerHTML = `<strong>System</strong> ${device} · tier <strong>${tier}</strong> · ${acc}${ram}${budget}${host}`;
}

async function loadSuggestions({ quiet = false } = {}) {
  try {
    const lang = encodeURIComponent(suggestionLanguage());
    const response = await fetch(
      labPath(`/api/v1/suggestions?language=${lang}&stable_only=true&limit=8`),
      { cache: "no-store" }
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    applySuggestionsPayload(payload);
    renderCapabilityLine();
    if (!quiet) {
      const n =
        (payload.suggested_presets?.length || 0) + (payload.suggested_pairs?.length || 0);
      log(`Vorschläge (${payload.scoring || "heuristic"}): ${n} Einträge für ${suggestionLanguage()}`);
    }
  } catch (error) {
    if (!quiet) log(`Vorschläge nicht geladen: ${error.message}`);
  }
}

/** Clamp demand rating to 1–5 stars. */
function clampStars(value, fallback = 1) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return Math.min(5, Math.max(1, fallback));
  return Math.min(5, Math.max(1, Math.round(n)));
}

/** Catalog demand (1=leicht … 5=schwer). Used when API omits resources.stars. */
const DEMAND_STARS = {
  "fw-tiny": { cpu: 2, gpu: 1, vram: 1 },
  "fw-base": { cpu: 2, gpu: 2, vram: 1 },
  "fw-small": { cpu: 2, gpu: 2, vram: 2 },
  "parakeet-tdt-0.6b-v3": { cpu: 2, gpu: 3, vram: 3 },
  "voxtral-mini-4b-realtime": { cpu: 3, gpu: 4, vram: 4 },
  "wcpp-base": { cpu: 2, gpu: 2, vram: 2 },
  supertonic: { cpu: 3, gpu: 1, vram: 1 },
  "qwen3-tts-0.6b": { cpu: 2, gpu: 3, vram: 3 },
  kokoro: { cpu: 3, gpu: 1, vram: 1 },
  "vibevoice-realtime-0.5b": { cpu: 3, gpu: 2, vram: 2 },
  "higgs-tts-3-4b": { cpu: 2, gpu: 5, vram: 5 },
  "external-openai": { cpu: 1, gpu: 1, vram: 1 },
  "local-fallback": { cpu: 2, gpu: 3, vram: 3 },
};

/** Derive CPU/GPU/VRAM star demand from catalog resources (1=leicht … 5=schwer). */
function resourceStars(resources, backendId = "") {
  const vramGb = Number(resources?.vram_gb || 0);
  const ramGb = Number(resources?.ram_gb || 0);
  const fromGb = (gb) => {
    if (gb <= 0) return 1;
    if (gb < 1) return 2;
    if (gb < 4) return 3;
    if (gb < 12) return 4;
    return 5;
  };
  const catalog = resources?.stars || {};
  const hasCatalog =
    Number(catalog.cpu) > 0 || Number(catalog.gpu) > 0 || Number(catalog.vram) > 0;
  const known = DEMAND_STARS[backendId] || null;
  if (hasCatalog) {
    return {
      cpu: clampStars(catalog.cpu, known?.cpu || fromGb(ramGb)),
      gpu: clampStars(catalog.gpu, known?.gpu || fromGb(vramGb)),
      vram: clampStars(catalog.vram, known?.vram || fromGb(vramGb)),
    };
  }
  if (known) {
    return {
      cpu: clampStars(known.cpu),
      gpu: clampStars(known.gpu),
      vram: clampStars(known.vram),
    };
  }
  return {
    cpu: clampStars(fromGb(ramGb)),
    gpu: clampStars(fromGb(vramGb)),
    vram: clampStars(fromGb(vramGb)),
  };
}

/** HTML for one demand row (label + 5 stars). */
function demandStarRow(label, filled, title) {
  const n = clampStars(filled, 1);
  const glyphs = Array.from({ length: 5 }, (_, i) =>
    i < n
      ? `<span class="star star-on" aria-hidden="true">★</span>`
      : `<span class="star star-off" aria-hidden="true">☆</span>`
  ).join("");
  return `<div class="demand-row" title="${title || `${label}: ${n}/5`}">
    <span class="demand-label">${label}</span>
    <span class="demand-stars" role="img" aria-label="${label} ${n} von 5">${glyphs}</span>
  </div>`;
}

function demandStarsBlock(stars) {
  return `<div class="demand-block" aria-label="Ressourcenbedarf">
    ${demandStarRow("CPU", stars.cpu, `CPU-Last: ${stars.cpu}/5`)}
    ${demandStarRow("GPU", stars.gpu, `GPU-Rechenlast: ${stars.gpu}/5`)}
    ${demandStarRow("VRAM", stars.vram, `VRAM-Bedarf: ${stars.vram}/5`)}
  </div>`;
}

/**
 * Derive CPU / AMD / Intel / NVIDIA / Vulkan support from catalog variants.
 * Remote-only backends (no local accelerator) get no chip row.
 */
function gpuSupportFromVariants(variants) {
  /** @type {GpuSupport} */
  const support = {
    show: false,
    cpu: false,
    nvidia: false,
    intel: false,
    amd: false,
    vulkan: false,
  };
  for (const variant of variants || []) {
    const acc = String(variant?.accelerator || "")
      .trim()
      .toLowerCase();
    if (!acc || acc === "remote") continue;
    const vendors = (Array.isArray(variant?.vendors) ? variant.vendors : [])
      .map((v) => String(v).trim().toLowerCase())
      .filter(Boolean);

    if (acc === "cpu") {
      support.cpu = true;
      support.show = true;
      continue;
    }

    const isGpuAccel = acc === "cuda" || acc === "sycl" || acc === "vulkan";
    if (!isGpuAccel && !vendors.some((v) => v === "nvidia" || v === "intel" || v === "amd")) {
      continue;
    }
    support.show = true;
    // CUDA is NVIDIA-only in this lab.
    if (acc === "cuda" || vendors.includes("nvidia")) support.nvidia = true;
    // oneAPI/SYCL paths target Intel Arc / XPU.
    if (acc === "sycl" || vendors.includes("intel")) support.intel = true;
    if (vendors.includes("amd")) support.amd = true;
    if (acc === "vulkan") {
      support.vulkan = true;
      // Vulkan variants declare brand support via vendors (often multi-vendor).
      if (vendors.includes("nvidia") || vendors.includes("any")) support.nvidia = true;
      if (vendors.includes("intel") || vendors.includes("any")) support.intel = true;
      if (vendors.includes("amd") || vendors.includes("any")) support.amd = true;
    }
  }
  return support;
}

function gpuSupportBlock(support) {
  if (!support?.show) return "";
  const chips = [
    { id: "cpu", label: "CPU", on: support.cpu, title: "CPU-Backend" },
    { id: "amd", label: "AMD", on: support.amd, title: "AMD GPU" },
    { id: "intel", label: "Intel", on: support.intel, title: "Intel GPU (Arc/XPU/SYCL)" },
    { id: "nvidia", label: "NVIDIA", on: support.nvidia, title: "NVIDIA GPU (CUDA)" },
    { id: "vulkan", label: "Vulkan", on: support.vulkan, title: "Vulkan-Backend" },
  ];
  const supported = chips.filter((c) => c.on).map((c) => c.label);
  const html = chips
    .map(
      (c) =>
        `<span class="gpu-chip gpu-chip-${c.id}${c.on ? " on" : " off"}" title="${
          c.on ? `${c.title}: unterstützt` : `${c.title}: nicht unterstützt`
        }" aria-label="${c.label} ${c.on ? "unterstützt" : "nicht unterstützt"}">${c.label}</span>`
    )
    .join("");
  return `<div class="gpu-support" role="group" aria-label="Backend-Support: ${
    supported.length ? supported.join(", ") : "keiner"
  }">${html}</div>`;
}

/**
 * Language chips from catalog `languages`.
 * Many codes → one Multilingual label; few → EN DE FR …
 * Skips auto/na and models with no concrete language list.
 */
function languageLabelsFromCatalog(languages) {
  const raw = (Array.isArray(languages) ? languages : [])
    .map((item) => String(item || "").trim().toLowerCase())
    .filter((item) => item && item !== "auto" && item !== "na");
  /** @type {string[]} */
  const codes = [];
  const seen = new Set();
  for (const item of raw) {
    // Prefer ISO-639-1 style 2-letter codes (zh-cn → ZH, en-us → EN).
    const code = (item.includes("-") ? item.split("-")[0] : item.slice(0, 2)).toUpperCase();
    if (!code || seen.has(code)) continue;
    seen.add(code);
    codes.push(code);
  }
  if (codes.length === 0) return [];
  if (codes.length >= 8) {
    return [
      {
        id: "multilingual",
        label: "Multilingual",
      },
    ];
  }
  return codes.map((code) => ({ id: code.toLowerCase(), label: code }));
}

function languageLabelsBlock(labels) {
  if (!Array.isArray(labels) || !labels.length) return "";
  const html = labels
    .map((item) => {
      const multi = item.id === "multilingual";
      return `<span class="lang-chip${multi ? " lang-chip-multi" : ""}" title="${
        multi ? "Viele Sprachen unterstützt" : `Sprache: ${item.label}`
      }">${item.label}</span>`;
    })
    .join("");
  const aria = labels.map((item) => item.label).join(", ");
  return `<div class="lang-support" role="group" aria-label="Sprachen: ${aria}">${html}</div>`;
}

function catalogOption(entry) {
  const variant = entry.selected_variant;
  const accelerator = variant?.accelerator || "unavailable";
  const endpoint = variant?.endpoint || "";
  const hostManaged = entry.host_managed === true;
  const runtimeState = entry.runtime_state || (hostManaged ? "unknown" : "external");
  const runtimeReason = entry.runtime_reason || "";
  const licenses = Array.isArray(entry.licenses) ? entry.licenses.filter(Boolean) : [];
  const accessUrl = entry.access_url || "";
  const authRequired =
    entry.auth_required === true ||
    Boolean(accessUrl) ||
    (Array.isArray(entry.artifacts) &&
      entry.artifacts.some((artifact) => artifact?.auth === "huggingface"));
  const hfTokenConfigured = entry.hf_token_configured === true;
  const compatible = entry.compatible !== undefined ? entry.compatible === true : entry.available === true;
  const activatable = entry.activatable !== undefined ? entry.activatable === true : entry.available === true;
  const reason = String(entry.reason || "").trim();
  const platform = /^not supported on ([a-z0-9_-]+)$/i.exec(reason)?.[1];
  const incompatibilityLabel = platform
    ? `Keine Variante für ${platform === "linux" ? "Linux" : platform === "windows" ? "Windows" : platform} verfügbar`
    : reason === "only experimental variants are registered"
      ? "Nur experimentelle Varianten verfügbar"
      : reason.startsWith("no compatible variant for ")
        ? "Keine passende Variante für diese Hardware verfügbar"
        : "Auf diesem System nicht verfügbar";
  const stateNote = runtimeReason
    ? `Host-Agent: ${runtimeReason}`
    : compatible
      ? `${variant.id}${variant.stable ? "" : " · experimental"}${hostManaged ? ` · ${runtimeState}` : ""}`
      : incompatibilityLabel;
  const accessNote = licenses.length
    ? `Lizenz: ${licenses.join(" + ")}${authRequired ? " · HF-Zugang erforderlich" : ""}`
    : authRequired
      ? "HF-Zugang erforderlich"
      : "";
  const env =
    entry.stage === "asr"
      ? { S2S_WHISPER_URL: endpoint }
      : entry.stage === "tts"
        ? { S2S_TTS: "http", S2S_TTS_URL: endpoint, S2S_TTS_MODEL: entry.model }
        : { S2S_LLM_URL: endpoint, S2S_LLM_MODEL: entry.model };
  return {
    id: entry.id,
    stage: entry.stage,
    name: entry.name,
    tag: entry.tag || accelerator,
    desc: entry.description || "",
    vramGb: Number(entry.resources?.vram_gb || 0),
    stars: resourceStars(entry.resources, entry.id),
    gpuSupport: gpuSupportFromVariants(entry.variants),
    languageLabels: languageLabelsFromCatalog(entry.languages),
    meta: `${entry.model || entry.protocol} · ${accelerator}${hostManaged ? " · Windows host" : ""}`,
    available: activatable && runtimeState !== "unavailable",
    compatible,
    incompatibilityLabel,
    activatable,
    managedRuntime: hostManaged || Boolean(variant?.container),
    variantId: entry.variant_id || variant?.id || "",
    installed: entry.installed === true,
    bundled: entry.bundled === true,
    downloadState: entry.download_state || (entry.installed ? "installed" : "missing"),
    downloadSizeBytes: Number(entry.download_size_bytes || 0),
    downloadedBytes: Number(entry.downloaded_bytes || 0),
    imageDownloadSizeBytes: Number(entry.image_download_size_bytes || 0),
    deletable: entry.deletable === true,
    downloadError: entry.download_error || "",
    defaultVoice: entry.default_voice || "",
    voices: Array.isArray(entry.voices) ? entry.voices.filter(Boolean) : [],
    voiceMode: ["request", "restart", "fixed"].includes(entry.voice_mode)
      ? entry.voice_mode
      : "fixed",
    licenses,
    accessUrl,
    authRequired,
    hfTokenConfigured,
    hostManaged,
    runtimeState,
    runtimeReason,
    env,
    note: [stateNote, accessNote].filter(Boolean).join(" · "),
  };
}

async function loadBackendCatalog({ quiet = false } = {}) {
  try {
    const response = await fetch(labPath("/api/v1/catalog"), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const catalog = await response.json();
    if (![1, 2].includes(catalog.schema_version) || !Array.isArray(catalog.backends)) {
      throw new Error("unsupported catalog response");
    }
    CATALOG_REVISION = String(catalog.catalog_revision || "");
    const options = catalog.backends.map(catalogOption);
    const asr = options.filter((option, index) => catalog.backends[index].stage === "asr");
    const tts = options.filter((option, index) => catalog.backends[index].stage === "tts");
    const llm = options.filter((option, index) => catalog.backends[index].stage === "llm");
    if (asr.length) ASR_OPTIONS = asr;
    if (tts.length) TTS_OPTIONS = tts;
    if (llm.length) LLM_OPTIONS = llm;
    if (Array.isArray(catalog.presets) && catalog.presets.length) {
      PRESETS = Object.fromEntries(
        catalog.presets.map((preset) => [
          preset.id,
          {
            label: preset.name,
            asr: preset.asr_id,
            tts: preset.tts_id,
            llm: preset.llm_id || "local-fallback",
            compose: `Catalog preset · ${preset.name}`,
          },
        ])
      );
    }
    if (catalog.hardware && typeof catalog.hardware === "object") {
      CAPABILITY = catalog.hardware;
      renderCapabilityLine();
    }
    if (!INITIAL_STACK_READ) {
      INITIAL_STACK_READ = true;
      try {
        const stackResponse = await fetch(labPath("/api/v1/stack"), { cache: "no-store" });
        if (!stackResponse.ok) throw new Error(`HTTP ${stackResponse.status}`);
        const active = (await stackResponse.json()).runtime || {};
        if (ASR_OPTIONS.some((option) => option.id === active.asr)) lab.asr = active.asr;
        if (TTS_OPTIONS.some((option) => option.id === active.tts)) lab.tts = active.tts;
        if (LLM_OPTIONS.some((option) => option.id === active.llm)) lab.llm = active.llm;
        if (active.voice) lab.voice = active.voice;
      } catch (error) {
        log(`Aktiven Stack nicht geladen: ${error.message}`);
      }
    }
    if (!ASR_OPTIONS.some((option) => option.id === lab.asr)) lab.asr = ASR_OPTIONS[0].id;
    if (!TTS_OPTIONS.some((option) => option.id === lab.tts)) lab.tts = TTS_OPTIONS[0].id;
    if (!LLM_OPTIONS.some((option) => option.id === lab.llm)) lab.llm = LLM_OPTIONS[0].id;
    syncVoiceForTts(TTS_OPTIONS.find((option) => option.id === lab.tts));
    if (!PRESETS[lab.preset]) lab.preset = "custom";
    persistLab();
    await loadSuggestions({ quiet: true });
    renderPresets();
    refreshLabUi();
    if (!quiet) {
      log(
        `Backend catalog v${catalog.schema_version}: ${catalog.hardware?.device_name || "host"} · tier ${catalog.hardware?.tier || "?"}`
      );
    }
    return catalog;
  } catch (error) {
    const message = `Backend-Katalog nicht erreichbar: ${error.message}`;
    log(message);
    if (els.stackHint) els.stackHint.textContent = message;
    if (els.asrChoices) els.asrChoices.textContent = message;
    throw error;
  }
}

// ── Lab selection state ─────────────────────────────────────────────
const LAB_DEFAULTS_VERSION = "2";
if (localStorage.getItem("s2s.lab.defaultsVersion") !== LAB_DEFAULTS_VERSION) {
  localStorage.setItem("s2s.lab.asr", "fw-tiny");
  localStorage.setItem("s2s.lab.tts", "supertonic");
  localStorage.setItem("s2s.lab.llm", "local-fallback");
  localStorage.setItem("s2s.lab.preset", "balanced");
  localStorage.setItem("s2s.lab.defaultsVersion", LAB_DEFAULTS_VERSION);
}

const lab = {
  asr: localStorage.getItem("s2s.lab.asr") || "fw-tiny",
  tts: localStorage.getItem("s2s.lab.tts") || "supertonic",
  llm: localStorage.getItem("s2s.lab.llm") || "local-fallback",
  voice: localStorage.getItem("s2s.lab.voice") || "",
  preset: localStorage.getItem("s2s.lab.preset") || "balanced",
  swapBusy: false,
};

const state = {
  ws: null,
  connected: false,
  connecting: false,
  /** User wants a live session (enables auto-reconnect until Trennen). */
  wantConnected: false,
  talking: false,
  speaking: false,
  processing: false,
  error: false,
  micLevel: 0,
  outLevel: 0,
  audioCtx: null,
  mediaStream: null,
  processor: null, // ScriptProcessor fallback
  workletNode: null, // AudioWorklet primary path
  workletReady: false,
  source: null,
  playTime: 0,
  playNodes: 0,
  talkMode: "hold",
  ttsActive: false,
  labOpen: false,
  /** @type {'offline'|'connecting'|'ready'|'listening'|'processing'|'speaking'|'error'} */
  ui: "offline",
};

const reconnect = {
  attempts: 0,
  timer: 0,
  countdown: 0,
  tickTimer: 0,
  /** True if we already showed a disconnect toast for this outage. */
  notifiedDrop: false,
  /** Was connected at least once in this wantConnected session (for "wieder verbunden"). */
  hadSession: false,
};

function clearReconnectTimers() {
  if (reconnect.timer) {
    clearTimeout(reconnect.timer);
    reconnect.timer = 0;
  }
  if (reconnect.countdown) {
    clearInterval(reconnect.countdown);
    reconnect.countdown = 0;
  }
  if (reconnect.tickTimer) {
    clearTimeout(reconnect.tickTimer);
    reconnect.tickTimer = 0;
  }
}

function stopReconnectLoop() {
  clearReconnectTimers();
  reconnect.attempts = 0;
  reconnect.notifiedDrop = false;
}

function reconnectDelayMs(attempt) {
  // attempt 1 → base, then 2s, 4s, … capped + jitter
  const exp = Math.min(
    RECONNECT_MAX_MS,
    RECONNECT_BASE_MS * Math.pow(2, Math.max(0, attempt - 1))
  );
  const jitter = exp * RECONNECT_JITTER * (Math.random() * 2 - 1);
  return Math.max(400, Math.round(exp + jitter));
}

/**
 * Schedule another connect() while the user still wants a session.
 * @param {string} [reason]
 */
function scheduleReconnect(reason) {
  if (!state.wantConnected) return;
  if (state.connected || state.connecting) return;
  if (reconnect.timer) return;

  reconnect.attempts += 1;
  const delay = reconnectDelayMs(reconnect.attempts);
  const due = Date.now() + delay;
  log(
    `Reconnect #${reconnect.attempts} in ${Math.round(delay / 1000)}s` +
      (reason ? ` (${reason})` : "")
  );

  const updateHint = () => {
    if (!state.wantConnected || state.connected || state.connecting) return;
    const sec = Math.max(1, Math.ceil((due - Date.now()) / 1000));
    setHint(copy.hintReconnecting(sec, reconnect.attempts), { sticky: true });
    if (els.statusChip) {
      els.statusChip.textContent = copy.connecting;
      els.statusChip.dataset.state = "connecting";
    }
  };
  updateHint();
  reconnect.countdown = window.setInterval(updateHint, 500);

  reconnect.timer = window.setTimeout(() => {
    reconnect.timer = 0;
    if (reconnect.countdown) {
      clearInterval(reconnect.countdown);
      reconnect.countdown = 0;
    }
    if (!state.wantConnected || state.connected) return;
    setHint(copy.hintReconnectNow(reconnect.attempts), { sticky: true });
    connect({ auto: true });
  }, delay);
}

// ── UI state machine ────────────────────────────────────────────────

function deriveUiState() {
  if (state.connecting || (state.wantConnected && reconnect.timer)) return "connecting";
  if (state.error && !state.connected && !state.wantConnected) return "error";
  if (!state.connected) return state.wantConnected ? "connecting" : "offline";
  if (state.talking) return "listening";
  if (state.speaking) return "speaking";
  if (state.processing) return "processing";
  return "ready";
}

function statusLabel(ui) {
  return copy[ui] || copy.offline;
}

function readyHint() {
  return state.talkMode === "toggle" ? copy.hintReadyToggle : copy.hintReadyHold;
}

function listeningHint() {
  return state.talkMode === "toggle" ? copy.hintListeningToggle : copy.hintListeningHold;
}

function micModeLabel() {
  if (!state.connected) return copy.micConnect;
  if (state.talking) return copy.micLive;
  return state.talkMode === "toggle" ? copy.micToggle : copy.micHold;
}

function syncConnectButtons() {
  // During auto-reconnect, still show Trennen so user can abort.
  const session = state.wantConnected || state.connected || state.connecting;
  const busy = state.connecting || (!!reconnect.timer && state.wantConnected);
  if (els.btnConnect) {
    els.btnConnect.disabled = busy || state.connected;
    els.btnConnect.hidden = session && (state.connected || state.wantConnected);
    els.btnConnect.textContent = busy ? copy.connecting : "Verbinden";
  }
  if (els.btnDisconnect) {
    els.btnDisconnect.disabled = !session;
    els.btnDisconnect.hidden = !session;
  }
  if (els.btnConnectLab) {
    els.btnConnectLab.disabled = busy || state.connected;
  }
  if (els.btnDisconnectLab) {
    els.btnDisconnectLab.disabled = !session;
  }
}

function setPipeActive(stage) {
  if (!els.pipeline) return;
  els.pipeline.querySelectorAll(".pipe-node").forEach((n) => {
    n.classList.toggle("is-active", !!stage && n.dataset.stage === stage);
  });
}

function refreshUi() {
  const ui = deriveUiState();
  state.ui = ui;

  if (els.statusChip) {
    els.statusChip.textContent = statusLabel(ui);
    els.statusChip.dataset.state = ui;
  }

  if (els.mic) {
    els.mic.classList.toggle("active", state.talking);
    els.mic.classList.toggle("speaking", state.speaking);
    els.mic.classList.toggle("is-offline", !state.connected);
    els.mic.setAttribute(
      "aria-label",
      !state.connected
        ? "Verbinden und sprechen"
        : state.talkMode === "toggle"
          ? "Tippen zum Sprechen"
          : "Halten zum Sprechen"
    );
  }
  if (els.micLabel) els.micLabel.textContent = micModeLabel();

  syncConnectButtons();

  // Pipeline stage highlight from session phase
  if (ui === "listening" || (ui === "processing" && !state.speaking)) {
    // after stop: processing often starts with ASR — leave as-is unless event sets it
  }
  if (ui === "speaking") setPipeActive("tts");
  if (ui === "ready" || ui === "offline") setPipeActive(null);

  if (els.hint && !els.hint.dataset.locked) {
    let text = copy.hintOffline;
    if (ui === "connecting") text = copy.hintConnecting;
    else if (ui === "ready") text = readyHint();
    else if (ui === "listening") text = listeningHint();
    else if (ui === "processing") text = copy.hintProcessing;
    else if (ui === "speaking") text = copy.hintSpeaking;
    else if (ui === "error") text = copy.hintDisconnected;
    els.hint.textContent = text;
    els.hint.classList.toggle("is-error", ui === "error");
  }
}

function setHint(text, { error = false, sticky = false } = {}) {
  if (!els.hint) return;
  els.hint.textContent = text;
  els.hint.classList.toggle("is-error", error);
  if (sticky) els.hint.dataset.locked = "1";
  else delete els.hint.dataset.locked;
}

function clearHintLock() {
  if (els.hint) delete els.hint.dataset.locked;
}

let toastTimer = 0;
function showToast(msg, { error = false, ms = 2200 } = {}) {
  if (!els.toast) return;
  els.toast.hidden = false;
  els.toast.textContent = msg;
  els.toast.classList.toggle("is-error", error);
  // force reflow for transition
  void els.toast.offsetWidth;
  els.toast.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    els.toast.classList.remove("is-visible");
    window.setTimeout(() => {
      if (!els.toast.classList.contains("is-visible")) els.toast.hidden = true;
    }, 220);
  }, ms);
}

// ── Lab drawer ──────────────────────────────────────────────────────

function openLab(sectionId) {
  state.labOpen = true;
  document.body.classList.add("lab-open");
  if (els.labDrawer) {
    els.labDrawer.classList.add("is-open");
    els.labDrawer.setAttribute("aria-hidden", "false");
  }
  if (els.labScrim) {
    els.labScrim.hidden = false;
    requestAnimationFrame(() => els.labScrim.classList.add("is-open"));
  }
  if (els.btnLab) els.btnLab.setAttribute("aria-expanded", "true");
  if (sectionId) {
    const el = document.getElementById(sectionId);
    if (el) {
      const details = el.closest("details");
      if (details) details.open = true;
      el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
  localStorage.setItem("s2s.labOpen", "1");
}

function closeLab() {
  state.labOpen = false;
  document.body.classList.remove("lab-open");
  if (els.labDrawer) {
    els.labDrawer.classList.remove("is-open");
    els.labDrawer.setAttribute("aria-hidden", "true");
  }
  if (els.labScrim) {
    els.labScrim.classList.remove("is-open");
    window.setTimeout(() => {
      if (!state.labOpen) els.labScrim.hidden = true;
    }, 220);
  }
  if (els.btnLab) els.btnLab.setAttribute("aria-expanded", "false");
  localStorage.setItem("s2s.labOpen", "0");
}

function toggleLab() {
  if (state.labOpen) closeLab();
  else openLab();
}

// ── Stack / lab selection ───────────────────────────────────────────

function setSwapBusy(busy, { lockVoice = true } = {}) {
  lab.swapBusy = !!busy;
  // Voice picker stays usable for pure voice changes and after completed swaps.
  // Only hard-disable during a real backend cutover (not voice-only).
  if (els.voiceSelect) {
    els.voiceSelect.disabled = !!(busy && lockVoice);
  }
}

function clearSwapBusy() {
  setSwapBusy(false);
  // Re-render so disabled state cannot stick after replaceChildren.
  const tts = findOpt(TTS_OPTIONS, lab.tts);
  if (tts) renderVoicePicker(tts);
}

async function sendStackToBackend({ voiceOnly = false } = {}) {
  const activeTts = findOpt(TTS_OPTIONS, lab.tts);
  const payload = voiceOnly
    ? { voice: lab.voice }
    : {
        asr_id: lab.asr,
        tts_id: lab.tts,
        llm_id: lab.llm,
        voice: lab.voice,
      };
  log(
    voiceOnly
      ? `Stimme → ${lab.voice}`
      : `Hot-swap → asr=${lab.asr} tts=${lab.tts} llm=${lab.llm} voice=${lab.voice}`
  );
  setHint(copy.hintStackSwap, { sticky: true });
  // Restart-backed voices lock the picker while their sidecar is replaced.
  setSwapBusy(true, { lockVoice: !voiceOnly || activeTts?.voiceMode === "restart" });
  try {
    const response = await fetch(labPath("/api/v1/stack"), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      // Host TTS probe is ~12s; allow multi-stage warm but never hang forever.
      signal: AbortSignal.timeout(180_000),
    });
    const status = await response.json().catch(() => ({}));
    if (!response.ok || status.ok === false) {
      throw new Error(status.message || status.error || `HTTP ${response.status}`);
    }
    handlePipelineEvent(
      JSON.stringify({
        type: "stack",
        ...(status.runtime || {}),
        ok: status.ok,
        message: status.message,
      })
    );
    clearSwapBusy();
  } catch (error) {
    clearSwapBusy();
    setHint(copy.hintStackErr(error.message), { error: true, sticky: true });
    showToast(copy.hintStackErr(error.message), { error: true });
  }
}

function findOpt(list, id) {
  return list.find((o) => o.id === id) || list[0];
}

function allModelOptions() {
  return [...ASR_OPTIONS, ...TTS_OPTIONS, ...LLM_OPTIONS];
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value >= 1024 ** 3) {
    return `${(value / 1024 ** 3).toLocaleString("de-DE", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })} GB`;
  }
  if (value >= 1024 ** 2) {
    return `${(value / 1024 ** 2).toLocaleString("de-DE", {
      maximumFractionDigits: 1,
    })} MB`;
  }
  return `${Math.max(0, value).toLocaleString("de-DE")} Bytes`;
}

function formatModelSize(bytes) {
  const value = Math.max(0, Number(bytes || 0));
  const gb = `${(value / 1024 ** 3).toLocaleString("de-DE", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} GB`;
  return value > 0 && value < 1024 ** 3 ? `${gb} · ${formatBytes(value)}` : gb;
}

async function apiRequest(url, options = {}) {
  const response = await fetch(String(url).startsWith('/api/') ? labPath(url) : url, options);
  if (response.ok) return response.status === 204 ? null : response.json();
  let message = `HTTP ${response.status}`;
  try {
    const body = await response.json();
    message = body.error || body.message || message;
  } catch (_) {}
  throw new Error(message);
}

function setModelDialogProgress(downloaded, total) {
  const safeTotal = Math.max(0, Number(total || 0));
  const safeDownloaded = Math.max(0, Number(downloaded || 0));
  const percent = safeTotal > 0 ? Math.min(100, (safeDownloaded / safeTotal) * 100) : 0;
  if (els.modelProgressFill) els.modelProgressFill.style.width = `${percent}%`;
  if (els.modelProgressLabel) {
    els.modelProgressLabel.textContent =
      `${Math.round(percent)} % · ${formatBytes(safeDownloaded)} / ${formatBytes(safeTotal)}`;
  }
}

function resetModelDialog() {
  if (els.modelProgress) els.modelProgress.hidden = true;
  if (els.modelDialogError) {
    els.modelDialogError.hidden = true;
    els.modelDialogError.textContent = "";
  }
  if (els.modelDialogHf) els.modelDialogHf.hidden = true;
  if (els.modelDialogHfToken) {
    els.modelDialogHfToken.value = "";
    els.modelDialogHfToken.required = false;
  }
  if (els.modelDialogHfRemember) els.modelDialogHfRemember.checked = true;
  if (els.modelDialogHfLink) {
    els.modelDialogHfLink.href = "#";
    els.modelDialogHfLink.hidden = true;
  }
  if (els.modelDialogConfirm) {
    els.modelDialogConfirm.hidden = false;
    els.modelDialogConfirm.disabled = false;
    els.modelDialogConfirm.classList.remove("danger");
  }
  if (els.modelDialogCancel) {
    els.modelDialogCancel.disabled = false;
    els.modelDialogCancel.textContent = "Abbrechen";
  }
}

/**
 * @param {{
 *   title: string,
 *   message: string,
 *   size?: string,
 *   confirmLabel: string,
 *   danger?: boolean,
 *   hfAuth?: { accessUrl?: string, tokenConfigured?: boolean } | null,
 * }} opts
 * @returns {Promise<false | { hfToken: string, remember: boolean }>}
 */
function confirmModelAction({
  title,
  message,
  size = "",
  confirmLabel,
  danger = false,
  hfAuth = null,
}) {
  if (!els.modelDialog) {
    const ok = window.confirm(`${message}\n${size}`);
    if (!ok) return Promise.resolve(false);
    if (!hfAuth) return Promise.resolve({ hfToken: "", remember: true });
    const token = window.prompt("Hugging Face Token (hf_…)", "") || "";
    return Promise.resolve({ hfToken: token.trim(), remember: true });
  }
  resetModelDialog();
  els.modelDialogTitle.textContent = title;
  els.modelDialogCopy.textContent = message;
  els.modelDialogSize.textContent = size;
  els.modelDialogConfirm.textContent = confirmLabel;
  els.modelDialogConfirm.classList.toggle("danger", danger);
  if (hfAuth && els.modelDialogHf) {
    els.modelDialogHf.hidden = false;
    if (els.modelDialogHfLink) {
      if (hfAuth.accessUrl) {
        els.modelDialogHfLink.href = hfAuth.accessUrl;
        els.modelDialogHfLink.hidden = false;
      } else {
        els.modelDialogHfLink.hidden = true;
      }
    }
    if (els.modelDialogHfToken) {
      // Server-side / remembered token is enough; field stays optional then.
      els.modelDialogHfToken.required = !hfAuth.tokenConfigured;
      els.modelDialogHfToken.placeholder = hfAuth.tokenConfigured
        ? "optional — Token ist bereits konfiguriert"
        : "hf_…";
    }
  }
  if (!els.modelDialog.open) els.modelDialog.showModal();
  return new Promise((resolve) => {
    const finish = (answer) => {
      els.modelDialogConfirm.onclick = null;
      els.modelDialogCancel.onclick = null;
      els.modelDialog.oncancel = null;
      if (els.modelDialog.open) els.modelDialog.close();
      resolve(answer);
    };
    els.modelDialogConfirm.onclick = () => {
      if (hfAuth && els.modelDialogHfToken) {
        const token = (els.modelDialogHfToken.value || "").trim();
        if (!token && !hfAuth.tokenConfigured) {
          if (els.modelDialogError) {
            els.modelDialogError.hidden = false;
            els.modelDialogError.textContent =
              "Bitte Hugging Face Token eintragen (oder HF_TOKEN serverseitig setzen).";
          }
          els.modelDialogHfToken.focus();
          return;
        }
        finish({
          hfToken: token,
          remember: Boolean(els.modelDialogHfRemember?.checked),
        });
        return;
      }
      finish({ hfToken: "", remember: true });
    };
    els.modelDialogCancel.onclick = () => finish(false);
    els.modelDialog.oncancel = (event) => {
      event.preventDefault();
      finish(false);
    };
  });
}

let activeModelDownload = null;

async function waitForModelDownload(opt) {
  if (!els.modelDialog) return false;
  resetModelDialog();
  const token = { backendId: opt.id, cancelled: false };
  activeModelDownload = token;
  els.modelDialogTitle.textContent = `${opt.name} wird heruntergeladen`;
  els.modelDialogCopy.textContent =
    "Das Modell wird geprüft und atomar im Modell-Volume installiert. Die bisherige Pipeline bleibt aktiv.";
  els.modelDialogSize.textContent = `Downloadgröße: ${formatModelSize(opt.downloadSizeBytes)}`;
  els.modelProgress.hidden = false;
  els.modelDialogConfirm.hidden = true;
  els.modelDialogCancel.textContent = "Download abbrechen";
  setModelDialogProgress(opt.downloadedBytes, opt.downloadSizeBytes);
  if (!els.modelDialog.open) els.modelDialog.showModal();

  els.modelDialog.oncancel = (event) => event.preventDefault();
  els.modelDialogCancel.onclick = async () => {
    if (token.cancelled) return;
    token.cancelled = true;
    els.modelDialogCancel.disabled = true;
    els.modelDialogCancel.textContent = "Wird abgebrochen…";
    try {
      await apiRequest(`/api/v1/models/${encodeURIComponent(opt.id)}/download`, {
        method: "DELETE",
      });
    } catch (error) {
      if (els.modelDialogError) {
        els.modelDialogError.hidden = false;
        els.modelDialogError.textContent = error.message;
      }
    }
  };

  try {
    while (!token.cancelled) {
      await new Promise((resolve) => window.setTimeout(resolve, 500));
      await loadBackendCatalog({ quiet: true });
      const current = allModelOptions().find((item) => item.id === opt.id);
      if (!current) throw new Error("Modell ist nicht mehr im Katalog vorhanden.");
      setModelDialogProgress(current.downloadedBytes, current.downloadSizeBytes);
      if (current.installed || current.downloadState === "installed") {
        if (els.modelDialog.open) els.modelDialog.close();
        showToast(`${current.name} ist installiert`);
        return true;
      }
      if (["failed", "cancelled"].includes(current.downloadState)) {
        throw new Error(
          current.downloadError ||
            (current.downloadState === "cancelled"
              ? "Download wurde abgebrochen."
              : "Download fehlgeschlagen.")
        );
      }
    }
    if (els.modelDialog.open) els.modelDialog.close();
    await loadBackendCatalog({ quiet: true });
    showToast("Download abgebrochen", { ms: 1600 });
    return false;
  } catch (error) {
    if (els.modelDialogError) {
      els.modelDialogError.hidden = false;
      els.modelDialogError.textContent = error.message;
    }
    els.modelDialogCancel.disabled = false;
    els.modelDialogCancel.textContent = "Schließen";
    els.modelDialogCancel.onclick = () => {
      if (els.modelDialog.open) els.modelDialog.close();
    };
    showToast(`Download fehlgeschlagen: ${error.message}`, { error: true, ms: 3200 });
    return false;
  } finally {
    if (activeModelDownload === token) activeModelDownload = null;
    els.modelDialog.oncancel = null;
  }
}

async function waitForModuleInstall(opt) {
  if (!els.modelDialog) return false;
  resetModelDialog();
  const token = {
    backendId: opt.id,
    cancelled: false,
    maxDownloaded: Math.max(0, Number(opt.downloadedBytes || 0)),
  };
  activeModelDownload = token;
  const totalBytes = opt.downloadSizeBytes + opt.imageDownloadSizeBytes;
  els.modelDialogTitle.textContent = `${opt.name} wird installiert`;
  els.modelDialogCopy.textContent =
    "Modell und signierte Laufzeit werden fortsetzbar installiert. Die aktive Pipeline bleibt bis zur erfolgreichen Aktivierung unverändert.";
  els.modelDialogSize.textContent =
    `Modell: ${formatModelSize(opt.downloadSizeBytes)} · Image: ${formatModelSize(opt.imageDownloadSizeBytes)}`;
  els.modelProgress.hidden = false;
  els.modelDialogConfirm.hidden = true;
  els.modelDialogCancel.textContent = "Installation abbrechen";
  setModelDialogProgress(token.maxDownloaded, totalBytes);
  if (!els.modelDialog.open) els.modelDialog.showModal();
  els.modelDialog.oncancel = (event) => event.preventDefault();
  els.modelDialogCancel.onclick = async () => {
    if (token.cancelled) return;
    token.cancelled = true;
    els.modelDialogCancel.disabled = true;
    els.modelDialogCancel.textContent = "Wird abgebrochen…";
    try {
      await apiRequest(`/api/v1/modules/${encodeURIComponent(opt.id)}/install`, {
        method: "DELETE",
      });
    } catch (error) {
      if (els.modelDialogError) {
        els.modelDialogError.hidden = false;
        els.modelDialogError.textContent = error.message;
      }
    }
  };
  try {
    while (!token.cancelled) {
      await new Promise((resolve) => window.setTimeout(resolve, 500));
      const current = await apiRequest(`/api/v1/modules/${encodeURIComponent(opt.id)}/install`);
      const downloaded = Number(current.model_downloaded_bytes || 0) +
        Number(current.image_downloaded_bytes || 0);
      const total = totalBytes || Number(current.model_total_bytes || 0) +
        Number(current.image_download_size_bytes || 0);
      token.maxDownloaded = Math.max(token.maxDownloaded, Math.min(downloaded, total));
      setModelDialogProgress(token.maxDownloaded, total);
      if (current.model_state === "installed" && current.runtime_state === "missing") {
        els.modelDialogCopy.textContent =
          "Modell installiert. Das signierte Laufzeit-Image wird vorbereitet und geladen. Die aktive Pipeline bleibt unverändert.";
        if (els.modelProgressLabel) {
          els.modelProgressLabel.textContent =
            `Modell geladen: ${formatBytes(Number(current.model_downloaded_bytes || 0))} · Laufzeit-Image wird geladen`;
        }
      }
      if (current.state === "ready") {
        if (els.modelDialog.open) els.modelDialog.close();
        await loadBackendCatalog({ quiet: true });
        showToast(`${opt.name} ist installiert und aktivierbar`);
        return true;
      }
      if (["failed", "cancelled", "unavailable", "needs_configuration", "needs_credentials", "unhealthy", "host_module_delivery_pending"].includes(current.state)) {
        throw new Error(current.error || `Laufzeitstatus: ${current.state}`);
      }
    }
    if (els.modelDialog.open) els.modelDialog.close();
    await loadBackendCatalog({ quiet: true });
    showToast("Installation abgebrochen", { ms: 1600 });
    return false;
  } catch (error) {
    if (els.modelDialogError) {
      els.modelDialogError.hidden = false;
      els.modelDialogError.textContent = error.message;
    }
    els.modelDialogCancel.disabled = false;
    els.modelDialogCancel.textContent = "Schließen";
    els.modelDialogCancel.onclick = () => {
      if (els.modelDialog.open) els.modelDialog.close();
    };
    showToast(`Installation fehlgeschlagen: ${error.message}`, { error: true, ms: 3200 });
    return false;
  } finally {
    if (activeModelDownload === token) activeModelDownload = null;
    els.modelDialog.oncancel = null;
  }
}

async function ensureModuleReady(opt) {
  if (!opt) {
    showToast("Modell ist nicht im Backend-Katalog vorhanden", { error: true, ms: 2800 });
    return false;
  }
  if (opt.activatable) return true;
  if (!opt.compatible) {
    showToast(opt.note || "Auf diesem System ist keine veröffentlichte Variante kompatibel", {
      error: true,
      ms: 3200,
    });
    return false;
  }
  if (opt.hostManaged) {
    if (["host_module_delivery_pending", "unavailable"].includes(opt.runtimeState)) {
      showToast(opt.runtimeReason || `Laufzeitstatus: ${opt.runtimeState}`, { error: true, ms: 3200 });
      return false;
    }
    if (opt.installed) return true;
    const needsHf = Boolean(opt.authRequired || opt.accessUrl);
    const approved = await confirmModelAction({
      title: "Speech-Lab-Modell herunterladen",
      message: `${opt.name} wird für den Windows-Host installiert. Lizenz: ${opt.licenses.join(" + ") || "siehe Modellkarte"}.`,
      size: `Modell: ${formatModelSize(opt.downloadSizeBytes)}`,
      confirmLabel: "OK · herunterladen",
      hfAuth: needsHf ? { accessUrl: opt.accessUrl || "", tokenConfigured: Boolean(opt.hfTokenConfigured) } : null,
    });
    if (!approved) return false;
    try {
      await apiRequest(`/api/v1/models/${encodeURIComponent(opt.id)}/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hf_token: approved.hfToken, remember: approved.remember }),
      });
    } catch (error) {
      showToast(`Download konnte nicht gestartet werden: ${error.message}`, { error: true, ms: 3200 });
      return false;
    }
    return waitForModelDownload(opt);
  }
  if (!opt.managedRuntime || ["host_module_delivery_pending", "needs_configuration", "needs_credentials", "unhealthy", "unavailable"].includes(opt.runtimeState)) {
    showToast(opt.runtimeReason || `Laufzeitstatus: ${opt.runtimeState}`, { error: true, ms: 3200 });
    return false;
  }
  const needsHf = Boolean(opt.authRequired || opt.accessUrl);
  const licenseNote = opt.licenses.length ? ` Lizenz: ${opt.licenses.join(" + ")}.` : "";
  const approved = await confirmModelAction({
    title: "Speech-Lab-Modul installieren",
    message:
      `${opt.name} installiert nur die ausgewählte Variante „${opt.variantId}“.` + licenseNote,
    size:
      `Modell: ${formatModelSize(opt.downloadSizeBytes)} · ` +
      `Image: ${formatModelSize(opt.imageDownloadSizeBytes)}`,
    confirmLabel: "OK · installieren",
    hfAuth: needsHf
      ? {
          accessUrl: opt.accessUrl || "",
          tokenConfigured: Boolean(opt.hfTokenConfigured),
        }
      : null,
  });
  if (!approved) return false;
  try {
    const body = { catalog_revision: CATALOG_REVISION };
    if (needsHf) {
      if (approved.hfToken) body.hf_token = approved.hfToken;
      body.remember = approved.remember !== false;
    }
    await apiRequest(`/api/v1/modules/${encodeURIComponent(opt.id)}/install`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    showToast(`Installation konnte nicht gestartet werden: ${error.message}`, {
      error: true,
      ms: 3200,
    });
    return false;
  }
  return waitForModuleInstall(opt);
}

async function deleteInstalledModel(opt) {
  if (!opt || (!opt.installed && !["ready", "running"].includes(opt.runtimeState))) return;
  const approved = await confirmModelAction({
    title: "Modell löschen",
    message: `${opt.name} wird deinstalliert. Aktive Module können nicht entfernt werden.`,
    size:
      `Modell: ${formatModelSize(opt.downloadSizeBytes)} · ` +
      `Image-Laufzeit: ${formatModelSize(opt.imageDownloadSizeBytes)}`,
    confirmLabel: "Modul deinstallieren",
    danger: true,
  });
  if (!approved) return;
  try {
    await apiRequest(`/api/v1/modules/${encodeURIComponent(opt.id)}`, { method: "DELETE" });
    await loadBackendCatalog({ quiet: true });
    showToast(`${opt.name} wurde deinstalliert`);
  } catch (error) {
    showToast(`Löschen nicht möglich: ${error.message}`, { error: true, ms: 3200 });
  }
}

function modelStatus(opt) {
  if (!opt.compatible) return opt.incompatibilityLabel || "Auf diesem System nicht verfügbar";
  if (opt.activatable) {
    return `Bereit · Modell ${formatModelSize(opt.downloadSizeBytes)} · Image ${formatModelSize(opt.imageDownloadSizeBytes)}`;
  }
  if (opt.compatible && opt.managedRuntime && !["unavailable", "needs_configuration", "host_module_delivery_pending"].includes(opt.runtimeState)) {
    return `Installation nötig · Modell ${formatModelSize(opt.downloadSizeBytes)} · Image ${formatModelSize(opt.imageDownloadSizeBytes)}`;
  }
  if (opt.bundled) return "Im Container enthalten";
  if (opt.downloadState === "downloading") {
    const percent =
      opt.downloadSizeBytes > 0
        ? Math.round((opt.downloadedBytes / opt.downloadSizeBytes) * 100)
        : 0;
    return `Download ${percent} %`;
  }
  if (opt.installed) return `Installiert · ${formatModelSize(opt.downloadSizeBytes)}`;
  if (opt.downloadState === "failed") return "Download fehlgeschlagen";
  return `Download nötig · ${formatModelSize(opt.downloadSizeBytes)}`;
}

function isRecommendedOption(opt) {
  if (!opt?.id) return false;
  if (opt.stage === "asr") return RECOMMENDED.asr.has(opt.id);
  if (opt.stage === "tts") return RECOMMENDED.tts.has(opt.id);
  // Fallback: engine lists may omit stage on some paths
  return RECOMMENDED.asr.has(opt.id) || RECOMMENDED.tts.has(opt.id);
}

function choiceButton(opt, selectedId) {
  const sel = opt.id === selectedId;
  const avail = opt.available !== false;
  const downloadOnly = !avail && opt.compatible && opt.managedRuntime && !["unavailable", "needs_configuration", "host_module_delivery_pending"].includes(opt.runtimeState);
  const selectable = avail || downloadOnly;
  const recommended = isRecommendedOption(opt);
  return `
    <div class="choice-wrap">
    <button type="button" class="choice choice-select${
      avail ? "" : downloadOnly ? " download-only" : " disabled"
    }${recommended ? " recommended" : ""}"
      role="option" data-id="${opt.id}"
      aria-selected="${sel ? "true" : "false"}"
      data-available="${avail ? "true" : "false"}"
      data-compatible="${opt.compatible ? "true" : "false"}"
      data-installed="${opt.installed ? "true" : "false"}"
      ${selectable ? "" : "disabled aria-disabled=\"true\""}
      ${
        avail
          ? recommended
            ? 'title="Voraussichtlich gut für dieses System (Heuristik)"'
            : ""
          : `title="${
              downloadOnly
                ? "Download möglich; Aktivierung nicht verfügbar"
                : "Auf diesem System nicht verfügbar"
            }"`
      }>
      <div class="choice-top">
        <span class="choice-name">${opt.name}${
          recommended ? '<span class="choice-rec-badge">Empfohlen</span>' : ""
        }</span>
        <span class="choice-tag">${opt.tag}</span>
      </div>
      <div class="choice-desc">${opt.desc}</div>
      ${languageLabelsBlock(opt.languageLabels)}
      ${gpuSupportBlock(opt.gpuSupport)}
      ${demandStarsBlock(opt.stars || resourceStars(null))}
      <span class="choice-vram">~${opt.vramGb.toFixed(1)} GB GPU · ${opt.meta}</span>
      <span class="choice-install-state" data-state="${opt.downloadState}">${modelStatus(opt)}</span>
    </button>
    ${
      (opt.deletable && opt.installed) || ["ready", "running"].includes(opt.runtimeState)
        ? `<button type="button" class="choice-delete" data-delete-id="${opt.id}"
            ${sel ? "disabled title=\"Aktives Modell kann nicht gelöscht werden\"" : ""}
            aria-label="${opt.name} löschen">Löschen</button>`
        : ""
    }
    </div>`;
}

function renderChoices(container, options, selectedId, onPick) {
  if (!container) return;
  const ordered = [...options].sort((a, b) => Number(b.compatible) - Number(a.compatible));
  container.innerHTML = ordered.map((o) => choiceButton(o, selectedId)).join("");
  container.querySelectorAll(".choice-select").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (btn.dataset.available === "false" && btn.dataset.compatible !== "true") {
        const opt = options.find((item) => item.id === btn.dataset.id);
        log(`„${btn.dataset.id}“ kann auf diesem System nicht aktiviert werden`);
        showToast(opt?.note || "Auf diesem System nicht verfügbar", {
          error: true,
          ms: 3000,
        });
        return;
      }
      await onPick(btn.dataset.id);
    });
  });
  container.querySelectorAll(".choice-delete").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const opt = options.find((item) => item.id === btn.dataset.deleteId);
      await deleteInstalledModel(opt);
    });
  });
}

function renderPresets() {
  if (!els.presetChips) return;
  els.presetChips.innerHTML = Object.entries(PRESETS)
    .map(([id, p]) => {
      const recommended = RECOMMENDED.presets.has(id);
      return `<button type="button" class="chip${
        recommended ? " recommended" : ""
      }" data-id="${id}" aria-pressed="${
        lab.preset === id ? "true" : "false"
      }" title="${
        recommended ? "Voraussichtlich gut für dieses System (Heuristik)" : ""
      }">${p.label}${
        recommended ? '<span class="chip-rec">Empfohlen</span>' : ""
      }</button>`;
    })
    .join("");
  els.presetChips.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => applyPreset(chip.dataset.id));
  });
}

function voicesForTts(tts) {
  if (!tts) return [];
  const voices = Array.isArray(tts.voices) ? tts.voices.filter(Boolean) : [];
  if (voices.length) return voices;
  return tts.defaultVoice ? [tts.defaultVoice] : [];
}

function syncVoiceForTts(tts, { forceDefault = false } = {}) {
  const voices = voicesForTts(tts);
  const current = voices.find(
    (voice) => !forceDefault && voice.toLowerCase() === String(lab.voice).toLowerCase()
  );
  lab.voice = current || tts?.defaultVoice || voices[0] || "";
}

function renderVoicePicker(tts) {
  if (!els.voiceField || !els.voiceSelect) return;
  const voices = voicesForTts(tts);
  syncVoiceForTts(tts);
  els.voiceField.hidden = voices.length < 2;
  els.voiceSelect.replaceChildren(
    ...voices.map((voice) => {
      const option = document.createElement("option");
      option.value = voice;
      option.textContent = voice.replaceAll("_", " ");
      return option;
    })
  );
  els.voiceSelect.value = lab.voice;
  // Never leave the control disabled from stale swap state after re-render.
  els.voiceSelect.disabled = false;
  if (els.voiceHint) {
    const behavior =
      tts?.voiceMode === "restart"
        ? "Der TTS-Sidecar wird beim Wechsel kontrolliert neu gestartet."
        : "Die Auswahl gilt ab der nächsten Ausgabe.";
    els.voiceHint.textContent = `${voices.length} Stimmen für ${tts?.name || "TTS"}. ${behavior}`;
  }
}

async function applyPreset(id) {
  const p = PRESETS[id];
  if (!p) return;
  const targets = [
    ASR_OPTIONS.find((item) => item.id === p.asr),
    TTS_OPTIONS.find((item) => item.id === p.tts),
    LLM_OPTIONS.find((item) => item.id === p.llm),
  ];
  for (const opt of targets) {
    if (!(await ensureModuleReady(opt))) return;
    const current = allModelOptions().find((item) => item.id === opt?.id);
    if (!current?.available) {
      showToast(`${current?.name || opt?.id} ist installiert, aber nicht aktivierbar`, {
        error: true,
        ms: 3200,
      });
      return;
    }
  }
  lab.preset = id;
  lab.asr = p.asr;
  lab.tts = p.tts;
  lab.llm = p.llm;
  syncVoiceForTts(TTS_OPTIONS.find((item) => item.id === lab.tts), { forceDefault: true });
  persistLab();
  refreshLabUi(p.compose);
  sendStackToBackend();
  log(`Preset: ${p.label}`);
  showToast(`Preset: ${p.label}`);
}

function persistLab() {
  localStorage.setItem("s2s.lab.asr", lab.asr);
  localStorage.setItem("s2s.lab.tts", lab.tts);
  localStorage.setItem("s2s.lab.llm", lab.llm);
  localStorage.setItem("s2s.lab.voice", lab.voice);
  localStorage.setItem("s2s.lab.preset", lab.preset);
}

function mergeEnv(asr, tts, llm) {
  return { ...(asr.env || {}), ...(llm.env || {}), ...(tts.env || {}) };
}

function refreshLabUi(composeOverride) {
  const asr = findOpt(ASR_OPTIONS, lab.asr);
  const tts = findOpt(TTS_OPTIONS, lab.tts);
  const llm = findOpt(LLM_OPTIONS, lab.llm);

  lab.asr = asr.id;
  lab.tts = tts.id;
  lab.llm = llm.id;
  syncVoiceForTts(tts);

  renderChoices(els.asrChoices, ASR_OPTIONS, lab.asr, async (id) => {
    if (!(await ensureModuleReady(ASR_OPTIONS.find((item) => item.id === id)))) return;
    const current = ASR_OPTIONS.find((item) => item.id === id);
    if (!current?.available) {
      showToast(`${current?.name || id} ist installiert, aber nicht aktivierbar`, {
        error: true,
        ms: 3000,
      });
      return;
    }
    lab.asr = id;
    lab.preset = "custom";
    persistLab();
    refreshLabUi();
    sendStackToBackend();
  });
  renderChoices(els.ttsChoices, TTS_OPTIONS, lab.tts, async (id) => {
    if (!(await ensureModuleReady(TTS_OPTIONS.find((item) => item.id === id)))) return;
    const current = TTS_OPTIONS.find((item) => item.id === id);
    if (!current?.available) {
      showToast(`${current?.name || id} ist installiert, aber nicht aktivierbar`, {
        error: true,
        ms: 3000,
      });
      return;
    }
    lab.tts = id;
    syncVoiceForTts(current, { forceDefault: true });
    lab.preset = "custom";
    persistLab();
    refreshLabUi();
    sendStackToBackend();
  });
  renderChoices(els.llmChoices, LLM_OPTIONS, lab.llm, async (id) => {
    if (!(await ensureModuleReady(LLM_OPTIONS.find((item) => item.id === id)))) return;
    const current = LLM_OPTIONS.find((item) => item.id === id);
    if (!current?.available) {
      showToast(`${current?.name || id} ist installiert, aber nicht aktivierbar`, {
        error: true,
        ms: 3000,
      });
      return;
    }
    lab.llm = id;
    lab.preset = "custom";
    persistLab();
    refreshLabUi();
    sendStackToBackend();
  });

  if (els.pipeAsrName) els.pipeAsrName.textContent = shortName(asr.name);
  if (els.pipeAsrMeta) els.pipeAsrMeta.textContent = asr.meta;
  if (els.pipeLlmName) els.pipeLlmName.textContent = shortName(llm.name);
  if (els.pipeLlmMeta) els.pipeLlmMeta.textContent = llm.meta;
  if (els.pipeTtsName) els.pipeTtsName.textContent = shortName(tts.name);
  if (els.pipeTtsMeta) els.pipeTtsMeta.textContent = tts.meta;
  renderVoicePicker(tts);

  setBadge(els.asrBadge, asr.available !== false ? "live" : "soon", asr.available === false);
  setBadge(els.ttsBadge, tts.available !== false ? "live" : "soon", tts.available === false);
  setBadge(els.llmBadge, llm.available !== false ? "live" : "soon", llm.available === false);

  const total = asr.vramGb + tts.vramGb + llm.vramGb;
  const pct = Math.min(100, (total / VRAM_BUDGET_GB) * 100);
  const warn = total > VRAM_BUDGET_GB * 0.85;
  if (els.budgetFill) {
    els.budgetFill.style.width = `${pct}%`;
    els.budgetFill.classList.toggle("warn", warn);
  }
  if (els.stackBudget) {
    els.stackBudget.textContent = `~${total.toFixed(1)} / ${VRAM_BUDGET_GB} GB GPU (geschätzt)`;
    els.stackBudget.classList.toggle("warn", warn);
  }
  if (els.labBadge) {
    els.labBadge.hidden = !warn;
    els.labBadge.title = warn ? "VRAM-Budget hoch" : "";
  }

  const env = mergeEnv(asr, tts, llm);
  const envLines = Object.entries(env)
    .map(([k, v]) => `${k}=${v}`)
    .join("\n");
  const notes = [asr.note, tts.note, llm.note].filter(Boolean).join(" · ");
  const compose =
    composeOverride ||
    (PRESETS[lab.preset] && PRESETS[lab.preset].compose) ||
    "Custom-Stack · Hot-swap per WebSocket.";
  if (els.stackHint) {
    els.stackHint.textContent = `${compose}\n\n.env / runtime:\n${envLines}${
      notes ? `\n\n${notes}` : ""
    }\n\nHot-swap: Auswahl wird live an s2s gesendet.`;
  }

  if (els.presetChips) {
    els.presetChips.querySelectorAll(".chip").forEach((c) => {
      c.setAttribute("aria-pressed", c.dataset.id === lab.preset ? "true" : "false");
    });
  }
}

function shortName(name) {
  return name
    .replace("faster-whisper ", "fw ")
    .replace("whisper.cpp ", "w.cpp ")
    .replace("Granite 3.3 ", "Granite ")
    .replace("Qwen3-TTS ", "Qwen ")
    .replace(" ONNX", "")
    .slice(0, 22);
}

function setBadge(el, text, isWarn, soft) {
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("warn", !!isWarn);
  el.classList.toggle("soft", !!soft && !isWarn);
}

// ── Init defaults ───────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
function defaultWsUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const sameOrigin = `${scheme}://${location.host}${labPath('/ws')}`;
  if (LAB_BASE_PATH !== '/') return sameOrigin;
  if (params.get("ws")) return params.get("ws");
  const saved = localStorage.getItem("s2s.ws");
  const legacyDirect =
    /^ws:\/\/(?:127\.0\.0\.1|localhost):8765\/?$/.test(saved || "") &&
    location.port !== "8765";
  if (saved && !legacyDirect) return saved;
  if (legacyDirect) localStorage.removeItem("s2s.ws");
  if (
    location.protocol === "http:" &&
    location.hostname !== "127.0.0.1" &&
    location.hostname !== "localhost"
  ) {
    console.warn(
      "Page is not HTTPS — microphone will be blocked on remote devices. Use: python serve.py"
    );
  }
  return sameOrigin;
}

if (els.wsUrl) els.wsUrl.value = defaultWsUrl();
if (els.wsUrl && LAB_BASE_PATH !== '/') els.wsUrl.disabled = true;
if (els.talkMode) {
  els.talkMode.value = localStorage.getItem("s2s.talkMode") || "hold";
  state.talkMode = els.talkMode.value;
}

refreshUi();
loadBackendCatalog().catch(() => {});

function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  if (els.log) {
    els.log.textContent = `${line}\n${els.log.textContent}`.slice(0, 4000);
  }
  console.log(msg);
}

function setCssLevels() {
  const l = Math.max(state.micLevel, state.outLevel * 0.9);
  document.documentElement.style.setProperty("--level", l.toFixed(3));
  document.documentElement.style.setProperty("--hotness", state.talking ? "1" : "0");
  if (els.barMic) els.barMic.style.width = `${Math.min(100, state.micLevel * 120)}%`;
  if (els.barOut) els.barOut.style.width = `${Math.min(100, state.outLevel * 120)}%`;
}

// ── Visual engine ───────────────────────────────────────────────────
const vis = {
  particles: [],
  rings: [],
  t: 0,
  last: performance.now(),
  frames: 0,
  fpsT: 0,
  pageVisible: !document.hidden,
  raf: 0,
};

function initParticles(n = 64) {
  vis.particles = Array.from({ length: n }, (_, i) => ({
    a: (i / n) * Math.PI * 2,
    r: 0.18 + Math.random() * 0.55,
    s: 0.15 + Math.random() * 0.9,
    size: 0.8 + Math.random() * 2.2,
    hue: Math.random() < 0.33 ? 190 : Math.random() < 0.5 ? 265 : 330,
  }));
}

function resizeCanvas() {
  if (!els.canvas) return null;
  const hot = state.connected || state.talking || state.speaking;
  const dpr = Math.min(window.devicePixelRatio || 1, hot ? 1.75 : 1.25);
  const w = window.innerWidth;
  const h = window.innerHeight;
  els.canvas.width = Math.floor(w * dpr);
  els.canvas.height = Math.floor(h * dpr);
  els.canvas.style.width = `${w}px`;
  els.canvas.style.height = `${h}px`;
  const c = els.canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return c;
}

let ctx = resizeCanvas();
window.addEventListener("resize", () => {
  ctx = resizeCanvas();
});
initParticles(64);

function isVisHot() {
  return (
    state.talking ||
    state.speaking ||
    state.playNodes > 0 ||
    state.micLevel > 0.04 ||
    state.outLevel > 0.04
  );
}

function scheduleNextFrame() {
  if (!vis.pageVisible) {
    vis.raf = 0;
    return;
  }
  const hot = isVisHot();
  if (!state.connected && !hot) {
    vis.raf = window.setTimeout(() => {
      vis.raf = 0;
      requestAnimationFrame(drawFrame);
    }, 120);
    return;
  }
  if (state.connected && !hot) {
    vis.raf = window.setTimeout(() => {
      vis.raf = 0;
      requestAnimationFrame(drawFrame);
    }, 50);
    return;
  }
  vis.raf = requestAnimationFrame(drawFrame);
}

function drawFrame(now) {
  if (!ctx) {
    scheduleNextFrame();
    return;
  }
  const dt = Math.min(0.05, (now - vis.last) / 1000);
  vis.last = now;
  vis.t += dt;
  vis.frames++;
  if (now - vis.fpsT > 500) {
    if (els.fps) {
      els.fps.textContent = `${Math.round((vis.frames * 1000) / (now - vis.fpsT))} fps`;
    }
    vis.frames = 0;
    vis.fpsT = now;
  }

  const w = window.innerWidth;
  const h = window.innerHeight;
  const cx = w * 0.5;
  const cy = h * 0.48;
  const energy = Math.max(state.micLevel, state.outLevel);
  const boost = state.talking ? 1.35 : state.speaking ? 1.15 : 0.85;
  const hot = isVisHot();

  if (hot) {
    ctx.fillStyle = "rgba(5, 6, 12, 0.22)";
    ctx.fillRect(0, 0, w, h);
  } else {
    ctx.fillStyle = "rgba(5, 6, 12, 1)";
    ctx.fillRect(0, 0, w, h);
  }

  const g = ctx.createRadialGradient(cx, cy, 20, cx, cy, Math.max(w, h) * 0.55);
  g.addColorStop(0, `rgba(92, 225, 255, ${0.03 + energy * 0.12})`);
  g.addColorStop(0.35, `rgba(167, 139, 250, ${0.04 + energy * 0.08})`);
  g.addColorStop(1, "rgba(5,6,12,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  const baseR = Math.min(w, h) * 0.16;
  const segs = hot ? 72 : 36;
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
  if (hot) {
    ctx.shadowColor = state.talking ? "#ff4d8d" : "#5ce1ff";
    ctx.shadowBlur = 8 + energy * 18;
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  if (hot) {
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
  }

  const particleBudget =
    !state.connected && !hot ? 0 : hot ? vis.particles.length : Math.min(24, vis.particles.length);
  for (let i = 0; i < particleBudget; i++) {
    const p = vis.particles[i];
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
  state.micLevel *= 0.92;
  state.outLevel *= 0.9;
  if (state.outLevel < 0.02 && state.playNodes === 0) {
    if (state.speaking) {
      state.speaking = false;
      if (!state.talking && state.connected && state.processing) {
        // keep processing until response_done
      } else if (!state.talking && state.connected) {
        state.processing = false;
      }
      refreshUi();
    }
  }

  scheduleNextFrame();
}

document.addEventListener("visibilitychange", () => {
  vis.pageVisible = !document.hidden;
  if (vis.pageVisible && !vis.raf) {
    requestAnimationFrame(drawFrame);
  } else if (!vis.pageVisible && vis.raf) {
    cancelAnimationFrame(vis.raf);
    clearTimeout(vis.raf);
    vis.raf = 0;
  }
  // Tab focus: if we wanted a session and are down, try sooner.
  if (
    vis.pageVisible &&
    state.wantConnected &&
    !state.connected &&
    !state.connecting
  ) {
    clearReconnectTimers();
    connect({ auto: true });
  }
});

// Browser online again → resume reconnect quickly.
window.addEventListener("online", () => {
  if (state.wantConnected && !state.connected && !state.connecting) {
    log("Network online — reconnecting");
    clearReconnectTimers();
    connect({ auto: true });
  }
});
window.addEventListener("offline", () => {
  if (state.wantConnected) {
    log("Network offline — waiting for connectivity");
    setHint("Netzwerk offline — warte auf Verbindung…", { sticky: true });
  }
});

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

function sendPcmFrames(frames) {
  if (!state.talking || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  for (const buf of frames) {
    state.ws.send(buf);
  }
}

async function ensureWorklet(actx) {
  if (state.workletReady) return true;
  if (!actx.audioWorklet || typeof actx.audioWorklet.addModule !== "function") {
    return false;
  }
  try {
    // Relative URL works for HTTPS lab UI and plain static serve.
    await actx.audioWorklet.addModule(new URL("pcm-worklet.js", location.href).href);
    state.workletReady = true;
    return true;
  } catch (e) {
    log(`AudioWorklet load failed — ScriptProcessor fallback: ${e.message || e}`);
    return false;
  }
}

async function startCaptureWorklet(actx) {
  state.workletNode = new AudioWorkletNode(actx, "pcm-capture", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    channelCount: 1,
    processorOptions: {
      targetRate: SAMPLE_RATE,
      frameSamples: FRAME_SAMPLES,
    },
  });
  state.workletNode.port.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.type === "frames") {
      if (typeof msg.level === "number") {
        state.micLevel = Math.min(1, msg.level * 4.5);
      }
      if (state.talking) sendPcmFrames(msg.frames || []);
    } else if (msg.type === "level" && typeof msg.level === "number") {
      state.micLevel = Math.min(1, msg.level * 4.5);
    }
  };
  state.workletNode.port.postMessage({ type: "arm", value: true });

  const mute = actx.createGain();
  mute.gain.value = 0;
  state.source.connect(state.workletNode);
  state.workletNode.connect(mute);
  mute.connect(actx.destination);
  log("Microphone capture started (AudioWorklet)");
}

function startCaptureScriptProcessor(actx) {
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
    const frames = [];
    while (offset + FRAME_SAMPLES <= merged.length) {
      const slice = merged.subarray(offset, offset + FRAME_SAMPLES);
      frames.push(floatTo16BitPCM(slice).buffer);
      offset += FRAME_SAMPLES;
    }
    leftover = merged.subarray(offset);
    sendPcmFrames(frames);
  };

  const mute = actx.createGain();
  mute.gain.value = 0;
  state.source.connect(state.processor);
  state.processor.connect(mute);
  mute.connect(actx.destination);
  log("Microphone capture started (ScriptProcessor fallback)");
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
  if (state.source || state.workletNode || state.processor) return;

  state.source = actx.createMediaStreamSource(state.mediaStream);

  const useWorklet = await ensureWorklet(actx);
  if (useWorklet) {
    try {
      await startCaptureWorklet(actx);
      return;
    } catch (e) {
      log(`AudioWorklet node failed — fallback: ${e.message || e}`);
      try {
        state.workletNode?.disconnect();
      } catch (_) {}
      state.workletNode = null;
    }
  }
  startCaptureScriptProcessor(actx);
}

function stopCaptureTracks() {
  if (state.workletNode) {
    try {
      state.workletNode.port.postMessage({ type: "arm", value: false });
      state.workletNode.port.onmessage = null;
      state.workletNode.disconnect();
    } catch (_) {}
    state.workletNode = null;
  }
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

function releaseMicrophone() {
  stopCaptureTracks();
  if (state.mediaStream) {
    for (const t of state.mediaStream.getTracks()) {
      try {
        t.stop();
      } catch (_) {}
    }
    state.mediaStream = null;
  }
  if (state.audioCtx) {
    state.audioCtx.suspend().catch(() => {});
  }
  state.playTime = 0;
  state.playNodes = 0;
  state.speaking = false;
  state.ttsActive = false;
  state.micLevel = 0;
  state.outLevel = 0;
}

function playPcmI16(arrayBuffer) {
  if (!state.audioCtx) return;
  const actx = state.audioCtx;
  const i16 = new Int16Array(arrayBuffer);
  if (!i16.length) {
    if (state.playNodes === 0) {
      state.speaking = false;
      state.ttsActive = false;
      refreshUi();
    }
    return;
  }

  let sum = 0;
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) {
    f32[i] = i16[i] / 32768;
    sum += f32[i] * f32[i];
  }
  state.outLevel = Math.min(1, Math.sqrt(sum / i16.length) * 4);
  state.speaking = true;
  state.ttsActive = true;
  state.processing = false;
  setPipeActive("tts");
  clearHintLock();
  refreshUi();

  const buf = actx.createBuffer(1, f32.length, SAMPLE_RATE);
  buf.copyToChannel(f32, 0);
  const src = actx.createBufferSource();
  src.buffer = buf;
  const gain = actx.createGain();
  const now = actx.currentTime;
  const lead = 0.03;
  let startAt = state.playTime;
  if (startAt < now + 0.005) {
    startAt = now + lead;
    gain.gain.setValueAtTime(0, startAt);
    gain.gain.linearRampToValueAtTime(1, startAt + 0.012);
  }
  src.connect(gain);
  gain.connect(actx.destination);
  src.start(startAt);
  state.playTime = startAt + buf.duration;
  state.playNodes++;
  src.onended = () => {
    state.playNodes = Math.max(0, state.playNodes - 1);
    if (state.playNodes === 0) {
      state.speaking = false;
      state.ttsActive = false;
      if (!state.talking) {
        state.processing = false;
        clearHintLock();
      }
      refreshUi();
    }
  };
}

// ── WebSocket ───────────────────────────────────────────────────────
/**
 * @param {{ auto?: boolean }} [opts] auto=true when called from reconnect scheduler
 */
function connect(opts = {}) {
  const auto = !!opts.auto;
  const url = (LAB_BASE_PATH !== '/' ? defaultWsUrl() : (els.wsUrl?.value || defaultWsUrl())).trim();
  if (!url) return;

  state.wantConnected = true;
  clearReconnectTimers();

  if (!(location.protocol === "https:" && url.endsWith("/ws"))) {
    localStorage.setItem("s2s.ws", url);
  }
  if (location.protocol === "https:" && url.startsWith("ws://")) {
    log("WARN: page is HTTPS but WS URL is ws:// — browser may block mixed content. Use wss://…/ws");
  }
  if (
    location.protocol === "http:" &&
    location.hostname !== "localhost" &&
    location.hostname !== "127.0.0.1"
  ) {
    log("WARN: microphone requires HTTPS (or localhost). Start UI with: python serve.py");
  }

  // Drop previous socket without treating it as a user disconnect / reconnect trigger.
  if (state.ws) {
    const old = state.ws;
    try {
      old.onopen = null;
      old.onclose = null;
      old.onerror = null;
      old.onmessage = null;
      old.close();
    } catch (_) {}
    state.ws = null;
  }

  log(`${auto ? "Reconnect" : "Connecting"} ${url} …`);
  state.connecting = true;
  state.error = false;
  state.connected = false;
  if (!auto) clearHintLock();
  refreshUi();

  let ws;
  try {
    ws = new WebSocket(url);
  } catch (e) {
    log(`WebSocket create failed: ${e.message || e}`);
    state.connecting = false;
    state.error = true;
    refreshUi();
    scheduleReconnect("create failed");
    return;
  }
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onopen = async () => {
    if (state.ws !== ws) return;
    const wasReconnect = reconnect.attempts > 0 || reconnect.hadSession;
    state.connected = true;
    state.connecting = false;
    state.error = false;
    state.processing = false;
    state.wantConnected = true;
    reconnect.attempts = 0;
    reconnect.notifiedDrop = false;
    clearReconnectTimers();
    clearHintLock();
    refreshUi();
    log("WebSocket open");
    showToast(wasReconnect && reconnect.hadSession ? copy.toastReconnected : copy.toastConnected);
    reconnect.hadSession = true;
    try {
      await ensureAudio();
    } catch (e) {
      log(`AudioContext: ${e.message || e}`);
    }
  };

  ws.onclose = (ev) => {
    if (state.ws !== ws) return; // superseded
    state.ws = null;
    const wasLive = state.connected;
    const wasConnecting = state.connecting;
    state.connected = false;
    state.connecting = false;
    state.talking = false;
    state.processing = false;
    state.speaking = false;
    releaseMicrophone();
    const why = ev.reason ? ` — ${ev.reason}` : "";
    log(`WebSocket closed (code ${ev.code}${why})`);
    if (ev.code === 1006) {
      log("HINT: abnormal close — is s2s-vulkan on :8765 up? Check https://…/health");
    } else if (ev.code === 1011) {
      log("HINT: proxy could not reach backend (start s2s-vulkan)");
    }

    // Intentional client close (Trennen) uses 1000 + wantConnected=false.
    if (!state.wantConnected) {
      state.error = false;
      clearHintLock();
      refreshUi();
      return;
    }

    // Unexpected drop or failed connect while user still wants session → auto-reconnect.
    state.error = true;
    if ((wasLive || wasConnecting) && !reconnect.notifiedDrop) {
      reconnect.notifiedDrop = true;
      if (wasLive) showToast(copy.toastDisconnected, { ms: 1600 });
    }
    clearHintLock();
    refreshUi();
    scheduleReconnect(wasLive ? "dropped" : `close ${ev.code}`);
  };

  ws.onerror = () => {
    if (state.ws !== ws) return;
    log("WebSocket error (see close code next)");
  };

  ws.onmessage = (ev) => {
    if (state.ws !== ws) return;
    if (ev.data instanceof ArrayBuffer) {
      playPcmI16(ev.data);
    } else if (typeof ev.data === "string") {
      handlePipelineEvent(ev.data);
    }
  };
}

function setLiveText(el, text, placeholder) {
  if (!el) return;
  const t = (text || "").trim();
  if (!t) {
    el.textContent = placeholder || "—";
    el.classList.add("placeholder");
    return;
  }
  el.textContent = t;
  el.classList.remove("placeholder");
}

function handlePipelineEvent(raw) {
  let msg;
  try {
    msg = JSON.parse(raw);
  } catch {
    log(`← ${raw.slice(0, 120)}`);
    return;
  }
  const type = msg.type || "";
  if (type === "final_transcript" || type === "partial_transcript") {
    setLiveText(els.liveAsrT, msg.text, copy.liveAsrEmpty);
    setLiveText(els.liveLlmT, "", copy.liveLlmWait);
    log(`ASR: ${msg.text}`);
    state.processing = true;
    setPipeActive("asr");
    setHint(copy.hintAsr(msg.text), { sticky: true });
    refreshUi();
    return;
  }
  if (type === "llm_chunk") {
    const prev = els.liveLlmT?.classList.contains("placeholder")
      ? ""
      : els.liveLlmT?.textContent || "";
    const next =
      (prev && prev !== copy.liveLlmEmpty && prev !== copy.liveLlmWait ? prev + " " : "") +
      (msg.text || "");
    setLiveText(els.liveLlmT, next, copy.liveLlmEmpty);
    state.processing = true;
    setPipeActive("llm");
    log(`LLM: ${msg.text}`);
    refreshUi();
    return;
  }
  if (type === "llm_full") {
    setLiveText(els.liveLlmT, msg.text, copy.liveLlmEmpty);
    state.processing = true;
    setPipeActive("llm");
    log(`LLM full: ${(msg.text || "").slice(0, 160)}`);
    refreshUi();
    return;
  }
  if (type === "response_done") {
    log("response done");
    if (!state.speaking && !state.talking) {
      state.processing = false;
      clearHintLock();
      setPipeActive(null);
      refreshUi();
    }
    return;
  }
  if (type === "metrics") {
    const values = msg.values || {};
    const rendered = Object.entries(values)
      .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
      .map(([key, value]) => `${key}=${value.toFixed(key.includes("rtf") ? 3 : 1)}`)
      .join(" · ");
    log(`METRICS ${(msg.stage || "turn").toUpperCase()}: ${rendered}`);
    return;
  }
  if (type === "stack_transition") {
    const stage = (msg.stage || "backend").toUpperCase();
    log(`${stage}: ${msg.phase} · ${msg.backend_id || ""} ${msg.message || ""}`);
    const terminal = ["ready", "failed", "rollback", "idle"].includes(msg.phase);
    // Intermediate phases only mark busy for the stack UI — never permanently
    // lock the voice picker (stale "draining" was leaving it disabled).
    if (terminal) {
      clearSwapBusy();
    } else {
      lab.swapBusy = true;
      // Keep voice select interactive unless a local sendStackToBackend locked it.
    }
    setHint(`${stage}: ${msg.message || msg.phase}`, {
      sticky: !terminal,
      error: msg.phase === "failed",
    });
    return;
  }
  if (type === "backend_health") {
    log(
      `${(msg.stage || "backend").toUpperCase()} health ${msg.ok ? "OK" : "FEHLER"}: ${
        msg.message || ""
      }`
    );
    return;
  }
  if (type === "download_progress") {
    const total = Number(msg.total || 0);
    const percent = total > 0 ? Math.round((Number(msg.downloaded || 0) / total) * 100) : 0;
    if (activeModelDownload?.backendId === msg.backend_id) {
      setModelDialogProgress(Number(msg.downloaded || 0), total);
    }
    setHint(`Download ${msg.artifact || msg.backend_id}: ${percent}%`, { sticky: true });
    return;
  }
  if (type === "idle_unload_scheduled") {
    const secs = Number(msg.delay_secs || 0);
    const text =
      msg.message ||
      (secs > 0
        ? `Keine Clients — Modelle werden in ${secs}s entladen`
        : "Idle-Unload geplant");
    log(text);
    setHint(text, { sticky: true });
    return;
  }
  if (type === "models_unloaded") {
    const text = msg.message || "Modelle entladen (Speicher freigegeben)";
    log(text);
    setHint(text, { sticky: true });
    showToast(text, { ms: 2800 });
    return;
  }
  if (type === "models_reloaded") {
    const text = msg.message || "Modelle wieder geladen";
    log(text);
    clearHintLock();
    setHint(text);
    showToast(text, { ms: 2000 });
    return;
  }
  if (type === "stack") {
    if (msg.asr) lab.asr = msg.asr;
    if (msg.tts) lab.tts = msg.tts;
    if (msg.llm) lab.llm = msg.llm;
    if (msg.voice) lab.voice = msg.voice;
    clearSwapBusy();
    persistLab();
    refreshLabUi();
    log(`Stack: asr=${msg.asr} tts=${msg.tts} llm=${msg.llm} · ${msg.message || ""}`);
    if (msg.ok === false) {
      setHint(copy.hintStackErr(msg.message), { error: true, sticky: true });
      showToast(copy.hintStackErr(msg.message), { error: true });
    } else {
      clearHintLock();
      setHint(copy.hintStackLive(msg.asr, msg.llm, msg.tts));
      showToast(copy.toastStack, { ms: 1600 });
      window.setTimeout(() => {
        if (!lab.swapBusy) {
          clearHintLock();
          refreshUi();
        }
      }, 1800);
    }
    refreshUi();
    return;
  }
  if (type === "error") {
    log(`ERR ${msg.stage || "?"}: ${msg.message || raw}`);
    state.processing = false;
    setHint(copy.hintErr(msg.stage, msg.message), { error: true, sticky: true });
    showToast(copy.hintErr(msg.stage, msg.message), { error: true, ms: 3200 });
    refreshUi();
    return;
  }
  log(`← ${raw.slice(0, 160)}`);
}

function benchmarkMemoryLabel(value) {
  const memory = Number(value);
  return value == null || !Number.isFinite(memory) ? "—" : `${memory.toFixed(0)} MiB`;
}

function renderBenchmark(run) {
  const result = run?.result;
  const summary = result?.summary;
  if (els.benchmarkStatus) {
    if (run.status === "running") {
      els.benchmarkStatus.textContent = "Messung läuft…";
    } else if (run.status === "failed") {
      els.benchmarkStatus.textContent = `Fehler: ${run.error || "unbekannt"}`;
    } else if (summary) {
      if (run.kind === "asr") {
        els.benchmarkStatus.textContent =
          `${summary.samples} Proben · ASR median ${Number(summary.asr_median_ms).toFixed(0)} ms` +
          ` · WER ${(Number(summary.wer_mean) * 100).toFixed(1)} %` +
          ` · CER ${(Number(summary.cer_mean) * 100).toFixed(1)} %` +
          ` · RAM ${benchmarkMemoryLabel(summary.memory_max_mib)}`;
      } else if (run.kind === "llm") {
        els.benchmarkStatus.textContent =
          `${summary.samples} Proben · TTFT median ${Number(summary.ttft_median_ms).toFixed(0)} ms` +
          ` · ${Number(summary.tokens_per_second_median).toFixed(1)} Token/s` +
          ` · RAM ${benchmarkMemoryLabel(summary.memory_max_mib)}`;
      } else {
        els.benchmarkStatus.textContent =
          `${summary.samples} Proben · TTFA median ${Number(summary.ttfa_median_ms).toFixed(0)} ms` +
          ` · RTF median ${Number(summary.rtf_median).toFixed(3)} / p95 ${Number(summary.rtf_p95).toFixed(3)}` +
          ` · RAM ${benchmarkMemoryLabel(summary.memory_max_mib)}`;
      }
    }
  }
  const samples = result?.samples || [];
  if (els.benchmarkHead) {
    els.benchmarkHead.innerHTML =
      run.kind === "asr"
        ? "<tr><th>Probe</th><th>Audio</th><th>Transkript</th><th>ASR</th><th>WER</th><th>CER</th><th>RAM</th></tr>"
        : run.kind === "llm"
          ? "<tr><th>Probe</th><th>Antwort</th><th>TTFT</th><th>Token/s</th><th>Gesamt</th><th>RAM</th></tr>"
          : "<tr><th>Probe</th><th>Audio</th><th>Bewertung</th><th>TTFA</th><th>RTF</th><th>Dauer</th><th>RAM</th></tr>";
  }
  if (els.benchmarkResults) {
    if (run.kind === "asr") {
      els.benchmarkResults.innerHTML = samples
        .map(
          (sample, index) => `<tr>
          <td>#${index + 1}</td>
          <td><audio controls preload="none" src="${escapeHtml(sample.audio_url)}"></audio></td>
          <td>${escapeHtml(sample.transcript || "—")}</td>
          <td>${Number(sample.asr_ms).toFixed(0)} ms</td>
          <td>${(Number(sample.wer) * 100).toFixed(1)} %</td>
          <td>${(Number(sample.cer) * 100).toFixed(1)} %</td>
          <td>${benchmarkMemoryLabel(sample.memory_mib)}</td>
        </tr>`
        )
        .join("");
    } else if (run.kind === "llm") {
      els.benchmarkResults.innerHTML = samples
        .map(
          (sample, index) => `<tr>
          <td>#${index + 1}</td>
          <td>${escapeHtml(sample.completion || "—")}</td>
          <td>${Number(sample.ttft_ms).toFixed(0)} ms</td>
          <td>${Number(sample.tokens_per_second).toFixed(1)}</td>
          <td>${Number(sample.total_ms).toFixed(0)} ms</td>
          <td>${benchmarkMemoryLabel(sample.memory_mib)}</td>
        </tr>`
        )
        .join("");
    } else {
      els.benchmarkResults.innerHTML = samples
        .map(
          (sample, index) => `<tr>
          <td>#${index + 1}</td>
          <td><audio controls preload="none" src="${escapeHtml(sample.audio_url)}"></audio></td>
          <td><div class="rating" aria-label="Probe ${index + 1} bewerten">
            ${[1, 2, 3, 4, 5]
              .map(
                (score) =>
                  `<button type="button" data-run="${escapeHtml(run.id)}" data-sample="${index}" data-score="${score}" aria-label="${score} von 5">${score}</button>`
              )
              .join("")}
          </div></td>
          <td>${Number(sample.ttfa_ms).toFixed(0)} ms</td>
          <td>${Number(sample.rtf).toFixed(3)}</td>
          <td>${Number(sample.audio_seconds).toFixed(2)} s</td>
          <td>${benchmarkMemoryLabel(sample.memory_mib)}</td>
        </tr>`
        )
        .join("");
    }
  }
  if (els.benchmarkTableWrap) els.benchmarkTableWrap.hidden = samples.length === 0;
  if (els.benchmarkExport) {
    els.benchmarkExport.hidden = run.status !== "completed";
    els.benchmarkExport.href = labPath(`/api/v1/benchmarks/${encodeURIComponent(run.id)}?format=csv`);
  }
}

async function submitBenchmarkRating(button) {
  const runId = button.dataset.run;
  const sampleIndex = Number(button.dataset.sample);
  const score = Number(button.dataset.score);
  const response = await fetch(labPath(`/api/v1/benchmarks/${encodeURIComponent(runId)}/ratings`), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ sample_index: sampleIndex, score }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Rating HTTP ${response.status}`);
  }
  button.parentElement
    ?.querySelectorAll("button")
    .forEach((candidate) => candidate.classList.toggle("selected", candidate === button));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function pollBenchmark(id) {
  for (;;) {
    const response = await fetch(labPath(`/api/v1/benchmarks/${encodeURIComponent(id)}`), {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Benchmark HTTP ${response.status}`);
    const run = await response.json();
    renderBenchmark(run);
    if (run.status !== "running") return run;
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
}

async function startBenchmark() {
  if (els.btnBenchmark) els.btnBenchmark.disabled = true;
  if (els.benchmarkStatus) els.benchmarkStatus.textContent = "Messung wird gestartet…";
  try {
    const prompts = (els.benchmarkPrompts?.value || "")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    const kind = els.benchmarkKind?.value || "tts";
    const response = await fetch(labPath("/api/v1/benchmarks"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ kind, prompts }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `Benchmark HTTP ${response.status}`);
    log(`Benchmark ${body.id} gestartet`);
    await pollBenchmark(body.id);
  } catch (error) {
    const message = error?.message || String(error);
    if (els.benchmarkStatus) els.benchmarkStatus.textContent = `Fehler: ${message}`;
    log(`Benchmark error: ${message}`);
  } finally {
    if (els.btnBenchmark) els.btnBenchmark.disabled = false;
  }
}

function disconnect() {
  // Stop auto-reconnect first so onclose does not reschedule.
  state.wantConnected = false;
  stopReconnectLoop();
  reconnect.hadSession = false;
  state.error = false;
  stopTalking(false);
  state.processing = false;
  state.speaking = false;
  state.connected = false;
  state.connecting = false;

  if (state.ws) {
    const ws = state.ws;
    try {
      ws.onopen = null;
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.close(1000, "client disconnect");
    } catch (_) {}
    state.ws = null;
  }
  releaseMicrophone();
  clearHintLock();
  refreshUi();
  log("Disconnected (manual)");
  showToast(copy.toastDisconnected, { ms: 1400 });
}

// ── Talk control ────────────────────────────────────────────────────
async function startTalking() {
  if (!state.connected || state.talking) return;
  try {
    await ensureAudio();
    await startCapture();
  } catch (e) {
    log(`Mic error: ${e.message || e}`);
    setHint(copy.hintMicDenied, { error: true, sticky: true });
    refreshUi();
    return;
  }
  state.talking = true;
  state.processing = false;
  state.error = false;
  clearHintLock();
  setPipeActive("asr");
  refreshUi();
}

function stopTalking(enterProcessing = true) {
  if (!state.talking) return;
  state.talking = false;
  flushSilenceToVad(800);
  if (enterProcessing && state.connected && !state.speaking) {
    state.processing = true;
  }
  clearHintLock();
  refreshUi();
}

function flushSilenceToVad(ms) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  const nFrames = Math.ceil(ms / FRAME_MS);
  const zeros = new Int16Array(FRAME_SAMPLES);
  for (let i = 0; i < nFrames; i++) {
    state.ws.send(zeros.slice().buffer);
  }
}

function isTypingTarget(el) {
  if (!el || el === document.body) return false;
  const tag = (el.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return true;
  if (el.isContentEditable) return true;
  return false;
}

// ── Event wiring ────────────────────────────────────────────────────
els.mic?.addEventListener("pointerdown", async (e) => {
  e.preventDefault();
  try {
    els.mic.setPointerCapture(e.pointerId);
  } catch (_) {}
  if (!state.connected) {
    if (!state.connecting) connect();
    return;
  }
  if (state.talkMode === "toggle") {
    if (state.talking) stopTalking();
    else await startTalking();
  } else {
    await startTalking();
  }
});

els.mic?.addEventListener("pointerup", () => {
  if (state.talkMode === "hold") stopTalking();
});
els.mic?.addEventListener("pointercancel", () => {
  if (state.talkMode === "hold") stopTalking();
});

els.btnConnect?.addEventListener("click", () => connect());
els.btnDisconnect?.addEventListener("click", () => disconnect());
els.btnConnectLab?.addEventListener("click", () => connect());
els.btnDisconnectLab?.addEventListener("click", () => disconnect());

els.btnLab?.addEventListener("click", () => toggleLab());
els.btnLabClose?.addEventListener("click", () => closeLab());
els.labScrim?.addEventListener("click", () => closeLab());

els.talkMode?.addEventListener("change", () => {
  state.talkMode = els.talkMode.value;
  localStorage.setItem("s2s.talkMode", state.talkMode);
  if (state.talking) stopTalking(false);
  clearHintLock();
  refreshUi();
});

els.voiceSelect?.addEventListener("change", () => {
  lab.voice = els.voiceSelect.value;
  persistLab();
  sendStackToBackend({ voiceOnly: true });
});

els.btnLogClear?.addEventListener("click", () => {
  if (els.log) els.log.textContent = "";
});

els.btnBenchmark?.addEventListener("click", () => startBenchmark());
els.benchmarkKind?.addEventListener("change", () => {
  if (els.btnBenchmark) {
    els.btnBenchmark.textContent = `${els.benchmarkKind.value.toUpperCase()} messen`;
  }
});
els.benchmarkResults?.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-score]");
  if (!button) return;
  submitBenchmarkRating(button).catch((error) => {
    const message = error?.message || String(error);
    showToast(`Bewertung fehlgeschlagen: ${message}`, { error: true, ms: 2800 });
  });
});

els.pipeline?.querySelectorAll(".pipe-node").forEach((node) => {
  node.addEventListener("click", () => {
    openLab(node.dataset.labSection || "sec-stack");
  });
});

window.addEventListener("keydown", async (e) => {
  if (e.code === "Escape" && state.labOpen) {
    e.preventDefault();
    closeLab();
    return;
  }
  // Lab shortcut
  if ((e.key === "l" || e.key === "L") && !isTypingTarget(e.target) && !e.metaKey && !e.ctrlKey) {
    e.preventDefault();
    toggleLab();
    return;
  }
  if (e.code === "Space" && !e.repeat && !isTypingTarget(e.target)) {
    e.preventDefault();
    if (!state.connected) {
      if (!state.connecting) connect();
      return;
    }
    if (state.talkMode === "toggle") {
      if (state.talking) stopTalking();
      else await startTalking();
    } else await startTalking();
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space" && state.talkMode === "hold" && !isTypingTarget(e.target)) {
    e.preventDefault();
    stopTalking();
  }
});

// First-run coach
function maybeShowCoach() {
  if (!els.coach) return;
  if (localStorage.getItem("s2s.coachDone") === "1") return;
  els.coach.hidden = false;
}
els.btnCoachDismiss?.addEventListener("click", () => {
  localStorage.setItem("s2s.coachDone", "1");
  if (els.coach) els.coach.hidden = true;
});

// Restore lab open preference only on desktop wide screens for power users
if (localStorage.getItem("s2s.labOpen") === "1" && window.innerWidth > 900) {
  // keep closed by default for talk-first; power users can press L
}

maybeShowCoach();
log("AuraGo S2S lab ready — Talk first, Lab second");
refreshUi();
