//! Versioned backend catalog and deterministic accelerator resolution.

use crate::gpu::{GpuKind, GpuReport};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::{Component, Path};

const EMBEDDED_CATALOG: &str = include_str!("../config/backends.json");

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendStage {
    Asr,
    Tts,
    Llm,
}

/// Relative resource demand for Lab UI (1 = leicht … 5 = sehr anspruchsvoll).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ResourceStars {
    #[serde(default)]
    pub cpu: u8,
    #[serde(default)]
    pub gpu: u8,
    #[serde(default)]
    pub vram: u8,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ResourceEstimate {
    #[serde(default)]
    pub vram_gb: f32,
    #[serde(default)]
    pub ram_gb: f32,
    /// Lab star ratings (CPU / GPU compute / VRAM). Values should be 1–5.
    #[serde(default)]
    pub stars: ResourceStars,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelArtifact {
    pub source: String,
    pub path: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub size: u64,
    /// Optional server-side download authentication scheme. Tokens are never
    /// serialized into the catalog.
    #[serde(default)]
    pub auth: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BackendVariant {
    pub id: String,
    pub accelerator: String,
    #[serde(default)]
    pub vendors: Vec<String>,
    #[serde(default)]
    pub platforms: Vec<String>,
    #[serde(default)]
    pub stable: bool,
    /// False when a catalog entry documents a future runtime but no immutable
    /// image is published for it yet. Such variants are never advertised as
    /// compatible, even with experimental opt-in.
    #[serde(default = "default_true")]
    pub published: bool,
    #[serde(default)]
    pub runtime_delivery_reason: String,
    #[serde(default)]
    pub device_match: Vec<String>,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub native_endpoint: String,
    #[serde(default)]
    pub container: String,
    #[serde(default)]
    pub image: String,
    /// Compressed image transfer estimate. The managed controller may replace
    /// this catalog hint with the exact signed-bundle value at runtime.
    #[serde(default)]
    pub image_download_size_bytes: u64,
    #[serde(default)]
    pub health_path: String,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    /// Optional model files for this runtime representation. Empty keeps the
    /// backend-level artifact list for schema-v1 compatibility.
    #[serde(default)]
    pub artifacts: Vec<ModelArtifact>,
    /// Optional bundled override for host runtimes whose files are not baked
    /// into the container image.
    #[serde(default)]
    pub bundled: Option<bool>,
    /// Optional protocol override (for example faster-whisper -> whisper-cpp).
    #[serde(default)]
    pub protocol: String,
    /// Allowlisted native host launcher profile. Empty means container/remote.
    #[serde(default)]
    pub host_profile: String,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceMode {
    Request,
    Restart,
    #[default]
    Fixed,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BackendDefinition {
    pub id: String,
    pub stage: BackendStage,
    pub name: String,
    #[serde(default)]
    pub tag: String,
    #[serde(default)]
    pub description: String,
    pub protocol: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub default_voice: String,
    #[serde(default)]
    pub voices: Vec<String>,
    /// How callers may select a TTS voice. Missing metadata fails closed to
    /// `fixed` so external legacy catalogs never claim unsupported switching.
    #[serde(default)]
    pub voice_mode: VoiceMode,
    #[serde(default)]
    pub native_sample_rate: u32,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub licenses: Vec<String>,
    #[serde(default)]
    pub access_url: String,
    pub resources: ResourceEstimate,
    #[serde(default)]
    pub bundled: bool,
    #[serde(default)]
    pub artifacts: Vec<ModelArtifact>,
    pub variants: Vec<BackendVariant>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StackPreset {
    pub id: String,
    pub name: String,
    pub asr_id: String,
    pub tts_id: String,
    #[serde(default = "default_preset_llm")]
    pub llm_id: String,
    /// Optional language tags for suggestion filtering (`de`, `en`, …).
    #[serde(default)]
    pub languages: Vec<String>,
    /// Soft VRAM ceiling for the pair (GB); used by suggestions.
    #[serde(default)]
    pub max_vram_gb: Option<f32>,
    /// Free-form latency hint for UI (`low`, `medium`, …).
    #[serde(default)]
    pub latency_hint: Option<String>,
    /// Free-form quality hint for UI (`balanced`, `high`, …).
    #[serde(default)]
    pub quality_hint: Option<String>,
}

fn default_preset_llm() -> String {
    "local-fallback".into()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BackendCatalog {
    pub schema_version: u32,
    pub backends: Vec<BackendDefinition>,
    #[serde(default)]
    pub presets: Vec<StackPreset>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CatalogBackendStatus {
    #[serde(flatten)]
    pub backend: BackendDefinition,
    pub available: bool,
    /// Hardware/platform compatibility independent of installation state.
    pub compatible: bool,
    /// True only when the selected runtime can be activated immediately.
    pub activatable: bool,
    pub reason: String,
    pub selected_variant: Option<BackendVariant>,
    pub variant_id: String,
    pub installed: bool,
    pub download_state: String,
    pub download_size_bytes: u64,
    pub downloaded_bytes: u64,
    pub deletable: bool,
    pub download_error: String,
    pub host_managed: bool,
    pub runtime_state: String,
    pub runtime_reason: String,
    pub image_download_size_bytes: u64,
    /// True when any selected (or base) artifact needs Hugging Face auth.
    /// Tokens themselves are never exposed through the catalog API.
    #[serde(default)]
    pub auth_required: bool,
    /// True when a server-side or in-memory session HF token is available.
    #[serde(default)]
    pub hf_token_configured: bool,
}

/// Hardware + best-effort capacity probe used by catalog and AuraGo setup.
///
/// Additive fields (`vram_*`, `ram_*`, `tier`, host agent) are optional /
/// defaulted so older clients ignore them safely. See `docs/aurago-integration.md`.
#[derive(Debug, Clone, Serialize)]
pub struct HardwareProfile {
    pub vendor: String,
    pub device_name: String,
    pub accelerators: Vec<String>,
    pub platform: String,
    pub in_container: bool,
    pub allow_experimental: bool,
    /// Best-effort device VRAM; `null` when unknown (common in CPU-only containers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_total_gb: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_free_gb: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ram_total_gb: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ram_available_gb: Option<f32>,
    /// Fresh Windows host supervisor heartbeat (file control plane).
    #[serde(default)]
    pub host_agent_online: bool,
    /// Host profiles the agent currently advertises as launchable.
    #[serde(default)]
    pub host_profiles: Vec<String>,
    /// Heuristic capacity class: `cpu-light` | `gpu-8gb` | `gpu-16gb+` | `host-vulkan`.
    #[serde(default = "default_capability_tier")]
    pub tier: String,
}

/// Alias used by AuraGo docs and the capability endpoint.
pub type CapabilityProfile = HardwareProfile;

fn default_capability_tier() -> String {
    "cpu-light".into()
}

impl HardwareProfile {
    pub fn from_gpu_report(report: &GpuReport) -> Self {
        let detected_vendor = report
            .selected
            .vendor_hint
            .clone()
            .unwrap_or_else(|| infer_vendor(&report.selected.name));
        let mut accelerators = Vec::new();
        for d in &report.all {
            let kind = accelerator_name(d.kind);
            if !accelerators.iter().any(|x| x == kind) {
                accelerators.push(kind.to_string());
            }
        }
        if !accelerators.iter().any(|x| x == "cpu") {
            accelerators.push("cpu".into());
        }
        if let Ok(extra) = std::env::var("S2S_LAB_ACCELERATORS") {
            for accelerator in extra
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
            {
                if is_known_accelerator(accelerator)
                    && !accelerators.iter().any(|item| item == accelerator)
                {
                    accelerators.push(accelerator.to_string());
                }
            }
        }
        let mut profile = Self {
            vendor: std::env::var("S2S_LAB_GPU_VENDOR")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(detected_vendor),
            device_name: std::env::var("S2S_LAB_DEVICE_NAME")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| report.selected.name.clone()),
            accelerators,
            platform: std::env::var("S2S_LAB_PLATFORM")
                .ok()
                .as_deref()
                .and_then(normalize_platform)
                .unwrap_or_else(|| std::env::consts::OS.to_string()),
            in_container: report.in_container,
            allow_experimental: env_truthy("S2S_ALLOW_EXPERIMENTAL"),
            vram_total_gb: None,
            vram_free_gb: None,
            ram_total_gb: None,
            ram_available_gb: None,
            host_agent_online: false,
            host_profiles: Vec::new(),
            tier: default_capability_tier(),
        };
        profile.apply_static_probes();
        profile.recompute_tier();
        profile
    }

    /// Refresh RAM/VRAM probes and recompute `tier` (sync, no host agent).
    pub fn apply_static_probes(&mut self) {
        if let Some(v) = env_f32("S2S_CAPABILITY_VRAM_TOTAL_GB") {
            self.vram_total_gb = Some(v);
        } else if self.vram_total_gb.is_none() {
            self.vram_total_gb = probe_vram_total_gb();
        }
        if let Some(v) = env_f32("S2S_CAPABILITY_VRAM_FREE_GB") {
            self.vram_free_gb = Some(v);
        } else if self.vram_free_gb.is_none() {
            self.vram_free_gb = probe_vram_free_gb();
        }
        if let Some(v) = env_f32("S2S_CAPABILITY_RAM_TOTAL_GB") {
            self.ram_total_gb = Some(v);
        } else {
            self.ram_total_gb = probe_ram_total_gb().or(self.ram_total_gb);
        }
        if let Some(v) = env_f32("S2S_CAPABILITY_RAM_AVAILABLE_GB") {
            self.ram_available_gb = Some(v);
        } else {
            self.ram_available_gb = probe_ram_available_gb().or(self.ram_available_gb);
        }
        self.recompute_tier();
    }

    /// Apply host-agent heartbeat (profiles + online flag) and recompute tier.
    pub fn apply_host_agent(&mut self, online: bool, profiles: impl IntoIterator<Item = String>) {
        self.host_agent_online = online;
        self.host_profiles = profiles.into_iter().collect();
        self.host_profiles.sort();
        self.host_profiles.dedup();
        self.recompute_tier();
    }

    pub fn recompute_tier(&mut self) {
        if let Ok(forced) = std::env::var("S2S_CAPABILITY_TIER") {
            let forced = forced.trim();
            if !forced.is_empty() {
                self.tier = forced.to_ascii_lowercase();
                return;
            }
        }
        self.tier = derive_capability_tier(self);
    }
}

/// Derive capacity tier per `docs/aurago-integration.md` (`heuristic_v1`).
pub fn derive_capability_tier(hw: &HardwareProfile) -> String {
    let has_vulkan = hw.accelerators.iter().any(|a| a == "vulkan");
    let has_cuda = hw.accelerators.iter().any(|a| a == "cuda");
    let has_sycl = hw.accelerators.iter().any(|a| a == "sycl");
    let vram = hw.vram_total_gb.or(hw.vram_free_gb);

    if hw.host_agent_online && has_vulkan {
        return "host-vulkan".into();
    }
    if vram.is_some_and(|g| g >= 14.0) {
        return "gpu-16gb+".into();
    }
    if vram.is_some_and(|g| g >= 6.0) {
        return "gpu-8gb".into();
    }
    // Accelerator present but VRAM unknown (typical Docker Desktop GPU passthrough gap).
    if has_cuda || has_sycl || (has_vulkan && !hw.in_container) {
        return "gpu-8gb".into();
    }
    if has_vulkan && hw.in_container {
        // Lab often advertises vulkan via env without device access inside the container.
        return "gpu-8gb".into();
    }
    "cpu-light".into()
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
}

fn probe_vram_total_gb() -> Option<f32> {
    probe_nvidia_smi_memory().map(|(total, _)| total)
}

fn probe_vram_free_gb() -> Option<f32> {
    probe_nvidia_smi_memory().map(|(_, free)| free)
}

/// Returns `(total_gb, free_gb)` from `nvidia-smi` when available.
fn probe_nvidia_smi_memory() -> Option<(f32, f32)> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?.trim();
    let mut parts = line.split(',').map(str::trim);
    let total_mib: f32 = parts.next()?.parse().ok()?;
    let free_mib: f32 = parts.next()?.parse().ok()?;
    Some((total_mib / 1024.0, free_mib / 1024.0))
}

fn probe_ram_total_gb() -> Option<f32> {
    #[cfg(target_os = "linux")]
    {
        return parse_meminfo_kb("MemTotal:").map(|kb| kb / 1024.0 / 1024.0);
    }
    #[cfg(target_os = "windows")]
    {
        return windows_ram_gb().map(|(total, _)| total);
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

fn probe_ram_available_gb() -> Option<f32> {
    #[cfg(target_os = "linux")]
    {
        // Prefer MemAvailable; fall back to MemFree.
        return parse_meminfo_kb("MemAvailable:")
            .or_else(|| parse_meminfo_kb("MemFree:"))
            .map(|kb| kb / 1024.0 / 1024.0);
    }
    #[cfg(target_os = "windows")]
    {
        return windows_ram_gb().map(|(_, avail)| avail);
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(prefix: &str) -> Option<f32> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(prefix) {
            let kb: f32 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb);
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_ram_gb() -> Option<(f32, f32)> {
    // GlobalMemoryStatusEx via PowerShell keeps us free of extra crates.
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty TotalVisibleMemorySize),((Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty FreePhysicalMemory))",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut nums = text
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter_map(|s| s.trim().parse::<f32>().ok());
    let total_kb = nums.next()?;
    let free_kb = nums.next()?;
    Some((total_kb / 1024.0 / 1024.0, free_kb / 1024.0 / 1024.0))
}

impl BackendCatalog {
    pub fn load() -> Result<Self> {
        let raw = if let Ok(path) = std::env::var("S2S_BACKEND_CATALOG") {
            std::fs::read_to_string(&path)
                .with_context(|| format!("read backend catalog {path}"))?
        } else {
            EMBEDDED_CATALOG.to_string()
        };
        let catalog: Self = serde_json::from_str(&raw).context("parse backend catalog")?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn validate(&self) -> Result<()> {
        if !matches!(self.schema_version, 1 | 2) {
            bail!(
                "unsupported backend catalog schema_version={} (expected 1 or 2)",
                self.schema_version
            );
        }
        let mut backend_ids = HashSet::new();
        let mut variant_ids = HashSet::new();
        for backend in &self.backends {
            validate_id(&backend.id, "backend")?;
            if !backend_ids.insert(backend.id.as_str()) {
                bail!("duplicate backend id '{}'", backend.id);
            }
            if backend.variants.is_empty() {
                bail!("backend '{}' has no variants", backend.id);
            }
            if backend.stage == BackendStage::Tts {
                let default_voice = backend.default_voice.trim();
                if default_voice.is_empty() {
                    bail!("TTS backend '{}' has no default_voice", backend.id);
                }
                let mut voices = HashSet::new();
                for voice in &backend.voices {
                    let trimmed = voice.trim();
                    if trimmed.is_empty()
                        || trimmed.chars().any(char::is_control)
                        || !voices.insert(trimmed.to_ascii_lowercase())
                    {
                        bail!(
                            "TTS backend '{}' has an invalid or duplicate voice",
                            backend.id
                        );
                    }
                }
                if !backend.voices.is_empty()
                    && !backend
                        .voices
                        .iter()
                        .any(|voice| voice.eq_ignore_ascii_case(default_voice))
                {
                    bail!(
                        "TTS backend '{}' default_voice '{}' is not listed in voices",
                        backend.id,
                        backend.default_voice
                    );
                }
                if matches!(backend.voice_mode, VoiceMode::Request | VoiceMode::Restart)
                    && backend.voices.is_empty()
                {
                    bail!(
                        "TTS backend '{}' voice_mode requires at least one catalog voice",
                        backend.id
                    );
                }
            }
            if !backend.access_url.is_empty() {
                validate_http_url(&backend.access_url)?;
            }
            if !backend.bundled
                && backend.artifacts.is_empty()
                && backend
                    .variants
                    .iter()
                    .all(|variant| variant.bundled != Some(true) && variant.artifacts.is_empty())
            {
                bail!(
                    "backend '{}' is neither bundled nor backed by downloadable artifacts",
                    backend.id
                );
            }
            validate_artifacts(&backend.id, &backend.artifacts)?;
            if backend.access_url.is_empty()
                && (backend
                    .artifacts
                    .iter()
                    .any(|artifact| artifact.auth == "huggingface")
                    || backend.variants.iter().any(|variant| {
                        variant
                            .artifacts
                            .iter()
                            .any(|artifact| artifact.auth == "huggingface")
                    }))
            {
                bail!(
                    "backend '{}' has authenticated artifacts but no access_url",
                    backend.id
                );
            }
            for variant in &backend.variants {
                validate_id(&variant.id, "variant")?;
                if !variant_belongs_to_backend(&backend.id, &variant.id) {
                    bail!(
                        "variant '{}' does not belong to backend '{}'",
                        variant.id,
                        backend.id
                    );
                }
                if !variant_ids.insert(variant.id.as_str()) {
                    bail!("duplicate variant id '{}'", variant.id);
                }
                if !is_known_accelerator(&variant.accelerator) {
                    bail!(
                        "variant '{}' has unknown accelerator '{}'",
                        variant.id,
                        variant.accelerator
                    );
                }
                if !variant.container.is_empty() {
                    validate_container_name(&variant.container)?;
                }
                if !variant.image.is_empty() && !is_safe_image_ref(&variant.image) {
                    bail!("variant '{}' has unsafe image reference", variant.id);
                }
                if !variant.endpoint.is_empty() {
                    validate_http_url(&variant.endpoint)?;
                }
                if !variant.native_endpoint.is_empty() {
                    validate_http_url(&variant.native_endpoint)?;
                }
                validate_artifacts(&variant.id, &variant.artifacts)?;
                if !variant.host_profile.is_empty() && !is_known_host_profile(&variant.host_profile)
                {
                    bail!(
                        "variant '{}' has unknown host profile '{}'",
                        variant.id,
                        variant.host_profile
                    );
                }
                if !variant.host_profile.is_empty() && !variant.container.is_empty() {
                    bail!(
                        "variant '{}' cannot declare both host_profile and container",
                        variant.id
                    );
                }
                if !variant.protocol.is_empty() && !is_known_protocol(&variant.protocol) {
                    bail!(
                        "variant '{}' has unknown protocol '{}'",
                        variant.id,
                        variant.protocol
                    );
                }
                if !variant_bundled(backend, variant)
                    && variant_artifacts(backend, variant).is_empty()
                {
                    bail!(
                        "variant '{}' is not bundled and has no downloadable artifacts",
                        variant.id
                    );
                }
            }
        }
        for preset in &self.presets {
            validate_id(&preset.id, "preset")?;
            if self.find(&preset.asr_id).is_none()
                || self.find(&preset.tts_id).is_none()
                || self.find(&preset.llm_id).is_none()
            {
                bail!("preset '{}' references an unknown backend", preset.id);
            }
        }
        Ok(())
    }

    pub fn find(&self, id: &str) -> Option<&BackendDefinition> {
        self.backends.iter().find(|backend| backend.id == id)
    }

    pub fn resolved(&self, hw: &HardwareProfile) -> Vec<CatalogBackendStatus> {
        self.backends
            .iter()
            .cloned()
            .map(|mut backend| {
                let selected_variant = resolve_variant(&backend, hw).cloned();
                let bundled = selected_variant
                    .as_ref()
                    .map(|variant| variant_bundled(&backend, variant))
                    .unwrap_or(backend.bundled);
                let artifacts = selected_variant
                    .as_ref()
                    .map(|variant| variant_artifacts(&backend, variant).to_vec())
                    .unwrap_or_else(|| backend.artifacts.clone());
                let protocol = selected_variant
                    .as_ref()
                    .map(|variant| variant_protocol(&backend, variant).to_string())
                    .unwrap_or_else(|| backend.protocol.clone());
                let host_managed = selected_variant
                    .as_ref()
                    .is_some_and(|variant| !variant.host_profile.is_empty());
                let available = selected_variant.is_some();
                let reason = if available {
                    String::new()
                } else {
                    incompatibility_reason(&backend, hw)
                };
                backend.bundled = bundled;
                backend.artifacts = artifacts;
                backend.protocol = protocol;
                let auth_required = backend
                    .artifacts
                    .iter()
                    .any(|artifact| artifact.auth == "huggingface");
                CatalogBackendStatus {
                    installed: bundled,
                    download_state: if bundled {
                        "bundled".into()
                    } else {
                        "missing".into()
                    },
                    download_size_bytes: backend
                        .artifacts
                        .iter()
                        .map(|artifact| artifact.size)
                        .sum(),
                    downloaded_bytes: 0,
                    deletable: !bundled,
                    download_error: String::new(),
                    host_managed,
                    runtime_state: selected_variant
                        .as_ref()
                        .map(|variant| {
                            if host_managed {
                                "host_module_delivery_pending"
                            } else if !variant.container.is_empty() {
                                "missing"
                            } else {
                                "external"
                            }
                        })
                        .unwrap_or("unavailable")
                        .into(),
                    runtime_reason: if host_managed {
                        "managed Windows host-module delivery is not installed".into()
                    } else {
                        String::new()
                    },
                    image_download_size_bytes: selected_variant
                        .as_ref()
                        .map(|variant| variant.image_download_size_bytes)
                        .unwrap_or_default(),
                    variant_id: selected_variant
                        .as_ref()
                        .map(|variant| variant.id.clone())
                        .unwrap_or_default(),
                    auth_required,
                    hf_token_configured: false,
                    backend,
                    available,
                    compatible: available,
                    activatable: available && !host_managed,
                    reason,
                    selected_variant,
                }
            })
            .collect()
    }
}

pub fn variant_artifacts<'a>(
    backend: &'a BackendDefinition,
    variant: &'a BackendVariant,
) -> &'a [ModelArtifact] {
    if variant.artifacts.is_empty() {
        backend.artifacts.as_slice()
    } else {
        variant.artifacts.as_slice()
    }
}

pub fn variant_bundled(backend: &BackendDefinition, variant: &BackendVariant) -> bool {
    variant.bundled.unwrap_or(backend.bundled)
}

pub fn variant_protocol<'a>(
    backend: &'a BackendDefinition,
    variant: &'a BackendVariant,
) -> &'a str {
    if variant.protocol.is_empty() {
        backend.protocol.as_str()
    } else {
        variant.protocol.as_str()
    }
}

pub fn resolve_variant<'a>(
    backend: &'a BackendDefinition,
    hw: &HardwareProfile,
) -> Option<&'a BackendVariant> {
    let mut candidates: Vec<&BackendVariant> = backend
        .variants
        .iter()
        .filter(|variant| variant_matches(variant, hw))
        .collect();
    candidates.sort_by_key(|variant| variant_rank(variant, hw));
    candidates.into_iter().next()
}

pub fn variant_is_compatible(variant: &BackendVariant, hw: &HardwareProfile) -> bool {
    variant_matches(variant, hw)
}

fn variant_matches(variant: &BackendVariant, hw: &HardwareProfile) -> bool {
    if !variant.published {
        return false;
    }
    let vendor_ok = variant.vendors.is_empty()
        || variant
            .vendors
            .iter()
            .any(|v| v == "any" || v == &hw.vendor);
    let platform_ok =
        variant.platforms.is_empty() || variant.platforms.iter().any(|p| p == &hw.platform);
    let accelerator_ok = variant.accelerator == "remote"
        || hw.accelerators.iter().any(|a| a == &variant.accelerator);
    let device_ok = variant.device_match.is_empty()
        || variant.device_match.iter().any(|needle| {
            hw.device_name
                .to_ascii_lowercase()
                .contains(&needle.to_ascii_lowercase())
        });
    let stability_ok = variant.stable || hw.allow_experimental;
    vendor_ok && platform_ok && accelerator_ok && device_ok && stability_ok
}

pub fn variant_rank(variant: &BackendVariant, hw: &HardwareProfile) -> u8 {
    // Certified Vulkan first. NVIDIA then prefers CUDA, Intel uses SYCL for
    // explicit model variants such as Qwen. A published local CPU runtime is
    // preferred over a remote endpoint so an unconfigured remote variant can
    // never hide an installable local module. Remote-only backends still
    // resolve normally and are gated by their credential and health checks.
    match variant.accelerator.as_str() {
        "vulkan" => 0,
        "cuda" if hw.vendor == "nvidia" => 1,
        "sycl" if hw.vendor == "intel" => 2,
        "cpu" => 3,
        "remote" => 4,
        _ => 5,
    }
}

fn incompatibility_reason(backend: &BackendDefinition, hw: &HardwareProfile) -> String {
    if backend.variants.iter().all(|variant| !variant.published) {
        return backend
            .variants
            .iter()
            .find_map(|variant| {
                (!variant.runtime_delivery_reason.is_empty())
                    .then(|| variant.runtime_delivery_reason.clone())
            })
            .unwrap_or_else(|| "runtime image is not published".into());
    }
    let platform_match = backend
        .variants
        .iter()
        .any(|v| v.platforms.is_empty() || v.platforms.iter().any(|p| p == &hw.platform));
    if !platform_match {
        return format!("not supported on {}", hw.platform);
    }
    let stable_match = backend.variants.iter().any(|v| v.stable);
    if !stable_match && !hw.allow_experimental {
        return "only experimental variants are registered".into();
    }
    format!(
        "no compatible variant for vendor={} accelerators={}",
        hw.vendor,
        hw.accelerators.join(",")
    )
}

pub fn endpoint_for(variant: &BackendVariant, in_container: bool) -> String {
    if in_container || variant.native_endpoint.is_empty() {
        variant.endpoint.clone()
    } else {
        variant.native_endpoint.clone()
    }
}

fn accelerator_name(kind: GpuKind) -> &'static str {
    match kind {
        GpuKind::Vulkan => "vulkan",
        GpuKind::Cuda => "cuda",
        GpuKind::Sycl => "sycl",
        GpuKind::Cpu => "cpu",
    }
}

fn infer_vendor(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    if lower.contains("nvidia") {
        "nvidia".into()
    } else if lower.contains("amd") || lower.contains("radeon") {
        "amd".into()
    } else if lower.contains("intel") || lower.contains("arc") {
        "intel".into()
    } else {
        "unknown".into()
    }
}

fn normalize_platform(value: &str) -> Option<String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "linux" => Some("linux".into()),
        "windows" => Some("windows".into()),
        "macos" => Some("macos".into()),
        _ => None,
    }
}

fn validate_id(id: &str, kind: &str) -> Result<()> {
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '.'))
    {
        bail!("{kind} id '{id}' must match [a-z0-9.-]+");
    }
    Ok(())
}

fn variant_belongs_to_backend(backend_id: &str, variant_id: &str) -> bool {
    if variant_id.starts_with(backend_id) {
        return true;
    }
    // Versioned logical IDs may omit their final version suffix in hardware
    // variant IDs (for example qwen3-tts-0.6b -> qwen3-tts-sycl-jit).
    let Some((family, suffix)) = backend_id.rsplit_once('-') else {
        return false;
    };
    suffix.as_bytes().first().is_some_and(u8::is_ascii_digit)
        && suffix.contains('.')
        && variant_id.starts_with(&format!("{family}-"))
}

fn validate_artifact_path(path: &str) -> Result<()> {
    let p = Path::new(path);
    if p.is_absolute()
        || path.contains('\\')
        || path.contains('\0')
        || p.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!("unsafe artifact path '{path}'");
    }
    Ok(())
}

fn validate_artifacts(owner: &str, artifacts: &[ModelArtifact]) -> Result<()> {
    for artifact in artifacts {
        validate_artifact_path(&artifact.path)?;
        if !artifact.sha256.is_empty()
            && (artifact.sha256.len() != 64
                || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            bail!(
                "'{owner}' artifact '{}' has an invalid SHA-256",
                artifact.path
            );
        }
        if !(artifact.source.starts_with("https://") || artifact.source.starts_with("http://")) {
            bail!("'{owner}' artifact source must be http(s)");
        }
        if !artifact.auth.is_empty() && artifact.auth != "huggingface" {
            bail!(
                "'{owner}' artifact '{}' has unknown auth scheme '{}'",
                artifact.path,
                artifact.auth
            );
        }
        if artifact.auth == "huggingface" {
            if !artifact.source.starts_with("https://huggingface.co/") {
                bail!(
                    "'{owner}' authenticated artifact '{}' must use huggingface.co over HTTPS",
                    artifact.path
                );
            }
            if artifact.sha256.len() != 64 || artifact.size == 0 {
                bail!(
                    "'{owner}' authenticated artifact '{}' must pin size and SHA-256",
                    artifact.path
                );
            }
        }
    }
    Ok(())
}

fn validate_container_name(name: &str) -> Result<()> {
    if name.len() > 128
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        bail!("unsafe container name '{name}'");
    }
    Ok(())
}

fn is_safe_image_ref(image: &str) -> bool {
    image.len() <= 256
        && image.chars().all(|c| {
            c.is_ascii_alphanumeric() || matches!(c, '/' | ':' | '.' | '-' | '_' | '@' | '+')
        })
}

fn validate_http_url(url: &str) -> Result<()> {
    if !(url.starts_with("http://") || url.starts_with("https://"))
        || url.contains(char::is_whitespace)
    {
        bail!("invalid http endpoint '{url}'");
    }
    Ok(())
}

fn is_known_accelerator(value: &str) -> bool {
    matches!(value, "vulkan" | "cuda" | "sycl" | "cpu" | "remote")
}

pub fn is_known_host_profile(value: &str) -> bool {
    matches!(
        value,
        "crispasr-whisper"
            | "crispasr-parakeet"
            | "crispasr-voxtral"
            | "crispasr-qwen3-asr"
            | "crispasr-canary"
            | "crispasr-funasr-mlt"
            | "crispasr-kokoro"
            | "crispasr-vibevoice"
            | "crispasr-chatterbox"
            | "crispasr-piper"
            | "crispasr-cosyvoice3"
            | "crispasr-omnivoice"
            | "chatterbox-python"
            | "inflect-python"
            | "audio8-python"
            | "llama-granite"
            | "qwen-sycl"
            | "supertonic-webgpu"
            | "xtts-webgpu"
            | "xtts-cpu"
    )
}

fn is_known_protocol(value: &str) -> bool {
    matches!(
        value,
        "faster-whisper"
            | "whisper-cpp"
            | "parakeet"
            | "voxtral"
            | "openai-tts"
            | "openai-tts-pcm"
            | "openai-chat"
    )
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(vendor: &str, accelerators: &[&str]) -> HardwareProfile {
        let mut profile = HardwareProfile {
            vendor: vendor.into(),
            device_name: if vendor == "intel" {
                "Intel Arc B580".into()
            } else {
                vendor.into()
            },
            accelerators: accelerators.iter().map(|x| x.to_string()).collect(),
            platform: "linux".into(),
            in_container: true,
            allow_experimental: false,
            vram_total_gb: None,
            vram_free_gb: None,
            ram_total_gb: None,
            ram_available_gb: None,
            host_agent_online: false,
            host_profiles: Vec::new(),
            tier: default_capability_tier(),
        };
        profile.recompute_tier();
        profile
    }

    #[test]
    fn capability_tier_cpu_light_by_default() {
        let profile = hw("unknown", &["cpu"]);
        assert_eq!(derive_capability_tier(&profile), "cpu-light");
    }

    #[test]
    fn capability_tier_gpu_8gb_from_vram() {
        let mut profile = hw("nvidia", &["cuda", "cpu"]);
        profile.vram_total_gb = Some(8.0);
        assert_eq!(derive_capability_tier(&profile), "gpu-8gb");
        profile.vram_total_gb = Some(16.0);
        assert_eq!(derive_capability_tier(&profile), "gpu-16gb+");
    }

    #[test]
    fn capability_tier_host_vulkan_when_agent_online() {
        let mut profile = hw("intel", &["vulkan", "cpu"]);
        profile.host_agent_online = true;
        assert_eq!(derive_capability_tier(&profile), "host-vulkan");
    }

    #[test]
    fn capability_tier_unknown_vram_with_cuda_is_gpu_8gb() {
        let profile = hw("nvidia", &["cuda", "cpu"]);
        assert_eq!(derive_capability_tier(&profile), "gpu-8gb");
    }

    #[test]
    fn capability_profile_serializes_tier() {
        let profile = hw("intel", &["cpu"]);
        let value = serde_json::to_value(&profile).unwrap();
        assert_eq!(value["tier"], "cpu-light");
        assert_eq!(value["host_agent_online"], false);
    }

    #[test]
    fn embedded_catalog_validates() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        catalog.validate().unwrap();
    }

    #[test]
    fn embedded_catalog_exposes_resource_stars() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        for backend in &catalog.backends {
            let stars = &backend.resources.stars;
            assert!(
                (1..=5).contains(&stars.cpu)
                    && (1..=5).contains(&stars.gpu)
                    && (1..=5).contains(&stars.vram),
                "backend '{}' stars must be 1–5 (got cpu={} gpu={} vram={})",
                backend.id,
                stars.cpu,
                stars.gpu,
                stars.vram
            );
        }
        let higgs = catalog.find("higgs-tts-3-4b").unwrap();
        assert_eq!(higgs.resources.stars.gpu, 5);
        assert_eq!(higgs.resources.stars.vram, 5);
        let tiny = catalog.find("fw-tiny").unwrap();
        assert_eq!(tiny.resources.stars.gpu, 1);
        assert_eq!(tiny.resources.stars.vram, 1);
    }

    #[test]
    fn qwen_catalog_exposes_named_voices() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("qwen3-tts-0.6b").unwrap();
        assert_eq!(backend.default_voice, "Aiden");
        assert_eq!(backend.voice_mode, VoiceMode::Request);
        assert_eq!(backend.voices.len(), 9);
        assert!(backend.voices.iter().any(|voice| voice == "Serena"));
        assert!(backend.voices.iter().any(|voice| voice == "Dylan"));
    }

    #[test]
    fn embedded_catalog_declares_truthful_tts_voice_modes() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        for id in [
            "qwen3-tts-0.6b",
            "xtts-v2",
            "vibevoice-realtime-0.5b",
            "higgs-tts-3-4b",
            "cosyvoice3-0.5b",
        ] {
            assert_eq!(
                catalog.find(id).unwrap().voice_mode,
                VoiceMode::Request,
                "{id}"
            );
        }
        let piper = catalog.find("piper").unwrap();
        assert_eq!(piper.voice_mode, VoiceMode::Restart);
        assert_eq!(piper.default_voice, "thorsten");
        assert_eq!(piper.voices, vec!["thorsten", "libritts"]);
        for id in [
            "supertonic",
            "chatterbox-multilingual-v3",
            "kokoro",
            "omnivoice",
            "inflect-micro-v2",
        ] {
            let backend = catalog.find(id).unwrap();
            assert_eq!(backend.voice_mode, VoiceMode::Fixed, "{id}");
            assert!(!backend.default_voice.trim().is_empty(), "{id}");
        }
    }

    #[test]
    fn missing_voice_mode_defaults_to_fixed_and_blank_tts_voice_is_rejected() {
        let mut value: serde_json::Value = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backends = value["backends"].as_array_mut().unwrap();
        let piper = backends
            .iter_mut()
            .find(|backend| backend["id"] == "piper")
            .unwrap();
        piper.as_object_mut().unwrap().remove("voice_mode");
        let catalog: BackendCatalog = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(catalog.find("piper").unwrap().voice_mode, VoiceMode::Fixed);

        let piper = value["backends"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|backend| backend["id"] == "piper")
            .unwrap();
        piper["default_voice"] = serde_json::Value::String(String::new());
        let catalog: BackendCatalog = serde_json::from_value(value).unwrap();
        assert!(catalog
            .validate()
            .unwrap_err()
            .to_string()
            .contains("default_voice"));
    }

    #[test]
    fn higgs_catalog_exposes_cuda_and_host_variants() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("higgs-tts-3-4b").unwrap();
        assert_eq!(backend.stage, BackendStage::Tts);
        assert_eq!(backend.model, "bosonai/higgs-tts-3-4b");
        assert_eq!(backend.native_sample_rate, 24_000);
        assert_eq!(backend.default_voice, "default");
        assert!(backend
            .artifacts
            .iter()
            .any(|a| a.path.ends_with("model.safetensors") && a.size > 9_000_000_000));
        let cuda = backend
            .variants
            .iter()
            .find(|v| v.id == "higgs-tts-3-4b-cuda")
            .expect("cuda variant");
        assert_eq!(cuda.accelerator, "cuda");
        assert_eq!(cuda.container, "s2s-tts-higgs");
        assert!(cuda.endpoint.contains("8086"));
        let host = backend
            .variants
            .iter()
            .find(|v| v.id == "higgs-tts-3-4b-host")
            .expect("host/remote variant for non-CUDA labs");
        assert_eq!(host.accelerator, "remote");
        assert!(host.container.is_empty());
        // Intel/AMD lab: host remote must resolve; NVIDIA prefers CUDA when experimental is on.
        let intel = hw("intel", &["sycl", "cpu"]);
        assert_eq!(
            resolve_variant(backend, &intel).map(|v| v.id.as_str()),
            Some("higgs-tts-3-4b-host")
        );
        let mut nvidia = hw("nvidia", &["cuda", "cpu"]);
        nvidia.allow_experimental = true;
        assert_eq!(
            resolve_variant(backend, &nvidia).map(|v| v.id.as_str()),
            Some("higgs-tts-3-4b-cuda")
        );
    }

    #[test]
    fn qwen_selects_sycl_on_intel_b580() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("qwen3-tts-0.6b").unwrap();
        let selected = resolve_variant(backend, &hw("intel", &["vulkan", "sycl", "cpu"])).unwrap();
        assert_eq!(selected.id, "qwen3-tts-b580-sycl-aot");
    }

    #[test]
    fn qwen_selects_preflighted_native_windows_sycl_variant() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("qwen3-tts-0.6b").unwrap();
        let mut hardware = hw("intel", &["sycl", "cpu"]);
        hardware.platform = "windows".into();
        hardware.in_container = true;
        hardware.allow_experimental = true;

        let selected = resolve_variant(backend, &hardware).unwrap();
        assert_eq!(selected.id, "qwen3-tts-sycl-windows-experimental");
        assert_eq!(selected.host_profile, "qwen-sycl");
        assert_eq!(
            selected.environment.get("GGML_BACKEND").map(String::as_str),
            Some("SYCL0")
        );
        assert_eq!(
            selected
                .environment
                .get("ONEAPI_DEVICE_SELECTOR")
                .map(String::as_str),
            Some("level_zero:0")
        );
        assert_eq!(
            endpoint_for(selected, hardware.in_container),
            "http://host.docker.internal:8083/v1/audio/speech"
        );
    }

    #[test]
    fn platform_override_is_restricted_to_known_values() {
        assert_eq!(normalize_platform(" Windows "), Some("windows".into()));
        assert_eq!(normalize_platform("linux"), Some("linux".into()));
        assert_eq!(normalize_platform("container-injected"), None);
    }

    #[test]
    fn faster_whisper_selects_cuda_on_nvidia() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("fw-small").unwrap();
        let selected = resolve_variant(backend, &hw("nvidia", &["cuda", "cpu"])).unwrap();
        assert_eq!(selected.accelerator, "cuda");
    }

    #[test]
    fn parakeet_selects_cuda_on_nvidia() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("parakeet-tdt-0.6b-v3").unwrap();
        let selected = resolve_variant(backend, &hw("nvidia", &["cuda", "cpu"])).unwrap();
        assert_eq!(selected.id, "parakeet-tdt-0.6b-v3-cuda");
    }

    #[test]
    fn vibevoice_catalog_exposes_cpu_and_voices() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("vibevoice-realtime-0.5b").unwrap();
        assert_eq!(backend.stage, BackendStage::Tts);
        assert_eq!(backend.default_voice, "emma");
        assert!(backend.voices.iter().any(|v| v == "emma"));
        assert!(backend.voices.iter().any(|v| v.contains("de-")));
        assert_eq!(backend.native_sample_rate, 24_000);
        assert!(backend
            .artifacts
            .iter()
            .any(|a| a.path.contains("q4_k.gguf") && a.size > 600_000_000));
        for voice in &backend.voices {
            let path = format!("vibevoice/{voice}.gguf");
            assert!(
                backend
                    .artifacts
                    .iter()
                    .any(|artifact| artifact.path == path),
                "missing directly addressable artifact for voice {voice}"
            );
        }
        assert!(backend
            .variants
            .iter()
            .any(|v| v.id == "vibevoice-realtime-0.5b-cpu" && v.accelerator == "cpu"));
        // A stable local runtime must not be hidden by an unconfigured remote.
        let selected = resolve_variant(backend, &hw("intel", &["sycl", "cpu"])).unwrap();
        assert_eq!(selected.id, "vibevoice-realtime-0.5b-cpu");
        assert!(!selected.container.is_empty());
    }

    #[test]
    fn chatterbox_catalog_is_pinned_and_selects_windows_cpu_or_cuda() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("chatterbox-multilingual-v3").unwrap();
        assert_eq!(backend.stage, BackendStage::Tts);
        assert_eq!(backend.model, "ResembleAI/chatterbox");
        assert_eq!(backend.default_voice, "default");
        assert_eq!(backend.native_sample_rate, 24_000);
        assert!(backend.languages.iter().any(|language| language == "de"));
        assert_eq!(backend.artifacts.len(), 6);
        assert!(backend.artifacts.iter().all(|artifact| {
            artifact
                .source
                .contains("/resolve/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18/")
                && artifact.sha256.len() == 64
                && artifact.size > 0
        }));

        let mut intel = hw("intel", &["cpu"]);
        intel.platform = "windows".into();
        intel.allow_experimental = true;
        let cpu = resolve_variant(backend, &intel).expect("Windows CPU variant");
        assert_eq!(cpu.id, "chatterbox-multilingual-v3-cpu-windows");
        assert_eq!(cpu.host_profile, "chatterbox-python");
        assert!(cpu.endpoint.contains("8090"));

        let mut nvidia = hw("nvidia", &["cuda", "cpu"]);
        nvidia.platform = "windows".into();
        nvidia.allow_experimental = true;
        let cuda = resolve_variant(backend, &nvidia).expect("Windows CUDA variant");
        assert_eq!(cuda.id, "chatterbox-multilingual-v3-cuda-windows");
        assert_eq!(cuda.host_profile, "chatterbox-python");
    }

    #[test]
    fn chatterbox_vulkan_variant_is_pinned_and_b580_scoped() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("chatterbox-multilingual-v3").unwrap();
        let variant = backend
            .variants
            .iter()
            .find(|variant| variant.id == "chatterbox-multilingual-v3-vulkan-windows-b580")
            .expect("Chatterbox Vulkan variant");

        assert_eq!(variant.accelerator, "vulkan");
        assert!(!variant.stable);
        assert_eq!(variant.host_profile, "crispasr-chatterbox");
        assert_eq!(variant.artifacts.len(), 2);
        assert!(variant.artifacts.iter().all(|artifact| {
            artifact
                .source
                .contains("/resolve/0295ba8dee365d84e5de44b818bb27ddfa705c43/")
                && artifact.path.starts_with("chatterbox/")
                && artifact.path.ends_with(".gguf")
                && artifact.sha256.len() == 64
                && artifact.size > 300_000_000
        }));

        let mut b580 = hw("intel", &["vulkan", "cpu"]);
        b580.platform = "windows".into();
        b580.device_name = "Intel Arc B580".into();
        b580.allow_experimental = true;
        let selected = resolve_variant(backend, &b580).expect("B580 Vulkan variant");
        assert_eq!(selected.id, variant.id);

        let mut other_intel = b580.clone();
        other_intel.device_name = "Intel Arc A770".into();
        assert_eq!(
            resolve_variant(backend, &other_intel).map(|selected| selected.id.as_str()),
            Some("chatterbox-multilingual-v3-cpu-windows")
        );
    }

    #[test]
    fn xtts_v2_catalog_is_pinned_gated_and_b580_scoped() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("xtts-v2").expect("XTTS-v2 backend");
        assert_eq!(backend.stage, BackendStage::Tts);
        assert_eq!(backend.model, "coqui/XTTS-v2");
        assert_eq!(backend.default_voice, "de_sample");
        assert_eq!(backend.native_sample_rate, 24_000);
        assert_eq!(backend.languages.len(), 17);
        assert!(backend.languages.iter().any(|language| language == "de"));
        assert!(backend
            .licenses
            .iter()
            .any(|license| license == "CC-BY-NC-4.0"));
        assert!(backend.access_url.contains("XTTSv2-Streaming-ONNX"));
        assert_eq!(backend.artifacts.len(), 11);
        assert!(backend
            .artifacts
            .iter()
            .all(|artifact| artifact.sha256.len() == 64 && artifact.size > 0));
        let gated: Vec<_> = backend
            .artifacts
            .iter()
            .filter(|artifact| artifact.auth == "huggingface")
            .collect();
        assert_eq!(gated.len(), 9);
        assert!(gated.iter().all(|artifact| artifact
            .source
            .contains("/resolve/975b202585dea4ae6ca7f6118121cdf1011d7d28/")));
        assert!(!backend
            .artifacts
            .iter()
            .any(|artifact| artifact.path.contains("int8")));

        let mut b580 = hw("intel", &["vulkan", "cpu"]);
        b580.platform = "windows".into();
        assert!(resolve_variant(backend, &b580).is_none());
        b580.allow_experimental = true;
        let selected = resolve_variant(backend, &b580).expect("B580 WebGPU variant");
        assert_eq!(selected.id, "xtts-v2-webgpu-vulkan-windows-b580");
        assert_eq!(selected.host_profile, "xtts-webgpu");
        assert!(selected.endpoint.contains("8091"));

        let mut other_intel = b580.clone();
        other_intel.device_name = "Intel Arc A770".into();
        let fallback = resolve_variant(backend, &other_intel).expect("Windows CPU fallback");
        assert_eq!(fallback.id, "xtts-v2-cpu-windows");
        assert_eq!(fallback.host_profile, "xtts-cpu");
    }

    #[test]
    fn authenticated_artifacts_require_huggingface_https_and_pins() {
        let mut catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let artifact = catalog
            .backends
            .iter_mut()
            .find(|backend| backend.id == "xtts-v2")
            .unwrap()
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.auth == "huggingface")
            .unwrap();
        artifact.source = "https://example.invalid/model.onnx".into();
        assert!(catalog.validate().is_err());

        let mut catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let artifact = catalog
            .backends
            .iter_mut()
            .find(|backend| backend.id == "xtts-v2")
            .unwrap()
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.auth == "huggingface")
            .unwrap();
        artifact.auth = "basic".into();
        assert!(catalog.validate().is_err());
    }

    #[test]
    fn voxtral_catalog_exposes_cuda_cpu_and_host() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("voxtral-mini-4b-realtime").unwrap();
        assert_eq!(backend.stage, BackendStage::Asr);
        assert_eq!(backend.model, "mistralai/Voxtral-Mini-4B-Realtime-2602");
        assert!(backend.resources.stars.vram >= 4);
        assert!(backend
            .artifacts
            .iter()
            .any(|a| a.path.ends_with("model.safetensors") && a.size > 8_000_000_000));
        let mut nvidia = hw("nvidia", &["cuda", "cpu"]);
        nvidia.allow_experimental = true;
        assert_eq!(
            resolve_variant(backend, &nvidia).map(|v| v.id.as_str()),
            Some("voxtral-mini-4b-realtime-cuda")
        );
        let intel = hw("intel", &["sycl", "cpu"]);
        assert_eq!(
            resolve_variant(backend, &intel).map(|v| v.id.as_str()),
            Some("voxtral-mini-4b-realtime-host")
        );
    }

    #[test]
    fn unpublished_parakeet_xpu_is_not_advertised_as_compatible() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("parakeet-tdt-0.6b-v3").unwrap();
        let mut hardware = hw("intel", &["sycl", "cpu"]);
        hardware.allow_experimental = true;
        let selected = resolve_variant(backend, &hardware).unwrap();
        assert_eq!(selected.id, "parakeet-tdt-0.6b-v3-cpu");
        let xpu = backend
            .variants
            .iter()
            .find(|variant| variant.id == "parakeet-tdt-0.6b-v3-xpu")
            .unwrap();
        assert!(!xpu.published);
        assert_eq!(
            xpu.runtime_delivery_reason,
            "image_not_published_for_architecture"
        );
    }

    #[test]
    fn parakeet_uses_cpu_on_windows_intel_docker() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("parakeet-tdt-0.6b-v3").unwrap();
        let mut hardware = hw("intel", &["sycl", "cpu"]);
        hardware.platform = "windows".into();
        hardware.allow_experimental = true;
        let selected = resolve_variant(backend, &hardware).unwrap();
        assert_eq!(selected.id, "parakeet-tdt-0.6b-v3-cpu");
    }

    #[test]
    fn unsafe_artifact_paths_are_rejected() {
        assert!(validate_artifact_path("../escape.gguf").is_err());
        assert!(validate_artifact_path("C:\\escape.gguf").is_err());
        assert!(validate_artifact_path("/escape.gguf").is_err());
        assert!(validate_artifact_path("qwen/model.gguf").is_ok());
    }

    #[test]
    fn multilingual_crispasr_backends_are_catalogued() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        catalog
            .validate()
            .expect("catalog with multilingual backends");

        let asr = [
            (
                "qwen3-asr-0.6b",
                "crispasr-qwen3-asr",
                "qwen3-asr/qwen3-asr-0.6b-q4_k.gguf",
            ),
            (
                "canary-1b-v2",
                "crispasr-canary",
                "canary/canary-1b-v2-q4_k.gguf",
            ),
            (
                "fun-asr-mlt-nano",
                "crispasr-funasr-mlt",
                "funasr/funasr-mlt-nano-2512-q4_k.gguf",
            ),
        ];
        for (id, profile, model_path) in asr {
            let backend = catalog.find(id).expect(id);
            assert_eq!(backend.stage, BackendStage::Asr);
            assert_eq!(backend.protocol, "whisper-cpp");
            let cpu = backend
                .variants
                .iter()
                .find(|v| v.id.ends_with("host-cpu"))
                .expect("host-cpu variant");
            assert!(cpu.stable);
            assert_eq!(cpu.platforms, vec!["windows".to_string()]);
            assert_eq!(cpu.host_profile, profile);
            assert!(variant_artifacts(backend, cpu)
                .iter()
                .any(|a| a.path == model_path));
            assert!(is_known_host_profile(profile));

            // Host-only backends must not resolve on Linux (no host agent there).
            let linux = hw("any", &["cpu"]);
            assert!(resolve_variant(backend, &linux).is_none());
        }

        let tts = [
            ("piper", "crispasr-piper", 8092),
            ("cosyvoice3-0.5b", "crispasr-cosyvoice3", 8093),
            ("omnivoice", "crispasr-omnivoice", 8094),
        ];
        for (id, profile, port) in tts {
            let backend = catalog.find(id).expect(id);
            assert_eq!(backend.stage, BackendStage::Tts);
            assert_eq!(backend.protocol, "openai-tts");
            let cpu = backend
                .variants
                .iter()
                .find(|v| v.id.ends_with("host-cpu"))
                .expect("host-cpu variant");
            assert_eq!(cpu.platforms, vec!["windows".to_string()]);
            assert_eq!(cpu.host_profile, profile);
            assert!(cpu.native_endpoint.contains(&port.to_string()));
            assert!(is_known_host_profile(profile));
        }

        assert!(catalog.presets.iter().any(|p| p.id == "multilingual-8gb"
            && p.asr_id == "qwen3-asr-0.6b"
            && p.tts_id == "piper"
            && p.llm_id == "local-fallback"));
    }

    #[test]
    fn inflect_micro_catalog_is_windows_host_cpu() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("inflect-micro-v2").expect("inflect-micro-v2");
        assert_eq!(backend.stage, BackendStage::Tts);
        assert_eq!(backend.protocol, "openai-tts");
        assert_eq!(backend.languages, vec!["en".to_string()]);
        assert_eq!(backend.native_sample_rate, 24_000);
        let cpu = backend
            .variants
            .iter()
            .find(|v| v.id == "inflect-micro-v2-host-cpu")
            .expect("host-cpu");
        assert!(cpu.stable);
        assert_eq!(cpu.platforms, vec!["windows".to_string()]);
        assert_eq!(cpu.host_profile, "inflect-python");
        assert!(is_known_host_profile("inflect-python"));
        assert!(variant_artifacts(backend, cpu)
            .iter()
            .any(|a| a.path == "inflect-micro-v2/model.pth" && a.size == 37_529_995));
        let linux = hw("any", &["cpu"]);
        assert!(resolve_variant(backend, &linux).is_none());
        assert!(catalog
            .presets
            .iter()
            .any(|p| p.id == "english-compact" && p.tts_id == "inflect-micro-v2"));
    }

    #[test]
    fn audio8_tts_catalog_supports_stable_windows_cpu() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog
            .find("audio8-tts-preview-0.6b")
            .expect("audio8-tts-preview-0.6b");
        assert_eq!(backend.stage, BackendStage::Tts);
        assert_eq!(backend.protocol, "openai-tts");
        assert_eq!(backend.model, "Audio8/Audio8-TTS-Preview-0.6b");
        assert_eq!(backend.native_sample_rate, 44_100);
        assert_eq!(backend.voice_mode, VoiceMode::Request);
        assert!(backend.languages.iter().any(|language| language == "de"));
        assert!(backend.artifacts.iter().any(|artifact| {
            artifact.path == "audio8-tts-preview-0.6b/model.safetensors"
                && artifact.size == 1_202_342_528
                && artifact
                    .source
                    .contains("/resolve/f9612f13a0ab40facf3d050fc908b9e6db05c2be/")
        }));
        assert!(is_known_host_profile("audio8-python"));

        // CPU host path is stable so downloads work without experimental opt-in.
        let mut windows = hw("any", &["cpu"]);
        windows.platform = "windows".into();
        windows.allow_experimental = false;
        let cpu = resolve_variant(backend, &windows).expect("Windows CPU variant");
        assert_eq!(cpu.id, "audio8-tts-preview-0.6b-host-cpu");
        assert_eq!(cpu.host_profile, "audio8-python");
        assert!(cpu.endpoint.contains("8096"));
        assert!(cpu.stable);

        // CUDA remains experimental and ranks above CPU when allowed.
        let mut nvidia = hw("nvidia", &["cuda", "cpu"]);
        nvidia.platform = "windows".into();
        nvidia.allow_experimental = true;
        let cuda = resolve_variant(backend, &nvidia).expect("Windows CUDA variant");
        assert_eq!(cuda.id, "audio8-tts-preview-0.6b-host-cuda");
        assert_eq!(cuda.host_profile, "audio8-python");
        assert!(!cuda.stable);

        let linux = hw("any", &["cpu"]);
        assert!(resolve_variant(backend, &linux).is_none());
    }

    #[test]
    fn cross_backend_variant_is_rejected() {
        let mut catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        catalog.backends[0].variants[0].id = "local-fallback-cuda".into();
        assert!(catalog.validate().is_err());
    }

    #[test]
    fn schema_one_remains_readable() {
        let mut catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        catalog.schema_version = 1;
        catalog.validate().unwrap();
    }

    #[test]
    fn variant_metadata_overrides_backend_artifacts_and_protocol() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("fw-tiny").unwrap();
        let variant = backend
            .variants
            .iter()
            .find(|variant| variant.id == "fw-tiny-vulkan-windows-b580")
            .unwrap();

        assert!(!variant_bundled(backend, variant));
        assert_eq!(variant_protocol(backend, variant), "whisper-cpp");
        assert_eq!(
            variant_artifacts(backend, variant)[0].path,
            "whisper/ggml-tiny.bin"
        );
        assert_eq!(variant.host_profile, "crispasr-whisper");

        let mut hardware = hw("intel", &["vulkan", "cpu"]);
        hardware.platform = "windows".into();
        hardware.allow_experimental = true;
        let resolved = catalog
            .resolved(&hardware)
            .into_iter()
            .find(|status| status.backend.id == "fw-tiny")
            .unwrap();
        assert!(!resolved.backend.bundled);
        assert_eq!(resolved.backend.protocol, "whisper-cpp");
        assert_eq!(resolved.backend.artifacts[0].path, "whisper/ggml-tiny.bin");
    }

    #[test]
    fn b580_vulkan_variants_require_experimental_opt_in() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let mut hardware = hw("intel", &["vulkan", "cpu"]);
        hardware.platform = "windows".into();

        let supertonic = catalog.find("supertonic").unwrap();
        assert_eq!(
            resolve_variant(supertonic, &hardware).map(|variant| variant.id.as_str()),
            Some("supertonic-cpu")
        );

        hardware.allow_experimental = true;
        let expected = [
            ("fw-tiny", "fw-tiny-vulkan-windows-b580"),
            (
                "parakeet-tdt-0.6b-v3",
                "parakeet-tdt-0.6b-v3-vulkan-windows-b580",
            ),
            (
                "voxtral-mini-4b-realtime",
                "voxtral-mini-4b-realtime-vulkan-windows-b580",
            ),
            ("supertonic", "supertonic-webgpu-vulkan-windows-b580"),
            (
                "chatterbox-multilingual-v3",
                "chatterbox-multilingual-v3-vulkan-windows-b580",
            ),
            ("xtts-v2", "xtts-v2-webgpu-vulkan-windows-b580"),
            ("kokoro", "kokoro-vulkan-windows-b580"),
            (
                "vibevoice-realtime-0.5b",
                "vibevoice-realtime-0.5b-vulkan-windows-b580",
            ),
            ("local-fallback", "local-fallback-vulkan-windows-b580"),
            ("qwen3-asr-0.6b", "qwen3-asr-0.6b-vulkan-windows-b580"),
            ("canary-1b-v2", "canary-1b-v2-vulkan-windows-b580"),
            ("fun-asr-mlt-nano", "fun-asr-mlt-nano-vulkan-windows-b580"),
            ("piper", "piper-vulkan-windows-b580"),
            ("cosyvoice3-0.5b", "cosyvoice3-0.5b-vulkan-windows-b580"),
            ("omnivoice", "omnivoice-vulkan-windows-b580"),
        ];
        for (backend_id, variant_id) in expected {
            let selected = resolve_variant(catalog.find(backend_id).unwrap(), &hardware).unwrap();
            assert_eq!(selected.id, variant_id);
            assert!(!selected.host_profile.is_empty());
        }

        hardware.accelerators.push("sycl".into());
        let qwen = resolve_variant(catalog.find("qwen3-tts-0.6b").unwrap(), &hardware).unwrap();
        assert_eq!(qwen.id, "qwen3-tts-sycl-windows-experimental");
        assert_eq!(qwen.host_profile, "qwen-sycl");
    }

    #[test]
    fn higgs_has_no_vulkan_or_host_managed_variant() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let higgs = catalog.find("higgs-tts-3-4b").unwrap();
        assert!(higgs.variants.iter().all(|variant| {
            variant.accelerator != "vulkan"
                && variant.host_profile.is_empty()
                && !variant.id.contains("vulkan")
        }));
    }
}
