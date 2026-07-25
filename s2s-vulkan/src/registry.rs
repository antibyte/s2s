//! Versioned backend catalog and deterministic accelerator resolution.

use crate::gpu::{GpuKind, GpuReport};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::{Component, Path};

const EMBEDDED_CATALOG: &str = include_str!("../config/backends.json");

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
    #[serde(default)]
    pub native_sample_rate: u32,
    #[serde(default)]
    pub languages: Vec<String>,
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
    pub reason: String,
    pub selected_variant: Option<BackendVariant>,
    pub installed: bool,
    pub download_state: String,
    pub download_size_bytes: u64,
    pub downloaded_bytes: u64,
    pub deletable: bool,
    pub download_error: String,
    pub host_managed: bool,
    pub runtime_state: String,
    pub runtime_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareProfile {
    pub vendor: String,
    pub device_name: String,
    pub accelerators: Vec<String>,
    pub platform: String,
    pub in_container: bool,
    pub allow_experimental: bool,
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
        Self {
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
        }
    }
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
                    runtime_state: if host_managed {
                        "unknown".into()
                    } else {
                        "not_managed".into()
                    },
                    runtime_reason: String::new(),
                    backend,
                    available,
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
    // explicit model variants such as Qwen, and CPU remains the last fallback.
    match variant.accelerator.as_str() {
        "vulkan" => 0,
        "cuda" if hw.vendor == "nvidia" => 1,
        "sycl" if hw.vendor == "intel" => 2,
        "remote" => 3,
        "cpu" => 4,
        _ => 5,
    }
}

fn incompatibility_reason(backend: &BackendDefinition, hw: &HardwareProfile) -> String {
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
            | "crispasr-kokoro"
            | "crispasr-vibevoice"
            | "llama-granite"
            | "qwen-vulkan"
            | "supertonic-webgpu"
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
        HardwareProfile {
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
        }
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
        assert_eq!(backend.voices.len(), 9);
        assert!(backend.voices.iter().any(|voice| voice == "Serena"));
        assert!(backend.voices.iter().any(|voice| voice == "Dylan"));
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
        assert!(backend
            .variants
            .iter()
            .any(|v| v.id == "vibevoice-realtime-0.5b-cpu" && v.accelerator == "cpu"));
        // remote host ranks above cpu when both are stable.
        let selected = resolve_variant(backend, &hw("intel", &["sycl", "cpu"])).unwrap();
        assert_eq!(selected.id, "vibevoice-realtime-0.5b-host");
        assert!(selected.endpoint.contains("8089"));
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
    fn parakeet_xpu_requires_linux_experimental_opt_in() {
        let catalog: BackendCatalog = serde_json::from_str(EMBEDDED_CATALOG).unwrap();
        let backend = catalog.find("parakeet-tdt-0.6b-v3").unwrap();
        let mut hardware = hw("intel", &["sycl", "cpu"]);
        hardware.allow_experimental = true;
        let selected = resolve_variant(backend, &hardware).unwrap();
        assert_eq!(selected.id, "parakeet-tdt-0.6b-v3-xpu");
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
        assert_eq!(
            resolved.backend.artifacts[0].path,
            "whisper/ggml-tiny.bin"
        );
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
            ("qwen3-tts-0.6b", "qwen3-tts-vulkan-windows-b580"),
            ("kokoro", "kokoro-vulkan-windows-b580"),
            (
                "vibevoice-realtime-0.5b",
                "vibevoice-realtime-0.5b-vulkan-windows-b580",
            ),
            ("local-fallback", "local-fallback-vulkan-windows-b580"),
        ];
        for (backend_id, variant_id) in expected {
            let selected = resolve_variant(catalog.find(backend_id).unwrap(), &hardware).unwrap();
            assert_eq!(selected.id, variant_id);
            assert!(!selected.host_profile.is_empty());
        }
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
