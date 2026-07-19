//! Automatic GPU / accelerator detection for host and Docker.
//!
//! Detection order (when `--gpu auto`):
//! 1. Explicit env (`GGML_BACKEND`, `S2S_GPU`, `CUDA_VISIBLE_DEVICES`, …)
//! 2. NVIDIA (`nvidia-smi`, `/dev/nvidia*`, container env)
//! 3. Vulkan devices (`vulkaninfo`, ICD files, `/dev/dri`)
//! 4. CPU fallback
//!
//! Results are applied to process env (`GGML_BACKEND`, `S2S_GPU_KIND`, …) so
//! child processes (whisper/llama/qwentts) and sibling containers see the same choice.

use crate::config::GpuPreference;
use serde::Serialize;
use std::path::Path;
use std::process::Command;
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuKind {
    Vulkan,
    Cuda,
    Cpu,
}

impl GpuKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub kind: GpuKind,
    /// GGML-style backend id, e.g. `Vulkan0`, `CUDA0`, `CPU`.
    pub ggml_backend: String,
    pub vendor_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuReport {
    pub selected: GpuDevice,
    pub all: Vec<GpuDevice>,
    pub in_container: bool,
    pub dri_nodes: Vec<String>,
    pub notes: Vec<String>,
}

impl GpuReport {
    pub fn apply_env(&self) {
        // Only set if the operator did not already pin GGML_BACKEND.
        if std::env::var_os("GGML_BACKEND").is_none() {
            // SAFETY: single-threaded at startup before spawning pipeline tasks that read env.
            unsafe {
                std::env::set_var("GGML_BACKEND", &self.selected.ggml_backend);
            }
        }
        unsafe {
            std::env::set_var("S2S_GPU_KIND", self.selected.kind.as_str());
            std::env::set_var("S2S_GPU_NAME", &self.selected.name);
            std::env::set_var("S2S_GGML_BACKEND", &self.selected.ggml_backend);
        }
        if self.in_container {
            unsafe {
                std::env::set_var("S2S_IN_CONTAINER", "1");
            }
        }
    }

    pub fn log(&self) {
        info!(
            "GPU auto-detect: selected {} ({}) → GGML_BACKEND={}",
            self.selected.name, self.selected.kind.as_str(), self.selected.ggml_backend
        );
        if self.in_container {
            info!("Running inside a container (/.dockerenv or cgroup)");
        }
        for d in &self.all {
            info!(
                "  [{}] {} kind={} ggml={}",
                d.index,
                d.name,
                d.kind.as_str(),
                d.ggml_backend
            );
        }
        if !self.dri_nodes.is_empty() {
            info!("  DRI nodes: {}", self.dri_nodes.join(", "));
        }
        for n in &self.notes {
            warn!("  note: {n}");
        }
    }
}

pub fn detect(pref: GpuPreference) -> GpuReport {
    let in_container = running_in_container();
    let dri_nodes = list_dri_nodes();
    let mut notes = Vec::new();
    let mut all = Vec::new();

    // --- NVIDIA / CUDA ---
    let nvidia = detect_nvidia();
    all.extend(nvidia);

    // --- Vulkan (includes AMD/Intel/NVIDIA ICD path) ---
    let vulkan = detect_vulkan(&mut notes);
    // Dedup by name roughly: keep vulkan devices even if NVIDIA also listed.
    for v in vulkan {
        if !all.iter().any(|a| a.ggml_backend == v.ggml_backend && a.kind == v.kind) {
            all.push(v);
        }
    }

    // Re-index
    for (i, d) in all.iter_mut().enumerate() {
        d.index = i as u32;
    }

    if all.is_empty() {
        notes.push(
            "No GPU found — using CPU. In Docker, pass devices (see docker-compose.yml) \
             and install NVIDIA Container Toolkit or mount /dev/dri."
                .into(),
        );
    }

    let selected = select_device(pref, &all, &mut notes);

    GpuReport {
        selected,
        all,
        in_container,
        dri_nodes,
        notes,
    }
}

fn select_device(pref: GpuPreference, all: &[GpuDevice], notes: &mut Vec<String>) -> GpuDevice {
    // Explicit env wins for auto mode.
    if pref == GpuPreference::Auto {
        if let Ok(backend) = std::env::var("GGML_BACKEND") {
            if !backend.is_empty() {
                let kind = if backend.to_ascii_uppercase().contains("VULKAN") {
                    GpuKind::Vulkan
                } else if backend.to_ascii_uppercase().contains("CUDA") {
                    GpuKind::Cuda
                } else {
                    GpuKind::Cpu
                };
                return GpuDevice {
                    index: 0,
                    name: format!("env GGML_BACKEND={backend}"),
                    kind,
                    ggml_backend: backend,
                    vendor_hint: None,
                };
            }
        }
        if let Ok(s) = std::env::var("S2S_GPU") {
            let s = s.to_ascii_lowercase();
            if s == "cpu" {
                return cpu_device();
            }
            if s.starts_with("vulkan") {
                if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Vulkan) {
                    return d.clone();
                }
                notes.push("S2S_GPU=vulkan but no Vulkan device detected".into());
            }
            if s.starts_with("cuda") {
                if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Cuda) {
                    return d.clone();
                }
                notes.push("S2S_GPU=cuda but no CUDA device detected".into());
            }
        }
    }

    let want = match pref {
        GpuPreference::Auto => {
            // Prefer discrete NVIDIA CUDA if present, else any Vulkan, else CPU.
            // For this project Vulkan is the primary target — prefer Vulkan first.
            if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Vulkan) {
                return d.clone();
            }
            if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Cuda) {
                notes.push(
                    "No Vulkan device; falling back to CUDA (use Vulkan-built GGML for this stack)"
                        .into(),
                );
                return d.clone();
            }
            return cpu_device();
        }
        GpuPreference::Vulkan => GpuKind::Vulkan,
        GpuPreference::Cuda => GpuKind::Cuda,
        GpuPreference::Cpu => return cpu_device(),
    };

    if let Some(d) = all.iter().find(|d| d.kind == want) {
        return d.clone();
    }

    notes.push(format!(
        "Requested GPU {:?} not found — falling back to CPU",
        want
    ));
    cpu_device()
}

fn cpu_device() -> GpuDevice {
    GpuDevice {
        index: 0,
        name: "CPU".into(),
        kind: GpuKind::Cpu,
        ggml_backend: "CPU".into(),
        vendor_hint: None,
    }
}

pub fn running_in_container() -> bool {
    if Path::new("/.dockerenv").exists() {
        return true;
    }
    if std::env::var_os("S2S_DOCKER").is_some() {
        return true;
    }
    // cgroup v1/v2 hints
    if let Ok(data) = std::fs::read_to_string("/proc/1/cgroup") {
        if data.contains("docker") || data.contains("kubepods") || data.contains("containerd") {
            return true;
        }
    }
    false
}

fn list_dri_nodes() -> Vec<String> {
    let mut out = Vec::new();
    let dri = Path::new("/dev/dri");
    if !dri.is_dir() {
        return out;
    }
    if let Ok(rd) = std::fs::read_dir(dri) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with("renderD") || name.starts_with("card") {
                out.push(format!("/dev/dri/{name}"));
            }
        }
    }
    out.sort();
    out
}

fn detect_nvidia() -> Vec<GpuDevice> {
    let mut devices = Vec::new();

    // Container / toolkit env often set even without nvidia-smi on PATH of app image.
    let has_nvidia_env = std::env::var_os("NVIDIA_VISIBLE_DEVICES").is_some()
        || std::env::var_os("NVIDIA_DRIVER_CAPABILITIES").is_some()
        || Path::new("/dev/nvidia0").exists()
        || Path::new("/dev/nvidiactl").exists();

    if let Some(list) = run_nvidia_smi() {
        for (i, name) in list.into_iter().enumerate() {
            devices.push(GpuDevice {
                index: i as u32,
                name: name.clone(),
                kind: GpuKind::Cuda,
                ggml_backend: format!("CUDA{i}"),
                vendor_hint: Some("nvidia".into()),
            });
        }
        return devices;
    }

    if has_nvidia_env {
        devices.push(GpuDevice {
            index: 0,
            name: "NVIDIA GPU (device node / container env)".into(),
            kind: GpuKind::Cuda,
            ggml_backend: "CUDA0".into(),
            vendor_hint: Some("nvidia".into()),
        });
    }

    devices
}

fn run_nvidia_smi() -> Option<Vec<String>> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let names: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    if names.is_empty() {
        None
    } else {
        Some(names)
    }
}

fn detect_vulkan(notes: &mut Vec<String>) -> Vec<GpuDevice> {
    let mut devices = Vec::new();

    // 1) vulkaninfo --summary (most reliable when tools installed)
    if let Some(list) = parse_vulkaninfo() {
        return list;
    }

    // 2) ICD files present?
    let icd_dirs = [
        "/usr/share/vulkan/icd.d",
        "/etc/vulkan/icd.d",
        "/usr/local/share/vulkan/icd.d",
    ];
    let mut icds = Vec::new();
    for dir in icd_dirs {
        let p = Path::new(dir);
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.ends_with(".json") {
                    icds.push(format!("{dir}/{name}"));
                }
            }
        }
    }

    // Also honor VK_ICD_FILENAMES
    if let Ok(paths) = std::env::var("VK_ICD_FILENAMES") {
        for p in paths.split(':') {
            if !p.is_empty() {
                icds.push(p.to_string());
            }
        }
    }

    let dri = list_dri_nodes();
    let has_render = dri.iter().any(|d| d.contains("renderD"));

    if !icds.is_empty() || has_render {
        let vendor = infer_vendor_from_icds(&icds);
        let name = match vendor.as_deref() {
            Some("nvidia") => "Vulkan GPU (NVIDIA ICD)".to_string(),
            Some("amd") => "Vulkan GPU (AMD RADV/AMDVLK)".to_string(),
            Some("intel") => "Vulkan GPU (Intel ANV)".to_string(),
            Some(other) => format!("Vulkan GPU ({other})"),
            None if has_render => "Vulkan GPU (/dev/dri)".to_string(),
            None => "Vulkan GPU (ICD present)".to_string(),
        };
        devices.push(GpuDevice {
            index: 0,
            name,
            kind: GpuKind::Vulkan,
            ggml_backend: "Vulkan0".into(),
            vendor_hint: vendor,
        });
    } else if Path::new("/dev/dri").exists() {
        notes.push(
            "/dev/dri exists but no Vulkan ICDs found — install mesa-vulkan-drivers \
             or NVIDIA Vulkan ICD in the image"
                .into(),
        );
    }

    devices
}

fn parse_vulkaninfo() -> Option<Vec<GpuDevice>> {
    let output = Command::new("vulkaninfo")
        .args(["--summary"])
        .output()
        .ok()?;
    // vulkaninfo may return non-zero when incomplete; still parse stdout.
    let text = String::from_utf8_lossy(&output.stdout);
    if text.trim().is_empty() {
        let err = String::from_utf8_lossy(&output.stderr);
        if err.trim().is_empty() {
            return None;
        }
    }

    let mut raw_names: Vec<String> = Vec::new();

    // Prefer "GPU0 = …" summary lines; fall back to deviceName.
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("GPU") {
            // GPU0 = Name  (first char of rest should be a digit)
            if rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                if let Some((_, n)) = rest.split_once('=') {
                    let n = n.trim();
                    if !n.is_empty() {
                        raw_names.push(n.to_string());
                    }
                }
            }
        }
    }

    if raw_names.is_empty() {
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("deviceName") {
                if let Some((_, n)) = rest.split_once('=') {
                    let n = n.trim();
                    if !n.is_empty() && !raw_names.iter().any(|x| x == n) {
                        raw_names.push(n.to_string());
                    }
                }
            }
        }
    }

    // Dedup while preserving order.
    let mut seen = std::collections::HashSet::new();
    raw_names.retain(|n| seen.insert(n.clone()));

    let mut devices = Vec::new();
    let mut vk_idx = 0u32;
    for name in raw_names {
        let lower = name.to_ascii_lowercase();
        let is_cpu = lower.contains("llvmpipe")
            || lower.contains("lavapipe")
            || lower.contains("swiftshader");
        let kind = if is_cpu {
            GpuKind::Cpu
        } else {
            GpuKind::Vulkan
        };
        let ggml = if is_cpu {
            "CPU".to_string()
        } else {
            let id = format!("Vulkan{vk_idx}");
            vk_idx += 1;
            id
        };
        let vendor = if lower.contains("nvidia") {
            Some("nvidia".into())
        } else if lower.contains("amd") || lower.contains("radeon") {
            Some("amd".into())
        } else if lower.contains("intel") {
            Some("intel".into())
        } else {
            None
        };
        devices.push(GpuDevice {
            index: devices.len() as u32,
            name,
            kind,
            ggml_backend: ggml,
            vendor_hint: vendor,
        });
    }

    if devices.is_empty() {
        None
    } else {
        devices.sort_by_key(|d| match d.kind {
            GpuKind::Vulkan => 0,
            GpuKind::Cuda => 1,
            GpuKind::Cpu => 2,
        });
        for (i, d) in devices.iter_mut().enumerate() {
            d.index = i as u32;
        }
        Some(devices)
    }
}

fn infer_vendor_from_icds(icds: &[String]) -> Option<String> {
    let joined = icds.join(" ").to_ascii_lowercase();
    if joined.contains("nvidia") {
        Some("nvidia".into())
    } else if joined.contains("radeon") || joined.contains("amd") {
        Some("amd".into())
    } else if joined.contains("intel") {
        Some("intel".into())
    } else if joined.contains("lvp") || joined.contains("lavapipe") {
        Some("lavapipe".into())
    } else {
        None
    }
}

/// JSON for `--list-gpus` and container health endpoints.
pub fn report_json(report: &GpuReport) -> String {
    serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".into())
}
