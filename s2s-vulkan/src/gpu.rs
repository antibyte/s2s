//! Automatic GPU / accelerator detection for host and Docker.
//!
//! Detection order (when `--gpu auto`):
//! 1. Explicit env (`GGML_BACKEND`, `S2S_GPU`, …)
//! 2. NVIDIA CUDA
//! 3. oneAPI SYCL (Intel Arc preferred when Level Zero / oneAPI present)
//! 4. Vulkan devices
//! 5. CPU fallback
//!
//! **Intel Arc policy:** Prefer SYCL over Vulkan when a SYCL runtime is
//! detected. Vulkan on Arc can produce silently wrong GGML numerics for
//! Qwen3-TTS; SYCL is Intel's first-class path. Override with
//! `S2S_GPU=vulkan` or `S2S_ALLOW_INTEL_VULKAN=1`.
//!
//! Results are applied to process env (`GGML_BACKEND`, `S2S_GPU_KIND`, …) so
//! child processes (whisper/llama/qwentts) see the same choice.
//!
//! Note: selecting SYCL only sets `GGML_BACKEND=SYCL0`. The *binary* that
//! loads (tts-server / llama-server) must be built with `-DGGML_SYCL=ON`.

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
    Sycl,
    Cpu,
}

impl GpuKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::Cuda => "cuda",
            Self::Sycl => "sycl",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub kind: GpuKind,
    /// GGML-style backend id, e.g. `Vulkan0`, `CUDA0`, `SYCL0`, `CPU`.
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
    /// True when Intel GPU is present and SYCL runtime looks available.
    pub sycl_runtime_available: bool,
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
            std::env::set_var(
                "S2S_SYCL_RUNTIME",
                if self.sycl_runtime_available {
                    "1"
                } else {
                    "0"
                },
            );
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
            self.selected.name,
            self.selected.kind.as_str(),
            self.selected.ggml_backend
        );
        if self.in_container {
            info!("Running inside a container (/.dockerenv or cgroup)");
        }
        info!("  SYCL runtime available: {}", self.sycl_runtime_available);
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
    all.extend(detect_nvidia());

    // --- SYCL / oneAPI (Intel) ---
    let sycl_runtime = sycl_runtime_available();
    let sycl_devs = detect_sycl(sycl_runtime, &mut notes);
    for s in sycl_devs {
        if !all.iter().any(|a| a.ggml_backend == s.ggml_backend) {
            all.push(s);
        }
    }

    // --- Vulkan ---
    let vulkan = detect_vulkan(&mut notes);
    for v in vulkan {
        if !all
            .iter()
            .any(|a| a.ggml_backend == v.ggml_backend && a.kind == v.kind)
        {
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

    let has_intel_gpu = all.iter().any(|d| {
        d.vendor_hint
            .as_deref()
            .is_some_and(|v| v.eq_ignore_ascii_case("intel"))
            || d.name.to_ascii_lowercase().contains("intel")
            || d.name.to_ascii_lowercase().contains("arc")
    });

    if has_intel_gpu && !sycl_runtime {
        let l0 = level_zero_present();
        notes.push(if l0 {
            "Intel GPU + Level Zero driver present, but full oneAPI/SYCL toolkit \
             not detected (no sycl-ls / oneAPI root). Auto keeps Vulkan0 for now; \
             install oneAPI Base Toolkit, build qwentts with -DGGML_SYCL=ON, then \
             auto will prefer SYCL0. Until then use Supertonic for TTS."
                .into()
        } else {
            "Intel GPU detected but no oneAPI/SYCL stack. Vulkan on Arc may produce \
             wrong numerics for Qwen3-TTS — prefer Supertonic."
                .into()
        });
    }
    if has_intel_gpu && sycl_runtime {
        notes.push(
            "Intel GPU + oneAPI/SYCL stack detected — auto prefers SYCL0 over Vulkan0. \
             Ensure qwentts/llama/whisper binaries are built with -DGGML_SYCL=ON."
                .into(),
        );
    }

    let selected = select_device(pref, &all, sycl_runtime, has_intel_gpu, &mut notes);

    GpuReport {
        selected,
        all,
        in_container,
        dri_nodes,
        notes,
        sycl_runtime_available: sycl_runtime,
    }
}

fn select_device(
    pref: GpuPreference,
    all: &[GpuDevice],
    sycl_runtime: bool,
    has_intel_gpu: bool,
    notes: &mut Vec<String>,
) -> GpuDevice {
    // Explicit env wins for auto mode.
    if pref == GpuPreference::Auto {
        if let Ok(backend) = std::env::var("GGML_BACKEND") {
            if !backend.is_empty() {
                let up = backend.to_ascii_uppercase();
                let kind = if up.contains("VULKAN") {
                    GpuKind::Vulkan
                } else if up.contains("CUDA") {
                    GpuKind::Cuda
                } else if up.contains("SYCL") {
                    GpuKind::Sycl
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
            if s.starts_with("sycl") {
                if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Sycl) {
                    return d.clone();
                }
                // Soft device even if only runtime detected
                if sycl_runtime {
                    notes.push(
                        "S2S_GPU=sycl — SYCL runtime present; using SYCL0 (binary must be SYCL-built)"
                            .into(),
                    );
                    return sycl_device(0, "Intel GPU (SYCL via S2S_GPU)");
                }
                notes.push("S2S_GPU=sycl but no SYCL runtime/device detected".into());
            }
            if s.starts_with("vulkan") {
                if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Vulkan) {
                    if has_intel_gpu {
                        notes.push(
                            "S2S_GPU=vulkan on Intel — Qwen3-TTS quality may be wrong; \
                             prefer S2S_GPU=sycl when available"
                                .into(),
                        );
                    }
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

    match pref {
        GpuPreference::Auto => {
            // NVIDIA: CUDA first
            if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Cuda) {
                return d.clone();
            }

            // Intel: SYCL over Vulkan (Vulkan numerics often wrong for Qwen on Arc).
            // Override: S2S_GPU=vulkan | S2S_GPU=sycl | GGML_BACKEND=…
            if has_intel_gpu {
                if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Sycl) {
                    return d.clone();
                }
                if sycl_runtime {
                    notes.push(
                        "Intel GPU: selecting SYCL0 (runtime present). \
                         Build backends with -DGGML_SYCL=ON or fall back to Supertonic/CPU."
                            .into(),
                    );
                    return sycl_device(0, "Intel GPU (SYCL preferred over Vulkan)");
                }
                if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Vulkan) {
                    notes.push(
                        "Intel GPU without usable SYCL stack: falling back to Vulkan0 \
                         (experimental for Qwen3-TTS). Install oneAPI + SYCL-built \
                         qwentts for SYCL0, or use Supertonic TTS."
                            .into(),
                    );
                    return d.clone();
                }
            }

            // Non-Intel or remaining: any Vulkan
            if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Vulkan) {
                return d.clone();
            }
            if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Sycl) {
                return d.clone();
            }
            cpu_device()
        }
        GpuPreference::Vulkan => {
            if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Vulkan) {
                if has_intel_gpu {
                    notes.push(
                        "Forced Vulkan on Intel GPU — Qwen3-TTS may produce bad audio".into(),
                    );
                }
                return d.clone();
            }
            notes.push("Requested Vulkan not found — falling back to CPU".into());
            cpu_device()
        }
        GpuPreference::Cuda => {
            if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Cuda) {
                return d.clone();
            }
            notes.push("Requested CUDA not found — falling back to CPU".into());
            cpu_device()
        }
        GpuPreference::Sycl => {
            if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Sycl) {
                return d.clone();
            }
            if sycl_runtime {
                notes.push("Forced SYCL: runtime present, no device enum — using SYCL0".into());
                return sycl_device(0, "SYCL0 (forced)");
            }
            // Fallback chain: SYCL → Vulkan (non-ideal) → CPU
            notes.push("Requested SYCL not available — trying Vulkan, then CPU".into());
            if let Some(d) = all.iter().find(|d| d.kind == GpuKind::Vulkan) {
                notes.push("SYCL→Vulkan fallback (experimental on Intel)".into());
                return d.clone();
            }
            notes.push("SYCL and Vulkan unavailable — CPU".into());
            cpu_device()
        }
        GpuPreference::Cpu => cpu_device(),
    }
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
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

fn sycl_device(index: u32, name: &str) -> GpuDevice {
    GpuDevice {
        index,
        name: name.into(),
        kind: GpuKind::Sycl,
        ggml_backend: format!("SYCL{index}"),
        vendor_hint: Some("intel".into()),
    }
}

/// True when a usable **oneAPI/SYCL developer or runtime stack** is present.
///
/// Level Zero alone (`ze_loader.dll`) is **not** enough: that ships with the
/// Intel GPU driver but does not provide DPC++/SYCL runtimes or `sycl-ls`.
/// Prefer SYCL only when oneAPI is installed / env is set / `sycl-ls` works.
/// Force with `S2S_SYCL_FORCE=1` after deploying a SYCL-built binary + runtime.
pub fn sycl_runtime_available() -> bool {
    if env_truthy("S2S_SYCL_FORCE") {
        return true;
    }

    // sycl-ls from oneAPI (strongest signal)
    if Command::new("sycl-ls")
        .output()
        .map(|o| o.status.success() || !o.stdout.is_empty())
        .unwrap_or(false)
    {
        return true;
    }

    // Compilers imply toolkit installed
    if which_ok("icx") || which_ok("icpx") || which_ok("dpcpp") {
        return true;
    }

    // Custom path from our installer / user env
    if let Ok(p) = std::env::var("S2S_ONEAPI_ROOT") {
        if Path::new(&p).is_dir() {
            return true;
        }
    }

    // oneAPI install roots (Windows + Linux; D: custom install supported)
    let roots = [
        r"D:\Intel\oneAPI",
        r"E:\Intel\oneAPI",
        r"C:\Program Files (x86)\Intel\oneAPI",
        r"C:\Program Files\Intel\oneAPI",
        "/opt/intel/oneapi",
    ];
    for r in roots {
        if Path::new(r).is_dir() {
            return true;
        }
    }

    // Env from setvars.sh / setvars.bat
    if std::env::var_os("ONEAPI_ROOT").is_some()
        || std::env::var_os("CMPLR_ROOT").is_some()
        || std::env::var_os("SETVARS_COMPLETED").is_some()
    {
        return true;
    }

    false
}

/// Driver-level Level Zero (Arc can run L0 without full oneAPI toolkit).
pub fn level_zero_present() -> bool {
    Path::new(r"C:\Windows\System32\ze_loader.dll").exists()
        || Path::new("/usr/lib/x86_64-linux-gnu/libze_loader.so").exists()
        || Path::new("/usr/lib/libze_loader.so.1").exists()
        || Path::new("/usr/local/lib/libze_loader.so").exists()
        || std::env::var_os("ZE_ENABLE_ALT_DRIVERS").is_some()
}

fn which_ok(bin: &str) -> bool {
    Command::new(bin)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn detect_sycl(runtime_ok: bool, notes: &mut Vec<String>) -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    if !runtime_ok {
        return devices;
    }

    // Prefer sycl-ls output for real device list
    if let Some(list) = parse_sycl_ls() {
        return list;
    }

    // Infer from Vulkan/Intel presence: one Level-Zero GPU is typical on Arc
    notes.push(
        "SYCL runtime present but sycl-ls unavailable — advertising SYCL0 for Intel GPU".into(),
    );
    devices.push(sycl_device(0, "Intel GPU (Level Zero / oneAPI)"));
    devices
}

fn parse_sycl_ls() -> Option<Vec<GpuDevice>> {
    let output = Command::new("sycl-ls").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    if text.trim().is_empty() {
        return None;
    }

    let mut devices = Vec::new();
    let mut idx = 0u32;
    for line in text.lines() {
        let line = line.trim();
        // Examples:
        // [level_zero:gpu][level_zero:0] Intel(R) Arc(TM) B580 Graphics
        // [opencl:cpu] ...
        let lower = line.to_ascii_lowercase();
        if !(lower.contains("gpu") || lower.contains("level_zero") || lower.contains("opencl:gpu"))
        {
            continue;
        }
        if lower.contains("cpu") && !lower.contains("gpu") {
            continue;
        }
        let name = line.to_string();
        let vendor = if lower.contains("intel") || lower.contains("arc") {
            Some("intel".into())
        } else if lower.contains("nvidia") {
            Some("nvidia".into())
        } else if lower.contains("amd") {
            Some("amd".into())
        } else {
            None
        };
        devices.push(GpuDevice {
            index: idx,
            name,
            kind: GpuKind::Sycl,
            ggml_backend: format!("SYCL{idx}"),
            vendor_hint: vendor,
        });
        idx += 1;
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

pub fn running_in_container() -> bool {
    if Path::new("/.dockerenv").exists() {
        return true;
    }
    if std::env::var_os("S2S_DOCKER").is_some() {
        return true;
    }
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

    // NVIDIA environment variables alone are not proof of a usable GPU:
    // Dockerfiles commonly set them even without an NVIDIA runtime/device.
    let has_nvidia_device =
        Path::new("/dev/nvidia0").exists() || Path::new("/dev/nvidiactl").exists();

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

    if has_nvidia_device {
        devices.push(GpuDevice {
            index: 0,
            name: "NVIDIA GPU (device node)".into(),
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

    if let Some(list) = parse_vulkaninfo() {
        return list;
    }

    let icd_dirs = [
        "/usr/share/vulkan/icd.d",
        "/etc/vulkan/icd.d",
        "/usr/local/share/vulkan/icd.d",
        r"C:\Windows\System32\DriverStore\FileRepository",
    ];
    let mut icds = Vec::new();
    for dir in icd_dirs {
        let p = Path::new(dir);
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.ends_with(".json") || name.to_ascii_lowercase().contains("intel") {
                    icds.push(format!("{dir}/{name}"));
                }
            }
        }
    }

    if let Ok(paths) = std::env::var("VK_ICD_FILENAMES") {
        for p in paths.split([';', ':']) {
            if !p.is_empty() {
                icds.push(p.to_string());
            }
        }
    }

    let dri = list_dri_nodes();
    let has_render = dri.iter().any(|d| d.contains("renderD"));

    // Windows: Vulkan SDK / ICD often without vulkaninfo on PATH in some shells
    #[cfg(windows)]
    {
        let vk_sdk = std::env::var_os("VULKAN_SDK");
        let has_sdk = vk_sdk.is_some() || Path::new(r"C:\VulkanSDK").is_dir();
        if devices.is_empty() && (has_sdk || !icds.is_empty()) {
            // Probe common Intel name via DXGI is heavy; use generic entry
            devices.push(GpuDevice {
                index: 0,
                name: "Vulkan GPU (Windows ICD/SDK)".into(),
                kind: GpuKind::Vulkan,
                ggml_backend: "Vulkan0".into(),
                vendor_hint: None,
            });
            return devices;
        }
    }

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
    let mut cmd = Command::new("vulkaninfo");
    cmd.args(["--summary"]);
    // Windows SDK path
    #[cfg(windows)]
    {
        if let Ok(sdk) = std::env::var("VULKAN_SDK") {
            let exe = Path::new(&sdk).join("Bin").join("vulkaninfo.exe");
            if exe.is_file() {
                return parse_vulkaninfo_from(Command::new(exe).args(["--summary"]));
            }
        }
        let sdk_root = Path::new(r"C:\VulkanSDK");
        if sdk_root.is_dir() {
            if let Ok(rd) = std::fs::read_dir(sdk_root) {
                for e in rd.flatten() {
                    let exe = e.path().join("Bin").join("vulkaninfo.exe");
                    if exe.is_file() {
                        return parse_vulkaninfo_from(Command::new(exe).args(["--summary"]));
                    }
                }
            }
        }
    }
    parse_vulkaninfo_from(&mut cmd)
}

fn parse_vulkaninfo_from(cmd: &mut Command) -> Option<Vec<GpuDevice>> {
    let output = cmd.output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    if text.trim().is_empty() {
        return None;
    }

    let mut raw_names: Vec<String> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("GPU") {
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
            GpuKind::Sycl => 0,
            GpuKind::Vulkan => 1,
            GpuKind::Cuda => 2,
            GpuKind::Cpu => 3,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_backend() {
        assert_eq!(cpu_device().ggml_backend, "CPU");
    }

    #[test]
    fn sycl_device_backend() {
        assert_eq!(sycl_device(0, "x").ggml_backend, "SYCL0");
        assert_eq!(sycl_device(1, "x").ggml_backend, "SYCL1");
    }
}
