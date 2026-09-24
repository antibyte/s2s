//! Backend lifecycle controller for the local Docker speech lab.
//!
//! Browser input can only select catalog ids. Container names, images,
//! endpoints and environment are read from the validated embedded catalog.

use crate::audio::pcm::encode_wav_f32;
use crate::config::{SttApi, TtsBackend};
use crate::host_runtime::{HostAgentStatus, HostRuntimeClient};
use crate::registry::{
    endpoint_for, resolve_variant, variant_artifacts, variant_bundled, variant_protocol,
    BackendCatalog, BackendDefinition, BackendStage, BackendVariant, CatalogBackendStatus,
    HardwareProfile, StackPreset, VoiceMode,
};
use crate::runtime::{self, SharedRuntime, StackStatus};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Seconds after the last WebSocket client disconnects before managed ASR/TTS/LLM
/// containers (and optional host processes via host agent) are stopped to free
/// VRAM/RAM. Override with `S2S_LAB_IDLE_UNLOAD_SECS` (`0` disables).
fn idle_unload_delay() -> Duration {
    let secs = std::env::var("S2S_LAB_IDLE_UNLOAD_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(120);
    Duration::from_secs(secs)
}

/// Host process names the optional Windows agent should kill on idle unload
/// (comma-separated). Default includes Qwen SYCL `tts-server`.
fn idle_unload_host_processes() -> Vec<String> {
    std::env::var("S2S_IDLE_UNLOAD_HOST_PROCESSES")
        .unwrap_or_else(|_| "tts-server".into())
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn host_unload_request_path() -> PathBuf {
    let root = std::env::var("S2S_DATA_DIR").unwrap_or_else(|_| "/data".into());
    PathBuf::from(root).join("idle-unload.request")
}

fn host_reload_request_path() -> PathBuf {
    let root = std::env::var("S2S_DATA_DIR").unwrap_or_else(|_| "/data".into());
    PathBuf::from(root).join("idle-reload.request")
}

/// Extract host port from an endpoint like `http://host.docker.internal:8083/v1/...`.
fn endpoint_host_port(endpoint: &str) -> Option<u16> {
    let url = endpoint.trim();
    let without_scheme = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let host_port = without_scheme.split('/').next().unwrap_or("");
    let port = host_port.rsplit_once(':').map(|(_, p)| p)?;
    port.parse().ok()
}

fn is_host_side_endpoint(endpoint: &str) -> bool {
    let lower = endpoint.to_ascii_lowercase();
    lower.contains("host.docker.internal")
        || lower.contains("127.0.0.1")
        || lower.contains("localhost")
}

/// Multi-GB model downloads (Voxtral/Higgs/…) need no whole-body timeout and
/// optional resume. Keep API/health timeouts on the regular `client`.
fn build_download_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_keepalive(Duration::from_secs(30))
        // reqwest 0.12 applies this to the full response body. Multi-GB HF
        // artifacts (Voxtral ~8.9 GB, Higgs ~9.3 GB) need hours on slow links.
        // Transient stalls still retry with Range resume.
        .timeout(Duration::from_secs(24 * 60 * 60))
        .build()
        .context("build model download HTTP client")
}

fn stage_label(stage: BackendStage) -> &'static str {
    match stage {
        BackendStage::Asr => "asr",
        BackendStage::Tts => "tts",
        BackendStage::Llm => "llm",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionPhase {
    Idle,
    Preparing,
    Draining,
    Stopping,
    Starting,
    Warming,
    Ready,
    Rollback,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageTransition {
    pub stage: BackendStage,
    pub phase: TransitionPhase,
    pub backend_id: String,
    pub variant_id: String,
    pub message: String,
}

/// Result of bring-up for one stage — commit uses the same variant/endpoint.
#[derive(Debug, Clone)]
struct StagePlan {
    stage: BackendStage,
    backend_id: String,
    variant_id: String,
    endpoint: String,
    container: String,
    started_container: Option<String>,
    started_host: bool,
    already_active: bool,
}

impl StageTransition {
    fn idle(stage: BackendStage) -> Self {
        Self {
            stage,
            phase: TransitionPhase::Idle,
            backend_id: String::new(),
            variant_id: String::new(),
            message: "idle".into(),
        }
    }
}

pub(crate) fn stage_is_ready(
    active: Option<&ActiveBackend>,
    transition: &StageTransition,
    runtime_id: &str,
) -> bool {
    active.is_some_and(|backend| {
        !runtime_id.is_empty()
            && backend.backend_id == runtime_id
            && matches!(
                transition.phase,
                TransitionPhase::Idle | TransitionPhase::Ready
            )
    })
}

fn activation_error_code(message: &str) -> &'static str {
    if message.contains("module_not_provisioned") {
        "module_not_provisioned"
    } else if message.contains("host_module_delivery_pending") {
        "host_module_delivery_pending"
    } else if message.contains("not installed") {
        "model_not_installed"
    } else {
        "stack_activation_failed"
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LabEvent {
    StackTransition {
        stage: BackendStage,
        phase: TransitionPhase,
        backend_id: String,
        variant_id: String,
        message: String,
    },
    BackendHealth {
        stage: BackendStage,
        backend_id: String,
        ok: bool,
        message: String,
    },
    DownloadProgress {
        backend_id: String,
        artifact: String,
        downloaded: u64,
        total: u64,
    },
    /// Scheduled after the last client disconnects.
    IdleUnloadScheduled { delay_secs: u64, message: String },
    /// Managed model containers were stopped to free memory.
    ModelsUnloaded { message: String },
    /// Containers restarted after a client reconnected.
    ModelsReloaded { message: String },
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivateStackRequest {
    #[serde(default, alias = "asr")]
    pub asr_id: Option<String>,
    #[serde(default, alias = "tts")]
    pub tts_id: Option<String>,
    #[serde(default, alias = "llm")]
    pub llm_id: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActiveBackend {
    pub backend_id: String,
    pub variant_id: String,
    pub accelerator: String,
    pub endpoint: String,
    pub container: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LabStackStatus {
    pub asr: Option<ActiveBackend>,
    pub tts: Option<ActiveBackend>,
    pub llm: Option<ActiveBackend>,
    pub runtime: StackStatus,
    pub asr_transition: StageTransition,
    pub tts_transition: StageTransition,
    pub llm_transition: StageTransition,
    pub ok: bool,
    pub error_code: String,
    pub message: String,
    pub last_transition_error: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LabCatalogResponse {
    pub schema_version: u32,
    pub catalog_revision: String,
    pub hardware: HardwareProfile,
    pub backends: Vec<CatalogBackendStatus>,
    pub presets: Vec<StackPreset>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelActionResponse {
    pub backend_id: String,
    pub state: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub message: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RuntimeProvisionStatus {
    pub state: String,
    #[serde(default)]
    pub image: String,
    #[serde(default)]
    pub image_download_size_bytes: u64,
    #[serde(default)]
    pub image_downloaded_bytes: u64,
    #[serde(default)]
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModuleStatusResponse {
    pub operation_id: String,
    pub backend_id: String,
    pub variant_id: String,
    pub state: String,
    pub model_state: String,
    pub runtime_state: String,
    pub model_downloaded_bytes: u64,
    pub model_total_bytes: u64,
    pub image_download_size_bytes: u64,
    pub image_downloaded_bytes: u64,
    pub error: String,
}

/// Optional body for `POST /api/v1/models/{id}/download`.
/// Tokens are accepted for gated Hugging Face artifacts and are never returned
/// by the catalog API.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDownloadRequest {
    /// Hugging Face access token (`hf_…`). Optional when `HF_TOKEN` / `S2S_HF_TOKEN`
    /// is already set server-side or remembered for this lab process.
    #[serde(default)]
    pub hf_token: Option<String>,
    /// When true (default), keep a non-empty token in memory for later downloads
    /// in this process. Never written to the catalog response or logs.
    #[serde(default = "default_remember_hf_token")]
    pub remember: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleInstallRequest {
    #[serde(default)]
    pub catalog_revision: String,
    #[serde(default)]
    pub hf_token: Option<String>,
    #[serde(default = "default_remember_hf_token")]
    pub remember: bool,
}

fn default_remember_hf_token() -> bool {
    true
}

#[derive(Debug, Clone)]
struct ModelDownloadRecord {
    variant_id: String,
    state: String,
    downloaded_bytes: u64,
    total_bytes: u64,
    error: String,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct ControllerState {
    asr: Option<ActiveBackend>,
    tts: Option<ActiveBackend>,
    llm: Option<ActiveBackend>,
    asr_transition: StageTransition,
    tts_transition: StageTransition,
    llm_transition: StageTransition,
    last_transition_error: String,
}

impl Default for ControllerState {
    fn default() -> Self {
        Self {
            asr: None,
            tts: None,
            llm: None,
            asr_transition: StageTransition::idle(BackendStage::Asr),
            tts_transition: StageTransition::idle(BackendStage::Tts),
            llm_transition: StageTransition::idle(BackendStage::Llm),
            last_transition_error: String::new(),
        }
    }
}

#[async_trait]
pub trait ContainerControl: Send + Sync {
    async fn validate(
        &self,
        container: &str,
        stage: BackendStage,
        backend_id: &str,
        image: &str,
    ) -> Result<()>;
    async fn start(&self, container: &str) -> Result<()>;
    async fn stop(&self, container: &str, timeout: Duration) -> Result<()>;
    async fn running(&self, container: &str) -> Result<bool>;
    /// Running containers with `s2s.lab.managed=true` and stage asr|tts|llm.
    async fn list_managed_running(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
    async fn runtime_status(&self, _variant_id: &str) -> Result<RuntimeProvisionStatus> {
        Ok(RuntimeProvisionStatus {
            state: "missing".into(),
            ..RuntimeProvisionStatus::default()
        })
    }
    async fn install(&self, _variant_id: &str) -> Result<RuntimeProvisionStatus> {
        Err(anyhow!("managed runtime installation is unavailable"))
    }
    async fn remove(&self, _variant_id: &str) -> Result<()> {
        Err(anyhow!("managed runtime removal is unavailable"))
    }
}

#[derive(Debug, Clone)]
struct ModuleInstallRecord {
    operation_id: String,
    variant_id: String,
    state: String,
    error: String,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug)]
struct DisabledContainerControl;

#[async_trait]
impl ContainerControl for DisabledContainerControl {
    async fn validate(
        &self,
        _container: &str,
        _stage: BackendStage,
        _backend_id: &str,
        _image: &str,
    ) -> Result<()> {
        Ok(())
    }

    async fn start(&self, _container: &str) -> Result<()> {
        Ok(())
    }

    async fn stop(&self, _container: &str, _timeout: Duration) -> Result<()> {
        Ok(())
    }

    async fn running(&self, _container: &str) -> Result<bool> {
        Ok(false)
    }
}

#[derive(Debug, Clone)]
pub struct DockerProxyControl {
    client: reqwest::Client,
    base_url: String,
    token: Option<String>,
}

impl DockerProxyControl {
    pub fn new(base_url: String) -> Result<Self> {
        let base_url = base_url.trim_end_matches('/').to_string();
        if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
            return Err(anyhow!("S2S_DOCKER_PROXY_URL must be http(s)"));
        }
        let token = std::env::var("S2S_DOCKER_PROXY_TOKEN_FILE")
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map(|token| token.trim().to_string())
            .filter(|token| !token.is_empty());
        if base_url != "http://docker-proxy:2375" && token.is_none() {
            return Err(anyhow!("S2S_DOCKER_PROXY_TOKEN_FILE is required"));
        }
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(20))
                .build()?,
            base_url,
            token,
        })
    }

    fn request(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let request = self.client.request(method, url);
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    async fn post_action(&self, container: &str, action: &str) -> Result<()> {
        self.inspect_managed(container).await?;
        let url = format!("{}/containers/{container}/{action}", self.base_url);
        let response = self
            .request(reqwest::Method::POST, &url)
            .send()
            .await
            .with_context(|| format!("Docker proxy POST {url}"))?;
        if response.status().is_success() || response.status().as_u16() == 304 {
            return Ok(());
        }
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if status.as_u16() == 404 {
            return Err(anyhow!(
                "managed container '{container}' does not exist; create the lab profiles first"
            ));
        }
        Err(anyhow!("Docker proxy {action} {status}: {body}"))
    }

    async fn inspect_managed(&self, container: &str) -> Result<serde_json::Value> {
        let url = format!("{}/containers/{container}/json", self.base_url);
        let response = self.request(reqwest::Method::GET, &url).send().await?;
        if response.status().as_u16() == 404 {
            return Err(anyhow!(
                "managed container '{container}' does not exist; create the lab profiles first"
            ));
        }
        if !response.status().is_success() {
            return Err(anyhow!(
                "Docker inspect for '{container}' failed: {}",
                response.status()
            ));
        }
        let value: serde_json::Value = response.json().await?;
        let labels = &value["Config"]["Labels"];
        if labels["s2s.lab.managed"].as_str() != Some("true")
            || !matches!(labels["stage"].as_str(), Some("asr" | "tts" | "llm"))
            || labels["backend-id"]
                .as_str()
                .is_none_or(|backend_id| backend_id.is_empty())
        {
            return Err(anyhow!(
                "refusing lifecycle action for '{container}': required s2s.lab.managed/stage/backend-id labels are missing"
            ));
        }
        Ok(value)
    }
}

#[async_trait]
impl ContainerControl for DockerProxyControl {
    async fn validate(
        &self,
        container: &str,
        stage: BackendStage,
        backend_id: &str,
        image: &str,
    ) -> Result<()> {
        let value = self.inspect_managed(container).await?;
        validate_managed_target(&value, container, stage, backend_id, image)
    }

    async fn start(&self, container: &str) -> Result<()> {
        self.post_action(container, "start").await
    }

    async fn stop(&self, container: &str, timeout: Duration) -> Result<()> {
        let secs = timeout.as_secs().max(1);
        self.post_action(container, &format!("stop?t={secs}")).await
    }

    async fn running(&self, container: &str) -> Result<bool> {
        let value = match self.inspect_managed(container).await {
            Ok(value) => value,
            Err(error) if error.to_string().contains("does not exist") => return Ok(false),
            Err(error) => return Err(error),
        };
        Ok(value["State"]["Running"].as_bool().unwrap_or(false))
    }

    async fn list_managed_running(&self) -> Result<Vec<String>> {
        // Docker Engine API filter JSON must be URL-encoded.
        let filters = serde_json::json!({
            "label": ["s2s.lab.managed=true"],
            "status": ["running"]
        });
        let encoded = urlencoding_encode(&filters.to_string());
        let url = format!("{}/containers/json?filters={encoded}", self.base_url);
        let response = self
            .request(reqwest::Method::GET, &url)
            .send()
            .await
            .with_context(|| format!("Docker proxy list {url}"))?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "Docker list managed containers failed: {}",
                response.status()
            ));
        }
        let values: Vec<serde_json::Value> = response.json().await?;
        let mut names = Vec::new();
        for value in values {
            let labels = &value["Labels"];
            let stage = labels["stage"].as_str().unwrap_or_default();
            if !matches!(stage, "asr" | "tts" | "llm") {
                continue;
            }
            if labels["backend-id"]
                .as_str()
                .is_none_or(|backend_id| backend_id.is_empty())
            {
                continue;
            }
            let name = value["Names"]
                .as_array()
                .and_then(|names| names.first())
                .and_then(|name| name.as_str())
                .unwrap_or_default()
                .trim_start_matches('/')
                .to_string();
            if !name.is_empty() {
                names.push(name);
            }
        }
        Ok(names)
    }

    async fn runtime_status(&self, variant_id: &str) -> Result<RuntimeProvisionStatus> {
        let url = format!(
            "{}/s2s/modules/{}",
            self.base_url,
            urlencoding_encode(variant_id)
        );
        let response = self
            .request(reqwest::Method::GET, &url)
            .send()
            .await
            .with_context(|| format!("controller GET {url}"))?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "controller runtime status returned {}",
                response.status()
            ));
        }
        Ok(response.json().await?)
    }

    async fn install(&self, variant_id: &str) -> Result<RuntimeProvisionStatus> {
        let url = format!(
            "{}/s2s/modules/{}/install",
            self.base_url,
            urlencoding_encode(variant_id)
        );
        let response = self
            .request(reqwest::Method::POST, &url)
            // Pulling a multi-GB runtime image can take much longer than the
            // 20-second timeout used for ordinary controller operations.
            .timeout(Duration::from_secs(2 * 60 * 60))
            .send()
            .await
            .with_context(|| format!("controller POST {url}"))?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("controller install {status}: {body}"));
        }
        Ok(serde_json::from_str(&body)?)
    }

    async fn remove(&self, variant_id: &str) -> Result<()> {
        let url = format!(
            "{}/s2s/modules/{}",
            self.base_url,
            urlencoding_encode(variant_id)
        );
        let response = self
            .request(reqwest::Method::DELETE, &url)
            .send()
            .await
            .with_context(|| format!("controller DELETE {url}"))?;
        if response.status().is_success() {
            return Ok(());
        }
        Err(anyhow!(
            "controller remove returned {}: {}",
            response.status(),
            response.text().await.unwrap_or_default()
        ))
    }
}

/// Minimal URL-encoding for Docker filter query values (no extra dependency).
fn urlencoding_encode(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() * 3);
    for byte in raw.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn validate_managed_target(
    value: &serde_json::Value,
    container: &str,
    stage: BackendStage,
    backend_id: &str,
    image: &str,
) -> Result<()> {
    let labels = &value["Config"]["Labels"];
    let actual_image = value["Config"]["Image"].as_str().unwrap_or_default();
    if labels["stage"].as_str() != Some(stage_label(stage))
        || labels["backend-id"].as_str() != Some(backend_id)
        || actual_image != image
    {
        return Err(anyhow!(
            "refusing '{container}': expected stage={} backend-id={backend_id} image={image}, got stage={} backend-id={} image={actual_image}",
            stage_label(stage),
            labels["stage"].as_str().unwrap_or(""),
            labels["backend-id"].as_str().unwrap_or("")
        ));
    }
    Ok(())
}

fn provisioned_image<'a>(runtime: &'a RuntimeProvisionStatus, catalog_image: &'a str) -> &'a str {
    if runtime.image.is_empty() {
        catalog_image
    } else {
        &runtime.image
    }
}

#[derive(Default)]
struct IdleUnloadState {
    /// Bumped whenever a timer is cancelled or replaced.
    generation: u64,
    task: Option<JoinHandle<()>>,
    /// True after containers were stopped due to idle; restart on next session.
    models_parked: bool,
}

#[derive(Clone)]
pub struct LabController {
    catalog: Arc<BackendCatalog>,
    catalog_revision: String,
    hardware: HardwareProfile,
    /// Shared with the gateway ASR/TTS proxy.
    pub(crate) runtime: SharedRuntime,
    control: Arc<dyn ContainerControl>,
    host_runtime: HostRuntimeClient,
    docker_control_enabled: bool,
    /// Shared HTTP client for health probes and gateway proxying.
    pub(crate) client: reqwest::Client,
    download_client: reqwest::Client,
    state: Arc<RwLock<ControllerState>>,
    asr_lock: Arc<Mutex<()>>,
    tts_lock: Arc<Mutex<()>>,
    llm_lock: Arc<Mutex<()>>,
    model_lock: Arc<Mutex<()>>,
    downloads: Arc<RwLock<HashMap<String, ModelDownloadRecord>>>,
    modules: Arc<RwLock<HashMap<String, ModuleInstallRecord>>>,
    /// Session-scoped Hugging Face token from the Lab UI (never catalog-exported).
    hf_token: Arc<RwLock<Option<String>>>,
    desired_llm: Arc<RwLock<String>>,
    events: broadcast::Sender<LabEvent>,
    /// Live PCM WebSocket sessions (voice lab clients).
    session_count: Arc<AtomicUsize>,
    idle_unload: Arc<Mutex<IdleUnloadState>>,
}

impl LabController {
    pub fn new(
        catalog: BackendCatalog,
        hardware: HardwareProfile,
        runtime: SharedRuntime,
    ) -> Result<Self> {
        let (control, docker_control_enabled): (Arc<dyn ContainerControl>, bool) =
            if let Ok(url) = std::env::var("S2S_DOCKER_PROXY_URL") {
                (Arc::new(DockerProxyControl::new(url)?), true)
            } else {
                (Arc::new(DisabledContainerControl), false)
            };
        Self::with_control(catalog, hardware, runtime, control, docker_control_enabled)
    }

    fn with_control(
        catalog: BackendCatalog,
        hardware: HardwareProfile,
        runtime: SharedRuntime,
        control: Arc<dyn ContainerControl>,
        docker_control_enabled: bool,
    ) -> Result<Self> {
        catalog.validate()?;
        let catalog_revision = hex::encode(Sha256::digest(serde_json::to_vec(&catalog)?));
        let desired_llm = runtime
            .try_read()
            .map(|state| state.llm_id.clone())
            .unwrap_or_else(|_| "local-fallback".into());
        let (events, _) = broadcast::channel(64);
        Ok(Self {
            catalog: Arc::new(catalog),
            catalog_revision,
            hardware,
            runtime,
            control,
            host_runtime: HostRuntimeClient::default(),
            docker_control_enabled,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()?,
            download_client: build_download_client()?,
            state: Arc::new(RwLock::new(ControllerState::default())),
            asr_lock: Arc::new(Mutex::new(())),
            tts_lock: Arc::new(Mutex::new(())),
            llm_lock: Arc::new(Mutex::new(())),
            model_lock: Arc::new(Mutex::new(())),
            downloads: Arc::new(RwLock::new(HashMap::new())),
            modules: Arc::new(RwLock::new(HashMap::new())),
            hf_token: Arc::new(RwLock::new(None)),
            desired_llm: Arc::new(RwLock::new(desired_llm)),
            events,
            session_count: Arc::new(AtomicUsize::new(0)),
            idle_unload: Arc::new(Mutex::new(IdleUnloadState::default())),
        })
    }

    fn env_hf_token() -> Option<String> {
        for key in ["S2S_HF_TOKEN", "HF_TOKEN"] {
            if let Ok(value) = std::env::var(key) {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
        None
    }

    async fn hf_token_is_configured(&self) -> bool {
        if Self::env_hf_token().is_some() {
            return true;
        }
        self.hf_token
            .read()
            .await
            .as_ref()
            .is_some_and(|token| !token.trim().is_empty())
    }

    /// Priority: request body → session memory → server env.
    async fn resolve_hf_token(&self, request_token: Option<&str>) -> Option<String> {
        if let Some(token) = request_token
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            return Some(token.to_string());
        }
        if let Some(token) = self.hf_token.read().await.clone() {
            let trimmed = token.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        Self::env_hf_token()
    }

    /// Call when a PCM WebSocket client connects. Cancels idle unload and
    /// restarts parked model containers if needed.
    pub async fn session_connected(&self) {
        let count = self.session_count.fetch_add(1, Ordering::SeqCst) + 1;
        info!(sessions = count, "lab session connected");
        self.cancel_idle_unload().await;
        if let Err(error) = self.ensure_models_loaded_if_parked().await {
            warn!("failed to reload parked models after reconnect: {error:#}");
            let _ = self.events.send(LabEvent::BackendHealth {
                stage: BackendStage::Tts,
                backend_id: String::new(),
                ok: false,
                message: format!("Modell-Reload nach Reconnect fehlgeschlagen: {error:#}"),
            });
        }
    }

    /// Call when a PCM WebSocket client disconnects. When no sessions remain,
    /// schedule model unload after the idle delay (default 2 minutes).
    pub async fn session_disconnected(&self) {
        let prev = self.session_count.load(Ordering::SeqCst);
        let count = if prev == 0 {
            0
        } else {
            self.session_count.fetch_sub(1, Ordering::SeqCst) - 1
        };
        info!(sessions = count, "lab session disconnected");
        if count == 0 {
            self.schedule_idle_unload().await;
        }
    }

    async fn cancel_idle_unload(&self) {
        let mut idle = self.idle_unload.lock().await;
        idle.generation = idle.generation.wrapping_add(1);
        if let Some(task) = idle.task.take() {
            task.abort();
            info!("cancelled idle model unload timer");
        }
    }

    async fn schedule_idle_unload(&self) {
        let delay = idle_unload_delay();
        if delay.is_zero() {
            info!("idle model unload disabled (S2S_LAB_IDLE_UNLOAD_SECS=0)");
            return;
        }

        let mut idle = self.idle_unload.lock().await;
        idle.generation = idle.generation.wrapping_add(1);
        let generation = idle.generation;
        if let Some(task) = idle.task.take() {
            task.abort();
        }

        let controller = self.clone();
        idle.task = Some(tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            if controller.session_count.load(Ordering::SeqCst) > 0 {
                return;
            }
            {
                let idle = controller.idle_unload.lock().await;
                if idle.generation != generation {
                    return;
                }
            }
            if let Err(error) = controller.unload_idle_models().await {
                warn!("idle model unload failed: {error:#}");
            }
        }));

        let secs = delay.as_secs();
        info!(delay_secs = secs, "scheduled idle model unload");
        let _ = self.events.send(LabEvent::IdleUnloadScheduled {
            delay_secs: secs,
            message: format!(
                "Keine Clients — Modelle werden in {secs}s entladen, um Speicher freizugeben"
            ),
        });
    }

    /// All non-empty managed container names declared for a backend id.
    fn containers_for_backend(&self, backend_id: &str) -> Vec<String> {
        let Some(backend) = self.catalog.find(backend_id) else {
            return Vec::new();
        };
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for variant in &backend.variants {
            if variant.container.is_empty() || !seen.insert(variant.container.clone()) {
                continue;
            }
            out.push(variant.container.clone());
        }
        out
    }

    /// Every managed ASR/TTS/LLM container name in the catalog.
    fn all_catalog_model_containers(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for backend in &self.catalog.backends {
            if !matches!(
                backend.stage,
                BackendStage::Asr | BackendStage::Tts | BackendStage::Llm
            ) {
                continue;
            }
            for variant in &backend.variants {
                if variant.container.is_empty() || !seen.insert(variant.container.clone()) {
                    continue;
                }
                out.push(variant.container.clone());
            }
        }
        out
    }

    async fn stop_container_best_effort(&self, container: &str, backend_id: &str) -> bool {
        let running = self.control.running(container).await.unwrap_or(false);
        if !running {
            return false;
        }
        info!(
            container,
            backend_id, "idle unload: stopping managed container"
        );
        let stop = self.control.stop(container, Duration::from_secs(15));
        match tokio::time::timeout(Duration::from_secs(25), stop).await {
            Ok(Ok(())) => true,
            Ok(Err(error)) => {
                warn!(container, error = %error, "idle unload: stop failed");
                false
            }
            Err(_) => {
                warn!(container, "idle unload: stop timed out");
                false
            }
        }
    }

    async fn write_host_unload_request(
        &self,
        actives: &[ActiveBackend],
        stopped_containers: &[String],
    ) {
        let ports: Vec<u16> = actives
            .iter()
            .filter_map(|active| {
                if is_host_side_endpoint(&active.endpoint) {
                    endpoint_host_port(&active.endpoint)
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        // Always include common host TTS ports used by start_qwen_*.ps1 / VibeVoice.
        let mut ports = ports;
        for extra in [8083u16, 8086, 8089] {
            if !ports.contains(&extra) {
                ports.push(extra);
            }
        }
        ports.sort_unstable();

        let processes = idle_unload_host_processes();
        let payload = serde_json::json!({
            "action": "unload",
            "processes": processes,
            "ports": ports,
            "containers": stopped_containers,
            "backends": actives.iter().map(|a| &a.backend_id).collect::<Vec<_>>(),
            "ts_unix": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        });
        let path = host_unload_request_path();
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        // Clear any pending reload so the host agent does not immediately restart.
        let _ = tokio::fs::remove_file(host_reload_request_path()).await;
        match tokio::fs::write(&path, payload.to_string()).await {
            Ok(()) => info!(
                path = %path.display(),
                "idle unload: wrote host unload request (run scripts/host_idle_agent.ps1 on Windows)"
            ),
            Err(error) => warn!(
                path = %path.display(),
                error = %error,
                "idle unload: failed to write host unload request"
            ),
        }
    }

    async fn write_host_reload_request(&self, actives: &[ActiveBackend]) {
        let ports: Vec<u16> = actives
            .iter()
            .filter_map(|active| {
                if is_host_side_endpoint(&active.endpoint) {
                    endpoint_host_port(&active.endpoint)
                } else {
                    None
                }
            })
            .collect();
        if ports.is_empty() && !actives.iter().any(|a| a.container.is_empty()) {
            return;
        }
        let payload = serde_json::json!({
            "action": "reload",
            "ports": ports,
            "backends": actives.iter().map(|a| {
                serde_json::json!({
                    "backend_id": a.backend_id,
                    "endpoint": a.endpoint,
                    "container": a.container,
                })
            }).collect::<Vec<_>>(),
            "ts_unix": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        });
        let path = host_reload_request_path();
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let _ = tokio::fs::remove_file(host_unload_request_path()).await;
        if let Err(error) = tokio::fs::write(&path, payload.to_string()).await {
            warn!(
                path = %path.display(),
                error = %error,
                "idle reload: failed to write host reload request"
            );
        }
    }

    async fn unload_idle_models(&self) -> Result<()> {
        if self.session_count.load(Ordering::SeqCst) > 0 {
            return Ok(());
        }
        // Wait briefly for any in-flight turn bookkeeping (pipeline is already aborted).
        let _ = self
            .runtime
            .read()
            .await
            .turns
            .wait_idle(Duration::from_secs(5))
            .await;

        if self.session_count.load(Ordering::SeqCst) > 0 {
            return Ok(());
        }

        let actives: Vec<ActiveBackend> = {
            let state = self.state.read().await;
            [&state.asr, &state.tts, &state.llm]
                .into_iter()
                .flatten()
                .cloned()
                .collect()
        };

        let mut to_stop: HashSet<String> = HashSet::new();
        // 1) Every container declared for the active stack backends (covers host
        //    variants that leave container="" while a sibling sidecar still runs).
        for active in &actives {
            for container in self.containers_for_backend(&active.backend_id) {
                to_stop.insert(container);
            }
            if !active.container.is_empty() {
                to_stop.insert(active.container.clone());
            }
        }
        // 2) Full park: any catalog model container still running.
        for container in self.all_catalog_model_containers() {
            to_stop.insert(container);
        }
        // 3) Docker truth: any currently running managed lab container.
        if self.docker_control_enabled {
            match self.control.list_managed_running().await {
                Ok(running) => {
                    for name in running {
                        to_stop.insert(name);
                    }
                }
                Err(error) => warn!("idle unload: list managed containers failed: {error:#}"),
            }
        }

        let mut stopped = Vec::new();
        if self.docker_control_enabled {
            for container in to_stop {
                let backend_id = self
                    .catalog
                    .backends
                    .iter()
                    .find(|b| b.variants.iter().any(|v| v.container == container))
                    .map(|b| b.id.as_str())
                    .unwrap_or("-");
                if self
                    .stop_container_best_effort(&container, backend_id)
                    .await
                {
                    stopped.push(format!("{backend_id} ({container})"));
                }
            }
        }

        // 4) Stop only child processes owned by the allowlisted host agent.
        if self
            .host_runtime
            .status()
            .await
            .ok()
            .flatten()
            .is_some_and(|status| status.fresh())
        {
            if let Err(error) = self.host_runtime.stop_all().await {
                warn!(error = %error, "idle unload: host agent stop_all failed");
            }
        }

        // 5) Legacy request remains for older agents and manually started Qwen.
        self.write_host_unload_request(&actives, &stopped).await;

        {
            let mut idle = self.idle_unload.lock().await;
            idle.models_parked = true;
            idle.task = None;
        }

        let message = if stopped.is_empty() {
            "Idle: Host-Unload angefordert (Container waren bereits aus); tts-server o.ä. via host_idle_agent.ps1".into()
        } else {
            format!(
                "Idle: Modelle entladen — {} (+ Host-Unload-Request für tts-server/Ports)",
                stopped.join(", ")
            )
        };
        info!("{message}");
        let _ = self.events.send(LabEvent::ModelsUnloaded { message });
        Ok(())
    }

    async fn ensure_models_loaded_if_parked(&self) -> Result<()> {
        let parked = {
            let idle = self.idle_unload.lock().await;
            idle.models_parked
        };
        if !parked {
            return Ok(());
        }

        let actives: Vec<ActiveBackend> = {
            let state = self.state.read().await;
            [&state.asr, &state.tts, &state.llm]
                .into_iter()
                .flatten()
                .cloned()
                .collect()
        };

        let mut restarted = Vec::new();
        if self.docker_control_enabled {
            for active in &actives {
                // Host variants: prefer starting a managed sibling container when present.
                let candidates: Vec<String> = if active.container.is_empty() {
                    let managed_host = self
                        .catalog
                        .find(&active.backend_id)
                        .and_then(|backend| {
                            backend
                                .variants
                                .iter()
                                .find(|variant| variant.id == active.variant_id)
                        })
                        .is_some_and(|variant| !variant.host_profile.is_empty());
                    if managed_host {
                        Vec::new()
                    } else {
                        self.containers_for_backend(&active.backend_id)
                    }
                } else {
                    vec![active.container.clone()]
                };
                for container in candidates {
                    let running = self.control.running(&container).await.unwrap_or(false);
                    if running {
                        if !restarted.contains(&active.backend_id) {
                            restarted.push(active.backend_id.clone());
                        }
                        continue;
                    }
                    info!(
                        container = %container,
                        backend_id = %active.backend_id,
                        "reloading parked model container after reconnect"
                    );
                    match self.control.start(&container).await {
                        Ok(()) => {
                            if let Some(backend) = self.catalog.find(&active.backend_id) {
                                if let Some(variant) = backend.variants.iter().find(|variant| {
                                    variant.id == active.variant_id
                                        || variant.container == container
                                }) {
                                    let endpoint = if active.endpoint.is_empty() {
                                        endpoint_for(variant, self.hardware.in_container)
                                    } else {
                                        active.endpoint.clone()
                                    };
                                    if let Err(error) =
                                        self.wait_for_health(backend, variant, &endpoint).await
                                    {
                                        warn!(
                                            backend_id = %active.backend_id,
                                            error = %error,
                                            "health wait after idle reload failed"
                                        );
                                    } else if let Err(error) =
                                        self.warm_backend(backend, &endpoint).await
                                    {
                                        warn!(
                                            backend_id = %active.backend_id,
                                            error = %error,
                                            "warmup after idle reload failed"
                                        );
                                    }
                                }
                            }
                            if !restarted.contains(&active.backend_id) {
                                restarted.push(active.backend_id.clone());
                            }
                            break;
                        }
                        Err(error) => {
                            warn!(
                                container = %container,
                                error = %error,
                                "failed to start parked container — trying next"
                            );
                        }
                    }
                }
            }
        }

        for active in &actives {
            let Some(backend) = self.catalog.find(&active.backend_id) else {
                continue;
            };
            let Some(variant) = backend
                .variants
                .iter()
                .find(|variant| variant.id == active.variant_id)
            else {
                continue;
            };
            if variant.host_profile.is_empty() {
                continue;
            }
            let active_voice = if backend.stage == BackendStage::Tts {
                self.runtime.read().await.cfg.supertonic_voice.clone()
            } else {
                String::new()
            };
            self.host_runtime
                .start(
                    backend.stage,
                    &active.backend_id,
                    variant,
                    &active.endpoint,
                    &active_voice,
                )
                .await
                .with_context(|| format!("reload host backend {}", active.backend_id))?;
            self.wait_for_health(backend, variant, &active.endpoint)
                .await
                .with_context(|| format!("reload host health {}", active.backend_id))?;
            self.warm_backend(backend, &active.endpoint)
                .await
                .with_context(|| format!("reload host warmup {}", active.backend_id))?;
            if !restarted.contains(&active.backend_id) {
                restarted.push(active.backend_id.clone());
            }
        }

        // Ask host agent to bring back native host TTS (Qwen SYCL, …) if needed.
        self.write_host_reload_request(&actives).await;

        {
            let mut idle = self.idle_unload.lock().await;
            idle.models_parked = false;
        }

        if !restarted.is_empty() {
            let message = format!("Modelle wieder geladen: {}", restarted.join(", "));
            info!("{message}");
            let _ = self.events.send(LabEvent::ModelsReloaded { message });
        }
        Ok(())
    }

    /// Heuristic ASR+TTS suggestions for setup (not benchmarks).
    pub async fn suggestions(
        &self,
        query: crate::suggest::SuggestionQuery,
    ) -> crate::suggest::SuggestionsResponse {
        let catalog = self.catalog().await;
        crate::suggest::build_suggestions(
            catalog.hardware,
            &catalog.backends,
            &catalog.presets,
            &query,
        )
    }

    /// Best-effort capability profile for AuraGo setup suggestions.
    ///
    /// Re-probes RAM/VRAM, merges host-agent heartbeat, and recomputes `tier`.
    /// See `docs/aurago-integration.md`.
    pub async fn capability(&self) -> crate::registry::CapabilityProfile {
        let mut profile = self.hardware.clone();
        profile.apply_static_probes();
        match self.host_runtime.status().await {
            Ok(Some(status)) if status.fresh() => {
                profile.apply_host_agent(true, status.profiles.clone());
                // Host agent sees the real GPU; Docker Desktop often only exposes CPU.
                for accelerator in &status.accelerators {
                    let acc = accelerator.trim();
                    if acc.is_empty() {
                        continue;
                    }
                    if !profile.accelerators.iter().any(|known| known == acc) {
                        profile.accelerators.push(acc.to_string());
                    }
                }
                if !status.device_name.trim().is_empty() {
                    profile.device_name = status.device_name.clone();
                }
                profile.recompute_tier();
            }
            _ => {
                profile.apply_host_agent(false, Vec::<String>::new());
            }
        }
        profile
    }

    pub async fn catalog(&self) -> LabCatalogResponse {
        let capability = self.capability().await;
        // Resolve variants against the base hardware filters (accelerators/vendor),
        // but report the enriched capability snapshot to clients.
        let mut backends = self.catalog.resolved(&self.hardware);
        if let Ok(voices) = discover_xtts_voice_ids_at(&models_root()).await {
            if let Some(status) = backends
                .iter_mut()
                .find(|status| status.backend.id == "xtts-v2")
            {
                for voice in voices {
                    if !status
                        .backend
                        .voices
                        .iter()
                        .any(|known| known.eq_ignore_ascii_case(&voice))
                    {
                        status.backend.voices.push(voice);
                    }
                }
                status.backend.voices.sort();
            }
        }
        let downloads = self.downloads.read().await.clone();
        let host_status = self.host_runtime.status().await.ok().flatten();
        let hf_token_configured = self.hf_token_is_configured().await;
        for status in &mut backends {
            status.hf_token_configured = hf_token_configured;
            let Some(variant) = status.selected_variant.as_ref() else {
                status.available = false;
                status.activatable = false;
                continue;
            };
            if status.host_managed {
                match host_status.as_ref() {
                    Some(agent) => {
                        let (state, reason) = agent.runtime_state(variant);
                        status.runtime_state = state.into();
                        status.runtime_reason = reason;
                    }
                    None => {
                        status.runtime_state = "host_module_delivery_pending".into();
                        status.runtime_reason =
                            "managed Windows host-module delivery is not installed".into();
                    }
                }
            } else if !variant.container.is_empty() {
                if !self.docker_control_enabled {
                    status.runtime_state = "unavailable".into();
                    status.runtime_reason = "managed Speech Lab controller is unavailable".into();
                } else {
                    match self.control.runtime_status(&variant.id).await {
                        Ok(runtime) => {
                            status.runtime_state = runtime.state;
                            status.runtime_reason = runtime.error;
                            if runtime.image_download_size_bytes > 0 {
                                status.image_download_size_bytes =
                                    runtime.image_download_size_bytes;
                            }
                        }
                        Err(error) => {
                            status.runtime_state = "error".into();
                            status.runtime_reason = format!("{error:#}");
                        }
                    }
                }
            } else {
                let endpoint = self.stage_endpoint(&status.backend, variant);
                let (state, reason) = self
                    .remote_runtime_status(&status.backend, variant, &endpoint)
                    .await;
                status.runtime_state = state;
                status.runtime_reason = reason;
            }
            if variant_bundled(&status.backend, variant) {
                status.installed = true;
                status.download_state = "bundled".into();
                status.downloaded_bytes = status.download_size_bytes;
                status.deletable = false;
            } else {
                let (installed, downloaded_bytes) =
                    model_installation_state(&status.backend, variant)
                        .await
                        .unwrap_or((false, 0));
                status.installed = installed;
                status.downloaded_bytes = downloaded_bytes;
                status.download_state = if installed {
                    "installed".into()
                } else {
                    "missing".into()
                };
                if let Some(record) = downloads
                    .get(&status.backend.id)
                    .filter(|record| record.variant_id == variant.id)
                {
                    status.download_state = record.state.clone();
                    status.downloaded_bytes = record.downloaded_bytes;
                    status.download_error = record.error.clone();
                    if record.state == "installed" {
                        status.installed = true;
                    }
                }
            }
            let runtime_ready = matches!(
                status.runtime_state.as_str(),
                "ready" | "running" | "external"
            );
            let host_startable = status.host_managed && status.runtime_state == "stopped";
            status.activatable =
                status.compatible && status.installed && (runtime_ready || host_startable);
            status.available = status.activatable;
        }
        LabCatalogResponse {
            schema_version: self.catalog.schema_version,
            catalog_revision: self.catalog_revision.clone(),
            hardware: capability,
            backends,
            presets: self.catalog.presets.clone(),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<LabEvent> {
        self.events.subscribe()
    }

    pub async fn module_status(&self, backend_id: &str) -> Result<ModuleStatusResponse> {
        let backend = self
            .catalog
            .find(backend_id)
            .ok_or_else(|| anyhow!("unknown backend id '{backend_id}'"))?;
        let variant = resolve_variant(backend, &self.hardware)
            .ok_or_else(|| anyhow!("backend '{backend_id}' has no compatible variant"))?;
        let total_bytes = variant_artifacts(backend, variant)
            .iter()
            .map(|artifact| artifact.size)
            .sum();
        let (installed, mut downloaded_bytes) = model_installation_state(backend, variant).await?;
        let mut model_state = if variant_bundled(backend, variant) {
            "bundled".to_string()
        } else if installed {
            "installed".to_string()
        } else {
            "missing".to_string()
        };
        if let Some(download) = self.downloads.read().await.get(backend_id) {
            if download.variant_id == variant.id {
                model_state = download.state.clone();
                downloaded_bytes = download.downloaded_bytes;
            }
        }
        let runtime = if !variant.host_profile.is_empty() {
            RuntimeProvisionStatus {
                state: "host_module_delivery_pending".into(),
                error: "managed Windows host-module delivery is not installed".into(),
                ..RuntimeProvisionStatus::default()
            }
        } else if variant.container.is_empty() {
            let endpoint = self.stage_endpoint(backend, variant);
            let (state, error) = self
                .remote_runtime_status(backend, variant, &endpoint)
                .await;
            RuntimeProvisionStatus {
                state,
                error,
                ..RuntimeProvisionStatus::default()
            }
        } else if self.docker_control_enabled {
            self.control.runtime_status(&variant.id).await?
        } else {
            RuntimeProvisionStatus {
                state: "unavailable".into(),
                error: "managed Speech Lab controller is unavailable".into(),
                ..RuntimeProvisionStatus::default()
            }
        };
        let record = self.modules.read().await.get(backend_id).cloned();
        let operation_id = record
            .as_ref()
            .map(|record| record.operation_id.clone())
            .unwrap_or_default();
        let mut state = if matches!(runtime.state.as_str(), "ready" | "running" | "external")
            && matches!(model_state.as_str(), "bundled" | "installed")
        {
            "ready".to_string()
        } else {
            runtime.state.clone()
        };
        let mut error = runtime.error;
        if let Some(record) = record {
            if record.variant_id == variant.id
                && matches!(record.state.as_str(), "installing" | "failed" | "cancelled")
            {
                state = record.state;
                if !record.error.is_empty() {
                    error = record.error;
                }
            }
        }
        let image_download_size_bytes = if runtime.image_download_size_bytes > 0 {
            runtime.image_download_size_bytes
        } else {
            variant.image_download_size_bytes
        };
        let image_downloaded_bytes = if matches!(runtime.state.as_str(), "ready" | "running") {
            image_download_size_bytes
        } else {
            runtime
                .image_downloaded_bytes
                .min(image_download_size_bytes)
        };
        Ok(ModuleStatusResponse {
            operation_id,
            backend_id: backend_id.into(),
            variant_id: variant.id.clone(),
            state,
            model_state,
            runtime_state: runtime.state,
            model_downloaded_bytes: downloaded_bytes,
            model_total_bytes: total_bytes,
            image_download_size_bytes,
            image_downloaded_bytes,
            error,
        })
    }

    pub async fn start_module_install(
        &self,
        backend_id: &str,
        request: ModuleInstallRequest,
    ) -> Result<ModuleStatusResponse> {
        if !request.catalog_revision.trim().is_empty()
            && request.catalog_revision.trim() != self.catalog_revision
        {
            return Err(anyhow!(
                "catalog revision changed; refresh the catalog before installing"
            ));
        }
        let backend = self
            .catalog
            .find(backend_id)
            .ok_or_else(|| anyhow!("unknown backend id '{backend_id}'"))?;
        let variant = resolve_variant(backend, &self.hardware)
            .ok_or_else(|| anyhow!("backend '{backend_id}' has no compatible variant"))?;
        if !variant.host_profile.is_empty() {
            return Err(anyhow!(
                "host_module_delivery_pending: managed Windows host-module delivery is not installed"
            ));
        }
        if variant.container.is_empty() {
            return self.module_status(backend_id).await;
        }
        if !self.docker_control_enabled {
            return Err(anyhow!("managed Speech Lab controller is unavailable"));
        }
        let bundled = variant_bundled(backend, variant);
        let model_ready = if !bundled {
            let (installed, downloaded) = model_installation_state(backend, variant).await?;
            if !installed {
                let _ = downloaded;
                self.start_model_download(
                    backend_id,
                    ModelDownloadRequest {
                        hf_token: request.hf_token,
                        remember: request.remember,
                    },
                )
                .await?;
            }
            installed
        } else {
            true
        };
        if let Some(record) = self.modules.read().await.get(backend_id) {
            if record.state == "installing" {
                return self.module_status(backend_id).await;
            }
        }
        let operation_id = uuid::Uuid::new_v4().to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        self.modules.write().await.insert(
            backend_id.into(),
            ModuleInstallRecord {
                operation_id: operation_id.clone(),
                variant_id: variant.id.clone(),
                state: "installing".into(),
                error: String::new(),
                cancel: cancel.clone(),
            },
        );
        let modules = self.modules.clone();
        let downloads = self.downloads.clone();
        let control = self.control.clone();
        let backend_key = backend_id.to_string();
        let variant_id = variant.id.clone();
        tokio::spawn(async move {
            let result: Result<RuntimeProvisionStatus> = async {
                if !model_ready {
                    loop {
                        if cancel.load(Ordering::Acquire) {
                            return Err(anyhow!("module installation cancelled"));
                        }
                        let download = downloads.read().await.get(&backend_key).cloned();
                        match download.as_ref().map(|record| record.state.as_str()) {
                            Some("installed") => break,
                            Some("failed") => {
                                return Err(anyhow!(
                                    "model download failed: {}",
                                    download
                                        .as_ref()
                                        .map(|record| record.error.as_str())
                                        .unwrap_or_default()
                                ));
                            }
                            Some("cancelled") => return Err(anyhow!("model download cancelled")),
                            _ => tokio::time::sleep(Duration::from_millis(250)).await,
                        }
                    }
                }
                control.install(&variant_id).await
            }
            .await;
            if cancel.load(Ordering::Acquire) && result.is_ok() {
                let _ = control.remove(&variant_id).await;
            }
            let mut records = modules.write().await;
            let Some(record) = records.get_mut(&backend_key) else {
                return;
            };
            if record.operation_id != operation_id {
                return;
            }
            if cancel.load(Ordering::Acquire) {
                record.state = "cancelled".into();
                record.error.clear();
            } else if let Err(error) = result {
                record.state = "failed".into();
                record.error = format!("{error:#}");
            } else {
                record.state = "ready".into();
                record.error.clear();
            }
        });
        self.module_status(backend_id).await
    }

    pub async fn cancel_module_install(&self, backend_id: &str) -> Result<ModuleStatusResponse> {
        let modules = self.modules.read().await;
        let record = modules
            .get(backend_id)
            .ok_or_else(|| anyhow!("no module installation for '{backend_id}'"))?;
        if record.state != "installing" {
            return Err(anyhow!("module installation is not running"));
        }
        record.cancel.store(true, Ordering::Release);
        drop(modules);
        if let Some(download) = self.downloads.read().await.get(backend_id) {
            download.cancel.store(true, Ordering::Release);
        }
        self.module_status(backend_id).await
    }

    pub async fn delete_module(&self, backend_id: &str) -> Result<ModuleStatusResponse> {
        let backend = self
            .catalog
            .find(backend_id)
            .cloned()
            .ok_or_else(|| anyhow!("unknown backend id '{backend_id}'"))?;
        let variant = resolve_variant(&backend, &self.hardware)
            .cloned()
            .ok_or_else(|| anyhow!("backend '{backend_id}' has no compatible variant"))?;
        let active = self.state.read().await;
        if [&active.asr, &active.tts, &active.llm]
            .into_iter()
            .flatten()
            .any(|candidate| candidate.backend_id == backend_id)
        {
            return Err(anyhow!("active module cannot be removed"));
        }
        drop(active);
        if !variant.container.is_empty() {
            self.control.remove(&variant.id).await?;
        }
        if !variant_bundled(&backend, &variant) {
            self.delete_model(backend_id).await?;
        }
        self.modules.write().await.remove(backend_id);
        self.module_status(backend_id).await
    }

    pub async fn start_model_download(
        &self,
        backend_id: &str,
        request: ModelDownloadRequest,
    ) -> Result<ModelActionResponse> {
        let backend = self
            .catalog
            .find(backend_id)
            .cloned()
            .ok_or_else(|| anyhow!("unknown backend id '{backend_id}'"))?;
        let variant = resolve_variant(&backend, &self.hardware)
            .cloned()
            .ok_or_else(|| anyhow!("backend '{backend_id}' has no compatible variant"))?;
        if variant_bundled(&backend, &variant) {
            return Ok(ModelActionResponse {
                backend_id: backend.id,
                state: "bundled".into(),
                downloaded_bytes: 0,
                total_bytes: 0,
                message: "model is bundled with its container image".into(),
            });
        }
        if variant_artifacts(&backend, &variant).is_empty() {
            return Err(anyhow!(
                "backend '{backend_id}' has no downloadable artifacts"
            ));
        }
        let total_bytes = variant_artifacts(&backend, &variant)
            .iter()
            .map(|artifact| artifact.size)
            .sum();
        let (installed, downloaded_bytes) = model_installation_state(&backend, &variant).await?;
        if installed {
            return Ok(ModelActionResponse {
                backend_id: backend.id,
                state: "installed".into(),
                downloaded_bytes,
                total_bytes,
                message: "model is already installed".into(),
            });
        }
        let downloaded_bytes = downloaded_bytes.saturating_add(
            partial_download_bytes(&backend, &variant)
                .await
                .unwrap_or(0),
        );
        {
            let downloads = self.downloads.read().await;
            if let Some(record) = downloads
                .get(backend_id)
                .filter(|record| record.variant_id == variant.id)
            {
                if record.state == "downloading" || record.state == "cancelling" {
                    return Ok(ModelActionResponse {
                        backend_id: backend_id.into(),
                        state: record.state.clone(),
                        downloaded_bytes: record.downloaded_bytes,
                        total_bytes: record.total_bytes,
                        message: "model download is already running".into(),
                    });
                }
            }
        }

        let request_token = request
            .hf_token
            .as_deref()
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_string);
        if let Some(token) = request_token.as_ref() {
            if request.remember {
                *self.hf_token.write().await = Some(token.clone());
            }
        }
        let hf_token = self.resolve_hf_token(request_token.as_deref()).await;
        let needs_hf = variant_artifacts(&backend, &variant)
            .iter()
            .any(|artifact| artifact.auth == "huggingface");
        if needs_hf
            && hf_token
                .as_ref()
                .is_none_or(|token| token.trim().is_empty())
        {
            return Err(anyhow!(
                "backend '{backend_id}' requires Hugging Face access. Accept the model terms at {} \
                 and enter a Hugging Face token in the Lab UI (or set HF_TOKEN server-side)",
                if backend.access_url.is_empty() {
                    "https://huggingface.co/"
                } else {
                    backend.access_url.as_str()
                }
            ));
        }

        let cancel = Arc::new(AtomicBool::new(false));
        self.downloads.write().await.insert(
            backend_id.into(),
            ModelDownloadRecord {
                variant_id: variant.id.clone(),
                state: "downloading".into(),
                downloaded_bytes,
                total_bytes,
                error: String::new(),
                cancel: cancel.clone(),
            },
        );
        let controller = self.clone();
        tokio::spawn(async move {
            let _download_guard = controller.model_lock.lock().await;
            let result = controller
                .download_artifacts(&backend, &variant, &cancel, hf_token.as_deref())
                .await;
            let cancelled = cancel.load(Ordering::Acquire);
            if result.is_err() && cancelled {
                // Only wipe on explicit cancel. Transient timeouts keep `.part`
                // files so multi-GB artifacts can resume.
                if let Err(cleanup_error) = remove_model_artifacts(&backend, &variant).await {
                    warn!(
                        backend_id = %backend.id,
                        error = %cleanup_error,
                        "failed to clean cancelled model files"
                    );
                }
            }
            let (installed, downloaded) = model_installation_state(&backend, &variant)
                .await
                .unwrap_or((false, 0));
            let partial = if installed {
                downloaded
            } else {
                downloaded.saturating_add(
                    partial_download_bytes(&backend, &variant)
                        .await
                        .unwrap_or(0),
                )
            };
            let mut downloads = controller.downloads.write().await;
            let record =
                downloads
                    .entry(backend.id.clone())
                    .or_insert_with(|| ModelDownloadRecord {
                        variant_id: variant.id.clone(),
                        state: "missing".into(),
                        downloaded_bytes: 0,
                        total_bytes,
                        error: String::new(),
                        cancel: cancel.clone(),
                    });
            record.downloaded_bytes = partial.min(total_bytes);
            record.variant_id = variant.id.clone();
            match result {
                Ok(()) if installed => {
                    record.state = "installed".into();
                    record.error.clear();
                }
                Ok(()) if cancelled => {
                    record.state = "cancelled".into();
                    record.error.clear();
                }
                Ok(()) => {
                    record.state = "missing".into();
                    record.error = "download ended without a complete model".into();
                }
                Err(_) if cancelled => {
                    record.state = "cancelled".into();
                    record.error.clear();
                    info!("Model download for {} cancelled", backend.id);
                }
                Err(error) => {
                    record.state = "failed".into();
                    record.error = format!("{error:#}");
                    warn!("Model download for {} failed: {error:#}", backend.id);
                }
            }
        });

        Ok(ModelActionResponse {
            backend_id: backend_id.into(),
            state: "downloading".into(),
            downloaded_bytes,
            total_bytes,
            message: "model download started".into(),
        })
    }

    pub async fn cancel_model_download(&self, backend_id: &str) -> Result<ModelActionResponse> {
        let backend = self
            .catalog
            .find(backend_id)
            .ok_or_else(|| anyhow!("unknown backend id '{backend_id}'"))?;
        let variant = resolve_variant(backend, &self.hardware)
            .ok_or_else(|| anyhow!("backend '{backend_id}' has no compatible variant"))?;
        let mut downloads = self.downloads.write().await;
        let record = downloads
            .get_mut(backend_id)
            .filter(|record| record.variant_id == variant.id)
            .ok_or_else(|| anyhow!("no model download is active for '{backend_id}'"))?;
        if record.state != "downloading" && record.state != "cancelling" {
            return Err(anyhow!("model download for '{backend_id}' is not running"));
        }
        record.cancel.store(true, Ordering::Release);
        record.state = "cancelling".into();
        Ok(ModelActionResponse {
            backend_id: backend_id.into(),
            state: record.state.clone(),
            downloaded_bytes: record.downloaded_bytes,
            total_bytes: record.total_bytes,
            message: "cancellation requested".into(),
        })
    }

    pub async fn delete_model(&self, backend_id: &str) -> Result<ModelActionResponse> {
        let backend = self
            .catalog
            .find(backend_id)
            .cloned()
            .ok_or_else(|| anyhow!("unknown backend id '{backend_id}'"))?;
        let variant = resolve_variant(&backend, &self.hardware)
            .cloned()
            .ok_or_else(|| anyhow!("backend '{backend_id}' has no compatible variant"))?;
        if variant_bundled(&backend, &variant) {
            return Err(anyhow!(
                "bundled model '{}' cannot be deleted separately",
                backend.id
            ));
        }
        if self.backend_is_active(&backend.id).await {
            return Err(anyhow!(
                "model '{}' is active; switch this stage before deleting it",
                backend.id
            ));
        }
        if let Some(record) = self
            .downloads
            .write()
            .await
            .get_mut(&backend.id)
            .filter(|record| record.variant_id == variant.id)
        {
            record.cancel.store(true, Ordering::Release);
            record.state = "cancelling".into();
        }
        let _download_guard = self.model_lock.lock().await;
        remove_model_artifacts(&backend, &variant).await?;
        let total_bytes = variant_artifacts(&backend, &variant)
            .iter()
            .map(|artifact| artifact.size)
            .sum();
        self.downloads.write().await.insert(
            backend.id.clone(),
            ModelDownloadRecord {
                variant_id: variant.id.clone(),
                state: "missing".into(),
                downloaded_bytes: 0,
                total_bytes,
                error: String::new(),
                cancel: Arc::new(AtomicBool::new(false)),
            },
        );
        Ok(ModelActionResponse {
            backend_id: backend.id,
            state: "missing".into(),
            downloaded_bytes: 0,
            total_bytes,
            message: "model files deleted".into(),
        })
    }

    async fn backend_is_active(&self, backend_id: &str) -> bool {
        let state = self.state.read().await;
        if state
            .asr
            .as_ref()
            .is_some_and(|active| active.backend_id == backend_id)
            || state
                .tts
                .as_ref()
                .is_some_and(|active| active.backend_id == backend_id)
            || state
                .llm
                .as_ref()
                .is_some_and(|active| active.backend_id == backend_id)
        {
            return true;
        }
        self.runtime.read().await.llm_id == backend_id
    }

    pub fn start_llm_monitor(&self) {
        let controller = self.clone();
        tokio::spawn(async move {
            controller.monitor_external_llm().await;
        });
    }

    pub async fn reconcile_active_stack(&self) {
        let runtime = self.runtime.read().await.clone();
        let asr = match self.active_from_runtime(&runtime.asr_id).await {
            Some(active) => Some(active),
            None => match self.find_running_stage(BackendStage::Asr).await {
                Some(active) => Some(active),
                None => self.active_bundled_asr(&runtime).await,
            },
        };
        let tts = match self.active_from_runtime(&runtime.tts_id).await {
            Some(active) => Some(active),
            None => self.find_running_stage(BackendStage::Tts).await,
        };
        let llm = match self.active_from_runtime(&runtime.llm_id).await {
            Some(active) => Some(active),
            None => self.find_running_stage(BackendStage::Llm).await,
        };
        for active in [&asr, &tts, &llm].into_iter().flatten() {
            let Some(backend) = self.catalog.find(&active.backend_id) else {
                continue;
            };
            let Some(variant) = backend
                .variants
                .iter()
                .find(|variant| variant.id == active.variant_id)
            else {
                continue;
            };
            // AuraGo owns the default ASR service under a stable network alias.
            // Replacing that URL with a catalog module alias would make it unreachable.
            if !(active.backend_id == "confucius4-r2t2" && active.container.is_empty()) {
                if let Err(error) = self
                    .apply_runtime(backend, variant, active.endpoint.clone(), None)
                    .await
                {
                    warn!(
                        "Unable to reconcile runtime for {}: {error:#}",
                        active.backend_id
                    );
                }
            }
            if let Err(error) = self
                .stop_other_stage_containers(backend.stage, &active.container, &active.backend_id)
                .await
            {
                warn!(
                    "Unable to enforce one {:?} container at startup: {error:#}",
                    backend.stage
                );
            }
        }
        let mut state = self.state.write().await;
        state.asr = asr;
        state.tts = tts;
        state.llm = llm;
    }

    async fn active_bundled_asr(&self, runtime: &runtime::RuntimeState) -> Option<ActiveBackend> {
        if !self.docker_control_enabled
            || !self.hardware.in_container
            || runtime.asr_id != "confucius4-r2t2"
            || runtime.cfg.whisper_url.trim_end_matches('/') != "http://confucius-asr:8082"
        {
            return None;
        }
        let backend = self.catalog.find(&runtime.asr_id)?;
        let variant = resolve_variant(backend, &self.hardware)?;
        if !self
            .client
            .get("http://confucius-asr:8082/health")
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .ok()?
            .status()
            .is_success()
        {
            return None;
        }
        Some(ActiveBackend {
            backend_id: backend.id.clone(),
            variant_id: variant.id.clone(),
            accelerator: variant.accelerator.clone(),
            endpoint: runtime.cfg.whisper_url.clone(),
            container: String::new(),
        })
    }

    async fn find_running_stage(&self, stage: BackendStage) -> Option<ActiveBackend> {
        if !self.docker_control_enabled {
            return None;
        }
        for backend in self
            .catalog
            .backends
            .iter()
            .filter(|backend| backend.stage == stage)
        {
            let Some(variant) = resolve_variant(backend, &self.hardware) else {
                continue;
            };
            if variant.container.is_empty()
                || !self
                    .control
                    .running(&variant.container)
                    .await
                    .unwrap_or(false)
            {
                continue;
            }
            return Some(ActiveBackend {
                backend_id: backend.id.clone(),
                variant_id: variant.id.clone(),
                accelerator: variant.accelerator.clone(),
                endpoint: endpoint_for(variant, self.hardware.in_container),
                container: variant.container.clone(),
            });
        }
        None
    }

    async fn active_from_runtime(&self, backend_id: &str) -> Option<ActiveBackend> {
        let backend = self.catalog.find(backend_id)?;
        let variant = resolve_variant(backend, &self.hardware)?;
        if self.docker_control_enabled
            && !variant.container.is_empty()
            && !self
                .control
                .running(&variant.container)
                .await
                .unwrap_or(false)
        {
            return None;
        }
        Some(ActiveBackend {
            backend_id: backend.id.clone(),
            variant_id: variant.id.clone(),
            accelerator: variant.accelerator.clone(),
            endpoint: endpoint_for(variant, self.hardware.in_container),
            container: variant.container.clone(),
        })
    }

    pub async fn status(&self) -> LabStackStatus {
        let state = self.state.read().await.clone();
        let rt = self.runtime.read().await;
        LabStackStatus {
            asr: state.asr,
            tts: state.tts,
            llm: state.llm,
            runtime: runtime::status_of(&rt),
            asr_transition: state.asr_transition,
            tts_transition: state.tts_transition,
            llm_transition: state.llm_transition,
            ok: true,
            error_code: String::new(),
            message: "ready".into(),
            last_transition_error: state.last_transition_error,
        }
    }

    async fn restore_ready_state_after_failure(&self, snapshot: &ControllerState, message: &str) {
        let mut restored = snapshot.clone();
        restored.last_transition_error = message.to_string();
        *self.state.write().await = restored;
    }

    async fn record_failed_rollback(&self, message: &str) {
        self.state.write().await.last_transition_error = message.to_string();
    }

    pub(crate) fn tts_voice_mode(&self, tts_id: &str) -> Option<VoiceMode> {
        self.catalog
            .find(tts_id)
            .filter(|backend| backend.stage == BackendStage::Tts)
            .map(|backend| backend.voice_mode)
    }

    pub(crate) fn resolve_tts_voice(&self, tts_id: &str, voice: &str) -> Result<String> {
        resolve_requested_voice(&self.catalog, tts_id, voice)
    }

    pub(crate) async fn tts_voice_loaded(&self, tts_id: &str, voice: &str) -> bool {
        let Some(backend) = self.catalog.find(tts_id) else {
            return false;
        };
        if backend.stage != BackendStage::Tts
            || resolve_requested_voice(&self.catalog, tts_id, voice).is_err()
        {
            return false;
        }
        if backend.voice_mode != VoiceMode::Restart {
            return true;
        }
        let active_variant = self
            .state
            .read()
            .await
            .tts
            .as_ref()
            .map(|active| active.variant_id.clone());
        let Some(active_variant) = active_variant else {
            return false;
        };
        self.host_runtime
            .status()
            .await
            .ok()
            .flatten()
            .filter(HostAgentStatus::fresh)
            .is_some_and(|status| {
                status.processes.iter().any(|process| {
                    process.variant_id == active_variant
                        && process.voice.eq_ignore_ascii_case(voice)
                        && matches!(process.state.as_str(), "starting" | "running")
                })
            })
    }

    pub async fn activate(&self, request: ActivateStackRequest) -> LabStackStatus {
        let snapshot_runtime = self.runtime.read().await.clone();
        let snapshot_state = self.state.read().await.clone();
        let mut errors = Vec::new();
        let mut failed_stage = None;
        let _asr_lock = if request.asr_id.is_some() {
            Some(self.asr_lock.lock().await)
        } else {
            None
        };
        let _tts_lock = if request.tts_id.is_some() || request.voice.is_some() {
            Some(self.tts_lock.lock().await)
        } else {
            None
        };
        let _llm_lock = if request.llm_id.is_some() {
            Some(self.llm_lock.lock().await)
        } else {
            None
        };

        let requested_voice = if let Some(voice) = request.voice.as_deref() {
            let active_tts_id = if let Some(tts_id) = request.tts_id.as_deref() {
                tts_id.to_string()
            } else if let Some(active) = self.state.read().await.tts.as_ref() {
                active.backend_id.clone()
            } else {
                self.runtime.read().await.tts_id.clone()
            };
            match resolve_requested_voice(&self.catalog, &active_tts_id, voice) {
                Ok(voice) => Some(voice),
                Err(error) => {
                    let mut status = self.status().await;
                    status.ok = false;
                    status.message = format!("TTS voice: {error:#}");
                    status.error_code = activation_error_code(&status.message).into();
                    return status;
                }
            }
        } else {
            None
        };
        let requested_tts_id = request
            .tts_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .unwrap_or(snapshot_runtime.tts_id.as_str());
        let voice_requires_restart = requested_voice.as_ref().is_some_and(|voice| {
            self.catalog
                .find(requested_tts_id)
                .is_some_and(|backend| backend.voice_mode == VoiceMode::Restart)
                && snapshot_runtime.tts_id == requested_tts_id
                && !voice.eq_ignore_ascii_case(&snapshot_runtime.cfg.supertonic_voice)
        });

        // Drop no-op stage switches *before* prepare so a full UI stack restore
        // does not mark already-live stages as "preparing" forever and does not
        // pause VAD while re-selecting the current backends.
        let asr_id = match request.asr_id.as_deref() {
            Some(id) if self.stage_already_active(BackendStage::Asr, id).await => None,
            other => other.map(str::to_string),
        };
        let tts_id = match request.tts_id.as_deref() {
            Some(id)
                if self.stage_already_active(BackendStage::Tts, id).await
                    && !voice_requires_restart =>
            {
                None
            }
            None if voice_requires_restart => Some(requested_tts_id.to_string()),
            other => other.map(str::to_string),
        };
        let llm_id = match request.llm_id.as_deref() {
            Some(id) if self.stage_already_active(BackendStage::Llm, id).await => None,
            other => other.map(str::to_string),
        };

        if let Some(asr_id) = asr_id.as_deref() {
            if let Err(error) = self.prepare_stage(BackendStage::Asr, asr_id).await {
                self.emit_transition(
                    BackendStage::Asr,
                    TransitionPhase::Failed,
                    asr_id,
                    "",
                    &format!("{error:#}"),
                )
                .await;
                failed_stage = Some(BackendStage::Asr);
                errors.push(format!("ASR: {error:#}"));
            }
        }
        if errors.is_empty() {
            if let Some(tts_id) = tts_id.as_deref() {
                if let Err(error) = self.prepare_stage(BackendStage::Tts, tts_id).await {
                    self.emit_transition(
                        BackendStage::Tts,
                        TransitionPhase::Failed,
                        tts_id,
                        "",
                        &format!("{error:#}"),
                    )
                    .await;
                    failed_stage = Some(BackendStage::Tts);
                    errors.push(format!("TTS: {error:#}"));
                }
            }
        }
        if errors.is_empty() {
            if let Some(llm_id) = llm_id.as_deref() {
                if let Err(error) = self.prepare_stage(BackendStage::Llm, llm_id).await {
                    self.emit_transition(
                        BackendStage::Llm,
                        TransitionPhase::Failed,
                        llm_id,
                        "",
                        &format!("{error:#}"),
                    )
                    .await;
                    failed_stage = Some(BackendStage::Llm);
                    errors.push(format!("LLM: {error:#}"));
                }
            }
        }

        if !errors.is_empty() {
            let message = errors.join("; ");
            self.restore_ready_state_after_failure(&snapshot_state, &message)
                .await;
            let mut status = self.status().await;
            status.ok = false;
            status.error_code = activation_error_code(&message).into();
            status.message = message;
            return status;
        }

        // Keep the turn gate closed for the complete transactional switch.
        // A native host backend may need the currently published Docker port,
        // so draining must finish before bring-up is allowed to pause it.
        let stage_switch_requested = asr_id.is_some() || tts_id.is_some() || llm_id.is_some();
        let needs_turn_pause = stage_switch_requested || requested_voice.is_some();
        let turn_coordinator = self.runtime.read().await.turns.clone();
        let _turn_pause = needs_turn_pause.then(|| turn_coordinator.pause());
        if stage_switch_requested {
            for (stage, backend_id) in [
                (BackendStage::Asr, asr_id.as_deref()),
                (BackendStage::Tts, tts_id.as_deref()),
                (BackendStage::Llm, llm_id.as_deref()),
            ] {
                if let Some(backend_id) = backend_id {
                    self.emit_draining(stage, backend_id).await;
                }
            }
            let drained = turn_coordinator.wait_idle(Duration::from_secs(10)).await;
            let message = if drained {
                "active speech turn drained"
            } else {
                "drain timeout reached after 10 seconds"
            };
            for (stage, backend_id) in [
                (BackendStage::Asr, asr_id.as_deref()),
                (BackendStage::Tts, tts_id.as_deref()),
                (BackendStage::Llm, llm_id.as_deref()),
            ] {
                if let Some(backend_id) = backend_id {
                    self.emit_draining_message(stage, backend_id, message).await;
                }
            }
        } else if requested_voice.is_some() {
            let _ = turn_coordinator.wait_idle(Duration::from_secs(3)).await;
        }

        // Start + health + warm the replacements while the gate remains closed.
        let mut plans: Vec<StagePlan> = Vec::new();
        if let Some(id) = asr_id.as_deref() {
            match self
                .bring_up_stage(BackendStage::Asr, id, false, None)
                .await
            {
                Ok(plan) => {
                    if !plan.already_active {
                        plans.push(plan);
                    }
                }
                Err(error) => {
                    self.emit_transition(
                        BackendStage::Asr,
                        TransitionPhase::Failed,
                        id,
                        "",
                        &format!("{error:#}"),
                    )
                    .await;
                    failed_stage = Some(BackendStage::Asr);
                    errors.push(format!("ASR: {error:#}"));
                }
            }
        }
        if errors.is_empty() {
            if let Some(id) = tts_id.as_deref() {
                match self
                    .bring_up_stage(
                        BackendStage::Tts,
                        id,
                        voice_requires_restart,
                        requested_voice.as_deref(),
                    )
                    .await
                {
                    Ok(plan) => {
                        if !plan.already_active {
                            plans.push(plan);
                        }
                    }
                    Err(error) => {
                        self.emit_transition(
                            BackendStage::Tts,
                            TransitionPhase::Failed,
                            id,
                            "",
                            &format!("{error:#}"),
                        )
                        .await;
                        failed_stage = Some(BackendStage::Tts);
                        errors.push(format!("TTS: {error:#}"));
                    }
                }
            }
        }
        if errors.is_empty() {
            if let Some(id) = llm_id.as_deref() {
                match self
                    .bring_up_stage(BackendStage::Llm, id, false, None)
                    .await
                {
                    Ok(plan) => {
                        if !plan.already_active {
                            plans.push(plan);
                        }
                    }
                    Err(error) => {
                        self.emit_transition(
                            BackendStage::Llm,
                            TransitionPhase::Failed,
                            id,
                            "",
                            &format!("{error:#}"),
                        )
                        .await;
                        failed_stage = Some(BackendStage::Llm);
                        errors.push(format!("LLM: {error:#}"));
                    }
                }
            }
        }
        if !errors.is_empty() {
            // Runtime was not cut over — stop any pre-started replacements so
            // we do not leave orphan stage containers running.
            self.abort_brought_up(&plans).await;
            let rollback_failed = if let Err(error) = self.restore_containers(&snapshot_state).await
            {
                errors.push(format!("rollback: {error:#}"));
                true
            } else {
                false
            };
            let message = errors.join("; ");
            if rollback_failed {
                self.record_failed_rollback(&message).await;
            } else {
                self.restore_ready_state_after_failure(&snapshot_state, &message)
                    .await;
            }
            let mut status = self.status().await;
            status.ok = false;
            status.error_code = activation_error_code(&message).into();
            status.message = message;
            return status;
        }

        let has_switch = !plans.is_empty();

        for plan in &plans {
            let voice_override = (plan.stage == BackendStage::Tts)
                .then_some(requested_voice.as_deref())
                .flatten();
            if let Err(error) = self.commit_stage(plan, voice_override).await {
                self.emit_transition(
                    plan.stage,
                    TransitionPhase::Failed,
                    &plan.backend_id,
                    &plan.variant_id,
                    &format!("{error:#}"),
                )
                .await;
                failed_stage = Some(plan.stage);
                errors.push(format!("{:?}: {error:#}", plan.stage));
                break;
            }
        }

        if !errors.is_empty() {
            self.emit_transition(
                failed_stage.unwrap_or(BackendStage::Asr),
                TransitionPhase::Rollback,
                "",
                "",
                "restoring previous stack",
            )
            .await;
            *self.runtime.write().await = snapshot_runtime;
            let rollback_failed = if let Err(error) = self.restore_containers(&snapshot_state).await
            {
                errors.push(format!("rollback: {error:#}"));
                true
            } else {
                false
            };
            let message = errors.join("; ");
            if rollback_failed {
                self.record_failed_rollback(&message).await;
            } else {
                self.restore_ready_state_after_failure(&snapshot_state, &message)
                    .await;
            }
            let mut status = self.status().await;
            status.ok = false;
            status.error_code = activation_error_code(&message).into();
            status.message = message;
            return status;
        }

        let has_tts_plan = plans.iter().any(|plan| plan.stage == BackendStage::Tts);
        if !has_tts_plan {
            if let Some(voice) = requested_voice.as_ref() {
                self.runtime.write().await.cfg.supertonic_voice = voice.clone();
            }
        }

        // Always publish a terminal "ready" for stages that are live so the UI
        // never stays on preparing/draining after a no-op or voice-only call.
        self.emit_active_stages_ready().await;
        self.state.write().await.last_transition_error.clear();

        let mut status = self.status().await;
        status.message = if has_switch {
            "stack activated".into()
        } else if requested_voice.is_some() {
            "voice activated".into()
        } else {
            "no changes".into()
        };
        status
    }

    async fn emit_active_stages_ready(&self) {
        let state = self.state.read().await.clone();
        for (stage, active) in [
            (BackendStage::Asr, state.asr.as_ref()),
            (BackendStage::Tts, state.tts.as_ref()),
            (BackendStage::Llm, state.llm.as_ref()),
        ] {
            let Some(active) = active else { continue };
            let phase = match stage {
                BackendStage::Asr => state.asr_transition.phase,
                BackendStage::Tts => state.tts_transition.phase,
                BackendStage::Llm => state.llm_transition.phase,
            };
            if matches!(phase, TransitionPhase::Ready | TransitionPhase::Idle) {
                continue;
            }
            self.emit_transition(
                stage,
                TransitionPhase::Ready,
                &active.backend_id,
                &active.variant_id,
                "backend ready",
            )
            .await;
        }
    }

    async fn prepare_stage(&self, stage: BackendStage, backend_id: &str) -> Result<()> {
        let backend = self
            .catalog
            .find(backend_id)
            .ok_or_else(|| anyhow!("unknown backend id '{backend_id}'"))?;
        if backend.stage != stage {
            return Err(anyhow!(
                "backend '{backend_id}' belongs to {:?}, not {:?}",
                backend.stage,
                stage
            ));
        }
        let variant = resolve_variant(backend, &self.hardware).ok_or_else(|| {
            anyhow!("backend '{backend_id}' has no compatible certified variant for this host")
        })?;
        if !variant.container.is_empty() {
            if !self.docker_control_enabled {
                return Err(anyhow!(
                    "module_not_provisioned: managed Speech Lab controller is unavailable"
                ));
            }
            let runtime = self.control.runtime_status(&variant.id).await?;
            if !matches!(runtime.state.as_str(), "ready" | "running") {
                return Err(anyhow!(
                    "module_not_provisioned: runtime '{}' is {}",
                    variant.id,
                    runtime.state
                ));
            }
            self.control
                .validate(
                    &variant.container,
                    stage,
                    &backend.id,
                    provisioned_image(&runtime, &variant.image),
                )
                .await?;
        }
        self.emit_transition(
            stage,
            TransitionPhase::Preparing,
            backend_id,
            &variant.id,
            "validating catalog and model artifacts",
        )
        .await;
        if !variant_bundled(backend, variant) {
            let (installed, downloaded_bytes) = model_installation_state(backend, variant).await?;
            if !installed {
                let total_bytes: u64 = variant_artifacts(backend, variant)
                    .iter()
                    .map(|artifact| artifact.size)
                    .sum();
                return Err(anyhow!(
                    "model '{backend_id}' is not installed ({downloaded_bytes}/{total_bytes} bytes); confirm the model download first"
                ));
            }
        }
        Ok(())
    }

    async fn emit_draining(&self, stage: BackendStage, backend_id: &str) {
        self.emit_draining_message(stage, backend_id, "draining active speech turn")
            .await;
    }

    async fn emit_draining_message(&self, stage: BackendStage, backend_id: &str, message: &str) {
        let variant_id = self
            .catalog
            .find(backend_id)
            .and_then(|backend| resolve_variant(backend, &self.hardware))
            .map(|variant| variant.id.as_str())
            .unwrap_or_default();
        self.emit_transition(
            stage,
            TransitionPhase::Draining,
            backend_id,
            variant_id,
            message,
        )
        .await;
    }

    async fn stage_already_active(&self, stage: BackendStage, backend_id: &str) -> bool {
        let active = {
            let state = self.state.read().await;
            match stage {
                BackendStage::Asr => state.asr.clone(),
                BackendStage::Tts => state.tts.clone(),
                BackendStage::Llm => state.llm.clone(),
            }
        };
        let Some(active) = active.filter(|active| active.backend_id == backend_id) else {
            return false;
        };
        let Some(variant) = self.catalog.find(backend_id).and_then(|backend| {
            backend
                .variants
                .iter()
                .find(|variant| variant.id == active.variant_id)
        }) else {
            return false;
        };

        if !variant.host_profile.is_empty() {
            return self
                .host_runtime
                .status()
                .await
                .ok()
                .flatten()
                .filter(HostAgentStatus::fresh)
                .is_some_and(|status| {
                    status.processes.iter().any(|process| {
                        process.stage == stage_label(stage)
                            && process.backend_id == backend_id
                            && process.variant_id == active.variant_id
                            && process.state == "running"
                    })
                });
        }
        if self.docker_control_enabled && !active.container.is_empty() {
            return self
                .control
                .running(&active.container)
                .await
                .unwrap_or(false);
        }
        true
    }

    fn stage_endpoint(&self, backend: &BackendDefinition, variant: &BackendVariant) -> String {
        if backend.id == "external-openai" {
            if let Ok(url) = std::env::var("S2S_LLM_EXTERNAL_URL") {
                if !url.trim().is_empty() {
                    return url;
                }
            }
            return self
                .runtime
                .try_read()
                .ok()
                .filter(|rt| rt.llm_id == "external-openai")
                .map(|rt| rt.cfg.llm_base_url.clone())
                .unwrap_or_default();
        }
        endpoint_for(variant, self.hardware.in_container)
    }

    async fn remote_runtime_status(
        &self,
        backend: &BackendDefinition,
        variant: &BackendVariant,
        endpoint: &str,
    ) -> (String, String) {
        if endpoint.trim().is_empty() {
            return (
                "needs_configuration".into(),
                "remote endpoint is not configured".into(),
            );
        }
        let api_key = if backend.id == "external-openai" {
            self.runtime.read().await.cfg.llm_api_key.clone()
        } else {
            String::new()
        };
        let requires_auth = backend.id == "external-openai"
            && std::env::var("S2S_LLM_EXTERNAL_REQUIRES_AUTH")
                .ok()
                .map(|value| !matches!(value.trim(), "0" | "false" | "no"))
                .unwrap_or_else(|| endpoint.trim().starts_with("https://"));
        if requires_auth && api_key.trim().is_empty() {
            return (
                "needs_credentials".into(),
                "required remote credentials are not configured".into(),
            );
        }
        let base = endpoint
            .split("/v1/")
            .next()
            .unwrap_or(endpoint)
            .trim_end_matches('/');
        let path = if variant.health_path.is_empty() {
            "/"
        } else {
            variant.health_path.as_str()
        };
        let url = format!("{base}{path}");
        let mut request = self.client.get(&url).timeout(Duration::from_secs(2));
        if !api_key.trim().is_empty() {
            request = request.bearer_auth(api_key);
        }
        match request.send().await {
            Ok(response) if response.status().is_success() => ("external".into(), String::new()),
            Ok(response) => (
                "unhealthy".into(),
                format!("remote healthcheck returned HTTP {}", response.status()),
            ),
            Err(error) => (
                "unhealthy".into(),
                format!("remote healthcheck failed: {error}"),
            ),
        }
    }

    fn current_stage_endpoint(
        &self,
        stage: BackendStage,
        rt: &crate::runtime::RuntimeState,
    ) -> String {
        match stage {
            BackendStage::Asr => rt.cfg.whisper_url.clone(),
            BackendStage::Tts => rt.cfg.tts_url.clone(),
            BackendStage::Llm => rt.cfg.llm_base_url.clone(),
        }
    }

    /// Candidate variants for a stage, preferred first (same ranking as catalog resolve).
    fn stage_candidates<'a>(
        &'a self,
        stage: BackendStage,
        backend_id: &str,
    ) -> Result<(&'a BackendDefinition, Vec<&'a BackendVariant>)> {
        let backend = self
            .catalog
            .find(backend_id)
            .ok_or_else(|| anyhow!("unknown backend id '{backend_id}'"))?;
        if backend.stage != stage {
            return Err(anyhow!(
                "backend '{backend_id}' belongs to {:?}, not {:?}",
                backend.stage,
                stage
            ));
        }
        let mut candidates: Vec<&BackendVariant> = backend
            .variants
            .iter()
            .filter(|variant| crate::registry::variant_is_compatible(variant, &self.hardware))
            .collect();
        if candidates.is_empty() {
            return Err(anyhow!(
                "backend '{backend_id}' has no compatible certified variant for this host"
            ));
        }
        candidates.sort_by_key(|variant| crate::registry::variant_rank(variant, &self.hardware));
        Ok((backend, candidates))
    }

    async fn pause_active_container_for_host(
        &self,
        stage: BackendStage,
    ) -> Result<Option<ActiveBackend>> {
        if !self.docker_control_enabled {
            return Ok(None);
        }
        let active = {
            let state = self.state.read().await;
            match stage {
                BackendStage::Asr => state.asr.clone(),
                BackendStage::Tts => state.tts.clone(),
                BackendStage::Llm => state.llm.clone(),
            }
        };
        let Some(active) = active else {
            return Ok(None);
        };
        if active.container.is_empty()
            || !self
                .control
                .running(&active.container)
                .await
                .unwrap_or(false)
        {
            return Ok(None);
        }

        self.emit_transition(
            stage,
            TransitionPhase::Stopping,
            &active.backend_id,
            &active.variant_id,
            "pausing current container for native host port",
        )
        .await;
        let stop = self
            .control
            .stop(&active.container, Duration::from_secs(15));
        match tokio::time::timeout(Duration::from_secs(25), stop).await {
            Ok(Ok(())) => Ok(Some(active)),
            Ok(Err(error)) => Err(error).with_context(|| {
                format!("pause current {stage:?} container '{}'", active.container)
            }),
            Err(_) => Err(anyhow!(
                "timed out pausing current {stage:?} container '{}'",
                active.container
            )),
        }
    }

    async fn restore_paused_container(&self, active: &ActiveBackend) -> Result<()> {
        self.control
            .start(&active.container)
            .await
            .with_context(|| format!("restart paused container '{}'", active.container))?;
        let backend = self
            .catalog
            .find(&active.backend_id)
            .ok_or_else(|| anyhow!("rollback backend '{}' is missing", active.backend_id))?;
        let variant = backend
            .variants
            .iter()
            .find(|variant| variant.id == active.variant_id)
            .ok_or_else(|| anyhow!("rollback variant '{}' is missing", active.variant_id))?;
        self.wait_for_health(backend, variant, &active.endpoint)
            .await
            .with_context(|| format!("health check restored backend '{}'", active.backend_id))?;
        self.warm_backend(backend, &active.endpoint)
            .await
            .with_context(|| format!("warm restored backend '{}'", active.backend_id))
    }

    /// Start + health + warm without cutting over runtime.
    /// Host/remote variants fail fast; if unreachable, fall back to a managed
    /// container variant when Docker control is enabled.
    async fn bring_up_stage(
        &self,
        stage: BackendStage,
        backend_id: &str,
        force_restart: bool,
        requested_voice: Option<&str>,
    ) -> Result<StagePlan> {
        let (backend, candidates) = self.stage_candidates(stage, backend_id)?;

        if !force_restart && self.stage_already_active(stage, backend_id).await {
            let rt = self.runtime.read().await;
            let active_ep = self.current_stage_endpoint(stage, &rt);
            if let Some(active_variant) = candidates
                .iter()
                .find(|variant| self.stage_endpoint(backend, variant) == active_ep)
                .copied()
            {
                return Ok(StagePlan {
                    stage,
                    backend_id: backend_id.into(),
                    variant_id: active_variant.id.clone(),
                    endpoint: active_ep,
                    container: active_variant.container.clone(),
                    started_container: None,
                    started_host: false,
                    already_active: true,
                });
            }
        }

        let mut last_error = None;
        let mut pause_attempted = false;
        let mut paused_for_host = None;
        for variant in &candidates {
            let endpoint = self.stage_endpoint(backend, variant);
            let host_only = variant.container.is_empty();

            // Managed host profiles are launched through the Windows agent.
            // Plain remote variants retain the existing probe-only behaviour.
            if host_only {
                let mut started_host = false;
                if !variant.host_profile.is_empty() {
                    if !pause_attempted {
                        pause_attempted = true;
                        match self.pause_active_container_for_host(stage).await {
                            Ok(paused) => paused_for_host = paused,
                            Err(error) => {
                                last_error = Some(error);
                                continue;
                            }
                        }
                    }
                    self.emit_transition(
                        stage,
                        TransitionPhase::Starting,
                        backend_id,
                        &variant.id,
                        &format!("starting Windows host profile {}", variant.host_profile),
                    )
                    .await;
                    match self
                        .host_runtime
                        .start(
                            stage,
                            backend_id,
                            variant,
                            &endpoint,
                            if stage == BackendStage::Tts {
                                requested_voice.unwrap_or(&backend.default_voice)
                            } else {
                                ""
                            },
                        )
                        .await
                    {
                        Ok(pid) => {
                            started_host = true;
                            info!(
                                backend_id,
                                variant = %variant.id,
                                pid,
                                "Windows host backend launched"
                            );
                        }
                        Err(error) => {
                            warn!(
                                backend_id,
                                variant = %variant.id,
                                error = %error,
                                "host agent launch failed — trying next candidate"
                            );
                            last_error = Some(error);
                            continue;
                        }
                    }
                }
                self.emit_transition(
                    stage,
                    TransitionPhase::Starting,
                    backend_id,
                    &variant.id,
                    &format!("probing host endpoint {endpoint}"),
                )
                .await;
                let host_timeout = if started_host {
                    Duration::from_secs(
                        std::env::var("S2S_LAB_HEALTH_TIMEOUT_SECS")
                            .ok()
                            .and_then(|value| value.parse::<u64>().ok())
                            .unwrap_or(300)
                            .clamp(10, 900),
                    )
                } else {
                    Duration::from_secs(12)
                };
                match self
                    .wait_for_health_with_timeout(backend, variant, &endpoint, host_timeout)
                    .await
                {
                    Ok(()) => {
                        self.emit_transition(
                            stage,
                            TransitionPhase::Warming,
                            backend_id,
                            &variant.id,
                            "warming inference path",
                        )
                        .await;
                        match self.warm_backend(backend, &endpoint).await {
                            Ok(()) => {
                                return Ok(StagePlan {
                                    stage,
                                    backend_id: backend_id.into(),
                                    variant_id: variant.id.clone(),
                                    endpoint,
                                    container: String::new(),
                                    started_container: None,
                                    started_host,
                                    already_active: false,
                                });
                            }
                            Err(error) => {
                                if started_host {
                                    let _ = self
                                        .host_runtime
                                        .stop(stage, backend_id, variant, &endpoint)
                                        .await;
                                }
                                last_error = Some(error);
                                continue;
                            }
                        }
                    }
                    Err(error) => {
                        if started_host {
                            let _ = self
                                .host_runtime
                                .stop(stage, backend_id, variant, &endpoint)
                                .await;
                        }
                        warn!(
                            backend_id,
                            variant = %variant.id,
                            error = %error,
                            "host/remote variant unreachable — trying next candidate"
                        );
                        last_error = Some(error);
                        continue;
                    }
                }
            }

            if !self.docker_control_enabled {
                last_error = Some(anyhow!(
                    "managed container '{}' requires Docker lab control",
                    variant.container
                ));
                continue;
            }

            self.emit_transition(
                stage,
                TransitionPhase::Starting,
                backend_id,
                &variant.id,
                "starting selected backend",
            )
            .await;
            match self.control.start(&variant.container).await {
                Ok(()) => {}
                Err(error) => {
                    warn!(
                        container = %variant.container,
                        error = %error,
                        "failed to start managed container — trying next candidate"
                    );
                    last_error = Some(error);
                    continue;
                }
            }

            if let Err(error) = self.wait_for_health(backend, variant, &endpoint).await {
                warn!(
                    container = %variant.container,
                    error = %error,
                    "managed backend health failed — trying next candidate"
                );
                last_error = Some(error);
                let _ = self
                    .control
                    .stop(&variant.container, Duration::from_secs(5))
                    .await;
                continue;
            }

            self.emit_transition(
                stage,
                TransitionPhase::Warming,
                backend_id,
                &variant.id,
                "warming inference path",
            )
            .await;
            if let Err(error) = self.warm_backend(backend, &endpoint).await {
                last_error = Some(error);
                let _ = self
                    .control
                    .stop(&variant.container, Duration::from_secs(5))
                    .await;
                continue;
            }

            return Ok(StagePlan {
                stage,
                backend_id: backend_id.into(),
                variant_id: variant.id.clone(),
                endpoint,
                container: variant.container.clone(),
                started_container: Some(variant.container.clone()),
                started_host: false,
                already_active: false,
            });
        }

        let mut final_error = last_error.unwrap_or_else(|| {
            anyhow!(
                "no reachable variant for '{backend_id}'. For host TTS start the server first \
                 (e.g. scripts/start_vibevoice.ps1 on :8089) or build the managed container image."
            )
        });
        if let Some(active) = paused_for_host.as_ref() {
            if let Err(restore_error) = self.restore_paused_container(active).await {
                final_error = anyhow!(
                    "{final_error:#}; failed to restore previous backend '{}': {restore_error:#}",
                    active.backend_id
                );
            }
        }
        Err(final_error)
    }

    async fn abort_brought_up(&self, brought_up: &[StagePlan]) {
        let keep: HashSet<String> = {
            let state = self.state.read().await;
            [&state.asr, &state.tts, &state.llm]
                .into_iter()
                .flatten()
                .map(|active| active.container.clone())
                .filter(|container| !container.is_empty())
                .collect()
        };
        for plan in brought_up {
            if plan.started_host {
                if let Some(backend) = self.catalog.find(&plan.backend_id) {
                    if let Some(variant) = backend
                        .variants
                        .iter()
                        .find(|variant| variant.id == plan.variant_id)
                    {
                        if let Err(error) = self
                            .host_runtime
                            .stop(plan.stage, &plan.backend_id, variant, &plan.endpoint)
                            .await
                        {
                            warn!(
                                backend_id = %plan.backend_id,
                                error = %error,
                                "failed to stop pre-started host backend"
                            );
                        }
                    }
                }
            }
            let Some(container) = plan.started_container.as_ref() else {
                continue;
            };
            if !self.docker_control_enabled || container.is_empty() || keep.contains(container) {
                continue;
            }
            warn!(
                container = %container,
                "aborting pre-started stage container after failed multi-stage activate"
            );
            let stop = self.control.stop(container, Duration::from_secs(5));
            if let Err(error) = tokio::time::timeout(Duration::from_secs(10), stop).await {
                warn!(container = %container, "abort stop timed out: {error:?}");
            } else {
                // ignore stop Result — best effort
            }
        }
    }

    /// Stop siblings and cut runtime over using the plan from bring_up.
    async fn commit_stage(&self, plan: &StagePlan, voice_override: Option<&str>) -> Result<()> {
        let backend = self
            .catalog
            .find(&plan.backend_id)
            .ok_or_else(|| anyhow!("unknown backend id '{}'", plan.backend_id))?;
        let variant = backend
            .variants
            .iter()
            .find(|variant| variant.id == plan.variant_id)
            .ok_or_else(|| anyhow!("variant '{}' missing after bring-up", plan.variant_id))?;

        if plan.already_active {
            self.emit_transition(
                plan.stage,
                TransitionPhase::Ready,
                &plan.backend_id,
                &plan.variant_id,
                "backend already active",
            )
            .await;
            return Ok(());
        }

        // A container/remote cutover must release a previously managed native
        // process. Host-to-host switches are already serialized by the agent.
        if variant.host_profile.is_empty() {
            let previous = {
                let state = self.state.read().await;
                match plan.stage {
                    BackendStage::Asr => state.asr.clone(),
                    BackendStage::Tts => state.tts.clone(),
                    BackendStage::Llm => state.llm.clone(),
                }
            };
            if let Some(previous) = previous {
                if let Some(previous_backend) = self.catalog.find(&previous.backend_id) {
                    if let Some(previous_variant) = previous_backend
                        .variants
                        .iter()
                        .find(|candidate| candidate.id == previous.variant_id)
                    {
                        if !previous_variant.host_profile.is_empty() {
                            self.host_runtime
                                .stop(
                                    plan.stage,
                                    &previous.backend_id,
                                    previous_variant,
                                    &previous.endpoint,
                                )
                                .await
                                .with_context(|| {
                                    format!("stop previous host backend {}", previous.backend_id)
                                })?;
                        }
                    }
                }
            }
        }

        if self.docker_control_enabled {
            self.emit_transition(
                plan.stage,
                TransitionPhase::Stopping,
                &plan.backend_id,
                &plan.variant_id,
                "stopping previous stage container",
            )
            .await;
            self.stop_other_stage_containers(plan.stage, &plan.container, &plan.backend_id)
                .await?;
        }

        self.apply_runtime(backend, variant, plan.endpoint.clone(), voice_override)
            .await?;
        let active = ActiveBackend {
            backend_id: backend.id.clone(),
            variant_id: variant.id.clone(),
            accelerator: variant.accelerator.clone(),
            endpoint: plan.endpoint.clone(),
            container: plan.container.clone(),
        };
        {
            let mut state = self.state.write().await;
            match plan.stage {
                BackendStage::Asr => state.asr = Some(active),
                BackendStage::Tts => state.tts = Some(active),
                BackendStage::Llm => state.llm = Some(active),
            }
        }
        if plan.stage == BackendStage::Llm {
            *self.desired_llm.write().await = backend.id.clone();
        }
        self.emit_transition(
            plan.stage,
            TransitionPhase::Ready,
            &plan.backend_id,
            &plan.variant_id,
            "backend ready",
        )
        .await;
        Ok(())
    }

    #[allow(dead_code)]
    async fn switch_stage(&self, stage: BackendStage, backend_id: &str) -> Result<()> {
        let plan = self.bring_up_stage(stage, backend_id, false, None).await?;
        self.commit_stage(&plan, None).await
    }

    async fn download_artifacts(
        &self,
        backend: &BackendDefinition,
        variant: &BackendVariant,
        cancel: &AtomicBool,
        hf_token: Option<&str>,
    ) -> Result<()> {
        let root = std::env::var("S2S_MODELS_DIR").unwrap_or_else(|_| "/models".into());
        let artifacts = variant_artifacts(backend, variant);
        let backend_total: u64 = artifacts.iter().map(|artifact| artifact.size).sum();
        let mut completed = 0u64;
        for artifact in artifacts {
            if cancel.load(Ordering::Acquire) {
                return Err(anyhow!("model download cancelled"));
            }
            let path = Path::new(&root).join(&artifact.path);
            if artifact_matches(&path, artifact.size, &artifact.sha256).await? {
                let add = if artifact.size > 0 {
                    artifact.size
                } else {
                    tokio::fs::metadata(&path)
                        .await
                        .map(|meta| meta.len())
                        .unwrap_or(0)
                };
                completed = completed.saturating_add(add);
                self.update_download_progress(&backend.id, completed).await;
                continue;
            }

            let downloaded = self
                .download_one_artifact(
                    backend,
                    artifact,
                    &path,
                    completed,
                    backend_total,
                    cancel,
                    hf_token,
                )
                .await?;
            completed = completed.saturating_add(downloaded);
            self.update_download_progress(&backend.id, completed).await;
            info!(
                "Prepared model artifact {} for {} ({} bytes)",
                path.display(),
                backend.id,
                downloaded
            );
        }
        Ok(())
    }

    async fn download_one_artifact(
        &self,
        backend: &BackendDefinition,
        artifact: &crate::registry::ModelArtifact,
        path: &Path,
        completed_before: u64,
        backend_total: u64,
        cancel: &AtomicBool,
        hf_token: Option<&str>,
    ) -> Result<u64> {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow!("artifact '{}' has no parent", path.display()))?;
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("create model directory {}", parent.display()))?;

        let part = part_path(path)?;
        const MAX_ATTEMPTS: u32 = 12;
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            if cancel.load(Ordering::Acquire) {
                return Err(anyhow!("model download cancelled"));
            }
            match self
                .download_one_artifact_attempt(
                    backend,
                    artifact,
                    path,
                    &part,
                    completed_before,
                    backend_total,
                    cancel,
                    hf_token,
                )
                .await
            {
                Ok(bytes) => return Ok(bytes),
                Err(error) if cancel.load(Ordering::Acquire) => return Err(error),
                Err(error) if attempt < MAX_ATTEMPTS && is_transient_download_error(&error) => {
                    let backoff = Duration::from_secs(2u64.saturating_pow(attempt.min(6)));
                    warn!(
                        backend_id = %backend.id,
                        artifact = %artifact.path,
                        attempt,
                        backoff_secs = backoff.as_secs(),
                        error = %error,
                        "transient model download error — retrying (partial kept for resume)"
                    );
                    tokio::time::sleep(backoff).await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    async fn download_one_artifact_attempt(
        &self,
        backend: &BackendDefinition,
        artifact: &crate::registry::ModelArtifact,
        path: &Path,
        part: &Path,
        completed_before: u64,
        backend_total: u64,
        cancel: &AtomicBool,
        hf_token: Option<&str>,
    ) -> Result<u64> {
        let mut existing = match tokio::fs::metadata(part).await {
            Ok(meta) if meta.is_file() => meta.len(),
            Ok(_) => 0,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error.into()),
        };
        if artifact.size > 0 && existing > artifact.size {
            warn!(
                artifact = %artifact.path,
                existing,
                expected = artifact.size,
                "partial download larger than expected — restarting artifact"
            );
            let _ = tokio::fs::remove_file(part).await;
            existing = 0;
        }

        // Already fully downloaded into .part — promote after integrity check.
        if artifact.size > 0 && existing == artifact.size {
            let digest = hash_file_hex(part).await?;
            if !artifact.sha256.is_empty() && !digest.eq_ignore_ascii_case(&artifact.sha256) {
                let _ = tokio::fs::remove_file(part).await;
                return Err(anyhow!(
                    "artifact '{}' SHA-256 mismatch on partial: got {digest}, expected {}",
                    artifact.path,
                    artifact.sha256
                ));
            }
            replace_artifact(part, path).await?;
            return Ok(artifact.size);
        }

        let mut hasher = Sha256::new();
        if existing > 0 {
            hash_file_into(part, &mut hasher).await?;
        }

        let mut request = self.download_client.get(&artifact.source);
        if artifact.auth == "huggingface" {
            // Prefer the token supplied with this download (UI / session); env is fallback.
            let token = match hf_token.map(str::trim).filter(|token| !token.is_empty()) {
                Some(token) => token.to_string(),
                None => self.resolve_hf_token(None).await.unwrap_or_default(),
            };
            let token = token.trim();
            if token.is_empty() {
                return Err(anyhow!(
                    "artifact '{}' requires Hugging Face access. Accept the model terms at {} \
                     and enter a Hugging Face token in the Lab UI (or set HF_TOKEN server-side)",
                    artifact.path,
                    backend.access_url
                ));
            }
            request = request.bearer_auth(token);
        }
        if existing > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={existing}-"));
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("download {}", artifact.source))?;
        if artifact.auth == "huggingface"
            && matches!(
                response.status(),
                reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
            )
        {
            return Err(anyhow!(
                "Hugging Face denied access to artifact '{}' (HTTP {}). Accept the model terms \
                 at {} and verify the Hugging Face token (Lab UI or HF_TOKEN)",
                artifact.path,
                response.status(),
                backend.access_url
            ));
        }
        let response = response
            .error_for_status()
            .with_context(|| format!("download {}", artifact.source))?;

        let status = response.status();
        let resume = existing > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
        if existing > 0 && !resume {
            // Server ignored Range (HTTP 200) — restart from scratch.
            warn!(
                artifact = %artifact.path,
                status = %status,
                "server did not honor Range resume — restarting artifact"
            );
            drop(hasher);
            let _ = tokio::fs::remove_file(part).await;
            existing = 0;
            hasher = Sha256::new();
        }

        let mut output = if existing > 0 {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(part)
                .await
                .with_context(|| format!("append partial artifact {}", part.display()))?
        } else {
            tokio::fs::File::create(part)
                .await
                .with_context(|| format!("create partial artifact {}", part.display()))?
        };

        let mut stream = response.bytes_stream();
        let mut downloaded = existing;
        while let Some(chunk) = stream.next().await {
            if cancel.load(Ordering::Acquire) {
                drop(output);
                return Err(anyhow!("model download cancelled"));
            }
            let chunk = chunk.context("read artifact download")?;
            downloaded = downloaded
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| anyhow!("artifact byte count overflow"))?;
            if artifact.size > 0 && downloaded > artifact.size {
                return Err(anyhow!(
                    "artifact '{}' exceeded expected size {}",
                    artifact.path,
                    artifact.size
                ));
            }
            output.write_all(&chunk).await?;
            hasher.update(&chunk);
            let _ = self.events.send(LabEvent::DownloadProgress {
                backend_id: backend.id.clone(),
                artifact: artifact.path.clone(),
                downloaded: completed_before.saturating_add(downloaded),
                total: backend_total,
            });
            self.update_download_progress(&backend.id, completed_before.saturating_add(downloaded))
                .await;
        }
        output.flush().await?;
        output.sync_all().await?;
        drop(output);

        if artifact.size > 0 && downloaded != artifact.size {
            return Err(anyhow!(
                "artifact '{}' has size {downloaded}, expected {} (incomplete — will resume)",
                artifact.path,
                artifact.size
            ));
        }
        let digest = hex::encode(hasher.finalize());
        if !artifact.sha256.is_empty() && !digest.eq_ignore_ascii_case(&artifact.sha256) {
            let _ = tokio::fs::remove_file(part).await;
            return Err(anyhow!(
                "artifact '{}' SHA-256 mismatch: got {digest}, expected {}",
                artifact.path,
                artifact.sha256
            ));
        }
        replace_artifact(part, path).await?;
        Ok(downloaded)
    }

    async fn update_download_progress(&self, backend_id: &str, downloaded_bytes: u64) {
        if let Some(record) = self.downloads.write().await.get_mut(backend_id) {
            record.downloaded_bytes = downloaded_bytes.min(record.total_bytes);
        }
    }

    async fn stop_other_stage_containers(
        &self,
        stage: BackendStage,
        selected: &str,
        keep_backend_id: &str,
    ) -> Result<()> {
        // Containers belonging to the backend we are activating (used when the
        // selected variant is host/remote with empty container name — e.g.
        // VibeVoice on :8089 is still the managed s2s-tts-vibevoice process).
        let keep_containers: HashSet<&str> = self
            .catalog
            .backends
            .iter()
            .filter(|b| b.id == keep_backend_id)
            .flat_map(|b| b.variants.iter().map(|v| v.container.as_str()))
            .filter(|c| !c.is_empty())
            .collect();

        let mut seen = HashSet::new();
        for backend in self.catalog.backends.iter().filter(|b| b.stage == stage) {
            for variant in &backend.variants {
                let container = variant.container.as_str();
                if container.is_empty()
                    || container == selected
                    || keep_containers.contains(container)
                    || !seen.insert(container)
                {
                    continue;
                }
                if self.control.running(container).await.unwrap_or(false) {
                    info!("Stopping managed {:?} container {container}", stage);
                    // Best-effort: a hung Docker stop must not block stack
                    // activate forever (that freezes VAD via turn pause).
                    let stop = self.control.stop(container, Duration::from_secs(8));
                    match tokio::time::timeout(Duration::from_secs(20), stop).await {
                        Ok(Ok(())) => {}
                        Ok(Err(error)) => {
                            warn!(
                                container,
                                error = %error,
                                "failed to stop managed container — continuing stack switch"
                            );
                        }
                        Err(_) => {
                            warn!(
                                container,
                                "timed out stopping managed container after 20s — continuing stack switch"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn wait_for_health(
        &self,
        backend: &BackendDefinition,
        variant: &BackendVariant,
        endpoint: &str,
    ) -> Result<()> {
        let health_timeout = std::env::var("S2S_LAB_HEALTH_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(300)
            .clamp(10, 900);
        // Host/remote (no managed container) must already be running — fail fast
        // so stack switches never freeze the UI for minutes on a dead port.
        let timeout = if variant.container.is_empty() {
            Duration::from_secs(12)
        } else if self.docker_control_enabled {
            Duration::from_secs(health_timeout)
        } else {
            Duration::from_secs(1)
        };
        self.wait_for_health_with_timeout(backend, variant, endpoint, timeout)
            .await
    }

    async fn wait_for_health_with_timeout(
        &self,
        backend: &BackendDefinition,
        variant: &BackendVariant,
        endpoint: &str,
        timeout: Duration,
    ) -> Result<()> {
        if endpoint.is_empty() {
            return Ok(());
        }
        let base = endpoint
            .split("/v1/")
            .next()
            .unwrap_or(endpoint)
            .trim_end_matches('/');
        let health_path = if variant.health_path.is_empty() {
            "/"
        } else {
            variant.health_path.as_str()
        };
        let url = if health_path.starts_with("/v1/") && endpoint.contains("/v1") {
            format!(
                "{}{}",
                endpoint.trim_end_matches('/'),
                health_path.trim_start_matches("/v1")
            )
        } else {
            format!("{base}{health_path}")
        };
        let deadline = tokio::time::Instant::now() + timeout;
        let last_error = loop {
            let failure = match self
                .client
                .get(&url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    // Reject HTML (e.g. web UI occupying the same host port).
                    let ctype = response
                        .headers()
                        .get(reqwest::header::CONTENT_TYPE)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if ctype.contains("text/html") {
                        format!("HTML response at {url} (wrong service / port conflict)")
                    } else {
                        let _ = self.events.send(LabEvent::BackendHealth {
                            stage: backend.stage,
                            backend_id: backend.id.clone(),
                            ok: true,
                            message: format!("reachable at {url}"),
                        });
                        return Ok(());
                    }
                }
                Ok(response) => format!("HTTP {}", response.status()),
                Err(error) => error.to_string(),
            };
            if tokio::time::Instant::now() >= deadline {
                break failure;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        };
        let _ = self.events.send(LabEvent::BackendHealth {
            stage: backend.stage,
            backend_id: backend.id.clone(),
            ok: false,
            message: last_error.clone(),
        });
        Err(anyhow!(
            "backend '{}' health check failed at {url}: {last_error}",
            backend.id
        ))
    }

    async fn warm_backend(&self, backend: &BackendDefinition, endpoint: &str) -> Result<()> {
        if endpoint.is_empty() {
            return Ok(());
        }
        if backend.stage == BackendStage::Asr {
            let wav = encode_wav_f32(&vec![0.0; 16_000], 16_000)?;
            let mut form = reqwest::multipart::Form::new()
                .part(
                    "file",
                    reqwest::multipart::Part::bytes(wav)
                        .file_name("warmup.wav")
                        .mime_str("audio/wav")?,
                )
                .text("language", "de")
                .text("response_format", "json");
            let path = if backend.protocol == "openai-asr" {
                form = form.text("model", backend.model.clone());
                "/v1/audio/transcriptions"
            } else {
                form = form.text("no_timestamps", "true");
                "/inference"
            };
            let url = format!("{}{path}", endpoint.trim_end_matches('/'));
            let response = self
                .client
                .post(&url)
                .timeout(Duration::from_secs(120))
                .multipart(form)
                .send()
                .await
                .with_context(|| format!("ASR warmup POST {url}"))?;
            if !response.status().is_success() {
                return Err(anyhow!(
                    "warmup for '{}' failed: HTTP {}",
                    backend.id,
                    response.status()
                ));
            }
            let body: serde_json::Value = response.json().await?;
            if !body.get("text").is_some_and(serde_json::Value::is_string) {
                return Err(anyhow!(
                    "warmup for '{}' returned no text field",
                    backend.id
                ));
            }
            return Ok(());
        }
        if backend.stage != BackendStage::Tts {
            return Ok(());
        }
        let qwen = backend.id == "qwen3-tts-0.6b";
        let higgs = backend.id == "higgs-tts-3-4b";
        let vibevoice = backend.id == "vibevoice-realtime-0.5b";
        let xtts = backend.id == "xtts-v2";
        let inflect = backend.id == "inflect-micro-v2";
        let audio8 = matches!(
            backend.id.as_str(),
            "audio8-tts-preview-0.6b" | "audio8-tts-preview-0.1b"
        );
        let crispasr_wav = matches!(
            backend.id.as_str(),
            "piper"
                | "cosyvoice3-0.5b"
                | "omnivoice"
                | "vibevoice-realtime-0.5b"
                | "kokoro"
                | "inflect-micro-v2"
                | "audio8-tts-preview-0.6b"
                | "audio8-tts-preview-0.1b"
        );
        let voice = if backend.default_voice.is_empty() {
            "default"
        } else {
            backend.default_voice.as_str()
        };
        // CrispASR / Higgs / XTTS / Inflect prefer WAV; Qwen uses raw PCM.
        let response_format = if higgs || vibevoice || xtts || crispasr_wav {
            "wav"
        } else {
            "pcm"
        };
        let mut body = serde_json::json!({
            "model": backend.model,
            "input": if qwen {
                "Hallo."
            } else if inflect {
                "Hello."
            } else {
                "Test."
            },
            "voice": voice,
            "language": if qwen {
                "german"
            } else if inflect {
                "en"
            } else {
                "de"
            },
            "response_format": response_format,
            "max_new_tokens": if qwen { 16 } else if higgs { 256 } else if audio8 { 256 } else { 64 }
        });
        if inflect {
            body["seed"] = serde_json::json!(7);
        }
        crate::tts::apply_preloaded_voice_policy(&mut body, &backend.model);
        // Piper/OmniVoice/Inflect load a fixed voice at process start; drop
        // placeholder OpenAI voice names so warmup does not 400.
        if matches!(
            body.get("voice").and_then(|v| v.as_str()),
            Some("default") | Some("male")
        ) && matches!(
            backend.id.as_str(),
            "piper" | "omnivoice" | "inflect-micro-v2"
        ) {
            if let Some(object) = body.as_object_mut() {
                object.remove("voice");
            }
        }
        let response = self
            .client
            .post(endpoint)
            .timeout(Duration::from_secs(if audio8 { 180 } else { 60 }))
            .json(&body)
            .send()
            .await
            .with_context(|| format!("warmup POST {endpoint}"))?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "warmup for '{}' failed: HTTP {}",
                backend.id,
                response.status()
            ));
        }
        // Drain streaming PCM so the server sees a completed request. Qwen is
        // only promoted after a real, non-empty, sample-aligned synthesis.
        let bytes = response.bytes().await?;
        validate_tts_warmup_audio(backend, &bytes)?;
        Ok(())
    }

    async fn apply_runtime(
        &self,
        backend: &BackendDefinition,
        variant: &BackendVariant,
        endpoint: String,
        voice_override: Option<&str>,
    ) -> Result<()> {
        let mut rt = self.runtime.write().await;
        match backend.stage {
            BackendStage::Asr => {
                rt.asr_id = backend.id.clone();
                rt.cfg.whisper_url = endpoint.clone();
                rt.cfg.stt_api = if variant_protocol(backend, variant) == "openai-asr" {
                    SttApi::Openai
                } else {
                    SttApi::Whisper
                };
                rt.cfg.stt_model = backend.model.clone();
            }
            BackendStage::Tts => {
                rt.tts_id = backend.id.clone();
                rt.cfg.tts = TtsBackend::Http;
                rt.cfg.tts_url = endpoint.clone();
                rt.cfg.tts_model = backend.model.clone();
                if backend.native_sample_rate > 0 {
                    rt.cfg.tts_native_sample_rate = backend.native_sample_rate;
                }
                rt.cfg.supertonic_voice = voice_override
                    .map(str::trim)
                    .filter(|voice| !voice.is_empty())
                    .unwrap_or(&backend.default_voice)
                    .to_string();
            }
            BackendStage::Llm => {
                rt.llm_id = backend.id.clone();
                rt.cfg.llm_base_url = endpoint.clone();
                rt.cfg.model_name = if backend.id == "external-openai" {
                    std::env::var("S2S_LLM_EXTERNAL_MODEL")
                        .unwrap_or_else(|_| backend.model.clone())
                } else {
                    backend.model.clone()
                };
            }
        }
        drop(rt);

        if backend.stage == BackendStage::Asr
            && variant_protocol(backend, variant) == "faster-whisper"
        {
            let model = variant
                .environment
                .get("S2S_WHISPER_MODEL")
                .map(String::as_str)
                .unwrap_or(&backend.model);
            runtime::reload_whisper_model(&endpoint, model)
                .await
                .map_err(|error| anyhow!("reload faster-whisper: {error:#}"))?;
        }
        Ok(())
    }

    async fn restore_containers(&self, snapshot: &ControllerState) -> Result<()> {
        let desired: HashSet<&str> = [&snapshot.asr, &snapshot.tts, &snapshot.llm]
            .into_iter()
            .flatten()
            .map(|active| active.container.as_str())
            .filter(|container| !container.is_empty())
            .collect();
        if self.docker_control_enabled {
            let mut seen = HashSet::new();
            for backend in self.catalog.backends.iter().filter(|backend| {
                matches!(
                    backend.stage,
                    BackendStage::Asr | BackendStage::Tts | BackendStage::Llm
                )
            }) {
                for variant in &backend.variants {
                    let container = variant.container.as_str();
                    if container.is_empty()
                        || desired.contains(container)
                        || !seen.insert(container.to_string())
                    {
                        continue;
                    }
                    if self.control.running(container).await.unwrap_or(false) {
                        if let Err(error) =
                            self.control.stop(container, Duration::from_secs(10)).await
                        {
                            warn!("Failed to stop rollback container {container}: {error:#}");
                        }
                    }
                }
            }
        }
        for active in [&snapshot.asr, &snapshot.tts, &snapshot.llm]
            .into_iter()
            .flatten()
        {
            let backend = self
                .catalog
                .find(&active.backend_id)
                .ok_or_else(|| anyhow!("rollback backend '{}' missing", active.backend_id))?;
            let variant = backend
                .variants
                .iter()
                .find(|variant| variant.id == active.variant_id)
                .ok_or_else(|| anyhow!("rollback variant '{}' missing", active.variant_id))?;
            if active.container.is_empty() {
                if !variant.host_profile.is_empty() {
                    let active_voice = if backend.stage == BackendStage::Tts {
                        self.runtime.read().await.cfg.supertonic_voice.clone()
                    } else {
                        String::new()
                    };
                    self.host_runtime
                        .start(
                            backend.stage,
                            &active.backend_id,
                            variant,
                            &active.endpoint,
                            &active_voice,
                        )
                        .await
                        .with_context(|| format!("restore host backend {}", active.backend_id))?;
                    self.wait_for_health(backend, variant, &active.endpoint)
                        .await
                        .with_context(|| {
                            format!("rollback host health for {}", active.backend_id)
                        })?;
                    self.warm_backend(backend, &active.endpoint)
                        .await
                        .with_context(|| {
                            format!("rollback host warmup for {}", active.backend_id)
                        })?;
                }
                continue;
            }
            if !self.docker_control_enabled {
                continue;
            }
            let runtime = self.control.runtime_status(&active.variant_id).await?;
            self.control
                .validate(
                    &active.container,
                    backend.stage,
                    &active.backend_id,
                    provisioned_image(&runtime, &variant.image),
                )
                .await?;
            self.control
                .start(&active.container)
                .await
                .with_context(|| format!("restore managed container {}", active.container))?;
            self.wait_for_health(backend, variant, &active.endpoint)
                .await
                .with_context(|| format!("rollback health for {}", active.backend_id))?;
            self.warm_backend(backend, &active.endpoint)
                .await
                .with_context(|| format!("rollback warmup for {}", active.backend_id))?;
        }
        Ok(())
    }

    async fn emit_transition(
        &self,
        stage: BackendStage,
        phase: TransitionPhase,
        backend_id: &str,
        variant_id: &str,
        message: &str,
    ) {
        let transition = StageTransition {
            stage,
            phase,
            backend_id: backend_id.into(),
            variant_id: variant_id.into(),
            message: message.into(),
        };
        {
            let mut state = self.state.write().await;
            match stage {
                BackendStage::Asr => state.asr_transition = transition.clone(),
                BackendStage::Tts => state.tts_transition = transition.clone(),
                BackendStage::Llm => state.llm_transition = transition.clone(),
            }
        }
        let _ = self.events.send(LabEvent::StackTransition {
            stage,
            phase,
            backend_id: backend_id.into(),
            variant_id: variant_id.into(),
            message: message.into(),
        });
    }

    async fn monitor_external_llm(self) {
        let external_url = std::env::var("S2S_LLM_EXTERNAL_URL").unwrap_or_else(|_| {
            self.runtime
                .try_read()
                .map(|runtime| runtime.cfg.llm_base_url.clone())
                .unwrap_or_default()
        });
        if external_url.is_empty() {
            warn!("LLM failover monitor disabled: no external URL configured");
            return;
        }
        let external_model = std::env::var("S2S_LLM_EXTERNAL_MODEL").unwrap_or_else(|_| {
            self.runtime
                .try_read()
                .map(|runtime| runtime.cfg.model_name.clone())
                .unwrap_or_else(|_| "external".into())
        });
        let Some(fallback) = self.catalog.find("local-fallback").cloned() else {
            warn!("LLM failover monitor disabled: local-fallback missing from catalog");
            return;
        };
        let Some(variant) = resolve_variant(&fallback, &self.hardware).cloned() else {
            warn!("LLM failover monitor: no compatible local fallback variant");
            return;
        };
        let fallback_endpoint = endpoint_for(&variant, self.hardware.in_container);
        let mut failures = 0u8;
        let mut successes = 0u8;
        let mut using_fallback = false;

        loop {
            if self.desired_llm.read().await.as_str() != "external-openai" {
                failures = 0;
                successes = 0;
                using_fallback = false;
                tokio::time::sleep(Duration::from_secs(15)).await;
                continue;
            }
            let external_ok = self.llm_healthy(&external_url).await;
            if external_ok {
                failures = 0;
                successes = successes.saturating_add(1);
                let pipeline_idle = self.runtime.read().await.turns.is_idle();
                if using_fallback && successes >= 2 && pipeline_idle {
                    {
                        let mut runtime = self.runtime.write().await;
                        runtime.cfg.llm_base_url = external_url.clone();
                        runtime.cfg.model_name = external_model.clone();
                        runtime.llm_id = "external-openai".into();
                    }
                    let _ = self.events.send(LabEvent::BackendHealth {
                        stage: BackendStage::Llm,
                        backend_id: "external-openai".into(),
                        ok: true,
                        message: "external LLM restored".into(),
                    });
                    // Existing requests hold a Config snapshot; a short grace
                    // period avoids stopping the fallback while it is finishing.
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    if self.docker_control_enabled && !variant.container.is_empty() {
                        if let Err(error) = self
                            .control
                            .stop(&variant.container, Duration::from_secs(10))
                            .await
                        {
                            warn!("Failed to stop LLM fallback: {error:#}");
                        }
                    }
                    using_fallback = false;
                }
            } else {
                successes = 0;
                failures = failures.saturating_add(1);
                if !using_fallback && failures >= 2 {
                    let activation = async {
                        if self.docker_control_enabled && !variant.container.is_empty() {
                            let runtime = self.control.runtime_status(&variant.id).await?;
                            self.control
                                .validate(
                                    &variant.container,
                                    BackendStage::Llm,
                                    &fallback.id,
                                    provisioned_image(&runtime, &variant.image),
                                )
                                .await?;
                            self.control.start(&variant.container).await?;
                        }
                        self.wait_for_health(&fallback, &variant, &fallback_endpoint)
                            .await?;
                        self.apply_runtime(&fallback, &variant, fallback_endpoint.clone(), None)
                            .await
                    }
                    .await;
                    match activation {
                        Ok(()) => {
                            using_fallback = true;
                            let _ = self.events.send(LabEvent::BackendHealth {
                                stage: BackendStage::Llm,
                                backend_id: fallback.id.clone(),
                                ok: true,
                                message: "external LLM unavailable; local fallback active".into(),
                            });
                        }
                        Err(error) => warn!("Unable to activate LLM fallback: {error:#}"),
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(15)).await;
        }
    }

    async fn llm_healthy(&self, base_url: &str) -> bool {
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        matches!(
            self.client
                .get(url)
                .timeout(Duration::from_secs(3))
                .send()
                .await,
            Ok(response) if response.status().is_success()
        )
    }
}

fn resolve_requested_voice(
    catalog: &BackendCatalog,
    tts_id: &str,
    requested_voice: &str,
) -> Result<String> {
    let backend = catalog
        .find(tts_id)
        .ok_or_else(|| anyhow!("unknown TTS backend id '{tts_id}'"))?;
    if backend.stage != BackendStage::Tts {
        return Err(anyhow!("backend '{tts_id}' is not a TTS backend"));
    }

    let requested_voice = requested_voice.trim();
    if requested_voice.is_empty() {
        return Err(anyhow!("voice must not be empty"));
    }
    if backend.id == "xtts-v2" {
        if !is_safe_xtts_voice_id(requested_voice) {
            return Err(anyhow!(
                "voice '{requested_voice}' is not a safe XTTS voice id"
            ));
        }
        let voice_path = models_root()
            .join("xtts-v2")
            .join("voices")
            .join(format!("{requested_voice}.wav"));
        if voice_path.is_file() {
            return Ok(requested_voice.to_string());
        }
        return Err(anyhow!(
            "voice '{requested_voice}' is not installed for 'xtts-v2'"
        ));
    }

    let voice = backend
        .voices
        .iter()
        .find(|voice| voice.eq_ignore_ascii_case(requested_voice))
        .or_else(|| {
            (!backend.default_voice.is_empty()
                && backend.default_voice.eq_ignore_ascii_case(requested_voice))
            .then_some(&backend.default_voice)
        });
    voice.cloned().ok_or_else(|| {
        let available = if backend.voices.is_empty() {
            backend.default_voice.clone()
        } else {
            backend.voices.join(", ")
        };
        anyhow!(
            "voice '{requested_voice}' is not available for '{tts_id}' (available: {available})"
        )
    })
}

fn validate_tts_warmup_audio(backend: &BackendDefinition, bytes: &[u8]) -> Result<()> {
    if backend.id == "qwen3-tts-0.6b" && (bytes.is_empty() || bytes.len() % 2 != 0) {
        return Err(anyhow!(
            "warmup for '{}' returned invalid PCM payload ({} bytes)",
            backend.id,
            bytes.len()
        ));
    }
    if backend.id == "xtts-v2" {
        let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
            .context("XTTS warmup returned invalid WAV data")?;
        let spec = reader.spec();
        if spec.channels != 1
            || spec.sample_rate != 24_000
            || spec.bits_per_sample != 16
            || reader.duration() == 0
        {
            return Err(anyhow!(
                "warmup for '{}' returned unsupported WAV format \
                 (channels={}, rate={}, bits={}, frames={})",
                backend.id,
                spec.channels,
                spec.sample_rate,
                spec.bits_per_sample,
                reader.duration()
            ));
        }
    }
    Ok(())
}

fn is_safe_xtts_voice_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 64
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'_' | b'-'))
}

async fn discover_xtts_voice_ids_at(root: &Path) -> Result<Vec<String>> {
    let voices_dir = root.join("xtts-v2").join("voices");
    let mut entries = match tokio::fs::read_dir(&voices_dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut voices = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if !entry.file_type().await?.is_file()
            || !path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
        {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if is_safe_xtts_voice_id(stem) {
            voices.push(stem.to_string());
        }
    }
    voices.sort();
    voices.dedup_by(|left, right| left.eq_ignore_ascii_case(right));
    Ok(voices)
}

async fn model_installation_state(
    backend: &BackendDefinition,
    variant: &BackendVariant,
) -> Result<(bool, u64)> {
    model_installation_state_at(backend, variant, &models_root()).await
}

fn models_root() -> PathBuf {
    PathBuf::from(std::env::var("S2S_MODELS_DIR").unwrap_or_else(|_| "/models".into()))
}

async fn model_installation_state_at(
    backend: &BackendDefinition,
    variant: &BackendVariant,
    root: &Path,
) -> Result<(bool, u64)> {
    let artifacts = variant_artifacts(backend, variant);
    if variant_bundled(backend, variant) {
        let total = artifacts.iter().map(|artifact| artifact.size).sum();
        return Ok((true, total));
    }
    let mut installed = !artifacts.is_empty();
    let mut downloaded = 0u64;
    for artifact in artifacts {
        let path = root.join(&artifact.path);
        match tokio::fs::metadata(&path).await {
            Ok(metadata) if metadata.is_file() => {
                let length = metadata.len();
                downloaded = downloaded.saturating_add(if artifact.size > 0 {
                    length.min(artifact.size)
                } else {
                    length
                });
                if artifact.size > 0 && length != artifact.size {
                    installed = false;
                }
            }
            Ok(_) => installed = false,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => installed = false,
            Err(error) => return Err(error.into()),
        }
    }
    Ok((installed, downloaded))
}

/// Bytes already present in `.part` files (for resume progress display).
async fn partial_download_bytes(
    backend: &BackendDefinition,
    variant: &BackendVariant,
) -> Result<u64> {
    let root = models_root();
    let mut total = 0u64;
    for artifact in variant_artifacts(backend, variant) {
        let final_path = root.join(&artifact.path);
        if artifact_matches(&final_path, artifact.size, &artifact.sha256).await? {
            continue;
        }
        let part = part_path(&final_path)?;
        match tokio::fs::metadata(&part).await {
            Ok(meta) if meta.is_file() => {
                let len = meta.len();
                total = total.saturating_add(if artifact.size > 0 {
                    len.min(artifact.size)
                } else {
                    len
                });
            }
            Ok(_) | Err(_) => {}
        }
    }
    Ok(total)
}

fn is_transient_download_error(error: &anyhow::Error) -> bool {
    let text = format!("{error:#}").to_ascii_lowercase();
    text.contains("timed out")
        || text.contains("timeout")
        || text.contains("connection reset")
        || text.contains("connection refused")
        || text.contains("broken pipe")
        || text.contains("error decoding response body")
        || text.contains("error sending request")
        || text.contains("incomplete")
        || text.contains("temporarily")
        || text.contains("503")
        || text.contains("429")
        || text.contains("502")
        || text.contains("504")
}

async fn hash_file_into(path: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("open partial for hash {}", path.display()))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(())
}

async fn hash_file_hex(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    hash_file_into(path, &mut hasher).await?;
    Ok(hex::encode(hasher.finalize()))
}

async fn remove_model_artifacts(
    backend: &BackendDefinition,
    variant: &BackendVariant,
) -> Result<()> {
    remove_model_artifacts_at(backend, variant, &models_root()).await
}

async fn remove_model_artifacts_at(
    backend: &BackendDefinition,
    variant: &BackendVariant,
    root: &Path,
) -> Result<()> {
    for artifact in variant_artifacts(backend, variant) {
        let path = root.join(&artifact.path);
        for target in [path.clone(), part_path(&path)?] {
            match tokio::fs::remove_file(&target).await {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("delete model file {}", target.display()));
                }
            }
        }
        let mut parent = path.parent();
        while let Some(directory) = parent {
            if directory == root {
                break;
            }
            match tokio::fs::remove_dir(directory).await {
                Ok(()) => parent = directory.parent(),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty
                    ) =>
                {
                    break;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(())
}

async fn artifact_matches(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<bool> {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(metadata) if metadata.is_file() => metadata,
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if expected_size > 0 && metadata.len() != expected_size {
        return Ok(false);
    }
    if expected_sha256.is_empty() {
        return Ok(true);
    }
    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = tokio::io::AsyncReadExt::read(&mut file, &mut buffer).await?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()).eq_ignore_ascii_case(expected_sha256))
}

fn part_path(path: &Path) -> Result<PathBuf> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("invalid artifact filename '{}'", path.display()))?;
    Ok(path.with_file_name(format!("{file_name}.part")))
}

async fn replace_artifact(part_path: &Path, target_path: &Path) -> Result<()> {
    #[cfg(windows)]
    if tokio::fs::try_exists(target_path).await? {
        // Docker reference builds run on Linux, where rename replaces atomically.
        // Windows cannot replace an existing file with std::fs::rename.
        tokio::fs::remove_file(target_path).await?;
    }
    tokio::fs::rename(part_path, target_path)
        .await
        .with_context(|| {
            format!(
                "activate artifact {} as {}",
                part_path.display(),
                target_path.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use clap::Parser;
    use std::collections::BTreeMap;

    #[tokio::test]
    async fn runtime_install_can_outlive_the_regular_controller_timeout() {
        let app = axum::Router::new().route(
            "/s2s/modules/test/install",
            axum::routing::post(|| async {
                tokio::time::sleep(Duration::from_millis(80)).await;
                axum::Json(serde_json::json!({"state": "ready"}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let control = DockerProxyControl {
            client: reqwest::Client::builder()
                .timeout(Duration::from_millis(10))
                .build()
                .unwrap(),
            base_url: format!("http://{address}"),
            token: None,
        };

        let status = control.install("test").await.unwrap();
        assert_eq!(status.state, "ready");
        server.abort();
    }

    #[derive(Default)]
    struct FakeControl {
        running: Mutex<HashSet<String>>,
        actions: Mutex<Vec<String>>,
        fail_start: Mutex<HashSet<String>>,
        missing_variants: Mutex<HashSet<String>>,
        override_status: Mutex<Option<RuntimeProvisionStatus>>,
    }

    #[async_trait]
    impl ContainerControl for FakeControl {
        async fn validate(
            &self,
            _container: &str,
            _stage: BackendStage,
            _backend_id: &str,
            _image: &str,
        ) -> Result<()> {
            Ok(())
        }

        async fn start(&self, container: &str) -> Result<()> {
            self.actions.lock().await.push(format!("start:{container}"));
            if self.fail_start.lock().await.contains(container) {
                return Err(anyhow!("injected start failure for {container}"));
            }
            self.running.lock().await.insert(container.into());
            Ok(())
        }

        async fn stop(&self, container: &str, _timeout: Duration) -> Result<()> {
            self.actions.lock().await.push(format!("stop:{container}"));
            self.running.lock().await.remove(container);
            Ok(())
        }

        async fn running(&self, container: &str) -> Result<bool> {
            Ok(self.running.lock().await.contains(container))
        }

        async fn list_managed_running(&self) -> Result<Vec<String>> {
            Ok(self.running.lock().await.iter().cloned().collect())
        }

        async fn runtime_status(&self, variant_id: &str) -> Result<RuntimeProvisionStatus> {
            if let Some(status) = self.override_status.lock().await.clone() {
                return Ok(status);
            }
            Ok(RuntimeProvisionStatus {
                state: if self.missing_variants.lock().await.contains(variant_id) {
                    "missing"
                } else {
                    "ready"
                }
                .into(),
                ..RuntimeProvisionStatus::default()
            })
        }
    }

    fn test_config() -> Config {
        Config::try_parse_from(["s2s-vulkan", "--skip-health"]).unwrap()
    }

    fn test_catalog(endpoint: String) -> BackendCatalog {
        BackendCatalog {
            schema_version: 1,
            presets: Vec::new(),
            backends: vec![BackendDefinition {
                id: "asr-a".into(),
                stage: BackendStage::Asr,
                name: "ASR A".into(),
                tag: String::new(),
                description: String::new(),
                protocol: "mock".into(),
                model: "a".into(),
                default_voice: String::new(),
                voices: vec![],
                voice_mode: VoiceMode::Fixed,
                native_sample_rate: 0,
                languages: vec![],
                licenses: vec![],
                access_url: String::new(),
                resources: crate::registry::ResourceEstimate {
                    vram_gb: 1.0,
                    ram_gb: 1.0,
                    stars: Default::default(),
                },
                bundled: true,
                artifacts: vec![],
                variants: vec![BackendVariant {
                    id: "asr-a-cpu".into(),
                    accelerator: "cpu".into(),
                    vendors: vec!["any".into()],
                    platforms: vec![std::env::consts::OS.into()],
                    stable: true,
                    published: true,
                    runtime_delivery_reason: String::new(),
                    device_match: vec![],
                    endpoint: endpoint.clone(),
                    native_endpoint: endpoint,
                    container: "asr-a".into(),
                    image: "test/asr:a".into(),
                    image_download_size_bytes: 0,
                    health_path: "/health".into(),
                    environment: BTreeMap::new(),
                    artifacts: Vec::new(),
                    bundled: None,
                    protocol: String::new(),
                    host_profile: String::new(),
                }],
            }],
        }
    }

    fn backend(id: &str, stage: BackendStage, container: &str) -> BackendDefinition {
        BackendDefinition {
            id: id.into(),
            stage,
            name: id.into(),
            tag: String::new(),
            description: String::new(),
            protocol: "mock".into(),
            model: id.into(),
            default_voice: if stage == BackendStage::Tts {
                "default".into()
            } else {
                String::new()
            },
            voices: vec![],
            voice_mode: VoiceMode::Fixed,
            native_sample_rate: 0,
            languages: vec![],
            licenses: vec![],
            access_url: String::new(),
            resources: crate::registry::ResourceEstimate {
                vram_gb: 1.0,
                ram_gb: 1.0,
                stars: Default::default(),
            },
            bundled: true,
            artifacts: vec![],
            variants: vec![BackendVariant {
                id: format!("{id}-cpu"),
                accelerator: "cpu".into(),
                vendors: vec!["any".into()],
                platforms: vec![std::env::consts::OS.into()],
                stable: true,
                published: true,
                runtime_delivery_reason: String::new(),
                device_match: vec![],
                endpoint: String::new(),
                native_endpoint: String::new(),
                container: container.into(),
                image: format!("test/{id}:latest"),
                image_download_size_bytes: 0,
                health_path: "/health".into(),
                environment: BTreeMap::new(),
                artifacts: Vec::new(),
                bundled: None,
                protocol: String::new(),
                host_profile: String::new(),
            }],
        }
    }

    fn switch_catalog() -> BackendCatalog {
        BackendCatalog {
            schema_version: 1,
            presets: Vec::new(),
            backends: vec![
                backend("asr-a", BackendStage::Asr, "asr-a"),
                backend("asr-b", BackendStage::Asr, "asr-b"),
                backend("tts-fail", BackendStage::Tts, "tts-fail"),
            ],
        }
    }

    fn cpu_hardware() -> HardwareProfile {
        let mut profile = HardwareProfile {
            vendor: "unknown".into(),
            device_name: "CPU".into(),
            accelerators: vec!["cpu".into()],
            platform: std::env::consts::OS.into(),
            in_container: false,
            allow_experimental: false,
            vram_total_gb: None,
            vram_free_gb: None,
            ram_total_gb: None,
            ram_available_gb: None,
            host_agent_online: false,
            host_profiles: Vec::new(),
            tier: "cpu-light".into(),
        };
        profile.recompute_tier();
        profile
    }

    #[test]
    fn stack_request_rejects_browser_supplied_docker_parameters() {
        let value = r#"{"asr_id":"fw-base","image":"attacker/image","command":["sh"]}"#;
        assert!(serde_json::from_str::<ActivateStackRequest>(value).is_err());
    }

    #[tokio::test]
    async fn native_host_port_pause_restores_only_the_active_stage_container() {
        let runtime = runtime::runtime_from(test_config());
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("asr-a".into());
        control.running.lock().await.insert("tts-fail".into());
        let lab = LabController::with_control(
            switch_catalog(),
            cpu_hardware(),
            runtime,
            control.clone(),
            true,
        )
        .unwrap();
        lab.state.write().await.asr = Some(ActiveBackend {
            backend_id: "asr-a".into(),
            variant_id: "asr-a-cpu".into(),
            accelerator: "cpu".into(),
            endpoint: String::new(),
            container: "asr-a".into(),
        });

        let paused = lab
            .pause_active_container_for_host(BackendStage::Asr)
            .await
            .unwrap()
            .expect("active ASR container should be paused");
        assert!(!control.running.lock().await.contains("asr-a"));
        assert!(control.running.lock().await.contains("tts-fail"));

        lab.restore_paused_container(&paused).await.unwrap();
        let actions = control.actions.lock().await.clone();
        assert_eq!(actions, vec!["stop:asr-a", "start:asr-a"]);
        assert!(control.running.lock().await.contains("asr-a"));
        assert!(control.running.lock().await.contains("tts-fail"));
    }

    #[tokio::test]
    async fn voice_only_activation_validates_qwen_catalog_and_preserves_runtime_on_error() {
        let runtime = runtime::runtime_from(test_config());
        {
            let mut rt = runtime.write().await;
            rt.tts_id = "qwen3-tts-0.6b".into();
            rt.cfg.supertonic_voice = "Aiden".into();
        }
        let mut qwen = backend("qwen3-tts-0.6b", BackendStage::Tts, "");
        qwen.default_voice = "Aiden".into();
        qwen.voices = vec!["Aiden".into(), "Serena".into(), "Vivian".into()];
        let catalog = BackendCatalog {
            schema_version: 1,
            presets: Vec::new(),
            backends: vec![qwen],
        };
        let lab = Arc::new(
            LabController::with_control(
                catalog,
                cpu_hardware(),
                runtime.clone(),
                Arc::new(FakeControl::default()),
                false,
            )
            .unwrap(),
        );
        lab.reconcile_active_stack().await;

        let turns = runtime.read().await.turns.clone();
        let active_turn = turns.try_acquire().expect("active turn");
        let voice_lab = lab.clone();
        let selecting = tokio::spawn(async move {
            voice_lab
                .activate(ActivateStackRequest {
                    voice: Some("serena".into()),
                    ..ActivateStackRequest::default()
                })
                .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!selecting.is_finished());
        drop(active_turn);
        let selected = tokio::time::timeout(Duration::from_millis(200), selecting)
            .await
            .expect("voice update should finish after turn release")
            .unwrap();
        assert!(selected.ok, "{}", selected.message);
        assert_eq!(selected.runtime.voice, "Serena");

        let rejected = lab
            .activate(ActivateStackRequest {
                voice: Some("not-a-qwen-voice".into()),
                ..ActivateStackRequest::default()
            })
            .await;
        assert!(!rejected.ok);
        assert!(rejected.message.contains("not available"));
        assert_eq!(runtime.read().await.cfg.supertonic_voice, "Serena");
    }

    #[test]
    fn managed_target_requires_exact_registry_metadata() {
        let inspect = serde_json::json!({
            "Config": {
                "Image": "registry/asr:sha",
                "Labels": {
                    "s2s.lab.managed": "true",
                    "stage": "asr",
                    "backend-id": "fw-base"
                }
            }
        });
        assert!(validate_managed_target(
            &inspect,
            "asr",
            BackendStage::Asr,
            "fw-base",
            "registry/asr:sha"
        )
        .is_ok());
        assert!(validate_managed_target(
            &inspect,
            "asr",
            BackendStage::Asr,
            "fw-base",
            "attacker/asr:latest"
        )
        .is_err());
        assert!(validate_managed_target(
            &inspect,
            "asr",
            BackendStage::Tts,
            "fw-base",
            "registry/asr:sha"
        )
        .is_err());
    }

    #[test]
    fn provisioned_digest_overrides_catalog_tag_for_lifecycle_validation() {
        let runtime = RuntimeProvisionStatus {
            image: "registry/asr@sha256:immutable".into(),
            ..RuntimeProvisionStatus::default()
        };
        assert_eq!(
            provisioned_image(&runtime, "registry/asr:cpu"),
            "registry/asr@sha256:immutable"
        );
        assert_eq!(
            provisioned_image(&RuntimeProvisionStatus::default(), "registry/asr:cpu"),
            "registry/asr:cpu"
        );
    }

    #[test]
    fn controller_image_progress_survives_gateway_status_decode() {
        let status: RuntimeProvisionStatus = serde_json::from_str(
            r#"{"state":"missing","image_download_size_bytes":100,"image_downloaded_bytes":42}"#,
        )
        .unwrap();
        assert_eq!(status.image_downloaded_bytes, 42);
        assert_eq!(status.image_download_size_bytes, 100);
    }

    #[tokio::test]
    async fn observed_image_total_replaces_catalog_estimate() {
        let mut catalog = test_catalog("http://127.0.0.1:9".into());
        catalog.backends[0].variants[0].image_download_size_bytes = 1000;
        let control = Arc::new(FakeControl::default());
        *control.override_status.lock().await = Some(RuntimeProvisionStatus {
            state: "missing".into(),
            image_download_size_bytes: 100,
            image_downloaded_bytes: 42,
            ..RuntimeProvisionStatus::default()
        });
        let lab = LabController::with_control(
            catalog,
            cpu_hardware(),
            runtime::runtime_from(test_config()),
            control,
            true,
        )
        .unwrap();
        let status = lab.module_status("asr-a").await.unwrap();
        assert_eq!(status.image_download_size_bytes, 100);
        assert_eq!(status.image_downloaded_bytes, 42);
    }

    #[tokio::test]
    async fn unknown_backend_does_not_mutate_runtime() {
        let runtime = runtime::runtime_from(test_config());
        let before = {
            let guard = runtime.read().await;
            runtime::status_of(&guard)
        };
        let control = Arc::new(FakeControl::default());
        let lab = LabController::with_control(
            test_catalog("http://127.0.0.1:9".into()),
            cpu_hardware(),
            runtime.clone(),
            control,
            false,
        )
        .unwrap();
        let result = lab
            .activate(ActivateStackRequest {
                asr_id: Some("missing".into()),
                tts_id: None,
                llm_id: None,
                voice: None,
            })
            .await;
        let after = {
            let guard = runtime.read().await;
            runtime::status_of(&guard)
        };
        assert!(!result.ok);
        assert_eq!(before.asr, after.asr);
        assert_eq!(before.whisper_url, after.whisper_url);
    }

    #[tokio::test]
    async fn asr_switch_without_llm_id_preserves_active_llm() {
        let runtime = runtime::runtime_from(test_config());
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("asr-a".into());
        let lab =
            LabController::with_control(switch_catalog(), cpu_hardware(), runtime, control, true)
                .unwrap();
        {
            let mut state = lab.state.write().await;
            state.asr = Some(ActiveBackend {
                backend_id: "asr-a".into(),
                variant_id: "asr-a-cpu".into(),
                accelerator: "cpu".into(),
                endpoint: String::new(),
                container: "asr-a".into(),
            });
            state.llm = Some(ActiveBackend {
                backend_id: "llm-kept".into(),
                variant_id: "llm-kept-cpu".into(),
                accelerator: "cpu".into(),
                endpoint: "http://127.0.0.1:9999".into(),
                container: "llm-kept".into(),
            });
        }

        let result = lab
            .activate(ActivateStackRequest {
                asr_id: Some("asr-b".into()),
                tts_id: None,
                llm_id: None,
                voice: None,
            })
            .await;

        assert!(result.ok, "{}", result.message);
        assert_eq!(
            result.asr.as_ref().map(|item| item.backend_id.as_str()),
            Some("asr-b")
        );
        assert_eq!(
            result.llm.as_ref().map(|item| item.backend_id.as_str()),
            Some("llm-kept")
        );
    }

    #[tokio::test]
    async fn switch_keeps_exactly_one_asr_container_running() {
        let runtime = runtime::runtime_from(test_config());
        {
            let mut rt = runtime.write().await;
            rt.asr_id = "asr-a".into();
        }
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("asr-a".into());
        let lab = LabController::with_control(
            switch_catalog(),
            cpu_hardware(),
            runtime,
            control.clone(),
            true,
        )
        .unwrap();
        lab.reconcile_active_stack().await;

        let result = lab
            .activate(ActivateStackRequest {
                asr_id: Some("asr-b".into()),
                tts_id: None,
                llm_id: None,
                voice: None,
            })
            .await;

        assert!(result.ok, "{}", result.message);
        assert_eq!(
            control.running.lock().await.clone(),
            HashSet::from(["asr-b".to_string()])
        );
    }

    #[tokio::test]
    async fn reselecting_stopped_active_container_restarts_it() {
        let runtime = runtime::runtime_from(test_config());
        runtime.write().await.asr_id = "asr-a".into();
        let control = Arc::new(FakeControl::default());
        let lab = LabController::with_control(
            switch_catalog(),
            cpu_hardware(),
            runtime,
            control.clone(),
            true,
        )
        .unwrap();
        lab.state.write().await.asr = Some(ActiveBackend {
            backend_id: "asr-a".into(),
            variant_id: "asr-a-cpu".into(),
            accelerator: "cpu".into(),
            endpoint: String::new(),
            container: "asr-a".into(),
        });

        let result = lab
            .activate(ActivateStackRequest {
                asr_id: Some("asr-a".into()),
                ..ActivateStackRequest::default()
            })
            .await;

        assert!(result.ok, "{}", result.message);
        assert!(control.running.lock().await.contains("asr-a"));
        assert!(control
            .actions
            .lock()
            .await
            .contains(&"start:asr-a".to_string()));
    }

    #[tokio::test]
    async fn switching_to_native_backend_stops_previous_stage_container() {
        let runtime = runtime::runtime_from(test_config());
        {
            let mut rt = runtime.write().await;
            rt.tts_id = "tts-old".into();
        }
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("tts-old".into());
        let catalog = BackendCatalog {
            schema_version: 1,
            presets: Vec::new(),
            backends: vec![
                backend("tts-old", BackendStage::Tts, "tts-old"),
                backend("tts-native", BackendStage::Tts, ""),
            ],
        };
        let lab =
            LabController::with_control(catalog, cpu_hardware(), runtime, control.clone(), true)
                .unwrap();
        lab.reconcile_active_stack().await;

        let result = lab
            .activate(ActivateStackRequest {
                asr_id: None,
                tts_id: Some("tts-native".into()),
                llm_id: None,
                voice: None,
            })
            .await;

        assert!(result.ok, "{}", result.message);
        assert!(control.running.lock().await.is_empty());
        assert!(control
            .actions
            .lock()
            .await
            .contains(&"stop:tts-old".to_string()));
    }

    #[test]
    fn qwen_warmup_requires_non_empty_aligned_pcm() {
        let qwen = backend("qwen3-tts-0.6b", BackendStage::Tts, "");
        assert!(validate_tts_warmup_audio(&qwen, &[]).is_err());
        assert!(validate_tts_warmup_audio(&qwen, &[0]).is_err());
        assert!(validate_tts_warmup_audio(&qwen, &[0, 0]).is_ok());
    }

    #[test]
    fn xtts_warmup_requires_mono_pcm16_wav_at_24khz() {
        fn wav(sample_rate: u32) -> Vec<u8> {
            let mut cursor = std::io::Cursor::new(Vec::new());
            {
                let spec = hound::WavSpec {
                    channels: 1,
                    sample_rate,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                };
                let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
                writer.write_sample::<i16>(0).unwrap();
                writer.finalize().unwrap();
            }
            cursor.into_inner()
        }

        let xtts = backend("xtts-v2", BackendStage::Tts, "");
        assert!(validate_tts_warmup_audio(&xtts, &[]).is_err());
        assert!(validate_tts_warmup_audio(&xtts, &wav(16_000)).is_err());
        assert!(validate_tts_warmup_audio(&xtts, &wav(24_000)).is_ok());
    }

    #[tokio::test]
    async fn xtts_voice_discovery_filters_paths_and_model_delete_keeps_user_voices() {
        let root = std::env::temp_dir().join(format!("s2s-xtts-test-{}", uuid::Uuid::new_v4()));
        let voices = root.join("xtts-v2/voices");
        tokio::fs::create_dir_all(&voices).await.unwrap();
        tokio::fs::write(voices.join("de_sample.wav"), b"pinned")
            .await
            .unwrap();
        tokio::fs::write(voices.join("user_voice-2.wav"), b"user")
            .await
            .unwrap();
        tokio::fs::write(voices.join("ignored.mp3"), b"ignored")
            .await
            .unwrap();

        assert_eq!(
            discover_xtts_voice_ids_at(&root).await.unwrap(),
            vec!["de_sample", "user_voice-2"]
        );
        assert!(is_safe_xtts_voice_id("user_voice-2"));
        assert!(!is_safe_xtts_voice_id("../outside"));
        assert!(!is_safe_xtts_voice_id("voice.wav"));

        let catalog: BackendCatalog =
            serde_json::from_str(include_str!("../config/backends.json")).unwrap();
        let backend = catalog.find("xtts-v2").unwrap();
        let variant = backend
            .variants
            .iter()
            .find(|variant| variant.id == "xtts-v2-cpu-windows")
            .unwrap();
        remove_model_artifacts_at(backend, variant, &root)
            .await
            .unwrap();
        assert!(!tokio::fs::try_exists(voices.join("de_sample.wav"))
            .await
            .unwrap());
        assert!(tokio::fs::try_exists(voices.join("user_voice-2.wav"))
            .await
            .unwrap());

        tokio::fs::remove_dir_all(root).await.unwrap();
    }

    #[tokio::test]
    async fn download_client_strips_authorization_on_foreign_redirect_host() {
        async fn read_headers(stream: &mut tokio::net::TcpStream) -> String {
            let mut bytes = Vec::new();
            let mut buffer = [0u8; 1024];
            loop {
                let count = stream.read(&mut buffer).await.unwrap();
                if count == 0 {
                    break;
                }
                bytes.extend_from_slice(&buffer[..count]);
                if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            String::from_utf8(bytes).unwrap()
        }

        let target = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
        let target_port = target.local_addr().unwrap().port();
        let target_task = tokio::spawn(async move {
            let (mut stream, _) = target.accept().await.unwrap();
            let headers = read_headers(&mut stream).await;
            assert!(
                !headers.to_ascii_lowercase().contains("authorization:"),
                "authorization leaked across redirect host: {headers}"
            );
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .await
                .unwrap();
        });

        let redirect = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let redirect_addr = redirect.local_addr().unwrap();
        let redirect_task = tokio::spawn(async move {
            let (mut stream, _) = redirect.accept().await.unwrap();
            let headers = read_headers(&mut stream).await;
            assert!(headers
                .to_ascii_lowercase()
                .contains("authorization: bearer secret-marker"));
            let response = format!(
                "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.2:{target_port}/model\r\n\
                 Content-Length: 0\r\n\r\n"
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let response = build_download_client()
            .unwrap()
            .get(format!("http://{redirect_addr}/start"))
            .bearer_auth("secret-marker")
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
        redirect_task.await.unwrap();
        target_task.await.unwrap();
    }

    #[tokio::test]
    async fn reconcile_discovers_running_backend_after_controller_restart() {
        let runtime = runtime::runtime_from(test_config());
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("asr-b".into());
        let lab = LabController::with_control(
            switch_catalog(),
            cpu_hardware(),
            runtime.clone(),
            control,
            true,
        )
        .unwrap();

        lab.reconcile_active_stack().await;

        assert_eq!(runtime.read().await.asr_id, "asr-b");
        assert_eq!(
            lab.status().await.asr.map(|active| active.backend_id),
            Some("asr-b".into())
        );
    }

    #[tokio::test]
    async fn reconcile_recognizes_aurago_owned_default_asr() {
        let app = axum::Router::new().route("/health", axum::routing::get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut cfg = test_config();
        cfg.whisper_url = "http://confucius-asr:8082".into();
        cfg.stt_api = SttApi::Openai;
        let runtime = runtime::runtime_from(cfg);
        let mut hardware = cpu_hardware();
        hardware.in_container = true;
        hardware.platform = "linux".into();
        let catalog = serde_json::from_str(include_str!("../config/backends.json")).unwrap();
        let mut lab = LabController::with_control(
            catalog,
            hardware,
            runtime.clone(),
            Arc::new(FakeControl::default()),
            true,
        )
        .unwrap();
        lab.client = reqwest::Client::builder()
            .proxy(reqwest::Proxy::http(format!("http://{address}")).unwrap())
            .build()
            .unwrap();

        assert_eq!(runtime.read().await.asr_id, "confucius4-r2t2");
        assert_eq!(
            lab.client
                .get("http://confucius-asr:8082/health")
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );

        lab.reconcile_active_stack().await;

        let status = lab.status().await;
        let active = status.asr.unwrap();
        assert_eq!(active.backend_id, "confucius4-r2t2");
        assert_eq!(active.variant_id, "confucius4-r2t2-cpu");
        assert_eq!(active.endpoint, "http://confucius-asr:8082");
        assert!(active.container.is_empty());
        assert_eq!(runtime.read().await.cfg.whisper_url, active.endpoint);
        server.abort();
    }

    #[tokio::test]
    async fn switch_waits_for_active_turn_and_reopens_gate_afterwards() {
        let runtime = runtime::runtime_from(test_config());
        {
            let mut rt = runtime.write().await;
            rt.asr_id = "asr-a".into();
        }
        let turns = runtime.read().await.turns.clone();
        let active_turn = turns.try_acquire().expect("active turn");
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("asr-a".into());
        let lab =
            LabController::with_control(switch_catalog(), cpu_hardware(), runtime, control, true)
                .unwrap();
        lab.reconcile_active_stack().await;

        let switching = tokio::spawn(async move {
            lab.activate(ActivateStackRequest {
                asr_id: Some("asr-b".into()),
                tts_id: None,
                llm_id: None,
                voice: None,
            })
            .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!switching.is_finished());
        assert!(turns.try_acquire().is_none());
        drop(active_turn);
        let result = tokio::time::timeout(Duration::from_millis(200), switching)
            .await
            .expect("switch should finish after turn release")
            .unwrap();
        assert!(result.ok, "{}", result.message);
        assert!(turns.try_acquire().is_some());
    }

    #[tokio::test]
    async fn failed_second_stage_rolls_back_runtime_and_containers() {
        let runtime = runtime::runtime_from(test_config());
        {
            let mut rt = runtime.write().await;
            rt.asr_id = "asr-a".into();
        }
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("asr-a".into());
        control.fail_start.lock().await.insert("tts-fail".into());
        let lab = LabController::with_control(
            switch_catalog(),
            cpu_hardware(),
            runtime.clone(),
            control.clone(),
            true,
        )
        .unwrap();
        lab.reconcile_active_stack().await;

        let result = lab
            .activate(ActivateStackRequest {
                asr_id: Some("asr-b".into()),
                tts_id: Some("tts-fail".into()),
                llm_id: None,
                voice: None,
            })
            .await;

        assert!(!result.ok);
        assert!(!result.last_transition_error.is_empty());
        assert_eq!(runtime.read().await.asr_id, "asr-a");
        assert!(stage_is_ready(
            result.asr.as_ref(),
            &result.asr_transition,
            "asr-a"
        ));
        assert_eq!(
            control.running.lock().await.clone(),
            HashSet::from(["asr-a".to_string()])
        );
    }

    #[tokio::test]
    async fn missing_module_preflight_preserves_ready_stack() {
        let runtime = runtime::runtime_from(test_config());
        runtime.write().await.asr_id = "asr-a".into();
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("asr-a".into());
        control
            .missing_variants
            .lock()
            .await
            .insert("asr-b-cpu".into());
        let lab = LabController::with_control(
            switch_catalog(),
            cpu_hardware(),
            runtime,
            control.clone(),
            true,
        )
        .unwrap();
        lab.reconcile_active_stack().await;

        let result = lab
            .activate(ActivateStackRequest {
                asr_id: Some("asr-b".into()),
                ..ActivateStackRequest::default()
            })
            .await;

        assert!(!result.ok);
        assert_eq!(result.error_code, "module_not_provisioned");
        assert!(result
            .last_transition_error
            .contains("module_not_provisioned"));
        assert!(stage_is_ready(
            result.asr.as_ref(),
            &result.asr_transition,
            "asr-a"
        ));
        assert_eq!(
            control.running.lock().await.clone(),
            HashSet::from(["asr-a".to_string()])
        );
    }

    #[tokio::test]
    async fn bundled_model_never_downloads_or_deletes() {
        let runtime = runtime::runtime_from(test_config());
        let lab = LabController::with_control(
            test_catalog("http://127.0.0.1:9".into()),
            cpu_hardware(),
            runtime,
            Arc::new(FakeControl::default()),
            false,
        )
        .unwrap();

        let download = lab
            .start_model_download("asr-a", ModelDownloadRequest::default())
            .await
            .unwrap();
        assert_eq!(download.state, "bundled");
        assert!(lab.delete_model("asr-a").await.is_err());
    }

    #[tokio::test]
    async fn resolve_hf_token_prefers_request_then_session_then_env() {
        let runtime = runtime::runtime_from(test_config());
        let lab = LabController::with_control(
            test_catalog("http://127.0.0.1:9".into()),
            cpu_hardware(),
            runtime,
            Arc::new(FakeControl::default()),
            false,
        )
        .unwrap();

        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("S2S_HF_TOKEN");
        assert!(lab.resolve_hf_token(None).await.is_none());

        std::env::set_var("HF_TOKEN", "env-token");
        assert_eq!(
            lab.resolve_hf_token(None).await.as_deref(),
            Some("env-token")
        );
        *lab.hf_token.write().await = Some("session-token".into());
        assert_eq!(
            lab.resolve_hf_token(None).await.as_deref(),
            Some("session-token")
        );
        assert_eq!(
            lab.resolve_hf_token(Some("request-token")).await.as_deref(),
            Some("request-token")
        );
        std::env::remove_var("HF_TOKEN");
    }

    #[tokio::test]
    async fn idle_unload_stops_containers_after_last_session_leaves() {
        std::env::set_var("S2S_LAB_IDLE_UNLOAD_SECS", "1");
        let data = std::env::temp_dir().join(format!("s2s-idle-data-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&data);
        std::env::set_var("S2S_DATA_DIR", data.to_string_lossy().as_ref());
        let runtime = runtime::runtime_from(test_config());
        {
            let mut rt = runtime.write().await;
            rt.asr_id = "asr-a".into();
        }
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("asr-a".into());
        let lab = LabController::with_control(
            switch_catalog(),
            cpu_hardware(),
            runtime,
            control.clone(),
            true,
        )
        .unwrap();
        lab.reconcile_active_stack().await;
        assert!(control.running.lock().await.contains("asr-a"));

        lab.session_connected().await;
        lab.session_disconnected().await;
        // Timer is 1s; wait a bit longer for unload.
        tokio::time::sleep(Duration::from_millis(1500)).await;

        assert!(
            !control.running.lock().await.contains("asr-a"),
            "ASR container should stop after idle unload"
        );
        assert!(control
            .actions
            .lock()
            .await
            .iter()
            .any(|action| action == "stop:asr-a"));
        // Host unload request is best-effort (path from S2S_DATA_DIR); env races
        // across parallel tests make a hard file assert flaky.

        // Reconnect restarts parked models.
        lab.session_connected().await;
        assert!(control.running.lock().await.contains("asr-a"));
        std::env::remove_var("S2S_LAB_IDLE_UNLOAD_SECS");
        std::env::remove_var("S2S_DATA_DIR");
        let _ = std::fs::remove_dir_all(&data);
    }

    #[tokio::test]
    async fn idle_unload_stops_sibling_container_when_host_variant_active() {
        std::env::set_var("S2S_LAB_IDLE_UNLOAD_SECS", "1");
        let data = std::env::temp_dir().join(format!("s2s-idle-host-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&data);
        std::env::set_var("S2S_DATA_DIR", data.to_string_lossy().as_ref());

        // Backend with host (empty container) + managed sidecar — mirrors VibeVoice.
        let mut hybrid = backend("tts-hybrid", BackendStage::Tts, "s2s-tts-hybrid");
        hybrid.variants.insert(
            0,
            BackendVariant {
                id: "tts-hybrid-host".into(),
                accelerator: "remote".into(),
                vendors: vec!["any".into()],
                platforms: vec![std::env::consts::OS.into()],
                stable: true,
                published: true,
                runtime_delivery_reason: String::new(),
                device_match: vec![],
                endpoint: "http://host.docker.internal:8089/v1/audio/speech".into(),
                native_endpoint: "http://127.0.0.1:8089/v1/audio/speech".into(),
                container: String::new(),
                image: String::new(),
                image_download_size_bytes: 0,
                health_path: "/health".into(),
                environment: BTreeMap::new(),
                artifacts: Vec::new(),
                bundled: None,
                protocol: String::new(),
                host_profile: String::new(),
            },
        );
        let catalog = BackendCatalog {
            schema_version: 1,
            presets: Vec::new(),
            backends: vec![hybrid],
        };

        let runtime = runtime::runtime_from(test_config());
        {
            let mut rt = runtime.write().await;
            rt.tts_id = "tts-hybrid".into();
            rt.cfg.tts_url = "http://host.docker.internal:8089/v1/audio/speech".into();
        }
        let control = Arc::new(FakeControl::default());
        control.running.lock().await.insert("s2s-tts-hybrid".into());
        let lab =
            LabController::with_control(catalog, cpu_hardware(), runtime, control.clone(), true)
                .unwrap();
        // Force active TTS to host variant with empty container.
        {
            let mut state = lab.state.write().await;
            state.tts = Some(ActiveBackend {
                backend_id: "tts-hybrid".into(),
                variant_id: "tts-hybrid-host".into(),
                accelerator: "remote".into(),
                endpoint: "http://host.docker.internal:8089/v1/audio/speech".into(),
                container: String::new(),
            });
        }

        lab.session_connected().await;
        lab.session_disconnected().await;
        tokio::time::sleep(Duration::from_millis(1500)).await;

        assert!(
            !control.running.lock().await.contains("s2s-tts-hybrid"),
            "sibling managed container must stop even when host variant is active"
        );
        std::env::remove_var("S2S_LAB_IDLE_UNLOAD_SECS");
        std::env::remove_var("S2S_DATA_DIR");
        let _ = std::fs::remove_dir_all(&data);
    }

    #[tokio::test]
    async fn optional_model_cleanup_only_removes_catalog_artifacts() {
        let root = std::env::temp_dir().join(format!("s2s-model-test-{}", uuid::Uuid::new_v4()));
        let model_path = root.join("optional/model.bin");
        let part = root.join("optional/model.bin.part");
        let unrelated = root.join("optional/keep.txt");
        tokio::fs::create_dir_all(model_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&model_path, [1u8, 2, 3, 4]).await.unwrap();
        tokio::fs::write(&part, [9u8]).await.unwrap();
        tokio::fs::write(&unrelated, b"keep").await.unwrap();

        let mut optional = backend("optional", BackendStage::Asr, "optional");
        optional.bundled = false;
        optional.artifacts = vec![crate::registry::ModelArtifact {
            source: "https://example.invalid/model.bin".into(),
            path: "optional/model.bin".into(),
            sha256: String::new(),
            size: 4,
            auth: String::new(),
        }];
        let variant = optional.variants[0].clone();

        assert_eq!(
            model_installation_state_at(&optional, &variant, &root)
                .await
                .unwrap(),
            (true, 4)
        );
        remove_model_artifacts_at(&optional, &variant, &root)
            .await
            .unwrap();
        assert!(!tokio::fs::try_exists(model_path).await.unwrap());
        assert!(!tokio::fs::try_exists(part).await.unwrap());
        assert!(tokio::fs::try_exists(unrelated).await.unwrap());

        tokio::fs::remove_dir_all(root).await.unwrap();
    }
}
