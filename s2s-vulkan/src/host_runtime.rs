//! File-based control plane for allowlisted native Windows inference servers.
//!
//! Docker Desktop cannot expose an Intel Arc Vulkan device to the Linux
//! containers. The lab therefore writes validated commands into the shared
//! data directory and a small Windows agent owns the actual child processes.

use crate::registry::{BackendStage, BackendVariant};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

const HOST_SCHEMA_VERSION: u32 = 1;
const HEARTBEAT_MAX_AGE_SECS: u64 = 20;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HostProcessStatus {
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub backend_id: String,
    #[serde(default)]
    pub variant_id: String,
    #[serde(default)]
    pub host_profile: String,
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub pid: u32,
    #[serde(default)]
    pub endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HostAgentStatus {
    #[serde(default)]
    pub schema_version: u32,
    #[serde(default)]
    pub agent_version: String,
    #[serde(default)]
    pub updated_at_unix: u64,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub device_name: String,
    #[serde(default)]
    pub accelerators: Vec<String>,
    #[serde(default)]
    pub profiles: Vec<String>,
    #[serde(default)]
    pub processes: Vec<HostProcessStatus>,
}

impl HostAgentStatus {
    pub fn fresh(&self) -> bool {
        let now = unix_time();
        self.schema_version == HOST_SCHEMA_VERSION
            && self.updated_at_unix > 0
            && now.saturating_sub(self.updated_at_unix) <= HEARTBEAT_MAX_AGE_SECS
    }

    pub fn profile_available(&self, profile: &str) -> bool {
        self.profiles.iter().any(|candidate| candidate == profile)
    }

    pub fn runtime_state(&self, variant: &BackendVariant) -> (&str, String) {
        if !self.fresh() {
            return ("unavailable", "host agent heartbeat is stale".into());
        }
        if !self.profile_available(&variant.host_profile) {
            return (
                "unavailable",
                format!("host profile '{}' is not configured", variant.host_profile),
            );
        }
        if let Some(process) = self
            .processes
            .iter()
            .find(|process| process.variant_id == variant.id)
        {
            return (
                if process.state.is_empty() {
                    "unknown"
                } else {
                    process.state.as_str()
                },
                String::new(),
            );
        }
        ("stopped", String::new())
    }
}

#[derive(Debug, Clone, Serialize)]
struct HostCommand {
    schema_version: u32,
    request_id: String,
    action: String,
    stage: String,
    backend_id: String,
    variant_id: String,
    host_profile: String,
    endpoint: String,
    created_at_unix: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct HostCommandResult {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    request_id: String,
    #[serde(default)]
    state: String,
    #[serde(default)]
    pid: u32,
    #[serde(default)]
    error: String,
}

#[derive(Debug, Clone)]
pub struct HostRuntimeClient {
    root: PathBuf,
}

impl Default for HostRuntimeClient {
    fn default() -> Self {
        Self::new(PathBuf::from(
            std::env::var("S2S_DATA_DIR").unwrap_or_else(|_| "/data".into()),
        ))
    }
}

impl HostRuntimeClient {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub async fn status(&self) -> Result<Option<HostAgentStatus>> {
        let path = self.root.join("host-agent").join("status.json");
        let raw = match tokio::fs::read(&path).await {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("read host agent status {}", path.display()))
            }
        };
        let status: HostAgentStatus =
            serde_json::from_slice(&raw).context("parse host agent status")?;
        Ok(Some(status))
    }

    pub async fn start(
        &self,
        stage: BackendStage,
        backend_id: &str,
        variant: &BackendVariant,
        endpoint: &str,
    ) -> Result<u32> {
        self.execute("start", stage, backend_id, variant, endpoint)
            .await
    }

    pub async fn stop(
        &self,
        stage: BackendStage,
        backend_id: &str,
        variant: &BackendVariant,
        endpoint: &str,
    ) -> Result<()> {
        self.execute("stop", stage, backend_id, variant, endpoint)
            .await
            .map(|_| ())
    }

    pub async fn stop_all(&self) -> Result<()> {
        let placeholder = BackendVariant {
            id: "all-host-runtimes".into(),
            accelerator: "remote".into(),
            vendors: Vec::new(),
            platforms: Vec::new(),
            stable: true,
            device_match: Vec::new(),
            endpoint: String::new(),
            native_endpoint: String::new(),
            container: String::new(),
            image: String::new(),
            health_path: String::new(),
            environment: Default::default(),
            artifacts: Vec::new(),
            bundled: Some(true),
            protocol: String::new(),
            host_profile: "all".into(),
        };
        self.execute(
            "stop_all",
            BackendStage::Tts,
            "all-host-runtimes",
            &placeholder,
            "",
        )
        .await
        .map(|_| ())
    }

    async fn execute(
        &self,
        action: &str,
        stage: BackendStage,
        backend_id: &str,
        variant: &BackendVariant,
        endpoint: &str,
    ) -> Result<u32> {
        if action != "stop_all" {
            if variant.host_profile.is_empty() {
                return Err(anyhow!("variant '{}' is not host-managed", variant.id));
            }
            let status = self
                .status()
                .await?
                .ok_or_else(|| anyhow!("Windows host agent is not running"))?;
            if !status.fresh() {
                return Err(anyhow!("Windows host agent heartbeat is stale"));
            }
            if !status.profile_available(&variant.host_profile) {
                return Err(anyhow!(
                    "Windows host profile '{}' is not configured",
                    variant.host_profile
                ));
            }
        }

        let request_id = Uuid::new_v4().to_string();
        let command = HostCommand {
            schema_version: HOST_SCHEMA_VERSION,
            request_id: request_id.clone(),
            action: action.into(),
            stage: stage_name(stage).into(),
            backend_id: backend_id.into(),
            variant_id: variant.id.clone(),
            host_profile: variant.host_profile.clone(),
            endpoint: endpoint.into(),
            created_at_unix: unix_time(),
        };
        let commands = self.root.join("host-agent").join("commands");
        let results = self.root.join("host-agent").join("results");
        tokio::fs::create_dir_all(&commands).await?;
        tokio::fs::create_dir_all(&results).await?;
        let target = commands.join(format!("{request_id}.json"));
        write_json_atomic(&target, &command).await?;

        let result_path = results.join(format!("{request_id}.json"));
        let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
        loop {
            match tokio::fs::read(&result_path).await {
                Ok(raw) => {
                    let result: HostCommandResult =
                        serde_json::from_slice(&raw).context("parse host command result")?;
                    let _ = tokio::fs::remove_file(&result_path).await;
                    if result.schema_version != HOST_SCHEMA_VERSION
                        || result.request_id != request_id
                    {
                        return Err(anyhow!("host agent returned a mismatched result"));
                    }
                    if !result.error.is_empty() || result.state == "error" {
                        return Err(anyhow!(
                            "host agent {} failed: {}",
                            action,
                            if result.error.is_empty() {
                                "unknown error"
                            } else {
                                &result.error
                            }
                        ));
                    }
                    return Ok(result.pid);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(anyhow!(
                    "host agent did not acknowledge '{}' within 15 seconds",
                    action
                ));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

async fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    let tmp = path.with_extension("json.tmp");
    let mut file = tokio::fs::File::create(&tmp)
        .await
        .with_context(|| format!("create {}", tmp.display()))?;
    file.write_all(&bytes).await?;
    file.flush().await?;
    drop(file);
    tokio::fs::rename(&tmp, path)
        .await
        .with_context(|| format!("activate host command {}", path.display()))
}

fn stage_name(stage: BackendStage) -> &'static str {
    match stage {
        BackendStage::Asr => "asr",
        BackendStage::Tts => "tts",
        BackendStage::Llm => "llm",
    }
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_heartbeat_is_not_accepted() {
        let status = HostAgentStatus {
            schema_version: 1,
            updated_at_unix: unix_time().saturating_sub(21),
            ..Default::default()
        };
        assert!(!status.fresh());
    }

    #[test]
    fn fresh_heartbeat_exposes_allowlisted_profile() {
        let status = HostAgentStatus {
            schema_version: 1,
            updated_at_unix: unix_time(),
            profiles: vec!["crispasr-whisper".into()],
            ..Default::default()
        };
        assert!(status.fresh());
        assert!(status.profile_available("crispasr-whisper"));
        assert!(!status.profile_available("powershell"));
    }

    #[tokio::test]
    async fn start_uses_uuid_command_and_matching_result() {
        let root = std::env::temp_dir().join(format!("s2s-host-runtime-{}", Uuid::new_v4()));
        let agent_root = root.join("host-agent");
        let commands = agent_root.join("commands");
        let results = agent_root.join("results");
        tokio::fs::create_dir_all(&commands).await.unwrap();
        tokio::fs::create_dir_all(&results).await.unwrap();
        let status = HostAgentStatus {
            schema_version: 1,
            updated_at_unix: unix_time(),
            platform: "windows".into(),
            accelerators: vec!["vulkan".into()],
            profiles: vec!["crispasr-whisper".into()],
            ..Default::default()
        };
        tokio::fs::write(
            agent_root.join("status.json"),
            serde_json::to_vec(&status).unwrap(),
        )
        .await
        .unwrap();

        let responder = tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
            loop {
                let mut entries = tokio::fs::read_dir(&commands).await.unwrap();
                if let Some(entry) = entries.next_entry().await.unwrap() {
                    let command: serde_json::Value =
                        serde_json::from_slice(&tokio::fs::read(entry.path()).await.unwrap())
                            .unwrap();
                    assert_eq!(command["action"], "start");
                    assert_eq!(command["host_profile"], "crispasr-whisper");
                    assert_eq!(command["backend_id"], "fw-base");
                    let request_id = command["request_id"].as_str().unwrap();
                    Uuid::parse_str(request_id).unwrap();
                    let result = serde_json::json!({
                        "schema_version": 1,
                        "request_id": request_id,
                        "state": "starting",
                        "pid": 4242,
                        "error": ""
                    });
                    tokio::fs::write(
                        results.join(format!("{request_id}.json")),
                        serde_json::to_vec(&result).unwrap(),
                    )
                    .await
                    .unwrap();
                    break;
                }
                assert!(tokio::time::Instant::now() < deadline);
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        });

        let variant = BackendVariant {
            id: "fw-base-vulkan-windows-b580".into(),
            accelerator: "vulkan".into(),
            vendors: vec!["intel".into()],
            platforms: vec!["windows".into()],
            stable: false,
            device_match: vec!["arc b580".into()],
            endpoint: "http://host.docker.internal:8082".into(),
            native_endpoint: "http://127.0.0.1:8082".into(),
            container: String::new(),
            image: String::new(),
            health_path: "/health".into(),
            environment: Default::default(),
            artifacts: Vec::new(),
            bundled: Some(false),
            protocol: "whisper-cpp".into(),
            host_profile: "crispasr-whisper".into(),
        };
        let client = HostRuntimeClient::new(root.clone());
        let pid = client
            .start(
                BackendStage::Asr,
                "fw-base",
                &variant,
                "http://host.docker.internal:8082",
            )
            .await
            .unwrap();
        assert_eq!(pid, 4242);
        responder.await.unwrap();
        tokio::fs::remove_dir_all(root).await.unwrap();
    }

    #[tokio::test]
    async fn missing_agent_blocks_host_start() {
        let root = std::env::temp_dir().join(format!("s2s-host-runtime-{}", Uuid::new_v4()));
        let variant = BackendVariant {
            id: "supertonic-webgpu-vulkan-windows-b580".into(),
            accelerator: "vulkan".into(),
            vendors: vec!["intel".into()],
            platforms: vec!["windows".into()],
            stable: false,
            device_match: vec!["arc b580".into()],
            endpoint: "http://host.docker.internal:8085/v1/audio/speech".into(),
            native_endpoint: "http://127.0.0.1:8085/v1/audio/speech".into(),
            container: String::new(),
            image: String::new(),
            health_path: "/health".into(),
            environment: Default::default(),
            artifacts: Vec::new(),
            bundled: Some(false),
            protocol: "openai-tts".into(),
            host_profile: "supertonic-webgpu".into(),
        };
        let error = HostRuntimeClient::new(root)
            .start(
                BackendStage::Tts,
                "supertonic",
                &variant,
                "http://host.docker.internal:8085/v1/audio/speech",
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("not running"));
    }
}
