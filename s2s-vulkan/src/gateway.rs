//! Stable ASR/TTS HTTP gateway for AuraGo chat/SIP.
//!
//! Paths stay fixed while the active stack sidecars may change after
//! `PUT /api/v1/stack`. See `docs/aurago-integration.md`.

use crate::config::Config;
use crate::lab::LabController;
use crate::stt;
use crate::tts::{apply_preloaded_voice_policy, http_is_qwen_public, iso_tts_language};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
pub struct GatewayReady {
    pub ready: bool,
    pub asr_id: String,
    pub tts_id: String,
    pub asr_ok: bool,
    pub tts_ok: bool,
    pub message: String,
    /// Echo of runtime language hint (not a measurement).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub language: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GatewaySpeechRequest {
    /// OpenAI-style field.
    #[serde(default)]
    pub input: Option<String>,
    /// Alias accepted by some clients.
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
}

impl GatewaySpeechRequest {
    pub fn text(&self) -> Result<&str> {
        self.input
            .as_deref()
            .or(self.text.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!("missing non-empty 'input' (or 'text') field"))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GatewayTranscript {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub asr_id: String,
}

#[derive(Debug, Clone)]
pub struct GatewayAudio {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub tts_id: String,
}

impl LabController {
    /// Liveness for load balancers — process is up regardless of backends.
    pub fn health_ok() -> Value {
        json!({ "status": "ok", "service": "s2s-vulkan" })
    }

    /// Readiness of the active ASR+TTS stack for AuraGo chat/SIP.
    pub async fn gateway_ready(&self) -> GatewayReady {
        let stack = self.status().await;
        let asr_id = stack.runtime.asr.clone();
        let tts_id = stack.runtime.tts.clone();
        let language = stack.runtime.language.clone();

        let asr_endpoint = stack
            .asr
            .as_ref()
            .map(|b| b.endpoint.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| stack.runtime.whisper_url.clone());
        let tts_endpoint = stack
            .tts
            .as_ref()
            .map(|b| b.endpoint.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| stack.runtime.tts_url.clone());

        let asr_ok = probe_service(&self.client, &asr_endpoint, true).await;
        let tts_ok = probe_service(&self.client, &tts_endpoint, false).await;
        let ready = asr_ok && tts_ok && !asr_id.is_empty() && !tts_id.is_empty();
        let message = if ready {
            format!("ready · asr={asr_id} tts={tts_id}")
        } else {
            let mut parts = Vec::new();
            if asr_id.is_empty() {
                parts.push("asr not selected".into());
            } else if !asr_ok {
                parts.push(format!("asr '{asr_id}' unreachable"));
            }
            if tts_id.is_empty() {
                parts.push("tts not selected".into());
            } else if !tts_ok {
                parts.push(format!("tts '{tts_id}' unreachable"));
            }
            if parts.is_empty() {
                "not ready".into()
            } else {
                parts.join("; ")
            }
        };

        GatewayReady {
            ready,
            asr_id,
            tts_id,
            asr_ok,
            tts_ok,
            message,
            language,
        }
    }

    /// Proxy transcription to the active ASR backend (`POST …/inference`).
    pub async fn gateway_transcribe(
        &self,
        wav: Vec<u8>,
        language: Option<&str>,
    ) -> Result<GatewayTranscript> {
        if wav.is_empty() {
            return Err(anyhow!("empty audio body"));
        }
        let rt = self.runtime.read().await;
        let asr_id = rt.asr_id.clone();
        let mut cfg = rt.cfg.clone();
        drop(rt);
        if let Some(lang) = language.map(str::trim).filter(|s| !s.is_empty()) {
            cfg.language = lang.to_string();
        }
        let text = stt::transcribe_wav_bytes(&self.client, &cfg, &wav)
            .await
            .context("gateway ASR")?;
        Ok(GatewayTranscript {
            text,
            language: if cfg.language == "auto" {
                None
            } else {
                Some(cfg.language)
            },
            asr_id,
        })
    }

    /// Proxy speech synthesis to the active TTS HTTP sidecar.
    pub async fn gateway_speech(&self, request: GatewaySpeechRequest) -> Result<GatewayAudio> {
        let text = request.text()?.to_string();
        let rt = self.runtime.read().await;
        let tts_id = rt.tts_id.clone();
        let cfg = rt.cfg.clone();
        drop(rt);

        let language = request
            .language
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(cfg.tts_language.as_str());
        let language = if language.eq_ignore_ascii_case("auto") {
            cfg.resolve_tts_language(Some(&cfg.language))
        } else {
            language.to_string()
        };
        // Prefer ISO codes for Supertonic-style servers; Qwen maps full names.
        let http_lang = if http_is_qwen_public(&cfg) {
            Config::qwen_tts_language(&language)
        } else {
            iso_tts_language(&language)
        };

        let fmt = request
            .response_format
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("wav");
        let voice = request
            .voice
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(cfg.supertonic_voice.as_str());
        let model = request
            .model
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(if cfg.tts_model.trim().is_empty() {
                "tts"
            } else {
                cfg.tts_model.as_str()
            });

        let mut body = json!({
            "model": model,
            "input": text,
            "language": http_lang,
            "voice": voice,
            "response_format": fmt,
        });
        apply_preloaded_voice_policy(&mut body, model);

        let url = cfg.tts_url.clone();
        if url.trim().is_empty() {
            return Err(anyhow!("no active TTS endpoint configured"));
        }
        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header(
                "Accept",
                if fmt.eq_ignore_ascii_case("pcm") {
                    "audio/pcm, audio/wav, */*"
                } else {
                    "audio/wav, audio/pcm, */*"
                },
            )
            .json(&body);
        if !cfg.tts_api_key.is_empty() {
            req = req.bearer_auth(&cfg.tts_api_key);
        }
        let response = req
            .send()
            .await
            .with_context(|| format!("gateway TTS POST {url}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let err_body = response.text().await.unwrap_or_default();
            return Err(anyhow!("TTS HTTP {status}: {err_body}"));
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or(if fmt.eq_ignore_ascii_case("pcm") {
                "audio/pcm"
            } else {
                "audio/wav"
            })
            .to_string();
        let bytes = response.bytes().await?.to_vec();
        if bytes.is_empty() {
            return Err(anyhow!("TTS returned empty body"));
        }
        Ok(GatewayAudio {
            bytes,
            content_type,
            tts_id,
        })
    }
}

async fn probe_service(client: &reqwest::Client, endpoint: &str, is_asr: bool) -> bool {
    if endpoint.trim().is_empty() {
        return false;
    }
    for url in candidate_health_urls(endpoint, is_asr) {
        match client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() || response.status().as_u16() == 404 => {
                let ctype = response
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if !ctype.contains("text/html") {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn candidate_health_urls(endpoint: &str, is_asr: bool) -> Vec<String> {
    let endpoint = endpoint.trim_end_matches('/');
    let base = endpoint
        .split("/v1/")
        .next()
        .unwrap_or(endpoint)
        .trim_end_matches('/');
    let mut urls = vec![
        format!("{base}/health"),
        format!("{base}/"),
        base.to_string(),
    ];
    if is_asr {
        // whisper-server often only answers on the root / inference path.
        urls.push(format!("{endpoint}/"));
    } else if endpoint.contains("/v1/") {
        urls.push(format!(
            "{}/health",
            endpoint
                .trim_end_matches("/audio/speech")
                .trim_end_matches('/')
        ));
    }
    urls
}

/// Unit-test helpers for URL derivation (no network).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_health_candidates_include_base_health() {
        let urls = candidate_health_urls("http://whisper-tiny:8082", true);
        assert!(urls.iter().any(|u| u.ends_with("/health")));
        assert!(urls.iter().any(|u| u == "http://whisper-tiny:8082"));
    }

    #[test]
    fn tts_health_candidates_strip_speech_path() {
        let urls = candidate_health_urls("http://supertonic:8083/v1/audio/speech", false);
        assert!(urls.iter().any(|u| u.contains("/health")));
        assert!(urls.iter().any(|u| u.starts_with("http://supertonic:8083")));
    }

    #[test]
    fn speech_request_requires_text() {
        let req = GatewaySpeechRequest {
            input: None,
            text: Some("  ".into()),
            model: None,
            voice: None,
            language: None,
            response_format: None,
        };
        assert!(req.text().is_err());
        let req = GatewaySpeechRequest {
            input: Some("Hallo".into()),
            text: None,
            model: None,
            voice: None,
            language: None,
            response_format: None,
        };
        assert_eq!(req.text().unwrap(), "Hallo");
    }
}
