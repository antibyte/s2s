//! Heuristic ASR+TTS suggestions for AuraGo / Lab setup.
//!
//! Scores are **predictions** from catalog metadata + capability profile — not
//! runtime benchmarks. See `docs/aurago-integration.md`.

use crate::registry::{
    variant_rank, BackendStage, CatalogBackendStatus, HardwareProfile, StackPreset,
};
use serde::Serialize;

pub const SCORING_VERSION: &str = "heuristic_v1";

#[derive(Debug, Clone, Default)]
pub struct SuggestionQuery {
    /// BCP-47 / short language code (`de`, `en`). Empty = no language filter.
    pub language: Option<String>,
    /// When true (default for AuraGo production), require a stable selected variant.
    pub stable_only: bool,
    /// Combined ASR+TTS VRAM budget in GB. Falls back to capability/tier defaults.
    pub max_vram_gb: Option<f32>,
    /// Max pairs / presets to return (default 8).
    pub limit: usize,
}

impl SuggestionQuery {
    pub fn from_query_pairs(pairs: &[(String, String)]) -> Self {
        let mut query = Self {
            language: None,
            stable_only: true,
            max_vram_gb: None,
            limit: 8,
        };
        for (key, value) in pairs {
            match key.as_str() {
                "language" | "lang" => {
                    let v = value.trim().to_ascii_lowercase();
                    if !v.is_empty() && v != "auto" {
                        query.language = Some(v);
                    }
                }
                "stable_only" | "stable" => {
                    query.stable_only = matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on" | ""
                    );
                }
                "max_vram_gb" | "max_vram" => {
                    if let Ok(v) = value.trim().parse::<f32>() {
                        if v.is_finite() && v > 0.0 {
                            query.max_vram_gb = Some(v);
                        }
                    }
                }
                "limit" => {
                    if let Ok(v) = value.trim().parse::<usize>() {
                        query.limit = v.clamp(1, 32);
                    }
                }
                _ => {}
            }
        }
        // Explicit false:
        for (key, value) in pairs {
            if matches!(key.as_str(), "stable_only" | "stable")
                && matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
            {
                query.stable_only = false;
            }
        }
        query
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SuggestedPreset {
    pub id: String,
    pub name: String,
    pub asr_id: String,
    pub tts_id: String,
    pub score: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuggestedPair {
    pub asr_id: String,
    pub tts_id: String,
    pub asr_name: String,
    pub tts_name: String,
    pub score: f32,
    pub reason: String,
    pub vram_gb: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuggestionCaution {
    pub backend_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuggestionsResponse {
    pub capability: HardwareProfile,
    pub suggested_presets: Vec<SuggestedPreset>,
    pub suggested_pairs: Vec<SuggestedPair>,
    pub caution: Vec<SuggestionCaution>,
    pub scoring: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_vram_gb: Option<f32>,
    #[serde(default)]
    pub note: String,
}

fn language_matches(backend_langs: &[String], wanted: &str) -> bool {
    if wanted.is_empty() || wanted == "auto" {
        return true;
    }
    if backend_langs.is_empty() {
        // Empty language list means "not constrained" for catalog entries that omit it.
        return true;
    }
    let wanted = wanted.to_ascii_lowercase();
    backend_langs.iter().any(|lang| {
        let lang = lang.to_ascii_lowercase();
        lang == "auto"
            || lang == wanted
            || lang.starts_with(&format!("{wanted}-"))
            || (wanted == "de" && matches!(lang.as_str(), "german" | "deu" | "ger"))
            || (wanted == "en" && matches!(lang.as_str(), "english" | "eng"))
    })
}

fn backend_vram_gb(status: &CatalogBackendStatus) -> f32 {
    status.backend.resources.vram_gb.max(0.0)
}

fn effective_budget(capability: &HardwareProfile, query: &SuggestionQuery) -> f32 {
    if let Some(v) = query.max_vram_gb {
        return v;
    }
    if let Some(free) = capability.vram_free_gb {
        return (free * 0.8).max(0.5);
    }
    if let Some(total) = capability.vram_total_gb {
        return (total * 0.8).max(0.5);
    }
    match capability.tier.as_str() {
        "gpu-16gb+" => 14.0,
        "gpu-8gb" | "host-vulkan" => 7.0,
        _ => 1.5, // cpu-light: prefer near-zero VRAM combos
    }
}

fn is_eligible(
    status: &CatalogBackendStatus,
    query: &SuggestionQuery,
    stage: BackendStage,
) -> bool {
    if status.backend.stage != stage {
        return false;
    }
    if !status.available {
        return false;
    }
    if let Some(variant) = status.selected_variant.as_ref() {
        if query.stable_only && !variant.stable {
            return false;
        }
    } else {
        return false;
    }
    if let Some(lang) = query.language.as_deref() {
        if !language_matches(&status.backend.languages, lang) {
            return false;
        }
    }
    true
}

fn readiness_bonus(status: &CatalogBackendStatus) -> f32 {
    let mut bonus = 0.0;
    if status.installed || status.backend.bundled || status.download_state == "bundled" {
        bonus += 0.12;
    }
    if status.host_managed {
        match status.runtime_state.as_str() {
            "running" | "stopped" | "not_managed" => bonus += 0.05,
            "unavailable" => bonus -= 0.35,
            _ => bonus -= 0.1,
        }
    } else {
        bonus += 0.08; // Docker-managed paths are preferred for AuraGo defaults
    }
    bonus
}

fn accelerator_score(status: &CatalogBackendStatus, hw: &HardwareProfile) -> f32 {
    let Some(variant) = status.selected_variant.as_ref() else {
        return 0.0;
    };
    // Lower rank is better (0 = vulkan).
    let rank = variant_rank(variant, hw) as f32;
    match rank {
        0.0 => 0.25,
        1.0 => 0.22,
        2.0 => 0.18,
        3.0 => 0.12,
        4.0 => 0.08,
        _ => 0.04,
    }
}

fn quality_hint_score(status: &CatalogBackendStatus) -> f32 {
    // Prefer slightly higher quality when budget allows (stars inverted lightly).
    let stars = status
        .backend
        .resources
        .stars
        .gpu
        .max(status.backend.resources.stars.vram);
    match stars {
        0 | 1 => 0.02,
        2 => 0.05,
        3 => 0.08,
        4 => 0.04, // heavy
        _ => 0.0,
    }
}

fn pair_score(
    asr: &CatalogBackendStatus,
    tts: &CatalogBackendStatus,
    hw: &HardwareProfile,
    budget: f32,
) -> (f32, String, f32) {
    let vram = backend_vram_gb(asr) + backend_vram_gb(tts);
    let mut score = 0.55;
    let mut reasons = Vec::new();

    score += accelerator_score(asr, hw) * 0.5;
    score += accelerator_score(tts, hw) * 0.5;
    score += readiness_bonus(asr);
    score += readiness_bonus(tts);
    score += quality_hint_score(asr) * 0.5;
    score += quality_hint_score(tts) * 0.5;

    if vram <= budget {
        score += 0.15;
        if vram <= budget * 0.5 {
            score += 0.05;
            reasons.push("leicht im VRAM-Budget");
        } else {
            reasons.push("passt ins VRAM-Budget");
        }
    } else {
        score -= 0.45;
        reasons.push("über VRAM-Budget");
    }

    if asr.installed && tts.installed {
        reasons.push("beide installiert");
    } else if asr.installed || tts.installed {
        reasons.push("teilweise installiert");
    }

    if !asr.host_managed && !tts.host_managed {
        reasons.push("Docker-fähig");
        score += 0.05;
    } else if asr.runtime_state == "unavailable" || tts.runtime_state == "unavailable" {
        reasons.push("Host-Runtime fehlt");
    }

    // Prefer known low-latency lab defaults slightly.
    if asr.backend.id == "fw-tiny" && tts.backend.id == "supertonic" {
        score += 0.08;
        reasons.push("schneller Default-Stack");
    }

    let score = score.clamp(0.0, 0.99);
    let reason = if reasons.is_empty() {
        format!("Heuristik {} · tier={}", SCORING_VERSION, hw.tier)
    } else {
        format!("{} · tier={}", reasons.join(", "), hw.tier)
    };
    (score, reason, vram)
}

fn caution_for(status: &CatalogBackendStatus, budget: f32) -> Option<SuggestionCaution> {
    if !status.available {
        return None;
    }
    let vram = backend_vram_gb(status);
    if vram > budget * 1.2 && vram >= 8.0 {
        return Some(SuggestionCaution {
            backend_id: status.backend.id.clone(),
            reason: format!(
                "~{vram:.1} GB VRAM laut Katalog — auf diesem System voraussichtlich zu schwer (Budget ~{budget:.1} GB)"
            ),
        });
    }
    if status.host_managed && status.runtime_state == "unavailable" {
        return Some(SuggestionCaution {
            backend_id: status.backend.id.clone(),
            reason: if status.runtime_reason.is_empty() {
                "Host-Profil nicht verfügbar".into()
            } else {
                status.runtime_reason.clone()
            },
        });
    }
    None
}

/// Build ranked suggestions from a resolved catalog snapshot.
pub fn build_suggestions(
    capability: HardwareProfile,
    backends: &[CatalogBackendStatus],
    presets: &[StackPreset],
    query: &SuggestionQuery,
) -> SuggestionsResponse {
    let budget = effective_budget(&capability, query);
    let asr: Vec<&CatalogBackendStatus> = backends
        .iter()
        .filter(|s| is_eligible(s, query, BackendStage::Asr))
        .collect();
    let tts: Vec<&CatalogBackendStatus> = backends
        .iter()
        .filter(|s| is_eligible(s, query, BackendStage::Tts))
        .collect();

    let mut pairs = Vec::new();
    for a in &asr {
        for t in &tts {
            let (score, reason, vram) = pair_score(a, t, &capability, budget);
            if score < 0.25 {
                continue;
            }
            pairs.push(SuggestedPair {
                asr_id: a.backend.id.clone(),
                tts_id: t.backend.id.clone(),
                asr_name: a.backend.name.clone(),
                tts_name: t.backend.name.clone(),
                score: (score * 100.0).round() / 100.0,
                reason,
                vram_gb: (vram * 100.0).round() / 100.0,
            });
        }
    }
    pairs.sort_by(|x, y| {
        y.score
            .partial_cmp(&x.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                x.vram_gb
                    .partial_cmp(&y.vram_gb)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| x.asr_id.cmp(&y.asr_id))
            .then_with(|| x.tts_id.cmp(&y.tts_id))
    });
    pairs.truncate(query.limit);

    let by_id = |id: &str| backends.iter().find(|b| b.backend.id == id);

    let mut suggested_presets = Vec::new();
    for preset in presets {
        // Optional preset language tags.
        if let Some(lang) = query.language.as_deref() {
            if !preset.languages.is_empty() && !language_matches(&preset.languages, lang) {
                continue;
            }
        }
        let Some(a) = by_id(&preset.asr_id) else {
            continue;
        };
        let Some(t) = by_id(&preset.tts_id) else {
            continue;
        };
        if !is_eligible(a, query, BackendStage::Asr) || !is_eligible(t, query, BackendStage::Tts) {
            continue;
        }
        let (mut score, reason, vram) = pair_score(a, t, &capability, budget);
        // Presets that survive filters get a small boost over ad-hoc pairs.
        score = (score + 0.06).clamp(0.0, 0.99);
        if let Some(max) = preset.max_vram_gb {
            if vram > max {
                continue;
            }
        }
        suggested_presets.push(SuggestedPreset {
            id: preset.id.clone(),
            name: preset.name.clone(),
            asr_id: preset.asr_id.clone(),
            tts_id: preset.tts_id.clone(),
            score: (score * 100.0).round() / 100.0,
            reason: format!("Preset · {reason}"),
        });
    }
    suggested_presets.sort_by(|x, y| {
        y.score
            .partial_cmp(&x.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.id.cmp(&y.id))
    });
    suggested_presets.truncate(query.limit);

    let mut caution = Vec::new();
    for status in backends {
        if !matches!(status.backend.stage, BackendStage::Asr | BackendStage::Tts) {
            continue;
        }
        if let Some(item) = caution_for(status, budget) {
            caution.push(item);
        }
    }
    caution.sort_by(|a, b| a.backend_id.cmp(&b.backend_id));
    caution.truncate(12);

    SuggestionsResponse {
        capability,
        suggested_presets,
        suggested_pairs: pairs,
        caution,
        scoring: SCORING_VERSION.into(),
        budget_vram_gb: Some((budget * 100.0).round() / 100.0),
        note: "Scores are heuristic predictions from catalog metadata and system capacity — not runtime benchmarks. User selection remains manual.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{
        BackendDefinition, BackendStage, BackendVariant, CatalogBackendStatus, ResourceEstimate,
        ResourceStars,
    };

    fn status(
        id: &str,
        stage: BackendStage,
        langs: &[&str],
        vram: f32,
        available: bool,
        stable: bool,
        host_managed: bool,
        runtime: &str,
        installed: bool,
    ) -> CatalogBackendStatus {
        CatalogBackendStatus {
            backend: BackendDefinition {
                id: id.into(),
                stage,
                name: id.into(),
                tag: String::new(),
                description: String::new(),
                protocol: "test".into(),
                model: id.into(),
                default_voice: if stage == BackendStage::Tts {
                    "default".into()
                } else {
                    String::new()
                },
                voices: vec![],
                voice_mode: crate::registry::VoiceMode::Fixed,
                native_sample_rate: 0,
                languages: langs.iter().map(|s| (*s).into()).collect(),
                licenses: vec![],
                access_url: String::new(),
                resources: ResourceEstimate {
                    vram_gb: vram,
                    ram_gb: 1.0,
                    stars: ResourceStars {
                        cpu: 2,
                        gpu: 2,
                        vram: 2,
                    },
                },
                bundled: false,
                artifacts: vec![],
                variants: vec![],
            },
            available,
            reason: String::new(),
            selected_variant: available.then(|| BackendVariant {
                id: format!("{id}-cpu"),
                accelerator: "cpu".into(),
                vendors: vec!["any".into()],
                platforms: vec!["linux".into()],
                stable,
                device_match: vec![],
                endpoint: format!("http://{id}"),
                native_endpoint: String::new(),
                container: String::new(),
                image: String::new(),
                health_path: "/health".into(),
                environment: Default::default(),
                artifacts: vec![],
                bundled: None,
                protocol: String::new(),
                host_profile: if host_managed {
                    "host-profile".into()
                } else {
                    String::new()
                },
            }),
            installed,
            download_state: if installed {
                "installed".into()
            } else {
                "missing".into()
            },
            download_size_bytes: 0,
            downloaded_bytes: 0,
            deletable: true,
            download_error: String::new(),
            host_managed,
            runtime_state: runtime.into(),
            runtime_reason: String::new(),
            auth_required: false,
            hf_token_configured: false,
        }
    }

    fn cap(tier: &str) -> HardwareProfile {
        HardwareProfile {
            vendor: "unknown".into(),
            device_name: "CPU".into(),
            accelerators: vec!["cpu".into()],
            platform: "linux".into(),
            in_container: true,
            allow_experimental: false,
            vram_total_gb: None,
            vram_free_gb: None,
            ram_total_gb: Some(16.0),
            ram_available_gb: Some(8.0),
            host_agent_online: false,
            host_profiles: vec![],
            tier: tier.into(),
        }
    }

    #[test]
    fn language_filter_drops_english_only_tts() {
        let backends = vec![
            status(
                "fw-tiny",
                BackendStage::Asr,
                &["de", "en"],
                0.25,
                true,
                true,
                false,
                "not_managed",
                true,
            ),
            status(
                "supertonic",
                BackendStage::Tts,
                &["de", "en"],
                0.0,
                true,
                true,
                false,
                "not_managed",
                true,
            ),
            status(
                "inflect-micro-v2",
                BackendStage::Tts,
                &["en"],
                0.0,
                true,
                true,
                true,
                "stopped",
                true,
            ),
        ];
        let query = SuggestionQuery {
            language: Some("de".into()),
            stable_only: true,
            max_vram_gb: Some(8.0),
            limit: 8,
        };
        let out = build_suggestions(cap("cpu-light"), &backends, &[], &query);
        assert!(out
            .suggested_pairs
            .iter()
            .all(|p| p.tts_id != "inflect-micro-v2"));
        assert!(out
            .suggested_pairs
            .iter()
            .any(|p| p.asr_id == "fw-tiny" && p.tts_id == "supertonic"));
    }

    #[test]
    fn heavy_backend_is_cautioned_on_small_budget() {
        let backends = vec![status(
            "voxtral-mini-4b-realtime",
            BackendStage::Asr,
            &["de", "en"],
            16.0,
            true,
            true,
            true,
            "stopped",
            false,
        )];
        let query = SuggestionQuery {
            language: Some("de".into()),
            stable_only: true,
            max_vram_gb: Some(8.0),
            limit: 8,
        };
        let out = build_suggestions(cap("gpu-8gb"), &backends, &[], &query);
        assert!(out
            .caution
            .iter()
            .any(|c| c.backend_id == "voxtral-mini-4b-realtime"));
    }

    #[test]
    fn unavailable_host_runtime_is_cautioned() {
        let backends = vec![status(
            "piper",
            BackendStage::Tts,
            &["de", "en"],
            0.2,
            true,
            true,
            true,
            "unavailable",
            true,
        )];
        let query = SuggestionQuery {
            language: None,
            stable_only: true,
            max_vram_gb: Some(8.0),
            limit: 8,
        };
        let out = build_suggestions(cap("cpu-light"), &backends, &[], &query);
        assert!(out.caution.iter().any(|c| c.backend_id == "piper"));
    }

    #[test]
    fn preset_boost_when_members_eligible() {
        let backends = vec![
            status(
                "fw-tiny",
                BackendStage::Asr,
                &["de", "en"],
                0.25,
                true,
                true,
                false,
                "not_managed",
                true,
            ),
            status(
                "supertonic",
                BackendStage::Tts,
                &["de", "en"],
                0.0,
                true,
                true,
                false,
                "not_managed",
                true,
            ),
        ];
        let presets = vec![StackPreset {
            id: "balanced".into(),
            name: "Sofort startklar".into(),
            asr_id: "fw-tiny".into(),
            tts_id: "supertonic".into(),
            llm_id: "local-fallback".into(),
            languages: vec!["de".into(), "en".into()],
            max_vram_gb: Some(2.0),
            latency_hint: Some("low".into()),
            quality_hint: Some("balanced".into()),
        }];
        let query = SuggestionQuery {
            language: Some("de".into()),
            stable_only: true,
            max_vram_gb: Some(8.0),
            limit: 8,
        };
        let out = build_suggestions(cap("cpu-light"), &backends, &presets, &query);
        assert_eq!(out.suggested_presets.len(), 1);
        assert_eq!(out.suggested_presets[0].id, "balanced");
        assert!(out.suggested_presets[0].score >= out.suggested_pairs[0].score);
    }

    #[test]
    fn query_parser_defaults_stable_only() {
        let q = SuggestionQuery::from_query_pairs(&[]);
        assert!(q.stable_only);
        let q = SuggestionQuery::from_query_pairs(&[("stable_only".into(), "false".into())]);
        assert!(!q.stable_only);
        let q = SuggestionQuery::from_query_pairs(&[
            ("language".into(), "DE".into()),
            ("max_vram_gb".into(), "8".into()),
            ("limit".into(), "3".into()),
        ]);
        assert_eq!(q.language.as_deref(), Some("de"));
        assert_eq!(q.max_vram_gb, Some(8.0));
        assert_eq!(q.limit, 3);
    }
}
