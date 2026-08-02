# Handoff pointer: AuraGo × s2s Speech Lab (for Codex)

**Date:** 2026-08-01  
**Audience:** Codex working on **AuraGo** (not re-implementing s2s).

## Primary handoff (AuraGo repo)

Open this first:

**`C:\Users\Andi\Documents\repo\AuraGo\documentation\handoff-s2s-speech-lab-codex.md`**

That document is the implementer checklist for **remaining** AuraGo work (`voice_mode` UI/validation, capability field mapping, manuals, suggestion prefill). The old “build baseline client” plan is **obsolete** — baseline is already on AuraGo `main`.

## s2s contract (API owner — this repo)

**`s2s-vulkan/docs/aurago-integration.md`**

## AuraGo operator notes

**`C:\Users\Andi\Documents\repo\AuraGo\documentation\s2s_speech_lab.md`**

## Status summary

| Layer | Status |
|-------|--------|
| s2s capability / suggestions / gateway / readiness / `voice_mode` / production env docs | **Done** in `D:\repo\s2s\s2s-vulkan` |
| AuraGo config + client + SIP/chat channels + Config UI stack editor + admin APIs | **Done** (verify; do not rewrite) |
| AuraGo remaining polish | See AuraGo handoff §4 (`voice_mode`, manuals, UX) |

## Repos

| Repo | Path |
|------|------|
| s2s | `D:\repo\s2s\s2s-vulkan` |
| AuraGo | `C:\Users\Andi\Documents\repo\AuraGo` |

## Production reminder (s2s)

```bash
export S2S_LAB_IDLE_UNLOAD_SECS=0
export S2S_ALLOW_EXPERIMENTAL=false
# AuraGo calls orchestrator :8765 (not SPA :8088) for /ready and /v1/audio/*
```

Codex should only touch s2s if an AuraGo gap requires an **additive** contract field; document it in `aurago-integration.md` first.
