# AuraGo ↔ s2s Lab integration contract

Status: **implemented contract** (capability, suggestions, gateway, readiness, and transactional stack activation).

s2s-Lab becomes a speech sidecar for [AuraGo](https://github.com/antibyte/AuraGo):

1. **Setup** — detect host capability and suggest ASR+TTS pairs (heuristic only; user always chooses).
2. **Live** — after selection, expose a stable ASR/TTS gateway for chat and classic SIP telephony.
   AuraGo keeps its own LLM/agent/Guardian.

No matrix benchmarks are required on the happy path.

## Roles

| Component | Responsibility |
|-----------|----------------|
| AuraGo | Agent, chat, SIP classic pipeline, config UI, secrets |
| s2s orchestrator (`--mode lab`) | Catalog, suggestions, readiness, stack switch, and fixed ASR/TTS gateway |
| s2s sidecars | Active ASR / TTS backends only |

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/catalog` | Backends + `hardware` capability fields |
| `GET` | `/api/v1/capability` | Capability profile only |
| `GET` | `/api/v1/suggestions?language=de&stable_only=1&max_vram_gb=8` | Heuristic ranked ASR/TTS pairs |
| `GET` | `/api/v1/stack` | Active ASR/TTS/LLM selection |
| `PUT` | `/api/v1/stack` | User-confirmed stack change; `200` only for `ok:true`, otherwise `409` |
| `POST` | `/api/v1/models/{id}/download` | Existing advanced-Lab model install API |
| `POST` | `/v1/audio/transcriptions` | Stable ASR gateway → active backend |
| `POST` | `/api/v1/asr` | Alias of transcriptions |
| `POST` | `/v1/audio/speech` | Stable TTS gateway → active backend (`model` is rejected) |
| `POST` | `/api/v1/tts` | Alias of speech |
| `GET` | `/ready` | Stack health for call answer / chat (`503` if not ready) |
| `GET` | `/health` | Process liveness |

## Capability profile

Returned by `GET /api/v1/capability` and embedded as `hardware` on the catalog.

```json
{
  "vendor": "intel",
  "device_name": "Intel Arc B580",
  "accelerators": ["cpu", "vulkan"],
  "platform": "windows",
  "in_container": true,
  "allow_experimental": false
}
```

### Field notes

| Field | Meaning |
|-------|---------|
| `vendor` / `device_name` | Detected GPU/device identity when available |
| `accelerators` | Available execution classes such as `cpu`, `vulkan`, `cuda`, or `sycl` |
| `platform` / `in_container` | Runtime placement used by catalog variant selection |
| `allow_experimental` | Whether experimental variants may be selected |

Scores are **predictions**, not measurements. UI copy should say “voraussichtlich”.

### Env overrides (ops / tests)

| Env | Effect |
|-----|--------|
| `S2S_LAB_GPU_VENDOR` | Force vendor string |
| `S2S_LAB_DEVICE_NAME` | Force device name |
| `S2S_LAB_ACCELERATORS` | Extra accelerators (comma-separated) |
| `S2S_LAB_PLATFORM` | `windows` / `linux` / `macos` |
| `S2S_ALLOW_EXPERIMENTAL` | Expose experimental catalog variants |

## Setup flow (AuraGo)

```text
1. GET /api/v1/capability
2. GET /api/v1/suggestions?language=de&stable_only=true
3. GET /api/v1/catalog                  (full manual list)
4. User picks ASR + TTS (optionally starting from a suggested pair)
5. PUT /api/v1/stack  { "asr_id", "tts_id", "voice"? } (never send llm_id)
6. GET /ready until ready=true
7. Point AuraGo whisper/tts providers at s2s gateway base URL
```

### Suggestions query parameters

| Query | Default | Meaning |
|-------|---------|---------|
| `language` | (none) | Filter backends that support this language |
| `stable_only` | `true` | Only stable selected variants |
| `max_vram_gb` | `8` | Combined ASR+TTS catalog VRAM budget |
| `limit` | `8` | Maximum pair count (1–32) |

Scores use `scoring: "heuristic_v1"` and are **not** benchmark measurements.

User may ignore suggestions and select any `available` backend.

## Live flow (AuraGo chat / SIP classic)

AuraGo `SpeechRecognizer` / `SpeechSynthesizer` call the **fixed** gateway paths, never the rotating sidecar hostnames.

| AuraGo need | s2s contract |
|-------------|------------------------|
| WAV → text | `POST /v1/audio/transcriptions` multipart `file` or raw `audio/wav`; valid PCM-WAV only, maximum 8 MiB → `{ "text", "asr_id" }` |
| Text → PCM/WAV | `POST /v1/audio/speech` JSON `{ "input", "voice", "language", "response_format": "wav" }` → audio + `x-s2s-tts-id`; a non-empty `model` is HTTP `400` and is never forwarded |
| Pre-answer check | `GET /ready` → `{ "ready", "asr_id", "tts_id", "asr_ok", "tts_ok", "message" }` (`503` unless both active IDs match the runtime, both stages are `idle`/`ready`, and both probes return non-HTML `2xx`) |
| Liveness | `GET /health` → `{ "status": "ok" }` |

LLM remains AuraGo. s2s lab pipeline LLM is unused for production telephony.
Omitting `llm_id` from a stack request preserves the active s2s LLM entry.

Gateway response limits are enforced while streaming: JSON and upstream error
bodies are capped at 1 MiB, TTS audio at 32 MiB. `Content-Length` is checked
before reading and chunked responses are stopped as soon as the limit is
exceeded. A transition in `warming`, `rollback`, or `failed` (and any other
non-terminal phase) is not ready; HTTP `404` is never treated as a successful
readiness probe.

## Production env hints

| Env | Lab default | AuraGo production |
|-----|-------------|-------------------|
| `S2S_LAB_IDLE_UNLOAD_SECS` | `120` | `0` (always warm for chat/SIP) |
| `S2S_ALLOW_EXPERIMENTAL` | operator choice | `false` for stable-first setup |
| Bind / network | UI via `:8088` | orchestrator on internal Docker network (`s2s-vulkan:8765`) or loopback-only host overlay |

AuraGo should call the orchestrator **directly** on port 8765 for `/health`, `/ready`,
and `/v1/audio/*`. The lab web UI on `:8088` only reverse-proxies `/api/` and `/ws`.
Containerized AuraGo does not require a published host port. A native AuraGo process
may use `docker/docker-compose.aurago-host.yml`, which publishes only
`127.0.0.1:8765` and never the LAN by default.

Example production snippet:

```bash
export S2S_LAB_IDLE_UNLOAD_SECS=0
export S2S_ALLOW_EXPERIMENTAL=false
# docker compose … up -d
curl -fsS http://s2s-vulkan:8765/ready
```

## Config mapping (AuraGo)

AuraGo ships `speech_lab` config + Compose overlay:

- `documentation/s2s_speech_lab.md`
- `deploy/docker/docker-compose.s2s.yml`

```yaml
speech_lab:
  enabled: true
  base_url: "http://s2s-vulkan:8765"
  advanced_ui_url: ""
  language: de
  voice: M1
  timeout_seconds: 60
  sip_enabled: false
  chat_input_enabled: false
  chat_output_enabled: false
```

AuraGo selects `speech_lab` explicitly in the Telephone agent for local ASR,
local TTS, or either hybrid combination. The three channel toggles above are
independent; there is no implicit provider override or cloud fallback.

## Non-goals

- Automatic stack switch without user confirmation
- Replacing AuraGo Realtime Speech cloud providers
- Mandatory WER/RTF benchmarks for setup
- Exposing Hugging Face tokens via catalog

## Versioning

- Catalog `schema_version` remains unchanged; the capability endpoint reuses the catalog `hardware` object.
- Suggestion scoring version string: `heuristic_v1`.
- This document is the contract; code must not silently change path names without a doc bump.
