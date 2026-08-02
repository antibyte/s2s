# s2s-vulkan

Rust reimplementation of the [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) **pipeline shape**:

```text
mic / WebSocket  →  VAD  →  STT  →  LLM  →  TTS  →  speakers / client
```

Heavy inference is **not** forced through PyTorch-on-Vulkan. Instead, the app talks to **GGML backends that already support Vulkan**:

| Stage | Default backend | Vulkan path |
| ----- | --------------- | ----------- |
| **VAD** | Energy + hangover (CPU) | leave on CPU |
| **STT** | `whisper-server` HTTP | build [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with `-DGGML_VULKAN=1` |
| **LLM** | OpenAI-compatible `llama-server` | build [llama.cpp](https://github.com/ggerganov/llama.cpp) with `-DGGML_VULKAN=ON` |
| **TTS** | HTTP sidecars | Qwen uses SYCL on Intel; certified Vulkan or CUDA variants can be registered for other vendors |

This is the same architecture conclusion as the Python fork analysis: ~90 % of FLOPs can sit on Vulkan without rewriting PyTorch kernels.

## Docker Speech Lab

The default Docker deployment is a local, global test stack with exactly one
active ASR and one active TTS. The versioned
[`config/backends.json`](config/backends.json) catalog is the single source for
the controller, API and browser. Model changes follow
`preparing → draining → stopping → starting → warming → ready`; failed health or
warmup rolls the previous containers and runtime endpoints back.

The browser and inference sidecars never receive a Docker socket. Only the
binary's `--mode lab` controller reaches a pinned socket proxy, and it can only
start or stop pre-created containers carrying the required
`s2s.lab.managed`, `stage` and `backend-id` labels.

```powershell
Copy-Item .env.example .env
# CPU base lab:
scripts\prepare_lab.ps1 -Platform base -Build
# NVIDIA GPU containers:
scripts\prepare_lab.ps1 -Platform nvidia -Build
# Linux Intel Arc/B580 reference:
scripts\prepare_lab.ps1 -Platform intel-sycl -Build
```

Linux:

```bash
S2S_LAB_BUILD=1 scripts/prepare_lab.sh intel-sycl
```

Open `http://127.0.0.1:8766`. This is the only host-published lab port; the
controller remains on the private Compose network. The UI loads compatible choices from
`GET /api/v1/catalog`, switches through `PUT /api/v1/stack`, displays download,
health and warmup events, and stores ASR (latency/WER/CER), LLM
(TTFT/tokens/s) and TTS (TTFA/RTF) measurements in the persistent
`s2s-lab-data` volume. TTS samples additionally support blind 1–5 ratings.
Benchmark JSON/CSV is available below `/api/v1/benchmarks`.

AuraGo integration (capability profile, suggestions, planned gateway) is documented in
[`docs/aurago-integration.md`](docs/aurago-integration.md). The orchestrator exposes:

- `GET /api/v1/capability` — RAM/VRAM (best effort), host-agent status, capacity `tier`
- `GET /api/v1/suggestions?language=de&stable_only=true` — heuristic ASR+TTS presets/pairs
  (not benchmarks; user still chooses via `PUT /api/v1/stack`)
- Stable live gateway (paths fixed across stack switches):
  - `POST /v1/audio/transcriptions` (alias `/api/v1/asr`)
  - `POST /v1/audio/speech` (alias `/api/v1/tts`)
  - `GET /ready`, `GET /health`
- Lab UI shows system tier + **Empfohlen** badges from suggestions (manual pick still)
- AuraGo production: `S2S_LAB_IDLE_UNLOAD_SECS=0` (see `.env.example` and `docs/aurago-integration.md`)

`faster-whisper tiny`, Supertonic and Granite 3.3 2B Q4 are immutable parts of
their respective images and form the first active stack. All other weights are
absent in a fresh model volume. Selecting one opens a size confirmation; only
after approval does the controller download it. A running download can be
cancelled, and an installed model can be deleted again while it is inactive:

```text
POST   /api/v1/models/{backend-id}/download   start after user approval
DELETE /api/v1/models/{backend-id}/download   cancel
DELETE /api/v1/models/{backend-id}            delete inactive model files
```

The Lab Compose file forces legacy first-boot download flags off even if an old
`.env` still contains them. This prevents model traffic before UI approval.

Accelerator resolution is catalog-driven: certified Vulkan first, CUDA on
NVIDIA, explicit SYCL variants on Intel (Qwen), then CPU. Experimental variants
remain hidden unless `S2S_ALLOW_EXPERIMENTAL=true`.

### NVIDIA Parakeet ASR

`nvidia/parakeet-tdt-0.6b-v3` is available as an optional multilingual ASR
sidecar. Its pinned Transformers artifact set is 2,509,473,204 bytes and is
downloaded only after confirmation in the Lab UI. The duplicate NeMo archive
is intentionally not downloaded.

The sidecar accepts the same multipart `/inference` contract as
faster-whisper and supports CPU, NVIDIA CUDA and Linux Intel XPU/SYCL. PyTorch
does not provide a Vulkan execution backend for this model, so the catalog
selects CUDA on NVIDIA, experimental XPU on an Intel Linux profile, and CPU on
Windows Docker Desktop. Parakeet automatically detects the language; the
language selector is retained for a consistent Lab API.

### Mistral Voxtral Mini 4B Realtime ASR

Catalog id: `voxtral-mini-4b-realtime` —
[mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).

- Multilingual realtime ASR (~13 languages); Apache-2.0 weights (~8.9 GB BF16).
- Lab sidecar: Transformers HTTP on multipart `/inference` (same contract as Whisper/Parakeet) for VAD utterances.
- Production streaming path: vLLM Realtime API (`/v1/realtime`) — not required for the lab turn pipeline.
- Practical NVIDIA VRAM **~16 GB+**; host port **8087**. Profiles: `voxtral` / `managed`.
- Lab: download the model, then select **Voxtral Mini 4B Realtime** (CUDA or host server on `:8087`).

```powershell
# After Lab download (or: hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir models/voxtral-mini-4b-realtime)
docker build -f docker/Dockerfile.voxtral --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 -t s2s-asr-voxtral:cuda .
# orchestrator:
#   --whisper-url http://127.0.0.1:8087
```

### Inflect Micro v2 (9.3M EN TTS)

Catalog id: `inflect-micro-v2` —
[owensong/Inflect-Micro-v2](https://huggingface.co/owensong/Inflect-Micro-v2)
(Apache-2.0).

- English-only, one fixed male voice, ~37.5 MB FP32 (~9.36M params).
- Python OpenAI sidecar on port **8095**; host profile `inflect-python`.
- Fits the ≤8 GB stack (CPU, near-zero VRAM).
- Setup: `.\scripts\setup_inflect.ps1`, then host agent + Lab download/switch.
- Preset: **`english-compact`** (fw-tiny + Inflect + Granite).

**Not for German.** Prefer Piper / CosyVoice / Qwen3-TTS for DE.

```powershell
.\scripts\setup_inflect.ps1
.\scripts\host_idle_agent.ps1
# Lab: download Inflect-Micro-v2 artifacts, select TTS, speak English
```

### Multilingual CrispASR backends (8 GB-friendly)

Optional catalog backends served by the existing Windows host agent and
[CrispASR](https://github.com/CrispStrobe/CrispASR) (`S2S_HOST_CRISPASR_EXE`).
They use the same HTTP contracts as the rest of the lab (`POST /inference` for
ASR, OpenAI `POST /v1/audio/speech` for TTS). LLM stays **Granite 2B** and/or
**external OpenAI/Ollama**.

| Catalog id | Stage | Host profile | Port | Notes |
|------------|-------|--------------|------|--------|
| `qwen3-asr-0.6b` | ASR | `crispasr-qwen3-asr` | 8082 | ~30 languages, Q4_K ~631 MB |
| `canary-1b-v2` | ASR | `crispasr-canary` | 8082 | 25 EU + speech translation, Q4_K ~611 MB |
| `fun-asr-mlt-nano` | ASR | `crispasr-funasr-mlt` | 8082 | 31 languages, Q4_K ~897 MB |
| `piper` | TTS | `crispasr-piper` | **8092** | ~30 MB voices (DE thorsten + EN libritts) |
| `cosyvoice3-0.5b` | TTS | `crispasr-cosyvoice3` | **8093** | 9 languages + dialects, baked voices |
| `omnivoice` | TTS | `crispasr-omnivoice` | **8094** | 600+ languages + tokenizer |

Preset: **`multilingual-8gb`** → Qwen3-ASR + Piper + local Granite.

Host-CPU variants are **Windows-only** and stable when the host agent is running
(Linux Docker does not advertise them — there is no host agent path yet). Vulkan
(Arc B580) variants remain experimental (`S2S_ALLOW_EXPERIMENTAL=true`). ASR
profiles share port **8082** (one active ASR at a time), like Whisper/Parakeet.
Host launches use `-l auto` for multilingual defaults; TTS requests WAV for
Piper/CosyVoice3/OmniVoice (not raw PCM).

```powershell
.\scripts\host_idle_agent.ps1
# Lab: download model artifacts, then switch ASR/TTS in the UI
# or point the orchestrator directly:
#   --whisper-url http://127.0.0.1:8082
#   --tts http --tts-url http://127.0.0.1:8092/v1/audio/speech
```

### Qwen Intel SYCL

The SYCL sidecar is reproducibly pinned to qwentts.cpp `82cd05b` and ggml
`c044c6f`. Its multi-stage build uses oneAPI 2026.1 for compilation and the
2026.0 runtime image. The B580 image builds AOT with `bmg_g21`; the second image
keeps portable JIT. Runtime defaults are Flash Attention on, graph execution
off, DMMV prioritized, Level Zero selected, Q4_K_M talker, Q8_0 codec,
FP16 clamp, 256 frames maximum and the exact host sampler.

The catalog pins both Qwen GGUFs to a specific Hugging Face repository
revision, expected byte sizes and SHA-256 digests. The controller downloads
them to `/models/qwen` through `.part` files before it stops the currently
active TTS backend.

The local oneAPI installation at `D:\Intel\oneAPI` can build and validate the
same pinned patch series without modifying the dirty reference checkout:

```powershell
scripts\build_qwentts_pinned.ps1
```

Docker Desktop cannot enumerate an Intel Arc GPU inside its Linux VM. For the
experimental native Windows bridge, first verify that the SYCL server returns
non-empty PCM at `http://127.0.0.1:8083/v1/audio/speech`, then configure:

```dotenv
S2S_LAB_PLATFORM=windows
S2S_LAB_GPU_VENDOR=intel
S2S_LAB_DEVICE_NAME=Intel Arc B580
S2S_LAB_ACCELERATORS=sycl,cpu
S2S_ALLOW_EXPERIMENTAL=true
```

The controller still performs its own health request and a real 16-frame
German synthesis before switching the active TTS endpoint. Empty or malformed
PCM keeps the variant unavailable and restores the previous backend.

`QWEN_CODE_SAMPLER=sycl` enables the experimental device sampler; `auto` stays
on `host` until the speed, WER, blind-rating and EOS promotion gates pass.
Pre-rebase and rebased patch sets are documented in
[`patches/qwentts/README.md`](patches/qwentts/README.md).

## Why not `--device vulkan`?

The original Python stack hard-wires PyTorch devices (`cuda` / `mps` / `cpu`) for Silero, Parakeet, and several TTS paths. Desktop Vulkan is not a first-class PyTorch device. Swapping the **STT / LLM / TTS processes** for GGML servers is the practical route.

## Quick start (Windows)

For Intel Arc on Docker Desktop, keep the controller and web UI in Docker and
run Vulkan backends through the native allowlisted host supervisor. First copy
any existing named-volume data without deleting or overwriting it:

```powershell
.\scripts\migrate_windows_volumes.ps1
$env:S2S_MODELS_HOST_DIR = "$PWD\models"
$env:S2S_DATA_HOST_DIR = "$PWD\data"
$env:S2S_ALLOW_EXPERIMENTAL = "true"
$env:S2S_WINDOWS_LAB_ACCELERATORS = "vulkan,sycl,cpu"
$env:S2S_HOST_QWEN_SYCL_EXE = "D:\tools\qwentts\build-sycl\bin\Release\tts-server.exe"
$env:S2S_ONEAPI_ROOT = "D:\Intel\oneAPI"
.\scripts\host_idle_agent.ps1
```

In a second terminal start the Windows overlay:

```powershell
docker compose -f docker-compose.yml -f docker/docker-compose.windows.yml `
  --profile backends --profile web up -d --build
```

The agent reports `vulkaninfo` device data and configured profiles in
`data\host-agent\status.json`. It accepts only UUID command files emitted by the
controller, fixed backend/profile/port combinations, configured executables and
model paths below `S2S_MODELS_HOST_DIR`. Unknown commands and unmanaged port
owners are rejected. Experimental variants remain hidden unless
`S2S_ALLOW_EXPERIMENTAL=true`; Higgs has no Vulkan variant.

### 1. Build this app

```powershell
cd s2s-vulkan
cargo build --release
# Native Supertonic WebGPU/Vulkan sidecar:
cargo build --release --features supertonic-webgpu
```

### 2. Build Vulkan backends

**whisper.cpp**

```powershell
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
# binary: build\bin\Release\whisper-server.exe  (or build\bin\whisper-server)
```

**llama.cpp**

```powershell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
# binary: build\bin\Release\llama-server.exe
```

**Qwen3-TTS (optional, neural)**

Build the Vulkan wheel for `qwentts-cpp-python` (needs Vulkan SDK + CMake + VS Build Tools):

```powershell
$env:GGML_BACKEND = "Vulkan0"
# after building/installing faster-qwen3-tts with vulkan backend:
pip install fastapi uvicorn soundfile numpy
python scripts\tts_qwen_server.py --port 8083 --quant Q4_K_M
```

### 3. Start servers

```powershell
# Terminal A — STT
whisper-server -m models\ggml-small.bin --host 127.0.0.1 --port 8082 --language auto --no-timestamps

# Terminal B — LLM
llama-server -m models\your-3b-or-8b-Q4.gguf -ngl 999 -c 8192 --port 8081

# Terminal C — TTS (pick one)
# neural:
python scripts\tts_qwen_server.py --port 8083 --quant Q4_K_M
# or skip neural and use Windows SAPI for bring-up:
#   (no server — use --tts system)
```

### 4. Run the pipeline

```powershell
# Bring-up with system TTS (no neural TTS required)
.\target\release\s2s-vulkan.exe --mode local --tts system

# Full local Vulkan stack
.\target\release\s2s-vulkan.exe `
  --mode local `
  --whisper-url http://127.0.0.1:8082 `
  --llm-base-url http://127.0.0.1:8081/v1 `
  --model-name local-model `
  --tts http `
  --tts-url http://127.0.0.1:8083/v1/audio/speech `
  --language de
```

Ryzen 7 5825U (Renoir iGPU) starting point:

```text
VAD:  energy CPU
STT:  whisper base/small, Vulkan
LLM:  3B–8B Q4, llama.cpp Vulkan, -ngl 999
TTS:  Qwen3-TTS GGML Q4_K_M Vulkan  (often the bottleneck)
```

## Modes

| `--mode` | Transport |
| -------- | --------- |
| `local` (default) | Microphone + speakers (`cpal`) |
| `websocket` | Raw 16 kHz mono i16 LE PCM over `ws://host:port/` |
| `lab` | Backend controller API + raw PCM WebSocket (Docker default) |
| `realtime` | Minimal OpenAI Realtime subset at `ws://host:port/v1/realtime` |
| `tts-server` | Supertonic OpenAI-compatible HTTP sidecar |

List devices:

```powershell
s2s-vulkan --list_devices
```

## CLI (main flags)

```
--mode local|websocket|realtime
--vad energy
--thresh 0.55
--min-speech-ms 384
--min-silence-ms 400

--whisper-url http://127.0.0.1:8082
--language auto|en|de|…

--llm-base-url http://127.0.0.1:8081/v1
--model-name <id>
--system-prompt "…"
--llm-stream true
--temperature 0.7
--max-tokens 256

--tts http|piper|system
--tts-url http://127.0.0.1:8083/v1/audio/speech
--piper-bin piper --piper-model voice.onnx

--host 127.0.0.1 --port 8765
--skip-health
--list-devices
```

Env overrides: `S2S_WHISPER_URL`, `S2S_LLM_URL`, `S2S_LLM_API_KEY`, `S2S_LLM_MODEL`, `S2S_TTS_URL`, `GGML_BACKEND`.

## Architecture

```text
                    ┌─────────────┐
  PCM chunks  ─────►│     VAD     │── VadAudio ──►┌─────────────┐
                    │  (energy)   │               │     STT     │
                    └─────────────┘               │ whisper HTTP│
                                                  └──────┬──────┘
                                                         │ Transcription
                                                         ▼
                                                  ┌─────────────┐
                                                  │     LLM     │
                                                  │ llama HTTP  │
                                                  └──────┬──────┘
                                                         │ LlmChunk (sentences)
                                                         ▼
                                                  ┌─────────────┐
                                                  │     TTS     │
                                                  │ http/piper  │
                                                  └──────┬──────┘
                                                         │ AudioOut PCM
                                                         ▼
                                                      speakers / WS
```

Each stage is a **Tokio task** with an `mpsc` channel — same idea as the Python `BaseHandler` + `Queue` design, without threads-per-handler.

Turn taking: after VAD emits a final segment, listening pauses until TTS signals `response_done` (mirrors HF “normal” mode `should_listen`).

## Mapping to the Python project

| Python | Rust (`s2s-vulkan`) |
| ------ | ------------------- |
| `BaseHandler` + queues | Tokio tasks + `mpsc` |
| `VADHandler` (Silero torch) | Energy VAD (CPU); Silero ONNX can be added later |
| Parakeet / Whisper STT | `WhisperCpp` via HTTP `/inference` |
| `responses-api` / `chat-completions` | OpenAI chat completions client (llama-server) |
| Qwen3-TTS in-process | HTTP wrapper `tts_qwen_server.py` (Vulkan GGML) |
| `--mode realtime` | Subset of Realtime events |
| `--mode websocket` | Raw PCM WebSocket |
| `--mode local` | `cpal` capture/playback |

## What is intentionally out of scope (v0.1)

- Full OpenAI Realtime tool-calling / interruption / speculative turns  
- In-process `qwentts.cpp` FFI (use the small Python HTTP wrapper)  
- Silero ONNX VAD (energy VAD is enough for PTT / quiet rooms)  
- Progressive live captions during speech  

## TTS backends

| `--tts` | Engine | GPU |
|--------|--------|-----|
| `auto` (default) | Supertonic if models present, else system | — |
| `supertonic` | Supertonic 3 (ONNX Runtime CPU; optional native WebGPU/Dawn Vulkan sidecar) | CPU / experimental Vulkan |
| `http` | External server (Qwen3/qwentts, Kokoro, Chatterbox, **Higgs TTS 3**, `supertonic serve`, …) | depends on server |
| `piper` | Piper CLI | CPU |
| `system` | Windows SAPI / espeak-ng | CPU |

### Idle model unload

When the last lab WebSocket client disconnects, managed ASR/TTS/LLM containers
are stopped after **2 minutes** (`S2S_LAB_IDLE_UNLOAD_SECS=120`) to free GPU/RAM.
All managed ASR/TTS/LLM sidecars are parked (including containers behind a host
variant such as VibeVoice). The Windows supervisor stops only native child PIDs
that it started. The older `/data/idle-unload.request` and
`/data/idle-reload.request` files remain accepted as compatibility inputs.
Reconnecting cancels the timer; if models were already parked they are started
and warmed again. Set `S2S_LAB_IDLE_UNLOAD_SECS=0` to disable.

### VibeVoice Realtime 0.5B TTS (optional lab backend)

Catalog id: `vibevoice-realtime-0.5b` —
[cstr/vibevoice-realtime-0.5b-GGUF](https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF)
(Microsoft VibeVoice-Realtime-0.5B, MIT).

- Low-latency streaming TTS served by **CrispASR** (`--backend vibevoice-tts`).
- OpenAI-compatible `POST /v1/audio/speech` on port **8089** (not 8766 — that is the web UI).
- Q4_K talker (~700 MB) + preset voice packs (EN/DE/FR/ES/…); default voice `emma`.
- Lab: download the model, build/start the sidecar, switch TTS to **VibeVoice Realtime 0.5B**.

```powershell
# After Lab download (or hf download cstr/vibevoice-realtime-0.5b-GGUF --local-dir models/vibevoice)
.\scripts\start_vibevoice.ps1 -Build
# --tts http --tts-url http://127.0.0.1:8089/v1/audio/speech --tts-model vibevoice-realtime-0.5b
```

### Chatterbox Multilingual V3 (experimental lab backend)

Catalog id: `chatterbox-multilingual-v3` —
[ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)
([official runtime](https://github.com/resemble-ai/chatterbox), MIT).

- Multilingual V3 supports 23 languages including German and emits 24 kHz audio.
- Intel Arc B580 can use the experimental native
  [CrispASR](https://github.com/CrispStrobe/CrispASR) GGML Vulkan variant.
  It keeps the autoregressive T3 stage on CPU because this CrispASR build
  documents a Vulkan step-graph crash there, while S3Gen runs on Vulkan.
  This is a genuine hybrid Vulkan path, not a silent CPU-only fallback.
- The official PyTorch CPU/CUDA variants remain available as fallbacks. NVIDIA
  continues to select CUDA; other Windows GPUs select CPU until separately
  qualified.
- The catalog pins every required model file to a revision, byte size and
  SHA-256. With experimental variants enabled, the Lab UI downloads the two
  Vulkan GGUFs for an Arc B580. The isolated Python runtime is only required
  for the CPU/CUDA variants:

```powershell
$env:S2S_ALLOW_EXPERIMENTAL = "true"
.\scripts\host_idle_agent.ps1

# CPU/CUDA fallback only:
.\scripts\setup_chatterbox.ps1
```

The supervisor starts the OpenAI-compatible sidecar at
`http://127.0.0.1:8090/v1/audio/speech`. The Vulkan variant uses the
`crispasr-chatterbox` host profile; CPU/CUDA use `chatterbox-python`. Both use
the bundled voice. Restart a previously running host agent after updating so it
advertises the new Vulkan profile.

### XTTS-v2 ONNX (experimental voice-cloning backend)

Catalog id: `xtts-v2` —
[coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) with the gated
[XTTSv2 Streaming ONNX conversion](https://huggingface.co/pltobing/XTTSv2-Streaming-ONNX).
The conversion is CC BY-NC 4.0 and is therefore not enabled for commercial use.

- Output is mono PCM/WAV at 24 kHz. The catalog exposes all 17 supported
  languages, including German.
- `xtts-v2-webgpu-vulkan-windows-b580` uses ONNX Runtime 1.28's native WebGPU
  plugin with Dawn forced to `Vulkan`, one shared Intel Arc B580 device and
  graph capture disabled for the dynamic GPT shapes. It is never considered
  healthy when Dawn reports D3D12 or GPT/HiFiGAN falls back completely to CPU.
- `xtts-v2-cpu-windows` uses the same FP32 ONNX files and is the rollback
  target when Vulkan activation fails. Both variants require
  `S2S_ALLOW_EXPERIMENTAL=true`.
- Voice names are safe ids from `models\xtts-v2\voices\*.wav`; arbitrary paths
  are rejected. `de_sample.wav` is the pinned default. Adding an administrative
  WAV file adds its stem to the Lab voice list. Deleting model artifacts does
  not remove other user-provided voice WAVs.
- Accept the gated repository terms. Provide a Hugging Face token either in the
  Lab UI download dialog (recommended) or via `HF_TOKEN` / `S2S_HF_TOKEN` in the
  server/Compose environment. The catalog never returns the token. Then create
  the isolated CPython 3.11 runtime if you use the host Python path:

```powershell
# Optional server-side token (UI entry is preferred for day-to-day use):
$env:HF_TOKEN = "hf_..."
$env:S2S_ALLOW_EXPERIMENTAL = "true"
.\scripts\setup_xtts_v2.ps1
.\scripts\host_idle_agent.ps1
```

The setup uses the hash-locked Windows dependency manifest, verifies the
upstream Python sources and `metadata.json` against their pinned Git blob ids,
and never prints the token. The Lab catalog independently verifies every large
ONNX artifact by byte size and SHA-256. The sidecar listens on
`http://127.0.0.1:8091/v1/audio/speech`; `/health` reports provider, Dawn
backend, adapter ids and WebGPU/CPU node counts per session.

### Higgs TTS 3 4B (optional lab backend)

Catalog id: `higgs-tts-3-4b` — [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b).

- OpenAI-compatible `POST /v1/audio/speech` via **SGLang-Omni** (Docker image `lmsysorg/sglang-omni:dev`).
- ~9.3 GB weights; practical NVIDIA VRAM **~24 GB+** (40 GB known-good). Research/non-commercial license.
- Lab: download the model, then switch TTS to **Higgs TTS 3 4B** (starts `s2s-tts-higgs` on port **8086**).
- Host helper: `scripts\start_higgs.ps1` (or compose profile `higgs` with the NVIDIA overlay).

```powershell
# After Lab download (or: hf download bosonai/higgs-tts-3-4b --local-dir models/higgs-tts-3-4b)
.\scripts\start_higgs.ps1 -Build
# orchestrator:
#   --tts http --tts-url http://127.0.0.1:8086/v1/audio/speech --tts-model bosonai/higgs-tts-3-4b
```

### Language selection (STT + TTS)

| Flag / env | Purpose |
|------------|---------|
| `--language` / `S2S_LANGUAGE` | STT language hint (`auto`, `en`, `de`, …) |
| `--tts-language` / `S2S_TTS_LANGUAGE` | TTS language for Supertonic, HTTP TTS, etc. |

Resolution for TTS when `--tts-language=auto`:

1. language detected on the current STT turn (if any)
2. else `--language` if not `auto`
3. else `en`

Force German speech output:

```bash
s2s-vulkan --tts supertonic --tts-language de --language de
# Docker / compose:
# S2S_TTS_LANGUAGE=de S2S_LANGUAGE=de
```

Supertonic also accepts `na` (language-agnostic).

### Supertonic (recommended CPU quality)

```bash
# one-time model download (~ONNX assets + voice styles)
python scripts/download_supertonic.py
# → models/supertonic/onnx + models/supertonic/voice_styles

cargo run --release -- \
  --mode websocket --host 0.0.0.0 --port 8765 \
  --tts supertonic \
  --supertonic-model-dir models/supertonic/onnx \
  --supertonic-voice M1 \
  --supertonic-steps 8 \
  --tts-language de \
  --tts-sample-rate 16000
```

Supertonic is **not** accelerated via `GGML_BACKEND=Vulkan0`. That flag still applies to whisper/llama/qwentts. Supertonic uses ONNX on CPU by design (official Rust: GPU not supported yet).

### Qwen3-TTS (Vulkan or Intel SYCL quality path)

Use `--tts http` against the catalog-selected Qwen sidecar. Intel Arc uses the
SYCL image; Vulkan stays available only for hardware/backend combinations whose
self-test is registered as stable.

## Web test UI (`web/`)

Optional browser lab for the raw PCM WebSocket backend:

- Full-viewport reactive particle / waveform canvas (mic + playback energy)
- Large animated circular hold-to-talk / toggle button
- Connection panel, level meters, log

### Local (no Docker)

```bash
# terminal 1 — backend in websocket mode
cargo run --release -- --mode websocket --host 0.0.0.0 --port 8765

# terminal 2 — HTTPS UI (required for microphone from other devices)
cd web
python serve.py --host 0.0.0.0 --port 9999 --backend 127.0.0.1:8765
# open https://127.0.0.1:9999  (or https://<LAN-IP>:9999 from another PC)
```

Browsers only allow the microphone in a **secure context** (`https://` or `http://localhost`).
`serve.py` generates a self-signed cert and proxies `wss://…/ws` → the backend so there is no mixed content.

From another PC: open `https://<this-machine-ip>:9999`, accept the certificate warning once, Connect, hold the orb.

### Docker (optional profile)

```bash
docker compose --profile web up -d --build
# UI:  http://localhost:8766
# WS:  same origin /ws  →  proxied to s2s:8765
```

For HTTPS microphone access while keeping the complete Docker stack, the local
HTTPS server can proxy both the Lab API and WebSocket through the Docker web
gateway:

```bash
cd web
python serve.py --host 0.0.0.0 --port 9999 \
  --backend 127.0.0.1:8766 --backend-ws-path /ws
```

Also included in `--profile full`.

Env: `WEB_PORT=8766`.

> The Compose default is `--mode lab`, which includes binary PCM WebSocket
> transport and the controller API.

## GitHub Container Registry (GHCR)

CI workflow: [`.github/workflows/ghcr.yml`](../.github/workflows/ghcr.yml) (monorepo root) and [`s2s-vulkan/.github/workflows/ghcr.yml`](.github/workflows/ghcr.yml).

On push to `main` / tags `v*`:

```text
ghcr.io/<github-owner>/s2s-vulkan:latest
ghcr.io/<github-owner>/s2s-vulkan:sha-<short>
ghcr.io/<github-owner>/s2s-vulkan:1.2.3   # from tag v1.2.3
```

### Publish (after first push)

1. Push this repo to GitHub (Actions enabled).
2. Workflow builds and pushes with `GITHUB_TOKEN`.
3. Package settings → set visibility **Public** if needed.
4. Pull:

```bash
# GitHub Packages often needs a login even for public images
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
docker pull ghcr.io/USERNAME/s2s-vulkan:latest
```

### Run from GHCR

```bash
export GHCR_OWNER=your-user   # lowercase
# or: export IMAGE=ghcr.io/your-user/s2s-vulkan TAG=latest

docker compose pull
docker compose up -d
# Tiny + Supertonic + Granite are already inside their sidecar images.
```

Optional models are managed through the Lab UI/API. The old download helper is
kept only for explicit maintenance outside the Lab lifecycle:

```bash
docker compose run --rm \
  -e S2S_DOWNLOAD_MODELS=true \
  -e S2S_DOWNLOAD_WHISPER=true \
  -e S2S_DOWNLOAD_FORCE=1 \
  model-init
```

### Legacy download-helper env reference

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `S2S_DOWNLOAD_MODELS` | `false` | `true` only for an explicit helper invocation |
| `S2S_DOWNLOAD_FORCE` | `0` | `1` re-download |
| `S2S_DOWNLOAD_WHISPER` | `false` | Fetch Whisper weights |
| `S2S_DOWNLOAD_LLM` | `false` | Fetch GGUF LLM |
| `S2S_DOWNLOAD_TTS` | `false` | Optional TTS weights |
| `S2S_WHISPER_PRESET` | `tiny` | `tiny`…`large-v3-turbo` |
| `S2S_WHISPER_HF_REPO` | `ggerganov/whisper.cpp` | HF repo |
| `S2S_WHISPER_HF_FILE` | (from preset) | Exact file on repo |
| `S2S_WHISPER_MODEL_URL` | — | Direct URL override |
| `S2S_LLM_HF_REPO` | `Edge-Quant/granite-3.3-2b-instruct-Q4_K_M-GGUF` | HF repo |
| `S2S_LLM_HF_FILE` | `granite-3.3-2b-instruct-q4_k_m.gguf` | File on repo |
| `S2S_LLM_MODEL_URL` | — | Direct URL override |
| `S2S_DOWNLOAD_EXTRA` | — | `url=>relpath,url2\|relpath2` |
| `S2S_HF_TOKEN` / `HF_TOKEN` | — | Optional server-side gated HF access (UI download dialog can also supply a token); never exposed through the catalog API |
| `S2S_MODELS_DIR` | `/models` | Volume mount path |

Examples of controller-managed optional model paths:

```text
/models/faster-whisper/base/model.bin
/models/qwen/qwen-talker-0.6b-customvoice-Q4_K_M.gguf
/models/kokoro/kokoro-v1.0.onnx
```

Manual download only:

```bash
docker run --rm -v s2s-models:/models \
  -e S2S_DOWNLOAD_MODELS=true \
  -e S2S_DOWNLOAD_WHISPER=true \
  -e S2S_WHISPER_PRESET=base \
  ghcr.io/USERNAME/s2s-vulkan:latest download-models
```

## Docker (GPU auto-detect)

The orchestrator image probes the container environment at startup:

1. `GGML_BACKEND` / `S2S_GPU` if already set  
2. NVIDIA (`nvidia-smi`, `/dev/nvidia*`, toolkit env)  
3. Vulkan (`vulkaninfo`, ICDs, `/dev/dri`)  
4. CPU fallback  

Selected values are exported as `GGML_BACKEND`, `S2S_GPU_KIND`, `S2S_GPU_NAME` for sidecar backends.

### Build & run orchestrator only

Backends can run on the host (default compose URLs use `host.docker.internal`):

```bash
cd s2s-vulkan
docker compose build s2s
docker compose up s2s

# Probe what the container sees:
docker compose run --rm s2s gpu-probe
# or:
docker compose run --rm s2s --list-gpus
```

Open Realtime: `ws://localhost:8765/v1/realtime`

### AMD / Intel iGPU (Linux)

`/dev/dri` is mounted; set host GIDs if permission errors appear:

```bash
getent group video render
# VIDEO_GID=44 RENDER_GID=109 docker compose up s2s
```

### NVIDIA

```bash
# host: install nvidia-container-toolkit, then:
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.nvidia.yml \
  up --build s2s
```

### CPU-only smoke

```bash
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.cpu.yml \
  up --build s2s
```

### Prepare managed containers

```bash
# Pre-create every fixed, labelled backend but start only one ASR and one TTS.
S2S_LAB_BUILD=1 scripts/prepare_lab.sh base
```

| Variable | Meaning |
| -------- | ------- |
| `S2S_GPU=auto\|vulkan\|cuda\|sycl\|cpu` | Preference (default `auto`) |
| `GGML_BACKEND=Vulkan0` | Pin exact GGML backend string |
| `S2S_WHISPER_URL` / `S2S_LLM_URL` / `S2S_TTS_URL` | Backend endpoints |
| `PARAKEET_TORCH_VERSION` | PyTorch version for optional Parakeet images (default `2.11.0`) |
| `S2S_MODE` | Native runs support `lab`, `realtime`, `websocket`, `local`; Compose is fixed to `lab` |
| `S2S_DEBUG_GPU=1` | Print `vulkaninfo --summary` in entrypoint |

Inside the app (host or container):

```bash
s2s-vulkan --list-gpus
s2s-vulkan --gpu auto --mode realtime --host 0.0.0.0
```

## License

Apache-2.0 (same spirit as upstream speech-to-speech). Component models keep their own licenses.
