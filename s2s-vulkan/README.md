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
| **TTS** | HTTP / Piper / system | [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) Vulkan via `scripts/tts_qwen_server.py` |

This is the same architecture conclusion as the Python fork analysis: ~90 % of FLOPs can sit on Vulkan without rewriting PyTorch kernels.

## Why not `--device vulkan`?

The original Python stack hard-wires PyTorch devices (`cuda` / `mps` / `cpu`) for Silero, Parakeet, and several TTS paths. Desktop Vulkan is not a first-class PyTorch device. Swapping the **STT / LLM / TTS processes** for GGML servers is the practical route.

## Quick start (Windows)

### 1. Build this app

```powershell
cd s2s-vulkan
cargo build --release
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
| `realtime` | Minimal OpenAI Realtime subset at `ws://host:port/v1/realtime` |

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
# UI:  http://localhost:8088
# WS:  same origin /ws  →  proxied to s2s:8765
```

Also included in `--profile full`.

Env: `WEB_PORT=8088`.

> Backend must run with `--mode websocket` for binary PCM. Compose default is `realtime`; for the lab set `S2S_MODE=websocket` (or change the `s2s` command).

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

### Run from GHCR with first-boot model download

```bash
export GHCR_OWNER=your-user   # lowercase
# or: export IMAGE=ghcr.io/your-user/s2s-vulkan TAG=latest

docker compose pull
docker compose up -d
# model-init downloads whisper + LLM into volume s2s-models, then s2s starts
```

Only re-download if files are missing (`S2S_DOWNLOAD_MODELS=auto`). Force:

```bash
S2S_DOWNLOAD_FORCE=1 docker compose run --rm model-init
```

### Model env reference

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `S2S_DOWNLOAD_MODELS` | `auto` | `auto`/`true` download missing; `false` skip |
| `S2S_DOWNLOAD_FORCE` | `0` | `1` re-download |
| `S2S_DOWNLOAD_WHISPER` | `true` | Fetch Whisper weights |
| `S2S_DOWNLOAD_LLM` | `true` | Fetch GGUF LLM |
| `S2S_DOWNLOAD_TTS` | `false` | Optional TTS weights |
| `S2S_WHISPER_PRESET` | `small` | `tiny`…`large-v3-turbo` |
| `S2S_WHISPER_HF_REPO` | `ggerganov/whisper.cpp` | HF repo |
| `S2S_WHISPER_HF_FILE` | (from preset) | Exact file on repo |
| `S2S_WHISPER_MODEL_URL` | — | Direct URL override |
| `S2S_LLM_HF_REPO` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | HF repo |
| `S2S_LLM_HF_FILE` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | File on repo |
| `S2S_LLM_MODEL_URL` | — | Direct URL override |
| `S2S_DOWNLOAD_EXTRA` | — | `url=>relpath,url2\|relpath2` |
| `S2S_HF_TOKEN` / `HF_TOKEN` | — | Gated HF models |
| `S2S_MODELS_DIR` | `/models` | Volume mount path |

Layout after download:

```text
/models/whisper/ggml-small.bin
/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
/models/.s2s-models-ready
```

Manual download only:

```bash
docker run --rm -v s2s-models:/models \
  -e S2S_WHISPER_PRESET=base \
  -e S2S_LLM_HF_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf \
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

### Full stack profiles

```bash
# whisper + llama (+ optional tts) with GPU devices
docker compose --profile full up --build
```

| Variable | Meaning |
| -------- | ------- |
| `S2S_GPU=auto\|vulkan\|cuda\|cpu` | Preference (default `auto`) |
| `GGML_BACKEND=Vulkan0` | Pin exact GGML backend string |
| `S2S_WHISPER_URL` / `S2S_LLM_URL` / `S2S_TTS_URL` | Backend endpoints |
| `S2S_MODE` | `realtime` (default in compose), `websocket`, `local` |
| `S2S_DEBUG_GPU=1` | Print `vulkaninfo --summary` in entrypoint |

Inside the app (host or container):

```bash
s2s-vulkan --list-gpus
s2s-vulkan --gpu auto --mode realtime --host 0.0.0.0
```

## License

Apache-2.0 (same spirit as upstream speech-to-speech). Component models keep their own licenses.
