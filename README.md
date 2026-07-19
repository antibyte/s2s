# s2s — Speech-to-Speech (Vulkan)

Dieses Repo enthält **`s2s-vulkan`**: eine Rust-Implementierung der Pipeline aus
[huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech),
mit Vulkan über GGML-Server statt PyTorch-`--device vulkan`.

```text
VAD (CPU) → whisper.cpp Vulkan → llama.cpp Vulkan → Qwen3-TTS / Piper / System
```

## Einstieg

```powershell
cd s2s-vulkan
cargo build --release
.\target\release\s2s-vulkan.exe --help
```

Ausführliche Vulkan-Build-Anleitung, Architektur und CLI: **[s2s-vulkan/README.md](s2s-vulkan/README.md)**.

## Minimaler Smoke-Test (ohne Modelle)

```powershell
cd s2s-vulkan
cargo run -- --mode local --tts system --skip_health
cargo run -- --list-gpus   # GPU auto-detect (Vulkan/CUDA/CPU)
```

## Docker + GHCR

Image (nach Push auf `main` / Tag `v*`): `ghcr.io/<owner>/s2s-vulkan`

```bash
cd s2s-vulkan
cp .env.example .env   # GHCR_OWNER setzen
docker compose pull
docker compose up -d   # model-init lädt Whisper+LLM, dann startet s2s
```

Modelle per Env (Defaults: Whisper `small`, Qwen2.5-1.5B Q4_K_M):

```bash
S2S_WHISPER_PRESET=base \
S2S_LLM_HF_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF \
S2S_LLM_HF_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf \
docker compose up -d
```

- **AMD/Intel:** `/dev/dri`
- **NVIDIA:** `docker compose -f docker-compose.yml -f docker/docker-compose.nvidia.yml up`
- **CPU:** `docker compose -f docker-compose.yml -f docker/docker-compose.cpu.yml up`

Details: [s2s-vulkan/README.md](s2s-vulkan/README.md#github-container-registry-ghcr).

Ohne laufende `whisper-server` / `llama-server` schlagen STT/LLM fehl — das ist erwartet. Mit Backends:

```powershell
# Terminals: whisper-server :8082, llama-server :8081, optional tts_qwen_server :8083
cargo run --release -- --mode local --tts system
```
