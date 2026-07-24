#!/usr/bin/env bash
set -euo pipefail

platform="${1:-base}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

compose=(docker compose -f docker-compose.yml)
case "$platform" in
  base) ;;
  linux-gpu)
    compose+=(-f docker/docker-compose.linux-gpu.yml)
    ;;
  nvidia)
    compose+=(-f docker/docker-compose.nvidia.yml)
    ;;
  intel-sycl)
    compose+=(-f docker/docker-compose.intel-sycl.yml)
    ;;
  windows)
    compose+=(-f docker/docker-compose.windows.yml)
    ;;
  *)
    echo "usage: $0 [base|linux-gpu|nvidia|intel-sycl|windows]" >&2
    exit 2
    ;;
esac
compose+=(--profile backends --profile managed --profile fallback --profile web)
if [[ "$platform" == "intel-sycl" ]]; then
  compose+=(--profile tts)
fi

if [[ "${S2S_LAB_BUILD:-0}" == "1" ]]; then
  "${compose[@]}" build
fi

# Pre-create the fixed, labelled registry containers. The controller may only
# start/stop these objects and cannot accept image, mount or device data from UI.
"${compose[@]}" create

optional_containers=(
  s2s-whisper-base
  s2s-whisper-small
  s2s-parakeet-cpu
  s2s-parakeet-cuda
  s2s-parakeet-xpu
  s2s-tts-qwen-vulkan
  s2s-tts-qwen-sycl-aot
  s2s-tts-qwen-sycl-jit
  s2s-tts-kokoro
)
for container in "${optional_containers[@]}"; do
  if docker container inspect "$container" >/dev/null 2>&1; then
    docker stop --time 10 "$container" >/dev/null 2>&1 || true
  fi
done

"${compose[@]}" up -d docker-proxy model-init supertonic whisper-tiny llama s2s web
echo "Speech Lab ready at http://127.0.0.1:${WEB_PORT:-8088} (ASR=fw-tiny, LLM=Granite 3.3, TTS=supertonic)"
