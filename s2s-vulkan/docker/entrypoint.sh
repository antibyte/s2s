#!/usr/bin/env bash
# Container entrypoint:
#   1) GPU pre-select → GGML_BACKEND
#   2) First-start model download (env-configurable)
#   3) Drop privileges → s2s-vulkan
set -euo pipefail

log() { echo "[entrypoint] $*" >&2; }

export S2S_DOCKER=1
export S2S_MODELS_DIR="${S2S_MODELS_DIR:-/models}"

exec_as_app() {
  if [[ "$(id -u)" -eq 0 ]]; then
    if command -v runuser >/dev/null 2>&1; then
      exec runuser -u s2s -- "$@"
    elif command -v gosu >/dev/null 2>&1; then
      exec gosu s2s "$@"
    else
      exec su -s /bin/bash s2s -c 'exec "$@"' -- "$@"
    fi
  else
    exec "$@"
  fi
}

# ── Device node hints ────────────────────────────────────────────────
if [[ -d /dev/dri ]]; then
  log "DRI devices: $(ls /dev/dri 2>/dev/null | tr '\n' ' ')"
else
  log "No /dev/dri (pass devices: or use NVIDIA toolkit for dGPU)"
fi

if [[ -e /dev/nvidia0 || -e /dev/nvidiactl ]]; then
  log "NVIDIA device nodes present"
fi

# ── GPU pre-seed (Rust probe may refine) ─────────────────────────────
if [[ -z "${GGML_BACKEND:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    if ls /usr/share/vulkan/icd.d/*nvidia* >/dev/null 2>&1 \
       || ls /etc/vulkan/icd.d/*nvidia* >/dev/null 2>&1 \
       || [[ -n "${VK_ICD_FILENAMES:-}" ]]; then
      export GGML_BACKEND="Vulkan0"
      log "Pre-select Vulkan0 (NVIDIA + Vulkan ICD)"
    else
      export GGML_BACKEND="CUDA0"
      log "Pre-select CUDA0 (nvidia-smi OK, no NVIDIA Vulkan ICD in image)"
    fi
  elif [[ -d /dev/dri ]] && ls /dev/dri/renderD* >/dev/null 2>&1; then
    export GGML_BACKEND="Vulkan0"
    log "Pre-select Vulkan0 (/dev/dri render node)"
  else
    export GGML_BACKEND="CPU"
    log "Pre-select CPU (no GPU nodes detected)"
  fi
else
  log "GGML_BACKEND already set to ${GGML_BACKEND}"
fi

if [[ "${S2S_DEBUG_GPU:-}" == "1" ]] && command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary 2>/dev/null | head -n 80 || true
fi

download_models() {
  if [[ -x /usr/local/bin/download-models.sh ]]; then
    /usr/local/bin/download-models.sh
  else
    log "download-models.sh missing — skip"
  fi
}

ensure_models_dir() {
  mkdir -p "$S2S_MODELS_DIR"
  if [[ "$(id -u)" -eq 0 ]]; then
    chown -R s2s:s2s "$S2S_MODELS_DIR" 2>/dev/null || true
  fi
}

case "${1:-}" in
  gpu-probe)
    shift || true
    exec_as_app s2s-vulkan --list-gpus --gpu auto "$@"
    ;;
  download-models|models)
    ensure_models_dir
    download_models
    exit 0
    ;;
  bash|sh)
    exec /bin/bash
    ;;
esac

ensure_models_dir
download_models

if [[ -f "${S2S_MODELS_DIR}/.s2s-models-ready" ]]; then
  while IFS='=' read -r k v; do
    case "$k" in
      whisper) export S2S_WHISPER_MODEL_PATH="$v" ;;
      llm)     export S2S_LLM_MODEL_PATH="$v" ;;
      tts)     export S2S_TTS_MODEL_PATH="$v" ;;
    esac
  done < <(grep -E '^(whisper|llm|tts)=' "${S2S_MODELS_DIR}/.s2s-models-ready" || true)
fi

export S2S_WHISPER_URL="${S2S_WHISPER_URL:-http://whisper:8082}"
export S2S_LLM_URL="${S2S_LLM_URL:-http://llama:8081/v1}"
export S2S_TTS_URL="${S2S_TTS_URL:-http://tts:8083/v1/audio/speech}"
export WHISPER_MODEL_PATH="${S2S_WHISPER_MODEL_PATH:-${WHISPER_MODEL_PATH:-}}"
export LLAMA_MODEL_PATH="${S2S_LLM_MODEL_PATH:-${LLAMA_MODEL_PATH:-}}"

log "Starting: s2s-vulkan $*"
if [[ $# -eq 0 ]]; then
  set -- --mode realtime --host 0.0.0.0 --port 8765 --gpu auto
fi
exec_as_app s2s-vulkan "$@"
