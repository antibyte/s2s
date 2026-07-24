#!/usr/bin/env bash
# Serve bosonai/higgs-tts-3-4b with SGLang-Omni (OpenAI /v1/audio/speech).
set -euo pipefail

MODEL_PATH="${HIGGS_MODEL_PATH:-/models/higgs-tts-3-4b}"
HOST="${HIGGS_HOST:-0.0.0.0}"
PORT="${HIGGS_PORT:-8086}"
HF_ID="${HIGGS_HF_ID:-bosonai/higgs-tts-3-4b}"

log() { echo "[higgs-tts] $*" >&2; }

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  log "ERROR: model config missing at ${MODEL_PATH}/config.json"
  log "Download weights via the Lab UI (backend id higgs-tts-3-4b) or:"
  log "  hf download ${HF_ID} --local-dir ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/model.safetensors" ]]; then
  log "ERROR: weights missing at ${MODEL_PATH}/model.safetensors (~9.3 GB)"
  exit 1
fi

SERVE_BIN=""
if command -v sgl-omni >/dev/null 2>&1; then
  SERVE_BIN="sgl-omni"
elif command -v sglang-omni >/dev/null 2>&1; then
  SERVE_BIN="sglang-omni"
elif python -c "import sglang_omni" >/dev/null 2>&1; then
  SERVE_BIN="python -m sglang_omni"
elif command -v vllm-omni >/dev/null 2>&1; then
  log "sgl-omni not found; falling back to vllm-omni"
  exec vllm-omni serve "${MODEL_PATH}" \
    --host "${HOST}" --port "${PORT}" \
    --trust-remote-code --omni
else
  log "ERROR: neither sgl-omni nor vllm-omni is available in this image"
  log "Base image should be lmsysorg/sglang-omni:dev (or install sglang-omni)"
  exit 1
fi

log "Starting ${SERVE_BIN} model=${MODEL_PATH} on ${HOST}:${PORT}"
# shellcheck disable=SC2086
exec ${SERVE_BIN} serve \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}"
