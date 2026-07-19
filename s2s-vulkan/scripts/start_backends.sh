#!/usr/bin/env bash
# Start Vulkan-capable GGML backends for s2s-vulkan (Linux).
set -euo pipefail

WHISPER_BIN="${WHISPER_BIN:-whisper-server}"
WHISPER_MODEL="${WHISPER_MODEL:-models/ggml-small.bin}"
WHISPER_PORT="${WHISPER_PORT:-8082}"

LLAMA_BIN="${LLAMA_BIN:-llama-server}"
LLAMA_MODEL="${LLAMA_MODEL:-models/model.gguf}"
LLAMA_PORT="${LLAMA_PORT:-8081}"

export GGML_BACKEND="${GGML_BACKEND:-Vulkan0}"
echo "GGML_BACKEND=$GGML_BACKEND"

if command -v "$WHISPER_BIN" >/dev/null 2>&1; then
  echo "Starting whisper-server on :$WHISPER_PORT ..."
  "$WHISPER_BIN" \
    -m "$WHISPER_MODEL" \
    --host 127.0.0.1 \
    --port "$WHISPER_PORT" \
    --language auto \
    --no-timestamps &
else
  echo "WARN: whisper-server not found. Build whisper.cpp with -DGGML_VULKAN=1" >&2
fi

if command -v "$LLAMA_BIN" >/dev/null 2>&1; then
  echo "Starting llama-server on :$LLAMA_PORT ..."
  "$LLAMA_BIN" \
    -m "$LLAMA_MODEL" \
    -ngl 999 \
    -c 8192 \
    --host 127.0.0.1 \
    --port "$LLAMA_PORT" &
else
  echo "WARN: llama-server not found. Build llama.cpp with -DGGML_VULKAN=ON" >&2
fi

echo
echo "Then run:"
echo "  cargo run --release -- --mode local --tts system"
echo "  cargo run --release -- --mode local --tts http --tts_url http://127.0.0.1:8083/v1/audio/speech"
