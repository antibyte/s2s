#!/usr/bin/env bash
# Serve VibeVoice Realtime 0.5B GGUF via CrispASR OpenAI /v1/audio/speech.
set -euo pipefail

MODEL="${VIBEVOICE_MODEL:-/models/vibevoice/vibevoice-realtime-0.5b-q4_k.gguf}"
VOICE="${VIBEVOICE_VOICE:-/models/vibevoice/vibevoice-voice-emma.gguf}"
VOICE_DIR="${VIBEVOICE_VOICE_DIR:-/models/vibevoice}"
HOST="${VIBEVOICE_HOST:-0.0.0.0}"
PORT="${VIBEVOICE_PORT:-8089}"
BACKEND="${VIBEVOICE_BACKEND:-vibevoice-tts}"

log() { echo "[vibevoice-tts] $*" >&2; }

if [[ ! -f "$MODEL" ]]; then
  log "ERROR: model missing at $MODEL"
  log "Lab download id: vibevoice-realtime-0.5b"
  log "  hf download cstr/vibevoice-realtime-0.5b-GGUF --local-dir /models/vibevoice"
  exit 1
fi

if [[ ! -f "$VOICE" ]]; then
  log "WARN: default voice missing at $VOICE — requests must pass voice="
fi

# CrispASR resolves voice="emma" as ${VOICE_DIR}/emma.gguf (or .wav). Lab
# artifacts are named vibevoice-voice-<id>.gguf — publish stable short names.
if [[ -d "$VOICE_DIR" ]]; then
  shopt -s nullglob
  for voice_file in "$VOICE_DIR"/vibevoice-voice-*.gguf; do
    base="$(basename "$voice_file" .gguf)"
    short="${base#vibevoice-voice-}"
    link="$VOICE_DIR/${short}.gguf"
    if [[ ! -e "$link" ]]; then
      ln -sfn "$voice_file" "$link" || cp -n "$voice_file" "$link" || true
      log "voice alias: $short → $(basename "$voice_file")"
    fi
  done
  shopt -u nullglob
fi

# Resolve crispasr binary (prebuilt image or local build).
if command -v crispasr >/dev/null 2>&1; then
  BIN=crispasr
elif command -v crispasr-server >/dev/null 2>&1; then
  BIN=crispasr-server
elif [[ -x /app/crispasr ]]; then
  BIN=/app/crispasr
elif [[ -x /usr/local/bin/crispasr ]]; then
  BIN=/usr/local/bin/crispasr
else
  log "ERROR: crispasr binary not found in image"
  exit 1
fi

ARGS=(
  --server
  --backend "$BACKEND"
  -m "$MODEL"
  --host "$HOST"
  --port "$PORT"
)

if [[ -f "$VOICE" ]]; then
  ARGS+=(--voice "$VOICE")
fi
if [[ -d "$VOICE_DIR" ]]; then
  ARGS+=(--voice-dir "$VOICE_DIR")
fi

# Lab pipeline already discloses AI speech; optional watermark slows short turns.
# CrispASR 0.8+ refuses --no-watermark without an explicit marking acceptance.
if [[ "${VIBEVOICE_NO_WATERMARK:-1}" == "1" ]]; then
  ARGS+=(--no-watermark --accept-marking-responsibility)
fi

log "Starting $BIN backend=$BACKEND model=$(basename "$MODEL") on ${HOST}:${PORT}"
exec "$BIN" "${ARGS[@]}"
