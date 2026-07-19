#!/usr/bin/env bash
# Download STT / LLM / optional TTS model files into S2S_MODELS_DIR.
# Safe to re-run: skips existing files unless S2S_DOWNLOAD_FORCE=1.
#
# Configuration (all optional — sensible defaults for first boot):
#
#   S2S_MODELS_DIR          default: /models
#   S2S_DOWNLOAD_MODELS     true|false|auto  (auto = download missing; default auto)
#   S2S_DOWNLOAD_FORCE      1 to re-download
#   S2S_DOWNLOAD_WHISPER    true|false  (default true)
#   S2S_DOWNLOAD_LLM        true|false  (default true)
#   S2S_DOWNLOAD_TTS        true|false  (default false — large)
#   S2S_HF_TOKEN            Hugging Face token (gated repos)
#   HF_TOKEN                alias for S2S_HF_TOKEN
#
# Whisper:
#   S2S_WHISPER_PRESET      tiny|base|small|medium|large-v3|large-v3-turbo  (default small)
#   S2S_WHISPER_HF_REPO     default ggerganov/whisper.cpp
#   S2S_WHISPER_HF_FILE     overrides preset → exact filename on the repo
#   S2S_WHISPER_MODEL_URL   full URL override (skips HF path)
#   S2S_WHISPER_PATH        local destination filename (under MODELS_DIR)
#
# LLM:
#   S2S_LLM_HF_REPO         default Qwen/Qwen2.5-1.5B-Instruct-GGUF
#   S2S_LLM_HF_FILE         default qwen2.5-1.5b-instruct-q4_k_m.gguf
#   S2S_LLM_MODEL_URL       full URL override
#   S2S_LLM_PATH            local destination filename
#
# TTS (optional, HF snapshot-style single file or URL):
#   S2S_TTS_HF_REPO
#   S2S_TTS_HF_FILE
#   S2S_TTS_MODEL_URL
#   S2S_TTS_PATH
#
# Extra arbitrary downloads (comma-separated "url=>relpath" or "url|relpath"):
#   S2S_DOWNLOAD_EXTRA="https://example/a.bin=>extra/a.bin,https://example/b.gguf|b.gguf"

set -euo pipefail

log()  { echo "[models] $*" >&2; }
warn() { echo "[models] WARN: $*" >&2; }
die()  { echo "[models] ERROR: $*" >&2; exit 1; }

MODELS_DIR="${S2S_MODELS_DIR:-/models}"
FORCE="${S2S_DOWNLOAD_FORCE:-0}"
MODE="${S2S_DOWNLOAD_MODELS:-auto}"
HF_TOKEN="${S2S_HF_TOKEN:-${HF_TOKEN:-}}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

should_download_category() {
  # $1 = per-category flag (true/false/empty), default true for whisper/llm
  local flag="${1:-true}"
  if is_true "$flag"; then return 0; fi
  return 1
}

mkdir -p "$MODELS_DIR"

if [[ "$MODE" == "false" ]] || [[ "$MODE" == "0" ]] || [[ "$MODE" == "no" ]]; then
  log "S2S_DOWNLOAD_MODELS=$MODE — skipping all downloads"
  return 0 2>/dev/null || exit 0
fi

# ── helpers ──────────────────────────────────────────────────────────

hf_url() {
  local repo="$1" file="$2"
  # encode spaces in file path lightly
  echo "https://huggingface.co/${repo}/resolve/main/${file}"
}

need_file() {
  local dest="$1"
  if [[ -f "$dest" && -s "$dest" && "$FORCE" != "1" ]]; then
    return 1 # already present
  fi
  return 0
}

download() {
  local url="$1"
  local dest="$2"
  local label="${3:-$dest}"

  mkdir -p "$(dirname "$dest")"

  if ! need_file "$dest"; then
    log "OK (cached): $label"
    return 0
  fi

  local tmp="${dest}.partial"
  log "Downloading $label"
  log "  URL:  $url"
  log "  Dest: $dest"

  local auth=()
  if [[ -n "$HF_TOKEN" && "$url" == *huggingface.co* ]]; then
    auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi

  # Follow redirects; resume partial; fail on HTTP errors
  if ! curl -fL --retry 5 --retry-delay 2 --retry-all-errors \
      -C - \
      "${auth[@]}" \
      --connect-timeout 30 \
      --progress-bar \
      -o "$tmp" \
      "$url"; then
    rm -f "$tmp"
    die "download failed: $url"
  fi

  if [[ ! -s "$tmp" ]]; then
    rm -f "$tmp"
    die "download empty: $url"
  fi

  mv -f "$tmp" "$dest"
  log "Done: $label ($(du -h "$dest" | awk '{print $1}'))"
}

whisper_file_for_preset() {
  case "${1:-small}" in
    tiny)             echo "ggml-tiny.bin" ;;
    tiny.en)          echo "ggml-tiny.en.bin" ;;
    base)             echo "ggml-base.bin" ;;
    base.en)          echo "ggml-base.en.bin" ;;
    small)            echo "ggml-small.bin" ;;
    small.en)         echo "ggml-small.en.bin" ;;
    medium)           echo "ggml-medium.bin" ;;
    medium.en)        echo "ggml-medium.en.bin" ;;
    large|large-v3)   echo "ggml-large-v3.bin" ;;
    large-v3-turbo|turbo) echo "ggml-large-v3-turbo.bin" ;;
    *)
      # allow raw filename as "preset"
      echo "$1"
      ;;
  esac
}

# ── Whisper ──────────────────────────────────────────────────────────

if should_download_category "${S2S_DOWNLOAD_WHISPER:-true}"; then
  W_PRESET="${S2S_WHISPER_PRESET:-small}"
  W_REPO="${S2S_WHISPER_HF_REPO:-ggerganov/whisper.cpp}"
  W_FILE="${S2S_WHISPER_HF_FILE:-$(whisper_file_for_preset "$W_PRESET")}"
  W_PATH="${S2S_WHISPER_PATH:-$W_FILE}"
  W_DEST="${MODELS_DIR%/}/whisper/${W_PATH##*/}"
  # keep flat optional: if path has no slash intent, still under whisper/
  if [[ "$W_PATH" == */* ]]; then
    W_DEST="${MODELS_DIR%/}/${W_PATH}"
  fi

  if [[ -n "${S2S_WHISPER_MODEL_URL:-}" ]]; then
    download "$S2S_WHISPER_MODEL_URL" "$W_DEST" "whisper/$W_PATH"
  else
    download "$(hf_url "$W_REPO" "$W_FILE")" "$W_DEST" "whisper/$W_FILE"
  fi
  # Export resolved path for sibling entrypoints
  export S2S_WHISPER_MODEL_PATH="$W_DEST"
else
  log "Whisper download disabled (S2S_DOWNLOAD_WHISPER)"
fi

# ── LLM ──────────────────────────────────────────────────────────────

if should_download_category "${S2S_DOWNLOAD_LLM:-true}"; then
  L_REPO="${S2S_LLM_HF_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
  L_FILE="${S2S_LLM_HF_FILE:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"
  L_PATH="${S2S_LLM_PATH:-$L_FILE}"
  L_DEST="${MODELS_DIR%/}/llm/${L_PATH##*/}"
  if [[ "$L_PATH" == */* ]]; then
    L_DEST="${MODELS_DIR%/}/${L_PATH}"
  fi

  if [[ -n "${S2S_LLM_MODEL_URL:-}" ]]; then
    download "$S2S_LLM_MODEL_URL" "$L_DEST" "llm/$L_PATH"
  else
    download "$(hf_url "$L_REPO" "$L_FILE")" "$L_DEST" "llm/$L_FILE"
  fi
  export S2S_LLM_MODEL_PATH="$L_DEST"
else
  log "LLM download disabled (S2S_DOWNLOAD_LLM)"
fi

# ── TTS (optional) ───────────────────────────────────────────────────

if should_download_category "${S2S_DOWNLOAD_TTS:-false}"; then
  if [[ -n "${S2S_TTS_MODEL_URL:-}" || -n "${S2S_TTS_HF_REPO:-}" ]]; then
    T_PATH="${S2S_TTS_PATH:-${S2S_TTS_HF_FILE:-tts-model}}"
    T_DEST="${MODELS_DIR%/}/tts/${T_PATH##*/}"
    if [[ "$T_PATH" == */* ]]; then
      T_DEST="${MODELS_DIR%/}/${T_PATH}"
    fi
    if [[ -n "${S2S_TTS_MODEL_URL:-}" ]]; then
      download "$S2S_TTS_MODEL_URL" "$T_DEST" "tts/$T_PATH"
    else
      download "$(hf_url "${S2S_TTS_HF_REPO}" "${S2S_TTS_HF_FILE}")" "$T_DEST" "tts/${S2S_TTS_HF_FILE}"
    fi
    export S2S_TTS_MODEL_PATH="$T_DEST"
  else
    warn "S2S_DOWNLOAD_TTS=true but no S2S_TTS_MODEL_URL / S2S_TTS_HF_REPO set — skip"
  fi
fi

# ── Extra ────────────────────────────────────────────────────────────

if [[ -n "${S2S_DOWNLOAD_EXTRA:-}" ]]; then
  IFS=',' read -ra EXTRAS <<< "$S2S_DOWNLOAD_EXTRA"
  for item in "${EXTRAS[@]}"; do
    item="$(echo "$item" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$item" ]] && continue
    local_url=""
    local_rel=""
    if [[ "$item" == *"=>"* ]]; then
      local_url="${item%%=>*}"
      local_rel="${item#*=>}"
    elif [[ "$item" == *"|"* ]]; then
      local_url="${item%%|*}"
      local_rel="${item#*|}"
    else
      # URL only → basename under extras/
      local_url="$item"
      local_rel="extras/$(basename "${item%%\?*}")"
    fi
    local_url="$(echo "$local_url" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    local_rel="$(echo "$local_rel" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    download "$local_url" "${MODELS_DIR%/}/${local_rel}" "extra/$local_rel"
  done
fi

# Marker for operators / health
{
  echo "downloaded_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "models_dir=$MODELS_DIR"
  [[ -n "${S2S_WHISPER_MODEL_PATH:-}" ]] && echo "whisper=$S2S_WHISPER_MODEL_PATH"
  [[ -n "${S2S_LLM_MODEL_PATH:-}" ]] && echo "llm=$S2S_LLM_MODEL_PATH"
  [[ -n "${S2S_TTS_MODEL_PATH:-}" ]] && echo "tts=$S2S_TTS_MODEL_PATH"
} > "${MODELS_DIR%/}/.s2s-models-ready"

log "Model directory ready: $MODELS_DIR"
ls -lah "$MODELS_DIR" 2>/dev/null | sed 's/^/[models]   /' >&2 || true
