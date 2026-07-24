# syntax=docker/dockerfile:1.7
#
# s2s-vulkan orchestrator — Ubuntu 24.04 for glibc ≥ 2.38 (ort/ONNX Runtime).

ARG RUST_VERSION=1.90.0
ARG APP_VERSION=0.1.0

# ── build ────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder
ARG RUST_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential pkg-config \
        libasound2-dev libssl-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal \
    && rustc --version \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && printf 'fn main() {}\n' > src/main.rs \
    && cargo build --release \
    && rm -rf src \
    && rm -f target/release/s2s-vulkan target/release/s2s-vulkan.d \
    && rm -rf target/release/deps/s2s_vulkan-* target/release/.fingerprint/s2s-vulkan-*

COPY src ./src
COPY config ./config
RUN cargo build --release \
    && strip target/release/s2s-vulkan

# ── bundled Supertonic model ────────────────────────────────────────
FROM python:3.12-slim-bookworm AS supertonic-models

RUN python - <<'PY'
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

revision = "3cadd1ee6394adea1bd021217a0e650ede09a323"
files = [
    ("onnx/duration_predictor.onnx", 3700147, "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db"),
    ("onnx/text_encoder.onnx", 36416150, "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff"),
    ("onnx/tts.json", 8253, ""),
    ("onnx/unicode_indexer.json", 277676, ""),
    ("onnx/vector_estimator.onnx", 256534781, "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c"),
    ("onnx/vocoder.onnx", 101424195, "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba"),
    ("voice_styles/F1.json", 292046, ""),
    ("voice_styles/F2.json", 292423, ""),
    ("voice_styles/F3.json", 290794, ""),
    ("voice_styles/F4.json", 291808, ""),
    ("voice_styles/F5.json", 291479, ""),
    ("voice_styles/M1.json", 291748, ""),
    ("voice_styles/M2.json", 292055, ""),
    ("voice_styles/M3.json", 290198, ""),
    ("voice_styles/M4.json", 291522, ""),
    ("voice_styles/M5.json", 291469, ""),
]
root = Path("/opt/s2s-models/supertonic")
for name, expected_size, expected_sha in files:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://huggingface.co/Supertone/supertonic-3/resolve/"
        f"{revision}/{quote(name)}?download=true"
    )
    digest = sha256()
    written = 0
    with urlopen(url, timeout=120) as source, target.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    if written != expected_size:
        raise SystemExit(f"{name}: expected {expected_size} bytes, got {written}")
    if expected_sha and digest.hexdigest() != expected_sha:
        raise SystemExit(f"{name}: SHA-256 mismatch")
PY

# ── runtime ──────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ARG APP_VERSION
ARG REVISION=unknown
ARG SOURCE_URL=https://github.com/unknown/s2s-vulkan

LABEL org.opencontainers.image.title="s2s-vulkan" \
      org.opencontainers.image.description="Speech-to-speech orchestrator (VAD→STT→LLM→TTS) with GPU auto-detect and first-boot model download" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="s2s-vulkan" \
      org.opencontainers.image.base.name="docker.io/library/ubuntu:24.04"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libasound2t64 \
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools \
        pciutils \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 -s /usr/sbin/nologin s2s \
    && mkdir -p /app /models /data \
    && chown -R s2s:s2s /app /models /data

COPY --from=builder /src/target/release/s2s-vulkan /usr/local/bin/s2s-vulkan
COPY --from=supertonic-models --chown=s2s:s2s /opt/s2s-models/supertonic /opt/s2s-models/supertonic
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/download-models.sh /usr/local/bin/download-models.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
             /usr/local/bin/download-models.sh \
             /usr/local/bin/s2s-vulkan

USER root

ENV S2S_DOCKER=1 \
    RUST_LOG=info \
    S2S_GPU=auto \
    S2S_AUTO_BIND=true \
    S2S_MODELS_DIR=/models \
    S2S_DATA_DIR=/data \
    S2S_DOWNLOAD_MODELS=false \
    S2S_DOWNLOAD_WHISPER=false \
    S2S_DOWNLOAD_LLM=false \
    S2S_DOWNLOAD_TTS=false \
    S2S_WHISPER_PRESET=tiny \
    S2S_WHISPER_HF_REPO=ggerganov/whisper.cpp \
    S2S_LLM_HF_REPO=Edge-Quant/granite-3.3-2b-instruct-Q4_K_M-GGUF \
    S2S_LLM_HF_FILE=granite-3.3-2b-instruct-q4_k_m.gguf \
    S2S_WHISPER_URL=http://whisper-tiny:8082 \
    S2S_LLM_URL=http://llama-fallback:8081/v1 \
    S2S_TTS_URL=http://supertonic:8083/v1/audio/speech \
    S2S_LANGUAGE=de \
    S2S_TTS_LANGUAGE=de \
    S2S_TTS=http \
    S2S_SUPERTONIC_VOICE=M1 \
    S2S_SUPERTONIC_MODEL_DIR=/opt/s2s-models/supertonic/onnx \
    GGML_BACKEND=

VOLUME ["/models", "/data"]
WORKDIR /app
EXPOSE 8765

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["--mode", "lab", "--host", "0.0.0.0", "--port", "8765", "--gpu", "auto"]
