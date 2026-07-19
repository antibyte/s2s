# syntax=docker/dockerfile:1.7
#
# s2s-vulkan orchestrator — GHCR-ready image with first-boot model download.
#
#   ghcr.io/<owner>/s2s-vulkan:latest

# icu_* / idna (via reqwest/url) need rustc >= 1.86; keep a recent stable.
ARG RUST_VERSION=1.90
ARG APP_VERSION=0.1.0

# ── build ────────────────────────────────────────────────────────────
FROM rust:${RUST_VERSION}-bookworm AS builder
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency builds with a stub main (invalidated only when lock/toml change).
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && printf 'fn main() {}\n' > src/main.rs \
    && cargo build --release \
    && rm -rf src \
    && rm -f target/release/s2s-vulkan target/release/s2s-vulkan.d \
    && rm -rf target/release/deps/s2s_vulkan-* target/release/.fingerprint/s2s-vulkan-*

COPY src ./src
RUN cargo build --release \
    && strip target/release/s2s-vulkan

# ── runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

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
      org.opencontainers.image.base.name="docker.io/library/debian:bookworm-slim"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libasound2 \
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools \
        pciutils \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 -s /usr/sbin/nologin s2s \
    && mkdir -p /app /models \
    && chown -R s2s:s2s /app /models

COPY --from=builder /src/target/release/s2s-vulkan /usr/local/bin/s2s-vulkan
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/download-models.sh /usr/local/bin/download-models.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
             /usr/local/bin/download-models.sh \
             /usr/local/bin/s2s-vulkan

# Entrypoint runs as root for volume chown + model download, then drops to s2s.
USER root

ENV S2S_DOCKER=1 \
    RUST_LOG=info \
    S2S_GPU=auto \
    S2S_AUTO_BIND=true \
    S2S_MODELS_DIR=/models \
    S2S_DOWNLOAD_MODELS=auto \
    S2S_DOWNLOAD_WHISPER=true \
    S2S_DOWNLOAD_LLM=true \
    S2S_DOWNLOAD_TTS=false \
    S2S_WHISPER_PRESET=small \
    S2S_WHISPER_HF_REPO=ggerganov/whisper.cpp \
    S2S_LLM_HF_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    S2S_LLM_HF_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf \
    S2S_WHISPER_URL=http://whisper:8082 \
    S2S_LLM_URL=http://llama:8081/v1 \
    S2S_TTS_URL=http://tts:8083/v1/audio/speech \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    GGML_BACKEND=

VOLUME ["/models"]
WORKDIR /app
EXPOSE 8765

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["--mode", "realtime", "--host", "0.0.0.0", "--port", "8765", "--gpu", "auto"]
