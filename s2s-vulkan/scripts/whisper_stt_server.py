#!/usr/bin/env python3
"""
Lightweight Whisper STT server compatible with s2s-vulkan's /inference client.

Uses faster-whisper (CTranslate2). First run downloads the model.

  pip install faster-whisper soundfile numpy aiohttp
  python scripts/whisper_stt_server.py --host 127.0.0.1 --port 8082 --model base

Env:
  S2S_WHISPER_MODEL=base|small|tiny|...
  S2S_WHISPER_DEVICE=cpu|cuda|auto
  S2S_WHISPER_COMPUTE=int8|float16|default
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--model", default=os.environ.get("S2S_WHISPER_MODEL", "base"))
    p.add_argument("--device", default=os.environ.get("S2S_WHISPER_DEVICE", "cpu"))
    p.add_argument(
        "--compute-type",
        default=os.environ.get("S2S_WHISPER_COMPUTE", "int8"),
    )
    p.add_argument("--language", default=None, help="force language code, e.g. de")
    args = p.parse_args()

    def ensure(pkg: str, import_name: str | None = None) -> None:
        name = import_name or pkg
        try:
            __import__(name)
        except ImportError:
            print(f"Installing {pkg}…", file=sys.stderr)
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

    ensure("faster-whisper", "faster_whisper")
    ensure("aiohttp")
    ensure("numpy")
    ensure("soundfile")

    from aiohttp import web
    from faster_whisper import WhisperModel

    print(
        f"Loading Whisper model={args.model} device={args.device} compute={args.compute_type} …",
        file=sys.stderr,
    )
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("Whisper ready.", file=sys.stderr)

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "model": args.model})

    async def inference(request: web.Request) -> web.Response:
        reader = await request.multipart()
        audio_bytes: bytes | None = None
        language = args.language
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name in ("file", "audio", "wav"):
                audio_bytes = await part.read(decode=False)
            elif part.name == "language":
                language = (await part.text()).strip() or language

        if not audio_bytes:
            # raw body fallback
            body = await request.read()
            if body:
                audio_bytes = body
        if not audio_bytes:
            return web.json_response({"error": "no audio"}, status=400)

        # Write temp wav/file for faster-whisper
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            segments, info = model.transcribe(
                path,
                language=None if not language or language == "auto" else language,
                beam_size=1,
                vad_filter=False,
            )
            text = "".join(s.text for s in segments).strip()
            print(f"STT [{getattr(info, 'language', '?')}]: {text}", file=sys.stderr)
            return web.json_response({"text": text})
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    app = web.Application(client_max_size=32 * 1024 * 1024)
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_post("/inference", inference)
    # OpenAI-ish alias
    app.router.add_post("/v1/audio/transcriptions", inference)

    print(f"Listening http://{args.host}:{args.port}/inference", file=sys.stderr)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
