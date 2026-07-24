#!/usr/bin/env python3
"""
OpenAI-compatible HTTP TTS for Kokoro-82M.

Default: http://127.0.0.1:8084/v1/audio/speech

Backends (auto-detect):
  1) kokoro (hexgrad) + misaki  — multi-lang pipeline
  2) kokoro-onnx                — lightweight ONNX (EN-first)

Install (pick one):
  pip install kokoro soundfile numpy fastapi uvicorn 'misaki[en]'
  # optional DE G2P extras if available for your misaki version
  #   or:
  pip install kokoro-onnx soundfile numpy fastapi uvicorn

Models (kokoro-onnx):
  auto-download into models/kokoro/  (or set --onnx / --voices)

s2s-vulkan:
  --tts http --tts-url http://127.0.0.1:8084/v1/audio/speech
  Lab hot-swap id: kokoro
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Kokoro native sample rate
KOKORO_SR = 24000

# HuggingFace assets for kokoro-onnx v1.0
ONNX_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)

# lang_code for hexgrad kokoro KPipeline
LANG_MAP = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "american": "a",
    "british": "b",
    "es": "e",
    "spanish": "e",
    "fr": "f",
    "french": "f",
    "hi": "h",
    "hindi": "h",
    "it": "i",
    "italian": "i",
    "ja": "j",
    "jp": "j",
    "japanese": "j",
    "pt": "p",
    "pt-br": "p",
    "portuguese": "p",
    "zh": "z",
    "zh-cn": "z",
    "chinese": "z",
    # German is not a first-class Kokoro lang_code; fall back to American EN G2P.
    "de": "a",
    "german": "a",
    "deutsch": "a",
    "auto": "a",
}

DEFAULT_VOICE = {
    "a": "af_bella",
    "b": "bf_emma",
    "e": "ef_dora",
    "f": "ff_siwis",
    "h": "hf_alpha",
    "i": "if_sara",
    "j": "jf_alpha",
    "p": "pf_dora",
    "z": "zf_xiaobei",
}


class SpeechRequest(BaseModel):
    model: Optional[str] = "kokoro"
    input: Optional[str] = None
    text: Optional[str] = None
    voice: Optional[str] = None
    language: Optional[str] = "auto"
    lang: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return
    print(f"Downloading {url} → {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _resolve_lang(language: Optional[str], lang: Optional[str]) -> str:
    raw = (lang or language or "auto").strip().lower()
    if raw in LANG_MAP:
        return LANG_MAP[raw]
    # full names already handled; try first 2 letters
    if len(raw) >= 2 and raw[:2] in LANG_MAP:
        return LANG_MAP[raw[:2]]
    return "a"


def _is_supertonic_or_qwen_voice(voice: str) -> bool:
    v = voice.strip()
    if v.upper() in {
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
    }:
        return True
    # Qwen custom-voice speakers
    if v.lower() in {
        "aiden",
        "dylan",
        "eric",
        "onyx",
        "ryan",
        "serena",
        "uncle_fu",
        "vivian",
    }:
        return True
    return False


class Engine:
    def synthesize(
        self, text: str, voice: str, lang_code: str, speed: float
    ) -> tuple[np.ndarray, int]:
        raise NotImplementedError


class KokoroOfficial(Engine):
    """hexgrad/kokoro KPipeline backend."""

    def __init__(self) -> None:
        from kokoro import KPipeline  # type: ignore

        self._KPipeline = KPipeline
        self._pipelines: dict[str, Any] = {}
        # Warm default American English
        self._pipelines["a"] = KPipeline(lang_code="a")
        print("Kokoro backend: hexgrad/kokoro (KPipeline)", flush=True)

    def _pipe(self, lang_code: str):
        if lang_code not in self._pipelines:
            print(f"Loading KPipeline lang_code={lang_code}", flush=True)
            self._pipelines[lang_code] = self._KPipeline(lang_code=lang_code)
        return self._pipelines[lang_code]

    def synthesize(
        self, text: str, voice: str, lang_code: str, speed: float
    ) -> tuple[np.ndarray, int]:
        pipe = self._pipe(lang_code)
        chunks: list[np.ndarray] = []
        for _gs, _ps, audio in pipe(text, voice=voice, speed=speed):
            if audio is None:
                continue
            a = np.asarray(audio, dtype=np.float32).reshape(-1)
            if a.size:
                chunks.append(a)
        if not chunks:
            raise RuntimeError("Kokoro produced empty audio")
        return np.concatenate(chunks), KOKORO_SR


class KokoroOnnx(Engine):
    """kokoro-onnx backend (EN-oriented, fast)."""

    def __init__(self, onnx: Path, voices: Path) -> None:
        from kokoro_onnx import Kokoro  # type: ignore

        self._k = Kokoro(str(onnx), str(voices))
        print(f"Kokoro backend: kokoro-onnx ({onnx.name})", flush=True)

    def synthesize(
        self, text: str, voice: str, lang_code: str, speed: float
    ) -> tuple[np.ndarray, int]:
        # kokoro-onnx lang strings differ slightly; map pipeline codes.
        lang = {
            "a": "en-us",
            "b": "en-gb",
            "e": "es",
            "f": "fr-fr",
            "h": "hi",
            "i": "it",
            "j": "ja",
            "p": "pt-br",
            "z": "zh",
        }.get(lang_code, "en-us")
        try:
            samples, sr = self._k.create(text, voice=voice, speed=speed, lang=lang)
        except TypeError:
            # Older API without lang=
            samples, sr = self._k.create(text, voice=voice, speed=speed)
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("kokoro-onnx produced empty audio")
        return audio, int(sr)


def build_engine(args: argparse.Namespace) -> Engine:
    if args.backend in ("auto", "kokoro"):
        try:
            return KokoroOfficial()
        except Exception as e:
            if args.backend == "kokoro":
                raise
            print(f"hexgrad/kokoro unavailable ({e}); trying kokoro-onnx…", flush=True)

    model_dir = Path(args.model_dir)
    onnx = Path(args.onnx) if args.onnx else model_dir / "kokoro-v1.0.onnx"
    voices = Path(args.voices) if args.voices else model_dir / "voices-v1.0.bin"
    if args.download or not onnx.is_file() or not voices.is_file():
        _download(ONNX_URL, onnx)
        _download(VOICES_URL, voices)
    return KokoroOnnx(onnx, voices)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_models = root / "models" / "kokoro"

    p = argparse.ArgumentParser(description="Kokoro OpenAI-compatible TTS server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8084)
    p.add_argument(
        "--backend",
        choices=["auto", "kokoro", "onnx"],
        default="auto",
        help="auto = kokoro package then kokoro-onnx",
    )
    p.add_argument("--model-dir", default=str(default_models))
    p.add_argument("--onnx", default=None, help="Path to kokoro-v1.0.onnx")
    p.add_argument("--voices", default=None, help="Path to voices-v1.0.bin")
    p.add_argument(
        "--download",
        action="store_true",
        help="Force download ONNX assets for kokoro-onnx",
    )
    p.add_argument("--speaker", default="af_bella", help="Default voice id")
    args = p.parse_args()

    try:
        engine = build_engine(args)
    except Exception as e:
        print(
            "Failed to load Kokoro. Install one of:\n"
            "  pip install kokoro 'misaki[en]' soundfile numpy fastapi uvicorn\n"
            "  pip install kokoro-onnx soundfile numpy fastapi uvicorn\n"
            f"Error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    app = FastAPI(title="kokoro-tts", version="1.0")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "engine": type(engine).__name__,
            "sample_rate": KOKORO_SR,
            "default_voice": args.speaker,
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {"id": "kokoro", "object": "model", "owned_by": "local"},
            ],
        }

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        text = (req.input or req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")

        lang_code = _resolve_lang(req.language, req.lang)
        voice = (req.voice or args.speaker or DEFAULT_VOICE.get(lang_code, "af_bella")).strip()
        if not voice or _is_supertonic_or_qwen_voice(voice):
            voice = DEFAULT_VOICE.get(lang_code, args.speaker)

        speed = float(req.speed or 1.0)
        speed = max(0.5, min(2.0, speed))
        fmt = (req.response_format or "wav").lower().strip()

        try:
            audio, sr = engine.synthesize(text, voice, lang_code, speed)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"synth failed: {e}") from e

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak

        if fmt in ("pcm", "raw", "s16le"):
            pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
            return Response(
                content=pcm.tobytes(),
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": str(sr),
                    "X-Channels": "1",
                },
            )

        # Default WAV (and anything else we don't specially handle)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    print(
        f"Kokoro TTS listening on http://{args.host}:{args.port}/v1/audio/speech "
        f"(default voice={args.speaker})",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
