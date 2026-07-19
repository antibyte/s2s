#!/usr/bin/env python3
"""
Minimal HTTP TTS wrapper around faster-qwen3-tts / qwentts.cpp (Vulkan via GGML).

Build the Vulkan wheel first (see README), then:

  export GGML_BACKEND=Vulkan0
  pip install fastapi uvicorn
  python scripts/tts_qwen_server.py --port 8083 --quant Q4_K_M

s2s-vulkan:

  s2s-vulkan --tts http --tts_url http://127.0.0.1:8083/v1/audio/speech
"""

from __future__ import annotations

import argparse
import io
import os
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: Optional[str] = None
    text: Optional[str] = None
    voice: Optional[str] = "Aiden"
    language: Optional[str] = "auto"
    response_format: Optional[str] = "wav"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8083)
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    p.add_argument("--device", default="cpu", help="Python device flag; GGML uses GGML_BACKEND")
    p.add_argument("--backend", default="ggml")
    p.add_argument("--quant", default="Q4_K_M", choices=["BF16", "Q8_0", "Q4_K_M"])
    p.add_argument("--speaker", default="Aiden")
    args = p.parse_args()

    # Prefer explicit Vulkan adapter when the native lib supports it.
    os.environ.setdefault("GGML_BACKEND", "Vulkan0")

    from faster_qwen3_tts import FasterQwen3TTS  # type: ignore

    print(f"Loading Qwen3-TTS {args.model} backend={args.backend} quant={args.quant} …")
    model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        backend=args.backend,
        quant=args.quant,
    )
    print("Model ready.")

    app = FastAPI(title="qwen3-tts-vulkan-wrapper")

    @app.get("/health")
    def health():
        return {"ok": True, "ggml_backend": os.environ.get("GGML_BACKEND")}

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        text = (req.input or req.text or "").strip()
        if not text:
            return Response(status_code=400, content=b"empty text")

        # faster-qwen3-tts API surface varies slightly across versions; try common paths.
        wav = None
        sr = 24000
        speaker = req.voice or args.speaker
        language = req.language or "auto"

        if hasattr(model, "generate"):
            out = model.generate(text=text, speaker=speaker, language=language)
            if isinstance(out, tuple):
                wav, sr = out[0], int(out[1])
            else:
                wav = out
        elif hasattr(model, "synthesize"):
            out = model.synthesize(text, speaker=speaker, language=language)
            if isinstance(out, tuple):
                wav, sr = out[0], int(out[1])
            else:
                wav = out
        else:
            raise RuntimeError("FasterQwen3TTS has neither generate() nor synthesize()")

        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
