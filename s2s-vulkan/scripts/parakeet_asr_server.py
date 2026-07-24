#!/usr/bin/env python3
"""Local-only HTTP sidecar for NVIDIA Parakeet TDT 0.6B v3."""

from __future__ import annotations

import argparse
import asyncio
import io
import os
from dataclasses import dataclass
from typing import Any


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


def select_device(torch: Any, requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PARAKEET_DEVICE=cuda requested, but CUDA is unavailable")
    if requested == "xpu" and (
        not hasattr(torch, "xpu") or not torch.xpu.is_available()
    ):
        raise RuntimeError("PARAKEET_DEVICE=xpu requested, but Intel XPU is unavailable")
    if requested not in {"cpu", "cuda", "xpu"}:
        raise RuntimeError(f"unsupported PARAKEET_DEVICE={requested!r}")
    return requested


def select_dtype(torch: Any, device: str, requested: str) -> Any:
    requested = requested.strip().lower()
    if requested == "auto":
        requested = "float16" if device == "cuda" else "bfloat16" if device == "xpu" else "float32"
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if requested not in choices:
        raise RuntimeError(f"unsupported PARAKEET_DTYPE={requested!r}")
    if device == "cpu" and requested != "float32":
        raise RuntimeError("CPU inference requires PARAKEET_DTYPE=float32")
    return choices[requested]


def normalize_decoded(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


@dataclass
class Runtime:
    torch: Any
    torchaudio: Any
    soundfile: Any
    processor: Any
    model: Any
    device: str
    dtype: Any
    model_dir: str
    lock: asyncio.Lock

    def transcribe_sync(self, audio_bytes: bytes) -> str:
        audio, sample_rate = self.soundfile.read(
            io.BytesIO(audio_bytes), dtype="float32", always_2d=True
        )
        if audio.shape[0] == 0:
            raise ValueError("empty audio")
        mono = audio.mean(axis=1)
        target_rate = int(self.processor.feature_extractor.sampling_rate)
        waveform = self.torch.from_numpy(mono)
        if int(sample_rate) != target_rate:
            waveform = self.torchaudio.functional.resample(
                waveform, int(sample_rate), target_rate
            )

        inputs = self.processor(
            waveform.numpy(), sampling_rate=target_rate, return_tensors="pt"
        )
        for key, tensor in inputs.items():
            if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
                inputs[key] = tensor.to(device=self.device, dtype=self.dtype)
            else:
                inputs[key] = tensor.to(device=self.device)

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, return_dict_in_generate=True)
        return normalize_decoded(
            self.processor.decode(output.sequences, skip_special_tokens=True)
        )

    async def transcribe(self, audio_bytes: bytes) -> str:
        async with self.lock:
            return await asyncio.to_thread(self.transcribe_sync, audio_bytes)


async def read_audio_part(request: Any) -> bytes:
    reader = await request.multipart()
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name in {"file", "audio"}:
            return await part.read(decode=False)
    raise ValueError("multipart field 'file' is required")


def create_app(web: Any, runtime: Runtime) -> Any:
    async def health(_request: Any) -> Any:
        return web.json_response(
            {
                "status": "ok",
                "model": MODEL_ID,
                "device": runtime.device,
                "dtype": str(runtime.dtype),
            }
        )

    async def inference(request: Any) -> Any:
        try:
            audio_bytes = await read_audio_part(request)
            text = await runtime.transcribe(audio_bytes)
            return web.json_response({"text": text, "model": MODEL_ID})
        except (ValueError, RuntimeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        except Exception as error:
            return web.json_response({"error": f"transcription failed: {error}"}, status=500)

    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_post("/inference", inference)
    app.router.add_post("/v1/audio/transcriptions", inference)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("PARAKEET_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PARAKEET_PORT", "8082")))
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import soundfile
    import torch
    import torchaudio
    from aiohttp import web
    from transformers import AutoModelForTDT, AutoProcessor

    model_dir = os.getenv("PARAKEET_MODEL_DIR", "/models/parakeet-tdt-0.6b-v3")
    device = select_device(torch, os.getenv("PARAKEET_DEVICE", "auto"))
    dtype = select_dtype(torch, device, os.getenv("PARAKEET_DTYPE", "auto"))

    print(f"Loading {MODEL_ID} from {model_dir} on {device} ({dtype})", flush=True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForTDT.from_pretrained(
        model_dir, local_files_only=True, dtype=dtype
    )
    model.to(device)
    model.eval()

    runtime = Runtime(
        torch=torch,
        torchaudio=torchaudio,
        soundfile=soundfile,
        processor=processor,
        model=model,
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        lock=asyncio.Lock(),
    )
    print(f"Parakeet ready on http://{args.host}:{args.port}", flush=True)
    web.run_app(create_app(web, runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
