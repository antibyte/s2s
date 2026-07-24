#!/usr/bin/env python3
"""Local HTTP sidecar for Mistral Voxtral Mini 4B Realtime (batch /inference).

Compatible with s2s-vulkan STT: POST multipart /inference → JSON {"text": "..."}.

Recommended stack is vLLM realtime streaming; this sidecar uses Transformers for
utterance-level VAD clips in the lab pipeline (Apache-2.0 weights).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
from dataclasses import dataclass
from typing import Any


MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def select_device(torch: Any, requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VOXTRAL_DEVICE=cuda requested, but CUDA is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise RuntimeError(f"unsupported VOXTRAL_DEVICE={requested!r}")
    return requested


def select_dtype(torch: Any, device: str, requested: str) -> Any:
    requested = requested.strip().lower()
    if requested == "auto":
        requested = "bfloat16" if device == "cuda" else "float32"
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if requested not in choices:
        raise RuntimeError(f"unsupported VOXTRAL_DTYPE={requested!r}")
    if device == "cpu" and requested not in {"float32", "bfloat16"}:
        # CPU path: prefer float32 for widest compatibility.
        return torch.float32
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
        # Voxtral audio feature extractor typically expects 16 kHz.
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        target_rate = int(getattr(feature_extractor, "sampling_rate", 16_000) or 16_000)
        waveform = self.torch.from_numpy(mono)
        if int(sample_rate) != target_rate:
            waveform = self.torchaudio.functional.resample(
                waveform, int(sample_rate), target_rate
            )

        inputs = self.processor(waveform.numpy(), return_tensors="pt")
        for key, tensor in list(inputs.items()):
            if hasattr(tensor, "to"):
                if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
                    inputs[key] = tensor.to(device=self.device, dtype=self.dtype)
                else:
                    inputs[key] = tensor.to(device=self.device)

        with self.torch.inference_mode():
            # Realtime model also supports non-streaming generate for short clips.
            outputs = self.model.generate(**inputs)
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
        if isinstance(decoded, list) and decoded:
            return normalize_decoded(decoded[0])
        return normalize_decoded(decoded)

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
        except Exception as error:  # noqa: BLE001 — surface as HTTP 500 for the lab
            return web.json_response({"error": f"transcription failed: {error}"}, status=500)

    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_post("/inference", inference)
    app.router.add_post("/v1/audio/transcriptions", inference)
    return app


def load_model(torch: Any, model_dir: str, device: str, dtype: Any) -> tuple[Any, Any]:
    """Load Voxtral Realtime; fall back to AutoModel if class name differs."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    try:
        from transformers import VoxtralRealtimeForConditionalGeneration

        model_cls = VoxtralRealtimeForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForCausalLM

        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return processor, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("VOXTRAL_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VOXTRAL_PORT", "8082")))
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import soundfile
    import torch
    import torchaudio
    from aiohttp import web

    model_dir = os.getenv("VOXTRAL_MODEL_DIR", "/models/voxtral-mini-4b-realtime")
    device = select_device(torch, os.getenv("VOXTRAL_DEVICE", "auto"))
    dtype = select_dtype(torch, device, os.getenv("VOXTRAL_DTYPE", "auto"))

    print(f"Loading {MODEL_ID} from {model_dir} on {device} ({dtype})", flush=True)
    processor, model = load_model(torch, model_dir, device, dtype)

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
    print(f"Voxtral ready on http://{args.host}:{args.port}", flush=True)
    web.run_app(create_app(web, runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
