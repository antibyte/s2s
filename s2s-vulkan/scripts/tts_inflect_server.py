#!/usr/bin/env python3
"""Allowlisted OpenAI-compatible HTTP sidecar for Inflect-Micro-v2 (9.3M EN TTS)."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import threading
import traceback
import wave
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_ID = "owensong/Inflect-Micro-v2"
MODEL_ALIAS = "inflect-micro-v2"
MODEL_REVISION = "1e0f60061e50c6849ce4c79c9aff0887fa631d81"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_TEXT_CHARS = 5000
SAMPLE_RATE = 24_000
REQUIRED_ARTIFACTS = (
    "model.pth",
    "config.json",
    "inference.py",
)
# Official FP32 weight size / digest (release_manifest.json).
MODEL_PTH_SIZE = 37_529_995
MODEL_PTH_SHA256 = "3eede065c9ccfa88ade0a5a9a5c23de34afcbbb32213e59aad44d5cf100fdee8"


def bounded_float(
    value: Any,
    *,
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def bounded_int(
    value: Any,
    *,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def normalize_language(value: Any) -> str:
    language = str(value or "").strip().lower().replace("_", "-")
    if language in {"", "auto", "default", "en", "en-us", "en-gb", "english"}:
        return "en"
    language = language.split("-", 1)[0]
    if language != "en":
        raise ValueError(
            f"unsupported language '{value}'; Inflect-Micro-v2 is English-only"
        )
    return "en"


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    language: str
    response_format: str
    speed: float
    variation: float
    seed: int

    @classmethod
    def from_payload(cls, payload: Any) -> "SpeechRequest":
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        text = str(payload.get("input") or payload.get("text") or "").strip()
        if not text:
            raise ValueError("input text must not be empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"input text exceeds {MAX_TEXT_CHARS} characters")

        # Fixed synthetic male voice; ignore OpenAI voice names except placeholders.
        voice = str(payload.get("voice") or "default").strip().lower()
        if voice not in {"", "default", "male", "builtin", "auto"}:
            raise ValueError(
                "only the pinned built-in male voice is available; use voice='default'"
            )

        response_format = str(payload.get("response_format") or "wav").lower()
        if response_format not in {"wav", "pcm"}:
            raise ValueError("response_format must be 'wav' or 'pcm'")

        return cls(
            text=text,
            language=normalize_language(
                payload.get("language") or payload.get("lang")
            ),
            response_format=response_format,
            speed=bounded_float(
                payload.get("speed"),
                name="speed",
                default=1.0,
                minimum=0.5,
                maximum=2.0,
            ),
            variation=bounded_float(
                payload.get("variation"),
                name="variation",
                default=0.667,
                minimum=0.0,
                maximum=1.0,
            ),
            seed=bounded_int(
                payload.get("seed"),
                name="seed",
                default=0,
                minimum=0,
                maximum=2_147_483_647,
            ),
        )


def float32_mono_to_pcm16(waveform: Any) -> bytes:
    import numpy as np

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("model returned empty or non-finite audio")
    normalized = np.clip(audio, -1.0, 1.0)
    return (normalized * 32767.0).round().astype("<i2").tobytes()


def wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return buffer.getvalue()


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_dir(model_dir: Path, *, verify_weights: bool = True) -> Path:
    resolved = model_dir.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"model directory is not a directory: {resolved}")
    missing = [name for name in REQUIRED_ARTIFACTS if not (resolved / name).is_file()]
    if missing:
        raise ValueError("missing Inflect artifacts: " + ", ".join(missing))
    if verify_weights:
        weights = resolved / "model.pth"
        size = weights.stat().st_size
        if size != MODEL_PTH_SIZE:
            raise ValueError(
                f"model.pth size mismatch: got {size}, expected {MODEL_PTH_SIZE}"
            )
        digest = file_sha256(weights)
        if digest != MODEL_PTH_SHA256:
            raise ValueError(
                f"model.pth sha256 mismatch: got {digest}, expected {MODEL_PTH_SHA256}"
            )
    return resolved


class InflectRuntime:
    def __init__(self, model_dir: Path, device: str) -> None:
        model_dir = validate_model_dir(model_dir, verify_weights=True)
        # Official package imports from its own directory.
        sys.path.insert(0, str(model_dir))
        from inference import InflectTTS  # type: ignore

        print(
            f"Loading {MODEL_ALIAS} revision={MODEL_REVISION} "
            f"device={device} model_dir={model_dir}",
            flush=True,
        )
        self.engine = InflectTTS(str(model_dir), device=device)
        self.device = device
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()
        print(f"Inflect ready at {self.sample_rate} Hz on {device}.", flush=True)

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        with self.lock:
            result = self.engine.synthesize(
                request.text,
                speed=request.speed,
                variation=request.variation,
                seed=request.seed,
            )
        # Official API: (sample_rate, waveform) or object with similar fields.
        if isinstance(result, tuple) and len(result) == 2:
            sample_rate, waveform = result
        else:
            sample_rate = getattr(result, "sample_rate", SAMPLE_RATE)
            waveform = getattr(result, "waveform", result)
        sample_rate = int(sample_rate) or SAMPLE_RATE
        pcm = float32_mono_to_pcm16(waveform)
        if request.response_format == "pcm":
            return pcm, "audio/pcm"
        return wav_bytes(pcm, sample_rate), "audio/wav"


def make_handler(runtime: InflectRuntime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            sys.stderr.write(
                "%s - %s\n" % (self.address_string(), format % args)
            )

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._send(status, body, "application/json")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in {"/health", "/"}:
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "backend": MODEL_ALIAS,
                        "model": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "sample_rate": runtime.sample_rate,
                        "device": runtime.device,
                    },
                )
                return
            if path == "/v1/models":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": MODEL_ALIAS,
                                "object": "model",
                                "owned_by": "local",
                            }
                        ],
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in {"/v1/audio/speech", "/inference"}:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid Content-Length"},
                )
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                request = SpeechRequest.from_payload(payload)
                audio, content_type = runtime.synthesize(request)
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception as error:  # noqa: BLE001
                traceback.print_exc()
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"synthesis failed: {error}"},
                )
                return
            self._send(
                HTTPStatus.OK,
                audio,
                content_type,
                extra_headers={"X-Sample-Rate": str(runtime.sample_rate)},
            )

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("S2S_INFLECT_MODEL_DIR", "models/inflect-micro-v2"),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("S2S_INFLECT_DEVICE", "cpu"),
        choices=("cpu", "cuda"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = Path(args.model_dir)
    try:
        runtime = InflectRuntime(model_dir, device=args.device)
    except Exception as error:  # noqa: BLE001
        print(f"failed to load Inflect: {error}", file=sys.stderr)
        return 1
    handler = make_handler(runtime)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Inflect-Micro-v2 listening on http://{args.host}:{args.port} "
        f"(POST /v1/audio/speech)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
