#!/usr/bin/env python3
"""Allowlisted OpenAI-compatible HTTP sidecar for Chatterbox Multilingual V3."""

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


MODEL_ID = "ResembleAI/chatterbox"
MODEL_ALIAS = "chatterbox-multilingual-v3"
MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
T3_MODEL = "t3_mtl23ls_v3.safetensors"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_TEXT_CHARS = 5000
REQUIRED_ARTIFACTS = (
    "ve.pt",
    T3_MODEL,
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)
SUPPORTED_LANGUAGES = frozenset(
    {
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "hi",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "sw",
        "tr",
        "zh",
    }
)
LANGUAGE_ALIASES = {
    "arabic": "ar",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "finnish": "fi",
    "french": "fr",
    "hebrew": "he",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "malay": "ms",
    "dutch": "nl",
    "norwegian": "no",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "swedish": "sv",
    "swahili": "sw",
    "turkish": "tr",
    "chinese": "zh",
}

# from_local() must never reach around the catalog's pinned model artifacts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def normalize_language(value: Any, default: str = "de") -> str:
    language = str(value or "").strip().lower().replace("_", "-")
    if language in {"", "auto", "default"}:
        language = default
    language = LANGUAGE_ALIASES.get(language, language.split("-", 1)[0])
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"unsupported language '{value}'; expected one of "
            + ", ".join(sorted(SUPPORTED_LANGUAGES))
        )
    return language


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


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    language: str
    response_format: str
    exaggeration: float
    cfg_weight: float
    temperature: float

    @classmethod
    def from_payload(cls, payload: Any, default_language: str) -> "SpeechRequest":
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        text = str(payload.get("input") or payload.get("text") or "").strip()
        if not text:
            raise ValueError("input text must not be empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"input text exceeds {MAX_TEXT_CHARS} characters")

        voice = str(payload.get("voice") or "default").strip().lower()
        if voice not in {"default", "builtin"}:
            raise ValueError(
                "only the pinned built-in voice is available; use voice='default'"
            )
        response_format = str(payload.get("response_format") or "wav").lower()
        if response_format not in {"wav", "pcm"}:
            raise ValueError("response_format must be 'wav' or 'pcm'")

        return cls(
            text=text,
            language=normalize_language(
                payload.get("language") or payload.get("lang"),
                default_language,
            ),
            response_format=response_format,
            exaggeration=bounded_float(
                payload.get("exaggeration"),
                name="exaggeration",
                default=0.5,
                minimum=0.0,
                maximum=2.0,
            ),
            cfg_weight=bounded_float(
                payload.get("cfg_weight"),
                name="cfg_weight",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
            ),
            temperature=bounded_float(
                payload.get("temperature"),
                name="temperature",
                default=0.8,
                minimum=0.05,
                maximum=2.0,
            ),
        )


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return buffer.getvalue()


def validate_model_dir(model_dir: Path) -> Path:
    resolved = model_dir.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"model directory is not a directory: {resolved}")
    missing = [name for name in REQUIRED_ARTIFACTS if not (resolved / name).is_file()]
    if missing:
        raise ValueError("missing Chatterbox artifacts: " + ", ".join(missing))
    return resolved


def load_local_cangjie(model: Any, model_dir: Path) -> None:
    """Populate the tokenizer from the catalog artifact instead of the Hub."""
    converter = model.tokenizer.cangjie_converter
    with (model_dir / "Cangjie5_TC.json").open("r", encoding="utf-8") as source:
        entries = json.load(source)
    word2cj: dict[str, str] = {}
    cj2word: dict[str, list[str]] = {}
    for entry in entries:
        fields = str(entry).split("\t")
        if len(fields) < 2:
            continue
        word, code = fields[0], fields[1]
        word2cj[word] = code
        cj2word.setdefault(code, []).append(word)
    if not word2cj:
        raise ValueError("Cangjie mapping is empty")
    converter.word2cj = word2cj
    converter.cj2word = cj2word


class ChatterboxRuntime:
    def __init__(
        self,
        model_dir: Path,
        device: str,
        default_language: str,
        threads: int,
    ) -> None:
        import numpy as np
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA variant selected, but torch.cuda is unavailable")
        if threads > 0:
            torch.set_num_threads(threads)

        print(
            f"Loading {MODEL_ALIAS} revision={MODEL_REVISION} "
            f"device={device} model_dir={model_dir}",
            flush=True,
        )
        model = ChatterboxMultilingualTTS.from_local(
            model_dir,
            device=device,
            t3_model=T3_MODEL,
        )
        load_local_cangjie(model, model_dir)
        self.model = model
        self.np = np
        self.device = device
        self.default_language = normalize_language(default_language)
        self.sample_rate = int(model.sr)
        self.lock = threading.Lock()
        print(f"Chatterbox ready at {self.sample_rate} Hz.", flush=True)

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        with self.lock:
            generated = self.model.generate(
                request.text,
                language_id=request.language,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
            )
        audio = generated.detach().cpu().numpy().reshape(-1)
        if audio.size == 0 or not self.np.isfinite(audio).all():
            raise RuntimeError("model returned empty or non-finite audio")
        normalized = self.np.clip(audio, -1.0, 1.0).astype(
            self.np.float32,
            copy=False,
        )
        pcm = (normalized * 32767.0).round().astype("<i2").tobytes()
        if request.response_format == "pcm":
            return pcm, "audio/pcm"
        return wav_bytes(pcm, self.sample_rate), "audio/wav"


class ChatterboxServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        runtime: ChatterboxRuntime,
    ) -> None:
        super().__init__(address, ChatterboxHandler)
        self.runtime = runtime


class ChatterboxHandler(BaseHTTPRequestHandler):
    server: ChatterboxServer

    def send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self.send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "model": MODEL_ALIAS,
                    "revision": MODEL_REVISION,
                    "device": self.server.runtime.device,
                    "sample_rate": self.server.runtime.sample_rate,
                },
            )
            return
        if path == "/v1/models":
            self.send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ALIAS,
                            "object": "model",
                            "owned_by": "ResembleAI",
                        }
                    ],
                },
            )
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/v1/audio/speech":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            raw_length = self.headers.get("Content-Length", "")
            length = int(raw_length)
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError(
                    f"Content-Length must be between 1 and {MAX_REQUEST_BYTES}"
                )
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            request = SpeechRequest.from_payload(
                payload,
                self.server.runtime.default_language,
            )
            body, content_type = self.server.runtime.synthesize(request)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:  # pragma: no cover - hardware/model path
            traceback.print_exc(file=sys.stderr)
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Chatterbox synthesis failed: {error}"},
            )
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Audio-Sample-Rate", str(self.server.runtime.sample_rate))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, template: str, *args: Any) -> None:
        sys.stderr.write(
            f"[chatterbox] {self.address_string()} "
            + (template % args)
            + "\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible Chatterbox Multilingual V3 sidecar"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--default-language", default="de")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if not 0 <= args.threads <= 256:
        raise SystemExit("--threads must be between 0 and 256")
    model_dir = validate_model_dir(args.model_dir)
    runtime = ChatterboxRuntime(
        model_dir,
        args.device,
        args.default_language,
        args.threads,
    )
    server = ChatterboxServer((args.host, args.port), runtime)
    print(f"Listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
