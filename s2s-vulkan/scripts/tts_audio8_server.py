#!/usr/bin/env python3
"""Allowlisted OpenAI-compatible HTTP sidecar for Audio8 TTS Preview models."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import threading
import traceback
import wave
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_ID = "Audio8/Audio8-TTS-Preview-0.6b"
MODEL_ALIAS = "audio8-tts-preview-0.6b"
MODEL_REVISION = "f9612f13a0ab40facf3d050fc908b9e6db05c2be"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_TEXT_CHARS = 5000
DEFAULT_SAMPLE_RATE = 44_100
REQUIRED_ARTIFACTS = (
    "model.safetensors",
    "codec.pth",
    "config.json",
    "configuration_arktts.py",
    "modeling_arktts.py",
    "modeling_arktts_codec.py",
    "processing_arktts.py",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)
SUPPORTED_LANGUAGES = frozenset(
    {
        "yue",
        "zh",
        "nl",
        "en",
        "fr",
        "de",
        "it",
        "ja",
        "ko",
        "pl",
        "es",
    }
)
LANGUAGE_ALIASES = {
    "cantonese": "yue",
    "chinese": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "polish": "pl",
    "spanish": "es",
}

# Keep offline after the catalog download so the sidecar cannot pull unpinned weights.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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


_SAFE_VOICE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    language: str
    response_format: str
    voice: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int

    @classmethod
    def from_payload(cls, payload: Any, default_language: str) -> "SpeechRequest":
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        text = str(payload.get("input") or payload.get("text") or "").strip()
        if not text:
            raise ValueError("input text must not be empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"input text exceeds {MAX_TEXT_CHARS} characters")

        voice = str(payload.get("voice") or "default").strip()
        if voice.lower() in {"", "default", "builtin", "auto"}:
            voice = "default"
        elif not _SAFE_VOICE.match(voice):
            raise ValueError(
                "voice must be 'default' or a registered local voice name "
                "(alphanumeric, '.', '_', '-')"
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
            voice=voice,
            temperature=bounded_float(
                payload.get("temperature"),
                name="temperature",
                default=0.8,
                minimum=0.05,
                maximum=2.0,
            ),
            top_p=bounded_float(
                payload.get("top_p"),
                name="top_p",
                default=0.95,
                minimum=0.05,
                maximum=1.0,
            ),
            top_k=bounded_int(
                payload.get("top_k"),
                name="top_k",
                default=50,
                minimum=1,
                maximum=200,
            ),
            max_new_tokens=bounded_int(
                payload.get("max_new_tokens"),
                name="max_new_tokens",
                default=1024,
                minimum=16,
                maximum=4096,
            ),
        )


def float32_mono_to_pcm16(waveform: Any) -> bytes:
    import numpy as np

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("model returned empty or non-finite audio")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    normalized = np.clip(audio, -1.0, 1.0)
    return (normalized * 32767.0).round().astype("<i2").tobytes()


def wav_bytes(pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
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
        raise ValueError("missing Audio8 artifacts: " + ", ".join(missing))
    return resolved


def resolve_reference(
    model_dir: Path, voice: str
) -> tuple[str | None, str | None]:
    """Optional zero-shot clone: models/.../voices/<name>.wav + <name>.txt."""
    if voice == "default":
        return None, None
    voices_dir = (model_dir / "voices").resolve()
    if not voices_dir.is_dir():
        raise ValueError(
            f"voice '{voice}' requires a voices/ directory under the model dir"
        )
    audio = (voices_dir / f"{voice}.wav").resolve()
    transcript = (voices_dir / f"{voice}.txt").resolve()
    if not str(audio).startswith(str(voices_dir)) or not str(transcript).startswith(
        str(voices_dir)
    ):
        raise ValueError("voice path escapes the voices directory")
    if not audio.is_file():
        raise ValueError(f"reference audio missing: {audio.name}")
    if not transcript.is_file():
        raise ValueError(
            f"reference transcript missing: {transcript.name} "
            "(must match spoken content in the WAV)"
        )
    text = transcript.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"reference transcript is empty: {transcript.name}")
    return str(audio), text


class Audio8Runtime:
    def __init__(
        self,
        model_dir: Path,
        device: str,
        *,
        model_id: str = MODEL_ID,
        model_alias: str = MODEL_ALIAS,
        revision: str = MODEL_REVISION,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        model_dir = validate_model_dir(model_dir)
        self.model_dir = model_dir
        self.device = device
        self.model_id = model_id
        self.model_alias = model_alias
        self.revision = revision
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(
            f"Loading {self.model_alias} revision={self.revision} "
            f"device={device} model_dir={model_dir}",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        # Transformers 4.57 prefers `dtype`; older builds use `torch_dtype`.
        try:
            self.model = AutoModel.from_pretrained(
                str(model_dir), dtype=dtype, **load_kwargs
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(
                str(model_dir), torch_dtype=dtype, **load_kwargs
            )
        self.model = self.model.eval().to(device)
        self.sample_rate = int(
            getattr(self.model.config, "codec_sample_rate", DEFAULT_SAMPLE_RATE)
            or DEFAULT_SAMPLE_RATE
        )
        self.lock = threading.Lock()
        print(
            f"Audio8 ready at {self.sample_rate} Hz on {device}.",
            flush=True,
        )

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        import torch

        reference_audio, reference_text = resolve_reference(
            self.model_dir, request.voice
        )
        processor_kwargs: dict[str, Any] = {
            "text": [request.text],
            "return_tensors": "pt",
        }
        if reference_audio is not None and reference_text is not None:
            processor_kwargs["reference_audio"] = [reference_audio]
            processor_kwargs["reference_text"] = [reference_text]

        with self.lock:
            inputs = self.processor(**processor_kwargs)
            inputs = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in inputs.items()
            }
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    do_sample=True,
                    return_dict_in_generate=True,
                )
                waveforms, waveform_lengths = self.model.decode_audio(output.codes)

        length = int(waveform_lengths[0])
        waveform = waveforms[0, :length].float().cpu().numpy()
        pcm = float32_mono_to_pcm16(waveform)
        if request.response_format == "pcm":
            return pcm, "audio/pcm"
        return wav_bytes(pcm, self.sample_rate), "audio/wav"


def make_handler(
    runtime: Audio8Runtime, default_language: str
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

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
                        "backend": runtime.model_alias,
                        "model": runtime.model_id,
                        "revision": runtime.revision,
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
                                "id": runtime.model_alias,
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
                request = SpeechRequest.from_payload(payload, default_language)
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
                extra_headers={
                    "X-Sample-Rate": str(runtime.sample_rate),
                    "X-S2S-Model": runtime.model_alias,
                },
            )

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "S2S_AUDIO8_MODEL_DIR", "models/audio8-tts-preview-0.6b"
        ),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("S2S_AUDIO8_DEVICE", "cpu"),
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--default-language",
        default=os.environ.get("S2S_AUDIO8_LANGUAGE", "de"),
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("S2S_AUDIO8_MODEL_ID", MODEL_ID),
    )
    parser.add_argument(
        "--model-alias",
        default=os.environ.get("S2S_AUDIO8_MODEL_ALIAS", MODEL_ALIAS),
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("S2S_AUDIO8_REVISION", MODEL_REVISION),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        default_language = normalize_language(args.default_language, "de")
        runtime = Audio8Runtime(
            Path(args.model_dir),
            device=args.device,
            model_id=args.model_id,
            model_alias=args.model_alias,
            revision=args.revision,
        )
    except Exception as error:  # noqa: BLE001
        print(f"failed to load Audio8: {error}", file=sys.stderr)
        return 1
    handler = make_handler(runtime, default_language)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Audio8 TTS listening on http://{args.host}:{args.port} "
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
