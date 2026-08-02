#!/usr/bin/env python3
"""Allowlisted OpenAI-compatible XTTS-v2 ONNX sidecar.

The WebGPU mode uses ONNX Runtime's native plugin EP. Every ONNX session is
created against the same selected Intel Arc B580 WebGPU device with Dawn forced
to Vulkan. The process refuses to become healthy when GPT or HiFiGAN executes
entirely on the CPU fallback.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import re
import sys
import threading
import traceback
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable


MODEL_ID = "coqui/XTTS-v2"
MODEL_ALIAS = "xtts-v2"
ONNX_REVISION = "975b202585dea4ae6ca7f6118121cdf1011d7d28"
SAMPLE_RATE = 24_000
MAX_REQUEST_BYTES = 1024 * 1024
MAX_TEXT_CHARS = 5000
VOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SUPPORTED_LANGUAGES = frozenset(
    {
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "zh-cn",
        "ja",
        "hu",
        "ko",
        "hi",
    }
)
LANGUAGE_ALIASES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "polish": "pl",
    "turkish": "tr",
    "russian": "ru",
    "dutch": "nl",
    "czech": "cs",
    "arabic": "ar",
    "chinese": "zh-cn",
    "zh": "zh-cn",
    "japanese": "ja",
    "hungarian": "hu",
    "korean": "ko",
    "hindi": "hi",
}
REQUIRED_MODEL_FILES = (
    "metadata.json",
    "vocab.json",
    "mel_stats.npy",
    "conditioning_encoder.onnx",
    "speaker_encoder.onnx",
    "gpt_model.onnx",
    "hifigan_vocoder.onnx",
    "embeddings/mel_embedding.npy",
    "embeddings/mel_pos_embedding.npy",
    "embeddings/text_embedding.npy",
    "embeddings/text_pos_embedding.npy",
)
REQUIRED_UPSTREAM_FILES = (
    "xtts_streaming_pipeline.py",
    "xtts_onnx_orchestrator.py",
    "xtts_tokenizer.py",
    "zh_num2words.py",
)
WEBGPU_PROVIDER_OPTIONS = {
    "dawnBackendType": "Vulkan",
    "powerPreference": "high-performance",
    "enableGraphCapture": "0",
    "preferredLayout": "NCHW",
    "validationMode": "basic",
    "deviceId": "0",
    "preserveDevice": "1",
}


def normalize_language(value: Any, default: str = "de") -> str:
    language = str(value or "").strip().lower().replace("_", "-")
    if language in {"", "auto", "default"}:
        language = default
    language = LANGUAGE_ALIASES.get(language, language)
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
    voice: str
    language: str
    response_format: str
    speed: float

    @classmethod
    def from_payload(cls, payload: Any, default_language: str) -> "SpeechRequest":
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        model = str(payload.get("model") or MODEL_ID).strip()
        if model.lower() not in {MODEL_ID.lower(), MODEL_ALIAS}:
            raise ValueError(f"unsupported model '{model}'")
        text = str(payload.get("input") or payload.get("text") or "").strip()
        if not text:
            raise ValueError("input text must not be empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"input text exceeds {MAX_TEXT_CHARS} characters")
        voice = str(payload.get("voice") or "de_sample").strip()
        if not VOICE_ID_RE.fullmatch(voice):
            raise ValueError("voice must be a safe installed voice id")
        response_format = str(payload.get("response_format") or "wav").lower()
        if response_format not in {"wav", "pcm"}:
            raise ValueError("response_format must be 'wav' or 'pcm'")
        return cls(
            text=text,
            voice=voice,
            language=normalize_language(
                payload.get("language") or payload.get("lang"),
                default_language,
            ),
            response_format=response_format,
            speed=bounded_float(
                payload.get("speed"),
                name="speed",
                default=1.0,
                minimum=0.5,
                maximum=2.0,
            ),
        )


class VoiceStore:
    def __init__(self, voices_dir: Path) -> None:
        self.root = voices_dir.expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError(f"voices directory is not a directory: {self.root}")

    def list_ids(self) -> list[str]:
        return sorted(
            path.stem
            for path in self.root.glob("*.wav")
            if path.is_file() and VOICE_ID_RE.fullmatch(path.stem)
        )

    def resolve(self, voice_id: str) -> Path:
        if not VOICE_ID_RE.fullmatch(voice_id):
            raise ValueError("voice must be a safe installed voice id")
        candidate = (self.root / f"{voice_id}.wav").resolve(strict=False)
        if candidate.parent != self.root or not candidate.is_file():
            raise ValueError(f"voice '{voice_id}' is not installed")
        validate_reference_wav(candidate)
        return candidate


def validate_reference_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() not in {1, 2}
                or source.getsampwidth() not in {2, 3, 4}
                or source.getframerate() < 8_000
                or source.getframerate() > 192_000
                or source.getnframes() == 0
            ):
                raise ValueError(f"voice '{path.stem}' contains unsupported WAV data")
    except (EOFError, wave.Error) as error:
        raise ValueError(f"voice '{path.stem}' is not a valid WAV file") from error


def validate_model_dir(model_dir: Path) -> Path:
    resolved = model_dir.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"model directory is not a directory: {resolved}")
    missing = [name for name in REQUIRED_MODEL_FILES if not (resolved / name).is_file()]
    if missing:
        raise ValueError("missing XTTS-v2 artifacts: " + ", ".join(missing))
    if (resolved / "gpt_model_int8.onnx").exists():
        print("Ignoring gpt_model_int8.onnx; this sidecar always uses pinned FP32.", flush=True)
    return resolved


def validate_upstream_dir(upstream_dir: Path) -> Path:
    resolved = upstream_dir.expanduser().resolve(strict=True)
    missing = [name for name in REQUIRED_UPSTREAM_FILES if not (resolved / name).is_file()]
    if missing:
        raise ValueError("missing pinned XTTS-v2 runtime sources: " + ", ".join(missing))
    return resolved


def pcm16_bytes(samples: Any) -> bytes:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("model returned empty or non-finite audio")
    if audio.size > SAMPLE_RATE * 120:
        raise RuntimeError("model returned more than 120 seconds of audio")
    return (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return buffer.getvalue()


def _device_value(device: Any, *names: str) -> Any:
    hardware = getattr(device, "hardware_device", None)
    for source in (hardware, device):
        if source is None:
            continue
        for name in names:
            value = getattr(source, name, None)
            if value is not None and value != "":
                return value
    return None


def _device_metadata(device: Any) -> dict[str, Any]:
    for name in ("ep_metadata", "metadata"):
        value = getattr(device, name, None)
        if isinstance(value, dict):
            return value
    return {}


def _int_id(value: Any) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value or 0)


def select_arc_b580_device(devices: Iterable[Any], ep_name: str) -> tuple[Any, dict[str, Any]]:
    candidates = [device for device in devices if getattr(device, "ep_name", "") == ep_name]
    for index, device in enumerate(candidates):
        name = str(_device_value(device, "name", "device_name") or "")
        vendor_id = _int_id(_device_value(device, "vendor_id", "vendorId"))
        device_id = _int_id(_device_value(device, "device_id", "deviceId"))
        metadata = _device_metadata(device)
        metadata_text = json.dumps(metadata, sort_keys=True).lower()
        if "d3d12" in metadata_text or "direct3d" in metadata_text:
            continue
        if vendor_id == 0x8086 and "arc b580" in name.lower():
            if metadata_text and "backend" in metadata_text and "vulkan" not in metadata_text:
                continue
            return device, {
                "name": name,
                "vendor_id": f"0x{vendor_id:04x}",
                "device_id": f"0x{device_id:04x}",
                "device_index": index,
                "metadata": metadata,
            }
    details = [
        {
            "name": str(_device_value(device, "name", "device_name") or ""),
            "vendor_id": f"0x{_int_id(_device_value(device, 'vendor_id', 'vendorId')):04x}",
            "metadata": _device_metadata(device),
        }
        for device in candidates
    ]
    raise RuntimeError(
        "WebGPU Vulkan requires an Intel Arc B580 adapter; discovered "
        + json.dumps(details, ensure_ascii=False)
    )


def session_role(model_path: Any) -> str:
    name = Path(str(model_path)).name.lower()
    if "conditioning" in name:
        return "conditioning"
    if "speaker" in name:
        return "speaker"
    if "gpt" in name:
        return "gpt"
    if "hifigan" in name or "vocoder" in name:
        return "hifigan"
    return Path(str(model_path)).stem


def profile_provider_counts(profile_path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    path = Path(profile_path)
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return counts
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    for event in events:
        provider = str((event.get("args") or {}).get("provider") or "").strip()
        if provider:
            counts[provider] = counts.get(provider, 0) + 1
    return counts


def validate_webgpu_session_placement(sessions: dict[str, dict[str, Any]]) -> None:
    for role in ("gpt", "hifigan"):
        placement = sessions.get(role, {})
        webgpu_nodes = int(placement.get("webgpu_nodes", 0))
        if webgpu_nodes <= 0:
            raise RuntimeError(
                f"XTTS {role} fell back fully to CPU; refusing Vulkan health"
            )


class OrtSessionFactory:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.records: list[tuple[str, Any]] = []
        self.adapter: dict[str, Any] = {}
        self.provider = "CPUExecutionProvider"
        self.dawn_backend = ""
        self._ort: Any = None
        self._real: Any = None
        self._device: Any = None

    def initialize(self) -> None:
        import onnxruntime as ort

        self._ort = ort
        self._real = ort.InferenceSession
        if self.mode == "cpu":
            return
        webgpu_ep = importlib.import_module("onnxruntime_ep_webgpu")
        ort.register_execution_provider_library(
            "s2s_xtts_webgpu",
            webgpu_ep.get_library_path(),
        )
        self._device, self.adapter = select_arc_b580_device(
            ort.get_ep_devices(),
            webgpu_ep.get_ep_name(),
        )
        self.provider = webgpu_ep.get_ep_name()
        self.dawn_backend = "Vulkan"

    def create(self, model_path: Any, *args: Any, **kwargs: Any) -> Any:
        del args
        kwargs.pop("sess_options", None)
        options = self._ort.SessionOptions()
        options.enable_profiling = True
        if self.mode == "webgpu-vulkan":
            provider_options = dict(WEBGPU_PROVIDER_OPTIONS)
            provider_options["deviceId"] = str(self.adapter["device_index"])
            options.add_provider_for_devices([self._device], provider_options)
            kwargs.pop("providers", None)
            kwargs.pop("provider_options", None)
            session = self._real(model_path, sess_options=options, **kwargs)
        else:
            kwargs.pop("providers", None)
            kwargs.pop("provider_options", None)
            session = self._real(
                model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
                **kwargs,
            )
        self.records.append((session_role(model_path), session))
        return session

    @contextmanager
    def installed(self) -> Iterable[None]:
        self.initialize()
        self._ort.InferenceSession = self.create
        try:
            yield
        finally:
            self._ort.InferenceSession = self._real

    def finish_profiles(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for role, session in self.records:
            counts = profile_provider_counts(session.end_profiling())
            webgpu_nodes = sum(
                count for provider, count in counts.items() if "webgpu" in provider.lower()
            )
            cpu_nodes = sum(
                count for provider, count in counts.items() if "cpu" in provider.lower()
            )
            current = result.setdefault(
                role,
                {
                    "provider": self.provider,
                    "webgpu_nodes": 0,
                    "cpu_nodes": 0,
                    "node_providers": {},
                },
            )
            current["webgpu_nodes"] += webgpu_nodes
            current["cpu_nodes"] += cpu_nodes
            for provider, count in counts.items():
                current["node_providers"][provider] = (
                    current["node_providers"].get(provider, 0) + count
                )
        return result


class XTTSRuntime:
    def __init__(
        self,
        model_dir: Path,
        voices_dir: Path,
        upstream_dir: Path,
        mode: str,
        default_language: str,
        threads: int,
    ) -> None:
        import numpy as np

        self.np = np
        self.mode = mode
        self.default_language = normalize_language(default_language)
        self.sample_rate = SAMPLE_RATE
        self.voices = VoiceStore(voices_dir)
        self.lock = threading.Lock()
        self.conditioning: dict[str, tuple[Any, Any]] = {}
        self.sessions: dict[str, dict[str, Any]] = {}

        if str(upstream_dir) not in sys.path:
            sys.path.insert(0, str(upstream_dir))
        factory = OrtSessionFactory(mode)
        print(
            f"Loading {MODEL_ALIAS} revision={ONNX_REVISION} mode={mode} "
            f"model_dir={model_dir}",
            flush=True,
        )
        with factory.installed():
            module = importlib.import_module("xtts_streaming_pipeline")
            pipeline_type = module.StreamingTTSPipeline
            self.pipeline = pipeline_type(
                model_dir=str(model_dir),
                vocab_path=str(model_dir / "vocab.json"),
                mel_norms_path=str(model_dir / "mel_stats.npy"),
                use_int8_gpt=False,
                num_threads_gpt=threads,
            )
        self.provider = factory.provider
        self.dawn_backend = factory.dawn_backend
        self.adapter = factory.adapter

        default_voice = self.voices.resolve("de_sample")
        self._conditioning_for("de_sample", default_voice)
        warmup = SpeechRequest("Test.", "de_sample", "de", "pcm", 1.0)
        pcm, _ = self.synthesize(warmup)
        if len(pcm) < 2:
            raise RuntimeError("XTTS warmup returned no audio")
        self.sessions = factory.finish_profiles()
        if mode == "webgpu-vulkan":
            validate_webgpu_session_placement(self.sessions)
        print(
            f"XTTS ready provider={self.provider} dawn={self.dawn_backend or 'n/a'} "
            f"adapter={self.adapter.get('name', 'CPU')}",
            flush=True,
        )

    def _conditioning_for(self, voice_id: str, path: Path) -> tuple[Any, Any]:
        cached = self.conditioning.get(voice_id)
        if cached is None:
            cached = self.pipeline.get_conditioning_latents(str(path))
            self.conditioning[voice_id] = cached
        return cached

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        voice_path = self.voices.resolve(request.voice)
        with self.lock:
            conditioning, speaker = self._conditioning_for(request.voice, voice_path)
            chunks = list(
                self.pipeline.inference_stream(
                    text=request.text,
                    language=request.language,
                    gpt_cond_latent=conditioning,
                    speaker_embedding=speaker,
                    stream_chunk_size=20,
                    speed=request.speed,
                )
            )
        if not chunks:
            raise RuntimeError("model returned no audio chunks")
        pcm = pcm16_bytes(self.np.concatenate(chunks, axis=0))
        if request.response_format == "pcm":
            return pcm, "audio/pcm"
        return wav_bytes(pcm), "audio/wav"

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model": MODEL_ID,
            "alias": MODEL_ALIAS,
            "revision": ONNX_REVISION,
            "provider": self.provider,
            "dawn_backend": self.dawn_backend,
            "adapter": self.adapter,
            "sample_rate": self.sample_rate,
            "voices": self.voices.list_ids(),
            "sessions": self.sessions,
        }


class XTTSServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: XTTSRuntime) -> None:
        super().__init__(address, XTTSHandler)
        self.runtime = runtime


class XTTSHandler(BaseHTTPRequestHandler):
    server: XTTSServer

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
            self.send_json(HTTPStatus.OK, self.server.runtime.health())
            return
        if path == "/v1/models":
            self.send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {"id": MODEL_ID, "object": "model", "owned_by": "coqui"}
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
            length = int(self.headers.get("Content-Length", ""))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError(
                    f"Content-Length must be between 1 and {MAX_REQUEST_BYTES}"
                )
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
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
                {"error": f"XTTS-v2 synthesis failed: {error}"},
            )
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Audio-Sample-Rate", str(SAMPLE_RATE))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, template: str, *args: Any) -> None:
        sys.stderr.write(f"[xtts-v2] {self.address_string()} " + (template % args) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible XTTS-v2 ONNX WebGPU/CPU sidecar"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--voices-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("webgpu-vulkan", "cpu"),
        default="cpu",
    )
    parser.add_argument("--default-language", default="de")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if not 0 <= args.threads <= 256:
        raise SystemExit("--threads must be between 0 and 256")
    model_dir = validate_model_dir(args.model_dir)
    voices_dir = args.voices_dir.expanduser().resolve(strict=True)
    upstream_dir = validate_upstream_dir(args.upstream_dir)
    runtime = XTTSRuntime(
        model_dir,
        voices_dir,
        upstream_dir,
        args.mode,
        args.default_language,
        args.threads,
    )
    server = XTTSServer((args.host, args.port), runtime)
    print(f"Listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
