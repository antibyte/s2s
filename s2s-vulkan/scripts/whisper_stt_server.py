#!/usr/bin/env python3
"""faster-whisper STT for s2s — quality-focused defaults + model hot-reload."""
from __future__ import annotations
import argparse, asyncio, os, sys, tempfile
from pathlib import Path


def should_drop_segment(
    avg_logprob: float | None, no_speech_prob: float | None
) -> bool:
    """Reject only clearly unreliable text.

    Tiny frequently assigns a high no-speech probability to short valid
    utterances. Whisper's decision rule combines that probability with a poor
    log probability, so a high no-speech value alone must not discard text.
    """
    if avg_logprob is not None and avg_logprob < -1.2:
        return True
    return (
        no_speech_prob is not None
        and no_speech_prob > 0.95
        and avg_logprob is not None
        and avg_logprob < -1.0
    )


def transcription_options(language: str | None) -> dict:
    """Return quality-focused options without seeding transcript text.

    A language-specific initial prompt can be emitted verbatim for low-signal
    audio, turning silence into a convincing but false transcription.
    """
    return {
        "language": language,
        "task": "transcribe",
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
        "temperature": 0.0,
        "vad_filter": True,
        "vad_parameters": {
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 200,
            "threshold": 0.5,
        },
        "condition_on_previous_text": False,
        "without_timestamps": True,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--model", default=os.environ.get("S2S_WHISPER_MODEL", "small"))
    p.add_argument("--device", default=os.environ.get("S2S_WHISPER_DEVICE", "cpu"))
    p.add_argument("--compute-type", default=os.environ.get("S2S_WHISPER_COMPUTE", "int8"))
    p.add_argument("--language", default=os.environ.get("S2S_WHISPER_LANGUAGE", "de"))
    args = p.parse_args()

    from aiohttp import web
    from faster_whisper import WhisperModel

    state = {
        "model_name": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "language": args.language,
        "model": None,
        "lock": asyncio.Lock(),
    }

    def load_model(name: str):
        print(
            f"Loading Whisper model={name} device={args.device} compute={args.compute_type}",
            file=sys.stderr,
            flush=True,
        )
        m = WhisperModel(name, device=args.device, compute_type=args.compute_type)
        state["model"] = m
        state["model_name"] = name
        print(f"Whisper ready (model={name}).", file=sys.stderr, flush=True)
        return m

    load_model(args.model)

    async def health(_):
        return web.json_response(
            {
                "ok": True,
                "model": state["model_name"],
                "engine": "faster-whisper",
                "device": state["device"],
                "compute_type": state["compute_type"],
            }
        )

    async def reload(request: web.Request):
        """Hot-swap Whisper weights: POST {"model":"base"|"small"|"tiny"|...}"""
        try:
            body = await request.json()
        except Exception:
            body = {}
        name = (body.get("model") or request.rel_url.query.get("model") or "").strip()
        if not name:
            return web.json_response({"error": "missing model"}, status=400)
        if name == state["model_name"] and state["model"] is not None:
            return web.json_response({"ok": True, "model": name, "reloaded": False})
        async with state["lock"]:
            try:
                # Load in thread so event loop stays responsive for health checks.
                await asyncio.to_thread(load_model, name)
            except Exception as e:
                print(f"reload failed: {e}", file=sys.stderr, flush=True)
                return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"ok": True, "model": name, "reloaded": True})

    async def inference(request: web.Request):
        reader = await request.multipart()
        audio_bytes = None
        language = state["language"]
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name in ("file", "audio", "wav"):
                audio_bytes = await part.read(decode=False)
            elif part.name == "language":
                language = (await part.text()).strip() or language
        if not audio_bytes:
            body = await request.read()
            if body:
                audio_bytes = body
        if not audio_bytes:
            return web.json_response({"error": "no audio"}, status=400)

        if len(audio_bytes) < 8000:
            print(f"STT skip tiny payload {len(audio_bytes)} bytes", file=sys.stderr, flush=True)
            return web.json_response({"text": ""})

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            lang = None if not language or language == "auto" else language

            def _transcribe():
                model = state["model"]
                segments, info = model.transcribe(
                    path,
                    **transcription_options(lang),
                )
                parts = []
                for s in segments:
                    t = (s.text or "").strip()
                    if not t:
                        continue
                    avg_logprob = getattr(s, "avg_logprob", None)
                    no_speech_prob = getattr(s, "no_speech_prob", None)
                    if should_drop_segment(avg_logprob, no_speech_prob):
                        print(
                            "  drop unreliable segment "
                            f"logprob={avg_logprob!r} no_speech={no_speech_prob!r}: {t}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    parts.append(t)
                return " ".join(parts).strip(), info

            async with state["lock"]:
                text, info = await asyncio.to_thread(_transcribe)
            bad = {
                "* musik *",
                "[musik]",
                "(musik)",
                "music",
                "♪",
                "...",
                "untertitel",
                "www.",
                "amara.org",
                "♪♪",
            }
            if text.lower() in bad or any(
                b in text.lower() for b in ("untertitel", "amara.org", "www.youtube")
            ):
                text = ""
            print(
                f"STT [{getattr(info,'language','?')} p={getattr(info,'language_probability',0):.2f} model={state['model_name']}]: {text!r}",
                file=sys.stderr,
                flush=True,
            )
            return web.json_response({"text": text, "model": state["model_name"]})
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    app = web.Application(client_max_size=32 * 1024 * 1024)
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_post("/reload", reload)
    app.router.add_post("/inference", inference)
    app.router.add_post("/v1/audio/transcriptions", inference)
    print(
        f"Listening http://{args.host}:{args.port}/inference engine=faster-whisper model={args.model} (hot-reload /reload)",
        file=sys.stderr,
        flush=True,
    )
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
