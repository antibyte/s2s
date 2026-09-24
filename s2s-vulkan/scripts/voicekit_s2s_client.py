#!/usr/bin/env python3
"""AIY Voice Kit tap-to-talk client for the s2s Speech Lab WebSocket.

Button press starts capture+stream. Local VAD ends the turn after 1 s of
silence (or a second press). LED:
  solid     recording
  blink     waiting for the lab reply
  fast 2 s  error or timeout
  off       idle / reply finished
"""
from __future__ import annotations

import array
import asyncio
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time

import websockets
from gpiozero import Button, LED

WS_URL = os.environ.get("S2S_WS_URL", "ws://127.0.0.1:8765/ws")
CAPTURE_DEV = os.environ.get("S2S_CAPTURE_DEVICE", "capture")
PLAY_DEV = os.environ.get("S2S_PLAY_DEVICE", "playback")
SAMPLE_RATE = 16000
# 20 ms frames: WebRTC VAD requires 10/20/30 ms; also fine for the lab WS.
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
SILENCE_MS = 800
END_SILENCE_MS = int(os.environ.get("S2S_END_SILENCE_MS", "1000"))
NO_SPEECH_TIMEOUT_MS = int(os.environ.get("S2S_NO_SPEECH_TIMEOUT_MS", "8000"))
MIN_SPEECH_MS = 80
BUTTON_GPIO = 23
LED_GPIO = 25
GAIN = float(os.environ.get("S2S_MIC_GAIN", "1"))
PROCESS_TIMEOUT = float(os.environ.get("S2S_PROCESS_TIMEOUT_SEC", "45"))
# Lab used to swallow empty STT and reopen the mic; fail the LED instead of blinking 45 s.
FIRST_EVENT_TIMEOUT = float(os.environ.get("S2S_FIRST_EVENT_SEC", "8"))
# Gateway JSON `response_done` races ahead of binary PCM (separate WS tasks).
# Wait for in-flight frames after the event; shorten once TTS metrics say we
# have the full clip. Do not idle between LLM sentences — that gap is ~1 s.
PCM_IDLE_AFTER_DONE = float(os.environ.get("S2S_PCM_IDLE_SEC", "1.2"))
PCM_IDLE_WHEN_COMPLETE = 0.2
TRAIL_SILENCE_MS = 300
# Start speaker after this much audio is queued — low latency, enough to survive jitter.
PREBUFFER_MS = 280
PREBUFFER_BYTES = SAMPLE_RATE * 2 * PREBUFFER_MS // 1000
_PLAY_END = object()
_PLAY_ABORT = object()


def boost_s16(frame: bytes, gain: float) -> bytes:
    samples = array.array("h")
    samples.frombytes(frame)
    out = array.array("h")
    for x in samples:
        y = int(x * gain)
        if y > 32767:
            y = 32767
        elif y < -32768:
            y = -32768
        out.append(y)
    return out.tobytes()


try:
    import webrtcvad  # type: ignore

    _WEBRTC = webrtcvad.Vad(int(os.environ.get("S2S_WEBRTC_AGGR", "2")))
except Exception:
    _WEBRTC = None


class TurnVad:
    """Start-on-speech, stop after END_SILENCE_MS of non-voice.

    Prefers WebRTC VAD when the C extension is present; otherwise adaptive
    energy VAD (fits Pi Zero W).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.heard_speech = False
        self.speech_ms = 0
        self.silence_ms = 0
        self.total_ms = 0
        self.noise_rms = 1800.0

    def push(self, frame: bytes, rms: float) -> str | None:
        """Return 'end' when the turn should stop, else None."""
        self.total_ms += FRAME_MS
        voiced = self._voiced(frame, rms)
        if not self.heard_speech:
            if rms > 200:
                self.noise_rms = 0.9 * self.noise_rms + 0.1 * rms
            if voiced:
                self.speech_ms += FRAME_MS
                if self.speech_ms >= MIN_SPEECH_MS:
                    self.heard_speech = True
                    self.silence_ms = 0
            else:
                self.speech_ms = 0
            if self.total_ms >= NO_SPEECH_TIMEOUT_MS:
                return "nospeech"
            return None
        if voiced:
            self.silence_ms = 0
            self.speech_ms += FRAME_MS
        else:
            self.silence_ms += FRAME_MS
            if self.silence_ms >= END_SILENCE_MS:
                return "end"
        return None

    def _voiced(self, frame: bytes, rms: float) -> bool:
        energy = rms >= max(self.noise_rms * 1.7, 2500.0)
        if _WEBRTC is None:
            return energy
        try:
            return bool(_WEBRTC.is_speech(frame, SAMPLE_RATE)) or energy
        except Exception:
            return energy


def rms_s16(frame: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(frame)
    if not samples:
        return 0.0
    acc = 0
    for x in samples:
        acc += x * x
    return math.sqrt(acc / len(samples))


class VoiceKitClient:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws = None
        self.phase = "idle"  # idle | talking | process | playing | error
        self.talking = False
        self.press_t = 0.0
        self.frames_sent = 0
        self.peak_rms = 0.0
        self.got_pcm = False
        self.pcm_bytes = 0
        self.expected_pcm_bytes = 0
        self.awaiting_end = False
        self._pcm_ended = False
        self.capture_proc: subprocess.Popen | None = None
        self.play_proc: subprocess.Popen | None = None
        self.play_lock = threading.Lock()
        self.play_q: queue.Queue = queue.Queue(maxsize=1024)
        self.play_thread: threading.Thread | None = None
        self.play_done = threading.Event()
        self.play_done.set()
        self.send_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.led = LED(LED_GPIO)
        self.button = Button(BUTTON_GPIO, pull_up=True, bounce_time=0.05)
        self.stop = threading.Event()
        self.connected = threading.Event()
        self.timeout_handle: asyncio.TimerHandle | None = None
        self.first_event_handle: asyncio.TimerHandle | None = None
        self.pcm_idle_handle: asyncio.TimerHandle | None = None
        self.error_handle: asyncio.TimerHandle | None = None
        self.vad = TurnVad()
        self._ending = False

    def log(self, *parts) -> None:
        print(time.strftime("%H:%M:%S"), *parts, flush=True)

    def _cancel_timer(self, handle_name: str) -> None:
        handle = getattr(self, handle_name)
        if handle is not None:
            handle.cancel()
            setattr(self, handle_name, None)

    def set_led_mode(self, mode: str) -> None:
        try:
            if mode == "on":
                self.led.on()
            elif mode == "off":
                self.led.off()
            elif mode == "process":
                self.led.blink(on_time=0.22, off_time=0.22, background=True)
            elif mode == "error":
                # ~8 Hz for 2 seconds, then gpiozero leaves the LED off.
                self.led.blink(
                    on_time=0.06,
                    off_time=0.06,
                    n=16,
                    background=True,
                )
        except Exception as exc:
            self.log("led error", exc)

    def begin_process(self) -> None:
        self.phase = "process"
        self.got_pcm = False
        self.pcm_bytes = 0
        self.expected_pcm_bytes = 0
        self.awaiting_end = False
        self._pcm_ended = False
        self.set_led_mode("process")
        self._cancel_timer("timeout_handle")
        self._cancel_timer("first_event_handle")
        self._cancel_timer("pcm_idle_handle")
        loop = self.loop
        if loop:
            self.timeout_handle = loop.call_later(PROCESS_TIMEOUT, self._on_timeout)
            self.first_event_handle = loop.call_later(
                FIRST_EVENT_TIMEOUT, self._on_first_event
            )

    def finish_process(self) -> None:
        self._cancel_timer("timeout_handle")
        self._cancel_timer("first_event_handle")
        self._cancel_timer("pcm_idle_handle")
        if self.phase in ("process", "playing"):
            self.phase = "idle"
        self.set_led_mode("off")

    def signal_error(self, reason: str) -> None:
        self.log("error", reason)
        self._cancel_timer("timeout_handle")
        self._cancel_timer("first_event_handle")
        self._cancel_timer("pcm_idle_handle")
        self.stop_playback()
        self.phase = "error"
        self.set_led_mode("error")
        loop = self.loop
        if loop:
            self._cancel_timer("error_handle")
            self.error_handle = loop.call_later(2.05, self._error_done)

    def _on_timeout(self) -> None:
        if self.phase == "process":
            self.signal_error("timeout")

    def _on_first_event(self) -> None:
        if self.phase == "process" and not self.got_pcm:
            self.signal_error("no reply")

    def _note_lab_event(self) -> None:
        self._cancel_timer("first_event_handle")

    def _error_done(self) -> None:
        if self.phase == "error":
            self.phase = "idle"
            self.set_led_mode("off")

    def _enqueue(self, data: bytes | None) -> None:
        loop = self.loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self.send_q.put_nowait, data)

    def request_stop(self) -> None:
        self.stop.set()
        self._enqueue(None)
        loop = self.loop
        ws = self.ws
        if loop is not None and ws is not None:
            loop.call_soon_threadsafe(lambda: asyncio.create_task(ws.close()))

    def ensure_capture(self) -> None:
        if self.capture_proc and self.capture_proc.poll() is None:
            return
        err = open("/tmp/voicekit-arecord.err", "ab")
        self.capture_proc = subprocess.Popen(
            [
                "arecord",
                "-D",
                CAPTURE_DEV,
                "-f",
                "S16_LE",
                "-r",
                str(SAMPLE_RATE),
                "-c",
                "1",
                "-t",
                "raw",
                "--buffer-size=2048",
            ],
            stdout=subprocess.PIPE,
            stderr=err,
        )
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop_capture(self) -> None:
        proc = self.capture_proc
        self.capture_proc = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _capture_loop(self) -> None:
        proc = self.capture_proc
        if not proc or not proc.stdout:
            return
        buf = b""
        while proc is self.capture_proc and proc.poll() is None:
            chunk = proc.stdout.read(FRAME_BYTES)
            if not chunk:
                break
            if not self.talking:
                continue
            buf += chunk
            while len(buf) >= FRAME_BYTES:
                raw, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
                boosted = boost_s16(raw, GAIN)
                level = rms_s16(boosted)
                if level > self.peak_rms:
                    self.peak_rms = level
                self.frames_sent += 1
                self._enqueue(boosted)
                decision = self.vad.push(boosted, level)
                if decision:
                    self._request_end_recording(decision)
        if proc is self.capture_proc:
            self.log("arecord stopped", proc.poll())

    def flush_silence(self) -> None:
        zeros = b"\x00" * FRAME_BYTES
        for _ in range(SILENCE_MS // FRAME_MS):
            self._enqueue(zeros)

    def stop_playback(self) -> None:
        while True:
            try:
                self.play_q.get_nowait()
            except queue.Empty:
                break
        if self.play_thread and self.play_thread.is_alive():
            self.play_q.put_nowait(_PLAY_ABORT)
        with self.play_lock:
            proc = self.play_proc
            self.play_proc = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=2)
        self.play_done.set()

    def _ensure_player(self) -> None:
        if self.play_thread and self.play_thread.is_alive():
            return
        self.play_done.clear()
        self.play_thread = threading.Thread(target=self._player_loop, daemon=True)
        self.play_thread.start()

    def _feed_pcm(self, data: bytes) -> None:
        if self._pcm_ended:
            self._pcm_ended = False
            self.awaiting_end = True
            if self.phase == "idle":
                self.phase = "playing"
                self.set_led_mode("on")
            self.log(f"late pcm {len(data)}B, resume")
        self.pcm_bytes += len(data)
        self._ensure_player()
        try:
            self.play_q.put_nowait(data)
        except queue.Full:
            self.log("play queue full, drop chunk")
        if self.awaiting_end:
            self._maybe_finish_pcm()

    def _end_pcm(self) -> None:
        if self._pcm_ended:
            return
        self._pcm_ended = True
        self.awaiting_end = False
        self._cancel_timer("pcm_idle_handle")
        self._ensure_player()
        trail = b"\x00" * (SAMPLE_RATE * 2 * TRAIL_SILENCE_MS // 1000)
        try:
            self.play_q.put_nowait(trail)
            self.play_q.put_nowait(_PLAY_END)
        except queue.Full:
            self.log("play queue full on end")
            try:
                self.play_q.put_nowait(_PLAY_END)
            except queue.Full:
                pass

    def _arm_pcm_idle(self, delay: float) -> None:
        self._cancel_timer("pcm_idle_handle")
        loop = self.loop
        if loop:
            self.pcm_idle_handle = loop.call_later(delay, self._on_pcm_idle)

    def _on_pcm_idle(self) -> None:
        self.pcm_idle_handle = None
        if self.awaiting_end and not self._pcm_ended:
            self.log(
                f"pcm idle recv={self.pcm_bytes / (SAMPLE_RATE * 2):.2f}s "
                f"expect={self.expected_pcm_bytes / (SAMPLE_RATE * 2):.2f}s"
            )
            self._end_pcm()

    def _maybe_finish_pcm(self) -> None:
        if not self.awaiting_end or self._pcm_ended:
            return
        slack = FRAME_BYTES * 2
        if (
            self.expected_pcm_bytes > 0
            and self.pcm_bytes + slack >= self.expected_pcm_bytes
        ):
            self._arm_pcm_idle(PCM_IDLE_WHEN_COMPLETE)
        else:
            self._arm_pcm_idle(PCM_IDLE_AFTER_DONE)

    def _open_aplay(self) -> subprocess.Popen:
        proc = subprocess.Popen(
            [
                "aplay",
                "-q",
                "-D",
                PLAY_DEV,
                "-f",
                "S16_LE",
                "-r",
                str(SAMPLE_RATE),
                "-c",
                "1",
                "-t",
                "raw",
                "--buffer-time=800000",
                "--period-time=40000",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        with self.play_lock:
            self.play_proc = proc
        return proc

    def _write_aplay(self, proc: subprocess.Popen, data: bytes) -> subprocess.Popen:
        try:
            assert proc.stdin is not None
            proc.stdin.write(data)
            return proc
        except BrokenPipeError:
            self.log("aplay underrun, restart")
            proc = self._open_aplay()
            try:
                assert proc.stdin is not None
                proc.stdin.write(data)
            except BrokenPipeError:
                pass
            return proc

    def _player_loop(self) -> None:
        pending = bytearray()
        proc: subprocess.Popen | None = None
        started = False
        played = 0
        try:
            while True:
                item = self.play_q.get()
                if item is _PLAY_ABORT:
                    break
                if item is _PLAY_END:
                    if pending and proc is None:
                        proc = self._open_aplay()
                        started = True
                        self._on_playback_started()
                    if pending and proc is not None:
                        proc = self._write_aplay(proc, bytes(pending))
                        played += len(pending)
                        pending.clear()
                    if proc is not None and proc.stdin:
                        try:
                            proc.stdin.close()
                        except Exception:
                            pass
                    if proc is not None:
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    break
                pending.extend(item)
                if not started:
                    if len(pending) < PREBUFFER_BYTES:
                        continue
                    proc = self._open_aplay()
                    started = True
                    self._on_playback_started()
                    proc = self._write_aplay(proc, bytes(pending))
                    played += len(pending)
                    pending.clear()
                    continue
                if proc is None:
                    proc = self._open_aplay()
                proc = self._write_aplay(proc, bytes(pending))
                played += len(pending)
                pending.clear()
        finally:
            with self.play_lock:
                leftover = self.play_proc
                self.play_proc = None
            if leftover and leftover.poll() is None:
                leftover.terminate()
                try:
                    leftover.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    leftover.kill()
            self.log(
                f"play done wrote={played / (SAMPLE_RATE * 2):.1f}s "
                f"recv={self.pcm_bytes / (SAMPLE_RATE * 2):.1f}s "
                f"expect={self.expected_pcm_bytes / (SAMPLE_RATE * 2):.1f}s"
            )
            self.play_done.set()
            loop = self.loop
            if loop and self.phase == "playing":
                loop.call_soon_threadsafe(self.finish_process)

    def _on_playback_started(self) -> None:
        loop = self.loop
        if loop:
            loop.call_soon_threadsafe(self._mark_playing)

    def _mark_playing(self) -> None:
        if self.phase == "process":
            self._cancel_timer("timeout_handle")
            self.phase = "playing"
            self.set_led_mode("on")

    def on_press(self) -> None:
        if not self.connected.is_set():
            return
        if self.talking:
            self._request_end_recording("button")
            return
        self.stop_playback()
        self._cancel_timer("timeout_handle")
        self._cancel_timer("first_event_handle")
        self._cancel_timer("pcm_idle_handle")
        self._cancel_timer("error_handle")
        self.vad.reset()
        self._ending = False
        self.talking = True
        self.phase = "talking"
        self.press_t = time.monotonic()
        self.frames_sent = 0
        self.peak_rms = 0.0
        self.got_pcm = False
        self.set_led_mode("on")
        backend = "webrtc" if _WEBRTC else "energy"
        self.log("record start", backend)
        self.ensure_capture()

    def on_release(self) -> None:
        return

    def _request_end_recording(self, reason: str) -> None:
        loop = self.loop
        if loop is None:
            self._finish_recording(reason)
            return
        loop.call_soon_threadsafe(self._finish_recording, reason)

    def _finish_recording(self, reason: str) -> None:
        if not self.talking or self._ending:
            return
        self._ending = True
        self.talking = False
        self.log(
            "record end",
            reason,
            f"frames={self.frames_sent}",
            f"peak_rms={self.peak_rms:.0f}",
            f"heard={self.vad.heard_speech}",
        )
        if reason == "nospeech":
            self.flush_silence()
            self.signal_error("no speech")
            return
        self.flush_silence()
        self.begin_process()

    async def sender(self) -> None:
        while True:
            item = await self.send_q.get()
            if item is None:
                return
            ws = self.ws
            if ws is None:
                continue
            try:
                await ws.send(item)
            except Exception as exc:
                self.log("send error", exc)
                self.signal_error("send failed")
                return

    async def receiver(self) -> None:
        ws = self.ws
        assert ws is not None
        async for message in ws:
            if isinstance(message, (bytes, bytearray, memoryview)):
                data = bytes(message)
                if not data:
                    continue
                if self.phase not in ("process", "playing"):
                    if not (
                        self.phase == "idle"
                        and (self.awaiting_end or not self.play_done.is_set())
                    ):
                        continue
                    self.phase = "playing"
                    self.set_led_mode("on")
                self._note_lab_event()
                if not self.got_pcm:
                    self.log(f"← pcm start {len(data)}B")
                self.got_pcm = True
                self._feed_pcm(data)
                continue
            if not isinstance(message, str):
                continue
            text = message.strip()
            if self.phase in ("process", "playing"):
                self._note_lab_event()
            self._handle_event(text)
            if len(text) > 200:
                text = text[:200] + "…"
            self.log("←", text)

    def _handle_event(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        kind = str(msg.get("type") or "")
        if kind == "error":
            stage = msg.get("stage") or "pipeline"
            message = msg.get("message") or "lab error"
            self.signal_error(f"{stage}: {message}")
            return
        if kind == "metrics" and str(msg.get("stage") or "") == "tts":
            values = msg.get("values") or {}
            dur = values.get("audio_duration_ms")
            if isinstance(dur, (int, float)) and dur > 0:
                self.expected_pcm_bytes += int(dur * SAMPLE_RATE / 1000.0 * 2)
                if self.awaiting_end:
                    self._maybe_finish_pcm()
            return
        if kind == "response_done":
            if not self.got_pcm:
                self.signal_error("empty reply")
                return
            self.awaiting_end = True
            self.log(
                f"response_done pcm={self.pcm_bytes / (SAMPLE_RATE * 2):.2f}s "
                f"expect={self.expected_pcm_bytes / (SAMPLE_RATE * 2):.2f}s"
            )
            self._maybe_finish_pcm()

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.button.when_pressed = self.on_press
        self.button.when_released = self.on_release
        backoff = 1.0
        while not self.stop.is_set():
            self.log("connecting to Speech Lab")
            try:
                async with websockets.connect(
                    WS_URL,
                    max_size=8 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                    open_timeout=10,
                ) as ws:
                    self.ws = ws
                    self.connected.set()
                    backoff = 1.0
                    self.log(
                        "connected — press button, speak, auto-stop after "
                        f"{END_SILENCE_MS}ms silence"
                    )
                    self.set_led_mode("off")
                    self.ensure_capture()
                    sender = asyncio.create_task(self.sender())
                    try:
                        await self.receiver()
                    finally:
                        sender.cancel()
            except Exception as exc:
                self.log("ws error", type(exc).__name__)
                self.signal_error("websocket")
            finally:
                self.connected.clear()
                self.ws = None
                self.talking = False
                self.stop_capture()
                self.stop_playback()
                self._cancel_timer("timeout_handle")
                self._cancel_timer("first_event_handle")
                self._cancel_timer("pcm_idle_handle")
                if self.phase != "error":
                    self.phase = "idle"
                    self.set_led_mode("off")
            if self.stop.is_set():
                break
            self.log("reconnect in", backoff, "s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 20.0)


def init_mixer() -> None:
    subprocess.run(
        [
            "arecord",
            "-D",
            CAPTURE_DEV,
            "-d",
            "1",
            "-f",
            "S16_LE",
            "-r",
            str(SAMPLE_RATE),
            "-c",
            "1",
            "-t",
            "raw",
            "/dev/null",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
    )
    for args in (
        ["amixer", "-q", "set", "Mic", "100%"],
        ["amixer", "-q", "set", "Master", "80%"],
        ["amixer", "-q", "-c", "sndrpigooglevoi", "set", "Mic", "100%"],
        ["amixer", "-q", "-c", "sndrpigooglevoi", "set", "Master", "80%"],
    ):
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    init_mixer()
    client = VoiceKitClient()

    def handle_stop(*_args) -> None:
        client.request_stop()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass
    finally:
        client.stop_capture()
        client.stop_playback()
        client.set_led_mode("off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
