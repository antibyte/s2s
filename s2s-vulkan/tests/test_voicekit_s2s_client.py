"""Host-only state tests for the optional Voice Kit push-to-talk client."""

from __future__ import annotations

import asyncio
import importlib.util
import queue
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _LED:
    def __init__(self, *_args, **_kwargs):
        pass

    def on(self):
        pass

    def off(self):
        pass

    def blink(self, **_kwargs):
        pass


class _Button:
    def __init__(self, *_args, **_kwargs):
        self.when_pressed = None
        self.when_released = None


GPIOZERO = types.ModuleType("gpiozero")
GPIOZERO.LED = _LED
GPIOZERO.Button = _Button
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "voicekit_s2s_client.py"
SPEC = importlib.util.spec_from_file_location("voicekit_s2s_client_test", SCRIPT)
assert SPEC and SPEC.loader
CLIENT = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"gpiozero": GPIOZERO}):
    SPEC.loader.exec_module(CLIENT)


class _AliveThread:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float) -> None:
        pass


class VoiceKitStateTests(unittest.TestCase):
    def test_stop_playback_discards_old_audio_but_keeps_abort_for_live_player(self):
        client = CLIENT.VoiceKitClient()
        client.play_thread = _AliveThread()
        client.play_q.put_nowait(b"old reply")

        client.stop_playback()

        self.assertIs(client.play_q.get_nowait(), CLIENT._PLAY_ABORT)
        with self.assertRaises(queue.Empty):
            client.play_q.get_nowait()

    def test_stop_playback_without_player_leaves_no_abort_for_next_turn(self):
        client = CLIENT.VoiceKitClient()
        client.play_q.put_nowait(b"old reply")

        client.stop_playback()

        self.assertTrue(client.play_q.empty())

    def test_voice_activity_ends_after_silence(self):
        with patch.object(CLIENT, "_WEBRTC", None):
            vad = CLIENT.TurnVad()
            frame = b"\x00" * CLIENT.FRAME_BYTES
            for _ in range(CLIENT.MIN_SPEECH_MS // CLIENT.FRAME_MS):
                self.assertIsNone(vad.push(frame, 10000.0))
            self.assertTrue(vad.heard_speech)
            for _ in range(CLIENT.END_SILENCE_MS // CLIENT.FRAME_MS - 1):
                self.assertIsNone(vad.push(frame, 0.0))
            self.assertEqual(vad.push(frame, 0.0), "end")


class VoiceKitStopTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_closes_active_websocket(self):
        client = CLIENT.VoiceKitClient()
        client.loop = asyncio.get_running_loop()

        class Socket:
            closed = False

            async def close(self):
                self.closed = True

        socket = Socket()
        client.ws = socket
        client.request_stop()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertTrue(client.stop.is_set())
        self.assertTrue(socket.closed)
        self.assertIsNone(client.send_q.get_nowait())


if __name__ == "__main__":
    unittest.main()