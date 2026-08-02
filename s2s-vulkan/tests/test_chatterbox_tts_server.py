import importlib.util
import io
import pathlib
import sys
import tempfile
import unittest
import wave


MODULE_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "tts_chatterbox_server.py"
SPEC = importlib.util.spec_from_file_location("tts_chatterbox_server", MODULE_PATH)
SERVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


class ChatterboxRequestTests(unittest.TestCase):
    def test_language_aliases_and_regions_are_normalized(self):
        self.assertEqual(SERVER.normalize_language("German"), "de")
        self.assertEqual(SERVER.normalize_language("pt-BR"), "pt")
        self.assertEqual(SERVER.normalize_language("auto", "fr"), "fr")

    def test_unsupported_language_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported language"):
            SERVER.normalize_language("xx")

    def test_openai_request_defaults_to_builtin_voice_and_wav(self):
        request = SERVER.SpeechRequest.from_payload(
            {"input": "Hallo Welt.", "language": "de"},
            "en",
        )
        self.assertEqual(request.text, "Hallo Welt.")
        self.assertEqual(request.language, "de")
        self.assertEqual(request.response_format, "wav")
        self.assertEqual(request.exaggeration, 0.5)

    def test_request_rejects_paths_as_voice_ids(self):
        with self.assertRaisesRegex(ValueError, "built-in voice"):
            SERVER.SpeechRequest.from_payload(
                {
                    "input": "Hallo.",
                    "voice": r"C:\outside\reference.wav",
                },
                "de",
            )

    def test_generation_controls_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            SERVER.SpeechRequest.from_payload(
                {"input": "Hallo.", "temperature": float("nan")},
                "de",
            )


class ChatterboxAudioTests(unittest.TestCase):
    def test_wav_encoding_is_mono_pcm16_at_native_rate(self):
        pcm = b"\x00\x00\xff\x7f\x00\x80"
        encoded = SERVER.wav_bytes(pcm, 24000)
        with wave.open(io.BytesIO(encoded), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 24000)
            self.assertEqual(audio.readframes(audio.getnframes()), pcm)

    def test_model_directory_requires_every_pinned_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = pathlib.Path(directory)
            for name in SERVER.REQUIRED_ARTIFACTS:
                (model_dir / name).write_bytes(b"test")
            self.assertEqual(SERVER.validate_model_dir(model_dir), model_dir.resolve())
            (model_dir / SERVER.T3_MODEL).unlink()
            with self.assertRaisesRegex(ValueError, SERVER.T3_MODEL):
                SERVER.validate_model_dir(model_dir)


if __name__ == "__main__":
    unittest.main()
