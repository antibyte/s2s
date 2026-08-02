import importlib.util
import io
import pathlib
import sys
import tempfile
import unittest
import wave


MODULE_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "tts_inflect_server.py"
SPEC = importlib.util.spec_from_file_location("tts_inflect_server", MODULE_PATH)
SERVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


class InflectRequestTests(unittest.TestCase):
    def test_language_accepts_english_aliases(self):
        self.assertEqual(SERVER.normalize_language("en"), "en")
        self.assertEqual(SERVER.normalize_language("English"), "en")
        self.assertEqual(SERVER.normalize_language("en-US"), "en")
        self.assertEqual(SERVER.normalize_language("auto"), "en")

    def test_non_english_language_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "English-only"):
            SERVER.normalize_language("de")

    def test_openai_request_defaults_to_wav_and_builtin_voice(self):
        request = SERVER.SpeechRequest.from_payload(
            {"input": "Hello world.", "language": "en"}
        )
        self.assertEqual(request.text, "Hello world.")
        self.assertEqual(request.language, "en")
        self.assertEqual(request.response_format, "wav")
        self.assertEqual(request.speed, 1.0)
        self.assertAlmostEqual(request.variation, 0.667)
        self.assertEqual(request.seed, 0)

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            SERVER.SpeechRequest.from_payload({"input": "  "})

    def test_request_rejects_external_voice_paths(self):
        with self.assertRaisesRegex(ValueError, "built-in"):
            SERVER.SpeechRequest.from_payload(
                {
                    "input": "Hello.",
                    "voice": r"C:\outside\reference.wav",
                }
            )

    def test_speed_is_clamped_to_public_range(self):
        with self.assertRaisesRegex(ValueError, "speed"):
            SERVER.SpeechRequest.from_payload({"input": "Hello.", "speed": 3.0})
        with self.assertRaisesRegex(ValueError, "speed"):
            SERVER.SpeechRequest.from_payload({"input": "Hello.", "speed": 0.1})

    def test_seed_and_variation_bounds(self):
        request = SERVER.SpeechRequest.from_payload(
            {"input": "Hello.", "seed": 7, "variation": 0.0}
        )
        self.assertEqual(request.seed, 7)
        self.assertEqual(request.variation, 0.0)
        with self.assertRaisesRegex(ValueError, "variation"):
            SERVER.SpeechRequest.from_payload(
                {"input": "Hello.", "variation": 1.5}
            )


class InflectAudioTests(unittest.TestCase):
    def test_wav_encoding_is_mono_pcm16_at_24khz(self):
        pcm = b"\x00\x00\xff\x7f\x00\x80"
        encoded = SERVER.wav_bytes(pcm, 24000)
        with wave.open(io.BytesIO(encoded), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 24000)
            self.assertEqual(audio.readframes(audio.getnframes()), pcm)

    def test_model_directory_requires_pinned_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = pathlib.Path(directory)
            for name in SERVER.REQUIRED_ARTIFACTS:
                (model_dir / name).write_bytes(b"test")
            # Skip weight size/sha when files are stubs.
            self.assertEqual(
                SERVER.validate_model_dir(model_dir, verify_weights=False),
                model_dir.resolve(),
            )
            (model_dir / "model.pth").unlink()
            with self.assertRaisesRegex(ValueError, "model.pth"):
                SERVER.validate_model_dir(model_dir, verify_weights=False)


if __name__ == "__main__":
    unittest.main()
