import importlib.util
import io
import pathlib
import sys
import tempfile
import unittest
import wave


MODULE_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "tts_audio8_server.py"
SPEC = importlib.util.spec_from_file_location("tts_audio8_server", MODULE_PATH)
SERVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


class Audio8RequestTests(unittest.TestCase):
    def test_language_accepts_german_and_aliases(self):
        self.assertEqual(SERVER.normalize_language("de"), "de")
        self.assertEqual(SERVER.normalize_language("German"), "de")
        self.assertEqual(SERVER.normalize_language("en-US"), "en")
        self.assertEqual(SERVER.normalize_language("chinese"), "zh")
        self.assertEqual(SERVER.normalize_language("auto", default="de"), "de")

    def test_unsupported_language_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported language"):
            SERVER.normalize_language("ru")

    def test_openai_request_defaults(self):
        request = SERVER.SpeechRequest.from_payload(
            {"input": "Hallo Welt.", "language": "de"},
            default_language="de",
        )
        self.assertEqual(request.text, "Hallo Welt.")
        self.assertEqual(request.language, "de")
        self.assertEqual(request.response_format, "wav")
        self.assertEqual(request.voice, "default")
        self.assertAlmostEqual(request.temperature, 0.8)
        self.assertEqual(request.max_new_tokens, 1024)

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            SERVER.SpeechRequest.from_payload(
                {"input": "  "}, default_language="de"
            )

    def test_unsafe_voice_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "voice must be"):
            SERVER.SpeechRequest.from_payload(
                {
                    "input": "Hallo.",
                    "voice": r"C:\outside\reference.wav",
                },
                default_language="de",
            )

    def test_sampling_bounds(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            SERVER.SpeechRequest.from_payload(
                {"input": "Hallo.", "temperature": 0.0},
                default_language="de",
            )
        with self.assertRaisesRegex(ValueError, "top_p"):
            SERVER.SpeechRequest.from_payload(
                {"input": "Hallo.", "top_p": 1.5},
                default_language="de",
            )

    def test_cli_defaults_remain_preview_06b(self):
        args = SERVER.parse_args([])
        self.assertEqual(args.model_id, "Audio8/Audio8-TTS-Preview-0.6b")
        self.assertEqual(args.model_alias, "audio8-tts-preview-0.6b")
        self.assertEqual(args.revision, "f9612f13a0ab40facf3d050fc908b9e6db05c2be")
        self.assertEqual(args.port, 8096)
        self.assertEqual(args.model_dir, "models/audio8-tts-preview-0.6b")

    def test_cli_accepts_preview_01b_identity(self):
        args = SERVER.parse_args(
            [
                "--model-id",
                "Audio8/Audio8-TTS-Preview-0.1b",
                "--model-alias",
                "audio8-tts-preview-0.1b",
                "--revision",
                "7a644014c398a0495d5efd1da7461bfeb4dbddcd",
                "--model-dir",
                "models/audio8-tts-preview-0.1b",
                "--port",
                "8096",
            ]
        )
        self.assertEqual(args.model_id, "Audio8/Audio8-TTS-Preview-0.1b")
        self.assertEqual(args.model_alias, "audio8-tts-preview-0.1b")
        self.assertEqual(
            args.revision, "7a644014c398a0495d5efd1da7461bfeb4dbddcd"
        )
        self.assertEqual(args.model_dir, "models/audio8-tts-preview-0.1b")


class Audio8AudioTests(unittest.TestCase):
    def test_wav_encoding_is_mono_pcm16_at_44100(self):
        pcm = b"\x00\x00\xff\x7f\x00\x80"
        encoded = SERVER.wav_bytes(pcm, 44100)
        with wave.open(io.BytesIO(encoded), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 44100)
            self.assertEqual(audio.readframes(audio.getnframes()), pcm)

    def test_model_directory_requires_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = pathlib.Path(directory)
            for name in SERVER.REQUIRED_ARTIFACTS:
                (model_dir / name).write_bytes(b"test")
            self.assertEqual(
                SERVER.validate_model_dir(model_dir),
                model_dir.resolve(),
            )
            (model_dir / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "model.safetensors"):
                SERVER.validate_model_dir(model_dir)

    def test_resolve_reference_requires_paired_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = pathlib.Path(directory)
            voices = model_dir / "voices"
            voices.mkdir()
            (voices / "demo.wav").write_bytes(b"RIFF")
            with self.assertRaisesRegex(ValueError, "transcript"):
                SERVER.resolve_reference(model_dir, "demo")
            (voices / "demo.txt").write_text("spoken words", encoding="utf-8")
            audio, text = SERVER.resolve_reference(model_dir, "demo")
            self.assertTrue(audio.endswith("demo.wav"))
            self.assertEqual(text, "spoken words")
            audio, text = SERVER.resolve_reference(model_dir, "default")
            self.assertIsNone(audio)
            self.assertIsNone(text)


if __name__ == "__main__":
    unittest.main()
