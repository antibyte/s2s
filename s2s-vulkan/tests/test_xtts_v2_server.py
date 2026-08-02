import importlib.util
import io
import pathlib
import sys
import tempfile
import unittest
import wave
from types import SimpleNamespace


MODULE_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "tts_xtts_v2_server.py"
SPEC = importlib.util.spec_from_file_location("tts_xtts_v2_server", MODULE_PATH)
SERVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


def write_wav(path: pathlib.Path, *, frames: bytes = b"\0\0" * 240) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(frames)


class XTTSRequestTests(unittest.TestCase):
    def test_openai_german_payload_defaults(self):
        request = SERVER.SpeechRequest.from_payload(
            {
                "model": "coqui/XTTS-v2",
                "input": "Hallo Welt.",
                "language": "German",
            },
            "en",
        )
        self.assertEqual(request.language, "de")
        self.assertEqual(request.voice, "de_sample")
        self.assertEqual(request.response_format, "wav")
        self.assertEqual(request.speed, 1.0)

    def test_language_regions_and_chinese_alias_are_normalized(self):
        self.assertEqual(SERVER.normalize_language("zh"), "zh-cn")
        self.assertEqual(SERVER.normalize_language("pt"), "pt")
        with self.assertRaisesRegex(ValueError, "unsupported language"):
            SERVER.normalize_language("pt-BR")

    def test_payload_rejects_paths_as_voice_ids(self):
        for voice in (r"..\outside", "../outside", r"C:\voice", "voice.wav"):
            with self.subTest(voice=voice):
                with self.assertRaisesRegex(ValueError, "safe installed voice id"):
                    SERVER.SpeechRequest.from_payload(
                        {"input": "Hallo.", "voice": voice},
                        "de",
                    )

    def test_payload_rejects_unknown_model_and_invalid_speed(self):
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            SERVER.SpeechRequest.from_payload(
                {"model": "other", "input": "Hallo."},
                "de",
            )
        with self.assertRaisesRegex(ValueError, "speed"):
            SERVER.SpeechRequest.from_payload(
                {"input": "Hallo.", "speed": float("nan")},
                "de",
            )


class XTTSVoiceTests(unittest.TestCase):
    def test_voice_store_lists_only_safe_wav_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            write_wav(root / "de_sample.wav")
            write_wav(root / "my-voice_2.wav")
            (root / "not-a-voice.txt").write_text("x", encoding="utf-8")
            store = SERVER.VoiceStore(root)
            self.assertEqual(store.list_ids(), ["de_sample", "my-voice_2"])
            self.assertEqual(store.resolve("de_sample"), root / "de_sample.wav")

    def test_voice_store_rejects_traversal_missing_and_invalid_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "broken.wav").write_bytes(b"not wav")
            store = SERVER.VoiceStore(root)
            with self.assertRaisesRegex(ValueError, "safe installed voice id"):
                store.resolve("../outside")
            with self.assertRaisesRegex(ValueError, "not installed"):
                store.resolve("missing")
            with self.assertRaisesRegex(ValueError, "valid WAV"):
                store.resolve("broken")


class XTTSAudioTests(unittest.TestCase):
    def test_wav_encoding_is_mono_pcm16_at_24khz(self):
        pcm = b"\0\0\xff\x7f\0\x80"
        encoded = SERVER.wav_bytes(pcm)
        with wave.open(io.BytesIO(encoded), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 24_000)
            self.assertEqual(audio.readframes(audio.getnframes()), pcm)

    def test_pcm_rejects_empty_nan_and_excessive_audio(self):
        import numpy as np

        with self.assertRaisesRegex(RuntimeError, "empty or non-finite"):
            SERVER.pcm16_bytes([])
        with self.assertRaisesRegex(RuntimeError, "empty or non-finite"):
            SERVER.pcm16_bytes([np.nan])
        with self.assertRaisesRegex(RuntimeError, "120 seconds"):
            SERVER.pcm16_bytes(np.zeros(SERVER.SAMPLE_RATE * 120 + 1))


class XTTSProviderTests(unittest.TestCase):
    @staticmethod
    def device(
        name="Intel Arc B580",
        vendor_id=0x8086,
        device_id=0xE20B,
        metadata=None,
    ):
        return SimpleNamespace(
            ep_name="WebGpuExecutionProvider",
            hardware_device=SimpleNamespace(
                name=name,
                vendor_id=vendor_id,
                device_id=device_id,
            ),
            ep_metadata=metadata or {"backend": "Vulkan"},
        )

    def test_arc_b580_vulkan_adapter_is_selected_and_reported(self):
        device, report = SERVER.select_arc_b580_device(
            [self.device()],
            "WebGpuExecutionProvider",
        )
        self.assertEqual(device.hardware_device.name, "Intel Arc B580")
        self.assertEqual(report["vendor_id"], "0x8086")
        self.assertEqual(report["device_id"], "0xe20b")

    def test_d3d12_and_non_b580_devices_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Intel Arc B580"):
            SERVER.select_arc_b580_device(
                [self.device(metadata={"backend": "D3D12"})],
                "WebGpuExecutionProvider",
            )
        with self.assertRaisesRegex(RuntimeError, "Intel Arc B580"):
            SERVER.select_arc_b580_device(
                [self.device(name="Intel Arc A770")],
                "WebGpuExecutionProvider",
            )

    def test_all_sessions_share_required_vulkan_options(self):
        self.assertEqual(SERVER.WEBGPU_PROVIDER_OPTIONS["dawnBackendType"], "Vulkan")
        self.assertEqual(
            SERVER.WEBGPU_PROVIDER_OPTIONS["powerPreference"],
            "high-performance",
        )
        self.assertEqual(SERVER.WEBGPU_PROVIDER_OPTIONS["enableGraphCapture"], "0")
        self.assertEqual(SERVER.WEBGPU_PROVIDER_OPTIONS["deviceId"], "0")

    def test_cpu_only_gpt_or_hifigan_cannot_report_vulkan_healthy(self):
        healthy = {
            "gpt": {"webgpu_nodes": 7, "cpu_nodes": 2},
            "hifigan": {"webgpu_nodes": 5, "cpu_nodes": 0},
        }
        SERVER.validate_webgpu_session_placement(healthy)
        for role in ("gpt", "hifigan"):
            broken = {name: dict(value) for name, value in healthy.items()}
            broken[role]["webgpu_nodes"] = 0
            with self.subTest(role=role):
                with self.assertRaisesRegex(RuntimeError, "fell back fully to CPU"):
                    SERVER.validate_webgpu_session_placement(broken)


class XTTSArtifactTests(unittest.TestCase):
    def test_model_and_upstream_directories_require_all_pinned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = root / "onnx"
            upstream = root / "upstream"
            model.mkdir()
            upstream.mkdir()
            for name in SERVER.REQUIRED_MODEL_FILES:
                path = model / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            for name in SERVER.REQUIRED_UPSTREAM_FILES:
                (upstream / name).write_bytes(b"x")
            self.assertEqual(SERVER.validate_model_dir(model), model.resolve())
            self.assertEqual(
                SERVER.validate_upstream_dir(upstream),
                upstream.resolve(),
            )
            (model / "gpt_model.onnx").unlink()
            with self.assertRaisesRegex(ValueError, "gpt_model.onnx"):
                SERVER.validate_model_dir(model)


if __name__ == "__main__":
    unittest.main()
