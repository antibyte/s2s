import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "whisper_stt_server.py"
SPEC = importlib.util.spec_from_file_location("whisper_stt_server", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SegmentFilterTests(unittest.TestCase):
    def test_keeps_short_speech_with_high_no_speech_probability(self):
        self.assertFalse(MODULE.should_drop_segment(-0.4, 0.87))

    def test_keeps_tiny_voicekit_clip(self):
        # Real Voice HAT turns dropped by the old logprob < -1.2 rule.
        self.assertFalse(MODULE.should_drop_segment(-1.38, 0.12))
        self.assertFalse(MODULE.should_drop_segment(-1.27, 0.25))

    def test_drops_garbage_logprob(self):
        self.assertTrue(MODULE.should_drop_segment(-2.1, 0.2))

    def test_drops_combined_strong_silence_signal(self):
        self.assertTrue(MODULE.should_drop_segment(-1.1, 0.97))

    def test_transcription_options_do_not_seed_transcript_text(self):
        options = MODULE.transcription_options("de")
        self.assertEqual(options["language"], "de")
        self.assertFalse(options["vad_filter"])
        self.assertNotIn("initial_prompt", options)


if __name__ == "__main__":
    unittest.main()
