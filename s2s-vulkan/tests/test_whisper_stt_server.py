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

    def test_drops_low_probability_text(self):
        self.assertTrue(MODULE.should_drop_segment(-1.3, 0.1))

    def test_drops_combined_strong_silence_signal(self):
        self.assertTrue(MODULE.should_drop_segment(-1.1, 0.97))

    def test_transcription_options_do_not_seed_transcript_text(self):
        options = MODULE.transcription_options("de")
        self.assertEqual(options["language"], "de")
        self.assertTrue(options["vad_filter"])
        self.assertNotIn("initial_prompt", options)


if __name__ == "__main__":
    unittest.main()
