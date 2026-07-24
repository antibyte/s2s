import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "parakeet_asr_server.py"
SPEC = importlib.util.spec_from_file_location("parakeet_asr_server", MODULE_PATH)
SERVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


class Availability:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self, cuda=False, xpu=False):
        self.cuda = Availability(cuda)
        self.xpu = Availability(xpu)


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_prefers_cuda(self):
        self.assertEqual(SERVER.select_device(FakeTorch(True, True), "auto"), "cuda")

    def test_auto_uses_xpu_before_cpu(self):
        self.assertEqual(SERVER.select_device(FakeTorch(False, True), "auto"), "xpu")

    def test_auto_falls_back_to_cpu(self):
        self.assertEqual(SERVER.select_device(FakeTorch(False, False), "auto"), "cpu")

    def test_explicit_unavailable_device_fails(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            SERVER.select_device(FakeTorch(False, False), "cuda")

    def test_device_specific_default_dtypes(self):
        torch = FakeTorch()
        self.assertEqual(SERVER.select_dtype(torch, "cpu", "auto"), "float32")
        self.assertEqual(SERVER.select_dtype(torch, "cuda", "auto"), "float16")
        self.assertEqual(SERVER.select_dtype(torch, "xpu", "auto"), "bfloat16")

    def test_cpu_rejects_reduced_precision(self):
        with self.assertRaisesRegex(RuntimeError, "CPU inference requires"):
            SERVER.select_dtype(FakeTorch(), "cpu", "float16")

    def test_decode_normalization(self):
        self.assertEqual(SERVER.normalize_decoded([" Hallo. ", "Welt!"]), "Hallo. Welt!")


if __name__ == "__main__":
    unittest.main()
