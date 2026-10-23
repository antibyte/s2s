"""Pinned model, safe extraction and bundle contracts, without downloading weights."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_phonon
import speech_lab_bundle


class SpeechResourcesTest(unittest.TestCase):
    def test_manifest_and_runtime_matrix(self):
        catalog = json.loads((ROOT / "config/backends.json").read_text(encoding="utf-8"))
        bundle = json.loads((ROOT / "deploy/speech-lab-bundle.json").read_text(encoding="utf-8"))
        speech_lab_bundle.validate(bundle, catalog)
        backends = {item["id"]: item for item in catalog["backends"]}
        phonon = backends["phonon-40m-multilingual"]
        manifest = json.loads((ROOT / "config/phonon.json").read_text(encoding="utf-8"))
        self.assertEqual(set(phonon["voices"]), {v["id"] for v in manifest["voices"]})
        self.assertEqual(len(phonon["voices"]), 10)
        self.assertTrue(phonon["voice_labels"][phonon["default_voice"]].startswith("Nele"))
        self.assertEqual(phonon["voice_mode"], "request")
        self.assertTrue(phonon["bundled"])
        granite = backends["granite-speech-5.0-2b"]
        self.assertFalse(granite["bundled"])
        self.assertEqual(sum(a["size"] for a in granite["artifacts"]), 1_950_119_008)
        self.assertIn(".i1-Q4_K_M.gguf", granite["artifacts"][0]["path"])
        for artifact in granite["artifacts"]:
            self.assertEqual(len(artifact["sha256"]), 64)
            self.assertNotIn("/resolve/main/", artifact["source"])
        runtimes = [r for r in bundle["runtimes"] if r["backend_id"] in (granite["id"], phonon["id"])]
        self.assertEqual(len(runtimes), 4)
        for runtime in runtimes:
            self.assertNotIn("/opt/s2s-models", runtime["model_mounts"])
            self.assertEqual(runtime["architectures"], ["amd64"])
            self.assertIn(runtime["image_key"], bundle["images"])
        default = next(p for p in catalog["presets"] if p["id"] == "balanced")
        self.assertEqual((default["asr_id"], default["tts_id"]), ("confucius4-r2t2", "supertonic"))

    def test_corrupt_missing_and_repeated_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.dat"
            entry = {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            with self.assertRaises(ValueError):
                prepare_phonon.verify(target, entry)
            target.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                prepare_phonon.verify(target, entry)
            target.write_bytes(b"abc")
            prepare_phonon.verify(target, entry)
            with self.assertRaises(ValueError):
                prepare_phonon.prepare(root, root, {})
            self.assertEqual(target.read_bytes(), b"abc")

    def test_archives_reject_traversal_links_and_oversized_members(self):
        for name, kind, size in (("phonon/b/../escape", tarfile.REGTYPE, 1),
                                 ("elsewhere", tarfile.REGTYPE, 1),
                                 ("phonon/b/link", tarfile.SYMTYPE, 0),
                                 ("phonon/b/huge", tarfile.REGTYPE, 129 * 1024**2)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package = root / "package.tar.gz"
                with tarfile.open(package, "w:gz") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.size = size
                    # Header-only oversized members are rejected before reading.
                    if size > 1024:
                        archive.fileobj.write(member.tobuf())
                    else:
                        archive.addfile(member, io.BytesIO(b"x" * size))
                with self.assertRaises(ValueError):
                    prepare_phonon.extract(package, root / "output", "phonon/b/")
                self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
