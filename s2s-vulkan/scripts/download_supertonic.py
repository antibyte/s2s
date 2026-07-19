#!/usr/bin/env python3
"""
Download Supertonic 3 ONNX assets + preset voices for in-process Rust TTS.

  python scripts/download_supertonic.py
  python scripts/download_supertonic.py --out models/supertonic

Requires: pip install huggingface_hub
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("models/supertonic"),
        help="Destination directory (will contain onnx/ and voice_styles/)",
    )
    p.add_argument("--repo", default="Supertone/supertonic-3")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import snapshot_download

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} → {args.out} …")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(args.out),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "onnx/*",
            "voice_styles/*",
            "*.json",
            "LICENSE*",
            "README*",
        ],
    )
    onnx = args.out / "onnx"
    voices = args.out / "voice_styles"
    print("Done.")
    print(f"  ONNX:   {onnx}  exists={onnx.is_dir()}")
    print(f"  Voices: {voices} exists={voices.is_dir()}")
    print()
    print("Run s2s with:")
    print(
        f"  s2s-vulkan --tts supertonic "
        f"--supertonic-model-dir {onnx} "
        f"--supertonic-voice M1 --tts-sample-rate 16000"
    )


if __name__ == "__main__":
    main()
