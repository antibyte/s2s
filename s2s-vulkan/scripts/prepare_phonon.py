#!/usr/bin/env python3
"""Verify and unpack the pinned AuraGo Phonon release resources for an image build."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def verify(path, entry):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["size"]:
        raise ValueError(f"{path.name}: resource size mismatch")
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != entry["sha256"]:
            raise ValueError(f"{path.name}: SHA-256 mismatch")


def extract(package, destination, prefix):
    total = 0
    with tarfile.open(package, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.name.startswith(prefix) or ".." in name.parts or "\\" in member.name:
                raise ValueError("unsafe Phonon archive path")
            relative = PurePosixPath(member.name[len(prefix):])
            if not relative.parts or relative.is_absolute() or not member.isfile():
                raise ValueError("Phonon archives may contain only regular files")
            total += member.size
            if member.size > 128 * 1024**2 or total > 256 * 1024**2:
                raise ValueError("Phonon archive exceeds extraction limit")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def prepare(packages, output, manifest):
    if output.exists():
        raise ValueError("output directory already exists")
    for entry in manifest["packages"]:
        verify(packages / entry["name"], entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phonon-", dir=output.parent) as temporary:
        staged = Path(temporary) / "resources"
        staged.mkdir()
        prefix = f"phonon/{manifest['bundle_id']}/"
        for entry in manifest["packages"]:
            extract(packages / entry["name"], staged, prefix)
        for entry in manifest["model_files"]:
            verify(staged / "model" / entry["name"], entry)
        for entry in manifest["voices"]:
            verify(staged / entry["filename"], entry)
        verify(staged / manifest["runtime"]["name"], manifest["runtime"])
        (staged / manifest["runtime"]["name"]).chmod(0o755)
        os.replace(staged, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "config/phonon.json").read_text(encoding="utf-8"))
    prepare(args.packages, args.output, manifest)
    print(f"Verified Phonon {manifest['bundle_id']}: {len(manifest['voices'])} voices")


if __name__ == "__main__":
    main()
