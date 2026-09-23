#!/usr/bin/env python3
"""Generate and validate the Speech Lab bundle-v2 runtime matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "backends.json"
BUNDLE_PATH = ROOT / "deploy" / "speech-lab-bundle.json"
DIGEST = "0" * 64

IMAGES = {
    "gateway": ("s2s-vulkan", 320_000_000),
    "controller": ("s2s-speech-lab-controller", 8_000_000),
    "whisper": ("s2s-whisper-fw", 2_400_000_000),
    "confucius_cpu": ("s2s-asr-confucius-cpu", 1_800_000_000),
    "confucius_cuda": ("s2s-asr-confucius-cuda", 5_000_000_000),
    "confucius_vulkan": ("s2s-asr-confucius-vulkan", 2_200_000_000),
    "parakeet_cpu": ("s2s-asr-parakeet-cpu", 2_800_000_000),
    "parakeet_cuda": ("s2s-asr-parakeet-cuda", 7_500_000_000),
    "voxtral_cpu": ("s2s-asr-voxtral-cpu", 3_200_000_000),
    "voxtral_cuda": ("s2s-asr-voxtral-cuda", 7_800_000_000),
    "qwen_sycl_aot": ("s2s-tts-qwen-sycl-aot", 3_800_000_000),
    "qwen_sycl_jit": ("s2s-tts-qwen-sycl-jit", 3_800_000_000),
    "qwen_vulkan": ("s2s-tts-qwen-vulkan", 1_800_000_000),
    "qwen_cuda": ("s2s-tts-qwen-cuda", 3_600_000_000),
    "kokoro": ("s2s-tts-kokoro", 650_000_000),
    "vibevoice_cpu": ("s2s-tts-vibevoice-cpu", 1_200_000_000),
    "vibevoice_cuda": ("s2s-tts-vibevoice-cuda", 3_600_000_000),
    "higgs_cuda": ("s2s-tts-higgs-cuda", 9_500_000_000),
    "llama_cpu": ("s2s-llama-granite", 2_200_000_000),
    "llama_cuda": ("s2s-llama-granite-cuda", 4_500_000_000),
    "web": ("s2s-web", 35_000_000),
    "model_init": ("s2s-model-init", 25_000_000),
}


def image_key(variant: dict) -> str:
    image = variant["image"]
    accel = variant["accelerator"]
    variant_id = variant["id"]
    if image.startswith("s2s-whisper-fw:"):
        return "whisper"
    if image.startswith("s2s-asr-confucius:"):
        return f"confucius_{accel}"
    if image.startswith("s2s-asr-parakeet:"):
        return f"parakeet_{'cuda' if accel == 'cuda' else 'cpu'}"
    if image.startswith("s2s-asr-voxtral:"):
        return f"voxtral_{accel}"
    if image == "s2s-vulkan:local":
        return "gateway"
    if variant_id == "qwen3-tts-b580-sycl-aot":
        return "qwen_sycl_aot"
    if variant_id == "qwen3-tts-sycl-jit":
        return "qwen_sycl_jit"
    if image.startswith("s2s-tts-qwen-vulkan:"):
        return "qwen_vulkan"
    if image.startswith("s2s-tts-qwen-cuda:"):
        return "qwen_cuda"
    if image.startswith("s2s-tts-kokoro:"):
        return "kokoro"
    if image.startswith("s2s-tts-vibevoice:"):
        return f"vibevoice_{'cuda' if accel == 'cuda' else 'cpu'}"
    if image.startswith("s2s-tts-higgs:"):
        return "higgs_cuda"
    if image.startswith("s2s-llama-granite-cuda:"):
        return "llama_cuda"
    if image.startswith("s2s-llama-granite:"):
        return "llama_cpu"
    raise ValueError(f"no published image mapping for {variant_id}: {image}")


def image_ref(key: str) -> str:
    repository, _ = IMAGES[key]
    return f"ghcr.io/antibyte/{repository}@sha256:{DIGEST}"


def runtime_command(backend_id: str, accelerator: str = "") -> list[str]:
    if backend_id == "confucius4-r2t2":
        return [
            "-m", "/opt/s2s-models/confucius4-r2t2.Q4_K_M.gguf",
            "--mmproj", "/opt/s2s-models/confucius4-r2t2.mmproj-Q8_0.gguf",
            "--host", "0.0.0.0", "--port", "8082", "-c", "4096",
            "-ngl", "0" if accelerator == "cpu" else "99",
            "--alias", "confucius4-r2t2",
        ]
    if backend_id == "supertonic":
        return [
            "--mode", "tts-server", "--host", "0.0.0.0", "--port", "8083",
            "--skip-health", "--supertonic-model-dir", "/opt/s2s-models/supertonic/onnx",
            "--supertonic-voice", "M1", "--tts-sample-rate", "16000",
        ]
    if backend_id == "local-fallback":
        return [
            "-m", "/opt/s2s-models/granite-3.3-2b-instruct-q4_k_m.gguf",
            "--host", "0.0.0.0", "--port", "8081", "-c", "2048",
            "--alias", "granite-3.3-2b-instruct",
        ]
    return []


def generate(bundle: dict, catalog: dict) -> dict:
    bundle["schema_version"] = 2
    bundle["contract_version"] = "speech-lab/v2"
    bundle["images"] = {key: image_ref(key) for key in IMAGES}
    bundle["volumes"] = [
        "aurago-speech-lab-models",
        "aurago-speech-lab-data",
        "aurago-speech-lab-control",
    ]
    bundle["start_order"] = [
        "model_init", "control_init", "asr", "llm", "tts", "controller", "gateway", "web"
    ]
    services = bundle["services"]
    if not any(service["role"] == "control_init" for service in services):
        gateway_index = next(i for i, service in enumerate(services) if service["role"] == "gateway")
        services.insert(
            gateway_index,
            {
                "role": "control_init",
                "image": "controller",
                "restart": "no",
                "internal_only": True,
                "environment": ["S2S_CONTROLLER_INIT_TOKEN_ONLY=true", "S2S_CONTROLLER_TOKEN_FILE=/control/token"],
            },
        )
    if not any(service["role"] == "controller" for service in services):
        gateway_index = next(i for i, service in enumerate(services) if service["role"] == "gateway")
        services.insert(
            gateway_index,
            {
                "role": "controller",
                "image": "controller",
                "health_path": "/health",
                "port": 2375,
                "internal_only": True,
                "docker_socket": True,
            },
        )
    runtimes = []
    for backend in catalog["backends"]:
        for variant in backend.get("variants", []):
            if not variant.get("container"):
                continue
            if variant.get("published", True) is False:
                if not variant.get("runtime_delivery_reason"):
                    raise ValueError(f"unpublished variant {variant['id']} needs runtime_delivery_reason")
                continue
            key = image_key(variant)
            endpoint = urlparse(variant["endpoint"])
            if not endpoint.hostname or not endpoint.port:
                raise ValueError(f"variant {variant['id']} has no concrete endpoint")
            health_path = variant.get("health_path") or "/health"
            model_paths = {"/models"}
            if any("/opt/s2s-models" in value for value in variant.get("environment", {}).values()):
                model_paths.add("/opt/s2s-models")
            licenses = backend.get("licenses", [])
            artifacts = variant.get("artifacts", backend.get("artifacts", []))
            ram_gb = float(backend.get("resources", {}).get("ram_gb", 0))
            vram_gb = float(backend.get("resources", {}).get("vram_gb", 0))
            runtimes.append(
                {
                    "backend_id": backend["id"],
                    "variant_id": variant["id"],
                    "stage": backend["stage"],
                    "container": variant["container"],
                    "image": image_ref(key),
                    "image_key": key,
                    "image_download_size_bytes": IMAGES[key][1],
                    "architectures": ["amd64"],
                    "accelerator": variant["accelerator"],
                    "experimental": not variant.get("stable", False),
                    "command": runtime_command(backend["id"], variant["accelerator"]),
                    "environment": variant.get("environment", {}),
                    "aliases": sorted({endpoint.hostname}),
                    "model_mounts": sorted(model_paths),
                    "network": {"name": bundle["network"], "internal_only": True},
                    "volumes": [
                        {"name": "aurago-speech-lab-models", "targets": sorted(model_paths), "read_only": True},
                        {"name": "aurago-speech-lab-data", "targets": ["/data"], "read_only": False},
                    ],
                    "healthcheck": {
                        "path": health_path if health_path.startswith("/") else f"/{health_path}",
                        "port": endpoint.port,
                        "timeout_seconds": 5,
                        "interval_seconds": 3,
                    },
                    "resources": {
                        "memory_bytes": max(536_870_912, int(ram_gb * 1_073_741_824)),
                        "nano_cpus": 2_000_000_000,
                        "shm_bytes": 1_073_741_824 if ram_gb >= 4 else 268_435_456,
                        "gpu_count": 0 if variant["accelerator"] == "cpu" else 1,
                        "vram_gb": vram_gb,
                    },
                    "license": ", ".join(licenses),
                    "auth_required": any(artifact.get("auth") for artifact in artifacts),
                }
            )
    bundle["runtimes"] = sorted(runtimes, key=lambda item: item["variant_id"])
    return bundle


def validate(bundle: dict, catalog: dict) -> None:
    errors: list[str] = []
    if bundle.get("schema_version") != 2 or bundle.get("contract_version") != "speech-lab/v2":
        errors.append("bundle must use schema 2 and speech-lab/v2")
    digest_pattern = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
    catalog_variants = {}
    for backend in catalog["backends"]:
        for variant in backend.get("variants", []):
            if not variant.get("container"):
                continue
            if variant.get("published", True) is False:
                if not variant.get("runtime_delivery_reason"):
                    errors.append(f"unpublished {variant['id']} has no reason")
                continue
            catalog_variants[variant["id"]] = (backend, variant)
    runtime_variants = {runtime["variant_id"]: runtime for runtime in bundle.get("runtimes", [])}
    if set(catalog_variants) != set(runtime_variants):
        errors.append(
            f"catalog/runtime drift missing={sorted(set(catalog_variants)-set(runtime_variants))} "
            f"extra={sorted(set(runtime_variants)-set(catalog_variants))}"
        )
    for variant_id, runtime in runtime_variants.items():
        pair = catalog_variants.get(variant_id)
        if not pair:
            continue
        backend, variant = pair
        if runtime.get("backend_id") != backend["id"] or runtime.get("stage") != backend["stage"]:
            errors.append(f"{variant_id}: backend/stage mismatch")
        if runtime.get("container") != variant["container"]:
            errors.append(f"{variant_id}: container mismatch")
        if not digest_pattern.match(runtime.get("image", "")):
            errors.append(f"{variant_id}: image is not digest pinned")
        if not runtime.get("architectures"):
            errors.append(f"{variant_id}: no architectures")
        if runtime.get("image_download_size_bytes", 0) <= 0:
            errors.append(f"{variant_id}: no image size estimate")
        health = runtime.get("healthcheck", {})
        if not health.get("path") or not 0 < health.get("port", 0) < 65536:
            errors.append(f"{variant_id}: invalid healthcheck")
        if runtime.get("experimental") == variant.get("stable", False):
            errors.append(f"{variant_id}: experimental flag drift")
        if not runtime.get("volumes") or not runtime.get("network") or not runtime.get("resources"):
            errors.append(f"{variant_id}: incomplete isolation/resource contract")
    controller = [service for service in bundle.get("services", []) if service.get("role") == "controller"]
    if len(controller) != 1 or not controller[0].get("internal_only") or not controller[0].get("docker_socket"):
        errors.append("exactly one private controller service must own the Docker socket")
    if any(service.get("docker_socket") for service in bundle.get("services", []) if service.get("role") != "controller"):
        errors.append("a non-controller service has Docker socket access")
    if errors:
        raise SystemExit("\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    if args.write:
        bundle = generate(bundle, catalog)
        BUNDLE_PATH.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    validate(bundle, catalog)
    print(f"validated {len(bundle['runtimes'])} published Speech Lab runtimes")


if __name__ == "__main__":
    main()
