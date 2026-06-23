#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tilemem as TM  # noqa: E402


SCHEMA_VERSION = "tilemem_industrial_quickstart_v1"


def build_payload() -> dict[str, Any]:
    spec = TM.model_spec(
        name="tilemem_industrial_quickstart",
        layers=2,
        experts_per_layer=4,
        hidden_size=16,
        intermediate_size=32,
        expert_budget=2,
        workload="mixed",
        tile={
            "hidden_tile": 8,
            "intermediate_tile": 16,
            "shard_count": 2,
            "projection_groups": ["gate_up", "down"],
        },
        memory={
            "gpu_cache_budget_gib": 1.0,
            "cpu_cache_budget_gib": 8.0,
        },
    )
    compiled = TM.plan(spec)

    registry = TM.BackendRegistry()
    capability = TM.BackendCapability(
        name="customer_cuda_fp8",
        formats=["fp8_e4m3_sample"],
        layouts=["block_n32_fp32"],
        scale_granularities=["block"],
        projection_groups=["gate_up"],
        runtime_entrypoint="kernels/gemm_fp8.cu:tilemem_launch_gemm_fp8",
        owns_quantization=True,
        owns_calibration=True,
        owns_quality=True,
        hardware_targets=["cuda_native_arch"],
        fallback_dtype="bf16",
    )
    TM.register_backend(capability, registry=registry)
    external_manifest = _sample_external_fp8_manifest()
    external_handles = TM.build_tile_handles(external_manifest, registry=registry)
    external_handle = external_handles[0]

    headline = TM.v0_1_headline_gain()

    return {
        "schema_version": SCHEMA_VERSION,
        "api_style": "import tilemem as TM",
        "sdk_plan": {
            "model": compiled.mir.name,
            "manifest_checksum": compiled.manifest["checksum"],
            "tiles": len(compiled.handles),
            "dispatch": compiled.dispatch_summary(iterations=3),
        },
        "external_kernel": {
            "source": "kernels/gemm_fp8.cu",
            "launch_function": "tilemem_launch_gemm_fp8",
            "backend_capability": capability.to_dict(),
            "handle": external_handle.to_dict(),
            "compile_note": (
                "The real CUDA sample lives in kernels/gemm_fp8.cu. "
                "This quickstart validates the TileMEM runtime contract without requiring nvcc."
            ),
        },
        "v0_1_headline_gain": headline,
        "responsibility_boundary": {
            "tilemem_owns": [
                "tile splitting and tile id management",
                "dtype tag and tile format metadata",
                "scale metadata address, size, and layout descriptors",
                "backend capability registration",
                "manifest generation",
                "runtime dispatch tile handles",
                "fallback chain descriptors",
            ],
            "external_developer_owns": [
                "concrete FP8/FP6/FP4 kernels",
                "model structure adaptation",
                "weight quantization",
                "calibration method",
                "quality evaluation",
                "backend-specific physical layout",
                "CUDA/TileLang/Triton/ROCm implementation",
            ],
        },
    }


def _sample_external_fp8_manifest() -> dict[str, Any]:
    key = "L0:E0:gate_up:S0:N0-64"
    return {
        "schema_version": "tilemem_industrial_quickstart_manifest_v1",
        "tile_ids": {
            key: {
                "layer": 0,
                "expert": 0,
                "projection_group": "gate_up",
                "shard_id": 0,
                "n_start": 0,
                "n_end": 64,
            }
        },
        "tile_offsets": {key: 0},
        "tile_bytes": {key: 4096},
        "tile_dtype_map": {key: "fp8"},
        "tile_format_map": {
            key: {
                "format": "fp8_e4m3_sample",
                "storage_bits": 8,
                "compute_dtype": "fp8_e4m3",
                "accum_dtype": "bf16",
                "layout_owner": "customer_cuda_fp8",
            }
        },
        "scale_offsets": {key: 0},
        "scale_bytes": {key: 256},
        "scale_layout_map": {key: "block_n32_fp32"},
        "backend_owner_map": {key: "customer_cuda_fp8"},
        "tile_fallback_map": {key: {"dtype": "bf16", "backend": "kt_fallback"}},
        "gpu_hot_tiles": [key],
        "fallback_chain": ["bf16", "kt_fallback"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TileMEM industrial SDK quickstart.")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    payload = build_payload()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(serialized)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
