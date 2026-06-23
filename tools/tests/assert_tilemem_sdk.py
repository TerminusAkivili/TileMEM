#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_import_tilemem_sdk_surface() -> None:
    import tilemem as TM

    required = [
        "BackendCapability",
        "BackendRegistry",
        "CheckpointArtifact",
        "MoETopology",
        "ScaleLayout",
        "ServingCommand",
        "ServingResult",
        "TileFormat",
        "TileHandle",
        "WeightMatchResult",
        "build_manifest",
        "build_mir",
        "build_runtime_weight_aliases",
        "build_serving_command",
        "build_tile_checkpoint_map",
        "build_tile_handles",
        "checkpoint_weight_names",
        "export_checkpoint_artifact",
        "infer_moe_topology",
        "load_checkpoint_weight_map",
        "load_hf_config",
        "match_checkpoint_weights",
        "model_spec",
        "model_spec_from_hf_config",
        "plan",
        "plan_from_hf_config",
        "register_backend",
        "run_serving_backend",
        "v0_1_headline_gain",
    ]
    missing = [name for name in required if not hasattr(TM, name)]
    assert missing == []


def test_sdk_builds_manifest_and_dispatch_handles() -> None:
    import tilemem as TM

    spec = TM.model_spec(
        name="tilemem_sdk_quickstart",
        layers=2,
        experts_per_layer=4,
        hidden_size=16,
        intermediate_size=32,
        expert_budget=2,
        workload="mixed",
        tile={"hidden_tile": 8, "intermediate_tile": 16, "shard_count": 2},
    )
    compiled = TM.plan(spec)
    assert compiled.mir.name == "tilemem_sdk_quickstart"
    assert compiled.manifest["model"] == "tilemem_sdk_quickstart"
    assert compiled.handles
    assert all(handle.backend == "kt_fallback" for handle in compiled.handles)

    dispatch = compiled.dispatch_summary(iterations=3)
    assert dispatch["tiles"] == len(compiled.handles)
    assert dispatch["dispatchable_tiles"] == len(compiled.handles)
    assert dispatch["fallback_tiles"] == 0


def test_sdk_registers_external_backend_and_exposes_tile_handles() -> None:
    import tilemem as TM

    registry = TM.BackendRegistry()
    TM.register_backend(
        TM.BackendCapability(
            name="customer_cuda",
            formats=["fp8_e4m3_sample"],
            layouts=["block_n32_fp32"],
            scale_granularities=["block"],
            projection_groups=["gate_up"],
            runtime_entrypoint="kernels/gemm_fp8.cu:tilemem_launch_gemm_fp8",
            owns_quantization=True,
            owns_calibration=True,
            owns_quality=True,
            hardware_targets=["cuda_sm90"],
            fallback_dtype="bf16",
        ),
        registry=registry,
    )
    manifest = {
        "tile_ids": {
            "L0:E0:gate_up:S0:N0-64": {
                "layer": 0,
                "expert": 0,
                "projection_group": "gate_up",
                "shard_id": 0,
                "n_start": 0,
                "n_end": 64,
            }
        },
        "tile_offsets": {"L0:E0:gate_up:S0:N0-64": 0},
        "tile_bytes": {"L0:E0:gate_up:S0:N0-64": 4096},
        "tile_dtype_map": {"L0:E0:gate_up:S0:N0-64": "fp8"},
        "tile_format_map": {
            "L0:E0:gate_up:S0:N0-64": {
                "format": "fp8_e4m3_sample",
                "storage_bits": 8,
                "compute_dtype": "fp8_e4m3",
                "accum_dtype": "bf16",
                "layout_owner": "customer_cuda",
            }
        },
        "scale_offsets": {"L0:E0:gate_up:S0:N0-64": 0},
        "scale_bytes": {"L0:E0:gate_up:S0:N0-64": 256},
        "scale_layout_map": {"L0:E0:gate_up:S0:N0-64": "block_n32_fp32"},
        "backend_owner_map": {"L0:E0:gate_up:S0:N0-64": "customer_cuda"},
        "tile_fallback_map": {"L0:E0:gate_up:S0:N0-64": {"dtype": "bf16", "backend": "kt_fallback"}},
        "gpu_hot_tiles": ["L0:E0:gate_up:S0:N0-64"],
    }

    handles = TM.build_tile_handles(manifest, registry=registry)
    assert len(handles) == 1
    handle = handles[0]
    assert handle.dispatchable is True
    assert handle.backend == "customer_cuda"
    assert handle.format == "fp8_e4m3_sample"
    assert handle.scale_offset == 0
    assert handle.scale_bytes == 256
    assert handle.fallback_backend == "kt_fallback"


def test_sdk_replays_v0_1_headline_gain() -> None:
    import tilemem as TM

    headline = TM.v0_1_headline_gain()
    assert headline["best"]["tok_gain_pct"] >= 30.0
    assert headline["best"]["policy"].startswith("tilepo_")
    assert headline["gate"]["status"] == "PASS"


def main() -> None:
    test_import_tilemem_sdk_surface()
    test_sdk_builds_manifest_and_dispatch_handles()
    test_sdk_registers_external_backend_and_exposes_tile_handles()
    test_sdk_replays_v0_1_headline_gain()
    print("TileMEM SDK tests passed")


if __name__ == "__main__":
    main()
