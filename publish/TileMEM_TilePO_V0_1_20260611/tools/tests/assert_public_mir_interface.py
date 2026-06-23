#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilepo.model_interface import (  # noqa: E402
    ModelAdapter,
    ModelSpec,
    build_mir_from_model_spec,
    model_spec_from_dict,
)
from tilepo.mir import ModelIR, load_mir, save_mir, validate_mir_dict  # noqa: E402
from tilepo import (  # noqa: E402
    ModelSpec as PublicModelSpec,
    build_mir_from_model_spec as public_build_mir_from_model_spec,
    model_spec_from_dict as public_model_spec_from_dict,
)


MODEL_SPEC = {
    "schema_version": "tilemem_model_spec_v1",
    "name": "toy_moe",
    "layers": 2,
    "experts_per_layer": 4,
    "hidden_size": 16,
    "intermediate_size": 32,
    "expert_budget": 2,
    "workload": "mixed",
    "tile": {
        "hidden_tile": 8,
        "intermediate_tile": 16,
        "shard_count": 2,
        "projection_groups": ["gate_up", "down"],
    },
    "memory": {
        "gpu_cache_budget_gib": 1.0,
        "cpu_cache_budget_gib": 8.0,
    },
    "precision": {
        "dtype_policy": "bf16",
        "allowed": ["bf16"],
        "calibration_required": False,
    },
    "schedule": {
        "deployment_mode": "balanced",
        "backend_priority": ["cuda", "tilelang", "kt_fallback"],
        "mode": "verify",
        "runtime_gates": ["locality", "correctness"],
        "prewarm_policy": "hotset",
        "miss_policy": "fallback",
    },
}


class DictModelAdapter(ModelAdapter):
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_tilemem_model_spec(self) -> ModelSpec:
        return model_spec_from_dict(self.payload)


def test_model_spec_builds_public_mir_and_round_trips() -> None:
    spec = model_spec_from_dict(MODEL_SPEC)
    assert spec.name == "toy_moe"
    mir = build_mir_from_model_spec(spec)
    assert isinstance(mir, ModelIR)
    assert mir.name == "toy_moe"
    assert mir.layers == 2
    assert mir.experts_per_layer == 4
    assert len(mir.tiles) == 16
    assert mir.schedule.backend_priority[-1].value == "kt_fallback"

    mir_dict = mir.to_dict()
    validate_mir_dict(mir_dict)
    assert mir_dict["schema_version"] == "tilepo_mir_v1"
    assert mir_dict["public_interface"] == "tilemem_public_mir_v0_1_1"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy.mir.json"
        save_mir(mir, path)
        loaded = load_mir(path)
        assert loaded == mir


def test_replaceable_model_adapter_can_supply_spec() -> None:
    adapter = DictModelAdapter(MODEL_SPEC)
    mir = build_mir_from_model_spec(adapter)
    assert mir.name == "toy_moe"
    assert mir.routes[0].workload == "mixed"


def test_model_spec_validation_rejects_bad_shape() -> None:
    bad = dict(MODEL_SPEC)
    bad["experts_per_layer"] = 0
    try:
        model_spec_from_dict(bad)
    except ValueError as exc:
        assert "experts_per_layer must be positive" in str(exc)
    else:
        raise AssertionError("accepted a model spec with no experts")


def test_public_model_spec_v1_rejects_mixed_precision_for_now() -> None:
    fp8_spec = json.loads(json.dumps(MODEL_SPEC))
    fp8_spec["precision"] = {
        "dtype_policy": "fp8",
        "allowed": ["bf16", "fp8"],
        "calibration_required": False,
    }
    try:
        build_mir_from_model_spec(fp8_spec)
    except ValueError as exc:
        assert "BF16-only" in str(exc)
    else:
        raise AssertionError("public model spec v1 accepted mixed precision")


def test_cli_compiles_model_spec_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        spec_path = tmp_path / "model_spec.json"
        spec_path.write_text(json.dumps(MODEL_SPEC, indent=2) + "\n")
        out_dir = tmp_path / "out"
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "tilepo_compile_plan"),
            "--model-spec",
            str(spec_path),
            "--out-dir",
            str(out_dir),
        ]
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        assert "toy_moe.mir.json" in completed.stdout
        mir_path = out_dir / "toy_moe.mir.json"
        manifest_path = out_dir / "toy_moe.manifest.json"
        assert mir_path.exists()
        assert manifest_path.exists()
        mir_data = json.loads(mir_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        normalized_spec = json.loads((out_dir / "toy_moe.model_spec.json").read_text())
        assert mir_data["public_interface"] == "tilemem_public_mir_v0_1_1"
        assert manifest["model"] == "toy_moe"
        assert manifest["schema_version"] == "tilepo_manifest_v1"
        assert normalized_spec["schema_version"] == "tilemem_model_spec_v1"


def test_public_top_level_api_exports_model_interface() -> None:
    spec = public_model_spec_from_dict(MODEL_SPEC)
    assert isinstance(spec, PublicModelSpec)
    mir = public_build_mir_from_model_spec(spec)
    assert mir.name == "toy_moe"


def test_checked_in_model_spec_template_compiles() -> None:
    spec_path = ROOT / "configs" / "models" / "model_spec_template.json"
    assert spec_path.exists(), "missing public model spec template"
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "template_out"
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "tilepo_compile_plan"),
            "--model-spec",
            str(spec_path),
            "--out-dir",
            str(out_dir),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        outputs = sorted(path.name for path in out_dir.iterdir())
        assert outputs == [
            "tilemem_v0_1_1_model_spec_template.manifest.json",
            "tilemem_v0_1_1_model_spec_template.mir.json",
            "tilemem_v0_1_1_model_spec_template.model_spec.json",
        ]


def main() -> None:
    test_model_spec_builds_public_mir_and_round_trips()
    test_replaceable_model_adapter_can_supply_spec()
    test_model_spec_validation_rejects_bad_shape()
    test_public_model_spec_v1_rejects_mixed_precision_for_now()
    test_cli_compiles_model_spec_json()
    test_public_top_level_api_exports_model_interface()
    test_checked_in_model_spec_template_compiles()
    print("Public MIR/model interface tests passed")


if __name__ == "__main__":
    main()
