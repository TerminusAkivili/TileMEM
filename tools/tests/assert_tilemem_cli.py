#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tools" / "tilemem"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_doctor_outputs_json_status() -> None:
    completed = _run("doctor", "--json")
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "tilemem_cli_doctor_v1"
    assert payload["tilemem_importable"] is True
    assert payload["api_symbols"] >= 50
    assert "python" in payload


def test_verify_quick_runs_core_assertions() -> None:
    completed = _run("verify", "--quick")
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "tilemem_cli_verify_v1"
    assert payload["mode"] == "quick"
    assert payload["status"] == "passed"
    assert "tools/tests/assert_tilemem_sdk.py" in payload["commands"]


def test_compile_wraps_existing_model_spec_compiler() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "compile"
        completed = _run(
            "compile",
            "--model-spec",
            "configs/models/model_spec_template.json",
            "--out-dir",
            str(out_dir),
        )
        payload = json.loads(completed.stdout)
        assert payload["schema_version"] == "tilemem_cli_compile_v1"
        assert payload["status"] == "compiled"
        assert Path(payload["mir_path"]).exists()
        assert Path(payload["manifest_path"]).exists()
        assert Path(payload["compiled_plan_path"]).exists()


def test_compile_wraps_existing_tmem_plan_compiler() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "compile_plan"
        completed = _run(
            "compile",
            "--plan",
            "configs/models/model_template.tmem",
            "--out-dir",
            str(out_dir),
        )
        payload = json.loads(completed.stdout)
        assert payload["schema_version"] == "tilemem_cli_compile_v1"
        assert payload["status"] == "compiled"
        assert payload["input_kind"] == "plan"
        assert Path(payload["mir_path"]).exists()
        assert Path(payload["manifest_path"]).exists()
        assert Path(payload["compiled_plan_path"]).exists()


def test_checkpoint_prepare_dry_run_generates_artifact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint_dir = root / "checkpoint"
        out_dir = root / "out"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2_moe",
                    "num_hidden_layers": 2,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "hidden_size": 128,
                    "moe_intermediate_size": 256,
                },
                indent=2,
            )
            + "\n"
        )
        (checkpoint_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 1},
                    "weight_map": {
                        "model.layers.0.mlp.experts.0.gate_proj.weight": "model.safetensors",
                        "model.layers.0.mlp.experts.0.up_proj.weight": "model.safetensors",
                        "model.layers.0.mlp.experts.0.down_proj.weight": "model.safetensors",
                    },
                },
                indent=2,
            )
            + "\n"
        )
        completed = _run(
            "checkpoint",
            "prepare",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--out-dir",
            str(out_dir),
            "--backend",
            "sglang",
            "--layers",
            "0",
            "--experts",
            "0",
            "--dry-run",
        )
        payload = json.loads(completed.stdout)
        assert payload["schema_version"] == "tilemem_checkpoint_prepare_v1"
        assert payload["serving"]["status"] == "dry_run"


def main() -> None:
    test_doctor_outputs_json_status()
    test_verify_quick_runs_core_assertions()
    test_compile_wraps_existing_model_spec_compiler()
    test_compile_wraps_existing_tmem_plan_compiler()
    test_checkpoint_prepare_dry_run_generates_artifact()
    print("TileMEM CLI tests passed")


if __name__ == "__main__":
    main()
