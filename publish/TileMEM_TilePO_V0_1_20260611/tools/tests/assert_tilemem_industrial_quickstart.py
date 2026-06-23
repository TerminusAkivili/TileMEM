#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def test_industrial_quickstart_runs_through_tilemem_sdk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "tilemem_industrial_quickstart.json"
        completed = subprocess.run(
            [
                sys.executable,
                "examples/tilemem_industrial_quickstart.py",
                "--out-json",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "industrial quickstart failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        payload = json.loads(output.read_text())

    assert payload["schema_version"] == "tilemem_industrial_quickstart_v1"
    assert payload["api_style"] == "import tilemem as TM"
    assert payload["sdk_plan"]["dispatch"]["tiles"] > 0
    assert payload["external_kernel"]["source"] == "kernels/gemm_fp8.cu"
    assert payload["external_kernel"]["handle"]["dispatchable"] is True
    assert payload["v0_1_headline_gain"]["best"]["tok_gain_pct"] >= 30.0


def main() -> None:
    test_industrial_quickstart_runs_through_tilemem_sdk()
    print("TileMEM industrial quickstart test passed")


if __name__ == "__main__":
    main()
