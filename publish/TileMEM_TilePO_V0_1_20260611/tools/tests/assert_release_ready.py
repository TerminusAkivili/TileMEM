#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tools" / "tilemem"

EXPECTED_WORKLOADS = ["mixed", "long_context"]
EXPECTED_EXPERTS = [2, 4, 6, 8, 10]
EXPECTED_POLICIES = ["kt_expert", "tilepo_coarse", "tilepo_fine", "tilepo_hybrid"]
EXPECTED_ASYNC = ["off", "on"]


def test_cli_verifies_release_evidence_matrix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "evidence"
        completed = subprocess.run(
            [
                sys.executable,
                str(CLI),
                "evidence",
                "verify",
                "--out-dir",
                str(out_dir),
                "--json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)

        assert payload["schema_version"] == "tilemem_release_evidence_v1"
        assert payload["status"] == "PASS"
        assert payload["gate"] == "PASS"
        assert payload["workloads"] == EXPECTED_WORKLOADS
        assert payload["experts"] == EXPECTED_EXPERTS
        assert payload["policies"] == EXPECTED_POLICIES
        assert payload["async_planning"] == EXPECTED_ASYNC
        assert payload["repeats"] == 3
        assert payload["request_count"] == 5
        assert payload["actual_rows"] == 210
        assert payload["expected_rows"] == 210
        assert payload["real_success_rows"] == 210
        assert payload["serving_precision"] == "BF16 / KT-native path"
        assert Path(payload["summary_path"]).exists()
        assert Path(payload["report_path"]).exists()


def test_readme_and_reproduce_doc_show_open_box_commands() -> None:
    readme = (ROOT / "README.md").read_text()
    reproduce = (ROOT / "REPRODUCE.md").read_text()
    required_phrases = [
        "tools/tilemem evidence verify --json",
        "Workloads: mixed, long_context",
        "Experts: 2, 4, 6, 8, 10",
        "Policies: kt_expert, tilepo_coarse, tilepo_fine, tilepo_hybrid",
        "Async planning: off, on",
        "Repeats: 3",
        "Request count: 5",
        "Rows: 210 / 210 real success",
        "Gate: PASS",
        "Serving precision: BF16 / KT-native path",
    ]
    for phrase in required_phrases:
        assert phrase in readme, phrase
        assert phrase in reproduce, phrase


def test_public_mir_story_is_v0_1_1_not_v0_12() -> None:
    readme = (ROOT / "README.md").read_text()
    model_template = (ROOT / "configs" / "models" / "model_spec_template.json").read_text()
    init_py = (ROOT / "tilemem" / "__init__.py").read_text()
    mir_schema = (ROOT / "tilepo" / "mir" / "schema.py").read_text()

    required_phrases = [
        "## v0.1.1 Public MIR And Replaceable Model Interface",
        "tilemem_v0_1_1_model_spec_template",
        '"public_interface": "tilemem_public_mir_v0_1_1"',
        '__version__ = "0.1.1"',
    ]
    combined = "\n".join([readme, model_template, init_py, mir_schema])
    for phrase in required_phrases:
        assert phrase in combined, phrase

    forbidden_phrases = [
        "V0.12",
        "v012",
        "v0_12",
        "0.12.0",
        "tilemem_public_mir_v0_12",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in combined, phrase


def main() -> None:
    test_cli_verifies_release_evidence_matrix()
    test_readme_and_reproduce_doc_show_open_box_commands()
    test_public_mir_story_is_v0_1_1_not_v0_12()
    print("TileMEM release-ready tests passed")


if __name__ == "__main__":
    main()
