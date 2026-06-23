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


def main() -> None:
    test_cli_verifies_release_evidence_matrix()
    test_readme_and_reproduce_doc_show_open_box_commands()
    print("TileMEM release-ready tests passed")


if __name__ == "__main__":
    main()
