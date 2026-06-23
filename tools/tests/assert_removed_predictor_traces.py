#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_UPPER = "T" + "MAP"
FORBIDDEN_LOWER = "t" + "map"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def test_no_predictor_paths() -> None:
    ignored_roots = {".git", "build", "__pycache__"}
    offenders = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in ignored_roots for part in relative.parts):
            continue
        if FORBIDDEN_LOWER in path.name.lower():
            offenders.append(str(relative))
    assert offenders == []


def test_no_predictor_text_references() -> None:
    pattern = rf"\b{FORBIDDEN_UPPER}\b|\b{FORBIDDEN_LOWER}\b"
    result = _run(
        [
            "rg",
            "-n",
            "-i",
            pattern,
            ".",
            "-g",
            "!build/**",
            "-g",
            "!*.tar.gz",
            "-g",
            "!*.zip",
            "-g",
            "!*.pyc",
        ]
    )
    assert result.returncode == 1, result.stdout + result.stderr


def test_cli_no_predictor_subcommand() -> None:
    result = _run([sys.executable, "tools/tilemem", "--help"])
    assert result.returncode == 0, result.stderr
    assert FORBIDDEN_LOWER not in result.stdout.lower()


def test_sdk_surface_no_predictor_symbols() -> None:
    sys.path.insert(0, str(ROOT))
    import tilemem as TM

    forbidden = [
        "HardwareProfile",
        "PredictionResult",
        FORBIDDEN_UPPER + "Decision",
        "hardware_profile",
        "predict_policy",
    ]
    exported = set(getattr(TM, "__all__", []))
    for name in forbidden:
        assert not hasattr(TM, name), name
        assert name not in exported, name


def main() -> None:
    test_no_predictor_paths()
    test_no_predictor_text_references()
    test_cli_no_predictor_subcommand()
    test_sdk_surface_no_predictor_symbols()
    print("Removed predictor trace check passed")


if __name__ == "__main__":
    main()
