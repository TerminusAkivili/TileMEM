#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilepo import sweep  # noqa: E402


def main() -> int:
    bench = ROOT / "tools" / "openai_varprompt_bench"
    assert bench.exists(), "release checkout must include tools/openai_varprompt_bench"
    assert bench.is_file()
    found = sweep._find_bench_tool()
    assert found == bench, found
    assert all(
        not str(candidate).startswith("/home/")
        for candidate in sweep.DEFAULT_BENCH_TOOL_CANDIDATES
    ), sweep.DEFAULT_BENCH_TOOL_CANDIDATES
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
