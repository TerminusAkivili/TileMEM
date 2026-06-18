#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    namespace = runpy.run_path(str(ROOT / "tools" / "openai_varprompt_bench"), run_name="tilepo_bench_test")
    marker_summary = namespace["bootstrap_marker_summary"]
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "run.jsonl"
        marker = out.with_suffix(".tilepo_bootstrap.json")
        marker.write_text(
            json.dumps(
                {
                    "kt_executor_preserved": True,
                    "runtime_metrics_source": "kt_preserving_native_tc_kernel",
                    "serving_hook": {
                        "serving_hook_active": True,
                        "serving_hook_invocations": 1,
                        "serving_hook_replaced_count": 1,
                        "serving_hook_returned_original": False,
                        "serving_hook_mode": "native_tc_adapter",
                        "tc_native_consumed_coalesced_groups": True,
                        "tc_native_descriptor_count": 8,
                        "tc_native_entrypoint": "tilepo_cuda_dispatch_coalesced_gemm",
                        "tc_native_descriptor_layout": "tilepo_cuda_coalesced_group_desc_v1",
                    },
                },
                indent=2,
            )
        )
        summary = marker_summary(out)
        assert summary["kt_executor_preserved"] is True
        assert summary["runtime_metrics_source"] == "kt_preserving_native_tc_kernel"
        assert summary["serving_hook_active"] is True
        assert summary["serving_hook_replaced_count"] == 1
        assert summary["serving_hook_returned_original"] is False
        assert summary["serving_hook_mode"] == "native_tc_adapter"
        assert summary["tc_native_consumed_coalesced_groups"] is True
        assert summary["tc_native_descriptor_count"] == 8
        assert summary["tc_native_entrypoint"] == "tilepo_cuda_dispatch_coalesced_gemm"
        assert summary["tc_native_descriptor_layout"] == "tilepo_cuda_coalesced_group_desc_v1"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
