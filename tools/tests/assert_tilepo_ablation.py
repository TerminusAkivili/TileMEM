#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tilepo.compiler import compile_plan
from tilepo import env as tilepo_env
from tilepo.mir import RuntimeMode
from tilepo.runtime import TileMEMRuntime
from tilepo.ablation import render_tilepo_plan, write_merged_manifest


def _success_row(policy: str, async_planning: str, system: str, experts: int, repeat: int) -> dict:
    return {
        "system": system,
        "workload": "mixed",
        "experts_per_layer": experts,
        "repeat": repeat,
        "request_count": 5,
        "warmup_request_count": 1,
        "output_tokens": 8,
        "tok_per_sec": 10.0 + repeat + (2.0 if policy.startswith("tilepo") else 0.0),
        "p50_ms": 100.0,
        "p95_ms": 120.0 - repeat,
        "p99_ms": 150.0 - repeat,
        "gpu_peak_gib": 8.0 + experts / 10.0,
        "cpu_ram_peak_gib": 18.0 + experts / 20.0,
        "server_ready_s": 4.0,
        "fallback_count": 0,
        "dtype_counts": {"bf16": 1},
        "command": ["python", "-m", "sglang.launch_server", "--kt-method", "BF16", "--dtype", "bfloat16"],
        "evidence_level": "real",
        "simulated": False,
        "status": "success",
        "tilepo_policy": policy,
        "tilepo_async_planning": async_planning,
        "raw_path": f"raw/{policy}_{async_planning}_{system}_{experts}_{repeat}.jsonl",
    }


def main() -> int:
    with TemporaryDirectory() as tmp:
        _assert_env_parsing()
        root = Path(tmp)
        base = Path("configs/tilepo_olmoe_bf16_only.tmem")

        manifests = {}
        for policy in ("tilepo_coarse", "tilepo_fine", "tilepo_hybrid"):
            plan = root / f"{policy}.tmem"
            plan.write_text(render_tilepo_plan(base, expert_budget=6, policy=policy, async_planning=True))
            compiled = compile_plan(plan, root / f"compiled_{policy}")
            manifest = compiled.manifest
            manifests[policy] = manifest
            assert manifest["tilepo_plan"]["policy"] == policy
            assert manifest["tilepo_plan"]["async_planning"] is True
            assert manifest["tilepo_plan"]["expert_budget"] == 6
            assert manifest["tilepo_plan"]["tile_count"] == len(manifest["tile_offsets"])

        coarse_tiles = manifests["tilepo_coarse"]["tilepo_plan"]["tile_count"]
        fine_tiles = manifests["tilepo_fine"]["tilepo_plan"]["tile_count"]
        hybrid_tiles = manifests["tilepo_hybrid"]["tilepo_plan"]["tile_count"]
        assert coarse_tiles < hybrid_tiles < fine_tiles, (coarse_tiles, hybrid_tiles, fine_tiles)

        runtime = TileMEMRuntime(manifests["tilepo_fine"], {}, mode=RuntimeMode.SERVE)
        request = {"topk": [(0, 0)], "require_tilemem": True}
        runtime.prefetch_plan(request)
        runtime.execute(request)
        metrics = runtime.metrics.snapshot()
        assert metrics["async_planning_mode"] == "on"
        assert metrics["async_plan_cache_hits"] == 1
        assert metrics["async_plan_cache_misses"] == 0
        assert metrics["plan_lookup_us"] == 0.0
        assert metrics["plan_lookup_total_us"] >= 0.0

        manifest_a = root / "manifest_a.json"
        manifest_b = root / "manifest_b.json"
        manifest_a.write_text(
            json.dumps(
                {
                    "schema_version": "tilepo_sweep_manifest_v1",
                    "runs": [_success_row("kt_expert", "off", "B", 6, repeat) for repeat in range(3)],
                    "command_runs": [],
                },
                indent=2,
            )
        )
        manifest_b.write_text(
            json.dumps(
                {
                    "schema_version": "tilepo_sweep_manifest_v1",
                    "runs": [
                        _success_row("tilepo_coarse", async_mode, "C", 6, repeat)
                        for async_mode in ("off", "on")
                        for repeat in range(3)
                    ],
                    "command_runs": [],
                },
                indent=2,
            )
        )
        merged = root / "merged.json"
        write_merged_manifest([manifest_a, manifest_b], merged)
        merged_data = json.loads(merged.read_text())
        assert merged_data["schema_version"] == "tilepo_merged_manifest_v1"
        assert len(merged_data["runs"]) == 9

        report_dir = root / "report"
        proc = subprocess.run(
            [
                sys.executable,
                "tools/report_tilepo_ablation",
                "--manifest",
                str(merged),
                "--out-dir",
                str(report_dir),
                "--workloads",
                "mixed",
                "--experts",
                "6",
                "--policies",
                "kt_expert,tilepo_coarse",
                "--async-modes",
                "off,on",
                "--repeats",
                "3",
                "--require-real",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        summary = json.loads((report_dir / "tilepo_ablation_summary.json").read_text())
        assert summary["gate"]["status"] == "PASS"
        by_key = {
            (
                item["workload"],
                item["experts_per_layer"],
                item["policy"],
                item["async_planning"],
                item["system"],
            ): item
            for item in summary["groups"]
        }
        coarse = by_key[("mixed", 6, "tilepo_coarse", "on", "C")]
        assert coarse["metrics"]["gpu_peak_gib"]["median"] > 0
        assert coarse["metrics"]["cpu_ram_peak_gib"]["median"] > 0
        assert (report_dir / "tilepo_ablation_report.md").exists()

    return 0


def _assert_env_parsing() -> None:
    original = {
        key: os.environ.get(key)
        for key in (
            tilepo_env.TILEPO_HOOK_BACKEND_PROBE_LIMIT,
            tilepo_env.TILEPO_HOOK_FLUSH_INTERVAL,
            tilepo_env.TILEPO_REQUIRE_NATIVE_BACKEND,
            tilepo_env.TILEPO_VERIFY_ATOL,
        )
    }
    try:
        os.environ[tilepo_env.TILEPO_HOOK_BACKEND_PROBE_LIMIT] = "bad"
        assert tilepo_env.hook_backend_probe_limit() == 1
        os.environ[tilepo_env.TILEPO_HOOK_BACKEND_PROBE_LIMIT] = "-4"
        assert tilepo_env.hook_backend_probe_limit() == 0
        os.environ[tilepo_env.TILEPO_HOOK_FLUSH_INTERVAL] = "0"
        assert tilepo_env.hook_flush_interval() == 1
        os.environ[tilepo_env.TILEPO_REQUIRE_NATIVE_BACKEND] = "yes"
        assert tilepo_env.require_native_backend() is True
        os.environ[tilepo_env.TILEPO_VERIFY_ATOL] = "-1.0"
        assert tilepo_env.verify_atol() == 0.0
        os.environ[tilepo_env.TILEPO_VERIFY_ATOL] = "nan"
        assert tilepo_env.verify_atol() == 0.0
        os.environ[tilepo_env.TILEPO_VERIFY_ATOL] = "inf"
        assert tilepo_env.verify_atol() == 0.0
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
