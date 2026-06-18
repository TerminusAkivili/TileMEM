#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _assert_cuda_backend_emits_native_tc_descriptor_metrics(root)
        manifest = root / "adaptive_manifest.json"
        _write_raw_files(root, _rows())
        manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": _rows()}, indent=2))
        out_dir = root / "report"

        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
                "--manifest",
                str(manifest),
                "--out-dir",
                str(out_dir),
                "--require-real",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        summary = json.loads((out_dir / "tilepo_adaptive_granularity_summary.json").read_text())
        assert summary["schema_version"] == "tilepo_adaptive_granularity_report_v1"
        assert summary["gate"]["status"] == "PASS", summary["gate"]
        assert summary["requested"]["expected_rows"] == 24
        assert summary["requested"]["serving_shell"] == "kt_sglang"
        assert summary["requested"]["comparison"] == "tilepo_self_ablation"
        assert len(summary["groups"]) == 24
        assert len(summary["comparisons"]) == 6

        mixed8 = _comparison(summary, "mixed", 8)
        assert mixed8["best_fixed_policy"] == "tilepo_fine"
        assert mixed8["strict_per_point_win"] is True
        assert mixed8["v2_vs_best_fixed"]["tok_gain_pct"] > 0.0
        assert mixed8["v2_vs_best_fixed"]["tok_margin"] > 0.0
        assert "gpu_peak_delta_pct" in mixed8["v2_vs_best_fixed"]
        assert "cpu_ram_peak_delta_pct" in mixed8["v2_vs_best_fixed"]
        assert mixed8["tile_count_comparison"]["coarse"] < mixed8["tile_count_comparison"]["v2"]
        assert mixed8["tile_count_comparison"]["v2"] < mixed8["tile_count_comparison"]["fine"]
        assert mixed8["dispatch_proxy"]["v2_vs_fine_pct"] < 100.0
        assert mixed8["memory_comparison"]["best_fixed_gpu_peak_gib"] == 6.4
        assert mixed8["memory_comparison"]["v2_gpu_peak_gib"] <= mixed8["memory_comparison"]["coarse_gpu_peak_gib"]

        long10 = _comparison(summary, "long_context", 10)
        assert long10["best_fixed_policy"] == "tilepo_coarse"
        assert long10["strict_per_point_win"] is True

        markdown = (out_dir / "tilepo_adaptive_granularity_report.md").read_text()
        assert "ATG + TC + BAA V0.2 Report" in markdown
        assert "V0.2 vs Best Fixed TilePO" in markdown
        assert "Tile Count and Dispatch Proxy" in markdown
        _assert_non_finite_metrics_are_rejected(root)
        _assert_v0_2_requires_strict_win(root)
        _assert_v0_2_requires_native_tc_descriptor_evidence(root)
        _assert_v0_2_requires_native_tc_consumption(root)
        _assert_v0_2_requires_native_tc_adapter_path(root)
        _assert_v0_2_rejects_masked_fallback_counters(root)
        _assert_v0_2_requires_runtime_efficiency_evidence(root)
        _assert_v0_2_requires_measured_runtime_metric_provenance(root)
        _assert_extra_matrix_rows_are_rejected(root)
        _assert_real_command_failure_writes_blocked_manifest(root)
        _assert_real_failed_row_writes_blocked_manifest(root)
        _assert_bench_failure_exits_nonzero(root)
        _assert_offline_runner_preflight_is_local_only(root)
        _assert_reproduction_scripts_wire_strict_native_tc_offline_acceptance()
        _assert_reproduction_scripts_wire_repeats()
        _assert_offline_runner_wires_resume()
        _assert_v0_2_offline_packager_writes_required_entrypoint()
        _assert_sweep_uses_local_bench_and_disables_windows_host_guard(root)
        _assert_sweep_uses_kt_preserving_native_tc_for_v0_2(root)
        _assert_self_ablation_preserves_kt_optimizations_consistently(root)
        _assert_bootstrap_prime_does_not_count_as_measured_replacement()
        _assert_sweep_wires_force_prompt_for_focused_restore(root)
        _assert_windows_host_guard_collection_is_bounded()
        _assert_runtime_env_limits_compile_parallelism(root)
        _assert_runtime_env_uses_shared_jit_cache(root)

    return 0


def _assert_cuda_backend_emits_native_tc_descriptor_metrics(root: Path) -> None:
    from tilepo.ablation import render_tilepo_plan
    from tilepo.backends.cuda_backend import CUDABackend
    from tilepo.compiler import compile_plan

    plan = root / "tilepo_atg_tc_baa_mixed8.tmem"
    plan.write_text(
        render_tilepo_plan(
            ROOT / "configs" / "tilepo_olmoe_bf16_only.tmem",
            expert_budget=8,
            policy="tilepo_atg_tc_baa",
            async_planning=True,
            workload_profile="mixed",
        )
    )
    manifest = compile_plan(plan, root / "compiled_tilepo_atg_tc_baa_mixed8").manifest
    result = CUDABackend(require_native=True).execute(
        {"topk": [(0, 0)], "require_tilemem": True, "payload": "ok"},
        manifest,
    )
    assert result["tc_native_consumed_coalesced_groups"] is True
    assert result["tc_native_descriptor_count"] == 8
    assert result["tc_native_entrypoint"] == "tilepo_cuda_dispatch_coalesced_gemm"
    assert result["tc_native_descriptor_layout"] == "tilepo_cuda_coalesced_group_desc_v1"
    assert result["cuda_descriptor_metrics_measured"] is True
    assert result["cuda_descriptor_traversal_us"] >= 0.0


def _assert_non_finite_metrics_are_rejected(root: Path) -> None:
    rows = _rows()
    for row in rows:
        if row["tilepo_policy"] == "tilepo_atg_tc_baa" and row["workload"] == "mixed" and row["experts_per_layer"] == 8:
            row["tok_per_sec"] = "nan"
            break
    _write_raw_files(root, rows)
    bad_manifest = root / "bad_adaptive_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": rows}, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "bad_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "non-finite tok_per_sec" in proc.stderr


def _assert_v0_2_requires_strict_win(root: Path) -> None:
    rows = _rows()
    for row in rows:
        if row["tilepo_policy"] == "tilepo_atg_tc_baa" and row["workload"] == "mixed" and row["experts_per_layer"] == 8:
            row["tok_per_sec"] = 132.0
            break
    _write_raw_files(root, rows)
    bad_manifest = root / "no_strict_win_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": rows}, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "no_strict_win_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "V0.2 tok/s does not strictly beat best fixed TilePO policy" in proc.stderr


def _assert_v0_2_requires_native_tc_descriptor_evidence(root: Path) -> None:
    rows = _rows()
    for row in rows:
        if row["tilepo_policy"] == "tilepo_atg_tc_baa" and row["workload"] == "mixed" and row["experts_per_layer"] == 8:
            row.pop("tc_native_descriptor_count", None)
            row.pop("tc_native_consumption_source", None)
            break
    _write_raw_files(root, rows)
    bad_manifest = root / "missing_v2_backend_evidence_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": rows}, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "missing_v2_backend_evidence_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "missing V0.2 native TC descriptor count" in proc.stderr
    assert "V0.2 native TC consumption source is not KT/CUDA" in proc.stderr


def _assert_v0_2_requires_native_tc_consumption(root: Path) -> None:
    rows = _rows()
    for row in rows:
        if row["tilepo_policy"] == "tilepo_atg_tc_baa" and row["workload"] == "mixed" and row["experts_per_layer"] == 8:
            row["tc_native_consumed"] = False
            break
    _write_raw_files(root, rows)
    bad_manifest = root / "python_cuda_shim_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": rows}, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "python_cuda_shim_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "missing V0.2 native TC consumption evidence" in proc.stderr


def _assert_v0_2_requires_native_tc_adapter_path(root: Path) -> None:
    rows = _rows()
    for row in rows:
        if row["tilepo_policy"] == "tilepo_atg_tc_baa" and row["workload"] == "mixed" and row["experts_per_layer"] == 8:
            row["serving_hook_replaced_count"] = 0
            row["serving_hook_returned_original"] = True
            row["serving_hook_fallback_count"] = 1
            row["plain_kt_fallback_events"] = 1
            break
    _write_raw_files(root, rows)
    bad_manifest = root / "observe_only_hook_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": rows}, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "observe_only_hook_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "V0.2 native TC did not replace the measured serving path" in proc.stderr
    assert "V0.2 native TC replacement count is zero" in proc.stderr


def _assert_v0_2_rejects_masked_fallback_counters(root: Path) -> None:
    rows = _rows()
    for row in rows:
        if row["tilepo_policy"] == "tilepo_atg_tc_baa" and row["workload"] == "mixed" and row["experts_per_layer"] == 8:
            row["plain_kt_fallback_events"] = 0
            row["fallback_count"] = 1
            row["serving_hook_fallback_count"] = 1
            break
    _write_raw_files(root, rows)
    bad_manifest = root / "masked_fallback_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": rows}, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "masked_fallback_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "V0.2 used fallback counter fallback_count=1" in proc.stderr
    assert "V0.2 used fallback counter serving_hook_fallback_count=1" in proc.stderr


def _assert_v0_2_requires_runtime_efficiency_evidence(root: Path) -> None:
    rows = _rows()
    for row in rows:
        if row["tilepo_policy"] == "tilepo_atg_tc_baa" and row["workload"] == "mixed" and row["experts_per_layer"] == 8:
            row.pop("baa_critical_path_us", None)
            row.pop("cuda_launch_count", None)
            row.pop("cuda_descriptor_traversal_us", None)
            break
    _write_raw_files(root, rows)
    bad_manifest = root / "missing_v2_runtime_evidence_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": rows}, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "missing_v2_runtime_evidence_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "missing V0.2 BAA critical-path metric" in proc.stderr
    assert "missing V0.2 CUDA launch metric" in proc.stderr
    assert "missing V0.2 CUDA descriptor traversal metric" in proc.stderr


def _assert_v0_2_requires_measured_runtime_metric_provenance(root: Path) -> None:
    rows = _rows()
    for row in rows:
        if row["tilepo_policy"] == "tilepo_atg_tc_baa" and row["workload"] == "mixed" and row["experts_per_layer"] == 8:
            row["baa_metrics_measured"] = False
            row["cuda_descriptor_metrics_measured"] = False
            row["runtime_metrics_source"] = "side_probe"
            break
    _write_raw_files(root, rows)
    bad_manifest = root / "unmeasured_runtime_metrics_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": rows}, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "unmeasured_runtime_metrics_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "V0.2 runtime metrics source is disconnected side_probe" in proc.stderr
    assert "V0.2 BAA metric is not marked measured" in proc.stderr
    assert "V0.2 CUDA descriptor metric is not marked measured" in proc.stderr


def _assert_extra_matrix_rows_are_rejected(root: Path) -> None:
    rows = _rows()
    extra = dict(rows[0])
    extra["system"] = "B"
    extra["tilepo_async_planning"] = "off"
    extra["raw_path"] = "raw/extra_non_matrix_row.jsonl"
    rows.append(extra)
    _write_raw_files(root, rows)
    bad_manifest = root / "extra_row_manifest.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "schema_version": "tilepo_merged_manifest_v1",
                "expected_result_rows": 24,
                "actual_result_rows": len(rows),
                "runs": rows,
            },
            indent=2,
        )
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "extra_row_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "unexpected non-matrix row" in proc.stderr


def _assert_real_command_failure_writes_blocked_manifest(root: Path) -> None:
    from tilepo import sweep

    original_probe = sweep._probe_environment
    original_linux_available = sweep._linux_available_gib
    original_subprocess_run = sweep.subprocess.run

    def ready_probe(**kwargs) -> dict:
        return {
            "ready": True,
            "model_path": kwargs["model_dir"],
            "init_path": kwargs["init_path"],
            "c_init_path": kwargs.get("c_init_path"),
            "kt_env": kwargs["kt_env"],
            "bench_tool": str(kwargs["bench_tool"]),
            "blockers": [],
        }

    def fail_run(command, **kwargs):
        raise subprocess.CalledProcessError(137, command)

    out_dir = root / "failed_sweep"
    try:
        sweep._probe_environment = ready_probe
        sweep._linux_available_gib = lambda: 64.0
        sweep.subprocess.run = fail_run
        result = sweep.run_sweep(
            "verify",
            ROOT / "configs" / "tilepo_olmoe_bf16_only.tmem",
            out_dir,
            workloads=["mixed"],
            experts=[6],
            repeats=1,
            require_real=True,
            execute=True,
            bench_tool=ROOT / "tools" / "report_tilepo_adaptive_granularity",
            kt_env="fake-ready-env",
            systems=["B"],
            request_count=1,
            warmup_request_count=0,
            output_tokens=1,
        )
    finally:
        sweep._probe_environment = original_probe
        sweep._linux_available_gib = original_linux_available
        sweep.subprocess.run = original_subprocess_run

    assert result["blocked"] is True
    assert "returncode 137" in "; ".join(result["blockers"])
    manifest = json.loads((out_dir / "tilepo_sweep_manifest.json").read_text())
    assert manifest["blocked"] is True
    assert manifest["checkpoint"] is True
    assert manifest["failed_command_run"]["returncode"] == 137
    assert manifest["expected_result_rows"] == 1
    assert manifest["actual_result_rows"] == 0


def _assert_real_failed_row_writes_blocked_manifest(root: Path) -> None:
    from tilepo import sweep

    original_probe = sweep._probe_environment
    original_linux_available = sweep._linux_available_gib
    original_subprocess_run = sweep.subprocess.run

    def ready_probe(**kwargs) -> dict:
        return {
            "ready": True,
            "model_path": kwargs["model_dir"],
            "init_path": kwargs["init_path"],
            "c_init_path": kwargs.get("c_init_path"),
            "kt_env": kwargs["kt_env"],
            "bench_tool": str(kwargs["bench_tool"]),
            "blockers": [],
        }

    def write_failed_row(command, **kwargs):
        out_path = Path(command[command.index("--out") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "status": "failed_with_log",
                    "simulated": False,
                    "evidence_level": "real",
                    "failure_reason": "synthetic failed row",
                    "tok_per_sec": 0.0,
                }
            )
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0)

    out_dir = root / "failed_row_sweep"
    try:
        sweep._probe_environment = ready_probe
        sweep._linux_available_gib = lambda: 64.0
        sweep.subprocess.run = write_failed_row
        result = sweep.run_sweep(
            "verify",
            ROOT / "configs" / "tilepo_olmoe_bf16_only.tmem",
            out_dir,
            workloads=["mixed"],
            experts=[6],
            repeats=1,
            require_real=True,
            execute=True,
            bench_tool=ROOT / "tools" / "report_tilepo_adaptive_granularity",
            kt_env="fake-ready-env",
            systems=["B"],
            request_count=1,
            warmup_request_count=0,
            output_tokens=1,
        )
    finally:
        sweep._probe_environment = original_probe
        sweep._linux_available_gib = original_linux_available
        sweep.subprocess.run = original_subprocess_run

    assert result["blocked"] is True
    assert "failed_with_log" in "; ".join(result["blockers"])
    manifest = json.loads((out_dir / "tilepo_sweep_manifest.json").read_text())
    assert manifest["blocked"] is True
    assert manifest["failed_command_run"]["row_status"] == "failed_with_log"
    assert manifest["actual_result_rows"] == 1


def _assert_bench_failure_exits_nonzero(root: Path) -> None:
    prompts = root / "bench_failure_prompts.txt"
    prompts.write_text("hello\n")
    out = root / "bench_failure.jsonl"
    log = root / "bench_failure.log"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "openai_varprompt_bench"),
            "--out",
            str(out),
            "--log",
            str(log),
            "--system",
            "B",
            "--run-name",
            "bench_failure_exit",
            "--model",
            "OLMoE-1B-7B",
            "--request-count",
            "1",
            "--warmup-request-count",
            "0",
            "--output-tokens",
            "1",
            "--startup-timeout-sec",
            "1",
            "--prompts-file",
            str(prompts),
            "--port",
            "9",
            "--server-command",
            sys.executable,
            "-c",
            "import sys; sys.exit(17)",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert rows[0]["status"] == "failed_with_log"


def _assert_offline_runner_preflight_is_local_only(root: Path) -> None:
    model_dir = root / "local_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "olmoe"}))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    init_path = root / "hotset.pt"
    init_path.write_bytes(b"local hotset")
    bench_tool = root / "bench.py"
    bench_tool.write_text("#!/usr/bin/env python3\n")
    fake_bin = root / "bin"
    fake_bin.mkdir()
    conda = fake_bin / "conda"
    conda.write_text("#!/usr/bin/env bash\nexit 0\n")
    conda.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_adaptive_granularity_offline.sh"),
            "--preflight-only",
            "--execute",
            "--strict-native-tc",
            "--offline-acceptance",
            "--ignore-active-run",
            "--skip-gpu-check",
            "--skip-quick-verify",
            "--model-dir",
            str(model_dir),
            "--init-expert-location",
            str(init_path),
            "--bench-tool",
            str(bench_tool),
            "--kt-env",
            "fake-kt-env",
            "--min-linux-available-gib",
            "0",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == "tilemem_adaptive_offline_preflight_v1"
    assert payload["status"] == "passed"
    assert payload["strict_native_tc"] is True
    assert payload["offline_acceptance"] is True
    assert payload["native_tc_preflight"]["tc_native_consumed_coalesced_groups"] is True
    assert payload["native_tc_preflight"]["tc_native_descriptor_count"] == 8
    assert payload["offline_env"]["HF_HUB_OFFLINE"] == "1"
    assert payload["offline_env"]["TILEPO_DISABLE_NETWORK"] == "1"
    assert payload["execute"] is True
    assert payload["preflight_only"] is True


def _assert_offline_runner_wires_resume() -> None:
    offline = (ROOT / "scripts" / "run_adaptive_granularity_offline.sh").read_text()
    reproduce = (ROOT / "scripts" / "reproduce_adaptive_granularity.sh").read_text()
    assert "--resume" in offline
    assert "command+=(--resume)" in offline
    assert "--resume" in reproduce
    assert "TILEMEM_ADAPTIVE_RESUME" in reproduce
    assert "if not resume:" in reproduce
    assert "--min-linux-available-gib" in reproduce
    assert "TILEMEM_ADAPTIVE_MIN_LINUX_AVAILABLE_GIB" in reproduce
    assert "command+=(--min-linux-available-gib \"$MIN_LINUX_AVAILABLE_GIB\")" in offline
    assert "min_linux_available_gib=min_linux_available_gib" in reproduce
    assert "shutil.rmtree(RUNS_DIR" in reproduce


def _assert_v0_2_offline_packager_writes_required_entrypoint() -> None:
    packager = ROOT / "scripts" / "package_tilepo_v0_2_offline_experiment.sh"
    text = packager.read_text()
    assert "TILEMEM_OFFLINE=1" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "scripts/run_adaptive_granularity_offline.sh" in text
    assert "--execute" in text
    assert "--strict-v0-2-win" in text


def _assert_sweep_uses_local_bench_and_disables_windows_host_guard(root: Path) -> None:
    from tilepo.sweep import build_tilepo_bench_command

    local_bench = ROOT / "tools" / "openai_varprompt_bench"
    assert local_bench.exists(), "adaptive offline runs must package openai_varprompt_bench"

    run = build_tilepo_bench_command(
        out_dir=root / "bench_command",
        workload="mixed",
        repeat=0,
        experts=6,
        system="C",
        port=35100,
        model_dir="/local/model",
        init_path="/local/hotset.pt",
        tilepo_manifest_path="/local/plan.manifest.json",
        mode="verify",
        bench_tool=local_bench,
        repo_root=ROOT,
        kt_env="fake-kt-env",
        request_count=1,
        warmup_request_count=0,
        output_tokens=1,
        ablation_policy="tilepo_atg_tc_baa",
        async_planning_mode="on",
    )
    command = run["command"]
    assert command[1] == str(local_bench)
    host_index = command.index("--max-host-commit-percent")
    assert command[host_index + 1] == "100"
    vmmem_index = command.index("--max-vmmem-gib")
    assert command[vmmem_index + 1] == "0"


def _assert_reproduction_scripts_wire_strict_native_tc_offline_acceptance() -> None:
    offline = (ROOT / "scripts" / "run_adaptive_granularity_offline.sh").read_text()
    reproduce = (ROOT / "scripts" / "reproduce_adaptive_granularity.sh").read_text()
    for script in (offline, reproduce):
        assert "--strict-native-tc" in script
        assert "--offline-acceptance" in script
        assert "TILEPO_STRICT_NATIVE_TC" in script
        assert "TILEPO_OFFLINE_ACCEPTANCE" in script
        assert "tc_native_consumed_coalesced_groups" in script
    assert "HF_HUB_OFFLINE=1" in offline
    assert "TRANSFORMERS_OFFLINE=1" in offline
    assert "HF_DATASETS_OFFLINE=1" in offline
    assert "TILEPO_DISABLE_NETWORK=1" in offline


def _assert_reproduction_scripts_wire_repeats() -> None:
    offline = (ROOT / "scripts" / "run_adaptive_granularity_offline.sh").read_text()
    reproduce = (ROOT / "scripts" / "reproduce_adaptive_granularity.sh").read_text()
    report = (ROOT / "tilepo" / "reporting" / "adaptive_granularity.py").read_text()
    for script in (offline, reproduce):
        assert "--repeats" in script
        assert "REPEATS" in script
    assert "command+=(--repeats \"$REPEATS\")" in offline
    assert "TILEMEM_ADAPTIVE_REPEATS" in reproduce
    assert "repeats=repeats" in reproduce
    assert '"repeats": repeats' in reproduce
    assert "expected_rows = len(workloads) * len(experts) * len(policies) * repeats" in report


def _assert_sweep_uses_kt_preserving_native_tc_for_v0_2(root: Path) -> None:
    from tilepo import env as tilepo_env
    from tilepo.sweep import build_tilepo_bench_command

    local_bench = ROOT / "tools" / "openai_varprompt_bench"
    v2_run = build_tilepo_bench_command(
        out_dir=root / "v2_bench_command",
        workload="mixed",
        repeat=0,
        experts=6,
        system="C",
        port=35101,
        model_dir="/local/model",
        init_path="/local/hotset.pt",
        tilepo_manifest_path="/local/plan.manifest.json",
        mode="verify",
        bench_tool=local_bench,
        repo_root=ROOT,
        kt_env="fake-kt-env",
        request_count=1,
        warmup_request_count=0,
        output_tokens=1,
        ablation_policy="tilepo_atg_tc_baa",
        async_planning_mode="on",
    )
    v2_command = v2_run["command"]
    assert f"{tilepo_env.TILEPO_REQUIRE_NATIVE_BACKEND}=1" in v2_command
    assert f"{tilepo_env.TILEPO_HOOK_BACKEND_PROBE_LIMIT}=1" in v2_command

    fixed_run = build_tilepo_bench_command(
        out_dir=root / "fixed_bench_command",
        workload="mixed",
        repeat=0,
        experts=6,
        system="C",
        port=35102,
        model_dir="/local/model",
        init_path="/local/hotset.pt",
        tilepo_manifest_path="/local/plan.manifest.json",
        mode="verify",
        bench_tool=local_bench,
        repo_root=ROOT,
        kt_env="fake-kt-env",
        request_count=1,
        warmup_request_count=0,
        output_tokens=1,
        ablation_policy="tilepo_coarse",
        async_planning_mode="on",
    )
    fixed_command = fixed_run["command"]
    assert f"{tilepo_env.TILEPO_REQUIRE_NATIVE_BACKEND}=1" not in fixed_command
    assert f"{tilepo_env.TILEPO_SERVE_REPLACE}=1" not in fixed_command


def _assert_self_ablation_preserves_kt_optimizations_consistently(root: Path) -> None:
    from tilepo.sweep import build_tilepo_bench_command

    local_bench = ROOT / "tools" / "openai_varprompt_bench"
    common = {
        "workload": "mixed",
        "repeat": 0,
        "experts": 8,
        "system": "C",
        "model_dir": "/local/model",
        "init_path": "/local/hotset.pt",
        "tilepo_manifest_path": "/local/plan.manifest.json",
        "mode": "verify",
        "bench_tool": local_bench,
        "repo_root": ROOT,
        "kt_env": "fake-kt-env",
        "request_count": 1,
        "warmup_request_count": 0,
        "output_tokens": 8,
        "async_planning_mode": "on",
    }
    fixed = build_tilepo_bench_command(
        out_dir=root / "fixed_same_shell",
        port=35200,
        ablation_policy="tilepo_fine",
        **common,
    )
    v2 = build_tilepo_bench_command(
        out_dir=root / "v2_same_shell",
        port=35201,
        ablation_policy="tilepo_atg_tc_baa",
        **common,
    )
    disabled_flags = {
        "--skip-server-warmup",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--disable-shared-experts-fusion",
    }
    fixed_flags = disabled_flags.intersection(fixed["server_command"])
    v2_flags = disabled_flags.intersection(v2["server_command"])
    assert fixed_flags == v2_flags == disabled_flags


def _assert_bootstrap_prime_does_not_count_as_measured_replacement() -> None:
    from tilepo.kt_patch.bootstrap import _merge_native_tc_prime_into_hook

    merged = _merge_native_tc_prime_into_hook(
        {"installed": True, "serving_hook_replaced_count": 0},
        {
            "tc_native_consumed": True,
            "tc_native_consumed_coalesced_groups": True,
            "tc_native_descriptor_count": 8,
            "tc_native_entrypoint": "tilepo_cuda_dispatch_coalesced_gemm",
            "tc_native_descriptor_layout": "tilepo_cuda_coalesced_group_desc_v1",
            "tc_native_consumed_group_count": 8,
            "tc_native_consumed_tile_count": 16384,
            "tc_native_consumed_bytes": 8589934592,
            "tc_native_consumption_source": "kt_grouped_moe_cuda_adapter",
        },
    )
    assert merged["serving_hook_replaced_count"] == 0
    assert merged["serving_hook_returned_original"] is True
    assert merged["serving_hook_mode"] == "observe_only"
    assert merged["tc_native_consumed"] is True
    assert merged["tc_native_descriptor_count"] == 8
    assert merged["serving_hook_backend_prime_native_ready"] is True


def _assert_sweep_wires_force_prompt_for_focused_restore(root: Path) -> None:
    from tilepo.sweep import build_tilepo_bench_command

    prompt = "summarize the memory tradeoff in expert placement"
    local_bench = ROOT / "tools" / "openai_varprompt_bench"
    run = build_tilepo_bench_command(
        out_dir=root / "force_prompt_bench_command",
        workload="mixed",
        repeat=0,
        experts=8,
        system="C",
        port=35103,
        model_dir="/local/model",
        init_path="/local/hotset.pt",
        tilepo_manifest_path="/local/plan.manifest.json",
        mode="verify",
        bench_tool=local_bench,
        repo_root=ROOT,
        kt_env="fake-kt-env",
        request_count=5,
        warmup_request_count=5,
        output_tokens=8,
        ablation_policy="tilepo_atg_tc_baa",
        async_planning_mode="on",
        force_prompt=prompt,
    )
    command = run["command"]
    assert run["force_prompt"] == prompt
    assert "--prompts-file" not in command
    assert command.count("--prompt") == 10
    assert command[command.index("--prompt") + 1] == prompt
    assert command.index("--prompt") < command.index("--server-command")

    offline = (ROOT / "scripts" / "run_adaptive_granularity_offline.sh").read_text()
    reproduce = (ROOT / "scripts" / "reproduce_adaptive_granularity.sh").read_text()
    assert "--force-prompt" in offline
    assert "TILEMEM_ADAPTIVE_FORCE_PROMPT" in reproduce


def _assert_windows_host_guard_collection_is_bounded() -> None:
    sys.path.insert(0, str(ROOT / "tools" / "v2"))
    import runtime_safety

    calls: dict[str, float] = {}
    original_check_output = runtime_safety.subprocess.check_output

    def fake_check_output(command, **kwargs):
        calls["timeout"] = kwargs.get("timeout", 0.0)
        return json.dumps(
            {
                "committed_gib": 31.0,
                "commit_limit_gib": 56.0,
                "commit_percent": 55.4,
                "vmmem_virtual_gib": 20.0,
                "vmmem_working_set_gib": 15.0,
                "vmmem_paged_memory_gib": 8.0,
            }
        )

    try:
        runtime_safety.subprocess.check_output = fake_check_output
        info = runtime_safety.read_or_collect_host_info(None, attempts=1, timeout_sec=3.5)
    finally:
        runtime_safety.subprocess.check_output = original_check_output

    assert info["commit_percent"] == 55.4
    assert calls["timeout"] == 3.5


def _assert_runtime_env_limits_compile_parallelism(root: Path) -> None:
    sys.path.insert(0, str(ROOT / "tools" / "v2"))
    import runtime_safety

    env = runtime_safety.apply_runtime_env({}, root / "runtime_env", root / "native_tmp")
    assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"

    custom = runtime_safety.apply_runtime_env(
        {
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        },
        root / "runtime_env_custom",
        root / "native_tmp_custom",
    )
    assert custom["TORCHINDUCTOR_COMPILE_THREADS"] == "2"
    assert custom["OMP_NUM_THREADS"] == "4"
    assert custom["MKL_NUM_THREADS"] == "4"


def _assert_runtime_env_uses_shared_jit_cache(root: Path) -> None:
    sys.path.insert(0, str(ROOT / "tools" / "v2"))
    import runtime_safety

    shared_root = root / "shared_jit_cache"
    env = runtime_safety.apply_runtime_env(
        {"TILEMEM_SHARED_JIT_CACHE_DIR": str(shared_root)},
        root / "runtime_env_shared",
        root / "native_tmp_shared",
    )
    assert env["FLASHINFER_WORKSPACE_BASE"].startswith(str(shared_root))
    assert env["FLASHINFER_CUBIN_DIR"].startswith(str(shared_root))
    assert env["TRITON_CACHE_DIR"].startswith(str(shared_root))
    assert env["TORCHINDUCTOR_CACHE_DIR"].startswith(str(shared_root))
    assert env["HF_HOME"].startswith(str((root / "runtime_env_shared").resolve()))

    offline = (ROOT / "scripts" / "run_adaptive_granularity_offline.sh").read_text()
    assert "TILEMEM_SHARED_JIT_CACHE_DIR" in offline


def _write_raw_files(root: Path, rows: list[dict]) -> None:
    for row in rows:
        raw_path = root / row["raw_path"]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(row) + "\n")


def _comparison(summary: dict, workload: str, experts: int) -> dict:
    for item in summary["comparisons"]:
        if item["workload"] == workload and item["experts_per_layer"] == experts:
            return item
    raise AssertionError(f"missing comparison for {workload}/{experts}")


def _rows() -> list[dict]:
    rows = []
    for workload in ("mixed", "long_context"):
        for experts in (6, 8, 10):
            if workload == "mixed" and experts == 8:
                fixed = {
                    "tilepo_coarse": (124.0, 83.0, 101.0, 7.2, 256, 256),
                    "tilepo_fine": (132.0, 80.0, 98.0, 6.4, 16384, 16384),
                    "tilepo_hybrid": (128.0, 82.0, 100.0, 6.9, 8200, 8200),
                    "tilepo_atg_tc_baa": (135.0, 78.0, 96.0, 6.6, 4480, 8),
                }
            elif workload == "long_context" and experts == 10:
                fixed = {
                    "tilepo_coarse": (130.0, 78.0, 96.0, 7.1, 320, 320),
                    "tilepo_fine": (124.0, 82.0, 101.0, 6.3, 20480, 20480),
                    "tilepo_hybrid": (127.0, 80.0, 98.0, 6.7, 12300, 12300),
                    "tilepo_atg_tc_baa": (133.0, 77.0, 95.0, 6.5, 4640, 10),
                }
            else:
                fixed = {
                    "tilepo_coarse": (120.0, 86.0, 104.0, 7.3, experts * 32, experts * 32),
                    "tilepo_fine": (122.0, 84.0, 102.0, 6.4, experts * 2048, experts * 2048),
                    "tilepo_hybrid": (123.0, 83.0, 101.0, 6.8, experts * 1100, experts * 1100),
                    "tilepo_atg_tc_baa": (126.0, 82.0, 100.0, 6.6, experts * 760, experts),
                }
            for policy, (tok, p95, p99, gpu, tile_count, dispatch_units) in fixed.items():
                rows.append(_row(policy, "on", "C", workload, experts, 0, tok, p95, p99, gpu, tile_count, dispatch_units))
    return rows


def _row(
    policy: str,
    async_mode: str,
    system: str,
    workload: str,
    experts: int,
    repeat: int,
    tok: float,
    p95: float,
    p99: float,
    gpu: float,
    tile_count: int | None = None,
    dispatch_units: int | None = None,
) -> dict:
    row = {
        "system": system,
        "workload": workload,
        "experts_per_layer": experts,
        "repeat": repeat,
        "request_count": 5,
        "warmup_request_count": 1,
        "tok_per_sec": tok,
        "p50_ms": p95 * 0.8,
        "p95_ms": p95,
        "p99_ms": p99,
        "gpu_peak_gib": gpu,
        "cpu_ram_peak_gib": 18.0,
        "server_ready_s": 4.0,
        "fallback_count": 0,
        "dtype_counts": {"bf16": 1},
        "command": ["python", "-m", "sglang.launch_server", "--kt-method", "BF16", "--dtype", "bfloat16"],
        "evidence_level": "real",
        "simulated": False,
        "status": "success",
        "tilepo_policy": policy,
        "tilepo_async_planning": async_mode,
        "raw_path": f"raw/{policy}_{async_mode}_{system}_{experts}_{repeat}.jsonl",
        "plan_lookup_us": 8.0,
        "cuda_launch_count": 1,
        "cuda_descriptor_traversal_us": 0.05,
    }
    if tile_count is not None:
        row["tile_count"] = tile_count
    if dispatch_units is not None:
        row["estimated_dispatch_units"] = dispatch_units
        row["execution_dispatch_units"] = dispatch_units
    if policy == "tilepo_atg_tc_baa":
        row["tilepo_backend"] = "kt_preserving_native_tc_cuda"
        row["plain_kt_fallback_events"] = 0
        row["hot_backend_native"] = True
        row["backend_owner"] = "kt_sglang"
        row["kt_executor_preserved"] = True
        row["tilepo_plan_applied_in_serving_path"] = True
        row["tc_coalescing_active"] = True
        row["tc_native_consumed"] = True
        row["tc_native_consumed_coalesced_groups"] = True
        row["tc_native_consumed_group_count"] = dispatch_units or tile_count or 1
        row["tc_native_descriptor_count"] = dispatch_units or tile_count or 1
        row["tc_native_consumed_tile_count"] = tile_count or dispatch_units or 1
        row["tc_native_consumed_bytes"] = (tile_count or dispatch_units or 1) * 4096
        row["tc_native_entrypoint"] = "tilepo_cuda_dispatch_coalesced_gemm"
        row["tc_native_descriptor_layout"] = "tilepo_cuda_coalesced_group_desc_v1"
        row["tc_native_consumption_source"] = "kt_grouped_moe_cuda_adapter"
        row["tc_native_launch_path"] = "kt_grouped_moe_cuda_adapter"
        row["tc_native_launch_count"] = 1
        row["serving_hook_active"] = True
        row["serving_hook_invocations"] = 1
        row["serving_hook_replaced_count"] = 1
        row["serving_hook_fallback_count"] = 0
        row["serving_hook_returned_original"] = False
        row["serving_hook_mode"] = "native_tc_adapter"
        row["serving_hook_replacement_real"] = True
        row["serving_hook_verify_count"] = 1
        row["serving_hook_verify_pass_count"] = 1
        row["serving_hook_verify_fail_count"] = 0
        row["serving_hook_candidate_available"] = True
        row["unexpected_plain_kt_bypass_events"] = 0
        row["baa_double_buffered"] = True
        row["baa_critical_path_us"] = 0.0
        row["runtime_metrics_source"] = "kt_preserving_native_tc_kernel"
        row["baa_metrics_measured"] = True
        row["cuda_descriptor_metrics_measured"] = True
    return row


if __name__ == "__main__":
    raise SystemExit(main())
