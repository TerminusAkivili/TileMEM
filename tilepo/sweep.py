from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from . import env as tilepo_env
from .compiler import compile_plan


DEFAULT_WORKLOADS = ["anchor_unique", "profile_matched", "mixed", "long_output"]
DEFAULT_EXPERTS = [4, 6, 8, 10, 12, 16]
DEFAULT_MODEL_DIR = "/mnt/d/tilemem_runtime/models/OLMoE-1B-7B-0924-Instruct"
DEFAULT_INIT = "/mnt/d/tilemem_runtime/results/kt_tilemem_hotset_20260523/tilemem_hotset_counts.pt"
DEFAULT_KT_ENV = "tilemem-tilepo-ktransformers"
DEFAULT_BENCH_TOOL_CANDIDATES = [
    Path("tools/openai_varprompt_bench"),
]
C_MODE_CHOICES = ("hook", "kt_native")


def run_sweep(
    mode: str,
    plan_path: Path,
    out_dir: Path,
    workloads: list[str] | None = None,
    experts: list[int] | None = None,
    repeats: int = 3,
    require_real: bool = False,
    dry_run_commands: bool = False,
    execute: bool = False,
    base_port: int = 34000,
    model_dir: str = DEFAULT_MODEL_DIR,
    init_path: str = DEFAULT_INIT,
    c_init_path: str | None = None,
    kt_env: str = DEFAULT_KT_ENV,
    bench_tool: Path | None = None,
    systems: list[str] | None = None,
    request_count: int = 4,
    warmup_request_count: int = 2,
    output_tokens: int = 4,
    startup_timeout_sec: int = 900,
    min_c_free_gib: float = 20.0,
    min_d_free_gib: float = 20.0,
    max_host_commit_percent: float = 95.0,
    max_vmmem_gib: float = 0.0,
    min_linux_available_gib: float = 8.0,
    skip_existing_success: bool = False,
    c_mode: str = "hook",
    ablation_policy: str = "",
    async_planning_mode: str = "",
) -> dict[str, Any]:
    _validate_c_mode(c_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    compile_result = compile_plan(plan_path, out_dir / "compiled_plan")
    bench_tool = bench_tool or _find_bench_tool()
    env = _probe_environment(
        model_dir=model_dir,
        init_path=init_path,
        c_init_path=c_init_path,
        kt_env=kt_env,
        bench_tool=bench_tool,
    )
    manifest_path = out_dir / "tilepo_sweep_manifest.json"
    if require_real and not execute:
        blockers = ["--require-real needs --execute; dry-run command manifests are not real evidence"]
        manifest = {
            "schema_version": "tilepo_sweep_manifest_v1",
            "mode": mode,
            "c_mode": c_mode,
            "simulated": False,
            "blocked": True,
            "blockers": blockers,
            "environment": env,
            "compiled_manifest": str(compile_result.manifest_path),
            "serving_shell": "KT/SGLang",
            "systems": ["A", "B", "C"],
            "c_init_path": c_init_path,
            "ablation_policy": ablation_policy,
            "async_planning_mode": async_planning_mode,
            "runs": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {"manifest_path": str(manifest_path), "blocked": True, "blockers": blockers}
    if require_real and not env["ready"]:
        manifest = {
            "schema_version": "tilepo_sweep_manifest_v1",
            "mode": mode,
            "c_mode": c_mode,
            "simulated": False,
            "blocked": True,
            "blockers": env["blockers"],
            "environment": env,
            "compiled_manifest": str(compile_result.manifest_path),
            "serving_shell": "KT/SGLang",
            "systems": ["A", "B", "C"],
            "c_init_path": c_init_path,
            "ablation_policy": ablation_policy,
            "async_planning_mode": async_planning_mode,
            "runs": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {"manifest_path": str(manifest_path), "blocked": True, "blockers": env["blockers"]}
    linux_available_gib = _linux_available_gib()
    if execute and linux_available_gib < min_linux_available_gib:
        blockers = [
            (
                f"Linux available memory {linux_available_gib:.2f} GiB is below "
                f"required {min_linux_available_gib:.2f} GiB for KT/SGLang cold start"
            )
        ]
        manifest = {
            "schema_version": "tilepo_sweep_manifest_v1",
            "mode": mode,
            "c_mode": c_mode,
            "simulated": False,
            "blocked": True,
            "blockers": blockers,
            "environment": env,
            "compiled_manifest": str(compile_result.manifest_path),
            "serving_shell": "KT/SGLang",
            "systems": ["A", "B", "C"],
            "c_init_path": c_init_path,
            "ablation_policy": ablation_policy,
            "async_planning_mode": async_planning_mode,
            "runs": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if require_real:
            return {"manifest_path": str(manifest_path), "blocked": True, "blockers": blockers}
        raise RuntimeError(blockers[0])

    selected_workloads = workloads or DEFAULT_WORKLOADS
    selected_experts = experts or DEFAULT_EXPERTS
    selected_systems = systems or ["A", "B", "C"]
    write_prompts(out_dir, selected_workloads)
    command_runs = _command_runs(
        mode=mode,
        out_dir=out_dir,
        workloads=selected_workloads,
        experts=selected_experts,
        repeats=repeats,
        base_port=base_port,
        model_dir=model_dir,
        init_path=init_path,
        c_init_path=c_init_path,
        tilepo_manifest_path=str(compile_result.manifest_path),
        bench_tool=bench_tool,
        repo_root=Path(__file__).resolve().parents[1],
        kt_env=kt_env,
        systems=selected_systems,
        request_count=request_count,
        warmup_request_count=warmup_request_count,
        output_tokens=output_tokens,
        startup_timeout_sec=startup_timeout_sec,
        min_c_free_gib=min_c_free_gib,
        min_d_free_gib=min_d_free_gib,
        max_host_commit_percent=max_host_commit_percent,
        max_vmmem_gib=max_vmmem_gib,
        min_linux_available_gib=min_linux_available_gib,
        c_mode=c_mode,
        ablation_policy=ablation_policy,
        async_planning_mode=async_planning_mode,
    )

    if execute:
        if not env["ready"]:
            raise RuntimeError("cannot execute real KT/SGLang sweep: " + "; ".join(env["blockers"]))
        skipped_existing_runs = 0
        for run in command_runs:
            if skip_existing_success and _mark_existing_success(run):
                skipped_existing_runs += 1
                _write_sweep_checkpoint(
                    manifest_path,
                    mode=mode,
                    c_mode=c_mode,
                    simulated=False,
                    env=env,
                    compile_result=compile_result,
                    selected_systems=selected_systems,
                    selected_workloads=selected_workloads,
                    selected_experts=selected_experts,
                    repeats=repeats,
                    command_runs=command_runs,
                    skipped_existing_runs=skipped_existing_runs,
                    c_init_path=c_init_path,
                    ablation_policy=ablation_policy,
                    async_planning_mode=async_planning_mode,
                )
                continue
            subprocess.run(run["command"], cwd=Path(__file__).resolve().parents[1], env=os.environ.copy(), check=True)
            _write_sweep_checkpoint(
                manifest_path,
                mode=mode,
                c_mode=c_mode,
                simulated=False,
                env=env,
                compile_result=compile_result,
                selected_systems=selected_systems,
                selected_workloads=selected_workloads,
                selected_experts=selected_experts,
                repeats=repeats,
                command_runs=command_runs,
                skipped_existing_runs=skipped_existing_runs,
                c_init_path=c_init_path,
                ablation_policy=ablation_policy,
                async_planning_mode=async_planning_mode,
            )
        rows = _load_real_rows(command_runs)
        simulated = False
    else:
        skipped_existing_runs = 0
    if not execute and dry_run_commands:
        rows = []
        simulated = True
    elif not execute:
        rows = _fixture_rows(
            selected_workloads,
            selected_experts,
            repeats,
            mode,
            ablation_policy=ablation_policy,
            async_planning_mode=async_planning_mode,
        )
        simulated = not require_real

    manifest = {
        "schema_version": "tilepo_sweep_manifest_v1",
        "mode": mode,
        "c_mode": c_mode,
        "simulated": simulated,
        "blocked": False,
        "environment": env,
        "compiled_manifest": str(compile_result.manifest_path),
        "serving_shell": "KT/SGLang",
        "systems": ["A", "B", "C"],
        "selected_systems": selected_systems,
        "selected_workloads": selected_workloads,
        "selected_experts": selected_experts,
        "selected_repeats": repeats,
        "c_init_path": c_init_path,
        "ablation_policy": ablation_policy,
        "async_planning_mode": async_planning_mode,
        "expected_command_runs": len(command_runs),
        "expected_result_rows": len(command_runs),
        "command_runs": command_runs,
        "skipped_existing_runs": skipped_existing_runs,
        "runs": rows,
        "created_at_unix": time.time(),
        "checkpoint": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "manifest_path": str(manifest_path),
        "blocked": False,
        "c_mode": c_mode,
        "runs": len(rows),
        "command_runs": len(command_runs),
        "skipped_existing_runs": skipped_existing_runs,
    }


def build_kt_sglang_server_command(
    *,
    port: int,
    experts: int,
    system: str,
    model_dir: str,
    init_path: str,
    tilepo_manifest_path: str,
    mode: str,
    kt_env: str = DEFAULT_KT_ENV,
) -> list[str]:
    strategy = "uniform" if system == "A" else "frequency"
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        kt_env,
        "python",
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        model_dir,
        "--served-model-name",
        "tilemem-active",
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--context-length",
        "64",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        "0.70",
        "--max-running-requests",
        "1",
        "--max-total-tokens",
        "128",
        "--max-prefill-tokens",
        "64",
        "--kt-weight-path",
        model_dir,
        "--kt-method",
        "BF16",
        "--kt-cpuinfer",
        "0",
        "--kt-threadpool-count",
        "1",
        "--kt-num-gpu-experts",
        str(experts),
        "--kt-expert-placement-strategy",
        strategy,
    ]
    if system in {"B", "C"}:
        cmd.extend(["--init-expert-location", init_path])
    cmd.extend(
        [
            "--skip-server-warmup",
            "--disable-radix-cache",
            "--disable-overlap-schedule",
            "--disable-cuda-graph",
            "--disable-shared-experts-fusion",
        ]
    )
    return cmd


def build_tilepo_bench_command(
    *,
    out_dir: Path,
    workload: str,
    repeat: int,
    experts: int,
    system: str,
    port: int,
    model_dir: str,
    init_path: str,
    tilepo_manifest_path: str,
    mode: str,
    bench_tool: Path,
    repo_root: Path,
    kt_env: str = DEFAULT_KT_ENV,
    c_init_path: str | None = None,
    request_count: int = 4,
    warmup_request_count: int = 2,
    output_tokens: int = 4,
    startup_timeout_sec: int = 900,
    min_c_free_gib: float = 20.0,
    min_d_free_gib: float = 20.0,
    max_host_commit_percent: float = 95.0,
    max_vmmem_gib: float = 0.0,
    min_linux_available_gib: float = 8.0,
    c_mode: str = "hook",
    ablation_policy: str = "",
    async_planning_mode: str = "",
) -> dict[str, Any]:
    _validate_c_mode(c_mode)
    system_name = {"A": "kt_uniform", "B": "kt_tilemem_placement", "C": "kt_sglang_tilepo"}[system]
    suffix_parts = []
    if ablation_policy:
        suffix_parts.append(_safe_name(ablation_policy))
    if async_planning_mode:
        suffix_parts.append(f"async{_safe_name(async_planning_mode)}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    run_name = f"{system_name}_experts{experts}_{workload}{suffix}_rep{repeat}"
    jsonl = out_dir / "raw" / f"{run_name}.jsonl"
    log = out_dir / "raw" / f"{run_name}.log"
    plugin = out_dir / "raw" / f"{run_name}.plugin.json"
    runtime_dir = out_dir / "runtime" / run_name
    native_tmp = Path("/tmp") / f"tilepo_{run_name}"
    prompts_file = out_dir / "prompts" / f"{workload}.txt"
    server_system = "B" if system == "C" and c_mode == "kt_native" else system
    effective_init_path = c_init_path if system == "C" and c_init_path else init_path
    server = build_kt_sglang_server_command(
        port=port,
        experts=experts,
        system=server_system,
        model_dir=model_dir,
        init_path=effective_init_path,
        tilepo_manifest_path=tilepo_manifest_path,
        mode=mode,
        kt_env=kt_env,
    )
    run_id = f"{run_name}-{uuid.uuid4().hex}"
    extra_env = []
    if system == "C" and c_mode == "hook":
        marker = out_dir / "raw" / f"{run_name}.tilepo_bootstrap.json"
        pythonpath = os.pathsep.join([str(repo_root), str(bench_tool.resolve().parents[1])])
        extra_env = [
            "--extra-env",
            f"{tilepo_env.TILEPO_ENABLE}=1",
            "--extra-env",
            f"{tilepo_env.TILEPO_MANIFEST}={tilepo_manifest_path}",
            "--extra-env",
            f"{tilepo_env.TILEPO_MODE}={mode}",
            "--extra-env",
            f"{tilepo_env.TILEPO_BACKEND}=cuda,tilelang,kt_fallback",
            "--extra-env",
            f"{tilepo_env.TILEPO_BOOTSTRAP_MARKER}={marker}",
            "--extra-env",
            f"{tilepo_env.TILEPO_RUN_ID}={run_id}",
            "--extra-env",
            f"{tilepo_env.TILEPO_POLICY}={ablation_policy}",
            "--extra-env",
            f"{tilepo_env.TILEPO_ASYNC_PLANNING}={async_planning_mode}",
            "--extra-env",
            f"PYTHONPATH={pythonpath}",
        ]
    command = [
        "python3",
        str(bench_tool),
        "--out",
        str(jsonl),
        "--log",
        str(log),
        "--system",
        system,
        "--run-name",
        run_name,
        "--model",
        "OLMoE-1B-7B",
        "--host",
        "127.0.0.1",
        "--served-model-name",
        "tilemem-active",
        "--request-count",
        str(request_count),
        "--warmup-request-count",
        str(warmup_request_count),
        "--output-tokens",
        str(output_tokens),
        "--startup-timeout-sec",
        str(startup_timeout_sec),
        "--request-timeout-sec",
        "300",
        "--evidence-level",
        "real",
        "--port",
        str(port),
        "--prompts-file",
        str(prompts_file),
        "--runtime-dir",
        str(runtime_dir),
        "--native-tmp-dir",
        str(native_tmp),
        "--plugin-out",
        str(plugin),
        "--min-c-free-gib",
        _format_number(min_c_free_gib),
        "--min-d-free-gib",
        _format_number(min_d_free_gib),
        "--max-host-commit-percent",
        _format_number(max_host_commit_percent),
        "--max-vmmem-gib",
        _format_number(max_vmmem_gib),
        *extra_env,
        "--server-command",
        *server,
    ]
    return {
        "system": system,
        "system_name": system_name,
        "c_mode": c_mode,
        "workload": workload,
        "repeat": repeat,
        "experts_per_layer": experts,
        "port": port,
        "jsonl": str(jsonl),
        "log": str(log),
        "plugin": str(plugin),
        "command": command,
        "server_command": server,
        "init_path": init_path,
        "effective_init_path": effective_init_path,
        "c_init_path": c_init_path,
        "run_id": run_id,
        "ablation_policy": ablation_policy,
        "async_planning_mode": async_planning_mode,
        "tilepo_policy": ablation_policy,
        "tilepo_async_planning": async_planning_mode,
    }


def write_prompts(out_dir: Path, workloads: list[str]) -> None:
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_sets = {
        "anchor_unique": [
            "ember fjord granite harbor iris juniper kernel lagoon",
            "frost grove helium inlet jade kelp lilac mesa",
            "glacier harbor iris juniper kelp lilac meadow nova",
            "harbor iris jade kelp lilac mesa nova orbit",
            "iris juniper kelp lilac mesa nova orbit prism",
            "juniper kelp lilac mesa nova orbit prism quartz",
        ],
        "profile_matched": ["Hello", "Hello", "Hello", "Hello", "Hello", "Hello"],
        "mixed": [
            "explain the routing behavior of a sparse mixture of experts model",
            "summarize the memory tradeoff in expert placement",
            "write a short C++ function for prefix lookup",
            "solve a tiny arithmetic puzzle with intermediate reasoning",
            "compare two GPU expert placement policies in one paragraph",
            "list the evidence needed for a reliable serving benchmark",
        ],
        "long_output": [
            "write a detailed paragraph about GPU memory residency for MoE inference",
            "compare static and frequency based expert placement in detail",
            "describe an experimental method for measuring p95 latency",
            "outline limitations of a small benchmark matrix",
            "explain why repeated measurements matter for serving systems research",
            "write a careful limitation section for a routing-aware placement study",
        ],
        "long_context": [
            "Given a routing histogram from a sparse mixture of experts server, explain how GPU residency, fallback traffic, and request shape interact during decoding.",
            "In a memory constrained MoE deployment, compare expert level placement with tile level placement when hot experts are stable but cold experts still appear.",
            "For a serving benchmark with repeated prompts, describe how request count, warmup, p95 latency, p99 latency, GPU memory, and CPU memory should be reported.",
            "Analyze a deployment where the model uses BF16 execution, a fixed router, and a variable GPU expert budget while preserving output quality.",
            "Summarize why a scheduler should admit TilePO only when measured throughput and tail latency beat the KT fallback path under the same expert budget.",
            "Write a careful systems paragraph about asynchronous planning, metadata overhead, kernel efficiency, and VRAM DRAM residency in MoE inference.",
        ],
    }
    for workload in workloads:
        prompts = prompt_sets.get(workload, prompt_sets["anchor_unique"])
        (prompts_dir / f"{workload}.txt").write_text("\n".join(prompts) + "\n")


def _command_runs(
    *,
    mode: str,
    out_dir: Path,
    workloads: list[str],
    experts: list[int],
    repeats: int,
    base_port: int,
    model_dir: str,
    init_path: str,
    c_init_path: str | None,
    tilepo_manifest_path: str,
    bench_tool: Path,
    repo_root: Path,
    kt_env: str,
    systems: list[str],
    request_count: int,
    warmup_request_count: int,
    output_tokens: int,
    startup_timeout_sec: int,
    min_c_free_gib: float,
    min_d_free_gib: float,
    max_host_commit_percent: float,
    max_vmmem_gib: float,
    min_linux_available_gib: float,
    c_mode: str,
    ablation_policy: str,
    async_planning_mode: str,
) -> list[dict[str, Any]]:
    runs = []
    port = base_port
    for workload in workloads:
        for expert_count in experts:
            for repeat in range(repeats):
                for system in systems:
                    runs.append(
                        build_tilepo_bench_command(
                            out_dir=out_dir,
                            workload=workload,
                            repeat=repeat,
                            experts=expert_count,
                            system=system,
                            port=port,
                            model_dir=model_dir,
                            init_path=init_path,
                            c_init_path=c_init_path,
                            tilepo_manifest_path=tilepo_manifest_path,
                            mode=mode,
                            bench_tool=bench_tool,
                            repo_root=repo_root,
                            kt_env=kt_env,
                            request_count=request_count,
                            warmup_request_count=warmup_request_count,
                            output_tokens=output_tokens,
                            startup_timeout_sec=startup_timeout_sec,
                            min_c_free_gib=min_c_free_gib,
                            min_d_free_gib=min_d_free_gib,
                            max_host_commit_percent=max_host_commit_percent,
                            max_vmmem_gib=max_vmmem_gib,
                            min_linux_available_gib=min_linux_available_gib,
                            c_mode=c_mode,
                            ablation_policy=ablation_policy,
                            async_planning_mode=async_planning_mode,
                        )
                    )
                    port += 1
    return runs


def _probe_environment(
    *,
    model_dir: str,
    init_path: str,
    c_init_path: str | None = None,
    kt_env: str,
    bench_tool: Path | None,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    blockers = []
    if not model_path.exists():
        blockers.append(f"missing model path: {model_path}")
    if not Path(init_path).exists():
        blockers.append(f"missing KT frequency init path: {init_path}")
    if c_init_path and not Path(c_init_path).exists():
        blockers.append(f"missing KT frequency init path for C: {c_init_path}")
    if bench_tool is None or not bench_tool.exists():
        blockers.append("missing tools/openai_varprompt_bench in this checkout")
    if shutil.which("python3") is None:
        blockers.append("python3 unavailable")
    if shutil.which("conda") is None:
        blockers.append("conda unavailable for KT/SGLang env")
    if shutil.which("nvidia-smi") is None:
        blockers.append("nvidia-smi unavailable")
    if shutil.which("conda") is not None:
        for module in ("sglang", "ktransformers"):
            proc = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    kt_env,
                    "python",
                    "-c",
                    (
                        "import importlib.util; "
                        f"raise SystemExit(0 if importlib.util.find_spec('{module}') else 3)"
                    ),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                blockers.append(f"KT/SGLang env '{kt_env}' cannot import {module}")
    return {
        "ready": not blockers,
        "model_path": str(model_path),
        "init_path": str(init_path),
        "c_init_path": c_init_path,
        "kt_env": kt_env,
        "bench_tool": str(bench_tool) if bench_tool else "",
        "blockers": blockers,
    }


def _find_bench_tool() -> Path | None:
    for candidate in DEFAULT_BENCH_TOOL_CANDIDATES:
        path = candidate if candidate.is_absolute() else Path(__file__).resolve().parents[1] / candidate
        if path.exists():
            return path
    return None


def _load_real_rows(command_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in command_runs:
        path = Path(run["jsonl"])
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["experts_per_layer"] = run["experts_per_layer"]
                row["repeat"] = run["repeat"]
                row["workload"] = run["workload"]
                row["ablation_policy"] = run.get("ablation_policy", "")
                row["async_planning_mode"] = run.get("async_planning_mode", "")
                row["tilepo_policy"] = run.get("tilepo_policy", run.get("ablation_policy", ""))
                row["tilepo_async_planning"] = run.get(
                    "tilepo_async_planning",
                    run.get("async_planning_mode", ""),
                )
                row["raw_path"] = str(path)
                row["command"] = run["command"]
                row["p50_ms"] = row.get("p50_latency_ms", row.get("p50_ms", 0.0))
                row["p95_ms"] = row.get("p95_latency_ms", row.get("p95_ms", 0.0))
                row["p99_ms"] = row.get("p99_latency_ms", row.get("p99_ms", 0.0))
                row["gpu_peak_gib"] = float(row.get("gpu_memory_peak_bytes", 0.0)) / (1024 ** 3)
                row["cpu_ram_peak_gib"] = float(row.get("cpu_memory_peak_bytes", 0.0)) / (1024 ** 3)
                row["server_ready_s"] = row.get("server_ready_after_sec", 0.0)
                hot_probe = _load_hot_backend_probe(run, path)
                if hot_probe:
                    row["hot_backend_probe_path"] = hot_probe["path"]
                    row["hot_backend_probe_status"] = hot_probe.get("status", "unknown")
                    if "failure_reason" in hot_probe:
                        row["hot_backend_probe_failure_reason"] = hot_probe["failure_reason"]
                row["runtime_overhead_us"] = row.get(
                    "runtime_overhead_us", hot_probe.get("runtime_overhead_us", 0.0)
                )
                for key in (
                    "plan_lookup_us",
                    "plan_lookup_total_us",
                    "gate_us",
                    "backend_launch_us",
                    "h2d_bytes",
                    "cache_hits",
                    "cache_misses",
                    "tile_count",
                    "async_plan_cache_hits",
                    "async_plan_cache_misses",
                ):
                    if key not in row and key in hot_probe:
                        row[key] = hot_probe[key]
                row["dtype_counts"] = row.get("dtype_counts", hot_probe.get("dtype_counts", {"bf16": 1}))
                row["fallback_count"] = row.get("fallback_count", hot_probe.get("fallback_count", 0))
                row["backend_launch_counts"] = row.get(
                    "backend_launch_counts", hot_probe.get("backend_launch_counts", {})
                )
                row["tilemem_backend_launch_count"] = row.get(
                    "tilemem_backend_launch_count", hot_probe.get("tilemem_backend_launch_count", 0)
                )
                if "hot_backend_native" not in row and "hot_backend_native" in hot_probe:
                    row["hot_backend_native"] = bool(hot_probe["hot_backend_native"])
                serving_hook = hot_probe.get("serving_hook", {})
                if isinstance(serving_hook, dict):
                    for key in (
                        "serving_hook_active",
                        "serving_hook_invocations",
                        "serving_hook_replaced_count",
                        "serving_hook_fallback_count",
                        "serving_hook_last_layer",
                        "serving_hook_last_shape",
                        "serving_hook_last_target",
                        "serving_hook_returned_original",
                        "serving_hook_replacement_blocked_reason",
                        "serving_hook_backend_launch_count",
                        "serving_hook_backend_launch_counts",
                        "serving_hook_backend_fallback_count",
                        "serving_hook_backend_dtype_counts",
                        "serving_hook_backend_h2d_bytes",
                        "serving_hook_backend_runtime_us",
                        "serving_hook_backend_result",
                        "serving_hook_backend_hot_tile",
                        "serving_hook_backend_launch_failure",
                        "serving_hook_verify_count",
                        "serving_hook_verify_pass_count",
                        "serving_hook_verify_fail_count",
                        "serving_hook_verify_max_abs_error",
                        "serving_hook_verify_shape_match",
                        "serving_hook_verify_dtype_match",
                        "serving_hook_verify_device_match",
                        "serving_hook_verify_source",
                        "serving_hook_candidate_available",
                    ):
                        if key in serving_hook and key not in row:
                            row[key] = serving_hook[key]
                rows.append(row)
    return rows


def _write_sweep_checkpoint(
    manifest_path: Path,
    *,
    mode: str,
    c_mode: str,
    simulated: bool,
    env: dict[str, Any],
    compile_result: Any,
    selected_systems: list[str],
    selected_workloads: list[str],
    selected_experts: list[int],
    repeats: int,
    command_runs: list[dict[str, Any]],
    skipped_existing_runs: int,
    c_init_path: str | None = None,
    ablation_policy: str = "",
    async_planning_mode: str = "",
) -> None:
    rows = _load_real_rows(command_runs)
    manifest = {
        "schema_version": "tilepo_sweep_manifest_v1",
        "mode": mode,
        "c_mode": c_mode,
        "simulated": simulated,
        "blocked": False,
        "environment": env,
        "compiled_manifest": str(compile_result.manifest_path),
        "serving_shell": "KT/SGLang",
        "systems": ["A", "B", "C"],
        "selected_systems": selected_systems,
        "selected_workloads": selected_workloads,
        "selected_experts": selected_experts,
        "selected_repeats": repeats,
        "c_init_path": c_init_path,
        "ablation_policy": ablation_policy,
        "async_planning_mode": async_planning_mode,
        "expected_command_runs": len(command_runs),
        "expected_result_rows": len(command_runs),
        "command_runs": command_runs,
        "skipped_existing_runs": skipped_existing_runs,
        "runs": rows,
        "created_at_unix": time.time(),
        "checkpoint": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _mark_existing_success(run: dict[str, Any]) -> bool:
    path = Path(run["jsonl"])
    if not path.exists():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    if not rows:
        return False
    if not all(_row_is_real_success(row) for row in rows):
        return False
    marker_path = path.with_suffix(".tilepo_bootstrap.json")
    if marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text())
        except json.JSONDecodeError:
            marker = {}
        marker_run_id = str(marker.get("run_id", ""))
        if marker_run_id:
            run["run_id"] = marker_run_id
    run["skipped_existing"] = True
    return True


def _row_is_real_success(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == "success"
        and row.get("simulated") is False
        and row.get("evidence_level") == "real"
    )


def _load_hot_backend_probe(run: dict[str, Any], jsonl_path: Path) -> dict[str, Any]:
    if run.get("system") != "C":
        return {}
    marker_path = jsonl_path.with_suffix(".tilepo_bootstrap.json")
    if not marker_path.exists():
        return {}
    try:
        marker = json.loads(marker_path.read_text())
    except json.JSONDecodeError:
        return {"path": str(marker_path), "status": "unreadable"}
    expected_run_id = str(run.get("run_id", ""))
    marker_run_id = str(marker.get("run_id", ""))
    if expected_run_id and marker_run_id != expected_run_id:
        return {
            "path": str(marker_path),
            "status": "stale_run_id",
            "expected_run_id": expected_run_id,
            "marker_run_id": marker_run_id,
        }
    probe = marker.get("hot_backend_probe")
    if not isinstance(probe, dict):
        return {"path": str(marker_path), "status": "missing"}
    result = {"path": str(marker_path), **probe}
    serving_hook = marker.get("serving_hook")
    if isinstance(serving_hook, dict):
        result["serving_hook"] = serving_hook
    return result


def _format_number(value: float | int) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def _validate_c_mode(c_mode: str) -> None:
    if c_mode not in C_MODE_CHOICES:
        choices = ", ".join(C_MODE_CHOICES)
        raise ValueError(f"unsupported c_mode {c_mode!r}; expected one of: {choices}")


def _linux_available_gib() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return float("inf")
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / (1024 * 1024)
    return float("inf")


def _fixture_rows(
    workloads: list[str],
    experts: list[int],
    repeats: int,
    mode: str,
    *,
    ablation_policy: str = "",
    async_planning_mode: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in workloads:
        for expert in experts:
            for repeat in range(repeats):
                base_tok = 100.0 + expert * 0.6
                base_p95 = 100.0 - min(expert, 16) * 0.8
                placement_tok = base_tok * (1.03 if workload != "anchor_unique" else 0.98)
                placement_p95 = base_p95 * (0.96 if workload != "anchor_unique" else 1.05)
                c_tok = placement_tok
                c_p95 = placement_p95
                c_gpu = 7.0
                if workload == "long_output" and expert in {8, 10, 12, 16}:
                    c_tok *= 1.13
                    c_p95 *= 0.84
                    c_gpu = 5.2
                elif workload == "mixed" and expert == 4:
                    c_tok *= 1.18
                    c_p95 *= 0.84
                    c_gpu = 5.8
                else:
                    c_tok *= 0.97
                    c_p95 *= 1.04
                rows.extend(
                    [
                        _row(
                            "A",
                            workload,
                            expert,
                            repeat,
                            base_tok,
                            base_p95,
                            9.0,
                            mode,
                            ablation_policy,
                            async_planning_mode,
                        ),
                        _row(
                            "B",
                            workload,
                            expert,
                            repeat,
                            placement_tok,
                            placement_p95,
                            8.0,
                            mode,
                            ablation_policy,
                            async_planning_mode,
                        ),
                        _row(
                            "C",
                            workload,
                            expert,
                            repeat,
                            c_tok,
                            c_p95,
                            c_gpu,
                            mode,
                            ablation_policy,
                            async_planning_mode,
                        ),
                    ]
                )
    return rows


def _row(
    system: str,
    workload: str,
    experts: int,
    repeat: int,
    tok: float,
    p95: float,
    gpu: float,
    mode: str,
    ablation_policy: str = "",
    async_planning_mode: str = "",
) -> dict[str, Any]:
    return {
        "system": system,
        "workload": workload,
        "experts_per_layer": experts,
        "repeat": repeat,
        "tok_per_sec": tok + repeat * 0.1,
        "p50_ms": p95 * 0.5,
        "p95_ms": p95,
        "p99_ms": p95 * 1.3,
        "gpu_peak_gib": gpu,
        "cpu_ram_peak_gib": 24.0,
        "server_ready_s": 5.0,
        "runtime_overhead_us": 25.0 if system == "C" else 0.0,
        "dtype_counts": {"bf16": 1} if system != "C" else {"mxfp4": 1, "fp8": 1, "bf16": 1},
        "fallback_count": 0 if system != "C" else (0 if mode == "serve" else 1),
        "backend_launch_counts": {"cuda": 1} if system == "C" else {},
        "evidence_level": "simulated",
        "simulated": True,
        "raw_path": f"raw/{system}-{workload}-{experts}-{repeat}.jsonl",
        "command": f"run_tilepo_sweep --mode {mode} --workload {workload} --experts {experts}",
        "ablation_policy": ablation_policy,
        "async_planning_mode": async_planning_mode,
        "tilepo_policy": ablation_policy,
        "tilepo_async_planning": async_planning_mode,
    }
