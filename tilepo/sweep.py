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
    max_host_commit_percent: float = 100.0,
    max_vmmem_gib: float = 0.0,
    min_linux_available_gib: float = 8.0,
    skip_existing_success: bool = False,
    c_mode: str = "hook",
    ablation_policy: str = "",
    async_planning_mode: str = "",
    force_prompt: str = "",
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
        force_prompt=force_prompt,
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
            try:
                subprocess.run(
                    run["command"],
                    cwd=Path(__file__).resolve().parents[1],
                    env=os.environ.copy(),
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                blocker = _command_failure_blocker(run, exc)
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
                    blocked=True,
                    blockers=[blocker],
                    failed_command_run=_failed_command_run(run, exc),
                )
                return {"manifest_path": str(manifest_path), "blocked": True, "blockers": [blocker]}
            row_failure = _raw_row_failure_blocker(run)
            if row_failure:
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
                    blocked=True,
                    blockers=[row_failure],
                    failed_command_run=_failed_row_command_run(run),
                )
                return {"manifest_path": str(manifest_path), "blocked": True, "blockers": [row_failure]}
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
        "actual_result_rows": len(rows),
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
    preserve_kt_optimizations: bool = False,
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
    if not preserve_kt_optimizations:
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
    max_host_commit_percent: float = 100.0,
    max_vmmem_gib: float = 0.0,
    min_linux_available_gib: float = 8.0,
    c_mode: str = "hook",
    ablation_policy: str = "",
    async_planning_mode: str = "",
    force_prompt: str = "",
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
        preserve_kt_optimizations=False,
    )
    run_id = f"{run_name}-{uuid.uuid4().hex}"
    extra_env = []
    if system == "C" and c_mode == "hook":
        marker = out_dir / "raw" / f"{run_name}.tilepo_bootstrap.json"
        pythonpath = os.pathsep.join([str(repo_root), str(bench_tool.resolve().parents[1])])
        is_v0_2_policy = ablation_policy == "tilepo_atg_tc_baa"
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
            f"{tilepo_env.TILEPO_HOOK_VERIFY_LIMIT}=1",
            "--extra-env",
            f"{tilepo_env.TILEPO_HOOK_FLUSH_INTERVAL}=4096",
            "--extra-env",
            f"PYTHONPATH={pythonpath}",
        ]
        if is_v0_2_policy:
            extra_env.extend(
                [
                    "--extra-env",
                    f"{tilepo_env.TILEPO_REQUIRE_NATIVE_BACKEND}=1",
                    "--extra-env",
                    f"{tilepo_env.TILEPO_HOOK_BACKEND_PROBE_LIMIT}=1",
                ]
            )
    prompt_args = ["--prompts-file", str(prompts_file)]
    if force_prompt:
        prompt_args = []
        for _ in range(max(request_count + warmup_request_count, 0)):
            prompt_args.extend(["--prompt", force_prompt])

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
        *prompt_args,
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
        "force_prompt": force_prompt,
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
    force_prompt: str,
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
                            force_prompt=force_prompt,
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
                _merge_hot_backend_probe(row, hot_probe)
                _attach_v2_execution_evidence(row)
                rows.append(row)
    return rows


def _attach_v2_execution_evidence(row: dict[str, Any]) -> None:
    if row.get("tilepo_policy") != "tilepo_atg_tc_baa":
        return
    row.setdefault("backend_owner", "kt_sglang")
    row.setdefault("kt_executor_preserved", _bool_value(row.get("serving_hook_returned_original")))
    row.setdefault("tilepo_plan_applied_in_serving_path", _tilepo_plan_applied_in_serving_path(row))
    row.setdefault("tc_coalescing_active", _tc_coalescing_active(row))
    row.setdefault("baa_double_buffered", _bool_value(row.get("baa_double_buffered")))
    if row.get("unexpected_plain_kt_bypass_events") is None:
        fallback_keys = ("fallback_count", "serving_hook_backend_fallback_count", "kt_fallback_count", "baa_fallback_count")
        if any(key in row for key in fallback_keys):
            row["unexpected_plain_kt_bypass_events"] = sum(int(row.get(key, 0) or 0) for key in fallback_keys)
    if "execution_dispatch_units" not in row and "serving_hook_backend_execution_dispatch_units" in row:
        row["execution_dispatch_units"] = row["serving_hook_backend_execution_dispatch_units"]
    if "coalesced_group_count" not in row and "serving_hook_backend_coalesced_group_count" in row:
        row["coalesced_group_count"] = row["serving_hook_backend_coalesced_group_count"]
    if "baa_critical_path_us" not in row and "serving_hook_backend_baa_critical_path_us" in row:
        row["baa_critical_path_us"] = row["serving_hook_backend_baa_critical_path_us"]
    if "baa_metrics_measured" not in row and "serving_hook_backend_baa_metrics_measured" in row:
        row["baa_metrics_measured"] = row["serving_hook_backend_baa_metrics_measured"]
    if "cuda_descriptor_traversal_us" not in row and "serving_hook_backend_cuda_descriptor_traversal_us" in row:
        row["cuda_descriptor_traversal_us"] = row["serving_hook_backend_cuda_descriptor_traversal_us"]
    if "cuda_descriptor_metrics_measured" not in row and "serving_hook_backend_cuda_descriptor_metrics_measured" in row:
        row["cuda_descriptor_metrics_measured"] = row["serving_hook_backend_cuda_descriptor_metrics_measured"]
    if "tc_native_consumed" not in row and "serving_hook_backend_tc_native_consumed" in row:
        row["tc_native_consumed"] = row["serving_hook_backend_tc_native_consumed"]
    if "tc_native_consumed_coalesced_groups" not in row and "serving_hook_backend_tc_native_consumed_coalesced_groups" in row:
        row["tc_native_consumed_coalesced_groups"] = row["serving_hook_backend_tc_native_consumed_coalesced_groups"]
    for key in (
        "tc_native_consumed_group_count",
        "tc_native_descriptor_count",
        "tc_native_consumed_tile_count",
        "tc_native_consumed_bytes",
        "tc_native_launch_count",
        "tc_adapter_group_count",
        "tc_adapter_descriptor_count",
        "tc_adapter_tile_count",
        "tc_adapter_dispatch_units",
    ):
        hook_key = f"serving_hook_backend_{key}"
        if key not in row and hook_key in row:
            row[key] = row[hook_key]
    for key in (
        "tc_native_entrypoint",
        "tc_native_descriptor_layout",
        "tc_native_consumption_source",
        "tc_native_launch_path",
        "tc_adapter_source",
        "tc_adapter_target",
        "tc_adapter_mode",
        "tc_adapter_fallback_reason",
    ):
        hook_key = f"serving_hook_backend_{key}"
        if key not in row and hook_key in row:
            row[key] = row[hook_key]
    if "tc_adapter_consumed" not in row and "serving_hook_backend_tc_adapter_consumed" in row:
        row["tc_adapter_consumed"] = row["serving_hook_backend_tc_adapter_consumed"]
    for key, hook_key in (
        ("tc_adapter_group_count", "serving_hook_backend_tc_adapter_group_count"),
        ("tc_adapter_descriptor_count", "serving_hook_backend_tc_adapter_descriptor_count"),
        ("tc_adapter_tile_count", "serving_hook_backend_tc_adapter_tile_count"),
        ("tc_adapter_dispatch_units", "serving_hook_backend_tc_adapter_dispatch_units"),
        ("tc_adapter_source", "serving_hook_backend_tc_adapter_source"),
        ("tc_adapter_target", "serving_hook_backend_tc_adapter_target"),
        ("tc_adapter_mode", "serving_hook_backend_tc_adapter_mode"),
        ("tc_adapter_fallback_reason", "serving_hook_backend_tc_adapter_fallback_reason"),
    ):
        if key not in row and hook_key in row:
            row[key] = row[hook_key]
    if _bool_value(row.get("tc_native_consumed")):
        row["runtime_metrics_source"] = row.get("runtime_metrics_source") or "kt_preserving_native_tc_kernel"
    row.setdefault("runtime_metrics_source", "kt_preserving_hook")
    hook_counts = row.get("serving_hook_backend_launch_counts")
    if isinstance(hook_counts, dict) and "cuda_launch_count" not in row:
        row["cuda_launch_count"] = int(hook_counts.get("cuda", 0) or 0)
    if "cuda_launch_count" not in row and "backend_launch_counts" in row:
        counts = row.get("backend_launch_counts")
        if isinstance(counts, dict):
            row["cuda_launch_count"] = int(counts.get("cuda", 0) or 0)
    row["tilepo_backend"] = _row_tilepo_backend(row)
    blockers = _v0_2_runtime_blockers(row)
    row["v0_2_runtime_status"] = "ready" if not blockers else "blocked"
    row["v0_2_runtime_blockers"] = blockers


_HOT_PROBE_ROW_KEYS = (
    "kt_executor_preserved",
    "tilepo_plan_applied_in_serving_path",
    "tc_coalescing_active",
    "plan_lookup_us",
    "plan_lookup_total_us",
    "gate_us",
    "backend_launch_us",
    "h2d_bytes",
    "cache_hits",
    "cache_misses",
    "tile_count",
    "coalesced_group_count",
    "execution_dispatch_units",
    "baa_double_buffered",
    "baa_critical_path_us",
    "baa_active_map_id",
    "baa_standby_ready",
    "baa_metrics_measured",
    "cuda_launch_count",
    "native_cuda_available",
    "native_cuda_launch_count",
    "cuda_python_shim_launch_count",
    "cuda_descriptor_traversal_us",
    "cuda_descriptor_metrics_measured",
    "tc_native_consumed",
    "tc_native_consumed_coalesced_groups",
    "tc_native_consumed_group_count",
    "tc_native_descriptor_count",
    "tc_native_consumed_tile_count",
    "tc_native_consumed_bytes",
    "tc_native_entrypoint",
    "tc_native_descriptor_layout",
    "tc_native_consumption_source",
    "tc_native_launch_path",
    "tc_native_launch_count",
    "runtime_metrics_source",
    "async_plan_cache_hits",
    "async_plan_cache_misses",
)

_SERVING_HOOK_ROW_KEYS = (
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
    "serving_hook_backend_launch_source",
    "serving_hook_backend_native_cuda_available",
    "serving_hook_backend_native_cuda_launch_count",
    "serving_hook_backend_cuda_python_shim_launch_count",
    "serving_hook_backend_fallback_count",
    "serving_hook_backend_dtype_counts",
    "serving_hook_backend_h2d_bytes",
    "serving_hook_backend_runtime_us",
    "serving_hook_backend_result",
    "serving_hook_backend_hot_tile",
    "serving_hook_backend_coalesced_group_count",
    "serving_hook_backend_execution_dispatch_units",
    "serving_hook_backend_baa_double_buffered",
    "serving_hook_backend_baa_critical_path_us",
    "serving_hook_backend_baa_metrics_measured",
    "serving_hook_backend_cuda_descriptor_traversal_us",
    "serving_hook_backend_cuda_descriptor_metrics_measured",
    "serving_hook_backend_tc_native_consumed",
    "serving_hook_backend_tc_native_consumed_coalesced_groups",
    "serving_hook_backend_tc_native_consumed_group_count",
    "serving_hook_backend_tc_native_descriptor_count",
    "serving_hook_backend_tc_native_consumed_tile_count",
    "serving_hook_backend_tc_native_consumed_bytes",
    "serving_hook_backend_tc_native_entrypoint",
    "serving_hook_backend_tc_native_descriptor_layout",
    "serving_hook_backend_tc_native_consumption_source",
    "serving_hook_backend_tc_native_launch_path",
    "serving_hook_backend_tc_native_launch_count",
    "serving_hook_backend_tc_adapter_consumed",
    "serving_hook_backend_tc_adapter_source",
    "serving_hook_backend_tc_adapter_group_count",
    "serving_hook_backend_tc_adapter_descriptor_count",
    "serving_hook_backend_tc_adapter_tile_count",
    "serving_hook_backend_tc_adapter_dispatch_units",
    "serving_hook_backend_tc_adapter_target",
    "serving_hook_backend_tc_adapter_mode",
    "serving_hook_backend_tc_adapter_fallback_reason",
    "serving_hook_backend_launch_failure",
    "serving_hook_mode",
    "serving_hook_replacement_real",
    "serving_hook_verify_count",
    "serving_hook_verify_pass_count",
    "serving_hook_verify_fail_count",
    "serving_hook_verify_max_abs_error",
    "serving_hook_verify_shape_match",
    "serving_hook_verify_dtype_match",
    "serving_hook_verify_device_match",
    "serving_hook_verify_source",
    "serving_hook_candidate_available",
)


def _merge_hot_backend_probe(row: dict[str, Any], hot_probe: dict[str, Any]) -> None:
    if not hot_probe:
        row.setdefault("runtime_overhead_us", 0.0)
        row.setdefault("dtype_counts", {"bf16": 1})
        row.setdefault("fallback_count", 0)
        row.setdefault("backend_launch_counts", {})
        row.setdefault("tilemem_backend_launch_count", 0)
        return
    row["hot_backend_probe_path"] = hot_probe.get("path", row.get("hot_backend_probe_path", ""))
    row["hot_backend_probe_status"] = hot_probe.get("status", "unknown")
    if "failure_reason" in hot_probe:
        row["hot_backend_probe_failure_reason"] = hot_probe["failure_reason"]
    row["runtime_overhead_us"] = row.get("runtime_overhead_us", hot_probe.get("runtime_overhead_us", 0.0))
    for key in _HOT_PROBE_ROW_KEYS:
        if key not in row and key in hot_probe:
            row[key] = hot_probe[key]
    row["dtype_counts"] = row.get("dtype_counts", hot_probe.get("dtype_counts", {"bf16": 1}))
    row["fallback_count"] = row.get("fallback_count", hot_probe.get("fallback_count", 0))
    row["backend_launch_counts"] = row.get("backend_launch_counts", hot_probe.get("backend_launch_counts", {}))
    row["tilemem_backend_launch_count"] = row.get(
        "tilemem_backend_launch_count",
        hot_probe.get("tilemem_backend_launch_count", 0),
    )
    if "hot_backend_native" not in row and "hot_backend_native" in hot_probe:
        row["hot_backend_native"] = bool(hot_probe["hot_backend_native"])
    serving_hook = hot_probe.get("serving_hook", {})
    if isinstance(serving_hook, dict):
        for key in _SERVING_HOOK_ROW_KEYS:
            if key in serving_hook and key not in row:
                row[key] = serving_hook[key]


def _tilepo_plan_applied_in_serving_path(row: dict[str, Any]) -> bool:
    return (
        _bool_value(row.get("serving_hook_active"))
        and int(row.get("serving_hook_invocations", 0) or 0) > 0
        and (
            row.get("tilepo_plan") is not None
            or int(row.get("tile_count", 0) or 0) > 0
            or int(row.get("execution_dispatch_units", 0) or 0) > 0
        )
    )


def _tc_coalescing_active(row: dict[str, Any]) -> bool:
    if _bool_value(row.get("tc_coalescing_active")):
        return True
    if int(row.get("coalesced_group_count", 0) or 0) > 0:
        return True
    dispatch_units = int(row.get("execution_dispatch_units", 0) or 0)
    tile_count = int(row.get("tile_count", 0) or 0)
    return tile_count > 0 and dispatch_units > 0 and dispatch_units < tile_count


def _v0_2_runtime_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not _bool_value(row.get("serving_hook_active")):
        blockers.append("serving hook is not active")
    if int(row.get("serving_hook_invocations", 0) or 0) <= 0:
        blockers.append("serving hook was not invoked")
    if not _bool_value(row.get("kt_executor_preserved")):
        blockers.append("KT/SGLang serving shell preservation evidence is missing")
    if _bool_value(row.get("serving_hook_returned_original", True)):
        blockers.append("V0.2 native TC did not replace the measured serving path")
    if int(row.get("serving_hook_replaced_count", 0) or 0) <= 0:
        blockers.append("V0.2 native TC replacement count is zero")
    if not _bool_value(row.get("tilepo_plan_applied_in_serving_path")):
        blockers.append("TilePO plan was not applied in serving path")
    if not _bool_value(row.get("tc_coalescing_active")):
        blockers.append("TC coalescing evidence is missing")
    if not _bool_value(row.get("tc_native_consumed")):
        blockers.append("native TC consumption evidence is missing")
    if not _bool_value(row.get("tc_native_consumed_coalesced_groups")):
        blockers.append("native TC coalesced group consumption evidence is missing")
    if int(row.get("tc_native_descriptor_count", 0) or 0) <= 0:
        blockers.append("native TC descriptor count is missing")
    elif int(row.get("tc_native_descriptor_count", 0) or 0) != 8:
        blockers.append("native TC descriptor count is not 8")
    if str(row.get("tc_native_entrypoint", "")) != "tilepo_cuda_dispatch_coalesced_gemm":
        blockers.append("native TC entrypoint is not tilepo_cuda_dispatch_coalesced_gemm")
    if str(row.get("tc_native_descriptor_layout", "")) != "tilepo_cuda_coalesced_group_desc_v1":
        blockers.append("native TC descriptor layout is not tilepo_cuda_coalesced_group_desc_v1")
    if int(row.get("tc_native_consumed_group_count", 0) or 0) <= 0:
        blockers.append("native TC consumed group count is missing")
    if str(row.get("tc_native_consumption_source", "")) not in {
        "kt_grouped_moe_cuda_adapter",
        "kt_serving_cuda_kernel",
        "kt_launch_adapter_tc",
    }:
        blockers.append("native TC consumption source is not KT/CUDA")
    if not _bool_value(row.get("baa_double_buffered")):
        blockers.append("BAA double-buffer evidence is missing")
    if int(row.get("serving_hook_verify_fail_count", 0) or 0) != 0:
        blockers.append("KT-preserving hook verification failed")
    if int(row.get("unexpected_plain_kt_bypass_events", row.get("fallback_count", 0)) or 0) != 0:
        blockers.append("unexpected plain KT bypass events are nonzero")
    if not _bool_value(row.get("baa_metrics_measured")):
        blockers.append("BAA metrics were not measured")
    if not _bool_value(row.get("cuda_descriptor_metrics_measured")):
        blockers.append("CUDA descriptor metrics were not measured")
    return blockers


def _serving_replacement_is_real(row: dict[str, Any]) -> bool:
    return not _v0_2_runtime_blockers(row)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _row_v0_2_evidence_blocker(run: dict[str, Any], row: dict[str, Any]) -> str:
    if run.get("tilepo_policy") != "tilepo_atg_tc_baa":
        return ""
    blockers = _v0_2_runtime_blockers(row)
    if not blockers:
        return ""
    return (
        "V0.2 row is not real native TC ATG+TC+BAA evidence: "
        + "; ".join(blockers)
        + f"; log={run.get('log')} jsonl={run.get('jsonl')}"
    )


def _row_tilepo_backend(row: dict[str, Any]) -> str:
    if row.get("tilepo_policy") == "tilepo_atg_tc_baa" and not _serving_replacement_is_real(row):
        return ""
    if row.get("tilepo_policy") == "tilepo_atg_tc_baa":
        if _bool_value(row.get("tc_native_consumed")):
            return "kt_preserving_native_tc_cuda"
        if _bool_value(row.get("tc_adapter_consumed")):
            return "kt_preserving_launch_adapter_tc_cuda"
        counts = row.get("backend_launch_counts")
        hook_counts = row.get("serving_hook_backend_launch_counts")
        if isinstance(hook_counts, dict) and int(hook_counts.get("cuda", 0) or 0) > 0:
            return "kt_preserving_cuda_augmentation"
        if isinstance(counts, dict) and int(counts.get("cuda", 0) or 0) > 0:
            return "kt_preserving_cuda_augmentation"
        if int(row.get("cuda_launch_count", 0) or 0) > 0:
            return "kt_preserving_cuda_augmentation"
    counts = row.get("backend_launch_counts")
    hook_counts = row.get("serving_hook_backend_launch_counts")
    if isinstance(counts, dict) and int(counts.get("cuda", 0) or 0) > 0:
        return "cuda"
    if isinstance(hook_counts, dict) and int(hook_counts.get("cuda", 0) or 0) > 0:
        return "cuda"
    return ""


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
    blocked: bool = False,
    blockers: list[str] | None = None,
    failed_command_run: dict[str, Any] | None = None,
) -> None:
    rows = _load_real_rows(command_runs)
    manifest = {
        "schema_version": "tilepo_sweep_manifest_v1",
        "mode": mode,
        "c_mode": c_mode,
        "simulated": simulated,
        "blocked": blocked,
        "blockers": blockers or [],
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
        "actual_result_rows": len(rows),
        "created_at_unix": time.time(),
        "checkpoint": True,
    }
    if failed_command_run is not None:
        manifest["failed_command_run"] = failed_command_run
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _command_failure_blocker(run: dict[str, Any], exc: subprocess.CalledProcessError) -> str:
    return (
        f"real benchmark command failed with returncode {exc.returncode}: "
        f"{run.get('system_name', run.get('system', 'unknown'))} "
        f"experts={run.get('experts_per_layer')} workload={run.get('workload')} "
        f"repeat={run.get('repeat')} log={run.get('log')} jsonl={run.get('jsonl')}"
    )


def _failed_command_run(run: dict[str, Any], exc: subprocess.CalledProcessError) -> dict[str, Any]:
    return {
        "returncode": exc.returncode,
        "system": run.get("system"),
        "system_name": run.get("system_name"),
        "workload": run.get("workload"),
        "experts_per_layer": run.get("experts_per_layer"),
        "repeat": run.get("repeat"),
        "jsonl": run.get("jsonl"),
        "log": run.get("log"),
        "plugin": run.get("plugin"),
        "command": run.get("command"),
        "server_command": run.get("server_command"),
        "tilepo_policy": run.get("tilepo_policy"),
        "tilepo_async_planning": run.get("tilepo_async_planning"),
    }


def _raw_row_failure_blocker(run: dict[str, Any]) -> str:
    path = Path(run["jsonl"])
    if not path.exists():
        return (
            "real benchmark command returned success but did not write jsonl: "
            f"{run.get('system_name', run.get('system', 'unknown'))} "
            f"experts={run.get('experts_per_layer')} workload={run.get('workload')} "
            f"repeat={run.get('repeat')} log={run.get('log')} jsonl={run.get('jsonl')}"
        )
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return (
            "real benchmark command returned success but wrote unreadable jsonl: "
            f"{exc}; log={run.get('log')} jsonl={run.get('jsonl')}"
        )
    if not rows:
        return (
            "real benchmark command returned success but wrote empty jsonl: "
            f"log={run.get('log')} jsonl={run.get('jsonl')}"
        )
    for index, row in enumerate(rows):
        _attach_run_identity(row, run)
        if not _row_is_real_success(row):
            status = row.get("status")
            reason = row.get("failure_reason", "")
            return (
                "real benchmark command returned success but raw row is not successful real evidence: "
                f"row={index} status={status!r} simulated={row.get('simulated')!r} "
                f"evidence_level={row.get('evidence_level')!r} reason={reason!r} "
                f"log={run.get('log')} jsonl={run.get('jsonl')}"
            )
        hot_probe = _load_hot_backend_probe(run, path)
        if hot_probe:
            if "hot_backend_native" in hot_probe and "hot_backend_native" not in row:
                row["hot_backend_native"] = bool(hot_probe["hot_backend_native"])
            serving_hook = hot_probe.get("serving_hook", {})
            if isinstance(serving_hook, dict):
                for key, value in serving_hook.items():
                    row.setdefault(key, value)
        _attach_v2_execution_evidence(row)
        v0_2_blocker = _row_v0_2_evidence_blocker(run, row)
        if v0_2_blocker:
            return v0_2_blocker
    return ""


def _failed_row_command_run(run: dict[str, Any]) -> dict[str, Any]:
    record = {
        "returncode": 0,
        "system": run.get("system"),
        "system_name": run.get("system_name"),
        "workload": run.get("workload"),
        "experts_per_layer": run.get("experts_per_layer"),
        "repeat": run.get("repeat"),
        "jsonl": run.get("jsonl"),
        "log": run.get("log"),
        "plugin": run.get("plugin"),
        "command": run.get("command"),
        "server_command": run.get("server_command"),
        "tilepo_policy": run.get("tilepo_policy"),
        "tilepo_async_planning": run.get("tilepo_async_planning"),
    }
    path = Path(run["jsonl"])
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        rows = []
    if rows:
        row = rows[0]
        record["row_status"] = row.get("status")
        record["row_failure_reason"] = row.get("failure_reason", "")
        record["row_evidence_level"] = row.get("evidence_level")
        record["row_simulated"] = row.get("simulated")
    return record


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
    if run.get("tilepo_policy") == "tilepo_atg_tc_baa":
        for row in rows:
            _attach_run_identity(row, run)
            hot_probe = _load_hot_backend_probe(run, path, allow_stale_run_id=True)
            _merge_hot_backend_probe(row, hot_probe)
            _attach_v2_execution_evidence(row)
            if _row_v0_2_evidence_blocker(run, row):
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


def _attach_run_identity(row: dict[str, Any], run: dict[str, Any]) -> None:
    row.setdefault("experts_per_layer", run.get("experts_per_layer"))
    row.setdefault("repeat", run.get("repeat"))
    row.setdefault("workload", run.get("workload"))
    row.setdefault("ablation_policy", run.get("ablation_policy", ""))
    row.setdefault("async_planning_mode", run.get("async_planning_mode", ""))
    row.setdefault("tilepo_policy", run.get("tilepo_policy", run.get("ablation_policy", "")))
    row.setdefault(
        "tilepo_async_planning",
        run.get("tilepo_async_planning", run.get("async_planning_mode", "")),
    )


def _row_is_real_success(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == "success"
        and row.get("simulated") is False
        and row.get("evidence_level") == "real"
    )


def _load_hot_backend_probe(
    run: dict[str, Any],
    jsonl_path: Path,
    *,
    allow_stale_run_id: bool = False,
) -> dict[str, Any]:
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
    if expected_run_id and marker_run_id != expected_run_id and not allow_stale_run_id:
        return {
            "path": str(marker_path),
            "status": "stale_run_id",
            "expected_run_id": expected_run_id,
            "marker_run_id": marker_run_id,
        }
    if marker_run_id and marker_run_id != expected_run_id and allow_stale_run_id:
        run["run_id"] = marker_run_id
    probe = marker.get("hot_backend_probe")
    if not isinstance(probe, dict):
        result = {"path": str(marker_path), "status": "missing"}
    else:
        result = {"path": str(marker_path), **probe}
    serving_hook = marker.get("serving_hook")
    if isinstance(serving_hook, dict):
        result["serving_hook"] = serving_hook
    for key in (
        "kt_executor_preserved",
        "tilepo_plan_applied_in_serving_path",
        "tc_coalescing_active",
        "baa_double_buffered",
        "baa_critical_path_us",
        "baa_metrics_measured",
        "cuda_descriptor_metrics_measured",
        "cuda_descriptor_traversal_us",
        "cuda_launch_count",
        "runtime_metrics_source",
    ):
        if key in marker and key not in result:
            result[key] = marker[key]
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
