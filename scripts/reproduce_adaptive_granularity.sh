#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE=0
RESUME=0
BASE_PORT=35100
MODEL_DIR="/mnt/d/tilemem_runtime/models/OLMoE-1B-7B-0924-Instruct"
INIT_PATH="/mnt/d/tilemem_runtime/results/kt_tilemem_hotset_20260523/tilemem_hotset_counts.pt"
KT_ENV="tilemem-tilepo-ktransformers"
BENCH_TOOL=""
MIN_LINUX_AVAILABLE_GIB=8
STRICT_V0_2_WIN=0
STRICT_NATIVE_TC=0
OFFLINE_ACCEPTANCE=0
V0_2_ONLY=0
FOCUSED_WORKLOAD=""
FOCUSED_EXPERTS=""
OUTPUT_TOKENS=4
REPEATS=1
SNAPSHOT_LABEL=""
FORCE_PROMPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      EXECUTE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --init-expert-location)
      INIT_PATH="$2"
      shift 2
      ;;
    --kt-env)
      KT_ENV="$2"
      shift 2
      ;;
    --bench-tool)
      BENCH_TOOL="$2"
      shift 2
      ;;
    --min-linux-available-gib)
      MIN_LINUX_AVAILABLE_GIB="$2"
      shift 2
      ;;
    --strict-v0-2-win|--strict-atg-win)
      STRICT_V0_2_WIN=1
      shift
      ;;
    --strict-native-tc)
      STRICT_V0_2_WIN=1
      STRICT_NATIVE_TC=1
      shift
      ;;
    --offline-acceptance)
      OFFLINE_ACCEPTANCE=1
      shift
      ;;
    --v0-2-only)
      V0_2_ONLY=1
      shift
      ;;
    --workload)
      FOCUSED_WORKLOAD="$2"
      shift 2
      ;;
    --experts)
      FOCUSED_EXPERTS="$2"
      shift 2
      ;;
    --output-tokens)
      OUTPUT_TOKENS="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --force-prompt)
      FORCE_PROMPT="$2"
      shift 2
      ;;
    --snapshot-label)
      SNAPSHOT_LABEL="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export TILEMEM_ADAPTIVE_EXECUTE="$EXECUTE"
export TILEMEM_ADAPTIVE_RESUME="$RESUME"
export TILEMEM_ADAPTIVE_BASE_PORT="$BASE_PORT"
export TILEMEM_ADAPTIVE_MODEL_DIR="$MODEL_DIR"
export TILEMEM_ADAPTIVE_INIT_PATH="$INIT_PATH"
export TILEMEM_ADAPTIVE_KT_ENV="$KT_ENV"
export TILEMEM_ADAPTIVE_BENCH_TOOL="$BENCH_TOOL"
export TILEMEM_ADAPTIVE_MIN_LINUX_AVAILABLE_GIB="$MIN_LINUX_AVAILABLE_GIB"
export TILEMEM_ADAPTIVE_STRICT_V0_2_WIN="$STRICT_V0_2_WIN"
export TILEPO_STRICT_NATIVE_TC="$STRICT_NATIVE_TC"
export TILEPO_OFFLINE_ACCEPTANCE="$OFFLINE_ACCEPTANCE"
export TILEMEM_ADAPTIVE_V0_2_ONLY="$V0_2_ONLY"
export TILEMEM_ADAPTIVE_WORKLOAD="$FOCUSED_WORKLOAD"
export TILEMEM_ADAPTIVE_EXPERTS="$FOCUSED_EXPERTS"
export TILEMEM_ADAPTIVE_OUTPUT_TOKENS="$OUTPUT_TOKENS"
export TILEMEM_ADAPTIVE_REPEATS="$REPEATS"
export TILEMEM_ADAPTIVE_SNAPSHOT_LABEL="$SNAPSHOT_LABEL"
export TILEMEM_ADAPTIVE_FORCE_PROMPT="$FORCE_PROMPT"

if [[ "$OFFLINE_ACCEPTANCE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TILEPO_DISABLE_NETWORK=1
fi

python3 - <<'PY'
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from tilepo.ablation import write_merged_manifest, write_tilepo_plan
from tilepo.dsl import DSLBlock, parse_tmem
from tilepo.sweep import run_sweep


ROOT = Path.cwd()
OUT_DIR = ROOT / "evidence" / "adaptive_granularity"
PLANS_DIR = OUT_DIR / "plans"
RUNS_DIR = OUT_DIR / "runs"
SNAPSHOTS_DIR = OUT_DIR / "snapshots"
MERGED = OUT_DIR / "tilepo_adaptive_granularity_manifest.json"
BASE_PLAN = ROOT / "configs" / "tilepo_olmoe_bf16_only.tmem"
DEFAULT_WORKLOADS = ["mixed", "long_context"]
DEFAULT_EXPERTS = [6, 8, 10]
TILEPO_SELF_ABLATION_POLICIES = ["tilepo_coarse", "tilepo_fine", "tilepo_hybrid", "tilepo_atg_tc_baa"]


def main() -> int:
    execute = os.environ["TILEMEM_ADAPTIVE_EXECUTE"] == "1"
    resume = os.environ.get("TILEMEM_ADAPTIVE_RESUME") == "1"
    base_port = int(os.environ["TILEMEM_ADAPTIVE_BASE_PORT"])
    model_dir = os.environ["TILEMEM_ADAPTIVE_MODEL_DIR"]
    init_path = os.environ["TILEMEM_ADAPTIVE_INIT_PATH"]
    kt_env = os.environ["TILEMEM_ADAPTIVE_KT_ENV"]
    bench_tool_text = os.environ["TILEMEM_ADAPTIVE_BENCH_TOOL"]
    bench_tool = Path(bench_tool_text) if bench_tool_text else None
    min_linux_available_gib = float(os.environ["TILEMEM_ADAPTIVE_MIN_LINUX_AVAILABLE_GIB"])
    strict_v0_2_win = os.environ.get("TILEMEM_ADAPTIVE_STRICT_V0_2_WIN") == "1"
    strict_native_tc = os.environ.get("TILEPO_STRICT_NATIVE_TC") == "1"
    offline_acceptance = os.environ.get("TILEPO_OFFLINE_ACCEPTANCE") == "1"
    v0_2_only = os.environ.get("TILEMEM_ADAPTIVE_V0_2_ONLY") == "1"
    focused_workload = os.environ.get("TILEMEM_ADAPTIVE_WORKLOAD", "").strip()
    focused_experts = os.environ.get("TILEMEM_ADAPTIVE_EXPERTS", "").strip()
    output_tokens = int(os.environ.get("TILEMEM_ADAPTIVE_OUTPUT_TOKENS", "4"))
    repeats = int(os.environ.get("TILEMEM_ADAPTIVE_REPEATS", "1"))
    force_prompt = os.environ.get("TILEMEM_ADAPTIVE_FORCE_PROMPT", "")
    request_count = 5
    warmup_request_count = 5 if force_prompt else 1
    workloads = [focused_workload] if focused_workload else list(DEFAULT_WORKLOADS)
    experts_list = [int(focused_experts)] if focused_experts else list(DEFAULT_EXPERTS)
    tilepo_policies = ["tilepo_atg_tc_baa"] if v0_2_only else TILEPO_SELF_ABLATION_POLICIES
    expected_rows = len(workloads) * len(experts_list) * len(tilepo_policies) * repeats
    snapshot_path = _snapshot_existing_evidence(os.environ.get("TILEMEM_ADAPTIVE_SNAPSHOT_LABEL", ""))
    offline_blockers = _offline_preflight(ROOT, model_dir, init_path, bench_tool)
    if offline_acceptance:
        preflight_path = OUT_DIR / "tilepo_v0_2_offline_preflight.json"
        payload = {
            "schema_version": "tilemem_atb_v0_2_offline_preflight_v1",
            "offline_acceptance": True,
            "ready": not offline_blockers,
            "blockers": offline_blockers,
            "environment": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
                "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", ""),
                "TILEPO_DISABLE_NETWORK": os.environ.get("TILEPO_DISABLE_NETWORK", ""),
            },
            "model_dir": model_dir,
            "init_path": init_path,
            "bench_tool": str(bench_tool or (ROOT / "tools" / "openai_varprompt_bench")),
        }
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        if not preflight_path.exists():
            preflight_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if offline_blockers:
            raise SystemExit("offline acceptance preflight failed: " + "; ".join(offline_blockers))

    shutil.rmtree(PLANS_DIR, ignore_errors=True)
    if not resume:
        shutil.rmtree(RUNS_DIR, ignore_errors=True)
    for stale in (
        MERGED,
        OUT_DIR / "tilepo_adaptive_granularity_summary.json",
        OUT_DIR / "tilepo_adaptive_granularity_report.md",
    ):
        stale.unlink(missing_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    plan_paths = {}
    for workload in workloads:
        for expert in experts_list:
            for policy in tilepo_policies:
                workload_suffix = f"_{workload}" if v0_2_only or focused_workload else ""
                plan_path = PLANS_DIR / f"{policy}_experts{expert}{workload_suffix}_throughput_async_on.tmem"
                write_tilepo_plan(
                    BASE_PLAN,
                    plan_path,
                    expert_budget=expert,
                    policy=policy,
                    async_planning=True,
                    adaptive_mode="throughput",
                    workload_profile=workload if policy == "tilepo_atg_tc_baa" else "generic",
                )
                plan_paths[(policy, expert, workload)] = plan_path

    manifest_paths = []
    blockers = []
    runs = []
    for expert in experts_list:
        for policy in tilepo_policies:
            for workload in workloads:
                suffix = f"_{workload}" if v0_2_only or focused_workload else ""
                runs.append(
                    (
                        policy,
                        expert,
                        workload,
                        plan_paths[(policy, expert, workload)],
                        ["C"],
                        "on",
                        RUNS_DIR / f"{policy}_experts{expert}{suffix}_async_on",
                    )
                )

    plan_metadata: dict[tuple[str, int, str], dict] = {}
    for index, (policy, expert, workload, plan_path, systems, async_mode, out_dir) in enumerate(runs):
        result = run_sweep(
            "verify",
            plan_path,
            out_dir,
            workloads=[workload],
            experts=[expert],
            repeats=repeats,
            require_real=execute,
            dry_run_commands=not execute,
            execute=execute,
            base_port=base_port + index * 100,
            model_dir=model_dir,
            init_path=init_path,
            kt_env=kt_env,
            bench_tool=bench_tool,
            systems=systems,
            request_count=5,
            warmup_request_count=warmup_request_count,
            output_tokens=output_tokens,
            skip_existing_success=True,
            min_linux_available_gib=min_linux_available_gib,
            c_mode="hook",
            ablation_policy=policy,
            async_planning_mode=async_mode,
            force_prompt=force_prompt,
        )
        manifest_path = Path(result["manifest_path"])
        manifest_paths.append(manifest_path)
        metadata = _compiled_plan_metadata(manifest_path)
        if metadata:
            plan_metadata[(policy, expert, workload)] = metadata
        if result.get("blocked"):
            blockers.extend(str(item) for item in result.get("blockers", []))
        blockers.extend(_manifest_environment_blockers(manifest_path))
        if execute and result.get("blocked"):
            break

    write_merged_manifest(manifest_paths, MERGED)
    merged = json.loads(MERGED.read_text())
    rows = merged.get("runs", [])
    _attach_plan_metadata(rows, plan_metadata)
    blocked = bool(blockers)
    merged.update(
        {
            "schema_version": "tilepo_adaptive_granularity_manifest_v1",
            "adaptive_mode": "throughput",
            "matrix": {
                "workloads": workloads,
                "experts": experts_list,
                "policies": tilepo_policies,
                "serving_shell": "kt_sglang",
                "tilepo_async": "on",
                "repeats": repeats,
                "request_count": request_count,
                "warmup_request_count": warmup_request_count,
                "output_tokens": output_tokens,
                "force_prompt": force_prompt,
                "precision": "bf16_kt_native_serving_shell_cuda_tilepo_dispatch",
                "comparison": "tilepo_v0_2_only" if v0_2_only else ("tilepo_self_ablation_focused" if focused_workload or focused_experts else "tilepo_self_ablation"),
                "strict_v0_2_win": strict_v0_2_win,
                "strict_native_tc": strict_native_tc,
                "offline_acceptance": offline_acceptance,
                "v0_2_only": v0_2_only,
            },
            "previous_evidence_snapshot": str(snapshot_path) if snapshot_path else None,
            "blocked": blocked,
            "blockers": sorted(set(blockers)),
            "expected_result_rows": expected_rows,
            "actual_result_rows": len(rows),
            "evidence_level": "real" if execute and not blocked else ("blocked" if blocked else "dry_run_commands"),
        }
    )
    MERGED.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")

    if execute and not blocked and len(rows) == expected_rows:
        from tilepo.reporting.adaptive_granularity import generate_adaptive_granularity_report

        generate_adaptive_granularity_report(
            MERGED,
            OUT_DIR,
            require_real=True,
            strict_v0_2_win=strict_v0_2_win,
            v0_2_only=v0_2_only,
        )
        if strict_native_tc:
            _assert_strict_native_tc(merged)
    elif blocked:
        (OUT_DIR / "tilepo_adaptive_granularity_report.md").write_text(
            "# TilePO Adaptive Granularity Report\n\n"
            "Gate: **BLOCKED**\n\n"
            "Real KT/SGLang execution did not start because the environment is incomplete.\n\n"
            "## Blockers\n\n"
            + "\n".join(f"- {item}" for item in sorted(set(blockers)))
            + "\n"
        )
    elif execute:
        (OUT_DIR / "tilepo_adaptive_granularity_report.md").write_text(
            "# TilePO Adaptive Granularity Report\n\n"
            "Gate: **FAIL**\n\n"
            f"Expected {expected_rows} real rows but found {len(rows)}.\n"
        )
    print(MERGED)
    if execute and (blocked or len(rows) != expected_rows):
        return 1
    return 0


def _write_kt_baseline_plan(expert_budget: int) -> Path:
    output = PLANS_DIR / f"kt_expert_experts{expert_budget}_async_off.tmem"
    plan = parse_tmem(BASE_PLAN.read_text())
    blocks = []
    for block in plan.blocks:
        values = dict(block.values)
        if block.kind == "workload":
            values["label"] = f"kt_expert_experts{expert_budget}"
        elif block.kind == "memory":
            values["experts_per_layer"] = int(expert_budget)
        elif block.kind == "schedule":
            values["async_planning"] = False
            values["deployment_mode"] = "safe"
        blocks.append(DSLBlock(block.kind, block.name, values, block.line))
    output.write_text(type(plan)(blocks).compiled_text())
    return output


def _compiled_plan_metadata(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text())
    compiled_manifest = data.get("compiled_manifest")
    if not compiled_manifest:
        return {}
    compiled_path = Path(compiled_manifest)
    if not compiled_path.exists():
        return {}
    compiled = json.loads(compiled_path.read_text())
    plan = compiled.get("tilepo_plan", {})
    if not plan:
        return {}
    tile_count = int(plan.get("tile_count", 0))
    dispatch = int(plan.get("estimated_dispatch_units", tile_count))
    return {"tile_count": tile_count, "estimated_dispatch_units": dispatch, "tilepo_plan": plan}


def _attach_plan_metadata(rows: list[dict], plan_metadata: dict[tuple[str, int, str], dict]) -> None:
    for row in rows:
        policy = str(row.get("tilepo_policy") or row.get("ablation_policy") or "")
        expert = int(row.get("experts_per_layer", 0))
        workload = str(row.get("workload", ""))
        metadata = plan_metadata.get((policy, expert, workload))
        if not metadata:
            continue
        row.setdefault("tile_count", metadata["tile_count"])
        row.setdefault("estimated_dispatch_units", metadata["estimated_dispatch_units"])
        if policy in {"tilepo_adaptive", "tilepo_atg", "tilepo_atg_tc_baa"}:
            row.setdefault("tilepo_plan", metadata["tilepo_plan"])


def _manifest_environment_blockers(manifest_path: Path) -> list[str]:
    data = json.loads(manifest_path.read_text())
    env = data.get("environment", {})
    if not isinstance(env, dict) or env.get("ready", True):
        return []
    return [str(item) for item in env.get("blockers", [])]


def _offline_preflight(root: Path, model_dir: str, init_path: str, bench_tool: Path | None) -> list[str]:
    if os.environ.get("TILEPO_OFFLINE_ACCEPTANCE") != "1":
        return []
    blockers: list[str] = []
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            blockers.append(f"{key}=1 is required for offline acceptance")
    if os.environ.get("TILEPO_DISABLE_NETWORK") != "1":
        blockers.append("TILEPO_DISABLE_NETWORK=1 is required for offline acceptance")
    if not Path(model_dir).exists():
        blockers.append(f"offline acceptance missing local model path: {model_dir}")
    if not Path(init_path).exists():
        blockers.append(f"offline acceptance missing local KT init path: {init_path}")
    local_bench = bench_tool or (root / "tools" / "openai_varprompt_bench")
    if not local_bench.exists():
        blockers.append(f"offline acceptance missing local benchmark tool: {local_bench}")
    for required in (
        root / "tools" / "tilepo_render_plan",
        root / "tools" / "report_tilepo_adaptive_granularity",
        root / "scripts" / "reproduce_adaptive_granularity.sh",
        root / "scripts" / "run_adaptive_granularity_offline.sh",
    ):
        if not required.exists():
            blockers.append(f"offline acceptance missing packaged file: {required}")
    return blockers


def _assert_strict_native_tc(merged: dict) -> None:
    rows = merged.get("runs", [])
    atb_rows = [
        row for row in rows
        if row.get("tilepo_policy") == "tilepo_atg_tc_baa"
        and row.get("workload") == "mixed"
        and int(row.get("experts_per_layer", 0) or 0) == 8
    ]
    if not atb_rows:
        raise SystemExit("strict native TC failed: missing mixed/8 ATB row")
    for row in atb_rows:
        if not bool(row.get("tc_native_consumed_coalesced_groups", False)):
            raise SystemExit("strict native TC failed: tc_native_consumed_coalesced_groups is false")
        if int(row.get("tc_native_descriptor_count", 0) or 0) != 8:
            raise SystemExit("strict native TC failed: tc_native_descriptor_count is not 8")
        if bool(row.get("serving_hook_returned_original", True)):
            raise SystemExit("strict native TC failed: serving_hook_returned_original is true")
        if int(row.get("serving_hook_replaced_count", 0) or 0) <= 0:
            raise SystemExit("strict native TC failed: serving_hook_replaced_count is zero")


def _snapshot_existing_evidence(label: str) -> Path | None:
    if not OUT_DIR.exists():
        return None
    excluded_names = {"snapshots", "shared_jit_cache", "cache", "__pycache__"}
    source_entries = [
        path for path in OUT_DIR.iterdir()
        if path.name not in excluded_names and path.exists()
    ]
    if not source_entries:
        return None

    safe_label = _safe_snapshot_label(label) or "before_reproduce"
    snapshot_path = _unique_snapshot_path(safe_label)
    snapshot_path.mkdir(parents=True, exist_ok=False)
    for source in source_entries:
        target = snapshot_path / source.name
        if source.is_dir():
            shutil.copytree(source, target, ignore=_snapshot_ignore)
        else:
            shutil.copy2(source, target)
    manifest = {
        "schema_version": "tilepo_evidence_snapshot_v1",
        "source_dir": str(OUT_DIR),
        "snapshot_path": str(snapshot_path),
        "source_manifest": str(MERGED) if MERGED.exists() else None,
        "label": safe_label,
        "excluded_directory_names": sorted(excluded_names),
    }
    (snapshot_path / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"snapshot_path={snapshot_path}", flush=True)
    return snapshot_path


def _safe_snapshot_label(label: str) -> str:
    text = (label or "").strip().lower()
    chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        elif char in {" ", ".", "/", ":"}:
            chars.append("_")
    return "".join(chars).strip("_")[:80]


def _unique_snapshot_path(label: str) -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = SNAPSHOTS_DIR / f"{stamp}_{label}"
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = SNAPSHOTS_DIR / f"{base.name}_{suffix}"
    return candidate


def _snapshot_ignore(_path: str, names: list[str]) -> set[str]:
    return {name for name in names if name in {"snapshots", "shared_jit_cache", "cache", "__pycache__"}}


if __name__ == "__main__":
    raise SystemExit(main())
PY
