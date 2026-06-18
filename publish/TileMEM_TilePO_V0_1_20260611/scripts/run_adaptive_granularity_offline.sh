#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE=0
PREFLIGHT_ONLY=0
IGNORE_ACTIVE_RUN=0
RESUME=0
SKIP_GPU_CHECK=0
FREE_IDE_MEMORY=0
SKIP_QUICK_VERIFY=0
STRICT_V0_2_WIN=0
STRICT_NATIVE_TC=0
OFFLINE_ACCEPTANCE=0
V0_2_ONLY=0
BASE_PORT=35100
MODEL_DIR="/mnt/d/tilemem_runtime/models/OLMoE-1B-7B-0924-Instruct"
INIT_PATH="/mnt/d/tilemem_runtime/results/kt_tilemem_hotset_20260523/tilemem_hotset_counts.pt"
KT_ENV="${TILEMEM_KT_ENV:-tilemem-v2-ktransformers}"
BENCH_TOOL=""
MIN_LINUX_AVAILABLE_GIB=8
LOG_PATH=""
FOCUSED_WORKLOAD=""
FOCUSED_EXPERTS=""
OUTPUT_TOKENS=4
REPEATS=1
FORCE_PROMPT=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_adaptive_granularity_offline.sh [--execute] [options]

Runs the TilePO V0.2 self-ablation matrix with local-only inputs inside the
KT/SGLang serving shell. The script exports HF/Transformers offline flags and
never downloads models. It also uses the packaged local benchmark runner and
keeps Windows host-memory inspection off the per-request critical path; WSL
memory and GPU preflight checks remain enabled.

Options:
  --execute                         Run the real KT/SGLang benchmark matrix.
  --strict-v0-2-win                 Require V0.2 to strictly beat fixed TilePO policies.
  --strict-native-tc                Require mixed/8 ATB native TC serving-path evidence.
  --offline-acceptance              Record and enforce final offline acceptance environment.
  --v0-2-only                       Run only the new ATB policy.
  --workload NAME                   Focus the matrix on one workload.
  --experts N                       Focus the matrix on one expert budget.
  --output-tokens N                 Output tokens per request.
  --repeats N                       Number of repeats per focused point.
  --force-prompt TEXT               Use one repeated prompt for warmup and measured requests.
  --preflight-only                  Check local readiness and exit.
  --resume                          Keep existing successful rows and run only missing/failed rows.
  --model-dir PATH                  Local HF checkpoint directory.
  --init-expert-location PATH       Local KT hotset/frequency file.
  --kt-env NAME                     Conda env with sglang and ktransformers.
  --bench-tool PATH                 Local openai_varprompt_bench path.
  --base-port PORT                  First serving port.
  --min-linux-available-gib VALUE   Required WSL/Linux available memory.
  --free-ide-memory                 Stop large VS Code language-service workers.
  --ignore-active-run               Do not fail if a benchmark already appears active.
  --skip-gpu-check                  Skip nvidia-smi requirement.
  --skip-quick-verify               Skip tools/tilemem verify --quick preflight.
  --log PATH                        Tee real execution output to PATH.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      EXECUTE=1
      shift
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      shift
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
    --resume)
      RESUME=1
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
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --min-linux-available-gib)
      MIN_LINUX_AVAILABLE_GIB="$2"
      shift 2
      ;;
    --free-ide-memory)
      FREE_IDE_MEMORY=1
      shift
      ;;
    --ignore-active-run)
      IGNORE_ACTIVE_RUN=1
      shift
      ;;
    --skip-gpu-check)
      SKIP_GPU_CHECK=1
      shift
      ;;
    --skip-quick-verify)
      SKIP_QUICK_VERIFY=1
      shift
      ;;
    --log)
      LOG_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BENCH_TOOL" ]]; then
  if [[ -e "$ROOT/tools/openai_varprompt_bench" ]]; then
    BENCH_TOOL="$ROOT/tools/openai_varprompt_bench"
  fi
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
if [[ "$OFFLINE_ACCEPTANCE" == "1" ]]; then
  export TILEPO_OFFLINE_ACCEPTANCE=1
  export TILEPO_DISABLE_NETWORK=1
fi
export TILEPO_STRICT_NATIVE_TC="$STRICT_NATIVE_TC"
export TILEMEM_SHARED_JIT_CACHE_DIR="${TILEMEM_SHARED_JIT_CACHE_DIR:-$ROOT/evidence/adaptive_granularity/shared_jit_cache}"

if [[ "$FREE_IDE_MEMORY" == "1" ]]; then
  python3 - <<'PY'
from __future__ import annotations

import os
import signal
import subprocess

out = subprocess.check_output(["ps", "-eo", "pid,rss,args"], text=True)
for line in out.splitlines()[1:]:
    parts = line.strip().split(None, 2)
    if len(parts) < 3:
        continue
    pid = int(parts[0])
    rss = int(parts[1])
    args = parts[2]
    if ("/bin/cpptools" in args or "server.bundle.js" in args) and rss > 200_000:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"stopped language-service process pid={pid} rss_kib={rss}", flush=True)
        except ProcessLookupError:
            pass
PY
  sleep 3
fi

write_blocked_manifest() {
  local preflight_payload="$1"
  local preflight_path
  preflight_path="$(mktemp)"
  printf '%s\n' "$preflight_payload" > "$preflight_path"
  python3 - "$preflight_path" "$STRICT_V0_2_WIN" "$V0_2_ONLY" "$FOCUSED_WORKLOAD" "$FOCUSED_EXPERTS" "$OUTPUT_TOKENS" "$REPEATS" "$FORCE_PROMPT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

preflight_path = Path(sys.argv[1])
strict_v0_2_win = sys.argv[2] == "1"
v0_2_only = sys.argv[3] == "1"
focused_workload = sys.argv[4].strip()
focused_experts = sys.argv[5].strip()
output_tokens = int(sys.argv[6])
repeats = int(sys.argv[7])
force_prompt = sys.argv[8]
preflight = json.loads(preflight_path.read_text())
root = Path.cwd()
out_dir = root / "evidence" / "adaptive_granularity"
manifest_path = out_dir / "tilepo_adaptive_granularity_manifest.json"
artifact_path = root / "artifacts" / "blocked_manifest.json"
workloads = [focused_workload] if focused_workload else ["mixed", "long_context"]
experts = [int(focused_experts)] if focused_experts else [6, 8, 10]
policies = ["tilepo_atg_tc_baa"] if v0_2_only else ["tilepo_coarse", "tilepo_fine", "tilepo_hybrid", "tilepo_atg_tc_baa"]
expected_rows = len(workloads) * len(experts) * len(policies) * repeats
blockers = [str(item) for item in preflight.get("blockers", [])]

out_dir.mkdir(parents=True, exist_ok=True)
artifact_path.parent.mkdir(parents=True, exist_ok=True)
manifest = {
    "schema_version": "tilepo_adaptive_granularity_manifest_v1",
    "adaptive_mode": "throughput",
    "matrix": {
        "workloads": workloads,
        "experts": experts,
        "policies": policies,
        "serving_shell": "kt_sglang",
        "tilepo_async": "on",
        "repeats": repeats,
        "request_count": 5,
        "warmup_request_count": 5 if force_prompt else 1,
        "output_tokens": output_tokens,
        "force_prompt": force_prompt,
        "precision": "bf16_kt_native_serving_shell_cuda_tilepo_dispatch",
        "comparison": "tilepo_v0_2_only" if v0_2_only else "tilepo_self_ablation",
        "strict_v0_2_win": strict_v0_2_win,
        "v0_2_only": v0_2_only,
    },
    "runs": [],
    "blocked": True,
    "blockers": sorted(set(blockers)),
    "expected_result_rows": expected_rows,
    "actual_result_rows": 0,
    "evidence_level": "blocked",
    "offline_preflight": preflight,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
(out_dir / "tilepo_adaptive_granularity_report.md").write_text(
    "# TilePO Adaptive Granularity Report\n\n"
    "Gate: **BLOCKED**\n\n"
    "Real KT/SGLang execution did not start because offline preflight found blockers.\n\n"
    "## Blockers\n\n"
    + "\n".join(f"- {item}" for item in sorted(set(blockers)))
    + "\n"
)
shutil.copyfile(manifest_path, artifact_path)
PY
  rm -f "$preflight_path"
}

preflight_json="$(python3 - "$EXECUTE" "$PREFLIGHT_ONLY" "$IGNORE_ACTIVE_RUN" "$SKIP_GPU_CHECK" "$SKIP_QUICK_VERIFY" "$MODEL_DIR" "$INIT_PATH" "$KT_ENV" "$BENCH_TOOL" "$MIN_LINUX_AVAILABLE_GIB" "$STRICT_V0_2_WIN" "$STRICT_NATIVE_TC" "$OFFLINE_ACCEPTANCE" "$REPEATS" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

execute = sys.argv[1] == "1"
preflight_only = sys.argv[2] == "1"
ignore_active_run = sys.argv[3] == "1"
skip_gpu_check = sys.argv[4] == "1"
skip_quick_verify = sys.argv[5] == "1"
model_dir = Path(sys.argv[6])
init_path = Path(sys.argv[7])
kt_env = sys.argv[8]
bench_tool_text = sys.argv[9]
bench_tool = Path(bench_tool_text) if bench_tool_text else None
min_linux_available_gib = float(sys.argv[10])
strict_v0_2_win = sys.argv[11] == "1"
strict_native_tc = sys.argv[12] == "1"
offline_acceptance = sys.argv[13] == "1"
repeats = int(sys.argv[14])
root = Path.cwd()


def linux_available_gib() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return float("inf")
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return float("inf")


def active_runs() -> list[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception:
        return []
    needles = (
        "scripts/reproduce_adaptive_granularity.sh",
        "openai_varprompt_bench",
        "sglang.launch_server",
    )
    current = {str(os.getpid()), str(os.getppid())}
    rows = []
    for line in out.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        pid, _, args = stripped.partition(" ")
        if pid in current:
            continue
        if any(needle in args for needle in needles):
            rows.append(stripped)
    return rows


def conda_import_ready() -> tuple[bool, str]:
    if shutil.which("conda") is None:
        return False, "conda unavailable"
    code = (
        "import importlib.util, sys; "
        "missing=[m for m in ['sglang','ktransformers','torch','transformers'] "
        "if importlib.util.find_spec(m) is None]; "
        "print(','.join(missing)); "
        "raise SystemExit(1 if missing else 0)"
    )
    proc = subprocess.run(
        ["conda", "run", "-n", kt_env, "python", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=45,
    )
    if proc.returncode == 0:
        return True, ""
    missing = proc.stdout.strip() or proc.stderr.strip() or f"returncode={proc.returncode}"
    return False, f"KT env '{kt_env}' import check failed: {missing}"


def native_tc_descriptor_preflight(root: Path) -> dict:
    from tilepo.ablation import render_tilepo_plan
    from tilepo.backends.cuda_backend import CUDABackend
    from tilepo.compiler import compile_plan

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        plan_path = tmp_path / "tilepo_atg_tc_baa_mixed8.tmem"
        plan_path.write_text(
            render_tilepo_plan(
                root / "configs" / "tilepo_olmoe_bf16_only.tmem",
                expert_budget=8,
                policy="tilepo_atg_tc_baa",
                async_planning=True,
                workload_profile="mixed",
            )
        )
        manifest = compile_plan(plan_path, tmp_path / "compiled").manifest
        result = CUDABackend(require_native=True).execute(
            {"topk": [(0, 0)], "require_tilemem": True, "payload": "ok"},
            manifest,
        )
        return {
            "tc_native_consumed_coalesced_groups": bool(
                result.get("tc_native_consumed_coalesced_groups", False)
            ),
            "tc_native_descriptor_count": int(result.get("tc_native_descriptor_count", 0) or 0),
            "tc_native_entrypoint": str(result.get("tc_native_entrypoint", "")),
            "tc_native_descriptor_layout": str(result.get("tc_native_descriptor_layout", "")),
            "cuda_descriptor_metrics_measured": bool(
                result.get("cuda_descriptor_metrics_measured", False)
            ),
            "cuda_descriptor_traversal_us": float(result.get("cuda_descriptor_traversal_us", 0.0) or 0.0),
        }


blockers: list[str] = []
warnings: list[str] = []
native_tc_preflight: dict = {}
active = active_runs()
if active and not ignore_active_run:
    blockers.append("active TileMEM/KT/SGLang benchmark process detected")
if not model_dir.exists():
    blockers.append(f"missing local model directory: {model_dir}")
elif not (model_dir / "config.json").exists():
    blockers.append(f"missing local model config.json: {model_dir / 'config.json'}")
if not init_path.exists():
    blockers.append(f"missing local KT hotset/init file: {init_path}")
if bench_tool is None or not bench_tool.exists():
    blockers.append("missing local openai_varprompt_bench")
for required in (
    root / "tools" / "tilepo_render_plan",
    root / "tools" / "report_tilepo_adaptive_granularity",
    root / "scripts" / "reproduce_adaptive_granularity.sh",
    root / "scripts" / "run_adaptive_granularity_offline.sh",
):
    if not required.exists():
        blockers.append(f"offline acceptance missing packaged file: {required}")
if offline_acceptance:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            blockers.append(f"{key}=1 is required for offline acceptance")
    if os.environ.get("TILEPO_DISABLE_NETWORK") != "1":
        blockers.append("TILEPO_DISABLE_NETWORK=1 is required for offline acceptance")
if shutil.which("python3") is None:
    blockers.append("python3 unavailable")
if not skip_gpu_check and shutil.which("nvidia-smi") is None:
    blockers.append("nvidia-smi unavailable")
available = linux_available_gib()
if available < min_linux_available_gib:
    blockers.append(
        f"Linux available memory {available:.2f} GiB is below required {min_linux_available_gib:.2f} GiB"
    )
ready, reason = conda_import_ready()
if not ready:
    blockers.append(reason)
if execute and strict_native_tc:
    try:
        native_tc_preflight = native_tc_descriptor_preflight(root)
        if not bool(native_tc_preflight.get("tc_native_consumed_coalesced_groups", False)):
            blockers.append("TilePO V0.2 native TC descriptor preflight did not consume coalesced groups")
        if int(native_tc_preflight.get("tc_native_descriptor_count", 0) or 0) != 8:
            blockers.append("TilePO V0.2 native TC descriptor preflight did not produce 8 descriptors")
        if native_tc_preflight.get("tc_native_entrypoint") != "tilepo_cuda_dispatch_coalesced_gemm":
            blockers.append("TilePO V0.2 native TC descriptor preflight entrypoint mismatch")
        if native_tc_preflight.get("tc_native_descriptor_layout") != "tilepo_cuda_coalesced_group_desc_v1":
            blockers.append("TilePO V0.2 native TC descriptor preflight layout mismatch")
    except Exception as exc:
        blockers.append(f"TilePO V0.2 native TC descriptor preflight failed: {exc}")

quick_verify_stdout = ""
if not blockers and not skip_quick_verify:
    proc = subprocess.run(
        ["tools/tilemem", "verify", "--quick"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=120,
    )
    quick_verify_stdout = proc.stdout.strip()
    if proc.returncode != 0:
        blockers.append(f"tools/tilemem verify --quick failed with returncode {proc.returncode}: {proc.stderr.strip()}")

payload = {
    "schema_version": "tilemem_adaptive_offline_preflight_v1",
    "status": "blocked" if blockers else "passed",
    "execute": execute,
    "preflight_only": preflight_only,
    "repo_root": str(root),
    "model_dir": str(model_dir),
    "init_expert_location": str(init_path),
    "kt_env": kt_env,
    "bench_tool": str(bench_tool) if bench_tool else "",
    "linux_available_gib": available,
    "min_linux_available_gib": min_linux_available_gib,
    "skip_gpu_check": skip_gpu_check,
    "skip_quick_verify": skip_quick_verify,
    "strict_v0_2_win": strict_v0_2_win,
    "strict_native_tc": strict_native_tc,
    "native_tc_preflight": native_tc_preflight,
    "repeats": repeats,
    "offline_acceptance": offline_acceptance,
    "offline_acceptance_ready": offline_acceptance and not blockers,
    "ready": not blockers,
    "active_run_count": len(active),
    "active_runs": active[:5],
    "offline_env": {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", ""),
        "TILEPO_DISABLE_NETWORK": os.environ.get("TILEPO_DISABLE_NETWORK", ""),
        "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY", ""),
        "WANDB_DISABLED": os.environ.get("WANDB_DISABLED", ""),
    },
    "quick_verify_stdout": quick_verify_stdout,
    "blockers": blockers,
    "warnings": warnings,
}
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(1 if blockers else 0)
PY
)" || {
  status=$?
  if [[ -n "$preflight_json" ]]; then
    mkdir -p "$ROOT/evidence/adaptive_granularity"
    printf '%s\n' "$preflight_json" > "$ROOT/evidence/adaptive_granularity/tilepo_v0_2_offline_preflight.json"
    write_blocked_manifest "$preflight_json"
  fi
  printf '%s\n' "$preflight_json"
  exit "$status"
}

mkdir -p "$ROOT/evidence/adaptive_granularity"
printf '%s\n' "$preflight_json" > "$ROOT/evidence/adaptive_granularity/tilepo_v0_2_offline_preflight.json"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  printf '%s\n' "$preflight_json"
  exit 0
fi

echo "$preflight_json"

command=(
  bash scripts/reproduce_adaptive_granularity.sh
  --kt-env "$KT_ENV"
  --model-dir "$MODEL_DIR"
  --init-expert-location "$INIT_PATH"
  --base-port "$BASE_PORT"
)
if [[ -n "$BENCH_TOOL" ]]; then
  command+=(--bench-tool "$BENCH_TOOL")
fi
command+=(--min-linux-available-gib "$MIN_LINUX_AVAILABLE_GIB")
if [[ "$EXECUTE" == "1" ]]; then
  command+=(--execute)
fi
if [[ "$RESUME" == "1" ]]; then
  command+=(--resume)
fi
if [[ "$STRICT_V0_2_WIN" == "1" ]]; then
  command+=(--strict-v0-2-win)
fi
if [[ "$STRICT_NATIVE_TC" == "1" ]]; then
  command+=(--strict-native-tc)
fi
if [[ "$OFFLINE_ACCEPTANCE" == "1" ]]; then
  command+=(--offline-acceptance)
fi
if [[ "$V0_2_ONLY" == "1" ]]; then
  command+=(--v0-2-only)
fi
if [[ -n "$FOCUSED_WORKLOAD" ]]; then
  command+=(--workload "$FOCUSED_WORKLOAD")
fi
if [[ -n "$FOCUSED_EXPERTS" ]]; then
  command+=(--experts "$FOCUSED_EXPERTS")
fi
command+=(--output-tokens "$OUTPUT_TOKENS")
command+=(--repeats "$REPEATS")
if [[ -n "$FORCE_PROMPT" ]]; then
  command+=(--force-prompt "$FORCE_PROMPT")
fi

if [[ -n "$LOG_PATH" ]]; then
  mkdir -p "$(dirname "$LOG_PATH")"
  "${command[@]}" 2>&1 | tee "$LOG_PATH"
  command_status="${PIPESTATUS[0]}"
else
  "${command[@]}"
  command_status="$?"
fi

if [[ "$command_status" != "0" ]]; then
  exit "$command_status"
fi

if [[ "$STRICT_NATIVE_TC" == "1" ]]; then
  python3 - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

path = Path("evidence/adaptive_granularity/tilepo_adaptive_granularity_manifest.json")
data = json.loads(path.read_text())
rows = [
    row for row in data.get("runs", [])
    if row.get("tilepo_policy") == "tilepo_atg_tc_baa"
    and row.get("workload") == "mixed"
    and int(row.get("experts_per_layer", 0) or 0) == 8
]
if not rows:
    raise SystemExit("strict native TC failed: missing mixed/8 ATB row")
for row in rows:
    if not bool(row.get("tc_native_consumed_coalesced_groups", False)):
        raise SystemExit("strict native TC failed: tc_native_consumed_coalesced_groups is false")
    if int(row.get("tc_native_descriptor_count", 0) or 0) != 8:
        raise SystemExit("strict native TC failed: tc_native_descriptor_count is not 8")
    if bool(row.get("serving_hook_returned_original", True)):
        raise SystemExit("strict native TC failed: serving_hook_returned_original is true")
    if int(row.get("serving_hook_replaced_count", 0) or 0) <= 0:
        raise SystemExit("strict native TC failed: serving_hook_replaced_count is zero")
PY
fi
