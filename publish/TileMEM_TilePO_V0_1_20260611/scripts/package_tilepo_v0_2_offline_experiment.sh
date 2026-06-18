#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT=""
CONFIG="configs/tilepo_olmoe_bf16_only.tmem"
INIT_EXPERT_LOCATION=""
OUT_DIR="build/tilepo_v0_2_offline_bundle"
KT_ENV="${TILEMEM_KT_ENV:-tilemem-v2-ktransformers}"
BENCH_TOOL=""
BASE_PORT="35100"
MIN_LINUX_AVAILABLE_GIB="8"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/package_tilepo_v0_2_offline_experiment.sh \
    --checkpoint PATH \
    --out-dir PATH \
    [--config PATH] \
    [--init-expert-location PATH] \
    [--kt-env NAME] \
    [--bench-tool PATH] \
    [--base-port PORT] \
    [--min-linux-available-gib VALUE]

Creates a local-only TilePO V0.2 offline experiment bundle. The packager does
not download artifacts; checkpoint, config, and optional hotset inputs must
already exist locally.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 2
}

copy_path() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='__pycache__/' --exclude='*.pyc' "$src" "$dst"
  else
    cp -a "$src" "$dst"
  fi
}

json_quote() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"
}

shell_quote() {
  printf "%q" "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --init-expert-location)
      INIT_EXPERT_LOCATION="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$CHECKPOINT" ]] || die "--checkpoint is required"
[[ -e "$CHECKPOINT" ]] || die "checkpoint path does not exist: $CHECKPOINT"
[[ -e "$CONFIG" ]] || die "config path does not exist: $CONFIG"
if [[ -n "$INIT_EXPERT_LOCATION" && ! -e "$INIT_EXPERT_LOCATION" ]]; then
  die "init expert location does not exist: $INIT_EXPERT_LOCATION"
fi
if [[ -n "$BENCH_TOOL" && ! -e "$BENCH_TOOL" ]]; then
  die "bench tool path does not exist: $BENCH_TOOL"
fi
[[ -e scripts/run_adaptive_granularity_offline.sh ]] || die "missing scripts/run_adaptive_granularity_offline.sh"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"/{scripts,configs,tools,metadata,logs,artifacts,manifests,wheelhouse}

copy_path scripts/run_adaptive_granularity_offline.sh "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
copy_path scripts/reproduce_adaptive_granularity.sh "$OUT_DIR/scripts/reproduce_adaptive_granularity.sh"
copy_path scripts/package_tilepo_v0_2_offline_experiment.sh "$OUT_DIR/scripts/package_tilepo_v0_2_offline_experiment.sh"
copy_path "$CONFIG" "$OUT_DIR/$CONFIG"
copy_path tools/report_tilepo_adaptive_granularity "$OUT_DIR/tools/report_tilepo_adaptive_granularity"
copy_path tools/tilemem "$OUT_DIR/tools/tilemem"
copy_path tools/v2 "$OUT_DIR/tools/v2"
copy_path tilepo "$OUT_DIR/tilepo"
copy_path tilemem "$OUT_DIR/tilemem"
copy_path pyproject.toml "$OUT_DIR/pyproject.toml"
copy_path Makefile "$OUT_DIR/Makefile"
if [[ -e tools/openai_varprompt_bench ]]; then
  copy_path tools/openai_varprompt_bench "$OUT_DIR/tools/openai_varprompt_bench"
fi
if [[ -n "$BENCH_TOOL" ]]; then
  copy_path "$BENCH_TOOL" "$OUT_DIR/tools/$(basename "$BENCH_TOOL")"
fi

checkpoint_abs="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
config_abs="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
init_abs=""
if [[ -n "$INIT_EXPERT_LOCATION" ]]; then
  init_abs="$(cd "$(dirname "$INIT_EXPERT_LOCATION")" && pwd)/$(basename "$INIT_EXPERT_LOCATION")"
fi
bench_arg=""
if [[ -n "$BENCH_TOOL" ]]; then
  bench_arg="$BENCH_TOOL"
elif [[ -e "$OUT_DIR/tools/openai_varprompt_bench" ]]; then
  bench_arg='__BUNDLE_BENCH_TOOL__'
fi

cat > "$OUT_DIR/run_full_experiment_offline.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$BUNDLE_ROOT"

MODEL_DIR=$(shell_quote "$checkpoint_abs")
INIT_EXPERT_LOCATION=$(shell_quote "$init_abs")
KT_ENV=$(shell_quote "$KT_ENV")
BASE_PORT=$(shell_quote "$BASE_PORT")
MIN_LINUX_AVAILABLE_GIB=$(shell_quote "$MIN_LINUX_AVAILABLE_GIB")
BENCH_TOOL=$(shell_quote "$bench_arg")
if [[ "\$BENCH_TOOL" == "__BUNDLE_BENCH_TOOL__" ]]; then
  BENCH_TOOL="\$BUNDLE_ROOT/tools/openai_varprompt_bench"
fi

export TILEMEM_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true

EXTRA_ARGS=("\$@")
if bash scripts/run_adaptive_granularity_offline.sh --help 2>&1 | grep -q -- '--strict-v0-2-win'; then
  has_strict=0
  for arg in "\${EXTRA_ARGS[@]}"; do
    if [[ "\$arg" == "--strict-v0-2-win" ]]; then
      has_strict=1
      break
    fi
  done
  if [[ "\$has_strict" == "0" ]]; then
    EXTRA_ARGS+=(--strict-v0-2-win)
  fi
else
  filtered=()
  for arg in "\${EXTRA_ARGS[@]}"; do
    if [[ "\$arg" != "--strict-v0-2-win" ]]; then
      filtered+=("\$arg")
    fi
  done
  EXTRA_ARGS=("\${filtered[@]}")
fi
if bash scripts/run_adaptive_granularity_offline.sh --help 2>&1 | grep -q -- '--strict-native-tc'; then
  has_strict_native=0
  for arg in "\${EXTRA_ARGS[@]}"; do
    if [[ "\$arg" == "--strict-native-tc" ]]; then
      has_strict_native=1
      break
    fi
  done
  if [[ "\$has_strict_native" == "0" ]]; then
    EXTRA_ARGS+=(--strict-native-tc)
  fi
fi
if bash scripts/run_adaptive_granularity_offline.sh --help 2>&1 | grep -q -- '--offline-acceptance'; then
  has_offline_acceptance=0
  for arg in "\${EXTRA_ARGS[@]}"; do
    if [[ "\$arg" == "--offline-acceptance" ]]; then
      has_offline_acceptance=1
      break
    fi
  done
  if [[ "\$has_offline_acceptance" == "0" ]]; then
    EXTRA_ARGS+=(--offline-acceptance)
  fi
fi

command=(
  bash scripts/run_adaptive_granularity_offline.sh
  --execute
  --model-dir "\$MODEL_DIR"
  --kt-env "\$KT_ENV"
  --base-port "\$BASE_PORT"
  --min-linux-available-gib "\$MIN_LINUX_AVAILABLE_GIB"
)
if [[ -n "\$INIT_EXPERT_LOCATION" ]]; then
  command+=(--init-expert-location "\$INIT_EXPERT_LOCATION")
fi
if [[ -n "\$BENCH_TOOL" ]]; then
  command+=(--bench-tool "\$BENCH_TOOL")
fi
command+=("\${EXTRA_ARGS[@]}")

printf '%q ' "\${command[@]}" > logs/run_full_experiment_offline.command
printf '\n' >> logs/run_full_experiment_offline.command
exec "\${command[@]}"
EOF
chmod +x "$OUT_DIR/run_full_experiment_offline.sh"

git_revision="$(git rev-parse HEAD 2>/dev/null || true)"
git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
git_status="$(git status --short 2>/dev/null || true)"
git_diff_stat="$(git diff --stat 2>/dev/null || true)"
nvidia_smi="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true)"
nvcc_version="$(nvcc --version 2>/dev/null || true)"

find "$OUT_DIR" -type f ! -path "$OUT_DIR/metadata/SHA256SUMS" -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sed "s#  $OUT_DIR/#  #" \
  > "$OUT_DIR/metadata/SHA256SUMS"

grep -q -- '--strict-native-tc' "$OUT_DIR/scripts/reproduce_adaptive_granularity.sh"
grep -q -- '--strict-native-tc' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
grep -q -- '--offline-acceptance' "$OUT_DIR/scripts/reproduce_adaptive_granularity.sh"
grep -q -- '--offline-acceptance' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
grep -q -- 'HF_HUB_OFFLINE=1' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
grep -q -- 'TRANSFORMERS_OFFLINE=1' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
grep -q -- 'HF_DATASETS_OFFLINE=1' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
grep -q -- '--strict-native-tc' "$OUT_DIR/run_full_experiment_offline.sh"
grep -q -- '--offline-acceptance' "$OUT_DIR/run_full_experiment_offline.sh"

python3 - "$OUT_DIR" "$checkpoint_abs" "$config_abs" "$init_abs" "$KT_ENV" "$BASE_PORT" "$MIN_LINUX_AVAILABLE_GIB" "$git_revision" "$git_branch" "$git_status" "$git_diff_stat" "$nvidia_smi" "$nvcc_version" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

(
    out_dir,
    checkpoint,
    config,
    init_path,
    kt_env,
    base_port,
    min_linux_available_gib,
    git_revision,
    git_branch,
    git_status,
    git_diff_stat,
    nvidia_smi,
    nvcc_version,
) = sys.argv[1:]

root = Path(out_dir)

def digest(path_text: str) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

metadata = {
    "schema_version": "tilepo_v0_2_offline_bundle_v1",
    "offline": True,
    "source_root": str(Path.cwd()),
    "bundle_root": str(root.resolve()),
    "inputs": {
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest(checkpoint),
        "config": config,
        "config_sha256": digest(config),
        "init_expert_location": init_path or None,
        "init_expert_location_sha256": digest(init_path),
    },
    "commands": {
        "entrypoint": "bash run_full_experiment_offline.sh --execute --strict-v0-2-win --strict-native-tc --offline-acceptance",
        "runner": "bash scripts/run_adaptive_granularity_offline.sh --execute",
        "kt_env": kt_env,
        "base_port": base_port,
        "min_linux_available_gib": min_linux_available_gib,
    },
    "environment": {
        "TILEMEM_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    },
    "git": {
        "revision": git_revision or None,
        "branch": git_branch or None,
        "status_short": git_status.splitlines(),
        "diff_stat": git_diff_stat.splitlines(),
    },
    "cuda": {
        "nvidia_smi": nvidia_smi.splitlines(),
        "nvcc_version": nvcc_version.splitlines(),
    },
    "checksums": "metadata/SHA256SUMS",
    "blocked_manifest_path": "artifacts/blocked_manifest.json",
    "expected_outputs": {
        "logs": "logs/",
        "manifests": "manifests/",
        "report": "evidence/adaptive_granularity/",
    },
}
(root / "metadata" / "bundle_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY

find "$OUT_DIR" -type f ! -path "$OUT_DIR/metadata/SHA256SUMS" -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sed "s#  $OUT_DIR/#  #" \
  > "$OUT_DIR/metadata/SHA256SUMS"

echo "$OUT_DIR"
echo "$OUT_DIR/run_full_experiment_offline.sh"
echo "$OUT_DIR/metadata/bundle_metadata.json"
