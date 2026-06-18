#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

exec bash scripts/run_adaptive_granularity_offline.sh \
  --execute \
  --strict-native-tc \
  --offline-acceptance \
  --workload mixed \
  --experts 8 \
  --repeats 3 \
  --output-tokens 8 \
  --ignore-active-run \
  --log evidence/adaptive_granularity/atb_v0_2_mixed8_repeats3_offline_after_parity_fix.log \
  "$@"
