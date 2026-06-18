#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PKG_NAME="TileMEM_TilePO_V0_1_20260611"
if [[ -d "publish/$PKG_NAME" ]]; then
  PACKAGE_DIR="publish/$PKG_NAME"
else
  PACKAGE_DIR="."
fi

python3 -m compileall -q tilemem tilepo TMAP
python3 tools/tests/assert_agent_skills.py
python3 tools/tests/assert_tilemem_sdk.py
python3 tools/tests/assert_tilemem_cli.py
python3 tools/tests/assert_checkpoint_integration.py
python3 tools/tests/assert_tilemem_industrial_quickstart.py
python3 tools/tests/assert_public_mir_interface.py
python3 tools/tests/assert_integration_interface.py
python3 tools/tests/assert_olmoe_integration_benchmark.py
python3 tools/tests/assert_customer_integration_end_to_end.py
python3 tools/tests/assert_tilepo_ablation.py
python3 tools/tests/assert_tilepo_adaptive_granularity.py
python3 tools/tests/assert_openai_varprompt_bench.py
python3 tools/tests/assert_sweep_bench_tool.py
python3 tools/tests/assert_tmap.py
bash scripts/reproduce_ablation.sh

for required in \
  "$PACKAGE_DIR/TMAP/README.md" \
  "$PACKAGE_DIR/SKILL/tilemem-environment-setup/SKILL.md" \
  "$PACKAGE_DIR/SKILL/tilemem-environment-setup/agents/openai.yaml" \
  "$PACKAGE_DIR/SKILL/tilemem-acceleration-path/SKILL.md" \
  "$PACKAGE_DIR/SKILL/tilemem-acceleration-path/agents/openai.yaml" \
  "$PACKAGE_DIR/SKILL/tilemem-backend-precision-path/SKILL.md" \
  "$PACKAGE_DIR/SKILL/tilemem-backend-precision-path/agents/openai.yaml" \
  "$PACKAGE_DIR/docs/customer_integration_end_to_end_example_20260613.md" \
  "$PACKAGE_DIR/docs/tilemem_checkpoint_integration.md" \
  "$PACKAGE_DIR/docs/tilemem_python_sdk_quickstart.md" \
  "$PACKAGE_DIR/docs/tilemem_tilepo_v2_execution_efficiency_roadmap_20260613.md" \
  "$PACKAGE_DIR/configs/models/model_spec_template.json" \
  "$PACKAGE_DIR/tilemem/__init__.py" \
  "$PACKAGE_DIR/tilemem/checkpoint.py" \
  "$PACKAGE_DIR/tilemem/sdk.py" \
  "$PACKAGE_DIR/tilepo/model_interface.py" \
  "$PACKAGE_DIR/tilepo/integration.py" \
  "$PACKAGE_DIR/tilepo/mir/io.py" \
  "$PACKAGE_DIR/examples/olmoe_external_cuda_backend.py" \
  "$PACKAGE_DIR/examples/customer_integration_end_to_end.py" \
  "$PACKAGE_DIR/examples/tilemem_checkpoint_integration.py" \
  "$PACKAGE_DIR/examples/tilemem_industrial_quickstart.py" \
  "$PACKAGE_DIR/kernels/gemm_fp8.cu" \
  "$PACKAGE_DIR/kernels/gemm_fp6.cu" \
  "$PACKAGE_DIR/kernels/gemm_fp4.cu" \
  "$PACKAGE_DIR/tools/benchmark_olmoe_integration_interface" \
  "$PACKAGE_DIR/tools/openai_varprompt_bench" \
  "$PACKAGE_DIR/tools/tilemem" \
  "$PACKAGE_DIR/tools/tests/assert_agent_skills.py" \
  "$PACKAGE_DIR/tools/tests/assert_integration_interface.py" \
  "$PACKAGE_DIR/tools/tests/assert_tilemem_cli.py" \
  "$PACKAGE_DIR/tools/tests/assert_checkpoint_integration.py" \
  "$PACKAGE_DIR/tools/tests/assert_tilemem_sdk.py" \
  "$PACKAGE_DIR/tools/tests/assert_tilemem_industrial_quickstart.py" \
  "$PACKAGE_DIR/tools/tests/assert_olmoe_integration_benchmark.py" \
  "$PACKAGE_DIR/tools/tests/assert_customer_integration_end_to_end.py" \
  "$PACKAGE_DIR/tools/tests/assert_tilepo_adaptive_granularity.py" \
  "$PACKAGE_DIR/tools/report_tilepo_adaptive_granularity" \
  "$PACKAGE_DIR/scripts/reproduce_adaptive_granularity.sh" \
  "$PACKAGE_DIR/tilepo/reporting/adaptive_granularity.py" \
  "$PACKAGE_DIR/tools/tests/assert_openai_varprompt_bench.py" \
  "$PACKAGE_DIR/tools/tests/assert_sweep_bench_tool.py" \
  "$PACKAGE_DIR/tools/tilemem_checkpoint_prepare" \
  "$PACKAGE_DIR/tools/tmap_predict" \
  "$PACKAGE_DIR/tools/tests/assert_public_mir_interface.py" \
  "$PACKAGE_DIR/tools/tests/assert_tmap.py"; do
  if [[ ! -f "$required" ]]; then
    echo "missing packaged TMAP artifact: $required" >&2
    exit 1
  fi
done

"$PACKAGE_DIR/tools/tilemem" doctor --json >/dev/null
(
  cd "$PACKAGE_DIR"
  python3 tools/tests/assert_openai_varprompt_bench.py
  python3 tools/tests/assert_sweep_bench_tool.py
)

if [[ -f "$PACKAGE_DIR/SHA256SUMS" ]]; then
  (cd "$PACKAGE_DIR" && sha256sum -c SHA256SUMS)
fi

if [[ -f "publish/$PKG_NAME.tar.gz.sha256" ]]; then
  (cd publish && sha256sum -c "$PKG_NAME.tar.gz.sha256")
  tar -tzf "publish/$PKG_NAME.tar.gz" \
    "$PKG_NAME/TMAP/README.md" \
    "$PKG_NAME/SKILL/tilemem-environment-setup/SKILL.md" \
    "$PKG_NAME/SKILL/tilemem-environment-setup/agents/openai.yaml" \
    "$PKG_NAME/SKILL/tilemem-acceleration-path/SKILL.md" \
    "$PKG_NAME/SKILL/tilemem-acceleration-path/agents/openai.yaml" \
    "$PKG_NAME/SKILL/tilemem-backend-precision-path/SKILL.md" \
    "$PKG_NAME/SKILL/tilemem-backend-precision-path/agents/openai.yaml" \
    "$PKG_NAME/docs/customer_integration_end_to_end_example_20260613.md" \
    "$PKG_NAME/docs/tilemem_checkpoint_integration.md" \
    "$PKG_NAME/docs/tilemem_python_sdk_quickstart.md" \
    "$PKG_NAME/docs/tilemem_tilepo_v2_execution_efficiency_roadmap_20260613.md" \
    "$PKG_NAME/configs/models/model_spec_template.json" \
    "$PKG_NAME/tilemem/__init__.py" \
    "$PKG_NAME/tilemem/checkpoint.py" \
    "$PKG_NAME/tilemem/sdk.py" \
    "$PKG_NAME/tilepo/model_interface.py" \
    "$PKG_NAME/tilepo/integration.py" \
    "$PKG_NAME/tilepo/mir/io.py" \
    "$PKG_NAME/examples/olmoe_external_cuda_backend.py" \
    "$PKG_NAME/examples/customer_integration_end_to_end.py" \
    "$PKG_NAME/examples/tilemem_checkpoint_integration.py" \
    "$PKG_NAME/examples/tilemem_industrial_quickstart.py" \
    "$PKG_NAME/kernels/gemm_fp8.cu" \
    "$PKG_NAME/kernels/gemm_fp6.cu" \
    "$PKG_NAME/kernels/gemm_fp4.cu" \
    "$PKG_NAME/tools/benchmark_olmoe_integration_interface" \
    "$PKG_NAME/tools/openai_varprompt_bench" \
    "$PKG_NAME/tools/tilemem" \
    "$PKG_NAME/tools/tests/assert_agent_skills.py" \
    "$PKG_NAME/tools/tests/assert_integration_interface.py" \
    "$PKG_NAME/tools/tests/assert_tilemem_cli.py" \
    "$PKG_NAME/tools/tests/assert_checkpoint_integration.py" \
    "$PKG_NAME/tools/tests/assert_tilemem_sdk.py" \
    "$PKG_NAME/tools/tests/assert_tilemem_industrial_quickstart.py" \
    "$PKG_NAME/tools/tests/assert_olmoe_integration_benchmark.py" \
    "$PKG_NAME/tools/tests/assert_customer_integration_end_to_end.py" \
    "$PKG_NAME/tools/tests/assert_tilepo_adaptive_granularity.py" \
    "$PKG_NAME/tools/report_tilepo_adaptive_granularity" \
    "$PKG_NAME/scripts/reproduce_adaptive_granularity.sh" \
    "$PKG_NAME/tilepo/reporting/adaptive_granularity.py" \
    "$PKG_NAME/tools/tests/assert_openai_varprompt_bench.py" \
    "$PKG_NAME/tools/tests/assert_sweep_bench_tool.py" \
    "$PKG_NAME/tools/tilemem_checkpoint_prepare" \
    "$PKG_NAME/tools/tmap_predict" \
    "$PKG_NAME/tools/tests/assert_public_mir_interface.py" \
    "$PKG_NAME/tools/tests/assert_tmap.py" >/dev/null
fi

echo "TileMEM / TilePO artifact verification passed."
