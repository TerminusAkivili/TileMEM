---
name: tilemem-acceleration-path
description: Use when connecting a MoE model or checkpoint to TileMEM, compiling MIR/manifests, preparing checkpoint artifacts, running TilePO/KT comparisons, or choosing a TileMEM acceleration path.
---

# TileMEM Acceleration Path

## Overview

Treat acceleration as a staged evidence chain: health check -> model/checkpoint metadata -> MIR/manifest/tile handles -> dry-run backend command -> same-budget KT vs TilePO benchmark.

## Required First Checks

```bash
tools/tilemem doctor
tools/tilemem verify --quick
python3 examples/tilemem_checkpoint_integration.py
python3 examples/tilemem_industrial_quickstart.py \
  --out-json build/tilemem_industrial_quickstart.json
```

Verify TileMEM import, checkpoint topology inference, manifest/tile map generation, BF16 fallback availability, and backend dry-run command construction before running long experiments.

## Connect A Checkpoint

Use dry-run first:

```bash
tools/tilemem checkpoint prepare \
  --checkpoint-dir /path/to/hf_moe_checkpoint \
  --out-dir build/my_moe_checkpoint_artifact \
  --backend sglang \
  --dry-run
```

For KT-native:

```bash
tools/tilemem checkpoint prepare \
  --checkpoint-dir /path/to/hf_moe_checkpoint \
  --out-dir build/my_moe_checkpoint_artifact_kt \
  --backend kt_native \
  --dry-run
```

Inspect `model_spec.json`, `model.mir.json`, `model.manifest.json`, `checkpoint_weight_map.json`, `tile_checkpoint_map.json`, and `checkpoint_artifact_summary.json`. Only use `--execute` after the generated command, backend binary, model path, tile map, and fallback path are correct.

## Compile And Select A Probe

Compile a public spec or `.tmem` plan:

```bash
tools/tilemem compile \
  --model-spec configs/models/model_spec_template.json \
  --out-dir build/my_moe_compile

tools/tilemem compile \
  --plan configs/models/model_template.tmem \
  --out-dir build/my_moe_plan_compile
```

Use the checked-in V0.1 evidence to pick a small same-budget probe:

```bash
tools/report_tilepo_ablation \
  --summary evidence/ablation/tilepo_ablation_summary.json \
  --out build/tilepo_ablation_report.md
```

For an unseen expert budget, run a short direct probe instead of extrapolating:

```bash
tools/run_tilepo_ablation --experts 12 --workload mixed --repeats 1
```

## Benchmark Decision Rule

Compare at the same expert budget:

- KT expert placement
- TilePO coarse
- TilePO fine
- TilePO hybrid
- async planning off/on when supported

Choose TilePO only when it beats KT at the same expert budget on throughput without p95/p99 regression. If the direct probe is weak, noisy, or VRAM is abundant enough that KT wins, recommend KT fallback or a shorter targeted probe.

## Guardrails

- Do not claim universal speedup; TilePO benefits are workload/hardware/budget dependent.
- Do not skip manifest and tile map inspection for real checkpoints.
- Do not benchmark TilePO against KT with different expert budgets.
- Keep BF16/KT fallback available while integrating a model.
