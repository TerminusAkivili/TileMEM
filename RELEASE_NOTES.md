# TileMEM TilePO v0.1 Priority Release

Tag: `v0.1-priority-2026-06-11`

This release publicly discloses TilePO, a BF16 profile-guided tile-level
placement/admission system for MoE serving.

## Included

- TilePO source code under the `tilepo/` namespace.
- `.tmem` model/plan examples and model replacement templates.
- V0.1 ablation report, summary, public manifest, completion record, and
  generated plans.
- Technical report Markdown snapshot.
- Offline verification and custom-model quickstart scripts.
- SHA256 checksum tooling and release package script.
- OpenAI-compatible live serving bench helper for same-budget KT/TilePO
  rechecks.

## Main Evidence

```text
V0.1 rows: 210 / 210 real success
V0.1 gate: PASS
Workloads: mixed, long_context
Experts: 2, 4, 6, 8, 10
Repeats: 3
Request count: 5
Precision: BF16 / KT-native serving path
```

## Claim Boundary

This release does not claim full native CUDA MoE replacement, low-bit serving
quality, universal generalization, or that fine-grained tiles alone explain all
wins.

## Verify

```bash
bash examples/quickstart_offline.sh
bash scripts/verify_artifact.sh
```

The artifact verification gate checks the released evidence matrix, package
checksums, and the packaged live bench helper.

## Bring Your Own Model

```bash
export TILEMEM_MODEL_PATH=/path/to/moe/checkpoint
bash examples/quickstart_custom_model.sh
```
