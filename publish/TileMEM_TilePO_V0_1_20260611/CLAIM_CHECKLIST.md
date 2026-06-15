# TilePO V0.1 Claim Checklist

## Can Say

- TileMEM is the project and TilePO is the algorithm/system.
- TilePO is a BF16 profile-guided tile-level placement/admission system for MoE
  serving.
- V0.1 includes source code, reports, public manifests, scripts, checksums, and a
  public priority roadmap.
- V0.1 completed 210/210 real successful rows and passed its gate.
- V0.1 compares TilePO and KT under the same expert budgets.
- V0.1 records tok/s, p95, p99, GPU peak, and CPU/DRAM peak.
- V0.1 shows TilePO wins over KT on `mixed` and `long_context` in the tested
  BF16 same-budget matrix.
- TilePO exposes policy boundaries: coarse, fine, hybrid, and async planning do
  not dominate uniformly.
- The GitHub checkout includes a live OpenAI-compatible serving bench helper
  for same-budget KT/TilePO rechecks when the model/runtime assets are provided.

## Must Qualify

- The V0.1 result is limited to the tested model/runtime/hardware setup.
- `tilepo/` is the implementation namespace used by the V0.1 TilePO artifact.
- A user must provide a compatible MoE checkpoint for real serving evaluation.
- The default public path is BF16. Low-bit code paths are not serving-quality
  claims in V0.1.
- The live TilePO hook path is conservative and observe-only unless a future
  replacement path is explicitly enabled and verified.

## Cannot Say

- TilePO universally beats KT/SGLang.
- TilePO replaces all KT/SGLang MoE kernels with native CUDA.
- TilePO proves FP8 or MXFP4 serving quality.
- TilePO proves fine-grained tiles are always better.
- TilePO is the first tile, cache, offload, or memory-hierarchy idea in history.
- Results generalize to every MoE model, GPU, and serving system without more
  evidence.

## Recommended One-Sentence Claim

TilePO is a BF16 profile-guided tile-level placement/admission system that
outperforms KT expert-level placement in the V0.1 same-budget `mixed` and
`long_context` MoE serving matrix while preserving explicit claim boundaries.
