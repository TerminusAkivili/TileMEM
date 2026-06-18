# TileMEM ATB V0.2 Showcase

ATB is the TilePO V0.2 strategy that combines:

- **ATG**: adaptive tile granularity for placement.
- **TC**: tile coalescing for execution dispatch.
- **BAA**: bubble-aware async planning for keeping planning off the request critical path.

This directory is a lightweight presentation subsystem. It points to the implementation in the main TileMEM tree and keeps only compact evidence summaries that are suitable for GitHub.

## What This Demonstrates

The validated point is:

- Workload: `mixed`
- Experts per layer: `8`
- Output tokens: `8`
- Repeats: `3`
- Serving shell: KT/SGLang preserved
- Precision: BF16
- Acceptance: offline/local-only

ATB V0.2 passes the focused gate against the best fixed TilePO policy:

| Metric | ATB V0.2 | Best fixed | Delta |
| --- | ---: | ---: | ---: |
| Median tok/s | `13.842` | `8.606` | `+60.84%` |
| p95 latency | `1710.98 ms` | `3040.13 ms` | `-43.72%` |
| p99 latency | `1729.65 ms` | `3126.31 ms` | `-44.67%` |

The best fixed policy in this run is `tilepo_hybrid`.

## Core Interpretation

The win is not from enabling hidden KT optimizations. All policies in the focused matrix use the same KT/SGLang short-benchmark server flags:

- `--skip-server-warmup`
- `--disable-radix-cache`
- `--disable-overlap-schedule`
- `--disable-cuda-graph`
- `--disable-shared-experts-fusion`

ATB-only environment variables enforce native TC evidence:

- `TILEPO_REQUIRE_NATIVE_BACKEND=1`
- `TILEPO_HOOK_BACKEND_PROBE_LIMIT=1`

Those are evidence/fallback guards, not KT execution optimizations.

## Mechanism

For the first `mixed/8` point, ATG selects fine placement for all eight experts:

- Placement tiles: `16384`
- Native TC descriptors: `8`
- Execution dispatch units: `8`
- BAA critical path: `0.0 us`

The validated claim is therefore:

> Fine-grained placement is retained for memory admission flexibility, while TC coalesces those placement tiles into expert-level execution groups and BAA keeps planning off the critical path.

Known limitation: this point does not reduce placement tile count versus fixed fine. The report keeps this warning explicitly: `mixed/8 V0.2 tile count is not between coarse and fine`.

## Files

- [ATB_V0_2_BENEFITS.md](ATB_V0_2_BENEFITS.md): concise presentation write-up.
- [evidence_summary.json](evidence_summary.json): compact machine-readable result summary.
- [reproduce_atb_v0_2_mixed8_offline.sh](reproduce_atb_v0_2_mixed8_offline.sh): wrapper for the accepted offline command.
- [MANIFEST.md](MANIFEST.md): implementation and evidence path index.

## Full Evidence

The full local evidence is intentionally not stored in this showcase directory because raw serving logs and JIT caches are large. The source run produced:

- `evidence/adaptive_granularity/tilepo_adaptive_granularity_report.md`
- `evidence/adaptive_granularity/tilepo_adaptive_granularity_summary.json`
- `evidence/adaptive_granularity/tilepo_adaptive_granularity_manifest.json`
- `evidence/adaptive_granularity/tilepo_v0_2_offline_preflight.json`
