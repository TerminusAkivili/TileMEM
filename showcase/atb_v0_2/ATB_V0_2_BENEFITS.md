# ATB V0.2 Benefits

## Summary

ATB V0.2 targets a memory-bound MoE serving bottleneck: fine-grained memory placement improves admission flexibility, but naive fine-grained execution can hurt dispatch, transfer, and grouped GEMM efficiency.

ATB separates the two layers:

```text
placement: fine-grained tiles
execution: coalesced expert-level dispatch groups
planning: double-buffered and off the request critical path
```

## Measured Result

Focused offline acceptance:

- Workload: `mixed`
- Experts: `8`
- Output tokens: `8`
- Repeats: `3`
- Best fixed baseline: `tilepo_hybrid`

| Metric | ATB V0.2 | Best fixed | Delta |
| --- | ---: | ---: | ---: |
| Median tok/s | `13.842` | `8.606` | `+60.84%` |
| p95 latency | `1710.98 ms` | `3040.13 ms` | `-43.72%` |
| p99 latency | `1729.65 ms` | `3126.31 ms` | `-44.67%` |
| GPU peak | `5.130 GiB` | `4.929 GiB` | `+4.08%` |

The gain is throughput and tail-latency oriented. It is not a GPU-memory-footprint win at this first point.

## Why It Improves Performance

ATB V0.2 keeps fine placement but avoids fine execution:

- `tile_count = 16384`
- `tc_native_descriptor_count = 8`
- `execution_dispatch_units = 8`
- `tc_native_consumed_coalesced_groups = true`

This means TileMEM can keep fine placement decisions for VRAM/DRAM admission while TC presents larger expert-level units to the execution path. The result is lower scheduling/metadata overhead and a friendlier grouped-GEMM dispatch shape.

BAA contributes by keeping planning out of the request critical path:

- `baa_double_buffered = true`
- `baa_critical_path_us = 0.0`

## Fairness Notes

The accepted run uses the same KT/SGLang short-benchmark server flags for all four policies:

- `tilepo_coarse`
- `tilepo_fine`
- `tilepo_hybrid`
- `tilepo_atg_tc_baa`

All policies disable the same KT/SGLang warmup/cache/cuda-graph features. ATB-only flags require native TC evidence and prevent fallback from being counted as success.

## Limitation

At `mixed/8`, ATG selects fine placement for all eight experts. Therefore, placement tile count is equal to fixed fine. The validated claim is not "fewer placement tiles"; it is:

> Fine placement plus coalesced native TC execution plus BAA descriptor planning.
