# TilePO Mixed-8 Fine Single-Point Demo

Date: 2026-06-15

This note records small live-demo validation paths for the GitHub release
checkout. The one-repeat command is a smoke benchmark; the three-repeat command
below rechecks the V0.1 `mixed:8 / tilepo_fine / async on` evidence point.

## Command

```bash
python3 tools/run_tilepo_sweep \
  --mode serve \
  --c-mode hook \
  --plan configs/tilepo_olmoe_bf16_only.tmem \
  --out-dir build/demo_mixed8_fine_hook_real \
  --workloads mixed \
  --experts 8 \
  --systems B,C \
  --repeats 1 \
  --request-count 5 \
  --warmup-request-count 1 \
  --output-tokens 8 \
  --startup-timeout-sec 900 \
  --model-dir /mnt/d/tilemem_runtime/models/OLMoE-1B-7B-0924-Instruct \
  --init-expert-location /mnt/d/tilemem_runtime/results/kt_tilemem_hotset_20260523/tilemem_hotset_counts.pt \
  --kt-env tilemem-v2-ktransformers \
  --ablation-policy tilepo_fine \
  --async-planning-mode on \
  --execute \
  --require-real
```

## Result

The run completed with `blocked=false`, `simulated=false`, and two successful
real rows.

| System | Policy | Warmup ms | Measured ms | tok/s | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| B / KT baseline | tilepo_fine, async on | 30919.1 | 14025.9 | 2.852 | 5085.6 |
| C / TilePO hook | tilepo_fine, async on | 29551.7 | 13379.1 | 2.990 | 5077.5 |

Single-point C vs B:

- tok/s: `+4.83%`
- p95: `+0.16%`

The C-side bootstrap marker showed:

```text
hot_backend_probe_status: success
serving_hook_active: true
serving_hook_invocations: 2304
serving_hook_backend_launch_count: 1
serving_hook_verify_pass_count: 1536
serving_hook_replaced_count: 0
serving_hook_returned_original: true
```

## Interpretation

This demo confirms that the GitHub checkout can run the same-budget B/C serving
path for `mixed:8`, and that the TilePO hook is active in the C-side serving
process. The benchmark excludes one warmup request from measured latency so the
first-request KT/MoE lazy-load cost does not dominate the single-point result.

Because this is one repeat with five measured requests, it should be presented
as a live smoke validation. The V0.1 claim should still cite the released
three-repeat ablation matrix for the stronger `mixed:8` result.

## Three-Repeat Recheck

The same GitHub checkout was then run with the V0.1 repeat count:

```bash
python3 tools/run_tilepo_sweep \
  --mode serve \
  --c-mode hook \
  --plan configs/tilepo_olmoe_bf16_only.tmem \
  --out-dir build/mixed8_fine_hook_r3 \
  --workloads mixed \
  --experts 8 \
  --systems B,C \
  --repeats 3 \
  --request-count 5 \
  --warmup-request-count 1 \
  --output-tokens 8 \
  --startup-timeout-sec 900 \
  --model-dir /mnt/d/tilemem_runtime/models/OLMoE-1B-7B-0924-Instruct \
  --init-expert-location /mnt/d/tilemem_runtime/results/kt_tilemem_hotset_20260523/tilemem_hotset_counts.pt \
  --kt-env tilemem-v2-ktransformers \
  --ablation-policy tilepo_fine \
  --async-planning-mode on \
  --execute \
  --require-real
```

The run completed with `blocked=false`, `simulated=false`, and `6/6` successful
real rows.

| System | rep0 tok/s | rep1 tok/s | rep2 tok/s | median tok/s | median p95 ms | median p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B / KT baseline | 3.275 | 3.797 | 3.533 | 3.533 | 3976.7 | 4199.6 |
| C / TilePO hook | 2.410 | 4.938 | 5.246 | 4.938 | 2271.9 | 2424.9 |

Three-repeat C vs B:

- tok/s: `+39.78%`
- p95: `+42.87%`

For comparison, the released V0.1 evidence matrix reports `mixed:8` median
`kt_expert/off` at `17.310 tok/s` and `1199.99 ms` p95, and `tilepo_fine/on`
at `22.749 tok/s` and `964.23 ms` p95. That corresponds to `+31.42%` tok/s and
`+19.65%` p95 improvement in the archived matrix.

The C-side bootstrap markers for all three repeats showed:

```text
hot_backend_probe_status: success
tilepo_policy: tilepo_fine
tilepo_async_planning: on
tilepo_tile_count: 1024
serving_hook_active: true
serving_hook_invocations: 2304
serving_hook_backend_launch_count: 1
serving_hook_verify_pass_count: 1536
serving_hook_verify_fail_count: 0
serving_hook_replaced_count: 0
serving_hook_returned_original: true
```

The benchmark intentionally keeps the same-budget B/C path and records the
current conservative TilePO hook behavior. The hook reaches the SGLang/KT MoE
path and records TilePO runtime/backend evidence, but it still returns the
original SGLang/KT BF16 output. Do not present this recheck as full native MoE
kernel replacement.

## Runtime Optimization Notes

The GitHub sweep command uses a controlled SGLang/KT configuration for
comparability. In `tilepo/sweep.py`, the server command sets one running request
and explicitly disables radix cache, overlap scheduling, CUDA graph, and shared
experts fusion. This keeps the B/C comparison narrow, but it is not an
optimized absolute-throughput profile.

The installed SGLang runtime also exposes KT and serving optimization flags such
as `--kt-gpu-experts-ratio`, `--kt-enable-dynamic-expert-update`,
`--kt-max-deferred-experts-per-token`, `--kt-gpu-prefill-token-threshold`,
`--moe-runner-backend`, `--cuda-graph-max-bs`, and overlap/radix-cache controls.
An optimized demo profile should be reported separately from the V0.1 controlled
evidence path.
