# TilePO ATB V0.2 Native TC Execution

ATB V0.1 established the ATG + TC + BAA manifest contract, but its TC path was still descriptor evidence unless the serving backend consumed coalesced descriptors.

ATB V0.2 changes the success condition: `tilepo_atg_tc_baa` is accepted only when coalesced TC descriptors are consumed inside the measured KT/SGLang serving path.

## V0.2 Gate

- Native TC descriptors must be consumed.
- Descriptor-only evidence must fail the native gate.
- Fallback is allowed for safety but cannot be reported as a native TC win.
- The first required point is `workload=mixed`, `experts=8`, `output_tokens=8`.
- Final three-repeat acceptance must run offline/disconnected from local artifacts.

## Required Runtime Evidence

- `tc_native_consumed_coalesced_groups`
- `tc_native_descriptor_count`
- `tc_native_entrypoint`
- `tc_native_descriptor_layout`
- `execution_dispatch_units`
- `placement_tile_count`
- `baa_critical_path_us`
- `baa_metrics_measured`
- `cuda_descriptor_traversal_us`
- `cuda_descriptor_metrics_measured`
- `serving_hook_replaced_count`
- `serving_hook_returned_original`

## Offline Acceptance

The final mixed/8 three-repeat acceptance run must not require GitHub, pip, Hugging Face downloads, model downloads, package downloads, or any external HTTP endpoint.

The offline runner must use local model/checkpoint paths and local packaged scripts. It must export:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

The run must write an offline preflight record that proves the local model path, KT init path, benchmark tool, plan renderer, report tool, and package-local scripts exist before starting real serving.

## Acceptance

`tilepo_atg_tc_baa` must beat the best fixed TilePO policy on throughput and keep p95/p99 within 3 percent of the best fixed policy.

If native TC is unavailable or descriptor-only, the report must fail the V0.2 native gate. A fallback diagnostic row may be emitted, but it cannot be counted as a V0.2 win.

## 2026-06-18 Mixed/8 Offline Acceptance

Command:

```bash
bash scripts/run_adaptive_granularity_offline.sh \
  --execute \
  --strict-native-tc \
  --offline-acceptance \
  --workload mixed \
  --experts 8 \
  --repeats 3 \
  --output-tokens 8 \
  --ignore-active-run \
  --log evidence/adaptive_granularity/atb_v0_2_mixed8_repeats3_offline_after_parity_fix.log
```

Result: **PASS**.

- Offline preflight passed with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TILEPO_DISABLE_NETWORK=1`.
- Native TC descriptor preflight consumed coalesced groups with `tc_native_descriptor_count=8`, layout `tilepo_cuda_coalesced_group_desc_v1`, and entrypoint `tilepo_cuda_dispatch_coalesced_gemm`.
- The focused matrix produced 12 real rows: `tilepo_coarse`, `tilepo_fine`, `tilepo_hybrid`, and `tilepo_atg_tc_baa`, each repeated three times for `workload=mixed`, `experts=8`, `output_tokens=8`.
- All four policies now use the same KT/SGLang short-benchmark server flag profile: `--skip-server-warmup`, `--disable-radix-cache`, `--disable-overlap-schedule`, `--disable-cuda-graph`, and `--disable-shared-experts-fusion`.
- V0.2 median throughput was `13.842 tok/s` versus best fixed `tilepo_hybrid` at `8.606 tok/s`, a `60.84%` gain.
- V0.2 p95/p99 were `1710.98 ms` / `1729.65 ms`, both lower than best fixed `3040.13 ms` / `3126.31 ms`.
- V0.2 rows report `serving_hook_replaced_count=1`, `serving_hook_returned_original=false`, `tc_native_consumed_coalesced_groups=true`, and `tc_native_descriptor_count=8`.

Known limitation: for this first mixed/8 point, ATG deliberately selects fine placement for all eight experts, so `placement_tile_count=16384`, equal to the fixed fine policy. The validated claim is therefore **fine placement + coalesced native TC execution + BAA descriptor path**, not a reduced placement tile count. The report keeps this as a warning: `mixed/8 V0.2 tile count is not between coarse and fine`.

The main root cause of the previous failed run was benchmark parity: fixed policies ran with the short-benchmark KT/SGLang flags while `tilepo_atg_tc_baa` alone preserved KT optimizations and triggered a different cuda-graph/warmup path. V0.2 now keeps the server flags identical across the self-ablation matrix. Bootstrap prime evidence is also kept separate from measured serving replacement count, so `serving_hook_replaced_count` is incremented only by real hook invocation.
