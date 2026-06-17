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
