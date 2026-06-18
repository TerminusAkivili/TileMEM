# ATB V0.2 Manifest

## Implementation Paths

- ATG plan generation: `tilepo/ablation.py`
- TC descriptor generation: `tilepo/compiler.py`
- Native TC descriptor consumption: `tilepo/backends/cuda_backend.py`
- Runtime metrics: `tilepo/runtime/runtime.py`, `tilepo/runtime/metrics.py`
- KT/SGLang bootstrap and hook: `tilepo/kt_patch/bootstrap.py`, `tilepo/kt_patch/sglang_hook.py`
- Benchmark matrix and evidence merge: `tilepo/sweep.py`
- Report gate: `tilepo/reporting/adaptive_granularity.py`

## Reproduction Paths

- Online/local runner: `scripts/reproduce_adaptive_granularity.sh`
- Offline acceptance wrapper: `scripts/run_adaptive_granularity_offline.sh`
- Report tool: `tools/report_tilepo_adaptive_granularity`

## Accepted Evidence Paths

- Markdown report: `evidence/adaptive_granularity/tilepo_adaptive_granularity_report.md`
- Summary JSON: `evidence/adaptive_granularity/tilepo_adaptive_granularity_summary.json`
- Merged manifest: `evidence/adaptive_granularity/tilepo_adaptive_granularity_manifest.json`
- Offline preflight: `evidence/adaptive_granularity/tilepo_v0_2_offline_preflight.json`
- Full run log: `evidence/adaptive_granularity/atb_v0_2_mixed8_repeats3_offline_after_parity_fix.log`

## Gate

The focused gate requires:

- `tilepo_atg_tc_baa tok/s > max(tilepo_coarse, tilepo_fine, tilepo_hybrid)`
- p95 and p99 no worse than 3 percent versus the best fixed latency baseline
- native TC descriptor evidence inside the measured serving path
- no fallback counted as a native TC win
