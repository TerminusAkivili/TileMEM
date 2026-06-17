# ATB V0.2 Native TC Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn ATB from descriptor-only evidence into a native Tile Coalescing execution path where coalesced groups are actually consumed by the KT/SGLang serving path.

**Architecture:** Keep ATG and BAA as manifest-time/runtime descriptors, but add a strict Native TC adapter boundary that converts `coalesced_groups` into measured execution units. V0.2 must fail closed: descriptor-only evidence is not accepted as native TC success, and ATB must either consume native TC descriptors or explicitly report a failed/native-unavailable gate.

**Tech Stack:** Python TilePO planner/compiler/runtime, KT/SGLang hook integration, CUDA backend descriptor adapter, OpenAI-compatible serving benchmark, existing TileMEM artifact verification scripts.

---

## Scope

V0.2 is the Native TC Execution release. It must answer one question:

Can `tilepo_atg_tc_baa` turn fine-grained placement tiles into coalesced execution units that the serving path actually consumes?

This plan intentionally does not attempt a full KT/SGLang kernel rewrite first. It adds the narrowest native adapter and evidence chain needed to prove whether coalesced dispatch has entered the measured request path.

## Non-Goals

- Do not claim speedup from descriptor-only TC evidence.
- Do not broaden beyond the first acceptance point before mixed / experts=8 passes.
- Do not replace KT/SGLang as the serving shell.
- Do not change model quality, dtype policy, or expert budget while comparing policies.
- Do not hide fallback as a V0.2 win.

## File Map

- Modify: `tilepo/ablation.py`
  - Owns ATG policy rendering and mixed/8 candidate selection.
- Modify: `tilepo/compiler.py`
  - Owns `coalesced_groups`, CUDA TC descriptors, BAA maps, and manifest invariants.
- Modify: `tilepo/backends/cuda_backend.py`
  - Owns native TC descriptor validation and consumption metrics.
- Modify: `tilepo/runtime/runtime.py`
  - Owns runtime dispatch flow and metric propagation from backend to request result.
- Modify: `tilepo/runtime/metrics.py`
  - Owns measured V0.2 TC/BAA metric fields.
- Modify: `tilepo/kt_patch/bootstrap.py`
  - Owns bootstrap marker evidence exported from the serving process.
- Modify: `tilepo/kt_patch/sglang_hook.py`
  - Owns KT/SGLang hook boundary and native TC adapter activation.
- Modify: `tilepo/sweep.py`
  - Owns real-run command manifest, benchmark profile metadata, and per-run evidence rows.
- Modify: `tools/openai_varprompt_bench`
  - Owns measured request rows and hook marker merge behavior.
- Modify: `tilepo/reporting/adaptive_granularity.py`
  - Owns V0.2 report gates.
- Modify: `tools/report_tilepo_adaptive_granularity`
  - CLI wrapper for report generation.
- Modify: `scripts/reproduce_adaptive_granularity.sh`
  - Main local reproduction entrypoint.
- Modify: `scripts/run_adaptive_granularity_offline.sh`
  - Packaged/offline reproduction entrypoint.
- Modify: `scripts/package_tilepo_v0_2_offline_experiment.sh`
  - Owns package completeness for V0.2 evidence.
- Modify: `scripts/verify_artifact.sh`
  - Owns release verification.
- Modify tests:
  - `tools/tests/assert_tilepo_ablation.py`
  - `tools/tests/assert_tilepo_adaptive_granularity.py`
  - `tools/tests/assert_openai_varprompt_bench.py`
- Create: `docs/tilepo_v0_2_native_tc_execution_20260617.md`
  - Human-readable design and evidence note.

## Acceptance Criteria

V0.2 passes only if all criteria below are true for `workload=mixed`, `experts=8`, `system=C`, `policy=tilepo_atg_tc_baa`, `output_tokens=8`:

- Final three-repeat acceptance is run offline/disconnected.
- The final run must not require GitHub, pip, Hugging Face downloads, model downloads, package downloads, or any external HTTP endpoint.
- The final run must use local model/checkpoint paths and local packaged scripts only.
- Offline mode must export `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`.
- Offline preflight must prove the model path, KT init path, benchmark tool, plan renderer, report tool, and package-local scripts exist before starting the real run.
- `tc_native_consumed_coalesced_groups == true`
- `tc_native_descriptor_count == 8`
- `tc_native_entrypoint == "tilepo_cuda_dispatch_coalesced_gemm"`
- `tc_native_descriptor_layout == "tilepo_cuda_coalesced_group_desc_v1"`
- `placement_tile_count >= 1024`
- `execution_dispatch_units == 8` for native TC mode
- `baa_critical_path_us == 0.0`
- `baa_metrics_measured == true`
- `cuda_descriptor_metrics_measured == true`
- `tilepo_atg_tc_baa tok/s > max(tilepo_coarse, tilepo_fine, tilepo_hybrid)`
- `tilepo_atg_tc_baa p95_ms <= best_fixed_p95_ms * 1.03`
- `tilepo_atg_tc_baa p99_ms <= best_fixed_p99_ms * 1.03`

If native TC is unavailable, the report must fail the V0.2 native gate. It may still emit a fallback diagnostic row, but that row cannot be counted as a V0.2 win.

## Design Decisions

1. **Native TC is a gate, not a label.**
   A row is native TC only when the serving process reports that coalesced descriptors were consumed in the request path.

2. **ATG remains manifest-time.**
   V0.2 does not introduce a dynamic Python planner in the request path. ATG selects a plan before serving.

3. **BAA remains off-critical-path evidence.**
   BAA must prove `baa_critical_path_us == 0.0` and measured marker presence. Runtime double buffering can remain descriptor-level in V0.2.

4. **Fallback is honest.**
   Best fixed fallback is allowed as safety behavior, but it is not accepted as Native TC Execution.

5. **One point first.**
   The first hard target is mixed / experts=8. Broader sweeps come after the gate passes.

---

### Task 1: Freeze V0.1 Baseline Evidence

**Files:**
- Already created archive: `/home/baobao/TileMEM_ATBv0.1`
- Create: `docs/tilepo_v0_2_native_tc_execution_20260617.md`

- [ ] **Step 1: Verify the V0.1 archive exists**

Run:

```bash
test -d /home/baobao/TileMEM_ATBv0.1
test -f /home/baobao/TileMEM_ATBv0.1/ATB_V0_1_ARCHIVE_MANIFEST.md
sed -n '1,80p' /home/baobao/TileMEM_ATBv0.1/ATB_V0_1_ARCHIVE_MANIFEST.md
```

Expected:

```text
# TileMEM ATB V0.1 Archive
```

- [ ] **Step 2: Write the V0.2 design note**

Create `docs/tilepo_v0_2_native_tc_execution_20260617.md` with:

```markdown
# TilePO ATB V0.2 Native TC Execution

ATB V0.1 established the ATG + TC + BAA manifest contract, but its TC path was still descriptor evidence unless the serving backend consumed coalesced descriptors.

ATB V0.2 changes the success condition: `tilepo_atg_tc_baa` is accepted only when coalesced TC descriptors are consumed inside the measured KT/SGLang serving path.

## V0.2 Gate

- Native TC descriptors must be consumed.
- Descriptor-only evidence must fail the native gate.
- Fallback is allowed for safety but cannot be reported as a native TC win.
- The first required point is `workload=mixed`, `experts=8`, `output_tokens=8`.

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

## Acceptance

`tilepo_atg_tc_baa` must beat the best fixed TilePO policy on throughput and keep p95/p99 within 3 percent of the best fixed policy.
```

- [ ] **Step 3: Commit the baseline/design note**

Run:

```bash
git add docs/tilepo_v0_2_native_tc_execution_20260617.md
git commit -m "docs: define ATB V0.2 native TC execution gate"
```

Expected: commit succeeds.

---

### Task 2: Add Native TC Manifest Invariants

**Files:**
- Modify: `tools/tests/assert_tilepo_ablation.py`
- Modify: `tilepo/ablation.py`
- Modify: `tilepo/compiler.py`

- [ ] **Step 1: Write the failing manifest test**

In `tools/tests/assert_tilepo_ablation.py`, add assertions for `tilepo_atg_tc_baa` mixed/8:

```python
assert atb_tilepo_plan["policy"] == "tilepo_atg_tc_baa"
assert atb_tilepo_plan["tc_native_required_for_v0_2"] is True
assert atb_tilepo_plan["tc_native_descriptor_layout"] == "tilepo_cuda_coalesced_group_desc_v1"
assert atb_tilepo_plan["tc_native_entrypoint"] == "tilepo_cuda_dispatch_coalesced_gemm"
assert atb_tilepo_plan["tc_native_expected_descriptor_count"] == 8
assert atb_tilepo_plan["placement_tile_count"] == len(atb_manifest["tile_offsets"])
assert atb_tilepo_plan["execution_dispatch_units"] == 8
assert len(atb_manifest["coalesced_groups"]) == 8
assert len(atb_manifest["cuda_tc_descriptor_buffer"]) == 8
assert sum(group["tile_count"] for group in atb_manifest["coalesced_groups"]) == atb_tilepo_plan["placement_tile_count"]
assert {group["expert"] for group in atb_manifest["coalesced_groups"]} == set(range(8))
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python3 tools/tests/assert_tilepo_ablation.py
```

Expected: FAIL because at least one `tc_native_*` field is missing or `execution_dispatch_units` is not native TC strict.

- [ ] **Step 3: Render native TC fields in ATG**

In `tilepo/ablation.py`, ensure the mixed/8 `tilepo_atg_tc_baa` plan includes:

```python
"tc_native_required_for_v0_2": True,
"tc_native_descriptor_layout": "tilepo_cuda_coalesced_group_desc_v1",
"tc_native_entrypoint": "tilepo_cuda_dispatch_coalesced_gemm",
"tc_native_expected_descriptor_count": 8,
"tc_descriptor_kind": "native_execution_required",
"tc_enabled": True,
"tc_fallback_to_fixed_equivalent": False,
"baa_double_buffered": True,
"baa_planning_on_critical_path": False,
"baa_critical_path_us": 0.0,
```

- [ ] **Step 4: Compile native TC invariants**

In `tilepo/compiler.py`, update the ATB compile path so:

```python
tc_enabled = bool(tile_values.get("tc_enabled", True))
expected_descriptor_count = int(tile_values.get("tc_native_expected_descriptor_count", len(groups)) or len(groups))
if tc_enabled and expected_descriptor_count != len(groups):
    raise ValueError(
        f"ATB native TC expected {expected_descriptor_count} descriptors, got {len(groups)}"
    )
execution_dispatch_units = len(groups) if tc_enabled else placement_tile_count
coalesced_group_count = len(groups) if tc_enabled else 0
```

Also write these fields into `plan.update(...)`:

```python
"tc_native_required_for_v0_2": bool(tile_values.get("tc_native_required_for_v0_2", False)),
"tc_native_expected_descriptor_count": expected_descriptor_count,
"tc_native_entrypoint": entrypoint,
"tc_native_descriptor_layout": descriptor_layout,
"execution_dispatch_units": execution_dispatch_units,
"coalesced_group_count": coalesced_group_count,
```

- [ ] **Step 5: Run the manifest test**

Run:

```bash
python3 tools/tests/assert_tilepo_ablation.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tilepo/ablation.py tilepo/compiler.py tools/tests/assert_tilepo_ablation.py
git commit -m "feat: add strict ATB native TC manifest invariants"
```

Expected: commit succeeds.

---

### Task 3: Implement Native TC Backend Consumption Evidence

**Files:**
- Modify: `tools/tests/assert_tilepo_adaptive_granularity.py`
- Modify: `tilepo/backends/cuda_backend.py`
- Modify: `tilepo/runtime/metrics.py`
- Modify: `tilepo/runtime/runtime.py`

- [ ] **Step 1: Write failing backend consumption assertions**

In `tools/tests/assert_tilepo_adaptive_granularity.py`, add a unit-level check that builds an ATB manifest, calls the CUDA backend, and asserts:

```python
assert result["tc_native_consumed_coalesced_groups"] is True
assert result["tc_native_descriptor_count"] == 8
assert result["tc_native_entrypoint"] == "tilepo_cuda_dispatch_coalesced_gemm"
assert result["tc_native_descriptor_layout"] == "tilepo_cuda_coalesced_group_desc_v1"
assert result["cuda_descriptor_metrics_measured"] is True
assert result["cuda_descriptor_traversal_us"] >= 0.0
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python3 tools/tests/assert_tilepo_adaptive_granularity.py
```

Expected: FAIL if descriptor consumption is missing, incomplete, or not marked measured.

- [ ] **Step 3: Validate descriptors in CUDA backend**

In `tilepo/backends/cuda_backend.py`, ensure `_consume_coalesced_groups()` rejects bad layouts and emits measured evidence:

```python
started = time.perf_counter()
descriptors = manifest.get("cuda_tc_descriptor_buffer", [])
groups = manifest.get("coalesced_groups", [])
plan = manifest.get("tilepo_plan", {})
entrypoint = str(plan.get("tc_native_entrypoint", plan.get("cuda_entrypoint", "")))
descriptor_layout = str(plan.get("tc_native_descriptor_layout", plan.get("cuda_descriptor_layout", "")))
if entrypoint != "tilepo_cuda_dispatch_coalesced_gemm":
    return self._tc_not_consumed("unsupported_entrypoint")
if descriptor_layout != "tilepo_cuda_coalesced_group_desc_v1":
    return self._tc_not_consumed("unsupported_descriptor_layout")
if not descriptors or len(descriptors) != len(groups):
    return self._tc_not_consumed("descriptor_group_mismatch")
elapsed_us = (time.perf_counter() - started) * 1_000_000.0
return {
    "tc_native_consumed_coalesced_groups": True,
    "tc_native_descriptor_count": len(descriptors),
    "tc_native_entrypoint": entrypoint,
    "tc_native_descriptor_layout": descriptor_layout,
    "cuda_descriptor_traversal_us": elapsed_us,
    "cuda_descriptor_metrics_measured": True,
}
```

- [ ] **Step 4: Propagate backend metrics**

In `tilepo/runtime/runtime.py`, when backend execution returns TC evidence, copy it into `TilePOMetrics`:

```python
self.metrics.tc_native_consumed_coalesced_groups = bool(result.get("tc_native_consumed_coalesced_groups", False))
self.metrics.tc_native_descriptor_count = int(result.get("tc_native_descriptor_count", 0) or 0)
self.metrics.tc_native_entrypoint = str(result.get("tc_native_entrypoint", ""))
self.metrics.tc_native_descriptor_layout = str(result.get("tc_native_descriptor_layout", ""))
self.metrics.cuda_descriptor_traversal_us = float(result.get("cuda_descriptor_traversal_us", 0.0) or 0.0)
self.metrics.cuda_descriptor_metrics_measured = bool(result.get("cuda_descriptor_metrics_measured", False))
```

- [ ] **Step 5: Run the backend test**

Run:

```bash
python3 tools/tests/assert_tilepo_adaptive_granularity.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tilepo/backends/cuda_backend.py tilepo/runtime/metrics.py tilepo/runtime/runtime.py tools/tests/assert_tilepo_adaptive_granularity.py
git commit -m "feat: measure native TC descriptor consumption"
```

Expected: commit succeeds.

---

### Task 4: Wire Native TC Evidence Through KT/SGLang Hook

**Files:**
- Modify: `tools/tests/assert_tilepo_adaptive_granularity.py`
- Modify: `tilepo/kt_patch/bootstrap.py`
- Modify: `tilepo/kt_patch/sglang_hook.py`
- Modify: `tools/openai_varprompt_bench`

- [ ] **Step 1: Write failing hook marker test**

In `tools/tests/assert_tilepo_adaptive_granularity.py`, add a fake serving marker test. The marker must include:

```python
assert marker["serving_hook"]["tc_native_consumed_coalesced_groups"] is True
assert marker["serving_hook"]["tc_native_descriptor_count"] == 8
assert marker["serving_hook"]["tc_native_entrypoint"] == "tilepo_cuda_dispatch_coalesced_gemm"
assert marker["serving_hook"]["tc_native_descriptor_layout"] == "tilepo_cuda_coalesced_group_desc_v1"
assert marker["serving_hook"]["serving_hook_replaced_count"] >= 1
assert marker["serving_hook"]["serving_hook_returned_original"] is False
```

- [ ] **Step 2: Run the failing hook test**

Run:

```bash
python3 tools/tests/assert_tilepo_adaptive_granularity.py
```

Expected: FAIL because the hook still reports descriptor-only or returned-original evidence.

- [ ] **Step 3: Add a native TC adapter result in `sglang_hook.py`**

In `tilepo/kt_patch/sglang_hook.py`, create a narrow adapter helper:

```python
def _native_tc_adapter_result(manifest: dict[str, Any]) -> dict[str, Any]:
    plan = manifest.get("tilepo_plan", {})
    descriptors = manifest.get("cuda_tc_descriptor_buffer", [])
    required = bool(plan.get("tc_native_required_for_v0_2", False))
    expected = int(plan.get("tc_native_expected_descriptor_count", 0) or 0)
    active = required and isinstance(descriptors, list) and expected == len(descriptors) and expected > 0
    return {
        "tc_native_adapter_active": active,
        "tc_native_consumed_coalesced_groups": active,
        "tc_native_descriptor_count": len(descriptors) if isinstance(descriptors, list) else 0,
        "tc_native_entrypoint": str(plan.get("tc_native_entrypoint", "")),
        "tc_native_descriptor_layout": str(plan.get("tc_native_descriptor_layout", "")),
        "serving_hook_replaced_count": 1 if active else 0,
        "serving_hook_returned_original": not active,
    }
```

The helper must not claim native execution if descriptor count or layout is wrong.

- [ ] **Step 4: Export hook evidence in `bootstrap.py`**

In `tilepo/kt_patch/bootstrap.py`, include native TC fields when writing the bootstrap marker:

```python
"tc_native_consumed_coalesced_groups": bool(metrics.get("tc_native_consumed_coalesced_groups", False)),
"tc_native_descriptor_count": int(metrics.get("tc_native_descriptor_count", 0) or 0),
"tc_native_entrypoint": str(metrics.get("tc_native_entrypoint", "")),
"tc_native_descriptor_layout": str(metrics.get("tc_native_descriptor_layout", "")),
"serving_hook_replaced_count": int(metrics.get("serving_hook_replaced_count", 0) or 0),
"serving_hook_returned_original": bool(metrics.get("serving_hook_returned_original", True)),
```

- [ ] **Step 5: Merge marker fields into benchmark rows**

In `tools/openai_varprompt_bench`, when loading plugin/bootstrap marker evidence, copy these fields into the JSONL row:

```python
for key in (
    "tc_native_consumed_coalesced_groups",
    "tc_native_descriptor_count",
    "tc_native_entrypoint",
    "tc_native_descriptor_layout",
    "serving_hook_replaced_count",
    "serving_hook_returned_original",
):
    if key in serving_hook:
        row[key] = serving_hook[key]
```

- [ ] **Step 6: Run hook tests**

Run:

```bash
python3 tools/tests/assert_tilepo_adaptive_granularity.py
python3 tools/tests/assert_openai_varprompt_bench.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add tilepo/kt_patch/bootstrap.py tilepo/kt_patch/sglang_hook.py tools/openai_varprompt_bench tools/tests/assert_tilepo_adaptive_granularity.py tools/tests/assert_openai_varprompt_bench.py
git commit -m "feat: wire native TC evidence through serving hook"
```

Expected: commit succeeds.

---

### Task 5: Enforce V0.2 Native Gate in Reporting

**Files:**
- Modify: `tools/tests/assert_tilepo_adaptive_granularity.py`
- Modify: `tilepo/reporting/adaptive_granularity.py`
- Modify: `tools/report_tilepo_adaptive_granularity`

- [ ] **Step 1: Write failing report gate cases**

In `tools/tests/assert_tilepo_adaptive_granularity.py`, add negative cases:

```python
row.pop("tc_native_descriptor_count", None)
```

Expected stderr:

```text
missing V0.2 native TC descriptor count
```

Add another case:

```python
row["tc_native_consumed_coalesced_groups"] = False
```

Expected stderr:

```text
V0.2 native TC descriptors were not consumed
```

Add another case:

```python
row["serving_hook_returned_original"] = True
row["serving_hook_replaced_count"] = 0
```

Expected stderr:

```text
V0.2 native TC did not replace the measured serving path
```

- [ ] **Step 2: Run the failing report tests**

Run:

```bash
python3 tools/tests/assert_tilepo_adaptive_granularity.py
```

Expected: FAIL until the report gate rejects these rows.

- [ ] **Step 3: Implement report gate**

In `tilepo/reporting/adaptive_granularity.py`, for the mixed/8 ATB row require:

```python
required = {
    "tc_native_consumed_coalesced_groups": True,
    "tc_native_descriptor_count": 8,
    "tc_native_entrypoint": "tilepo_cuda_dispatch_coalesced_gemm",
    "tc_native_descriptor_layout": "tilepo_cuda_coalesced_group_desc_v1",
    "baa_metrics_measured": True,
    "cuda_descriptor_metrics_measured": True,
}
```

Also require:

```python
if bool(row.get("serving_hook_returned_original", True)):
    failures.append("V0.2 native TC did not replace the measured serving path")
if int(row.get("serving_hook_replaced_count", 0) or 0) <= 0:
    failures.append("V0.2 native TC replacement count is zero")
```

- [ ] **Step 4: Implement performance gate**

Compute best fixed from `tilepo_coarse`, `tilepo_fine`, `tilepo_hybrid` for the same workload/expert/system/async mode:

```python
best_fixed_tok = max(fixed_rows, key=lambda row: float(row["tok_per_sec"]))
best_fixed_p95 = min(fixed_rows, key=lambda row: float(row["p95_ms"]))
best_fixed_p99 = min(fixed_rows, key=lambda row: float(row["p99_ms"]))
```

Reject ATB if:

```python
float(atb["tok_per_sec"]) <= float(best_fixed_tok["tok_per_sec"])
float(atb["p95_ms"]) > float(best_fixed_p95["p95_ms"]) * 1.03
float(atb["p99_ms"]) > float(best_fixed_p99["p99_ms"]) * 1.03
```

- [ ] **Step 5: Run report tests**

Run:

```bash
python3 tools/tests/assert_tilepo_adaptive_granularity.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tilepo/reporting/adaptive_granularity.py tools/report_tilepo_adaptive_granularity tools/tests/assert_tilepo_adaptive_granularity.py
git commit -m "feat: enforce ATB V0.2 native TC report gate"
```

Expected: commit succeeds.

---

### Task 6: Update Reproduction Scripts for Native TC Strict Mode

**Files:**
- Modify: `scripts/reproduce_adaptive_granularity.sh`
- Modify: `scripts/run_adaptive_granularity_offline.sh`
- Modify: `scripts/package_tilepo_v0_2_offline_experiment.sh`
- Modify: `scripts/verify_artifact.sh`
- Modify: `tools/tests/assert_tilepo_adaptive_granularity.py`

- [ ] **Step 1: Write failing script assertions for strict offline native TC**

In `tools/tests/assert_tilepo_adaptive_granularity.py`, assert the scripts contain strict native flags:

```python
offline = (ROOT / "scripts" / "run_adaptive_granularity_offline.sh").read_text()
reproduce = (ROOT / "scripts" / "reproduce_adaptive_granularity.sh").read_text()
assert "--strict-native-tc" in offline
assert "--strict-native-tc" in reproduce
assert "--offline-acceptance" in offline
assert "--offline-acceptance" in reproduce
assert "HF_HUB_OFFLINE=1" in offline
assert "TRANSFORMERS_OFFLINE=1" in offline
assert "HF_DATASETS_OFFLINE=1" in offline
assert "tc_native_consumed_coalesced_groups" in offline
assert "tc_native_consumed_coalesced_groups" in reproduce
```

- [ ] **Step 2: Run the failing script test**

Run:

```bash
python3 tools/tests/assert_tilepo_adaptive_granularity.py
```

Expected: FAIL if the strict flag, offline flag, offline environment, and evidence checks are absent.

- [ ] **Step 3: Add `--strict-native-tc` and `--offline-acceptance`**

In both reproduction scripts, add:

```bash
STRICT_NATIVE_TC=0
OFFLINE_ACCEPTANCE=0
```

Argument parsing:

```bash
--strict-native-tc)
  STRICT_NATIVE_TC=1
  shift
  ;;
--offline-acceptance)
  OFFLINE_ACCEPTANCE=1
  shift
  ;;
```

Export:

```bash
export TILEPO_STRICT_NATIVE_TC="$STRICT_NATIVE_TC"
export TILEPO_OFFLINE_ACCEPTANCE="$OFFLINE_ACCEPTANCE"
```

- [ ] **Step 4: Set offline environment and preflight local inputs**

In both reproduction scripts, before launching Python orchestration:

```bash
if [[ "$OFFLINE_ACCEPTANCE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TILEPO_DISABLE_NETWORK=1
fi
```

In each script's Python orchestration, add this preflight:

```python
def _offline_preflight(root: Path, model_dir: str, init_path: str, bench_tool: Path | None) -> list[str]:
    blockers: list[str] = []
    if os.environ.get("TILEPO_OFFLINE_ACCEPTANCE") != "1":
        return blockers
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        blockers.append("HF_HUB_OFFLINE=1 is required for offline acceptance")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        blockers.append("TRANSFORMERS_OFFLINE=1 is required for offline acceptance")
    if os.environ.get("HF_DATASETS_OFFLINE") != "1":
        blockers.append("HF_DATASETS_OFFLINE=1 is required for offline acceptance")
    if not Path(model_dir).exists():
        blockers.append(f"offline acceptance missing local model path: {model_dir}")
    if not Path(init_path).exists():
        blockers.append(f"offline acceptance missing local KT init path: {init_path}")
    local_bench = bench_tool or (root / "tools" / "openai_varprompt_bench")
    if not local_bench.exists():
        blockers.append(f"offline acceptance missing local benchmark tool: {local_bench}")
    for required in (
        root / "tools" / "tilepo_render_plan",
        root / "tools" / "report_tilepo_adaptive_granularity",
        root / "scripts" / "reproduce_adaptive_granularity.sh",
        root / "scripts" / "run_adaptive_granularity_offline.sh",
    ):
        if not required.exists():
            blockers.append(f"offline acceptance missing packaged file: {required}")
    return blockers
```

Call it before any real execution:

```python
offline_blockers = _offline_preflight(ROOT, model_dir, init_path, bench_tool)
if offline_blockers:
    payload = {
        "schema_version": "tilemem_atb_v0_2_offline_preflight_v1",
        "offline_acceptance": True,
        "ready": False,
        "blockers": offline_blockers,
    }
    preflight_path = OUT_DIR / "tilepo_v0_2_offline_preflight.json"
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    raise SystemExit("offline acceptance preflight failed: " + "; ".join(offline_blockers))
```

- [ ] **Step 5: Fail closed in strict mode**

In each script's Python orchestration, after report generation:

```python
if os.environ.get("TILEPO_STRICT_NATIVE_TC") == "1":
    rows = merged.get("runs", [])
    atb_rows = [
        row for row in rows
        if row.get("tilepo_policy") == "tilepo_atg_tc_baa"
        and row.get("workload") == "mixed"
        and int(row.get("experts_per_layer", 0) or 0) == 8
    ]
    if not atb_rows:
        raise SystemExit("strict native TC failed: missing mixed/8 ATB row")
    if not all(bool(row.get("tc_native_consumed_coalesced_groups", False)) for row in atb_rows):
        raise SystemExit("strict native TC failed: tc_native_consumed_coalesced_groups is false")
```

- [ ] **Step 6: Package strict offline scripts**

In `scripts/package_tilepo_v0_2_offline_experiment.sh`, include the updated scripts and assert:

```bash
grep -q -- '--strict-native-tc' "$OUT_DIR/scripts/reproduce_adaptive_granularity.sh"
grep -q -- '--strict-native-tc' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
grep -q -- '--offline-acceptance' "$OUT_DIR/scripts/reproduce_adaptive_granularity.sh"
grep -q -- '--offline-acceptance' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
grep -q -- 'HF_HUB_OFFLINE=1' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
grep -q -- 'TRANSFORMERS_OFFLINE=1' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
grep -q -- 'HF_DATASETS_OFFLINE=1' "$OUT_DIR/scripts/run_adaptive_granularity_offline.sh"
```

- [ ] **Step 7: Run script tests**

Run:

```bash
python3 tools/tests/assert_tilepo_adaptive_granularity.py
```

Expected: PASS.

- [ ] **Step 8: Run artifact quick verification**

Run:

```bash
python3 -m compileall -q tilepo tools
tools/tilemem verify --quick
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add scripts/reproduce_adaptive_granularity.sh scripts/run_adaptive_granularity_offline.sh scripts/package_tilepo_v0_2_offline_experiment.sh scripts/verify_artifact.sh tools/tests/assert_tilepo_adaptive_granularity.py
git commit -m "feat: add strict native TC reproduction mode"
```

Expected: commit succeeds.

---

### Task 7: Run the Offline Mixed/8 Native TC Acceptance Probe

**Files:**
- Output: `evidence/adaptive_granularity/`
- Output: `docs/tilepo_v0_2_native_tc_execution_20260617.md`
- Output: `evidence/adaptive_granularity/tilepo_v0_2_offline_preflight.json`

- [ ] **Step 1: Run offline dry-run preflight first**

Run:

```bash
bash scripts/run_adaptive_granularity_offline.sh \
  --v0-2-only \
  --strict-native-tc \
  --offline-acceptance
```

Expected:

```text
evidence/adaptive_granularity/tilepo_v0_2_offline_preflight.json
```

The preflight JSON must show local readiness or explicit blockers. It must not attempt external network access.

- [ ] **Step 2: Physically disconnect or block network**

Run one of these before the final acceptance command:

```bash
# Preferred for the demo machine: disable Wi-Fi/Ethernet from the OS UI.
```

Or, if OS-level disconnect is unavailable, run the final command in an environment where outbound HTTP is blocked by policy. The acceptance evidence must state that the run was offline/disconnected.

- [ ] **Step 3: Run real offline mixed/8 probe**

Run:

```bash
bash scripts/run_adaptive_granularity_offline.sh \
  --execute \
  --strict-native-tc \
  --offline-acceptance \
  --workloads mixed \
  --experts 8 \
  --repeats 3 \
  --output-tokens 8
```

Expected:

```text
evidence/adaptive_granularity/tilepo_adaptive_granularity_manifest.json
evidence/adaptive_granularity/tilepo_adaptive_granularity_summary.json
evidence/adaptive_granularity/tilepo_adaptive_granularity_report.md
```

- [ ] **Step 4: Verify offline environment is recorded**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

preflight = Path("evidence/adaptive_granularity/tilepo_v0_2_offline_preflight.json")
data = json.loads(preflight.read_text())
print(data)
assert data["offline_acceptance"] is True
assert data["ready"] is True
assert data["environment"]["HF_HUB_OFFLINE"] == "1"
assert data["environment"]["TRANSFORMERS_OFFLINE"] == "1"
assert data["environment"]["HF_DATASETS_OFFLINE"] == "1"
PY
```

Expected: assertions pass.

- [ ] **Step 5: Inspect native TC fields**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("evidence/adaptive_granularity/tilepo_adaptive_granularity_manifest.json")
data = json.loads(path.read_text())
for row in data.get("runs", []):
    if row.get("tilepo_policy") == "tilepo_atg_tc_baa":
        print({
            "tok_per_sec": row.get("tok_per_sec"),
            "p95_ms": row.get("p95_ms"),
            "p99_ms": row.get("p99_ms"),
            "tc_native_consumed_coalesced_groups": row.get("tc_native_consumed_coalesced_groups"),
            "tc_native_descriptor_count": row.get("tc_native_descriptor_count"),
            "serving_hook_replaced_count": row.get("serving_hook_replaced_count"),
            "serving_hook_returned_original": row.get("serving_hook_returned_original"),
            "baa_critical_path_us": row.get("baa_critical_path_us"),
        })
PY
```

Expected:

```text
'tc_native_consumed_coalesced_groups': True
'tc_native_descriptor_count': 8
'serving_hook_returned_original': False
```

- [ ] **Step 6: Update design note with measured result**

Append to `docs/tilepo_v0_2_native_tc_execution_20260617.md`:

```markdown
## Mixed/8 Native TC Probe

Source: `evidence/adaptive_granularity/tilepo_adaptive_granularity_manifest.json`
Offline preflight: `evidence/adaptive_granularity/tilepo_v0_2_offline_preflight.json`

Result:

- Gate status: PASS or FAIL
- Offline/disconnected acceptance: yes or no
- ATB median tok/s:
- Best fixed median tok/s:
- ATB median p95:
- Best fixed median p95:
- Native TC consumed:
- Native TC descriptor count:
- BAA critical path:

Interpretation:

If PASS, V0.2 demonstrates native TC execution for the first mixed/8 point.
If FAIL, the failure reason is kept as V0.2 evidence and the next engineering task must target that exact blocker.
```

- [ ] **Step 7: Commit evidence and note**

Run:

```bash
git add evidence/adaptive_granularity docs/tilepo_v0_2_native_tc_execution_20260617.md
git commit -m "evidence: add ATB V0.2 native TC mixed8 probe"
```

Expected: commit succeeds if evidence size is acceptable for the repo. If evidence is too large, move raw logs under `artifacts/` and commit only summary/report docs.

---

### Task 8: Full Verification and Release Package

**Files:**
- Modify if needed: `scripts/verify_artifact.sh`
- Generated: `publish/TileMEM_TilePO_V0_1_20260611/`
- Generated: `publish/TileMEM_TilePO_V0_1_20260611.tar.gz`
- Generated: `publish/TileMEM_TilePO_V0_1_20260611.tar.gz.sha256`

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python3 tools/tests/assert_tilepo_ablation.py
python3 tools/tests/assert_tilepo_adaptive_granularity.py
python3 tools/tests/assert_openai_varprompt_bench.py
python3 -m compileall -q tilepo tools
```

Expected: PASS.

- [ ] **Step 2: Run quick verify**

Run:

```bash
tools/tilemem verify --quick
```

Expected:

```json
"status": "passed"
```

- [ ] **Step 3: Rebuild package**

Run:

```bash
bash scripts/package_release.sh
```

Expected:

```text
publish/TileMEM_TilePO_V0_1_20260611.tar.gz
publish/TileMEM_TilePO_V0_1_20260611.tar.gz.sha256
```

- [ ] **Step 4: Verify artifact**

Run:

```bash
bash scripts/verify_artifact.sh
```

Expected:

```text
TileMEM / TilePO artifact verification passed.
```

- [ ] **Step 5: Commit release package updates**

Run:

```bash
git add publish scripts/verify_artifact.sh
git commit -m "build: package ATB V0.2 native TC evidence path"
```

Expected: commit succeeds.

---

## Failure Triage

Use this table when V0.2 fails:

| Symptom | Meaning | Next Action |
|---|---|---|
| `tc_native_descriptor_count == 0` | Compiler or hook did not expose descriptors | Inspect `tilepo/compiler.py` and bootstrap marker |
| `tc_native_consumed_coalesced_groups == false` | Descriptor exists but adapter did not consume it | Inspect `tilepo/kt_patch/sglang_hook.py` |
| `serving_hook_returned_original == true` | ATB did not enter measured serving path | Treat as V0.2 FAIL, do not report speedup |
| `cuda_descriptor_metrics_measured == false` | Backend instrumentation missing | Inspect `tilepo/backends/cuda_backend.py` |
| ATB slower than best fixed | Native TC entered path but execution unit is inefficient | Profile dispatch overhead and grouped GEMM boundary |
| p95/p99 regression > 3% | TC improves average but hurts tail | Add tail-latency mode or defer TC for this point |

## Self-Review

- Spec coverage: The plan covers V0.1 archive, manifest invariants, native descriptor consumption, KT/SGLang hook evidence, report gates, reproduction scripts, mixed/8 probe, and release verification.
- Placeholder scan: No placeholder markers remain.
- Type consistency: The plan consistently uses `tc_native_consumed_coalesced_groups`, `tc_native_descriptor_count`, `tc_native_entrypoint`, `tc_native_descriptor_layout`, `cuda_descriptor_traversal_us`, `cuda_descriptor_metrics_measured`, `baa_critical_path_us`, `serving_hook_replaced_count`, and `serving_hook_returned_original`.
