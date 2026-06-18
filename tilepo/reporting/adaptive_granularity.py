from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


WORKLOADS = ("mixed", "long_context")
EXPERTS = (6, 8, 10)
FIXED_POLICIES = ("tilepo_coarse", "tilepo_fine", "tilepo_hybrid")
V2_POLICY = "tilepo_atg_tc_baa"
LEGACY_ADAPTIVE_POLICIES = {"tilepo_adaptive", "tilepo_atg"}
POLICIES = (*FIXED_POLICIES, V2_POLICY)
EXPECTED_ROWS = len(WORKLOADS) * len(EXPERTS) * len(POLICIES)
V2_ONLY_POLICIES = (V2_POLICY,)
V2_ONLY_EXPECTED_ROWS = len(WORKLOADS) * len(EXPERTS) * len(V2_ONLY_POLICIES)
CORE_METRICS = ("tok_per_sec", "p95_ms", "p99_ms", "gpu_peak_gib", "cpu_ram_peak_gib")
OPTIONAL_METRICS = (
    "tile_count",
    "execution_dispatch_units",
    "estimated_dispatch_units",
    "plan_lookup_us",
    "baa_critical_path_us",
    "cuda_launch_count",
    "cuda_descriptor_traversal_us",
    "tc_native_consumed_group_count",
    "tc_native_descriptor_count",
    "tc_native_consumed_tile_count",
    "tc_native_consumed_bytes",
    "tc_native_launch_count",
    "tc_adapter_group_count",
    "tc_adapter_descriptor_count",
    "tc_adapter_tile_count",
    "tc_adapter_dispatch_units",
)
METRICS = (*CORE_METRICS, *OPTIONAL_METRICS)


class AdaptiveGranularityReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdaptiveGranularityReportResult:
    summary_path: Path
    markdown_path: Path
    summary: dict[str, Any]


def _matrix_workloads(matrix: dict[str, Any]) -> tuple[str, ...]:
    values = matrix.get("workloads")
    if isinstance(values, list) and values:
        return tuple(str(item) for item in values)
    return WORKLOADS


def _matrix_experts(matrix: dict[str, Any]) -> tuple[int, ...]:
    values = matrix.get("experts")
    if isinstance(values, list) and values:
        return tuple(int(item) for item in values)
    return EXPERTS


def _matrix_policies(matrix: dict[str, Any], *, v0_2_only: bool) -> tuple[str, ...]:
    values = matrix.get("policies")
    if isinstance(values, list) and values:
        return tuple(_normalize_policy(str(item)) for item in values)
    return V2_ONLY_POLICIES if v0_2_only else POLICIES


def generate_adaptive_granularity_report(
    manifest_path: Path | str,
    out_dir: Path | str,
    *,
    require_real: bool = False,
    strict_v0_2_win: bool = True,
    v0_2_only: bool = False,
) -> AdaptiveGranularityReportResult:
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    data = json.loads(manifest_path.read_text())
    rows = [row for row in data.get("runs", []) if isinstance(row, dict)]
    failures: list[str] = []
    warnings: list[str] = []
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    matrix = data.get("matrix", {}) if isinstance(data.get("matrix"), dict) else {}
    workloads = _matrix_workloads(matrix)
    experts = _matrix_experts(matrix)
    policies = _matrix_policies(matrix, v0_2_only=v0_2_only)
    repeats = int(matrix.get("repeats", 1) or 1)
    expected_rows = len(workloads) * len(experts) * len(policies) * repeats
    comparison_name = str(
        matrix.get("comparison")
        or ("tilepo_v0_2_only" if v0_2_only else "tilepo_self_ablation")
    )
    output_tokens = matrix.get("output_tokens")

    if data.get("blocked") is True:
        failures.append("source manifest is blocked")
    if data.get("expected_result_rows") is not None and int(data.get("expected_result_rows", 0)) != expected_rows:
        failures.append(
            f"source manifest expected_result_rows is not {expected_rows}: {data.get('expected_result_rows')}"
        )
    if data.get("actual_result_rows") is not None and int(data.get("actual_result_rows", 0)) != len(rows):
        failures.append(
            f"source manifest actual_result_rows does not match runs length: {data.get('actual_result_rows')} != {len(rows)}"
        )
    if len(rows) != expected_rows:
        failures.append(f"source manifest contains {len(rows)} rows, expected exactly {expected_rows}")

    for index, row in enumerate(rows):
        failures.extend(
            _validate_row(
                index,
                row,
                manifest_path=manifest_path,
                require_real=require_real,
                allowed_workloads=workloads,
                allowed_experts=experts,
                allowed_policies=policies,
                v0_2_only=v0_2_only,
            )
        )
        workload = str(row.get("workload", ""))
        expert = _as_int(row.get("experts_per_layer"))
        policy = _normalize_policy(_row_policy(row))
        async_mode = _row_async(row)
        system = str(row.get("system", ""))
        if workload in workloads and expert in experts and policy in policies:
            grouped[(workload, expert, policy, async_mode, system)].append(row)

    groups = [_group_record(key, value) for key, value in sorted(grouped.items())]
    failures.extend(_coverage_failures(grouped, workloads, experts, policies))
    comparisons = []
    if not v0_2_only:
        for workload in workloads:
            for expert in EXPERTS:
                if expert not in experts:
                    continue
                comparison = _comparison(grouped, workload, expert)
                if comparison:
                    comparisons.append(comparison)
                    _gate_comparison(comparison, failures, warnings)

    summary = {
        "schema_version": "tilepo_adaptive_granularity_report_v1",
        "source_manifest": str(manifest_path),
        "requested": {
            "workloads": list(workloads),
            "experts": list(experts),
            "policies": list(policies),
            "expected_rows": expected_rows,
            "repeats": repeats,
            "require_real": require_real,
            "strict_v0_2_win": strict_v0_2_win,
            "serving_shell": "kt_sglang",
            "comparison": comparison_name,
            "v0_2_only": v0_2_only,
            "output_tokens": output_tokens,
        },
        "gate": {"status": "PASS" if not failures else "FAIL", "failures": failures, "warnings": warnings},
        "groups": groups,
        "comparisons": comparisons,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tilepo_adaptive_granularity_summary.json"
    markdown_path = out_dir / "tilepo_adaptive_granularity_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown_v0_2_only(summary) if v0_2_only else _markdown(summary))
    if failures:
        raise AdaptiveGranularityReportError("; ".join(failures))
    return AdaptiveGranularityReportResult(summary_path, markdown_path, summary)


def _validate_row(
    index: int,
    row: dict[str, Any],
    *,
    manifest_path: Path,
    require_real: bool,
    allowed_workloads: tuple[str, ...],
    allowed_experts: tuple[int, ...],
    allowed_policies: tuple[str, ...],
    v0_2_only: bool,
) -> list[str]:
    failures: list[str] = []
    for key in ("system", "workload", "experts_per_layer", "repeat", "tok_per_sec", "p95_ms", "p99_ms"):
        if key not in row:
            failures.append(f"row {index} missing {key}")
    for metric in ("gpu_peak_gib", "cpu_ram_peak_gib"):
        if metric not in row:
            failures.append(f"row {index} missing {metric}")
    for metric in METRICS:
        if row.get(metric) is None:
            continue
        try:
            value = _to_float(row.get(metric))
        except (TypeError, ValueError):
            failures.append(f"row {index} has non-numeric {metric}: {row.get(metric)!r}")
            continue
        if not math.isfinite(value):
            failures.append(f"row {index} has non-finite {metric}: {row.get(metric)!r}")
    if not _row_policy(row):
        failures.append(f"row {index} missing tilepo_policy/ablation_policy")
    if not _row_async(row):
        failures.append(f"row {index} missing tilepo_async_planning/async_planning_mode")
    workload = str(row.get("workload", ""))
    expert = _as_int(row.get("experts_per_layer"))
    policy = _normalize_policy(_row_policy(row))
    async_mode = _row_async(row)
    system = str(row.get("system", ""))
    if workload or expert != -1 or policy or async_mode or system:
        if not (
            workload in allowed_workloads
            and expert in allowed_experts
            and policy in allowed_policies
            and async_mode == "on"
            and system == "C"
        ):
            failures.append(
                "row {index} unexpected non-matrix row: "
                "expected workload in {workloads}, experts in {experts}, policy in {policies}, async=on, system=C; "
                "got ({workload!r}, {expert}, {policy!r}, {async_mode!r}, {system!r})".format(
                    index=index,
                    workloads=list(allowed_workloads),
                    experts=list(allowed_experts),
                    policies=list(allowed_policies),
                    workload=workload,
                    expert=expert,
                    policy=policy,
                    async_mode=async_mode,
                    system=system,
                )
            )
    if require_real:
        if row.get("simulated") is not False or row.get("evidence_level") != "real":
            failures.append(f"row {index} is not real evidence")
        if row.get("status") != "success":
            failures.append(f"row {index} is not success: {row.get('status')}")
        if policy in POLICIES and (policy == V2_POLICY or not v0_2_only):
            if _required_v2_float(row, "cuda_launch_count") is None:
                failures.append(f"row {index} missing CUDA launch metric")
            if _required_v2_float(row, "cuda_descriptor_traversal_us") is None:
                failures.append(f"row {index} missing CUDA descriptor traversal metric")
        if policy == V2_POLICY:
            failures.extend(_v2_fallback_counter_failures(index, row))
            backend = _v2_backend(row)
            if backend is None:
                failures.append(f"row {index} missing explicit V0.2 TilePO augmentation backend evidence")
                backend = ""
            if "cuda" not in backend:
                failures.append(f"row {index} V0.2 augmentation backend is not CUDA-based: {backend}")
            failures.extend(_v2_kt_preserving_serving_failures(index, row))
            if _required_v2_float(row, "baa_critical_path_us") is None:
                failures.append(f"row {index} missing V0.2 BAA critical-path metric")
            if _required_v2_float(row, "cuda_launch_count") is None:
                failures.append(f"row {index} missing V0.2 CUDA launch metric")
            if _required_v2_float(row, "cuda_descriptor_traversal_us") is None:
                failures.append(f"row {index} missing V0.2 CUDA descriptor traversal metric")
            failures.extend(_v2_runtime_metric_provenance_failures(index, row))
        raw_path = row.get("raw_path")
        if not raw_path:
            failures.append(f"row {index} missing raw_path")
        else:
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            if not candidate.exists():
                failures.append(f"row {index} raw_path does not exist: {raw_path}")
    return failures


def _coverage_failures(
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]],
    workloads: tuple[str, ...],
    experts: tuple[int, ...],
    policies: tuple[str, ...],
) -> list[str]:
    failures: list[str] = []
    for workload in workloads:
        for expert in experts:
            for policy in policies:
                key = (workload, expert, policy, "on", "C")
                if len(grouped.get(key, [])) < 1:
                    failures.append(f"missing TilePO row for {key}")
    return failures


def _group_record(key: tuple[str, int, str, str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    workload, expert, policy, async_mode, system = key
    return {
        "workload": workload,
        "experts_per_layer": expert,
        "policy": policy,
        "async_planning": async_mode,
        "system": system,
        "repeats": len(rows),
        "metrics": {metric: _stats(_metric_values(rows, metric)) for metric in METRICS},
    }


def _comparison(
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]],
    workload: str,
    expert: int,
) -> dict[str, Any] | None:
    v2_rows = grouped.get((workload, expert, V2_POLICY, "on", "C"), [])
    fixed = {
        policy: _aggregate(grouped.get((workload, expert, policy, "on", "C"), []))
        for policy in FIXED_POLICIES
    }
    if not v2_rows or any(value is None for value in fixed.values()):
        return None
    v2 = _aggregate(v2_rows)
    if v2 is None:
        return None
    best_fixed_policy, best_fixed = max(fixed.items(), key=lambda item: item[1]["tok_per_sec"])
    best_fixed_p95_policy, best_fixed_p95 = min(fixed.items(), key=lambda item: item[1]["p95_ms"])
    best_fixed_p99_policy, best_fixed_p99 = min(fixed.items(), key=lambda item: item[1]["p99_ms"])
    coarse = fixed["tilepo_coarse"]
    fine = fixed["tilepo_fine"]
    hybrid = fixed["tilepo_hybrid"]
    return {
        "workload": workload,
        "experts_per_layer": expert,
        "best_fixed_policy": best_fixed_policy,
        "strict_per_point_win": bool(v2["tok_per_sec"] > best_fixed["tok_per_sec"]),
        "v2_vs_best_fixed": {
            "tok_gain_pct": _gain_pct(v2["tok_per_sec"], best_fixed["tok_per_sec"]),
            "tok_margin": v2["tok_per_sec"] - best_fixed["tok_per_sec"],
            "p95_delta_pct": _gain_pct(v2["p95_ms"], best_fixed_p95["p95_ms"]),
            "p99_delta_pct": _gain_pct(v2["p99_ms"], best_fixed_p99["p99_ms"]),
            "gpu_peak_delta_pct": _gain_pct(v2["gpu_peak_gib"], best_fixed["gpu_peak_gib"]),
            "cpu_ram_peak_delta_pct": _gain_pct(v2["cpu_ram_peak_gib"], best_fixed["cpu_ram_peak_gib"]),
        },
        "tile_count_comparison": {
            "coarse": _maybe_int(coarse.get("tile_count")),
            "fine": _maybe_int(fine.get("tile_count")),
            "hybrid": _maybe_int(hybrid.get("tile_count")),
            "v2": _maybe_int(v2.get("tile_count")),
            "v2_vs_fine_pct": _ratio_pct(v2.get("tile_count"), fine.get("tile_count")),
            "v2_between_coarse_and_fine": _between(
                coarse.get("tile_count"),
                v2.get("tile_count"),
                fine.get("tile_count"),
            ),
        },
        "dispatch_proxy": {
            "coarse": _maybe_int(coarse.get("estimated_dispatch_units")),
            "fine": _maybe_int(fine.get("estimated_dispatch_units")),
            "hybrid": _maybe_int(hybrid.get("estimated_dispatch_units")),
            "v2": _maybe_int(_dispatch_units(v2)),
            "v2_vs_fine_pct": _ratio_pct(
                _dispatch_units(v2),
                _dispatch_units(fine),
            ),
            "v2_vs_hybrid_pct": _ratio_pct(
                _dispatch_units(v2),
                _dispatch_units(hybrid),
            ),
        },
        "memory_comparison": {
            "v2_gpu_peak_gib": v2["gpu_peak_gib"],
            "v2_cpu_ram_peak_gib": v2["cpu_ram_peak_gib"],
            "best_fixed_gpu_peak_gib": best_fixed["gpu_peak_gib"],
            "best_fixed_cpu_ram_peak_gib": best_fixed["cpu_ram_peak_gib"],
            "best_fixed_p95_policy": best_fixed_p95_policy,
            "best_fixed_p99_policy": best_fixed_p99_policy,
            "best_fixed_p95_ms": best_fixed_p95["p95_ms"],
            "best_fixed_p99_ms": best_fixed_p99["p99_ms"],
            "coarse_gpu_peak_gib": coarse["gpu_peak_gib"],
            "coarse_cpu_ram_peak_gib": coarse["cpu_ram_peak_gib"],
            "fine_gpu_peak_gib": fine["gpu_peak_gib"],
            "fine_cpu_ram_peak_gib": fine["cpu_ram_peak_gib"],
            "hybrid_gpu_peak_gib": hybrid["gpu_peak_gib"],
            "hybrid_cpu_ram_peak_gib": hybrid["cpu_ram_peak_gib"],
        },
        "metrics": {"v2": v2, "best_fixed": best_fixed},
    }


def _gate_comparison(comparison: dict[str, Any], failures: list[str], warnings: list[str]) -> None:
    workload = comparison["workload"]
    expert = comparison["experts_per_layer"]
    v2_vs_best = comparison["v2_vs_best_fixed"]
    if not comparison["strict_per_point_win"]:
        failures.append(f"{workload}/{expert} V0.2 tok/s does not strictly beat best fixed TilePO policy")
    if v2_vs_best["p95_delta_pct"] > 3.0:
        failures.append(f"{workload}/{expert} V0.2 p95 regresses more than 3% vs best fixed")
    if v2_vs_best["p99_delta_pct"] > 3.0:
        failures.append(f"{workload}/{expert} V0.2 p99 regresses more than 3% vs best fixed")
    tile_counts = comparison["tile_count_comparison"]
    if tile_counts["v2"] is None or tile_counts["coarse"] is None or tile_counts["fine"] is None:
        failures.append(f"{workload}/{expert} missing tile_count comparison data")
    elif not tile_counts["v2_between_coarse_and_fine"]:
        warnings.append(f"{workload}/{expert} V0.2 tile count is not between coarse and fine")
    dispatch = comparison["dispatch_proxy"]
    if dispatch["v2_vs_fine_pct"] is None:
        failures.append(f"{workload}/{expert} missing dispatch proxy comparison data")
    elif dispatch["v2_vs_fine_pct"] >= 100.0:
        failures.append(f"{workload}/{expert} V0.2 dispatch proxy is not lower than fine")
    metrics = comparison["metrics"]["v2"]
    if metrics.get("baa_critical_path_us") is None:
        failures.append(f"{workload}/{expert} missing V0.2 BAA critical-path metric")
    elif float(metrics["baa_critical_path_us"]) < 0.0:
        failures.append(f"{workload}/{expert} V0.2 BAA critical-path metric is negative")
    elif float(metrics["baa_critical_path_us"]) > 0.0:
        failures.append(f"{workload}/{expert} V0.2 BAA planning appears on critical path")
    best_fixed_lookup_us = comparison["metrics"]["best_fixed"].get("plan_lookup_us")
    if best_fixed_lookup_us is None:
        failures.append(f"{workload}/{expert} missing best fixed async-on planning metric")
    elif (
        metrics.get("baa_critical_path_us") is not None
        and float(best_fixed_lookup_us) > 0.0
        and float(metrics["baa_critical_path_us"]) >= float(best_fixed_lookup_us)
    ):
        failures.append(f"{workload}/{expert} V0.2 BAA critical-path metric is not lower than best fixed planning")
    if metrics.get("cuda_launch_count") is None:
        failures.append(f"{workload}/{expert} missing V0.2 CUDA launch metric")
    elif float(metrics["cuda_launch_count"]) <= 0.0:
        failures.append(f"{workload}/{expert} V0.2 CUDA launch count is not positive")
    if metrics.get("cuda_descriptor_traversal_us") is None:
        failures.append(f"{workload}/{expert} missing V0.2 CUDA descriptor traversal metric")
    elif float(metrics["cuda_descriptor_traversal_us"]) <= 0.0:
        failures.append(f"{workload}/{expert} V0.2 CUDA descriptor traversal metric is not positive")
    if metrics.get("tc_native_descriptor_count") is None:
        failures.append(f"{workload}/{expert} missing V0.2 native TC descriptor metric")
    elif float(metrics["tc_native_descriptor_count"]) <= 0.0:
        failures.append(f"{workload}/{expert} V0.2 native TC descriptor count is not positive")
    if metrics.get("tc_native_consumed_group_count") is None:
        failures.append(f"{workload}/{expert} missing V0.2 native TC group-consumption metric")
    elif float(metrics["tc_native_consumed_group_count"]) <= 0.0:
        failures.append(f"{workload}/{expert} V0.2 native TC group-consumption count is not positive")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    record = {metric: float(median(_metric_values(rows, metric))) for metric in CORE_METRICS}
    for metric in OPTIONAL_METRICS:
        values = _metric_values(rows, metric)
        record[metric] = float(median(values)) if values else None
    return record


def _dispatch_units(record: dict[str, float]) -> float | None:
    value = record.get("execution_dispatch_units")
    if value is not None and value > 0.0:
        return value
    return record.get("estimated_dispatch_units")


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"median": None, "min": None, "max": None, "count": 0}
    return {"median": float(median(values)), "min": min(values), "max": max(values), "count": len(values)}


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ATG + TC + BAA V0.2 Report",
        "",
        f"Gate: **{summary['gate']['status']}**",
        "",
        "## V0.2 vs Best Fixed TilePO",
        "",
        "| Workload | Experts | Best fixed policy | tok/s gain | tok/s margin | p95 delta | p99 delta | Strict win |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for comparison in summary["comparisons"]:
        best = comparison["v2_vs_best_fixed"]
        lines.append(
            "| {workload} | {experts} | {policy} | {tok:.2f}% | {margin:.3f} | {p95:.2f}% | {p99:.2f}% | {strict} |".format(
                workload=comparison["workload"],
                experts=comparison["experts_per_layer"],
                policy=comparison["best_fixed_policy"],
                tok=best["tok_gain_pct"],
                margin=best["tok_margin"],
                p95=best["p95_delta_pct"],
                p99=best["p99_delta_pct"],
                strict="yes" if comparison["strict_per_point_win"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Tile Count and Dispatch Proxy",
            "",
            "| Workload | Experts | Tile count C/F/H/V2 | Dispatch V2/F | GPU peak V2/Best | DRAM peak V2/Best |",
            "| --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for comparison in summary["comparisons"]:
        tile_counts = comparison["tile_count_comparison"]
        dispatch = comparison["dispatch_proxy"]
        memory = comparison["memory_comparison"]
        lines.append(
            "| {workload} | {experts} | {coarse}/{fine}/{hybrid}/{v2} | {dispatch:.2f}% | "
            "{v2_gpu:.3f}/{best_gpu:.3f} | "
            "{v2_cpu:.3f}/{best_cpu:.3f} |".format(
                workload=comparison["workload"],
                experts=comparison["experts_per_layer"],
                coarse=_fmt_nullable(tile_counts["coarse"]),
                fine=_fmt_nullable(tile_counts["fine"]),
                hybrid=_fmt_nullable(tile_counts["hybrid"]),
                v2=_fmt_nullable(tile_counts["v2"]),
                dispatch=dispatch["v2_vs_fine_pct"] if dispatch["v2_vs_fine_pct"] is not None else 0.0,
                v2_gpu=memory["v2_gpu_peak_gib"],
                best_gpu=memory["best_fixed_gpu_peak_gib"],
                v2_cpu=memory["v2_cpu_ram_peak_gib"],
                best_cpu=memory["best_fixed_cpu_ram_peak_gib"],
            )
        )
    if summary["gate"]["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["gate"]["warnings"])
    if summary["gate"]["failures"]:
        lines.extend(["", "## Gate Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["gate"]["failures"])
    return "\n".join(lines) + "\n"


def _markdown_v0_2_only(summary: dict[str, Any]) -> str:
    lines = [
        "# ATG + TC + BAA V0.2 Report",
        "",
        f"Gate: **{summary['gate']['status']}**",
        "",
        "## V0.2 Single-Policy Validation",
        "",
        "| Workload | Experts | tok/s | p95 ms | p99 ms | GPU GiB | DRAM GiB | Dispatch units | Native TC desc | CUDA launches | BAA critical path us |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["groups"]:
        metrics = group["metrics"]
        lines.append(
            "| {workload} | {experts} | {tok:.3f} | {p95:.3f} | {p99:.3f} | {gpu:.3f} | {cpu:.3f} | {dispatch} | {tc_desc} | {cuda} | {baa:.3f} |".format(
                workload=group["workload"],
                experts=group["experts_per_layer"],
                tok=_metric_median(metrics, "tok_per_sec"),
                p95=_metric_median(metrics, "p95_ms"),
                p99=_metric_median(metrics, "p99_ms"),
                gpu=_metric_median(metrics, "gpu_peak_gib"),
                cpu=_metric_median(metrics, "cpu_ram_peak_gib"),
                dispatch=_fmt_nullable(_metric_median(metrics, "execution_dispatch_units")),
                tc_desc=_fmt_nullable(_metric_median(metrics, "tc_native_descriptor_count")),
                cuda=_fmt_nullable(_metric_median(metrics, "cuda_launch_count")),
                baa=_metric_median(metrics, "baa_critical_path_us"),
            )
        )
    lines.extend(
        [
            "",
            "## V0.2 Runtime Evidence",
            "",
            "- KT/SGLang executor is preserved; TilePO attaches manifest-time ATG/TC/BAA metadata through the serving hook.",
            "- TC evidence requires native CUDA/KT-style consumption of coalesced group descriptors.",
            "- BAA evidence requires double-buffer metadata and zero critical-path planning time.",
        ]
    )
    if summary["gate"]["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["gate"]["warnings"])
    if summary["gate"]["failures"]:
        lines.extend(["", "## Gate Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["gate"]["failures"])
    return "\n".join(lines) + "\n"


def _metric_median(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, {}).get("median") if isinstance(metrics.get(key), dict) else None
    return float(value) if value is not None else 0.0


def _row_policy(row: dict[str, Any]) -> str:
    return str(row.get("tilepo_policy") or row.get("ablation_policy") or "")


def _normalize_policy(policy: str) -> str:
    return V2_POLICY if policy in LEGACY_ADAPTIVE_POLICIES else policy


def _row_async(row: dict[str, Any]) -> str:
    return str(row.get("tilepo_async_planning") or row.get("async_planning_mode") or "")


def _v2_plain_kt_fallback_events(row: dict[str, Any]) -> int | None:
    for key in ("unexpected_plain_kt_bypass_events", "plain_kt_fallback_events"):
        if key in row:
            return int(row.get(key) or 0)
    keys = ("kt_fallback_count",)
    if any(key in row for key in keys):
        return sum(int(row.get(key) or 0) for key in keys)
    return None


def _v2_fallback_counter_failures(index: int, row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in (
        "fallback_count",
        "serving_hook_fallback_count",
        "unexpected_plain_kt_bypass_events",
        "serving_hook_backend_fallback_count",
        "kt_fallback_count",
        "baa_fallback_count",
    ):
        if key not in row:
            continue
        try:
            value = int(row.get(key) or 0)
        except (TypeError, ValueError):
            failures.append(f"row {index} V0.2 fallback counter {key} is non-numeric: {row.get(key)!r}")
            continue
        if value != 0:
            failures.append(f"row {index} V0.2 used fallback counter {key}={value}")
    return failures


def _v2_backend(row: dict[str, Any]) -> str | None:
    for key in ("tilepo_backend", "backend_owner"):
        if row.get(key):
            return str(row[key])
    plan = row.get("tilepo_plan")
    if isinstance(plan, dict) and plan.get("backend_owner"):
        return str(plan["backend_owner"])
    counts = row.get("backend_launch_counts")
    if isinstance(counts, dict) and int(counts.get("cuda", 0) or 0) > 0:
        return "cuda"
    hook_counts = row.get("serving_hook_backend_launch_counts")
    if isinstance(hook_counts, dict) and int(hook_counts.get("cuda", 0) or 0) > 0:
        if _as_bool(row.get("tc_native_consumed")):
            return "kt_preserving_native_tc_cuda"
        if _as_bool(row.get("tc_adapter_consumed")):
            return "kt_preserving_launch_adapter_tc_cuda"
        return "kt_preserving_cuda_augmentation"
    if _as_bool(row.get("tc_native_consumed")):
        return "kt_preserving_native_tc_cuda"
    if _as_bool(row.get("tc_adapter_consumed")):
        return "kt_preserving_launch_adapter_tc_cuda"
    return None


def _v2_kt_preserving_serving_failures(index: int, row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    workload = str(row.get("workload", ""))
    expert = _as_int(row.get("experts_per_layer", 0))
    if not _as_bool(row.get("kt_executor_preserved")):
        failures.append(f"row {index} missing V0.2 KT/SGLang serving shell preservation evidence")
    if not _as_bool(row.get("tilepo_plan_applied_in_serving_path")):
        failures.append(f"row {index} missing V0.2 TilePO serving-path plan evidence")
    if not _as_bool(row.get("tc_coalescing_active")):
        failures.append(f"row {index} missing V0.2 TC coalescing evidence")
    if not _as_bool(row.get("tc_native_consumed")):
        failures.append(f"row {index} missing V0.2 native TC consumption evidence")
    if not _as_bool(row.get("tc_native_consumed_coalesced_groups")):
        failures.append(f"row {index} missing V0.2 native TC coalesced-group consumption evidence")
    descriptor_count = _required_v2_float(row, "tc_native_descriptor_count")
    if descriptor_count is None or descriptor_count <= 0.0:
        failures.append(f"row {index} missing V0.2 native TC descriptor count")
    elif workload == "mixed" and expert == 8 and int(descriptor_count) != 8:
        failures.append(f"row {index} V0.2 native TC descriptor count is not 8")
    if str(row.get("tc_native_entrypoint", "")) != "tilepo_cuda_dispatch_coalesced_gemm":
        failures.append(f"row {index} V0.2 native TC entrypoint is not tilepo_cuda_dispatch_coalesced_gemm")
    if str(row.get("tc_native_descriptor_layout", "")) != "tilepo_cuda_coalesced_group_desc_v1":
        failures.append(f"row {index} V0.2 native TC descriptor layout is not tilepo_cuda_coalesced_group_desc_v1")
    group_count = _required_v2_float(row, "tc_native_consumed_group_count")
    if group_count is None or group_count <= 0.0:
        failures.append(f"row {index} missing V0.2 native TC consumed group count")
    source = str(row.get("tc_native_consumption_source", ""))
    if source not in {"kt_grouped_moe_cuda_adapter", "kt_serving_cuda_kernel", "kt_launch_adapter_tc"}:
        failures.append(f"row {index} V0.2 native TC consumption source is not KT/CUDA: {source}")
    if not _as_bool(row.get("baa_double_buffered")):
        failures.append(f"row {index} missing V0.2 BAA double-buffer evidence")
    if not _as_bool(row.get("serving_hook_active")):
        failures.append(f"row {index} V0.2 serving hook was not active")
    if _required_v2_float(row, "serving_hook_invocations") is None:
        failures.append(f"row {index} missing V0.2 serving hook invocation evidence")
    elif int(float(row["serving_hook_invocations"])) <= 0:
        failures.append(f"row {index} V0.2 serving hook was not invoked")
    if "serving_hook_returned_original" not in row:
        failures.append(f"row {index} missing V0.2 serving hook return-mode evidence")
    elif _as_bool(row.get("serving_hook_returned_original")):
        failures.append(f"row {index} V0.2 native TC did not replace the measured serving path")
    replaced_count = _required_v2_float(row, "serving_hook_replaced_count")
    if replaced_count is None:
        failures.append(f"row {index} missing V0.2 native TC replacement count")
    elif int(replaced_count) <= 0:
        failures.append(f"row {index} V0.2 native TC replacement count is zero")
    if "serving_hook_verify_fail_count" in row:
        try:
            if int(row.get("serving_hook_verify_fail_count") or 0) != 0:
                failures.append(f"row {index} V0.2 KT-preserving hook verification failed")
        except (TypeError, ValueError):
            failures.append(f"row {index} V0.2 serving hook verification failure count is non-numeric")
    return failures


def _v2_kt_preserving_serving_real(row: dict[str, Any]) -> bool:
    return (
        _as_bool(row.get("kt_executor_preserved"))
        and _as_bool(row.get("tilepo_plan_applied_in_serving_path"))
        and _as_bool(row.get("tc_coalescing_active"))
        and _as_bool(row.get("tc_native_consumed"))
        and _as_bool(row.get("tc_native_consumed_coalesced_groups"))
        and (_required_v2_float(row, "tc_native_descriptor_count") or 0.0) > 0.0
        and str(row.get("tc_native_entrypoint", "")) == "tilepo_cuda_dispatch_coalesced_gemm"
        and str(row.get("tc_native_descriptor_layout", "")) == "tilepo_cuda_coalesced_group_desc_v1"
        and (_required_v2_float(row, "tc_native_consumed_group_count") or 0.0) > 0.0
        and str(row.get("tc_native_consumption_source", "")) in {
            "kt_grouped_moe_cuda_adapter",
            "kt_serving_cuda_kernel",
            "kt_launch_adapter_tc",
        }
        and _as_bool(row.get("baa_double_buffered"))
        and _as_bool(row.get("serving_hook_active"))
        and _required_v2_float(row, "serving_hook_invocations") is not None
        and int(float(row.get("serving_hook_invocations", 0))) > 0
        and not _as_bool(row.get("serving_hook_returned_original", True))
        and int(float(row.get("serving_hook_replaced_count", 0) or 0)) > 0
        and int(row.get("serving_hook_verify_fail_count", 0) or 0) == 0
    )


def _v2_runtime_metric_provenance_failures(index: int, row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    source = str(row.get("runtime_metrics_source", ""))
    if source == "side_probe":
        failures.append(f"row {index} V0.2 runtime metrics source is disconnected side_probe")
    elif source not in {
        "kt_preserving_hook",
        "serving_hook_backend",
        "hot_backend_probe",
        "kt_preserving_native_tc_kernel",
        "serving_hook_native_tc_backend",
        "kt_launch_adapter_tc",
    }:
        failures.append(f"row {index} V0.2 runtime metrics source is not KT-preserving TilePO evidence")
    if not _as_bool(row.get("baa_metrics_measured")):
        failures.append(f"row {index} V0.2 BAA metric is not marked measured")
    if not _as_bool(row.get("cuda_descriptor_metrics_measured")):
        failures.append(f"row {index} V0.2 CUDA descriptor metric is not marked measured")
    baa_us = _required_v2_float(row, "baa_critical_path_us")
    if baa_us is not None and baa_us < 0.0:
        failures.append(f"row {index} V0.2 BAA critical-path metric is negative")
    descriptor_us = _required_v2_float(row, "cuda_descriptor_traversal_us")
    launch_count = _required_v2_float(row, "cuda_launch_count")
    if descriptor_us is not None and launch_count is not None and launch_count > 0.0 and descriptor_us <= 0.0:
        failures.append(f"row {index} V0.2 CUDA descriptor traversal metric is not positive")
    return failures


def _required_v2_float(row: dict[str, Any], key: str) -> float | None:
    if key not in row:
        return None
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _to_float(value: Any) -> float:
    return float(value)


def _metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for row in rows:
        if row.get(metric) is not None:
            values.append(_to_float(row.get(metric)))
    return values


def _gain_pct(candidate: float, baseline: float) -> float:
    return ((candidate / baseline) - 1.0) * 100.0 if baseline else 0.0


def _reduction_pct(candidate: float, baseline: float) -> float:
    return (1.0 - (candidate / baseline)) * 100.0 if baseline else 0.0


def _gap_pct(candidate: float, best: float) -> float:
    return max(0.0, (1.0 - (candidate / best)) * 100.0) if best else 0.0


def _ratio_pct(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in {None, 0.0}:
        return None
    return (candidate / baseline) * 100.0


def _maybe_int(value: float | None) -> int | None:
    return int(value) if value is not None else None


def _between(lower: float | None, candidate: float | None, upper: float | None) -> bool | None:
    if lower is None or candidate is None or upper is None:
        return None
    return lower < candidate < upper


def _fmt_nullable(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(value))
