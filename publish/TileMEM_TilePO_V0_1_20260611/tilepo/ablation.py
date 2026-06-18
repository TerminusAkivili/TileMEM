from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

from .dsl import DSLBlock, parse_tmem


ABLATION_EXPERT_BUDGETS = [1, 2, 4, 6, 8, 10, 12, 14, 16]
ABLATION_WORKLOADS = ["mixed", "profile_matched", "long_output"]
ABLATION_POLICIES = [
    "kt_expert",
    "tilepo_coarse",
    "tilepo_fine",
    "tilepo_hybrid",
    "tilepo_adaptive",
    "tilepo_atg",
    "tilepo_atg_tc_baa",
]
ABLATION_TILE_POLICIES = [
    "tilepo_coarse",
    "tilepo_fine",
    "tilepo_hybrid",
    "tilepo_adaptive",
    "tilepo_atg",
    "tilepo_atg_tc_baa",
]
ABLATION_ASYNC_MODES = ["off", "on"]
ADAPTIVE_MODES = ("throughput", "balanced", "tail_latency")
ADAPTIVE_OBJECTIVE = "maximize_throughput_minus_metadata_dispatch_and_tail_penalty"
ATG_TC_BAA_OBJECTIVE = "strictly_beat_best_fixed_tilepo_policy_with_cuda_tc_baa"
ADAPTIVE_SHAPES: dict[str, dict[str, int]] = {
    "coarse": {"hidden_tile": 2048, "intermediate_tile": 8192, "shard_count": 1},
    "medium": {"hidden_tile": 512, "intermediate_tile": 2048, "shard_count": 4},
    "small": {"hidden_tile": 256, "intermediate_tile": 1024, "shard_count": 8},
    "fine": {"hidden_tile": 64, "intermediate_tile": 128, "shard_count": 64},
}


def render_tilepo_plan(
    base_plan_path: Path | str,
    *,
    expert_budget: int,
    policy: str,
    async_planning: bool,
    adaptive_mode: str = "throughput",
    workload_profile: str = "generic",
) -> str:
    if policy not in ABLATION_TILE_POLICIES:
        raise ValueError(f"policy must be one of {ABLATION_TILE_POLICIES}, got {policy!r}")
    if adaptive_mode not in ADAPTIVE_MODES:
        raise ValueError(f"adaptive_mode must be one of {ADAPTIVE_MODES}, got {adaptive_mode!r}")
    if expert_budget <= 0:
        raise ValueError("expert_budget must be positive")

    plan = parse_tmem(Path(base_plan_path).read_text())
    blocks: list[DSLBlock] = []
    for block in plan.blocks:
        values = dict(block.values)
        if block.kind == "workload":
            values["label"] = f"tilepo_{policy}_experts{expert_budget}"
            values["profile"] = str(workload_profile)
        elif block.kind == "tile":
            values.update(
                _tile_values(
                    policy,
                    expert_budget,
                    adaptive_mode=adaptive_mode,
                    workload_profile=workload_profile,
                )
            )
        elif block.kind == "memory":
            values["experts_per_layer"] = int(expert_budget)
        elif block.kind == "schedule":
            values["async_planning"] = bool(async_planning)
            values["deployment_mode"] = _deployment_mode(policy)
        blocks.append(replace(block, values=values))
    return type(plan)(blocks).compiled_text()


def write_tilepo_plan(
    base_plan_path: Path | str,
    output_path: Path | str,
    *,
    expert_budget: int,
    policy: str,
    async_planning: bool,
    adaptive_mode: str = "throughput",
    workload_profile: str = "generic",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_tilepo_plan(
            base_plan_path,
            expert_budget=expert_budget,
            policy=policy,
            async_planning=async_planning,
            adaptive_mode=adaptive_mode,
            workload_profile=workload_profile,
        )
    )
    return output


def write_merged_manifest(manifest_paths: list[Path | str], output_path: Path | str) -> Path:
    output = Path(output_path)
    runs: list[dict[str, Any]] = []
    command_runs: list[dict[str, Any]] = []
    sources: list[str] = []
    selected_workloads: set[str] = set()
    selected_experts: set[int] = set()
    selected_policies: set[str] = set()
    selected_async_modes: set[str] = set()
    for item in manifest_paths:
        path = Path(item)
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        sources.append(str(path))
        for row in data.get("runs", []):
            if isinstance(row, dict):
                runs.append(row)
                if row.get("workload") is not None:
                    selected_workloads.add(str(row["workload"]))
                if row.get("experts_per_layer") is not None:
                    selected_experts.add(int(row["experts_per_layer"]))
                if row.get("tilepo_policy") is not None:
                    selected_policies.add(str(row["tilepo_policy"]))
                if row.get("tilepo_async_planning") is not None:
                    selected_async_modes.add(str(row["tilepo_async_planning"]))
        for run in data.get("command_runs", []):
            if isinstance(run, dict):
                command_runs.append(run)
    merged = {
        "schema_version": "tilepo_merged_manifest_v1",
        "input_manifests": sources,
        "runs": runs,
        "command_runs": command_runs,
        "selected_workloads": sorted(selected_workloads),
        "selected_experts": sorted(selected_experts),
        "selected_policies": sorted(selected_policies),
        "selected_async_modes": sorted(selected_async_modes),
        "expected_result_rows": len(command_runs),
        "actual_result_rows": len(runs),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    return output


def _tile_values(
    policy: str,
    expert_budget: int,
    *,
    adaptive_mode: str = "throughput",
    workload_profile: str = "generic",
) -> dict[str, Any]:
    if policy == "tilepo_coarse":
        return {
            "tile_policy": policy,
            "hidden_tile": 2048,
            "intermediate_tile": 8192,
            "shard_count": 1,
        }
    if policy == "tilepo_fine":
        return {
            "tile_policy": policy,
            "hidden_tile": 64,
            "intermediate_tile": 128,
            "shard_count": 64,
        }
    if policy == "tilepo_hybrid":
        hot_budget = max(1, min(4, (expert_budget + 2) // 3))
        return {
            "tile_policy": policy,
            "hidden_tile": 64,
            "intermediate_tile": 128,
            "shard_count": 64,
            "hot_expert_budget": hot_budget,
            "hot_hidden_tile": 2048,
            "hot_intermediate_tile": 8192,
            "hot_shard_count": 1,
            "cold_hidden_tile": 64,
            "cold_intermediate_tile": 128,
            "cold_shard_count": 64,
        }
    if policy == "tilepo_adaptive":
        return _adaptive_tile_values(expert_budget, adaptive_mode)
    if policy == "tilepo_atg":
        values = _adaptive_tile_values(expert_budget, adaptive_mode)
        segments = values.get("adaptive_segments", [])
        values.update(
            {
                "tile_policy": "tilepo_atg",
                "adaptive_objective": ATG_TC_BAA_OBJECTIVE,
                "atg_candidate_id": _candidate_id(segments),
                "atg_selection_source": "cost_model",
                "fallback_policy": "tilepo_hybrid",
            }
        )
        return values
    if policy == "tilepo_atg_tc_baa":
        return _atg_tc_baa_tile_values(
            expert_budget,
            adaptive_mode,
            workload_profile=workload_profile,
        )
    raise ValueError(f"unsupported V0.1 tile policy: {policy}")


def _deployment_mode(policy: str) -> str:
    if policy in {"tilepo_coarse", "tilepo_adaptive", "tilepo_atg", "tilepo_atg_tc_baa"}:
        return "speed"
    if policy == "tilepo_fine":
        return "memory"
    return "balanced"


def _adaptive_tile_values(expert_budget: int, adaptive_mode: str) -> dict[str, Any]:
    hot_budget, warm_budget, cold_budget = _adaptive_budgets(expert_budget, adaptive_mode)
    shape_by_segment = _adaptive_shape_names(adaptive_mode)
    shape_by_segment = _coarsen_if_needed(expert_budget, hot_budget, warm_budget, cold_budget, shape_by_segment)
    segments = _adaptive_segments(hot_budget, warm_budget, cold_budget, shape_by_segment)
    values: dict[str, Any] = {
        "tile_policy": "tilepo_adaptive",
        "adaptive_mode": adaptive_mode,
        "adaptive_objective": ADAPTIVE_OBJECTIVE,
        "hidden_tile": ADAPTIVE_SHAPES["fine"]["hidden_tile"],
        "intermediate_tile": ADAPTIVE_SHAPES["fine"]["intermediate_tile"],
        "shard_count": ADAPTIVE_SHAPES["fine"]["shard_count"],
        "hot_expert_budget": hot_budget,
        "warm_expert_budget": warm_budget,
        "cold_expert_budget": cold_budget,
        "adaptive_segments": segments,
        "estimated_dispatch_units": _estimated_dispatch_units(segments),
        "coarse_equivalent_hot_ratio": round(hot_budget / expert_budget, 6),
    }
    for segment in segments:
        shape = ADAPTIVE_SHAPES[str(segment["shape"])]
        prefix = str(segment["name"])
        values[f"{prefix}_hidden_tile"] = shape["hidden_tile"]
        values[f"{prefix}_intermediate_tile"] = shape["intermediate_tile"]
        values[f"{prefix}_shard_count"] = shape["shard_count"]
    return values


def _atg_tc_baa_tile_values(
    expert_budget: int,
    adaptive_mode: str,
    *,
    workload_profile: str = "generic",
) -> dict[str, Any]:
    profile = _normalize_workload_profile(workload_profile)
    values = _atg_workload_tile_values(expert_budget, adaptive_mode, profile)
    segments = values.get("adaptive_segments", [])
    values.update(
        {
            "tile_policy": "tilepo_atg_tc_baa",
            "tilepo_policy_aliases": ["tilepo_adaptive", "tilepo_atg"],
            "adaptive_objective": ATG_TC_BAA_OBJECTIVE,
            "atg_candidate_id": _candidate_id(segments),
            "atg_selection_source": "workload_cost_model" if profile in {"mixed", "long_context"} else "cost_model",
            "workload_profile": profile,
            "serving_shell": "kt_sglang",
            "backend_owner": "cuda",
            "cuda_entrypoint": "tilepo_cuda_dispatch_coalesced_gemm",
            "cuda_descriptor_layout": "tilepo_cuda_coalesced_group_desc_v1",
            "tc_native_required_for_v0_2": True,
            "tc_native_entrypoint": "tilepo_cuda_dispatch_coalesced_gemm",
            "tc_native_descriptor_layout": "tilepo_cuda_coalesced_group_desc_v1",
            "tc_native_expected_descriptor_count": expert_budget,
            "tc_descriptor_kind": "native_execution_required",
            "tc_enabled": True,
            "tc_fallback_to_fixed_equivalent": False,
            "tc_grouping": "workload_aware_layer_expert_projection_cuda_bf16",
            "tc_max_group_bytes": 1 << 30,
            "baa_enabled": True,
            "baa_window_size": 32,
            "baa_confidence_threshold": 0.75,
            "baa_double_buffered": True,
            "baa_planning_on_critical_path": False,
            "baa_critical_path_us": 0.0,
            "fallback_policy": "tilepo_hybrid",
        }
    )
    return values


def _atg_workload_tile_values(expert_budget: int, adaptive_mode: str, workload_profile: str) -> dict[str, Any]:
    profile = _normalize_workload_profile(workload_profile)
    if profile == "mixed":
        hot_budget = expert_budget
        warm_budget = 0
        cold_budget = 0
        shape_by_segment = {"hot": "fine", "warm": "fine", "cold": "fine"}
    elif profile == "long_context":
        hot_budget = min(expert_budget, max(1, math.ceil(expert_budget * 0.20)))
        warm_budget = max(0, min(expert_budget - hot_budget, math.ceil(expert_budget * 0.30)))
        cold_budget = max(0, expert_budget - hot_budget - warm_budget)
        shape_by_segment = {"hot": "coarse", "warm": "small", "cold": "fine"}
    else:
        return _adaptive_tile_values(expert_budget, adaptive_mode)

    segments = _adaptive_segments(hot_budget, warm_budget, cold_budget, shape_by_segment)
    values: dict[str, Any] = {
        "tile_policy": "tilepo_adaptive",
        "adaptive_mode": adaptive_mode,
        "adaptive_objective": ADAPTIVE_OBJECTIVE,
        "workload_profile": profile,
        "hidden_tile": ADAPTIVE_SHAPES["fine"]["hidden_tile"],
        "intermediate_tile": ADAPTIVE_SHAPES["fine"]["intermediate_tile"],
        "shard_count": ADAPTIVE_SHAPES["fine"]["shard_count"],
        "hot_expert_budget": hot_budget,
        "warm_expert_budget": warm_budget,
        "cold_expert_budget": cold_budget,
        "adaptive_segments": segments,
        "estimated_dispatch_units": max(1, expert_budget),
        "coarse_equivalent_hot_ratio": round(hot_budget / max(1, expert_budget), 6),
    }
    for segment in segments:
        shape = ADAPTIVE_SHAPES[str(segment["shape"])]
        prefix = str(segment["name"])
        values[f"{prefix}_hidden_tile"] = shape["hidden_tile"]
        values[f"{prefix}_intermediate_tile"] = shape["intermediate_tile"]
        values[f"{prefix}_shard_count"] = shape["shard_count"]
    return values


def _normalize_workload_profile(workload_profile: str) -> str:
    text = str(workload_profile or "generic").strip().lower()
    if text in {"long", "long_output", "long-context"}:
        return "long_context"
    if text in {"mix", "mixed"}:
        return "mixed"
    return text or "generic"


def _candidate_id(segments: Any) -> str:
    if not isinstance(segments, list):
        return "atg_unknown"
    parts = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        parts.append(f"{segment.get('name', 'segment')}_{segment.get('shape', 'unknown')}")
    return "tc_baa_" + "_".join(parts) if parts else "tc_baa_atg_default"


def _adaptive_budgets(expert_budget: int, adaptive_mode: str) -> tuple[int, int, int]:
    hot_fraction, warm_fraction = {
        "throughput": (0.50, 0.25),
        "balanced": (0.25, 0.35),
        "tail_latency": (0.15, 0.25),
    }[adaptive_mode]
    hot_budget = max(1, math.ceil(expert_budget * hot_fraction))
    warm_end = min(expert_budget, max(hot_budget, math.ceil(expert_budget * (hot_fraction + warm_fraction))))
    warm_budget = max(0, warm_end - hot_budget)
    cold_budget = max(0, expert_budget - hot_budget - warm_budget)
    return hot_budget, warm_budget, cold_budget


def _adaptive_shape_names(adaptive_mode: str) -> dict[str, str]:
    if adaptive_mode == "throughput":
        return {"hot": "coarse", "warm": "medium", "cold": "fine"}
    if adaptive_mode == "balanced":
        return {"hot": "coarse", "warm": "small", "cold": "fine"}
    return {"hot": "coarse", "warm": "small", "cold": "fine"}


def _coarsen_if_needed(
    expert_budget: int,
    hot_budget: int,
    warm_budget: int,
    cold_budget: int,
    shape_by_segment: dict[str, str],
) -> dict[str, str]:
    shapes = dict(shape_by_segment)
    fine_units = expert_budget * _shape_units("fine")
    hybrid_hot = max(1, min(4, (expert_budget + 2) // 3))
    hybrid_units = hybrid_hot * _shape_units("coarse") + (expert_budget - hybrid_hot) * _shape_units("fine")
    for segment, replacement in (("cold", "small"), ("warm", "medium"), ("cold", "medium"), ("warm", "coarse")):
        units = (
            hot_budget * _shape_units(shapes["hot"])
            + warm_budget * _shape_units(shapes["warm"])
            + cold_budget * _shape_units(shapes["cold"])
        )
        if units <= fine_units * 0.85 and units <= hybrid_units * 1.25:
            break
        if shapes.get(segment) != replacement:
            shapes[segment] = replacement
    return shapes


def _adaptive_segments(
    hot_budget: int,
    warm_budget: int,
    cold_budget: int,
    shape_by_segment: dict[str, str],
) -> list[dict[str, Any]]:
    segments = []
    start = 0
    for name, budget, reason in (
        ("hot", hot_budget, "protect grouped GEMM and launch efficiency"),
        ("warm", warm_budget, "balance reuse, residency, and dispatch cost"),
        ("cold", cold_budget, "preserve VRAM admission flexibility"),
    ):
        if budget <= 0:
            continue
        shape_name = shape_by_segment[name]
        shape = ADAPTIVE_SHAPES[shape_name]
        end = start + budget
        segments.append(
            {
                "name": name,
                "shape": shape_name,
                "expert_start": start,
                "expert_end": end,
                "expert_count": budget,
                "hidden_tile": shape["hidden_tile"],
                "intermediate_tile": shape["intermediate_tile"],
                "shard_count": shape["shard_count"],
                "reason": reason,
            }
        )
        start = end
    return segments


def _estimated_dispatch_units(segments: list[dict[str, Any]]) -> int:
    return sum(int(segment["expert_count"]) * _shape_units(str(segment["shape"])) for segment in segments)


def _shape_units(shape_name: str) -> int:
    return max(1, math.ceil(8192 / ADAPTIVE_SHAPES[shape_name]["intermediate_tile"]))
