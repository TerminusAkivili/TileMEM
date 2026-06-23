from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from .dsl import DSLBlock, parse_tmem


ABLATION_EXPERT_BUDGETS = [1, 2, 4, 6, 8, 10, 12, 14, 16]
ABLATION_WORKLOADS = ["mixed", "profile_matched", "long_output"]
ABLATION_POLICIES = ["kt_expert", "tilepo_coarse", "tilepo_fine", "tilepo_hybrid"]
ABLATION_TILE_POLICIES = ["tilepo_coarse", "tilepo_fine", "tilepo_hybrid"]
ABLATION_ASYNC_MODES = ["off", "on"]


def render_tilepo_plan(
    base_plan_path: Path | str,
    *,
    expert_budget: int,
    policy: str,
    async_planning: bool,
) -> str:
    if policy not in ABLATION_TILE_POLICIES:
        raise ValueError(f"policy must be one of {ABLATION_TILE_POLICIES}, got {policy!r}")
    if expert_budget <= 0:
        raise ValueError("expert_budget must be positive")

    plan = parse_tmem(Path(base_plan_path).read_text())
    blocks: list[DSLBlock] = []
    for block in plan.blocks:
        values = dict(block.values)
        if block.kind == "workload":
            values["label"] = f"tilepo_{policy}_experts{expert_budget}"
        elif block.kind == "tile":
            values.update(_tile_values(policy, expert_budget))
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
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_tilepo_plan(
            base_plan_path,
            expert_budget=expert_budget,
            policy=policy,
            async_planning=async_planning,
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


def _tile_values(policy: str, expert_budget: int) -> dict[str, Any]:
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
    raise ValueError(f"unsupported V0.1 tile policy: {policy}")


def _deployment_mode(policy: str) -> str:
    if policy == "tilepo_coarse":
        return "speed"
    if policy == "tilepo_fine":
        return "memory"
    return "balanced"
