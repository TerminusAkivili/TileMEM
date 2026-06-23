from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .dsl import DSLPlan, parse_tmem
from .model_interface import build_mir_from_model_spec, model_spec_from_dict, model_spec_to_dict
from .mir import (
    Backend,
    DeploymentMode,
    ModelIR,
    PrecisionIR,
    ResidencyIR,
    RouteIR,
    RuntimeMode,
    ScheduleIR,
    TileDType,
    TileIR,
    TileId,
    build_manifest,
    save_mir,
)


@dataclass(frozen=True)
class CompileResult:
    mir_path: Path
    manifest_path: Path
    compiled_plan_path: Path
    mir: ModelIR
    manifest: dict[str, Any]


def compile_plan(plan_path: Path | str, out_dir: Path | str) -> CompileResult:
    plan_path = Path(plan_path)
    out_dir = Path(out_dir)
    plan = parse_tmem(plan_path.read_text())
    mir = lower_plan_to_mir(plan)
    manifest = build_manifest(mir)
    _attach_ablation_metadata(plan, manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = plan_path.stem
    mir_path = out_dir / f"{stem}.mir.json"
    manifest_path = out_dir / f"{stem}.manifest.json"
    compiled_plan_path = out_dir / f"{stem}.compiled.tmem"
    mir_path.write_text(json.dumps(mir.to_dict(), indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    compiled_plan_path.write_text(plan.compiled_text())
    return CompileResult(mir_path, manifest_path, compiled_plan_path, mir, manifest)


def compile_model_spec(model_spec_path: Path | str, out_dir: Path | str) -> CompileResult:
    model_spec_path = Path(model_spec_path)
    out_dir = Path(out_dir)
    spec = model_spec_from_dict(json.loads(model_spec_path.read_text()))
    mir = build_mir_from_model_spec(spec)
    manifest = build_manifest(mir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = spec.name
    mir_path = out_dir / f"{stem}.mir.json"
    manifest_path = out_dir / f"{stem}.manifest.json"
    compiled_plan_path = out_dir / f"{stem}.model_spec.json"
    save_mir(mir, mir_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    compiled_plan_path.write_text(json.dumps(model_spec_to_dict(spec), indent=2, sort_keys=True) + "\n")
    return CompileResult(mir_path, manifest_path, compiled_plan_path, mir, manifest)


def lower_plan_to_mir(plan: DSLPlan) -> ModelIR:
    model_block = plan.required_block("model")
    workload_block = plan.required_block("workload")
    tile_block = plan.required_block("tile")
    memory_block = plan.required_block("memory")
    precision_block = plan.required_block("precision")
    schedule_block = plan.required_block("schedule")
    runtime_block = plan.required_block("runtime")

    layers = _required_int(model_block.values, "layers")
    experts_per_layer = _required_int(model_block.values, "experts_per_layer")
    hidden_size = _required_int(model_block.values, "hidden_size")
    intermediate_size = _required_int(model_block.values, "intermediate_size")
    experts_budget = min(_required_int(memory_block.values, "experts_per_layer"), experts_per_layer)
    shard_count = max(1, _required_int(tile_block.values, "shard_count"))
    hidden_tile = _required_int(tile_block.values, "hidden_tile")
    intermediate_tile = _required_int(tile_block.values, "intermediate_tile")
    tile_policy = str(tile_block.values.get("tile_policy", "uniform"))
    projection_groups = [str(x) for x in tile_block.values.get("projection_groups", ["gate_up", "down"])]
    allowed = [TileDType(dtype) for dtype in precision_block.values.get("allow", ["bf16"])]
    dtype_policy = str(precision_block.values.get("dtype_policy", "bf16"))
    tile_dtype = _select_tile_dtype(allowed, dtype_policy)

    tiles: list[TileIR] = []
    hot_tile_ids: list[TileId] = []
    hot_experts: dict[str, list[int]] = {}
    for layer in range(layers):
        hot_experts[str(layer)] = list(range(experts_budget))
        for expert in range(experts_budget):
            for projection_group in projection_groups:
                extent = intermediate_size if projection_group in {"gate_up", "up", "down"} else hidden_size
                for shard, n_start, n_end in _tile_ranges(
                    tile_values=tile_block.values,
                    tile_policy=tile_policy,
                    expert=expert,
                    projection_group=projection_group,
                    extent=extent,
                    hidden_tile=hidden_tile,
                    intermediate_tile=intermediate_tile,
                    shard_count=shard_count,
                ):
                    tile_id = TileId(layer, expert, projection_group, shard, n_start, n_end)
                    tile_bytes = _tile_bytes(tile_id, hidden_size, tile_dtype)
                    scale_bytes = 16 if tile_dtype in {TileDType.FP8, TileDType.MXFP4} else 0
                    tile = TileIR(tile_id, tile_dtype, tile_bytes, scale_bytes)
                    tiles.append(tile)
                    hot_tile_ids.append(tile_id)

    mode_text = str(runtime_block.values.get("mode", "shadow"))
    mode = RuntimeMode(mode_text[:-5] if mode_text.endswith("_mode") else mode_text)
    deployment_text = str(schedule_block.values.get("deployment_mode", "balanced"))
    deployment_mode = DeploymentMode(deployment_text)
    backend_priority = [Backend(item) for item in schedule_block.values.get("backend_priority", ["cuda", "tilelang", "kt_fallback"])]
    fallback_chain = []
    for item in runtime_block.values.get("fallback_chain", ["mxfp4", "fp8", "bf16", "kt"]):
        fallback_chain.append(TileDType(item) if item in {dtype.value for dtype in TileDType} else str(item))

    mir = ModelIR(
        name=model_block.name,
        layers=layers,
        experts_per_layer=experts_per_layer,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        routes=[RouteIR(str(workload_block.values.get("label", workload_block.name)), hot_experts)],
        tiles=tiles,
        residency=ResidencyIR(
            gpu_cache_budget_gib=float(memory_block.values.get("gpu_cache_budget_gib", 0.0)),
            cpu_cache_budget_gib=float(memory_block.values.get("cpu_cache_budget_gib", 0.0)),
            gpu_hot_tiles=hot_tile_ids,
            fallback_chain=fallback_chain,
        ),
        precision=PrecisionIR(
            dtype_policy=dtype_policy,
            allowed=allowed,
            calibration_required=bool(precision_block.values.get("calibration_required", False)),
        ),
        schedule=ScheduleIR(
            mode=mode,
            deployment_mode=deployment_mode,
            backend_priority=backend_priority,
            runtime_gates=[str(x) for x in runtime_block.values.get("gates", [])],
            prewarm_policy=str(schedule_block.values.get("prewarm_policy", "none")),
            miss_policy=str(schedule_block.values.get("miss_policy", "fallback")),
        ),
    )
    mir.validate()
    return mir


def _required_int(values: dict[str, Any], key: str) -> int:
    if key not in values:
        raise ValueError(f"missing required DSL key: {key}")
    value = int(values[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _select_tile_dtype(allowed: list[TileDType], dtype_policy: str) -> TileDType:
    if dtype_policy == "bf16":
        return TileDType.BF16
    for dtype in (TileDType.MXFP4, TileDType.FP8, TileDType.BF16):
        if dtype in allowed:
            return dtype
    raise ValueError("no supported dtype in precision allow list")


def _tile_bytes(tile_id: TileId, hidden_size: int, dtype: TileDType) -> int:
    elements = max(1, (tile_id.n_end - tile_id.n_start) * hidden_size)
    if dtype == TileDType.BF16:
        return elements * 2
    if dtype == TileDType.FP8:
        return elements
    return max(1, elements // 2)


def _tile_ranges(
    *,
    tile_values: dict[str, Any],
    tile_policy: str,
    expert: int,
    projection_group: str,
    extent: int,
    hidden_tile: int,
    intermediate_tile: int,
    shard_count: int,
) -> list[tuple[int, int, int]]:
    if not tile_policy.startswith("tilepo_"):
        tile_width = intermediate_tile if projection_group in {"gate_up", "up", "down"} else hidden_tile
        return _fixed_shards(extent, tile_width, shard_count)

    tile_width, policy_shards = _ablation_tile_shape(
        tile_values=tile_values,
        tile_policy=tile_policy,
        expert=expert,
        projection_group=projection_group,
        hidden_tile=hidden_tile,
        intermediate_tile=intermediate_tile,
        shard_count=shard_count,
    )
    if tile_width >= extent:
        return [(0, 0, extent)]
    return _covering_shards(extent, tile_width, policy_shards)


def _ablation_tile_shape(
    *,
    tile_values: dict[str, Any],
    tile_policy: str,
    expert: int,
    projection_group: str,
    hidden_tile: int,
    intermediate_tile: int,
    shard_count: int,
) -> tuple[int, int]:
    if tile_policy == "tilepo_hybrid":
        hot_budget = int(tile_values.get("hot_expert_budget", 1))
        prefix = "hot" if expert < hot_budget else "cold"
        hidden_tile = int(tile_values.get(f"{prefix}_hidden_tile", hidden_tile))
        intermediate_tile = int(tile_values.get(f"{prefix}_intermediate_tile", intermediate_tile))
        shard_count = int(tile_values.get(f"{prefix}_shard_count", shard_count))
    tile_width = intermediate_tile if projection_group in {"gate_up", "up", "down"} else hidden_tile
    return max(1, tile_width), max(1, shard_count)


def _fixed_shards(extent: int, tile_width: int, shard_count: int) -> list[tuple[int, int, int]]:
    ranges = []
    for shard in range(max(1, shard_count)):
        n_start = min(extent, shard * tile_width)
        n_end = min(extent, n_start + tile_width)
        if n_start != n_end:
            ranges.append((shard, n_start, n_end))
    return ranges


def _covering_shards(extent: int, tile_width: int, shard_count: int) -> list[tuple[int, int, int]]:
    needed = (extent + tile_width - 1) // tile_width
    count = max(1, min(max(1, shard_count), needed))
    return [
        (shard, shard * tile_width, min(extent, (shard + 1) * tile_width))
        for shard in range(count)
        if shard * tile_width < extent
    ]


def _attach_ablation_metadata(plan: DSLPlan, manifest: dict[str, Any]) -> None:
    tile_block = plan.required_block("tile")
    tile_policy = str(tile_block.values.get("tile_policy", ""))
    if not tile_policy.startswith("tilepo_"):
        return
    memory_block = plan.required_block("memory")
    schedule_block = plan.required_block("schedule")
    manifest["tilepo_plan"] = {
        "policy": tile_policy,
        "async_planning": bool(schedule_block.values.get("async_planning", False)),
        "expert_budget": int(memory_block.values.get("experts_per_layer", 0)),
        "tile_count": len(manifest.get("tile_offsets", {})),
        "gpu_hot_tile_count": len(manifest.get("gpu_hot_tiles", [])),
        "hot_expert_budget": int(tile_block.values.get("hot_expert_budget", 0)),
    }
    manifest["tilepo_policy"] = tile_policy
    manifest["tilepo_async_planning"] = "on" if manifest["tilepo_plan"]["async_planning"] else "off"
    manifest["checksum"] = _manifest_checksum(manifest)


def _manifest_checksum(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    return hashlib.sha256(encoded).hexdigest()
