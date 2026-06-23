from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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
)


MODEL_SPEC_SCHEMA_VERSION = "tilemem_model_spec_v1"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    experts_per_layer: int
    hidden_size: int
    intermediate_size: int
    expert_budget: int
    workload: str
    tile: dict[str, Any]
    memory: dict[str, Any]
    precision: dict[str, Any]
    schedule: dict[str, Any]

    def validate(self) -> None:
        _require_positive(self.layers, "layers")
        _require_positive(self.experts_per_layer, "experts_per_layer")
        _require_positive(self.hidden_size, "hidden_size")
        _require_positive(self.intermediate_size, "intermediate_size")
        _require_positive(self.expert_budget, "expert_budget")
        if self.expert_budget > self.experts_per_layer:
            raise ValueError("expert_budget must be <= experts_per_layer")
        if not self.name:
            raise ValueError("model name is required")
        if not self.workload:
            raise ValueError("workload is required")
        dtype_policy = str(self.precision.get("dtype_policy", "bf16"))
        allowed = {str(dtype) for dtype in self.precision.get("allowed", self.precision.get("allow", ["bf16"]))}
        if dtype_policy != "bf16" or allowed != {"bf16"}:
            raise ValueError("public model spec v1 is BF16-only; mixed precision is out of scope for v0.1.1")


@runtime_checkable
class ModelAdapter(Protocol):
    def to_tilemem_model_spec(self) -> ModelSpec:
        """Return a TileMEM public model spec for this model/checkpoint."""


def model_spec_from_dict(data: dict[str, Any]) -> ModelSpec:
    schema_version = str(data.get("schema_version", MODEL_SPEC_SCHEMA_VERSION))
    if schema_version != MODEL_SPEC_SCHEMA_VERSION:
        raise ValueError(f"unsupported model spec schema_version: {schema_version}")
    spec = ModelSpec(
        name=str(data["name"]),
        layers=int(data["layers"]),
        experts_per_layer=int(data["experts_per_layer"]),
        hidden_size=int(data["hidden_size"]),
        intermediate_size=int(data["intermediate_size"]),
        expert_budget=int(data.get("expert_budget", data.get("experts_per_layer", 0))),
        workload=str(data.get("workload", "mixed")),
        tile=dict(data.get("tile", {})),
        memory=dict(data.get("memory", {})),
        precision=dict(data.get("precision", {})),
        schedule=dict(data.get("schedule", {})),
    )
    spec.validate()
    return spec


def model_spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    spec.validate()
    return {
        "schema_version": MODEL_SPEC_SCHEMA_VERSION,
        "name": spec.name,
        "layers": spec.layers,
        "experts_per_layer": spec.experts_per_layer,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "expert_budget": spec.expert_budget,
        "workload": spec.workload,
        "tile": dict(spec.tile),
        "memory": dict(spec.memory),
        "precision": dict(spec.precision),
        "schedule": dict(spec.schedule),
    }


def build_mir_from_model_spec(spec_or_adapter: ModelSpec | ModelAdapter | dict[str, Any]) -> ModelIR:
    spec = _coerce_model_spec(spec_or_adapter)
    tile = spec.tile
    memory = spec.memory
    precision = spec.precision
    schedule = spec.schedule

    hidden_tile = _positive_int(tile.get("hidden_tile", spec.hidden_size), "tile.hidden_tile")
    intermediate_tile = _positive_int(tile.get("intermediate_tile", spec.intermediate_size), "tile.intermediate_tile")
    shard_count = _positive_int(tile.get("shard_count", 1), "tile.shard_count")
    projection_groups = [str(item) for item in tile.get("projection_groups", ["gate_up", "down"])]
    if not projection_groups:
        raise ValueError("tile.projection_groups must not be empty")

    allowed = [TileDType(item) for item in precision.get("allowed", precision.get("allow", ["bf16"]))]
    dtype_policy = str(precision.get("dtype_policy", "bf16"))
    tile_dtype = _select_tile_dtype(allowed, dtype_policy)

    tiles: list[TileIR] = []
    hot_tile_ids: list[TileId] = []
    hot_experts: dict[str, list[int]] = {}
    for layer in range(spec.layers):
        hot_experts[str(layer)] = list(range(spec.expert_budget))
        for expert in range(spec.expert_budget):
            for projection_group in projection_groups:
                extent = spec.intermediate_size if projection_group in {"gate_up", "up", "down"} else spec.hidden_size
                tile_width = intermediate_tile if projection_group in {"gate_up", "up", "down"} else hidden_tile
                for shard, n_start, n_end in _fixed_shards(extent, tile_width, shard_count):
                    tile_id = TileId(layer, expert, projection_group, shard, n_start, n_end)
                    tile_ir = TileIR(
                        tile_id=tile_id,
                        dtype=tile_dtype,
                        bytes=_tile_bytes(tile_id, spec.hidden_size, tile_dtype),
                        scale_bytes=16 if tile_dtype in {TileDType.FP8, TileDType.MXFP4} else 0,
                    )
                    tiles.append(tile_ir)
                    hot_tile_ids.append(tile_id)

    fallback_chain = []
    for item in schedule.get("fallback_chain", ["bf16", "kt"]):
        fallback_chain.append(TileDType(item) if item in {dtype.value for dtype in TileDType} else str(item))

    mir = ModelIR(
        name=spec.name,
        layers=spec.layers,
        experts_per_layer=spec.experts_per_layer,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        routes=[RouteIR(spec.workload, hot_experts)],
        tiles=tiles,
        residency=ResidencyIR(
            gpu_cache_budget_gib=float(memory.get("gpu_cache_budget_gib", 0.0)),
            cpu_cache_budget_gib=float(memory.get("cpu_cache_budget_gib", 0.0)),
            gpu_hot_tiles=hot_tile_ids,
            fallback_chain=fallback_chain,
        ),
        precision=PrecisionIR(
            dtype_policy=dtype_policy,
            allowed=allowed,
            calibration_required=bool(precision.get("calibration_required", False)),
        ),
        schedule=ScheduleIR(
            mode=RuntimeMode(str(schedule.get("mode", "verify"))),
            deployment_mode=DeploymentMode(str(schedule.get("deployment_mode", "balanced"))),
            backend_priority=[Backend(item) for item in schedule.get("backend_priority", ["cuda", "tilelang", "kt_fallback"])],
            runtime_gates=[str(item) for item in schedule.get("runtime_gates", [])],
            prewarm_policy=str(schedule.get("prewarm_policy", "hotset")),
            miss_policy=str(schedule.get("miss_policy", "fallback")),
        ),
    )
    mir.validate()
    return mir


def _coerce_model_spec(spec_or_adapter: ModelSpec | ModelAdapter | dict[str, Any]) -> ModelSpec:
    if isinstance(spec_or_adapter, ModelSpec):
        spec_or_adapter.validate()
        return spec_or_adapter
    if isinstance(spec_or_adapter, dict):
        return model_spec_from_dict(spec_or_adapter)
    spec = spec_or_adapter.to_tilemem_model_spec()
    spec.validate()
    return spec


def _select_tile_dtype(allowed: list[TileDType], dtype_policy: str) -> TileDType:
    if dtype_policy == "bf16":
        return TileDType.BF16
    for dtype in (TileDType.MXFP4, TileDType.FP8, TileDType.BF16):
        if dtype in allowed:
            return dtype
    raise ValueError("no supported dtype in precision.allowed")


def _fixed_shards(extent: int, tile_width: int, shard_count: int) -> list[tuple[int, int, int]]:
    ranges = []
    for shard in range(shard_count):
        n_start = min(extent, shard * tile_width)
        n_end = min(extent, n_start + tile_width)
        if n_start != n_end:
            ranges.append((shard, n_start, n_end))
    return ranges


def _tile_bytes(tile_id: TileId, hidden_size: int, dtype: TileDType) -> int:
    elements = max(1, (tile_id.n_end - tile_id.n_start) * hidden_size)
    if dtype == TileDType.BF16:
        return elements * 2
    if dtype == TileDType.FP8:
        return elements
    return max(1, elements // 2)


def _positive_int(value: Any, field: str) -> int:
    result = int(value)
    _require_positive(result, field)
    return result


def _require_positive(value: int, field: str) -> None:
    if value <= 0:
        raise ValueError(f"{field} must be positive")
