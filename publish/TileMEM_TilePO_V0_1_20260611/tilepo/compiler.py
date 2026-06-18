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
    elif tile_policy in {"tilepo_adaptive", "tilepo_atg", "tilepo_atg_tc_baa"}:
        prefix = _adaptive_segment_name(tile_values, expert)
        hidden_tile = int(tile_values.get(f"{prefix}_hidden_tile", hidden_tile))
        intermediate_tile = int(tile_values.get(f"{prefix}_intermediate_tile", intermediate_tile))
        shard_count = int(tile_values.get(f"{prefix}_shard_count", shard_count))
    tile_width = intermediate_tile if projection_group in {"gate_up", "up", "down"} else hidden_tile
    return max(1, tile_width), max(1, shard_count)


def _adaptive_segment_name(tile_values: dict[str, Any], expert: int) -> str:
    for segment in tile_values.get("adaptive_segments", []):
        if not isinstance(segment, dict):
            continue
        start = int(segment.get("expert_start", 0))
        end = int(segment.get("expert_end", start))
        if start <= expert < end:
            return str(segment.get("name", "cold"))
    hot_budget = int(tile_values.get("hot_expert_budget", 0))
    warm_budget = int(tile_values.get("warm_expert_budget", 0))
    if expert < hot_budget:
        return "hot"
    if expert < hot_budget + warm_budget:
        return "warm"
    return "cold"


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
        "estimated_dispatch_units": _estimated_dispatch_units(tile_block.values, int(memory_block.values.get("experts_per_layer", 0))),
    }
    if tile_policy in {"tilepo_adaptive", "tilepo_atg", "tilepo_atg_tc_baa"}:
        tile_count = len(manifest.get("tile_offsets", {}))
        expert_budget = max(1, int(memory_block.values.get("experts_per_layer", 0)))
        manifest["tilepo_plan"].update(
            {
                "adaptive_mode": str(tile_block.values.get("adaptive_mode", "throughput")),
                "adaptive_objective": str(tile_block.values.get("adaptive_objective", "")),
                "warm_expert_budget": int(tile_block.values.get("warm_expert_budget", 0)),
                "cold_expert_budget": int(tile_block.values.get("cold_expert_budget", 0)),
                "adaptive_segments": tile_block.values.get("adaptive_segments", []),
                "estimated_tile_count": tile_count,
                "estimated_dispatch_units": int(tile_block.values.get("estimated_dispatch_units", manifest["tilepo_plan"]["estimated_dispatch_units"])),
                "coarse_equivalent_hot_ratio": float(
                    tile_block.values.get(
                        "coarse_equivalent_hot_ratio",
                        int(tile_block.values.get("hot_expert_budget", 0)) / expert_budget,
                    )
                ),
            }
        )
    if tile_policy in {"tilepo_atg", "tilepo_atg_tc_baa"}:
        _attach_atg_metadata(tile_block.values, manifest)
    if tile_policy == "tilepo_atg_tc_baa":
        _attach_tc_baa_metadata(tile_block.values, manifest)
    manifest["tilepo_policy"] = tile_policy
    manifest["tilepo_async_planning"] = "on" if manifest["tilepo_plan"]["async_planning"] else "off"
    manifest["checksum"] = _manifest_checksum(manifest)


def _attach_atg_metadata(tile_values: dict[str, Any], manifest: dict[str, Any]) -> None:
    plan = manifest["tilepo_plan"]
    aliases = [str(item) for item in tile_values.get("tilepo_policy_aliases", ["tilepo_adaptive"])]
    plan.update(
        {
            "tilepo_policy_aliases": aliases,
            "atg_candidate_id": str(tile_values.get("atg_candidate_id", "tc_baa_atg_default")),
            "atg_selection_source": str(tile_values.get("atg_selection_source", "cost_model")),
            "fallback_policy": str(tile_values.get("fallback_policy", "tilepo_hybrid")),
            "workload_profile": str(tile_values.get("workload_profile", plan.get("workload_profile", "generic"))),
            "placement_tile_count": len(manifest.get("tile_offsets", {})),
            "backend_owner": str(tile_values.get("backend_owner", "cuda")),
            "serving_shell": str(tile_values.get("serving_shell", "kt_sglang")),
        }
    )


def _attach_tc_baa_metadata(tile_values: dict[str, Any], manifest: dict[str, Any]) -> None:
    descriptor_layout = str(
        tile_values.get("cuda_descriptor_layout", "tilepo_cuda_coalesced_group_desc_v1")
    )
    cuda_entrypoint = str(tile_values.get("cuda_entrypoint", "tilepo_cuda_dispatch_coalesced_gemm"))
    groups, tile_to_group = _build_coalesced_groups(
        manifest,
        workload_profile=str(tile_values.get("workload_profile", "generic")),
        cuda_entrypoint=cuda_entrypoint,
        descriptor_layout=descriptor_layout,
    )
    descriptor_buffer = _build_cuda_tc_descriptor_buffer(
        groups,
        descriptor_layout=descriptor_layout,
        cuda_entrypoint=cuda_entrypoint,
    )
    plan = manifest["tilepo_plan"]
    tc_enabled = bool(tile_values.get("tc_enabled", True))
    expected_descriptor_count = int(
        tile_values.get("tc_native_expected_descriptor_count", len(descriptor_buffer)) or len(descriptor_buffer)
    )
    if tc_enabled and expected_descriptor_count != len(descriptor_buffer):
        raise ValueError(
            f"ATB native TC expected {expected_descriptor_count} descriptors, got {len(descriptor_buffer)}"
        )
    execution_dispatch_units = len(groups) if tc_enabled else len(manifest.get("tile_offsets", {}))
    coalesced_group_count = len(groups) if tc_enabled else 0
    plan.update(
        {
            "tc_enabled": tc_enabled,
            "tc_grouping": str(tile_values.get("tc_grouping", "workload_aware_layer_expert_projection_cuda_bf16")),
            "tc_descriptor_kind": str(tile_values.get("tc_descriptor_kind", "native_execution_required")),
            "tc_fallback_to_fixed_equivalent": bool(tile_values.get("tc_fallback_to_fixed_equivalent", False)),
            "placement_tile_count": len(manifest.get("tile_offsets", {})),
            "execution_dispatch_units": execution_dispatch_units,
            "coalesced_group_count": coalesced_group_count,
            "backend_owner": str(tile_values.get("backend_owner", "cuda")),
            "cuda_entrypoint": cuda_entrypoint,
            "cuda_descriptor_layout": descriptor_layout,
            "tc_native_consumption": "kt_grouped_moe_cuda_adapter",
            "tc_native_required_for_v0_2": bool(tile_values.get("tc_native_required_for_v0_2", False)),
            "tc_native_entrypoint": str(tile_values.get("tc_native_entrypoint", cuda_entrypoint)),
            "tc_native_descriptor_layout": str(tile_values.get("tc_native_descriptor_layout", descriptor_layout)),
            "tc_native_expected_descriptor_count": expected_descriptor_count,
            "tc_native_expected_group_count": coalesced_group_count,
            "baa_enabled": bool(tile_values.get("baa_enabled", True)),
            "baa_window_size": int(tile_values.get("baa_window_size", 32)),
            "baa_confidence_threshold": float(tile_values.get("baa_confidence_threshold", 0.75)),
            "baa_double_buffered": bool(tile_values.get("baa_double_buffered", True)),
            "baa_planning_on_critical_path": bool(tile_values.get("baa_planning_on_critical_path", False)),
            "baa_critical_path_us": float(tile_values.get("baa_critical_path_us", 0.0) or 0.0),
            "baa_active_map_id": "map_A",
            "baa_standby_map_id": "map_B",
        }
    )
    manifest["coalesced_groups"] = groups
    manifest["tile_to_coalesced_group"] = tile_to_group
    manifest["dispatch_coalescing_units"] = groups
    manifest["cuda_tc_descriptor_buffer"] = descriptor_buffer
    manifest["baa_maps"] = {
        "active_map_id": "map_A",
        "standby_map_id": "map_B",
        "double_buffered": True,
        "descriptor_layout": descriptor_layout,
        "planning_on_critical_path": False,
        "map_A": {"source": "manifest_default", "dispatch_groups": [group["coalesced_group_id"] for group in groups]},
        "map_B": {"source": "rolling_histogram_speculative", "dispatch_groups": []},
    }


def _build_coalesced_groups(
    manifest: dict[str, Any],
    *,
    workload_profile: str,
    cuda_entrypoint: str,
    descriptor_layout: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records = _tile_records(manifest)
    profile = str(workload_profile or "generic").strip().lower()
    if profile == "mixed":
        return _build_mixed_expert_groups(
            records,
            cuda_entrypoint=cuda_entrypoint,
            descriptor_layout=descriptor_layout,
            workload_profile=profile,
        )
    return _build_projection_groups(
        records,
        cuda_entrypoint=cuda_entrypoint,
        descriptor_layout=descriptor_layout,
        workload_profile=profile,
    )


def _tile_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tile_ids = manifest.get("tile_ids", {})
    tile_bytes = manifest.get("tile_bytes", {})
    tile_dtype_map = manifest.get("tile_dtype_map", {})
    records: list[dict[str, Any]] = []
    for key, tile_id in tile_ids.items():
        if not isinstance(tile_id, dict):
            continue
        records.append(
            {
                "key": key,
                "layer": int(tile_id["layer"]),
                "expert": int(tile_id["expert"]),
                "projection_group": str(tile_id["projection_group"]),
                "shard_id": int(tile_id["shard_id"]),
                "n_start": int(tile_id["n_start"]),
                "n_end": int(tile_id["n_end"]),
                "dtype": str(tile_dtype_map.get(key, "bf16")),
                "bytes": int(tile_bytes.get(key, 0)),
            }
        )
    records.sort(key=lambda item: (item["expert"], item["layer"], item["projection_group"], item["n_start"], item["n_end"]))
    return records


def _build_mixed_expert_groups(
    records: list[dict[str, Any]],
    *,
    cuda_entrypoint: str,
    descriptor_layout: str,
    workload_profile: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((int(record["expert"]), str(record["dtype"])), []).append(record)
    groups: list[dict[str, Any]] = []
    tile_to_group: dict[str, str] = {}
    for index, ((expert, dtype), items) in enumerate(sorted(grouped.items())):
        tile_keys = [str(item["key"]) for item in items]
        group_id = f"G:MIXED:E{expert}:{index}"
        group = {
            "coalesced_group_id": group_id,
            "layer": "all",
            "expert": expert,
            "projection_group": "all",
            "tile_keys": tile_keys,
            "tile_count": len(tile_keys),
            "n_start": 0,
            "n_end": max((int(item["n_end"]) for item in items), default=0),
            "weight_bytes": sum(int(item["bytes"]) for item in items),
            "scale_bytes": 0,
            "dtype": dtype,
            "backend": "cuda",
            "backend_owner": "cuda",
            "cuda_entrypoint": cuda_entrypoint,
            "dispatch_kind": "cuda_coalesced_expert_group",
            "descriptor_layout": descriptor_layout,
            "workload_profile": workload_profile,
            "coalescing_scope": "expert_all_layers_projection_groups",
        }
        groups.append(group)
        for tile_key in tile_keys:
            tile_to_group[tile_key] = group_id
    return groups, tile_to_group


def _build_projection_groups(
    records: list[dict[str, Any]],
    *,
    cuda_entrypoint: str,
    descriptor_layout: str,
    workload_profile: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[tuple[int, int, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            int(record["layer"]),
            int(record["expert"]),
            str(record["projection_group"]),
            str(record["dtype"]),
        )
        grouped.setdefault(key, []).append(record)
    groups: list[dict[str, Any]] = []
    tile_to_group: dict[str, str] = {}
    for index, ((layer, expert, projection_group, dtype), items) in enumerate(sorted(grouped.items())):
        tile_keys = [str(item["key"]) for item in items]
        group_id = f"G:L{layer}:E{expert}:{projection_group}:{dtype}:{index}"
        group = {
            "coalesced_group_id": group_id,
            "layer": layer,
            "expert": expert,
            "projection_group": projection_group,
            "tile_keys": tile_keys,
            "tile_count": len(tile_keys),
            "n_start": min((int(item["n_start"]) for item in items), default=0),
            "n_end": max((int(item["n_end"]) for item in items), default=0),
            "weight_bytes": sum(int(item["bytes"]) for item in items),
            "scale_bytes": 0,
            "dtype": dtype,
            "backend": "cuda",
            "backend_owner": "cuda",
            "cuda_entrypoint": cuda_entrypoint,
            "dispatch_kind": "cuda_coalesced_gemm_group",
            "descriptor_layout": descriptor_layout,
            "workload_profile": workload_profile,
            "coalescing_scope": "layer_expert_projection",
        }
        groups.append(group)
        for tile_key in tile_keys:
            tile_to_group[tile_key] = group_id
    return groups, tile_to_group


def _build_cuda_tc_descriptor_buffer(
    groups: list[dict[str, Any]],
    *,
    descriptor_layout: str,
    cuda_entrypoint: str,
) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    byte_offset = 0
    tile_begin = 0
    for index, group in enumerate(groups):
        byte_count = int(group.get("weight_bytes", 0) or 0)
        tile_count = int(group.get("tile_count", len(group.get("tile_keys", []))) or 0)
        projection_group = str(group.get("projection_group", "all"))
        dispatch_kind = str(group.get("dispatch_kind", "cuda_coalesced_gemm_group"))
        descriptor = {
            "descriptor_id": index,
            "group_id": str(group.get("coalesced_group_id", f"G:{index}")),
            "layer_id": _descriptor_layer_id(group.get("layer")),
            "expert_id": int(group.get("expert", -1)),
            "projection_id": _projection_id(projection_group),
            "projection_group": projection_group,
            "tile_begin": tile_begin,
            "tile_count": tile_count,
            "byte_offset": byte_offset,
            "byte_count": byte_count,
            "dtype": str(group.get("dtype", "bf16")),
            "dtype_code": _dtype_code(str(group.get("dtype", "bf16"))),
            "layout": "kt_grouped_moe_bf16",
            "alignment": 16,
            "dispatch_kind": dispatch_kind,
            "dispatch_kind_code": _dispatch_kind_code(dispatch_kind),
            "descriptor_layout": descriptor_layout,
            "cuda_entrypoint": cuda_entrypoint,
        }
        descriptors.append(descriptor)
        byte_offset += byte_count
        tile_begin += tile_count
    return descriptors


def _descriptor_layer_id(value: Any) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _projection_id(projection_group: str) -> int:
    return {"gate_up": 0, "up": 1, "down": 2, "all": 255}.get(projection_group, 254)


def _dtype_code(dtype: str) -> int:
    return {"bf16": 1, "fp8": 2, "mxfp4": 3}.get(dtype, 0)


def _dispatch_kind_code(dispatch_kind: str) -> int:
    if dispatch_kind == "cuda_coalesced_expert_group":
        return 2
    if dispatch_kind == "cuda_coalesced_gemm_group":
        return 1
    return 0


def _estimated_dispatch_units(tile_values: dict[str, Any], expert_budget: int) -> int:
    tile_policy = str(tile_values.get("tile_policy", ""))
    if expert_budget <= 0:
        return 0
    if tile_policy == "tilepo_hybrid":
        hot_budget = int(tile_values.get("hot_expert_budget", 1))
        cold_budget = max(0, expert_budget - hot_budget)
        return hot_budget * _shape_units_from_tile_width(int(tile_values.get("hot_intermediate_tile", 8192))) + (
            cold_budget * _shape_units_from_tile_width(int(tile_values.get("cold_intermediate_tile", 128)))
        )
    if tile_policy in {"tilepo_adaptive", "tilepo_atg", "tilepo_atg_tc_baa"}:
        value = tile_values.get("estimated_dispatch_units")
        if value is not None:
            return int(value)
    return expert_budget * _shape_units_from_tile_width(int(tile_values.get("intermediate_tile", 8192)))


def _shape_units_from_tile_width(intermediate_tile: int) -> int:
    return max(1, (8192 + max(1, intermediate_tile) - 1) // max(1, intermediate_tile))


def _manifest_checksum(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    return hashlib.sha256(encoded).hexdigest()
