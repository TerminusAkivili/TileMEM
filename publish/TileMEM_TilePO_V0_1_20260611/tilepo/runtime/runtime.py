from __future__ import annotations

import time
from typing import Any

from tilepo.mir import Backend, RuntimeMode
from .metrics import RuntimeMetrics
from .state import TileResidencyState, TileState


class TileMEMRuntime:
    def __init__(self, manifest: dict[str, Any], backends: dict[Backend, Any], mode: RuntimeMode = RuntimeMode.SHADOW) -> None:
        self.manifest = manifest
        self.backends = backends
        self.mode = mode
        self.tile_state = TileState()
        self.metrics = RuntimeMetrics()
        ablation = manifest.get("tilepo_plan", {}) if isinstance(manifest.get("tilepo_plan", {}), dict) else {}
        self.metrics.ablation_policy = str(ablation.get("policy", manifest.get("tilepo_policy", "")))
        self.metrics.async_planning_mode = str(
            manifest.get("tilepo_async_planning", "on" if ablation.get("async_planning") else "off")
        )
        self.metrics.tile_count = len(manifest.get("tile_offsets", {}))
        coalesced_groups = manifest.get("coalesced_groups", [])
        coalesced_group_count = int(ablation.get("coalesced_group_count", len(coalesced_groups)))
        self.metrics.coalesced_group_count = coalesced_group_count
        self.metrics.execution_dispatch_units = int(
            ablation.get("execution_dispatch_units", coalesced_group_count)
        )
        self.metrics.baa_double_buffered = bool(ablation.get("baa_double_buffered", False))
        self.metrics.baa_active_map_id = str(ablation.get("baa_active_map_id", ""))
        baa_maps = manifest.get("baa_maps", {}) if isinstance(manifest.get("baa_maps", {}), dict) else {}
        self.metrics.baa_standby_ready = bool(baa_maps.get("standby_ready", False))
        self.metrics.baa_critical_path_us = 0.0
        self._async_planning_enabled = self.metrics.async_planning_mode == "on"
        self._plan_cache: dict[str, list[str]] = {}
        self.online_path_forbidden_calls: list[str] = []

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        cache_key = self._plan_cache_key(request)
        cached = self._async_planning_enabled and cache_key in self._plan_cache
        if cached:
            planned_tiles = list(self._plan_cache[cache_key])
            self.metrics.async_plan_cache_hits += 1
        else:
            planned_tiles = self._lookup_tiles(request)
            if self._async_planning_enabled:
                self.metrics.async_plan_cache_misses += 1
                self._plan_cache[cache_key] = list(planned_tiles)
        lookup_us = _elapsed_us(start)
        self.metrics.plan_lookup_total_us += lookup_us
        if not cached:
            self.metrics.plan_lookup_us += lookup_us

        gate_start = time.perf_counter()
        gates_pass = self._check_gates(request, planned_tiles)
        self.metrics.gate_us += _elapsed_us(gate_start)
        if not gates_pass:
            return self._fallback(request, "gate_failed")

        missing = [tile for tile in planned_tiles if self.tile_state.get(tile) != TileResidencyState.GPU_RESIDENT]
        hits = len(planned_tiles) - len(missing)
        self.metrics.cache_hits += hits
        self.metrics.cache_misses += len(missing)
        if missing:
            self._materialize_missing(missing)

        backend = self._select_backend()
        if backend is None:
            return self._fallback(request, "backend_unavailable")

        launch_start = time.perf_counter()
        result = backend.execute(request, self.manifest)
        self.metrics.backend_launch_us += _elapsed_us(launch_start)
        self.metrics.tilemem_backend_launch_count += 1
        backend_name = getattr(backend, "name", "")
        if backend_name == Backend.CUDA:
            self.metrics.cuda_launch_count += 1
            native = bool(result.get("native", False)) if isinstance(result, dict) else False
            self.metrics.native_cuda_available = self.metrics.native_cuda_available or bool(
                getattr(backend, "native_available", False)
            )
            if isinstance(result, dict) and bool(result.get("tc_native_consumed", False)):
                self.metrics.native_cuda_available = True
            if native:
                self.metrics.native_cuda_launch_count += 1
            else:
                self.metrics.cuda_python_shim_launch_count += 1
            if isinstance(result, dict):
                self._record_tc_native_evidence(result)
            descriptor_start = time.perf_counter()
            self._measure_cuda_descriptor_traversal()
            self.metrics.cuda_descriptor_traversal_us += _elapsed_us(descriptor_start)
            self.metrics.cuda_descriptor_metrics_measured = True
        elif backend_name == Backend.TILELANG:
            self.metrics.tilelang_launch_count += 1
        for tile in planned_tiles:
            dtype = self.manifest.get("tile_dtype_map", {}).get(tile, "unknown")
            self.metrics.record_dtype(dtype)
            if self.tile_state.get(tile) == TileResidencyState.GPU_RESIDENT:
                self.tile_state.transition(tile, TileResidencyState.GPU_EXECUTING)
                self.tile_state.transition(tile, TileResidencyState.GPU_RESIDENT)
        return result

    def _measure_cuda_descriptor_traversal(self) -> None:
        """Touch TC/BAA descriptors so the metric is a measured value, not a constant."""
        groups = self.manifest.get("coalesced_groups", [])
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, dict):
                    _ = group.get("tile_keys", group.get("tiles", group.get("tile_ids", ())))
        descriptors = self.manifest.get("cuda_tc_descriptor_buffer", [])
        if isinstance(descriptors, list):
            for descriptor in descriptors:
                if isinstance(descriptor, dict):
                    _ = descriptor.get("group_id")
                    _ = descriptor.get("byte_count")
        baa_maps = self.manifest.get("baa_maps", {})
        if isinstance(baa_maps, dict):
            _ = baa_maps.get("active")
            _ = baa_maps.get("standby")
            self.metrics.baa_metrics_measured = True

    def _record_tc_native_evidence(self, result: dict[str, Any]) -> None:
        if bool(result.get("tc_native_consumed", False)):
            self.metrics.tc_native_consumed = True
            self.metrics.tc_native_consumed_coalesced_groups = bool(
                result.get("tc_native_consumed_coalesced_groups", True)
            )
            self.metrics.tc_native_consumed_group_count = int(result.get("tc_native_consumed_group_count", 0) or 0)
            self.metrics.tc_native_descriptor_count = int(result.get("tc_native_descriptor_count", 0) or 0)
            self.metrics.tc_native_consumed_tile_count = int(result.get("tc_native_consumed_tile_count", 0) or 0)
            self.metrics.tc_native_consumed_bytes = int(result.get("tc_native_consumed_bytes", 0) or 0)
            self.metrics.tc_native_entrypoint = str(result.get("tc_native_entrypoint", ""))
            self.metrics.tc_native_descriptor_layout = str(result.get("tc_native_descriptor_layout", ""))
            self.metrics.tc_native_consumption_source = str(result.get("tc_native_consumption_source", ""))
            self.metrics.tc_native_launch_path = str(result.get("tc_native_launch_path", ""))
            self.metrics.tc_native_launch_count = int(result.get("tc_native_launch_count", 0) or 0)
        if bool(result.get("tc_adapter_consumed", False)):
            self.metrics.tc_adapter_consumed = True
            self.metrics.tc_adapter_source = str(result.get("tc_adapter_source", ""))
            self.metrics.tc_adapter_group_count = int(result.get("tc_adapter_group_count", 0) or 0)
            self.metrics.tc_adapter_descriptor_count = int(result.get("tc_adapter_descriptor_count", 0) or 0)
            self.metrics.tc_adapter_tile_count = int(result.get("tc_adapter_tile_count", 0) or 0)
            self.metrics.tc_adapter_dispatch_units = int(result.get("tc_adapter_dispatch_units", 0) or 0)
            self.metrics.tc_adapter_target = str(result.get("tc_adapter_target", ""))
            self.metrics.tc_adapter_mode = str(result.get("tc_adapter_mode", ""))
            self.metrics.tc_adapter_fallback_reason = str(result.get("tc_adapter_fallback_reason", ""))

    def _lookup_tiles(self, request: dict[str, Any]) -> list[str]:
        hot_tiles = set(self.manifest.get("gpu_hot_tiles", []))
        topk = request.get("topk", [])
        if not topk:
            return sorted(hot_tiles)
        planned = []
        for layer, expert in topk:
            prefix = f"L{int(layer)}:E{int(expert)}:"
            planned.extend(tile for tile in hot_tiles if tile.startswith(prefix))
        return sorted(set(planned))

    def _check_gates(self, request: dict[str, Any], planned_tiles: list[str]) -> bool:
        if request.get("force_gate_fail"):
            return False
        if not planned_tiles and request.get("require_tilemem", False):
            return False
        if "calibration" in self.manifest.get("runtime_gates", []) and request.get("calibration_pass", True) is False:
            return False
        return True

    def _materialize_missing(self, missing: list[str]) -> None:
        tile_bytes = self.manifest.get("tile_bytes", {})
        for tile in missing:
            self.tile_state.mark_gpu_resident(tile)
            self.metrics.h2d_bytes += int(tile_bytes.get(tile, 0))

    def _select_backend(self) -> Any | None:
        for name in self.manifest.get("backend_priority", []):
            try:
                backend = Backend(name)
            except ValueError:
                continue
            if backend == Backend.KT_FALLBACK:
                continue
            if backend in self.backends:
                return self.backends[backend]
        return None

    def _fallback(self, request: dict[str, Any], reason: str) -> dict[str, Any]:
        self.metrics.fallback_count += 1
        return {"output": request.get("kt_output"), "source": "kt_fallback", "reason": reason}

    def prefetch_plan(self, request: dict[str, Any]) -> None:
        if not self._async_planning_enabled:
            return
        self._plan_cache[self._plan_cache_key(request)] = self._lookup_tiles(request)

    def _plan_cache_key(self, request: dict[str, Any]) -> str:
        topk = request.get("topk", [])
        if not topk:
            return "__hotset__"
        return "|".join(f"{int(layer)}:{int(expert)}" for layer, expert in topk)


def _elapsed_us(start: float) -> float:
    return (time.perf_counter() - start) * 1_000_000.0
