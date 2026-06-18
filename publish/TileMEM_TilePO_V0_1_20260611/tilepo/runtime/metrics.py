from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeMetrics:
    plan_lookup_us: float = 0.0
    plan_lookup_total_us: float = 0.0
    gate_us: float = 0.0
    backend_launch_us: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    h2d_bytes: int = 0
    fallback_count: int = 0
    tilemem_backend_launch_count: int = 0
    tilelang_launch_count: int = 0
    cuda_launch_count: int = 0
    native_cuda_available: bool = False
    native_cuda_launch_count: int = 0
    cuda_python_shim_launch_count: int = 0
    cuda_descriptor_traversal_us: float = 0.0
    cuda_descriptor_metrics_measured: bool = False
    tc_native_consumed: bool = False
    tc_native_consumed_coalesced_groups: bool = False
    tc_native_consumed_group_count: int = 0
    tc_native_descriptor_count: int = 0
    tc_native_consumed_tile_count: int = 0
    tc_native_consumed_bytes: int = 0
    tc_native_entrypoint: str = ""
    tc_native_descriptor_layout: str = ""
    tc_native_consumption_source: str = ""
    tc_native_launch_path: str = ""
    tc_native_launch_count: int = 0
    tc_adapter_consumed: bool = False
    tc_adapter_source: str = ""
    tc_adapter_group_count: int = 0
    tc_adapter_descriptor_count: int = 0
    tc_adapter_tile_count: int = 0
    tc_adapter_dispatch_units: int = 0
    tc_adapter_target: str = ""
    tc_adapter_mode: str = ""
    tc_adapter_fallback_reason: str = ""
    dtype_counts: dict[str, int] = field(default_factory=dict)
    ablation_policy: str = ""
    async_planning_mode: str = ""
    tile_count: int = 0
    async_plan_cache_hits: int = 0
    async_plan_cache_misses: int = 0
    coalesced_group_count: int = 0
    execution_dispatch_units: int = 0
    baa_double_buffered: bool = False
    baa_critical_path_us: float = 0.0
    baa_metrics_measured: bool = False
    baa_active_map_id: str = ""
    baa_standby_ready: bool = False
    baa_map_swaps: int = 0

    def record_dtype(self, dtype: str, count: int = 1) -> None:
        self.dtype_counts[dtype] = self.dtype_counts.get(dtype, 0) + count

    def snapshot(self) -> dict[str, object]:
        return {
            "plan_lookup_us": self.plan_lookup_us,
            "plan_lookup_total_us": self.plan_lookup_total_us,
            "gate_us": self.gate_us,
            "backend_launch_us": self.backend_launch_us,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "h2d_bytes": self.h2d_bytes,
            "fallback_count": self.fallback_count,
            "tilemem_backend_launch_count": self.tilemem_backend_launch_count,
            "tilelang_launch_count": self.tilelang_launch_count,
            "cuda_launch_count": self.cuda_launch_count,
            "native_cuda_available": self.native_cuda_available,
            "native_cuda_launch_count": self.native_cuda_launch_count,
            "cuda_python_shim_launch_count": self.cuda_python_shim_launch_count,
            "cuda_descriptor_traversal_us": self.cuda_descriptor_traversal_us,
            "cuda_descriptor_metrics_measured": self.cuda_descriptor_metrics_measured,
            "tc_native_consumed": self.tc_native_consumed,
            "tc_native_consumed_coalesced_groups": self.tc_native_consumed_coalesced_groups,
            "tc_native_consumed_group_count": self.tc_native_consumed_group_count,
            "tc_native_descriptor_count": self.tc_native_descriptor_count,
            "tc_native_consumed_tile_count": self.tc_native_consumed_tile_count,
            "tc_native_consumed_bytes": self.tc_native_consumed_bytes,
            "tc_native_entrypoint": self.tc_native_entrypoint,
            "tc_native_descriptor_layout": self.tc_native_descriptor_layout,
            "tc_native_consumption_source": self.tc_native_consumption_source,
            "tc_native_launch_path": self.tc_native_launch_path,
            "tc_native_launch_count": self.tc_native_launch_count,
            "tc_adapter_consumed": self.tc_adapter_consumed,
            "tc_adapter_source": self.tc_adapter_source,
            "tc_adapter_group_count": self.tc_adapter_group_count,
            "tc_adapter_descriptor_count": self.tc_adapter_descriptor_count,
            "tc_adapter_tile_count": self.tc_adapter_tile_count,
            "tc_adapter_dispatch_units": self.tc_adapter_dispatch_units,
            "tc_adapter_target": self.tc_adapter_target,
            "tc_adapter_mode": self.tc_adapter_mode,
            "tc_adapter_fallback_reason": self.tc_adapter_fallback_reason,
            "dtype_counts": dict(sorted(self.dtype_counts.items())),
            "ablation_policy": self.ablation_policy,
            "async_planning_mode": self.async_planning_mode,
            "tile_count": self.tile_count,
            "async_plan_cache_hits": self.async_plan_cache_hits,
            "async_plan_cache_misses": self.async_plan_cache_misses,
            "coalesced_group_count": self.coalesced_group_count,
            "execution_dispatch_units": self.execution_dispatch_units,
            "baa_double_buffered": self.baa_double_buffered,
            "baa_critical_path_us": self.baa_critical_path_us,
            "baa_metrics_measured": self.baa_metrics_measured,
            "baa_active_map_id": self.baa_active_map_id,
            "baa_standby_ready": self.baa_standby_ready,
            "baa_map_swaps": self.baa_map_swaps,
        }
