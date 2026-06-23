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
    dtype_counts: dict[str, int] = field(default_factory=dict)
    ablation_policy: str = ""
    async_planning_mode: str = ""
    tile_count: int = 0
    async_plan_cache_hits: int = 0
    async_plan_cache_misses: int = 0

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
            "dtype_counts": dict(sorted(self.dtype_counts.items())),
            "ablation_policy": self.ablation_policy,
            "async_planning_mode": self.async_planning_mode,
            "tile_count": self.tile_count,
            "async_plan_cache_hits": self.async_plan_cache_hits,
            "async_plan_cache_misses": self.async_plan_cache_misses,
        }
