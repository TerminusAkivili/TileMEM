from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any

from tilepo.integration import ScaleLayout, TileFormat


PUBLIC_MIR_INTERFACE = "tilemem_public_mir_v0_1_1"
MIR_SCHEMA_VERSION = "tilepo_mir_v1"


class TileDType(str, Enum):
    BF16 = "bf16"
    FP8 = "fp8"
    FP6 = "fp6"
    FP4 = "fp4"
    MXFP4 = "mxfp4"


class RuntimeMode(str, Enum):
    SHADOW = "shadow"
    VERIFY = "verify"
    SERVE = "serve"


class Backend(str, Enum):
    CUDA = "cuda"
    TILELANG = "tilelang"
    KT_FALLBACK = "kt_fallback"


class DeploymentMode(str, Enum):
    SPEED = "speed"
    MEMORY = "memory"
    BALANCED = "balanced"
    SAFE = "safe"


@dataclass(frozen=True, order=True)
class TileId:
    layer: int
    expert: int
    projection_group: str
    shard_id: int
    n_start: int
    n_end: int

    def __post_init__(self) -> None:
        if self.layer < 0 or self.expert < 0 or self.shard_id < 0:
            raise ValueError("tile id indices must be non-negative")
        if self.n_end <= self.n_start:
            raise ValueError("tile n_end must be greater than n_start")

    def stable_key(self) -> str:
        return (
            f"L{self.layer}:E{self.expert}:{self.projection_group}:"
            f"S{self.shard_id}:N{self.n_start}-{self.n_end}"
        )

    def checksum(self) -> str:
        return hashlib.sha256(self.stable_key().encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "projection_group": self.projection_group,
            "shard_id": self.shard_id,
            "n_start": self.n_start,
            "n_end": self.n_end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TileId":
        return cls(
            int(data["layer"]),
            int(data["expert"]),
            str(data["projection_group"]),
            int(data["shard_id"]),
            int(data["n_start"]),
            int(data["n_end"]),
        )


@dataclass(frozen=True)
class TileIR:
    tile_id: TileId
    dtype: TileDType
    bytes: int
    scale_bytes: int = 0
    format: TileFormat | None = None
    scale: ScaleLayout | None = None
    backend: str | None = None
    fallback_dtype: str | None = None
    fallback_backend: str | None = None

    def validate(self) -> None:
        if self.bytes <= 0:
            raise ValueError("tile bytes must be positive")
        if self.scale_bytes < 0:
            raise ValueError("scale bytes must be non-negative")
        if self.format is not None:
            self.format.validate()
        if self.scale is not None:
            self.scale.validate()
        if self.backend is not None and not self.backend:
            raise ValueError("tile backend must not be empty")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "tile_id": self.tile_id.to_dict(),
            "dtype": self.dtype.value,
            "bytes": self.bytes,
            "scale_bytes": self.scale_bytes,
        }
        if self.format is not None:
            data["format"] = self.format.to_dict()
        if self.scale is not None:
            data["scale"] = self.scale.to_dict()
        if self.backend is not None:
            data["backend"] = self.backend
        if self.fallback_dtype is not None:
            data["fallback_dtype"] = self.fallback_dtype
        if self.fallback_backend is not None:
            data["fallback_backend"] = self.fallback_backend
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TileIR":
        return cls(
            TileId.from_dict(data["tile_id"]),
            TileDType(data["dtype"]),
            int(data["bytes"]),
            int(data.get("scale_bytes", 0)),
            TileFormat.from_dict(data["format"]) if "format" in data else None,
            ScaleLayout.from_dict(data["scale"]) if "scale" in data else None,
            str(data["backend"]) if "backend" in data else None,
            str(data["fallback_dtype"]) if "fallback_dtype" in data else None,
            str(data["fallback_backend"]) if "fallback_backend" in data else None,
        )


@dataclass(frozen=True)
class RouteIR:
    workload: str
    hot_experts: dict[str, list[int]]

    def validate(self) -> None:
        if not self.workload:
            raise ValueError("route workload is required")
        for layer, experts in self.hot_experts.items():
            int(layer)
            if not experts:
                raise ValueError("hot expert list must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"workload": self.workload, "hot_experts": self.hot_experts}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteIR":
        return cls(str(data["workload"]), {str(k): [int(x) for x in v] for k, v in data["hot_experts"].items()})


@dataclass(frozen=True)
class ResidencyIR:
    gpu_cache_budget_gib: float
    cpu_cache_budget_gib: float
    gpu_hot_tiles: list[TileId]
    fallback_chain: list[TileDType | str]

    def validate(self) -> None:
        if self.gpu_cache_budget_gib < 0 or self.cpu_cache_budget_gib < 0:
            raise ValueError("memory budgets must be non-negative")
        if not self.fallback_chain:
            raise ValueError("fallback_chain must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_cache_budget_gib": self.gpu_cache_budget_gib,
            "cpu_cache_budget_gib": self.cpu_cache_budget_gib,
            "gpu_hot_tiles": [tile.to_dict() for tile in self.gpu_hot_tiles],
            "fallback_chain": [item.value if isinstance(item, TileDType) else item for item in self.fallback_chain],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResidencyIR":
        chain: list[TileDType | str] = []
        for item in data["fallback_chain"]:
            chain.append(TileDType(item) if item in {dtype.value for dtype in TileDType} else str(item))
        return cls(
            float(data["gpu_cache_budget_gib"]),
            float(data["cpu_cache_budget_gib"]),
            [TileId.from_dict(tile) for tile in data.get("gpu_hot_tiles", [])],
            chain,
        )


@dataclass(frozen=True)
class PrecisionIR:
    dtype_policy: str
    allowed: list[TileDType]
    calibration_required: bool

    def validate(self) -> None:
        if not self.allowed:
            raise ValueError("at least one dtype must be allowed")
        low_precision = {TileDType.FP8, TileDType.FP6, TileDType.FP4, TileDType.MXFP4}
        if any(dtype in self.allowed for dtype in low_precision) and not self.calibration_required:
            raise ValueError("low-precision tile formats require calibration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype_policy": self.dtype_policy,
            "allowed": [dtype.value for dtype in self.allowed],
            "calibration_required": self.calibration_required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrecisionIR":
        return cls(str(data["dtype_policy"]), [TileDType(x) for x in data["allowed"]], bool(data["calibration_required"]))


@dataclass(frozen=True)
class ScheduleIR:
    mode: RuntimeMode
    deployment_mode: DeploymentMode
    backend_priority: list[Backend]
    runtime_gates: list[str]
    prewarm_policy: str
    miss_policy: str

    def validate(self) -> None:
        if not self.backend_priority:
            raise ValueError("backend_priority must not be empty")
        if Backend.KT_FALLBACK not in self.backend_priority:
            raise ValueError("backend_priority must include kt_fallback")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "deployment_mode": self.deployment_mode.value,
            "backend_priority": [backend.value for backend in self.backend_priority],
            "runtime_gates": self.runtime_gates,
            "prewarm_policy": self.prewarm_policy,
            "miss_policy": self.miss_policy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduleIR":
        return cls(
            RuntimeMode(data["mode"]),
            DeploymentMode(data["deployment_mode"]),
            [Backend(x) for x in data["backend_priority"]],
            [str(x) for x in data.get("runtime_gates", [])],
            str(data["prewarm_policy"]),
            str(data["miss_policy"]),
        )


@dataclass(frozen=True)
class ModelIR:
    name: str
    layers: int
    experts_per_layer: int
    hidden_size: int
    intermediate_size: int
    routes: list[RouteIR]
    tiles: list[TileIR]
    residency: ResidencyIR
    precision: PrecisionIR
    schedule: ScheduleIR

    def validate(self) -> None:
        if self.layers <= 0 or self.experts_per_layer <= 0:
            raise ValueError("model layers and experts_per_layer must be positive")
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("model dimensions must be positive")
        if not self.tiles:
            raise ValueError("MIR must contain tiles")
        for route in self.routes:
            route.validate()
        for tile in self.tiles:
            tile.validate()
        self.residency.validate()
        self.precision.validate()
        self.schedule.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MIR_SCHEMA_VERSION,
            "public_interface": PUBLIC_MIR_INTERFACE,
            "name": self.name,
            "layers": self.layers,
            "experts_per_layer": self.experts_per_layer,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "routes": [route.to_dict() for route in self.routes],
            "tiles": [tile.to_dict() for tile in self.tiles],
            "residency": self.residency.to_dict(),
            "precision": self.precision.to_dict(),
            "schedule": self.schedule.to_dict(),
        }


def model_from_dict(data: dict[str, Any]) -> ModelIR:
    model = ModelIR(
        name=str(data["name"]),
        layers=int(data["layers"]),
        experts_per_layer=int(data["experts_per_layer"]),
        hidden_size=int(data["hidden_size"]),
        intermediate_size=int(data["intermediate_size"]),
        routes=[RouteIR.from_dict(route) for route in data.get("routes", [])],
        tiles=[TileIR.from_dict(tile) for tile in data["tiles"]],
        residency=ResidencyIR.from_dict(data["residency"]),
        precision=PrecisionIR.from_dict(data["precision"]),
        schedule=ScheduleIR.from_dict(data["schedule"]),
    )
    return model


def build_manifest(model: ModelIR) -> dict[str, Any]:
    model.validate()
    tiles = sorted(model.tiles, key=lambda tile: tile.tile_id)
    tile_offsets: dict[str, int] = {}
    tile_bytes: dict[str, int] = {}
    tile_dtype_map: dict[str, str] = {}
    tile_ids: dict[str, dict[str, Any]] = {}
    tile_format_map: dict[str, dict[str, Any]] = {}
    scale_offsets: dict[str, int] = {}
    scale_bytes: dict[str, int] = {}
    scale_layout_map: dict[str, str] = {}
    backend_owner_map: dict[str, str] = {}
    tile_fallback_map: dict[str, dict[str, str]] = {}
    offset = 0
    scale_offset = 0
    for tile in tiles:
        key = tile.tile_id.stable_key()
        tile_offsets[key] = offset
        tile_bytes[key] = tile.bytes
        tile_dtype_map[key] = tile.dtype.value
        tile_ids[key] = tile.tile_id.to_dict()
        if tile.format is not None:
            tile_format_map[key] = tile.format.to_dict()
        if tile.scale is not None:
            scale_layout_map[key] = tile.scale.layout
        if tile.backend is not None:
            backend_owner_map[key] = tile.backend
        if tile.fallback_dtype is not None or tile.fallback_backend is not None:
            tile_fallback_map[key] = {
                "dtype": tile.fallback_dtype or "bf16",
                "backend": tile.fallback_backend or "kt_fallback",
            }
        offset += tile.bytes
        if tile.scale_bytes:
            scale_offsets[key] = scale_offset
            scale_bytes[key] = tile.scale_bytes
            scale_offset += tile.scale_bytes
    manifest = {
        "schema_version": "tilepo_manifest_v1",
        "tile_ids": tile_ids,
        "tile_offsets": tile_offsets,
        "tile_bytes": tile_bytes,
        "tile_dtype_map": tile_dtype_map,
        "tile_format_map": tile_format_map,
        "scale_offsets": scale_offsets,
        "scale_bytes": scale_bytes,
        "scale_layout_map": scale_layout_map,
        "backend_owner_map": backend_owner_map,
        "tile_fallback_map": tile_fallback_map,
        "gpu_hot_tiles": [tile.stable_key() for tile in sorted(model.residency.gpu_hot_tiles)],
        "fallback_chain": [item.value if isinstance(item, TileDType) else item for item in model.residency.fallback_chain],
        "backend_priority": [backend.value for backend in model.schedule.backend_priority],
        "runtime_gates": model.schedule.runtime_gates,
        "mode": model.schedule.mode.value,
        "deployment_mode": model.schedule.deployment_mode.value,
        "model": model.name,
    }
    manifest["checksum"] = _checksum(manifest)
    return manifest


def _checksum(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
