from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tilepo import env
from tilepo.integration import (
    BackendCapability,
    BackendRegistry,
    ScaleLayout,
    TileFormat,
    TileHandle,
    backend_registry,
    benchmark_dispatch_plan,
    build_tile_handles,
    register_backend,
)
from tilepo.mir import (
    Backend,
    DeploymentMode,
    MIR_SCHEMA_VERSION,
    ModelIR,
    PUBLIC_MIR_INTERFACE,
    PrecisionIR,
    ResidencyIR,
    RouteIR,
    RuntimeMode,
    ScheduleIR,
    TileDType,
    TileIR,
    TileId,
    build_manifest,
    load_mir,
    model_from_dict,
    save_mir,
    validate_mir_dict,
)
from tilepo.model_interface import (
    MODEL_SPEC_SCHEMA_VERSION,
    ModelAdapter,
    ModelSpec,
    build_mir_from_model_spec,
    model_spec_from_dict,
    model_spec_to_dict,
)
from .checkpoint import (
    CheckpointArtifact,
    MoETopology,
    ServingCommand,
    ServingResult,
    WeightMatchResult,
    build_runtime_weight_aliases,
    build_serving_command,
    build_tile_checkpoint_map,
    checkpoint_weight_names,
    export_checkpoint_artifact,
    infer_moe_topology,
    load_checkpoint_weight_map,
    load_hf_config,
    match_checkpoint_weights,
    model_spec_from_hf_config,
    plan_from_hf_config,
    run_serving_backend,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V0_1_SUMMARY = Path("evidence") / "ablation" / "tilepo_ablation_summary.json"


@dataclass(frozen=True)
class TileMEMPlan:
    mir: ModelIR
    manifest: dict[str, Any]
    handles: list[TileHandle]

    def dispatch_summary(self, *, iterations: int = 1) -> dict[str, Any]:
        return benchmark_dispatch_plan(self.handles, iterations=iterations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mir": self.mir.to_dict(),
            "manifest": self.manifest,
            "handles": [handle.to_dict() for handle in self.handles],
        }


def model_spec(
    *,
    name: str,
    layers: int,
    experts_per_layer: int,
    hidden_size: int,
    intermediate_size: int,
    expert_budget: int | None = None,
    workload: str = "mixed",
    tile: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
    precision: dict[str, Any] | None = None,
    schedule: dict[str, Any] | None = None,
) -> ModelSpec:
    return model_spec_from_dict(
        {
            "schema_version": MODEL_SPEC_SCHEMA_VERSION,
            "name": name,
            "layers": layers,
            "experts_per_layer": experts_per_layer,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "expert_budget": expert_budget if expert_budget is not None else experts_per_layer,
            "workload": workload,
            "tile": {
                "hidden_tile": hidden_size,
                "intermediate_tile": intermediate_size,
                "shard_count": 1,
                "projection_groups": ["gate_up", "down"],
                **(tile or {}),
            },
            "memory": {
                "gpu_cache_budget_gib": 0.0,
                "cpu_cache_budget_gib": 0.0,
                **(memory or {}),
            },
            "precision": {
                "dtype_policy": "bf16",
                "allowed": ["bf16"],
                "calibration_required": False,
                **(precision or {}),
            },
            "schedule": {
                "mode": "verify",
                "deployment_mode": "balanced",
                "backend_priority": ["cuda", "tilelang", "kt_fallback"],
                "runtime_gates": [],
                "prewarm_policy": "hotset",
                "miss_policy": "fallback",
                **(schedule or {}),
            },
        }
    )


def build_mir(spec_or_adapter: ModelSpec | ModelAdapter | dict[str, Any]) -> ModelIR:
    return build_mir_from_model_spec(spec_or_adapter)


def plan(
    spec_or_adapter: ModelSpec | ModelAdapter | dict[str, Any],
    *,
    registry: BackendRegistry | None = None,
) -> TileMEMPlan:
    mir = build_mir(spec_or_adapter)
    manifest = build_manifest(mir)
    handles = build_tile_handles(manifest, registry=registry)
    return TileMEMPlan(mir=mir, manifest=manifest, handles=handles)


def v0_1_headline_gain(summary_path: Path | str | None = None) -> dict[str, Any]:
    summary = _load_v0_1_summary(summary_path)
    rows = _v0_1_gain_rows(summary)
    best = max(rows, key=lambda item: item["tok_gain_pct"])
    return {
        "schema_version": "tilemem_v0_1_headline_gain_v1",
        "source_summary": str(_default_summary_path(summary_path)),
        "gate": dict(summary.get("gate", {})),
        "requested": dict(summary.get("requested", {})),
        "best": best,
        "top": rows[:10],
    }


def summarize_v0_1_evidence(summary_path: Path | str | None = None) -> dict[str, Any]:
    summary = _load_v0_1_summary(summary_path)
    rows = _v0_1_gain_rows(summary)
    return {
        "schema_version": "tilemem_v0_1_evidence_summary_v1",
        "source_summary": str(_default_summary_path(summary_path)),
        "gate": dict(summary.get("gate", {})),
        "requested": dict(summary.get("requested", {})),
        "groups": len(summary.get("groups", [])),
        "best_tilepo_rows": rows,
    }


def _default_summary_path(summary_path: Path | str | None) -> Path:
    path = Path(summary_path) if summary_path is not None else DEFAULT_V0_1_SUMMARY
    if path.is_absolute() and path.exists():
        return path
    for root in _artifact_roots():
        candidate = root / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"TileMEM V0.1 summary not found: {path}")


def _artifact_root() -> Path:
    return _artifact_roots()[0]


def _artifact_roots() -> list[Path]:
    roots = []
    env_root = env.tilemem_repo_root()
    if env_root:
        roots.append(env_root)
    roots.extend([ROOT, Path.cwd()])
    roots.extend(Path.cwd().parents)

    deduped = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _load_v0_1_summary(summary_path: Path | str | None) -> dict[str, Any]:
    path = _default_summary_path(summary_path)
    data = json.loads(path.read_text())
    if data.get("schema_version") != "tilepo_ablation_report_v1":
        raise ValueError(f"unsupported V0.1 summary schema: {path}")
    if data.get("gate", {}).get("status") != "PASS":
        raise ValueError(f"V0.1 evidence gate is not PASS: {path}")
    return data


def _v0_1_gain_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in summary.get("groups", []):
        grouped[(str(group["workload"]), int(group["experts_per_layer"]))].append(group)

    rows: list[dict[str, Any]] = []
    for (workload, experts_per_layer), groups in sorted(grouped.items()):
        kt_rows = [group for group in groups if group.get("policy") == "kt_expert"]
        if len(kt_rows) != 1:
            raise ValueError(f"expected one KT row for {workload}/{experts_per_layer}, got {len(kt_rows)}")
        kt = kt_rows[0]
        kt_tok = _median_metric(kt, "tok_per_sec")
        kt_p95 = _median_metric(kt, "p95_ms")
        candidates = []
        for group in groups:
            if group.get("policy") == "kt_expert":
                continue
            tok = _median_metric(group, "tok_per_sec")
            p95 = _median_metric(group, "p95_ms")
            candidates.append(
                {
                    "workload": workload,
                    "experts_per_layer": experts_per_layer,
                    "policy": str(group["policy"]),
                    "async_planning": str(group.get("async_planning", "off")),
                    "system": str(group["system"]),
                    "tok_per_sec": round(tok, 6),
                    "kt_tok_per_sec": round(kt_tok, 6),
                    "tok_gain_pct": round((tok / kt_tok - 1.0) * 100.0, 4),
                    "p95_ms": round(p95, 6),
                    "kt_p95_ms": round(kt_p95, 6),
                    "p95_reduction_pct": round((1.0 - p95 / kt_p95) * 100.0, 4),
                }
            )
        if not candidates:
            continue
        rows.append(max(candidates, key=lambda item: item["tok_gain_pct"]))
    return sorted(rows, key=lambda item: item["tok_gain_pct"], reverse=True)


def _median_metric(group: dict[str, Any], metric: str) -> float:
    return float(group["metrics"][metric]["median"])
