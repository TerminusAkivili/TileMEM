from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from tilepo import env
from tilepo.backends.cuda_backend import CUDABackend
from tilepo.backends.tilelang_backend import TileLangBackend
from tilepo.kt_patch.sglang_hook import configure_sglang_hook_runtime, install_sglang_hook
from tilepo.mir import Backend, RuntimeMode
from tilepo.runtime import TileMEMRuntime


def bootstrap_from_env() -> dict[str, Any]:
    manifest_path = env.manifest_path()
    mode = env.mode()
    backend = env.backend_priority()
    if not manifest_path.exists():
        raise FileNotFoundError(f"{env.TILEPO_MANIFEST} does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if "checksum" not in manifest:
        raise ValueError("TilePO manifest missing checksum")
    state = {
        "enabled": True,
        "mode": mode,
        "backend": backend,
        "manifest_path": str(manifest_path),
        "manifest_checksum": manifest["checksum"],
        "run_id": env.run_id(),
        "tilepo_policy": _ablation_policy(manifest),
        "tilepo_async_planning": _ablation_async(manifest),
        "tilepo_tile_count": len(manifest.get("tile_offsets", {})),
    }
    state["hot_backend_probe"] = _probe_hot_backend(manifest, mode, backend)
    configure_sglang_hook_runtime(
        _build_runtime(manifest, mode, backend),
        _hot_backend_probe_request(manifest),
        max_launches=env.hook_backend_probe_limit(),
    )
    state["serving_hook"] = install_sglang_hook()
    marker = env.bootstrap_marker_path()
    if marker:
        _write_bootstrap_marker(marker, state)
    env.mark_bootstrapped(str(manifest["checksum"]))
    return state


def _probe_hot_backend(manifest: dict[str, Any], mode: str, backend_text: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        runtime = _build_runtime(manifest, mode, backend_text)
        request = _hot_backend_probe_request(manifest)
        if _ablation_async(manifest) == "on":
            runtime.prefetch_plan(request)
        result = runtime.execute(request)
        metrics = runtime.metrics.snapshot()
        return {
            "status": "success",
            "result_backend": result.get("backend", result.get("source", "")),
            "result_hot_tile_backend": bool(result.get("hot_tile_backend", False)),
            "serving_hook_active": False,
            "runtime_overhead_us": (time.perf_counter() - start) * 1_000_000.0,
            "plan_lookup_us": metrics["plan_lookup_us"],
            "plan_lookup_total_us": metrics["plan_lookup_total_us"],
            "gate_us": metrics["gate_us"],
            "backend_launch_us": metrics["backend_launch_us"],
            "dtype_counts": metrics["dtype_counts"],
            "fallback_count": metrics["fallback_count"],
            "backend_launch_counts": {
                "cuda": metrics["cuda_launch_count"],
                "tilelang": metrics["tilelang_launch_count"],
            },
            "tilemem_backend_launch_count": metrics["tilemem_backend_launch_count"],
            "h2d_bytes": metrics["h2d_bytes"],
            "cache_hits": metrics["cache_hits"],
            "cache_misses": metrics["cache_misses"],
            "hot_backend_native": _hot_backend_native(runtime.backends),
            "ablation_policy": metrics.get("ablation_policy", _ablation_policy(manifest)),
            "async_planning_mode": metrics.get("async_planning_mode", _ablation_async(manifest)),
            "tile_count": metrics.get("tile_count", len(manifest.get("tile_offsets", {}))),
            "async_plan_cache_hits": metrics.get("async_plan_cache_hits", 0),
            "async_plan_cache_misses": metrics.get("async_plan_cache_misses", 0),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "failure_reason": str(exc),
            "runtime_overhead_us": (time.perf_counter() - start) * 1_000_000.0,
            "dtype_counts": {},
            "fallback_count": 1,
            "backend_launch_counts": {},
            "hot_backend_native": False,
        }


def _build_backends(backend_names: list[str]) -> dict[Backend, Any]:
    backends: dict[Backend, Any] = {}
    for name in backend_names:
        try:
            backend = Backend(name)
        except ValueError:
            continue
        if backend == Backend.CUDA and backend not in backends:
            backends[backend] = CUDABackend(require_native=env.require_native_backend())
        elif backend == Backend.TILELANG and backend not in backends:
            backends[backend] = TileLangBackend()
    return backends


def _build_runtime(manifest: dict[str, Any], mode: str, backend_text: str) -> TileMEMRuntime:
    backend_names = [item.strip() for item in backend_text.split(",") if item.strip()]
    if not backend_names:
        backend_names = [str(item) for item in manifest.get("backend_priority", [])]
    runtime_mode = RuntimeMode(mode[:-5] if mode.endswith("_mode") else mode)
    return TileMEMRuntime(manifest, _build_backends(backend_names), mode=runtime_mode)


def _hot_backend_probe_request(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "op": "moe",
        "topk": _first_hot_tile_topk(manifest),
        "require_tilemem": True,
        "calibration_pass": True,
        "dtype": _first_hot_tile_dtype(manifest),
        "hidden": [0.5, -0.25],
        "gate_up": [
            [[0.2, 0.1, 0.3, -0.2], [0.4, -0.3, 0.1, 0.5]],
            [[-0.1, 0.2, 0.2, 0.1], [0.3, 0.4, -0.2, 0.3]],
        ],
        "down": [
            [[0.7, -0.1], [0.2, 0.3]],
            [[-0.4, 0.5], [0.6, -0.2]],
        ],
        "expert_ids": [0, 1],
        "router_scores": [0.6, 0.4],
    }


def _first_hot_tile_topk(manifest: dict[str, Any]) -> list[tuple[int, int]]:
    hot_tiles = manifest.get("gpu_hot_tiles", [])
    if not hot_tiles:
        return []
    parts = str(hot_tiles[0]).split(":")
    if len(parts) < 2:
        return []
    try:
        layer = int(parts[0].removeprefix("L"))
        expert = int(parts[1].removeprefix("E"))
    except ValueError:
        return []
    return [(layer, expert)]


def _first_hot_tile_dtype(manifest: dict[str, Any]) -> str:
    hot_tiles = manifest.get("gpu_hot_tiles", [])
    dtype_map = manifest.get("tile_dtype_map", {})
    if hot_tiles:
        return str(dtype_map.get(str(hot_tiles[0]), "bf16"))
    return "bf16"


def _hot_backend_native(backends: dict[Backend, Any]) -> bool:
    backend = backends.get(Backend.CUDA)
    return bool(getattr(backend, "native_available", False))


def _ablation_policy(manifest: dict[str, Any]) -> str:
    ablation = manifest.get("tilepo_plan", {})
    if isinstance(ablation, dict) and ablation.get("policy"):
        return str(ablation["policy"])
    return env.policy(str(manifest.get("tilepo_policy", "")))


def _ablation_async(manifest: dict[str, Any]) -> str:
    if manifest.get("tilepo_async_planning"):
        return str(manifest["tilepo_async_planning"])
    ablation = manifest.get("tilepo_plan", {})
    if isinstance(ablation, dict) and "async_planning" in ablation:
        return "on" if bool(ablation["async_planning"]) else "off"
    return env.async_planning()


def _write_bootstrap_marker(path: Path, state: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {}
    existing_run_id = str(existing.get("run_id", ""))
    state_run_id = str(state.get("run_id", ""))
    if existing_run_id and state_run_id and existing_run_id != state_run_id:
        existing = {}
    merged = {**existing, **state}
    existing_hook = existing.get("serving_hook")
    state_hook = state.get("serving_hook")
    if isinstance(existing_hook, dict) and isinstance(state_hook, dict):
        merged["serving_hook"] = {**state_hook, **existing_hook}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
