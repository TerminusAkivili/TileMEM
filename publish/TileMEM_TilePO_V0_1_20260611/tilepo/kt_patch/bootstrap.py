from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from tilepo import env
from tilepo.backends.cuda_backend import CUDABackend
from tilepo.backends.tilelang_backend import TileLangBackend
from tilepo.kt_patch.sglang_hook import (
    configure_sglang_hook_runtime,
    install_sglang_hook,
    prime_sglang_hook_runtime,
)
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
        "backend_owner": "kt_sglang",
        "kt_executor_preserved": True,
        "tilepo_plan_applied_in_serving_path": True,
    }
    state["hot_backend_probe"] = _probe_hot_backend(manifest, mode, backend)
    state.update(_kt_preserving_manifest_metadata(manifest, state["hot_backend_probe"]))
    configure_sglang_hook_runtime(
        _build_runtime(manifest, mode, backend),
        _hot_backend_probe_request(manifest),
        max_launches=env.hook_backend_probe_limit(),
    )
    state["serving_hook"] = install_sglang_hook()
    if _serving_hook_installed(state["serving_hook"]):
        state["serving_hook_backend_prime"] = prime_sglang_hook_runtime()
        state["serving_hook"] = _merge_native_tc_prime_into_hook(
            state["serving_hook"],
            state["serving_hook_backend_prime"],
        )
    marker = env.bootstrap_marker_path()
    if marker:
        _write_bootstrap_marker(marker, state)
    if _serving_hook_installed(state["serving_hook"]):
        env.mark_bootstrapped(str(manifest["checksum"]))
    return state


def _kt_preserving_manifest_metadata(manifest: dict[str, Any], hot_probe: dict[str, Any]) -> dict[str, Any]:
    plan = manifest.get("tilepo_plan", {})
    if not isinstance(plan, dict):
        plan = {}
    tile_count = int(plan.get("tile_count", len(manifest.get("tile_offsets", {}))) or 0)
    dispatch_units = int(plan.get("execution_dispatch_units", plan.get("estimated_dispatch_units", 0)) or 0)
    coalesced_groups = int(plan.get("coalesced_group_count", hot_probe.get("coalesced_group_count", 0)) or 0)
    return {
        "tc_coalescing_active": bool(coalesced_groups > 0 or (tile_count > 0 and 0 < dispatch_units < tile_count)),
        "baa_double_buffered": bool(plan.get("baa_double_buffered", hot_probe.get("baa_double_buffered", False))),
        "baa_critical_path_us": float(hot_probe.get("baa_critical_path_us", 0.0)),
        "baa_metrics_measured": bool(hot_probe.get("baa_metrics_measured", False)),
        "cuda_descriptor_traversal_us": float(hot_probe.get("cuda_descriptor_traversal_us", 0.0)),
        "cuda_descriptor_metrics_measured": bool(hot_probe.get("cuda_descriptor_metrics_measured", False)),
        "cuda_launch_count": int(hot_probe.get("cuda_launch_count", 0) or 0),
        "tc_native_consumed": bool(hot_probe.get("tc_native_consumed", False)),
        "tc_native_consumed_coalesced_groups": bool(
            hot_probe.get("tc_native_consumed_coalesced_groups", False)
        ),
        "tc_native_consumed_group_count": int(hot_probe.get("tc_native_consumed_group_count", 0) or 0),
        "tc_native_descriptor_count": int(hot_probe.get("tc_native_descriptor_count", 0) or 0),
        "tc_native_consumed_tile_count": int(hot_probe.get("tc_native_consumed_tile_count", 0) or 0),
        "tc_native_consumed_bytes": int(hot_probe.get("tc_native_consumed_bytes", 0) or 0),
        "tc_native_entrypoint": str(hot_probe.get("tc_native_entrypoint", "")),
        "tc_native_descriptor_layout": str(hot_probe.get("tc_native_descriptor_layout", "")),
        "tc_native_consumption_source": str(hot_probe.get("tc_native_consumption_source", "")),
        "tc_native_launch_path": str(hot_probe.get("tc_native_launch_path", "")),
        "tc_native_launch_count": int(hot_probe.get("tc_native_launch_count", 0) or 0),
        "tc_adapter_consumed": bool(hot_probe.get("tc_adapter_consumed", False)),
        "tc_adapter_source": str(hot_probe.get("tc_adapter_source", "")),
        "tc_adapter_group_count": int(hot_probe.get("tc_adapter_group_count", 0) or 0),
        "tc_adapter_descriptor_count": int(hot_probe.get("tc_adapter_descriptor_count", 0) or 0),
        "tc_adapter_tile_count": int(hot_probe.get("tc_adapter_tile_count", 0) or 0),
        "tc_adapter_dispatch_units": int(hot_probe.get("tc_adapter_dispatch_units", 0) or 0),
        "tc_adapter_target": str(hot_probe.get("tc_adapter_target", "")),
        "tc_adapter_mode": str(hot_probe.get("tc_adapter_mode", "")),
        "tc_adapter_fallback_reason": str(hot_probe.get("tc_adapter_fallback_reason", "")),
        "runtime_metrics_source": "hot_backend_probe",
    }


def _merge_native_tc_prime_into_hook(hook_state: Any, prime: Any) -> dict[str, Any]:
    hook = dict(hook_state) if isinstance(hook_state, dict) else {}
    if not isinstance(prime, dict):
        return hook
    tc_consumed = bool(prime.get("tc_native_consumed", False))
    tc_groups = bool(prime.get("tc_native_consumed_coalesced_groups", False))
    descriptor_count = int(prime.get("tc_native_descriptor_count", 0) or 0)
    native_ready = tc_consumed and tc_groups and descriptor_count > 0
    hook.update(
        {
            "tc_native_consumed": tc_consumed,
            "tc_native_consumed_coalesced_groups": tc_groups,
            "tc_native_descriptor_count": descriptor_count,
            "tc_native_entrypoint": str(prime.get("tc_native_entrypoint", "")),
            "tc_native_descriptor_layout": str(prime.get("tc_native_descriptor_layout", "")),
            "tc_native_consumed_group_count": int(prime.get("tc_native_consumed_group_count", 0) or 0),
            "tc_native_consumed_tile_count": int(prime.get("tc_native_consumed_tile_count", 0) or 0),
            "tc_native_consumed_bytes": int(prime.get("tc_native_consumed_bytes", 0) or 0),
            "tc_native_consumption_source": str(prime.get("tc_native_consumption_source", "")),
            "serving_hook_backend_prime_native_ready": native_ready,
            "serving_hook_replaced_count": int(hook.get("serving_hook_replaced_count", 0) or 0),
            "serving_hook_returned_original": bool(hook.get("serving_hook_returned_original", True)),
            "serving_hook_mode": str(hook.get("serving_hook_mode", "observe_only")),
            "serving_hook_replacement_real": bool(hook.get("serving_hook_replacement_real", False)),
        }
    )
    return hook


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
            "native_cuda_available": metrics.get("native_cuda_available", False),
            "native_cuda_launch_count": metrics.get("native_cuda_launch_count", 0),
            "cuda_python_shim_launch_count": metrics.get("cuda_python_shim_launch_count", 0),
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
            "coalesced_group_count": metrics.get("coalesced_group_count", 0),
            "execution_dispatch_units": metrics.get("execution_dispatch_units", 0),
            "baa_double_buffered": metrics.get("baa_double_buffered", False),
            "baa_critical_path_us": metrics.get("baa_critical_path_us", 0.0),
            "baa_metrics_measured": metrics.get("baa_metrics_measured", False),
            "baa_active_map_id": metrics.get("baa_active_map_id", ""),
            "baa_standby_ready": metrics.get("baa_standby_ready", False),
            "cuda_launch_count": metrics["cuda_launch_count"],
            "cuda_descriptor_traversal_us": metrics.get("cuda_descriptor_traversal_us", 0.0),
            "cuda_descriptor_metrics_measured": metrics.get("cuda_descriptor_metrics_measured", False),
            "tc_native_consumed": metrics.get("tc_native_consumed", False),
            "tc_native_consumed_coalesced_groups": metrics.get("tc_native_consumed_coalesced_groups", False),
            "tc_native_consumed_group_count": metrics.get("tc_native_consumed_group_count", 0),
            "tc_native_descriptor_count": metrics.get("tc_native_descriptor_count", 0),
            "tc_native_consumed_tile_count": metrics.get("tc_native_consumed_tile_count", 0),
            "tc_native_consumed_bytes": metrics.get("tc_native_consumed_bytes", 0),
            "tc_native_entrypoint": metrics.get("tc_native_entrypoint", ""),
            "tc_native_descriptor_layout": metrics.get("tc_native_descriptor_layout", ""),
            "tc_native_consumption_source": metrics.get("tc_native_consumption_source", ""),
            "tc_native_launch_path": metrics.get("tc_native_launch_path", ""),
            "tc_native_launch_count": metrics.get("tc_native_launch_count", 0),
            "tc_adapter_consumed": metrics.get("tc_adapter_consumed", False),
            "tc_adapter_source": metrics.get("tc_adapter_source", ""),
            "tc_adapter_group_count": metrics.get("tc_adapter_group_count", 0),
            "tc_adapter_descriptor_count": metrics.get("tc_adapter_descriptor_count", 0),
            "tc_adapter_tile_count": metrics.get("tc_adapter_tile_count", 0),
            "tc_adapter_dispatch_units": metrics.get("tc_adapter_dispatch_units", 0),
            "tc_adapter_target": metrics.get("tc_adapter_target", ""),
            "tc_adapter_mode": metrics.get("tc_adapter_mode", ""),
            "tc_adapter_fallback_reason": metrics.get("tc_adapter_fallback_reason", ""),
            "runtime_metrics_source": "hot_backend_probe",
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
    return bool(getattr(backend, "native_available", False) or getattr(backend, "tc_native_launch_count", 0))


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
        merged["serving_hook"] = {**existing_hook, **state_hook}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


def _serving_hook_installed(state: Any) -> bool:
    return isinstance(state, dict) and bool(state.get("installed_targets"))
