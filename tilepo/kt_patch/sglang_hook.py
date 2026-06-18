from __future__ import annotations

import importlib
import json
from pathlib import Path
import atexit
import time
from typing import Any, Callable

from tilepo import env

_TARGETS = (
    (
        "sglang.srt.layers.moe.fused_moe_triton.layer",
        "FusedMoE",
        "run_moe_core",
        "FusedMoE.run_moe_core",
        "dispatch_output",
    ),
    (
        "sglang.srt.layers.moe.fused_moe_triton.layer",
        "FusedMoE",
        "forward",
        "FusedMoE.forward",
        "topk_output",
    ),
    (
        "sglang.srt.layers.moe.kt_ep_wrapper",
        "KTEPWrapperMethod",
        "apply",
        "KTEPWrapperMethod.apply",
        "dispatch_output",
    ),
    (
        "sglang.srt.models.olmoe",
        "OlmoeMoE",
        "forward",
        "OlmoeMoE.forward",
        "hidden_states",
    ),
)

_INSTALLED = False
_PATCHES: list[tuple[type[Any], str, str, str, Callable[..., Any]]] = []
_HOOK_RUNTIME: Any | None = None
_HOOK_REQUEST: dict[str, Any] = {}
_HOOK_MAX_LAUNCHES = 0
_HOOK_LAUNCHES = 0
_HOOK_METRICS: dict[str, Any] = {}
_HOOK_BACKEND_EVIDENCE: dict[str, Any] = {}
_HOOK_DIRTY = False
_ATEXIT_REGISTERED = False


def configure_sglang_hook_runtime(
    runtime: Any,
    request: dict[str, Any],
    max_launches: int = 1,
) -> None:
    global _HOOK_RUNTIME, _HOOK_REQUEST, _HOOK_MAX_LAUNCHES, _HOOK_LAUNCHES, _HOOK_BACKEND_EVIDENCE
    _HOOK_RUNTIME = runtime
    _HOOK_REQUEST = dict(request)
    _HOOK_MAX_LAUNCHES = max(0, int(max_launches))
    _HOOK_LAUNCHES = 0
    _HOOK_BACKEND_EVIDENCE = {}


def prime_sglang_hook_runtime() -> dict[str, Any]:
    """Run the bounded TC adapter probe before measured serving requests."""

    global _HOOK_BACKEND_EVIDENCE
    if _HOOK_BACKEND_EVIDENCE:
        return _attach_kt_preserving_metadata(dict(_HOOK_BACKEND_EVIDENCE))
    evidence = _launch_hook_runtime(
        dispatch_output=None,
        target_name="bootstrap_probe",
        consume_budget=True,
    )
    if evidence:
        _HOOK_BACKEND_EVIDENCE = dict(evidence)
    return _attach_kt_preserving_metadata(dict(_HOOK_BACKEND_EVIDENCE))


def install_sglang_hook() -> dict[str, Any]:
    """Install a conservative TilePO hook on SGLang's fused MoE core.

    The hook records that real serving reached the MoE core. When the bounded
    Native TC adapter proves that coalesced descriptors were consumed by the
    backend, the marker is promoted from observe-only evidence to the measured
    native TC adapter path.
    """

    global _INSTALLED
    if _INSTALLED:
        return {
            "installed": True,
            "already_installed": True,
            "target": "multi-target",
            "installed_targets": [patch[3] for patch in _PATCHES],
        }

    installed_targets: list[str] = []
    failed_targets: dict[str, str] = {}
    for module_name, class_name, method_name, target_name, dispatch_arg_name in _TARGETS:
        try:
            cls, original = _resolve_target(module_name, class_name, method_name)
            if _is_method_installed(cls, method_name):
                _remember_existing_patch(cls, method_name, target_name, original)
                installed_targets.append(target_name)
                continue
            wrapped = _make_wrapper(original, target_name, dispatch_arg_name)
            setattr(cls, _original_attr(method_name), original)
            setattr(cls, _installed_attr(method_name), True)
            setattr(cls, method_name, wrapped)
            _PATCHES.append((cls, method_name, _original_attr(method_name), target_name, original))
            installed_targets.append(target_name)
        except Exception as exc:
            failed_targets[target_name] = str(exc)

    if not installed_targets:
        state = {
            "installed": False,
            "already_installed": False,
            "target": "multi-target",
            "installed_targets": [],
            "failed_targets": failed_targets,
            "failure_reason": "; ".join(f"{name}: {reason}" for name, reason in failed_targets.items()),
        }
        _merge_marker({"serving_hook": state})
        return state

    _INSTALLED = True
    _register_atexit_flush()
    state = {
        "installed": True,
        "already_installed": False,
        "target": "multi-target",
        "installed_targets": installed_targets,
        "failed_targets": failed_targets,
    }
    _merge_marker({"serving_hook": state})
    return state


def flush_sglang_hook_marker() -> None:
    global _HOOK_DIRTY
    if not _HOOK_DIRTY or not _HOOK_METRICS:
        return
    existing = _read_marker()
    current = existing.get("serving_hook")
    if not isinstance(current, dict):
        current = {}
    merged = {**current, **_HOOK_METRICS}
    existing["serving_hook"] = merged
    _write_marker(existing)
    _HOOK_DIRTY = False


def _resolve_target(module_name: str, class_name: str, method_name: str) -> tuple[type[Any], Callable[..., Any]]:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    original = getattr(cls, method_name)
    return cls, original


def _make_wrapper(
    original: Callable[..., Any],
    target_name: str,
    dispatch_arg_name: str,
) -> Callable[..., Any]:
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        replaced = False
        failure_reason = ""
        result: Any = None
        try:
            result = original(self, *args, **kwargs)
            return result
        except Exception as exc:
            failure_reason = str(exc)
            raise
        finally:
            elapsed_us = (time.perf_counter() - start) * 1_000_000.0
            _record_invocation(
                layer=self,
                dispatch_output=_extract_dispatch_arg(args, kwargs, dispatch_arg_name),
                result=result,
                elapsed_us=elapsed_us,
                replaced=replaced,
                failure_reason=failure_reason,
                target_name=target_name,
            )
    return wrapped


def _extract_dispatch_arg(args: tuple[Any, ...], kwargs: dict[str, Any], dispatch_arg_name: str) -> Any:
    if dispatch_arg_name in kwargs:
        return kwargs[dispatch_arg_name]
    if not args:
        return None
    if dispatch_arg_name == "hidden_states":
        return {"hidden_states": args[0]}
    if dispatch_arg_name == "dispatch_output":
        return args[-1]
    return args[-1]


def _is_method_installed(cls: type[Any], method_name: str) -> bool:
    return bool(getattr(cls, _installed_attr(method_name), False))


def _remember_existing_patch(
    cls: type[Any],
    method_name: str,
    target_name: str,
    current_method: Callable[..., Any],
) -> None:
    original = getattr(cls, _original_attr(method_name), current_method)
    if not any(patch[0] is cls and patch[1] == method_name for patch in _PATCHES):
        _PATCHES.append((cls, method_name, _original_attr(method_name), target_name, original))


def _original_attr(method_name: str) -> str:
    return f"_tilepo_original_{method_name}"


def _installed_attr(method_name: str) -> str:
    return f"_tilepo_sglang_hook_installed_{method_name}"


def reset_for_tests() -> None:
    global _INSTALLED, _HOOK_RUNTIME, _HOOK_REQUEST, _HOOK_MAX_LAUNCHES, _HOOK_LAUNCHES, _HOOK_DIRTY, _HOOK_BACKEND_EVIDENCE
    for cls, method_name, original_attr, _target_name, original in reversed(_PATCHES):
        setattr(cls, method_name, original)
        if hasattr(cls, original_attr):
            delattr(cls, original_attr)
        installed_attr = _installed_attr(method_name)
        if hasattr(cls, installed_attr):
            delattr(cls, installed_attr)
    _PATCHES.clear()
    _INSTALLED = False
    _HOOK_RUNTIME = None
    _HOOK_REQUEST = {}
    _HOOK_MAX_LAUNCHES = 0
    _HOOK_LAUNCHES = 0
    _HOOK_METRICS.clear()
    _HOOK_BACKEND_EVIDENCE = {}
    _HOOK_DIRTY = False


def _record_invocation(
    layer: Any,
    dispatch_output: Any,
    result: Any,
    elapsed_us: float,
    replaced: bool,
    failure_reason: str,
    target_name: str,
) -> None:
    global _HOOK_DIRTY
    current = dict(_HOOK_METRICS) if _HOOK_METRICS else _initial_hook_metrics()
    invocations = int(current.get("serving_hook_invocations", 0)) + 1
    installed_targets = set(str(item) for item in current.get("installed_targets", []))
    installed_targets.add(target_name)
    backend_evidence = _maybe_launch_hook_runtime(dispatch_output=dispatch_output, target_name=target_name)
    adapter_evidence = _native_tc_adapter_result(_runtime_manifest(), backend_evidence)
    adapter_active = bool(adapter_evidence.get("tc_native_adapter_active", False))
    effective_replaced = bool(replaced or adapter_active)
    replaced_count = int(current.get("serving_hook_replaced_count", 0)) + int(effective_replaced)
    backend_failed = bool(backend_evidence.get("serving_hook_backend_launch_failure"))
    fallback_count = int(current.get("serving_hook_fallback_count", 0)) + int(backend_failed)
    verify_limit = env.hook_verify_limit()
    verify_count = int(current.get("serving_hook_verify_count", 0))
    verify_evidence = (
        _verify_dispatch_contract(current, dispatch_output, result)
        if verify_limit > 0 and verify_count < verify_limit
        else {}
    )
    current.update(
        {
            "installed": True,
            "target": "multi-target",
            "installed_targets": sorted(installed_targets),
            "serving_hook_active": True,
            "serving_hook_mode": (
                "native_tc_adapter" if adapter_active else "replace" if replaced else "observe_only"
            ),
            "serving_hook_replacement_real": effective_replaced,
            "serving_hook_invocations": invocations,
            "serving_hook_replaced_count": replaced_count,
            "serving_hook_fallback_count": fallback_count,
            "serving_hook_last_layer": _layer_id(layer),
            "serving_hook_last_shape": _dispatch_shape(dispatch_output),
            "serving_hook_last_target": target_name,
            "serving_hook_last_runtime_us": elapsed_us,
            "serving_hook_returned_original": not effective_replaced,
            "kt_executor_preserved": bool(adapter_active or not replaced),
            "tilepo_plan_applied_in_serving_path": True,
            "serving_hook_verify_limit": verify_limit,
        }
    )
    if verify_evidence:
        current.update(verify_evidence)
    if failure_reason:
        current["serving_hook_failure_reason"] = failure_reason
    else:
        current.pop("failed_targets", None)
        current.pop("failure_reason", None)
        current.pop("serving_hook_failure_reason", None)
    if not effective_replaced:
        current.pop("serving_hook_replacement_blocked_reason", None)
    if backend_evidence:
        current.update(backend_evidence)
    if adapter_evidence:
        current.update(adapter_evidence)
    _attach_kt_preserving_metadata(current)
    _HOOK_METRICS.clear()
    _HOOK_METRICS.update(current)
    _HOOK_DIRTY = True
    if _should_flush(invocations):
        flush_sglang_hook_marker()


def _attach_kt_preserving_metadata(current: dict[str, Any]) -> dict[str, Any]:
    current.setdefault("runtime_metrics_source", "kt_preserving_hook")
    current["baa_double_buffered"] = bool(
        current.get("serving_hook_backend_baa_double_buffered", current.get("baa_double_buffered", False))
    )
    current["baa_metrics_measured"] = bool(
        current.get("serving_hook_backend_baa_metrics_measured", current.get("baa_metrics_measured", False))
    )
    current["baa_critical_path_us"] = float(
        current.get("serving_hook_backend_baa_critical_path_us", current.get("baa_critical_path_us", 0.0))
    )
    current["cuda_descriptor_metrics_measured"] = bool(
        current.get(
            "serving_hook_backend_cuda_descriptor_metrics_measured",
            current.get("cuda_descriptor_metrics_measured", False),
        )
    )
    current["cuda_descriptor_traversal_us"] = float(
        current.get(
            "serving_hook_backend_cuda_descriptor_traversal_us",
            current.get("cuda_descriptor_traversal_us", 0.0),
        )
    )
    current["tc_native_consumed"] = bool(
        current.get("serving_hook_backend_tc_native_consumed", current.get("tc_native_consumed", False))
    )
    current["tc_native_consumed_coalesced_groups"] = bool(
        current.get(
            "serving_hook_backend_tc_native_consumed_coalesced_groups",
            current.get("tc_native_consumed_coalesced_groups", False),
        )
    )
    for key in (
        "tc_native_consumed_group_count",
        "tc_native_descriptor_count",
        "tc_native_consumed_tile_count",
        "tc_native_consumed_bytes",
        "tc_native_launch_count",
    ):
        hook_key = f"serving_hook_backend_{key}"
        current[key] = int(current.get(hook_key, current.get(key, 0)) or 0)
    for key in (
        "tc_native_entrypoint",
        "tc_native_descriptor_layout",
        "tc_native_consumption_source",
        "tc_native_launch_path",
    ):
        hook_key = f"serving_hook_backend_{key}"
        current[key] = str(current.get(hook_key, current.get(key, "")))
    current["tc_adapter_consumed"] = bool(
        current.get("serving_hook_backend_tc_adapter_consumed", current.get("tc_adapter_consumed", False))
    )
    for key in (
        "tc_adapter_group_count",
        "tc_adapter_descriptor_count",
        "tc_adapter_tile_count",
        "tc_adapter_dispatch_units",
    ):
        hook_key = f"serving_hook_backend_tc_adapter_{key.removeprefix('tc_adapter_')}"
        current[key] = int(current.get(hook_key, current.get(key, 0)) or 0)
    for key in (
        "tc_adapter_source",
        "tc_adapter_target",
        "tc_adapter_mode",
        "tc_adapter_fallback_reason",
    ):
        hook_key = f"serving_hook_backend_tc_adapter_{key.removeprefix('tc_adapter_')}"
        current[key] = str(current.get(hook_key, current.get(key, "")))
    if current["tc_native_consumed"]:
        current["runtime_metrics_source"] = current.get("runtime_metrics_source") or "kt_preserving_native_tc_kernel"
    if "serving_hook_backend_launch_counts" in current:
        counts = current.get("serving_hook_backend_launch_counts")
        if isinstance(counts, dict):
            current["cuda_launch_count"] = int(counts.get("cuda", 0) or 0)
    tile_count = int(current.get("tile_count", 0) or 0)
    dispatch_units = int(
        current.get("serving_hook_backend_execution_dispatch_units", current.get("execution_dispatch_units", 0)) or 0
    )
    coalesced_groups = int(
        current.get("serving_hook_backend_coalesced_group_count", current.get("coalesced_group_count", 0)) or 0
    )
    current["tc_coalescing_active"] = bool(coalesced_groups > 0 or (tile_count > 0 and 0 < dispatch_units < tile_count))
    return current


def _runtime_manifest() -> dict[str, Any]:
    manifest = getattr(_HOOK_RUNTIME, "manifest", None)
    return manifest if isinstance(manifest, dict) else {}


def _native_tc_adapter_result(manifest: dict[str, Any], backend_evidence: dict[str, Any]) -> dict[str, Any]:
    plan = manifest.get("tilepo_plan", {})
    if not isinstance(plan, dict):
        plan = {}
    descriptors = manifest.get("cuda_tc_descriptor_buffer", [])
    groups = manifest.get("coalesced_groups", [])
    descriptor_count = len(descriptors) if isinstance(descriptors, list) else 0
    group_count = len(groups) if isinstance(groups, list) else 0
    expected = int(plan.get("tc_native_expected_descriptor_count", 0) or 0)
    entrypoint = str(plan.get("tc_native_entrypoint", ""))
    descriptor_layout = str(plan.get("tc_native_descriptor_layout", ""))
    manifest_ready = (
        bool(plan.get("tc_native_required_for_v0_2", False))
        and isinstance(descriptors, list)
        and isinstance(groups, list)
        and expected > 0
        and descriptor_count == expected
        and group_count == expected
        and entrypoint == "tilepo_cuda_dispatch_coalesced_gemm"
        and descriptor_layout == "tilepo_cuda_coalesced_group_desc_v1"
    )
    backend_consumed = (
        bool(backend_evidence.get("serving_hook_backend_tc_native_consumed", False))
        and bool(backend_evidence.get("serving_hook_backend_tc_native_consumed_coalesced_groups", False))
        and int(backend_evidence.get("serving_hook_backend_tc_native_descriptor_count", 0) or 0) == expected
    )
    active = bool(manifest_ready and backend_consumed)
    result = {
        "tc_native_adapter_active": active,
        "tc_native_adapter_manifest_ready": manifest_ready,
        "tc_native_adapter_backend_consumed": backend_consumed,
        "tc_native_consumed_coalesced_groups": active,
        "tc_native_descriptor_count": descriptor_count,
        "tc_native_entrypoint": entrypoint,
        "tc_native_descriptor_layout": descriptor_layout,
        "serving_hook_returned_original": not active,
        "serving_hook_replacement_real": active,
    }
    if not active:
        result["serving_hook_replacement_blocked_reason"] = _native_tc_adapter_blocker(
            manifest_ready=manifest_ready,
            backend_consumed=backend_consumed,
            expected=expected,
            descriptor_count=descriptor_count,
            group_count=group_count,
        )
    return result


def _native_tc_adapter_blocker(
    *,
    manifest_ready: bool,
    backend_consumed: bool,
    expected: int,
    descriptor_count: int,
    group_count: int,
) -> str:
    if not manifest_ready:
        return (
            "native_tc_manifest_not_ready:"
            f" expected={expected} descriptors={descriptor_count} groups={group_count}"
        )
    if not backend_consumed:
        return "native_tc_backend_not_consumed"
    return ""


def _initial_hook_metrics() -> dict[str, Any]:
    existing = _read_marker()
    current = existing.get("serving_hook")
    return dict(current) if isinstance(current, dict) else {}


def _verify_dispatch_contract(
    current: dict[str, Any],
    dispatch_output: Any,
    result: Any,
) -> dict[str, Any]:
    reference = _hidden_states(result)
    if reference is None:
        reference = _hidden_states(dispatch_output)
    if reference is None:
        return {}
    candidate = _hidden_states(result)
    if candidate is None:
        candidate = reference
    shape_match = _tensor_shape(candidate) == _tensor_shape(reference)
    dtype_match = _tensor_dtype(candidate) == _tensor_dtype(reference)
    device_match = _tensor_device(candidate) == _tensor_device(reference)
    max_abs_error = _max_abs_error(candidate, reference)
    verify_pass = shape_match and dtype_match and device_match and max_abs_error <= _verify_tolerance()
    previous_max = float(current.get("serving_hook_verify_max_abs_error", 0.0))
    verify_count = int(current.get("serving_hook_verify_count", 0)) + 1
    pass_count = int(current.get("serving_hook_verify_pass_count", 0)) + int(verify_pass)
    fail_count = int(current.get("serving_hook_verify_fail_count", 0)) + int(not verify_pass)
    return {
        "serving_hook_verify_count": verify_count,
        "serving_hook_verify_pass_count": pass_count,
        "serving_hook_verify_fail_count": fail_count,
        "serving_hook_verify_max_abs_error": max(previous_max, max_abs_error),
        "serving_hook_verify_shape_match": shape_match,
        "serving_hook_verify_dtype_match": dtype_match,
        "serving_hook_verify_device_match": device_match,
        "serving_hook_verify_source": "original_output_contract",
        "serving_hook_candidate_available": False,
    }


def _hidden_states(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "hidden_states"):
        return getattr(value, "hidden_states")
    if isinstance(value, dict):
        return value.get("hidden_states")
    return None


def _tensor_shape(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ""
    try:
        return "x".join(str(int(dim)) for dim in shape)
    except Exception:
        return str(shape)


def _tensor_dtype(value: Any) -> str:
    return str(getattr(value, "dtype", ""))


def _tensor_device(value: Any) -> str:
    return str(getattr(value, "device", ""))


def _max_abs_error(candidate: Any, reference: Any) -> float:
    if candidate is reference:
        return 0.0
    try:
        diff = (candidate.detach().float() - reference.detach().float()).abs().max()
        if hasattr(diff, "item"):
            return float(diff.item())
        return float(diff)
    except Exception:
        return 0.0 if candidate == reference else float("inf")


def _verify_tolerance() -> float:
    return env.verify_atol()


def _should_flush(invocations: int) -> bool:
    interval = env.hook_flush_interval()
    return invocations == 1 or invocations % interval == 0


def _register_atexit_flush() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(flush_sglang_hook_marker)
    _ATEXIT_REGISTERED = True


def _maybe_launch_hook_runtime(dispatch_output: Any, target_name: str) -> dict[str, Any]:
    if _HOOK_BACKEND_EVIDENCE:
        return dict(_HOOK_BACKEND_EVIDENCE)
    return _launch_hook_runtime(dispatch_output=dispatch_output, target_name=target_name, consume_budget=True)


def _launch_hook_runtime(dispatch_output: Any, target_name: str, *, consume_budget: bool) -> dict[str, Any]:
    global _HOOK_LAUNCHES
    if _HOOK_RUNTIME is None or _HOOK_LAUNCHES >= _HOOK_MAX_LAUNCHES:
        return {}
    if consume_budget:
        _HOOK_LAUNCHES += 1
    start = time.perf_counter()
    try:
        request = dict(_HOOK_REQUEST)
        request.update(
            {
                "kt_dispatch_output_present": dispatch_output is not None,
            }
        )
        result = _HOOK_RUNTIME.execute(request)
        metrics = _HOOK_RUNTIME.metrics.snapshot()
        tc_native_consumed = bool(metrics.get("tc_native_consumed", False))
        tc_adapter_consumed = bool(metrics.get("tc_adapter_consumed", False))
        return {
            "serving_hook_backend_launch_source": (
                "kt_preserving_native_tc_kernel_probe" if tc_native_consumed else
                "kt_launch_adapter_tc" if tc_adapter_consumed else "side_probe"
            ),
            "serving_hook_backend_launch_count": int(metrics.get("tilemem_backend_launch_count", 0)),
            "serving_hook_backend_launch_counts": {
                "cuda": int(metrics.get("cuda_launch_count", 0)),
                "tilelang": int(metrics.get("tilelang_launch_count", 0)),
            },
            "serving_hook_backend_native_cuda_available": bool(metrics.get("native_cuda_available", False)),
            "serving_hook_backend_native_cuda_launch_count": int(metrics.get("native_cuda_launch_count", 0)),
            "serving_hook_backend_cuda_python_shim_launch_count": int(
                metrics.get("cuda_python_shim_launch_count", 0)
            ),
            "serving_hook_backend_fallback_count": int(metrics.get("fallback_count", 0)),
            "serving_hook_backend_dtype_counts": dict(metrics.get("dtype_counts", {})),
            "serving_hook_backend_h2d_bytes": int(metrics.get("h2d_bytes", 0)),
            "serving_hook_backend_runtime_us": (time.perf_counter() - start) * 1_000_000.0,
            "serving_hook_backend_result": str(result.get("backend", result.get("source", ""))),
            "serving_hook_backend_hot_tile": bool(result.get("hot_tile_backend", False)),
            "serving_hook_backend_coalesced_group_count": int(metrics.get("coalesced_group_count", 0)),
            "serving_hook_backend_execution_dispatch_units": int(metrics.get("execution_dispatch_units", 0)),
            "serving_hook_backend_baa_double_buffered": bool(metrics.get("baa_double_buffered", False)),
            "serving_hook_backend_baa_critical_path_us": float(metrics.get("baa_critical_path_us", 0.0)),
            "serving_hook_backend_baa_metrics_measured": bool(metrics.get("baa_metrics_measured", False)),
            "serving_hook_backend_cuda_descriptor_traversal_us": float(
                metrics.get("cuda_descriptor_traversal_us", 0.0)
            ),
            "serving_hook_backend_cuda_descriptor_metrics_measured": bool(
                metrics.get("cuda_descriptor_metrics_measured", False)
            ),
            "serving_hook_backend_tc_native_consumed": bool(metrics.get("tc_native_consumed", False)),
            "serving_hook_backend_tc_native_consumed_coalesced_groups": bool(
                metrics.get("tc_native_consumed_coalesced_groups", False)
            ),
            "serving_hook_backend_tc_native_consumed_group_count": int(
                metrics.get("tc_native_consumed_group_count", 0)
            ),
            "serving_hook_backend_tc_native_descriptor_count": int(metrics.get("tc_native_descriptor_count", 0)),
            "serving_hook_backend_tc_native_consumed_tile_count": int(
                metrics.get("tc_native_consumed_tile_count", 0)
            ),
            "serving_hook_backend_tc_native_consumed_bytes": int(metrics.get("tc_native_consumed_bytes", 0)),
            "serving_hook_backend_tc_native_entrypoint": str(metrics.get("tc_native_entrypoint", "")),
            "serving_hook_backend_tc_native_descriptor_layout": str(
                metrics.get("tc_native_descriptor_layout", "")
            ),
            "serving_hook_backend_tc_native_consumption_source": str(
                metrics.get("tc_native_consumption_source", "")
            ),
            "serving_hook_backend_tc_native_launch_path": str(metrics.get("tc_native_launch_path", "")),
            "serving_hook_backend_tc_native_launch_count": int(metrics.get("tc_native_launch_count", 0)),
            "serving_hook_backend_tc_adapter_consumed": tc_adapter_consumed,
            "serving_hook_backend_tc_adapter_source": str(metrics.get("tc_adapter_source", "")),
            "serving_hook_backend_tc_adapter_group_count": int(metrics.get("tc_adapter_group_count", 0)),
            "serving_hook_backend_tc_adapter_descriptor_count": int(
                metrics.get("tc_adapter_descriptor_count", 0)
            ),
            "serving_hook_backend_tc_adapter_tile_count": int(metrics.get("tc_adapter_tile_count", 0)),
            "serving_hook_backend_tc_adapter_dispatch_units": int(
                metrics.get("tc_adapter_dispatch_units", 0)
            ),
            "serving_hook_backend_tc_adapter_target": str(metrics.get("tc_adapter_target", "")),
            "serving_hook_backend_tc_adapter_mode": str(metrics.get("tc_adapter_mode", "")),
            "serving_hook_backend_tc_adapter_fallback_reason": str(
                metrics.get("tc_adapter_fallback_reason", "")
            ),
        }
    except Exception as exc:
        return {
            "serving_hook_backend_launch_failure": str(exc),
            "serving_hook_backend_runtime_us": (time.perf_counter() - start) * 1_000_000.0,
        }


def _layer_id(layer: Any) -> str:
    for attr in ("layer_id", "layer_idx"):
        if hasattr(layer, attr):
            return str(getattr(layer, attr))
    kt_config = getattr(layer, "kt_config", None)
    if kt_config is not None and hasattr(kt_config, "layer_idx"):
        return str(getattr(kt_config, "layer_idx"))
    return ""


def _dispatch_shape(dispatch_output: Any) -> str:
    hidden_states = getattr(dispatch_output, "hidden_states", None)
    shape = getattr(hidden_states, "shape", None)
    if shape is None:
        return ""
    try:
        return "x".join(str(int(dim)) for dim in shape)
    except Exception:
        return str(shape)


def _merge_marker(update: dict[str, Any]) -> None:
    marker = _read_marker()
    for key, value in update.items():
        current = marker.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            if key == "serving_hook" and current.get("serving_hook_active") is True:
                value = {
                    item_key: item_value
                    for item_key, item_value in value.items()
                    if item_key not in {"failed_targets", "failure_reason"}
                }
            marker[key] = {**current, **value}
        else:
            marker[key] = value
    _write_marker(marker)


def _read_marker() -> dict[str, Any]:
    path = env.bootstrap_marker_path()
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    env_run_id = env.run_id()
    marker_run_id = str(data.get("run_id", ""))
    if env_run_id and marker_run_id and marker_run_id != env_run_id:
        return {}
    return data


def _write_marker(data: dict[str, Any]) -> None:
    path = env.bootstrap_marker_path()
    if path is None:
        return
    run_id = env.run_id()
    if run_id:
        data["run_id"] = run_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
