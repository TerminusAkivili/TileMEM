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
)

_INSTALLED = False
_PATCHES: list[tuple[type[Any], str, str, str, Callable[..., Any]]] = []
_HOOK_RUNTIME: Any | None = None
_HOOK_REQUEST: dict[str, Any] = {}
_HOOK_MAX_LAUNCHES = 0
_HOOK_LAUNCHES = 0
_HOOK_METRICS: dict[str, Any] = {}
_HOOK_DIRTY = False
_ATEXIT_REGISTERED = False


def configure_sglang_hook_runtime(
    runtime: Any,
    request: dict[str, Any],
    max_launches: int = 1,
) -> None:
    global _HOOK_RUNTIME, _HOOK_REQUEST, _HOOK_MAX_LAUNCHES, _HOOK_LAUNCHES
    _HOOK_RUNTIME = runtime
    _HOOK_REQUEST = dict(request)
    _HOOK_MAX_LAUNCHES = max(0, int(max_launches))
    _HOOK_LAUNCHES = 0


def install_sglang_hook() -> dict[str, Any]:
    """Install a conservative TilePO hook on SGLang's fused MoE core.

    The hook is intentionally observe-only by default: it records that real
    serving reached the MoE core and then returns SGLang/KT's original output.
    Replacement is not attempted unless a future guarded path explicitly opts
    in and proves the dispatch contract.
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
    global _INSTALLED, _HOOK_RUNTIME, _HOOK_REQUEST, _HOOK_MAX_LAUNCHES, _HOOK_LAUNCHES, _HOOK_DIRTY
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
    replaced_count = int(current.get("serving_hook_replaced_count", 0)) + int(replaced)
    fallback_count = int(current.get("serving_hook_fallback_count", 0)) + int(not replaced)
    installed_targets = set(str(item) for item in current.get("installed_targets", []))
    installed_targets.add(target_name)
    backend_evidence = _maybe_launch_hook_runtime()
    verify_evidence = _verify_dispatch_contract(current, dispatch_output, result)
    current.update(
        {
            "installed": True,
            "target": "multi-target",
            "installed_targets": sorted(installed_targets),
            "serving_hook_active": True,
            "serving_hook_invocations": invocations,
            "serving_hook_replaced_count": replaced_count,
            "serving_hook_fallback_count": fallback_count,
            "serving_hook_last_layer": _layer_id(layer),
            "serving_hook_last_shape": _dispatch_shape(dispatch_output),
            "serving_hook_last_target": target_name,
            "serving_hook_last_runtime_us": elapsed_us,
            "serving_hook_returned_original": not replaced,
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
    if env.serve_replace_enabled() and not replaced:
        current["serving_hook_replacement_blocked_reason"] = (
            "SGLang dispatch_output replacement contract is not enabled for TilePO yet"
        )
    if backend_evidence:
        current.update(backend_evidence)
    _HOOK_METRICS.clear()
    _HOOK_METRICS.update(current)
    _HOOK_DIRTY = True
    if _should_flush(invocations):
        flush_sglang_hook_marker()


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


def _maybe_launch_hook_runtime() -> dict[str, Any]:
    global _HOOK_LAUNCHES
    if _HOOK_RUNTIME is None or _HOOK_LAUNCHES >= _HOOK_MAX_LAUNCHES:
        return {}
    _HOOK_LAUNCHES += 1
    start = time.perf_counter()
    try:
        result = _HOOK_RUNTIME.execute(dict(_HOOK_REQUEST))
        metrics = _HOOK_RUNTIME.metrics.snapshot()
        return {
            "serving_hook_backend_launch_count": int(metrics.get("tilemem_backend_launch_count", 0)),
            "serving_hook_backend_launch_counts": {
                "cuda": int(metrics.get("cuda_launch_count", 0)),
                "tilelang": int(metrics.get("tilelang_launch_count", 0)),
            },
            "serving_hook_backend_fallback_count": int(metrics.get("fallback_count", 0)),
            "serving_hook_backend_dtype_counts": dict(metrics.get("dtype_counts", {})),
            "serving_hook_backend_h2d_bytes": int(metrics.get("h2d_bytes", 0)),
            "serving_hook_backend_runtime_us": (time.perf_counter() - start) * 1_000_000.0,
            "serving_hook_backend_result": str(result.get("backend", result.get("source", ""))),
            "serving_hook_backend_hot_tile": bool(result.get("hot_tile_backend", False)),
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
            marker[key] = {**value, **current}
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
