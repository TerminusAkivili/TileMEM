from __future__ import annotations

import os
import math
from pathlib import Path


TILEPO_ASYNC_PLANNING = "TILEPO_ASYNC_PLANNING"
TILEPO_BACKEND = "TILEPO_BACKEND"
TILEPO_BOOTSTRAP_MARKER = "TILEPO_BOOTSTRAP_MARKER"
TILEPO_BOOTSTRAPPED = "TILEPO_BOOTSTRAPPED"
TILEPO_ENABLE = "TILEPO_ENABLE"
TILEPO_HOOK_BACKEND_PROBE_LIMIT = "TILEPO_HOOK_BACKEND_PROBE_LIMIT"
TILEPO_HOOK_FLUSH_INTERVAL = "TILEPO_HOOK_FLUSH_INTERVAL"
TILEPO_MANIFEST = "TILEPO_MANIFEST"
TILEPO_MANIFEST_CHECKSUM = "TILEPO_MANIFEST_CHECKSUM"
TILEPO_MODE = "TILEPO_MODE"
TILEPO_POLICY = "TILEPO_POLICY"
TILEPO_REQUIRE_NATIVE_BACKEND = "TILEPO_REQUIRE_NATIVE_BACKEND"
TILEPO_RUN_ID = "TILEPO_RUN_ID"
TILEPO_SERVE_REPLACE = "TILEPO_SERVE_REPLACE"
TILEPO_VERIFY_ATOL = "TILEPO_VERIFY_ATOL"
TILEMEM_REPO_ROOT = "TILEMEM_REPO_ROOT"


def get_text(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def get_path(name: str, default: str = "") -> Path:
    return Path(get_text(name, default))


def get_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def get_int(name: str, default: int, *, min_value: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        parsed = default
    else:
        try:
            parsed = int(value)
        except ValueError:
            parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    return parsed


def get_float(name: str, default: float, *, min_value: float | None = None) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        parsed = default
    else:
        try:
            parsed = float(value)
        except ValueError:
            parsed = default
    if not math.isfinite(parsed):
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    return parsed


def manifest_path() -> Path:
    return get_path(TILEPO_MANIFEST)


def mode() -> str:
    return get_text(TILEPO_MODE, "shadow")


def backend_priority() -> str:
    return get_text(TILEPO_BACKEND)


def run_id() -> str:
    return get_text(TILEPO_RUN_ID)


def bootstrap_marker_path() -> Path | None:
    marker = get_text(TILEPO_BOOTSTRAP_MARKER)
    return Path(marker) if marker else None


def hook_backend_probe_limit() -> int:
    return get_int(TILEPO_HOOK_BACKEND_PROBE_LIMIT, 1, min_value=0)


def hook_flush_interval() -> int:
    return get_int(TILEPO_HOOK_FLUSH_INTERVAL, 128, min_value=1)


def require_native_backend() -> bool:
    return get_bool(TILEPO_REQUIRE_NATIVE_BACKEND)


def serve_replace_enabled() -> bool:
    return get_bool(TILEPO_SERVE_REPLACE)


def verify_atol() -> float:
    return get_float(TILEPO_VERIFY_ATOL, 0.0, min_value=0.0)


def policy(default: str = "") -> str:
    return get_text(TILEPO_POLICY, default)


def async_planning(default: str = "") -> str:
    return get_text(TILEPO_ASYNC_PLANNING, default)


def tilemem_repo_root() -> Path | None:
    root = get_text(TILEMEM_REPO_ROOT)
    return Path(root) if root else None


def mark_bootstrapped(checksum: str) -> None:
    os.environ[TILEPO_BOOTSTRAPPED] = "1"
    os.environ[TILEPO_MANIFEST_CHECKSUM] = checksum
