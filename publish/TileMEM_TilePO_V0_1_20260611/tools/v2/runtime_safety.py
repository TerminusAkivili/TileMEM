from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


GIB = 1024**3

TEMP_KEYS = ["TMPDIR", "TEMP", "TMP"]

CACHE_DIRS = {
    "XDG_CACHE_HOME": "cache/xdg",
    "HF_HOME": "cache/huggingface",
    "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub",
    "TRANSFORMERS_CACHE": "cache/huggingface/transformers",
    "TORCH_HOME": "cache/torch",
    "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    "TRITON_CACHE_DIR": "cache/triton",
    "FLASHINFER_CACHE_DIR": "cache/flashinfer",
    "FLASHINFER_WORKSPACE_BASE": "cache/flashinfer_workspace",
    "FLASHINFER_CUBIN_DIR": "cache/flashinfer_cubins",
    "CUTE_DSL_CACHE_DIR": "cache/cute_dsl",
    "CUDA_CACHE_PATH": "cache/cuda",
    "PIP_CACHE_DIR": "cache/pip",
    "NUMBA_CACHE_DIR": "cache/numba",
}

SHARED_JIT_CACHE_DIRS = {
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
    "FLASHINFER_CACHE_DIR": "flashinfer",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer_workspace",
    "FLASHINFER_CUBIN_DIR": "flashinfer_cubins",
    "CUTE_DSL_CACHE_DIR": "cute_dsl",
    "CUDA_CACHE_PATH": "cuda",
    "NUMBA_CACHE_DIR": "numba",
}

LOCALHOST_NO_PROXY = ["127.0.0.1", "localhost", "::1", "0.0.0.0", "10.255.255.254"]


def add_localhost_no_proxy(env: dict[str, str]) -> dict[str, str]:
    for key in ("NO_PROXY", "no_proxy"):
        existing = [
            value.strip()
            for value in env.get(key, "").split(",")
            if value.strip()
        ]
        merged = existing[:]
        for value in LOCALHOST_NO_PROXY:
            if value not in merged:
                merged.append(value)
        env[key] = ",".join(merged)
    return env


def apply_runtime_env(
    env: dict[str, str],
    runtime_dir: Path | str | None,
    native_tmp_dir: Path | str | None = None,
) -> dict[str, str]:
    if not runtime_dir:
        return env
    root = Path(runtime_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(native_tmp_dir).resolve() if native_tmp_dir else root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    for key in TEMP_KEYS:
        env[key] = str(tmp_root)
    for key, relative in CACHE_DIRS.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    shared_jit_root = env.get("TILEMEM_SHARED_JIT_CACHE_DIR", "").strip()
    if shared_jit_root:
        shared_root = Path(shared_jit_root).resolve()
        shared_root.mkdir(parents=True, exist_ok=True)
        for key, relative in SHARED_JIT_CACHE_DIRS.items():
            path = shared_root / relative
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
    env.setdefault("MAX_JOBS", "1")
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "1")
    env.setdefault("NINJAFLAGS", "-j1")
    env.setdefault("NVCC_THREADS", "1")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("PYTHONUNBUFFERED", "1")
    add_localhost_no_proxy(env)
    return env


def parse_df_output(text: str) -> dict[str, dict]:
    mounts: dict[str, dict] = {}
    for raw in text.splitlines()[1:]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        mount = parts[-1]
        use_percent = float(parts[-2].rstrip("%"))
        available = int(parts[-3])
        used = int(parts[-4])
        total = int(parts[-5])
        filesystem = " ".join(parts[:-5])
        mounts[mount] = {
            "filesystem": filesystem,
            "total_bytes": total,
            "used_bytes": used,
            "available_bytes": available,
            "free_gib": available / GIB,
            "use_percent": use_percent,
        }
    return mounts


def df_for_mounts(mounts: list[str]) -> str:
    return subprocess.check_output(["df", "-P", "-B1", *mounts], text=True)


def disk_snapshot(
    mounts: dict[str, dict],
    *,
    c_mount: str,
    d_mount: str,
    min_c_free_gib: float,
    max_c_use_percent: float,
    min_d_free_gib: float,
    phase: str = "",
) -> dict:
    failures: list[str] = []
    c_info = mounts.get(c_mount)
    d_info = mounts.get(d_mount)
    if c_info is None:
        failures.append(f"C drive mount {c_mount} was not found")
    else:
        if min_c_free_gib > 0 and c_info["free_gib"] < min_c_free_gib:
            failures.append(
                f"C drive {c_mount} free space {c_info['free_gib']:.1f} GiB "
                f"is below required {min_c_free_gib:.1f} GiB"
            )
        if max_c_use_percent < 100 and c_info["use_percent"] > max_c_use_percent:
            failures.append(
                f"C drive {c_mount} usage {c_info['use_percent']:.0f}% "
                f"is above allowed {max_c_use_percent:.0f}%"
            )
    if d_mount:
        if d_info is None:
            failures.append(f"D drive mount {d_mount} was not found")
        elif min_d_free_gib > 0 and d_info["free_gib"] < min_d_free_gib:
            failures.append(
                f"D drive {d_mount} free space {d_info['free_gib']:.1f} GiB "
                f"is below required {min_d_free_gib:.1f} GiB"
            )
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "thresholds": {
            "c_mount": c_mount,
            "d_mount": d_mount,
            "min_c_free_gib": min_c_free_gib,
            "max_c_use_percent": max_c_use_percent,
            "min_d_free_gib": min_d_free_gib,
        },
        "mounts": mounts,
    }


def write_disk_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2) + "\n")


def write_json_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2) + "\n")


def read_or_collect_mounts(
    *,
    df_fixture: Path | None,
    c_mount: str,
    d_mount: str,
) -> dict[str, dict]:
    if df_fixture:
        text = df_fixture.read_text()
    else:
        mount_args = [c_mount]
        if d_mount and d_mount != c_mount:
            mount_args.append(d_mount)
        text = df_for_mounts(mount_args)
    return parse_df_output(text)


def collect_windows_host_info(timeout_sec: float = 10.0) -> dict:
    script = r"""
$commit = Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit'
$committed = 0.0
$limit = 0.0
foreach ($sample in $commit.CounterSamples) {
  if ($sample.Path -like '*committed bytes') { $committed = $sample.CookedValue }
  if ($sample.Path -like '*commit limit') { $limit = $sample.CookedValue }
}
$vmmem = Get-Process -Name vmmemWSL -ErrorAction SilentlyContinue | Select-Object -First 1
$obj = [PSCustomObject]@{
  committed_gib = [math]::Round($committed / 1GB, 3)
  commit_limit_gib = [math]::Round($limit / 1GB, 3)
  commit_percent = if ($limit -gt 0) { [math]::Round(100.0 * $committed / $limit, 3) } else { 0.0 }
  vmmem_virtual_gib = if ($vmmem) { [math]::Round($vmmem.VirtualMemorySize64 / 1GB, 3) } else { 0.0 }
  vmmem_working_set_gib = if ($vmmem) { [math]::Round($vmmem.WorkingSet64 / 1GB, 3) } else { 0.0 }
  vmmem_paged_memory_gib = if ($vmmem) { [math]::Round($vmmem.PagedMemorySize64 / 1GB, 3) } else { 0.0 }
}
$obj | ConvertTo-Json -Compress
"""
    output = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", script],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
    )
    data = json.loads(output)
    return {key: float(value) for key, value in data.items()}


def read_or_collect_host_info(
    host_fixture: Path | None,
    *,
    attempts: int = 3,
    retry_delay_sec: float = 0.5,
    timeout_sec: float = 10.0,
) -> dict:
    if host_fixture:
        return json.loads(host_fixture.read_text())
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            try:
                return collect_windows_host_info(timeout_sec=timeout_sec)
            except TypeError as exc:
                if "timeout_sec" not in str(exc):
                    raise
                return collect_windows_host_info()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(retry_delay_sec)
    assert last_error is not None
    raise last_error


def host_snapshot(
    info: dict,
    *,
    max_host_commit_percent: float,
    max_vmmem_gib: float,
    phase: str = "",
) -> dict:
    failures: list[str] = []
    commit_percent = float(info.get("commit_percent", 0.0))
    vmmem_virtual_gib = float(info.get("vmmem_virtual_gib", 0.0))
    if max_host_commit_percent < 100 and commit_percent > max_host_commit_percent:
        failures.append(
            f"host commit {commit_percent:.1f}% is above allowed "
            f"{max_host_commit_percent:.1f}%"
        )
    if max_vmmem_gib > 0 and vmmem_virtual_gib > max_vmmem_gib:
        failures.append(
            f"vmmemWSL virtual memory {vmmem_virtual_gib:.1f} GiB is above "
            f"allowed {max_vmmem_gib:.1f} GiB"
        )
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        "status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "thresholds": {
            "max_host_commit_percent": max_host_commit_percent,
            "max_vmmem_gib": max_vmmem_gib,
        },
        "host": info,
    }
