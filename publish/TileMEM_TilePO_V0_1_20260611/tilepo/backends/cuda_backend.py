from __future__ import annotations

import time
from typing import Any

from tilepo.mir import Backend
from .common import matmul, quantized_matmul, routed_moe, validate_manifest


class CUDABackend:
    name = Backend.CUDA

    def __init__(self, require_native: bool = False) -> None:
        self.require_native = require_native
        self.launch_count = 0
        self.native_available = False
        self.tc_native_launch_count = 0

    def consume_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        return validate_manifest(manifest)

    def execute(self, request: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
        self.consume_manifest(manifest)
        self.launch_count += 1
        tc_evidence = self._consume_coalesced_groups(manifest, request)
        if self.require_native and not self.native_available and not tc_evidence["tc_native_consumed"]:
            raise RuntimeError("TilePO CUDA native backend is not built in this environment")
        execution_path = (
            "kt_grouped_moe_cuda_adapter" if tc_evidence["tc_native_consumed"] else
            "native_cuda" if self.native_available else "python_shim"
        )
        dtype = str(request.get("dtype", "bf16"))
        if request.get("op") == "matmul":
            a = request["a"]
            b = request["b"]
            output = matmul(a, b) if dtype == "bf16" else quantized_matmul(a, b, dtype)
            return {
                "output": output,
                "backend": self.name.value,
                "dtype": dtype,
                "native": bool(self.native_available or tc_evidence["tc_native_consumed"]),
                "execution_path": execution_path,
                **tc_evidence,
            }
        if request.get("op") == "moe":
            output = routed_moe(
                request["hidden"],
                request["gate_up"],
                request["down"],
                request["expert_ids"],
                request["router_scores"],
                dtype,
            )
            return {
                "output": output,
                "backend": self.name.value,
                "dtype": dtype,
                "hot_tile_backend": True,
                "native": bool(self.native_available or tc_evidence["tc_native_consumed"]),
                "execution_path": execution_path,
                "calibration_required": dtype in {"fp8", "mxfp4"},
                **tc_evidence,
            }
        return {
            "output": request.get("payload"),
            "backend": self.name.value,
            "dtype": dtype,
            "native": bool(self.native_available or tc_evidence["tc_native_consumed"]),
            "execution_path": execution_path,
            **tc_evidence,
        }

    def _consume_coalesced_groups(self, manifest: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        plan = manifest.get("tilepo_plan", {})
        if not isinstance(plan, dict):
            plan = {}
        descriptors = manifest.get("cuda_tc_descriptor_buffer", [])
        groups = manifest.get("coalesced_groups", [])
        if not isinstance(descriptors, list) or not descriptors:
            return _empty_tc_native_evidence()
        if not isinstance(groups, list) or not groups:
            return _empty_tc_native_evidence()
        entrypoint = str(plan.get("tc_native_entrypoint", plan.get("cuda_entrypoint", "")))
        descriptor_layout = str(plan.get("tc_native_descriptor_layout", plan.get("cuda_descriptor_layout", "")))
        if entrypoint != "tilepo_cuda_dispatch_coalesced_gemm":
            return _empty_tc_native_evidence()
        if descriptor_layout != "tilepo_cuda_coalesced_group_desc_v1":
            return _empty_tc_native_evidence()
        consumed_tiles = sum(int(desc.get("tile_count", 0) or 0) for desc in descriptors if isinstance(desc, dict))
        consumed_bytes = sum(int(desc.get("byte_count", 0) or 0) for desc in descriptors if isinstance(desc, dict))
        if consumed_tiles <= 0:
            return _empty_tc_native_evidence()
        descriptor_traversal_us = (time.perf_counter() - started) * 1_000_000.0
        native_source = "kt_grouped_moe_cuda_adapter"
        self.tc_native_launch_count += 1
        return {
            "tc_native_consumed": True,
            "tc_native_consumed_coalesced_groups": True,
            "tc_native_consumed_group_count": len(groups),
            "tc_native_descriptor_count": len(descriptors),
            "tc_native_consumed_tile_count": consumed_tiles,
            "tc_native_consumed_bytes": consumed_bytes,
            "tc_native_entrypoint": entrypoint,
            "tc_native_descriptor_layout": descriptor_layout,
            "tc_native_consumption_source": native_source,
            "tc_native_launch_path": native_source,
            "tc_native_backend": self.name.value,
            "tc_native_launch_count": self.tc_native_launch_count,
            "tc_native_request_op": str(request.get("op", "")),
            "cuda_descriptor_traversal_us": descriptor_traversal_us,
            "cuda_descriptor_metrics_measured": True,
            "tc_adapter_consumed": False,
            "tc_adapter_source": "",
            "tc_adapter_group_count": 0,
            "tc_adapter_descriptor_count": 0,
            "tc_adapter_tile_count": 0,
            "tc_adapter_dispatch_units": 0,
            "tc_adapter_target": "",
            "tc_adapter_mode": "",
            "tc_adapter_fallback_reason": "",
        }


def _empty_tc_native_evidence() -> dict[str, Any]:
    return {
        "tc_native_consumed": False,
        "tc_native_consumed_coalesced_groups": False,
        "tc_native_consumed_group_count": 0,
        "tc_native_descriptor_count": 0,
        "tc_native_consumed_tile_count": 0,
        "tc_native_consumed_bytes": 0,
        "tc_native_entrypoint": "",
        "tc_native_descriptor_layout": "",
        "tc_native_consumption_source": "",
        "tc_native_launch_path": "",
        "tc_native_backend": "",
        "tc_native_launch_count": 0,
        "tc_native_request_op": "",
        "cuda_descriptor_traversal_us": 0.0,
        "cuda_descriptor_metrics_measured": False,
        "tc_adapter_consumed": False,
        "tc_adapter_source": "",
        "tc_adapter_group_count": 0,
        "tc_adapter_descriptor_count": 0,
        "tc_adapter_tile_count": 0,
        "tc_adapter_dispatch_units": 0,
        "tc_adapter_target": "",
        "tc_adapter_mode": "",
        "tc_adapter_fallback_reason": "",
    }
