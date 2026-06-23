from __future__ import annotations

from typing import Any

from tilepo.mir import Backend
from .common import matmul, quantized_matmul, routed_moe, validate_manifest


class CUDABackend:
    name = Backend.CUDA

    def __init__(self, require_native: bool = False) -> None:
        self.require_native = require_native
        self.launch_count = 0
        self.native_available = False

    def consume_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        return validate_manifest(manifest)

    def execute(self, request: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
        self.consume_manifest(manifest)
        if self.require_native and not self.native_available:
            raise RuntimeError("TilePO CUDA native backend is not built in this environment")
        self.launch_count += 1
        dtype = str(request.get("dtype", "bf16"))
        if request.get("op") == "matmul":
            a = request["a"]
            b = request["b"]
            output = matmul(a, b) if dtype == "bf16" else quantized_matmul(a, b, dtype)
            return {"output": output, "backend": self.name.value, "dtype": dtype}
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
                "native": self.native_available,
                "calibration_required": dtype in {"fp8", "mxfp4"},
            }
        return {"output": request.get("payload"), "backend": self.name.value, "dtype": dtype}
