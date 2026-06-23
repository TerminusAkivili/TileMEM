# TileMEM Python SDK Quickstart

This quickstart documents the industrial Python SDK surface:

```python
import tilemem as TM
```

For shell workflows, use the thin CLI wrapper:

```bash
tools/tilemem doctor
tools/tilemem verify --quick
```

The SDK turns a MoE model description or local Hugging Face-style checkpoint
config into a TileMEM MIR, deployment manifest, dispatchable tile handles,
and backend capability checks. TileMEM owns tile planning and metadata.
External developers own low-precision kernels and numerical validation.

## 1. Build A Small MoE Model Spec

Start with a compact MoE fixture. The same shape can later be produced by a
model adapter instead of handwritten fields.

```python
import tilemem as TM

spec = TM.model_spec(
    name="quickstart_moe",
    layers=2,
    experts_per_layer=4,
    hidden_size=16,
    intermediate_size=32,
    expert_budget=2,
    workload="mixed",
    tile={
        "hidden_tile": 8,
        "intermediate_tile": 16,
        "shard_count": 2,
        "projection_groups": ["gate_up", "down"],
    },
)
```

## 2. Compile A Plan And Dispatch Summary

`TM.plan()` builds the MIR, manifest, and runtime tile handles in one call. Use
`TM.build_mir()` and `TM.build_manifest()` directly when you want to inspect or
persist each stage separately.

```python
compiled = TM.plan(spec)

mir = compiled.mir
manifest = compiled.manifest
handles = compiled.handles
dispatch = compiled.dispatch_summary(iterations=5)

print(mir.name)
print(manifest["model"])
print(dispatch)
```

Equivalent staged form:

```python
mir = TM.build_mir(spec)
manifest = TM.build_manifest(mir)
handles = TM.build_tile_handles(manifest)
```

Without an external low-precision backend, BF16 tiles remain routed through the
KT fallback path while preserving stable tile IDs and fallback descriptors.

## 3. Register An External CUDA FP8 Backend

Backends declare the formats, scale layouts, projection groups, runtime
entrypoint, and ownership boundaries they support. TileMEM uses this capability
record to decide whether a tile handle is dispatchable to the external runtime
or should keep the fallback path.

```python
registry = TM.BackendRegistry()

TM.register_backend(
    TM.BackendCapability(
        name="customer_cuda",
        formats=["fp8_e4m3_sample"],
        layouts=["block_n32_fp32"],
        scale_granularities=["block"],
        projection_groups=["gate_up"],
        runtime_entrypoint="kernels/gemm_fp8.cu:tilemem_launch_gemm_fp8",
        owns_quantization=True,
        owns_calibration=True,
        owns_quality=True,
        hardware_targets=["cuda_sm90"],
        fallback_dtype="bf16",
    ),
    registry=registry,
)
```

A manifest that assigns compatible tiles to `customer_cuda` will now produce
dispatchable handles:

```python
handles = TM.build_tile_handles(manifest, registry=registry)

for handle in handles:
    if handle.backend == "customer_cuda" and handle.dispatchable:
        launch = {
            "tile": handle.stable_key,
            "weight_offset": handle.weight_offset,
            "weight_bytes": handle.weight_bytes,
            "scale_offset": handle.scale_offset,
            "scale_bytes": handle.scale_bytes,
            "scale_layout": handle.scale_layout,
            "fallback": handle.fallback_backend,
        }
        print(launch)
```

The sample FP8 entrypoint above is an integration contract example. Production
launchers should replace it with a customer-owned CUDA, HIP, vendor library, or
serving-runtime path.

## 4. Read The V0.1 Headline KT Comparison

Use `TM.v0_1_headline_gain()` to replay the public V0.1 BF16 evidence summary.
When the checked-in evidence is present, the best TilePO row is expected to show
approximately a 30% or higher token-throughput gain over the same-budget KT
expert-placement baseline.

```python
headline = TM.v0_1_headline_gain()
best = headline["best"]

print(headline["gate"]["status"])
print(best["workload"], best["experts_per_layer"])
print(best["policy"], best["async_planning"])
print(f'{best["tok_gain_pct"]:.2f}% tok/s over KT')
print(f'{best["p95_reduction_pct"]:.2f}% p95 reduction')
```

The V0.1 headline is BF16 / KT-native evidence. It is not an FP8, F6, F4, or
production Tensor Core quality claim.

## 6. Prepare A Local Hugging Face-Style Checkpoint

TileMEM can infer MoE topology from local `config.json` files and prepare a
serving artifact without downloading a model or launching a server. Supported
topology patterns include OLMoE, Qwen MoE, Mixtral, and generic MoE configs
with standard expert-count fields.

```python
import tilemem as TM

checkpoint_dir = "/path/to/checkpoint"

topology = TM.infer_moe_topology(checkpoint_dir)
spec = TM.model_spec_from_hf_config(checkpoint_dir)
compiled = TM.plan_from_hf_config(checkpoint_dir)

matches = TM.match_checkpoint_weights(
    TM.checkpoint_weight_names(checkpoint_dir),
    spec=spec,
    family=topology.family,
    layers=[0],
    experts=[0],
)
aliases = TM.build_runtime_weight_aliases(
    family=topology.family,
    layers=[0],
    experts=[0],
)

artifact = TM.export_checkpoint_artifact(
    checkpoint_dir,
    out_dir="build/checkpoint_artifact",
    layers=[0],
    experts=[0],
    materialize=False,
)

serving = TM.run_serving_backend(
    checkpoint_dir=checkpoint_dir,
    backend="sglang",
    plan_path=artifact.manifest_path,
    expert_budget=spec.expert_budget,
    execute=False,
)

print(compiled.mir.name)
print(matches.missing)
print(artifact.tile_checkpoint_map_path)
print(aliases["L0:E0"]["sglang"])
print(serving.command)
```

Equivalent CLI:

```bash
tools/tilemem checkpoint prepare \
  --checkpoint-dir /path/to/checkpoint \
  --out-dir build/checkpoint_artifact \
  --backend sglang \
  --dry-run
```

`artifact.tile_checkpoint_map_path` points to a tile-level JSON map. Each
TileMEM tile records its stable key, N range, TileMEM payload offset, source
checkpoint tensors, and source shard files. `materialize=True` copies referenced
checkpoint shard files into the artifact. It does not claim tensor-level
repacking or quality validation; external runtimes still own exact tensor
loading, layout conversion, calibration, and serving lifecycle.

## Ownership Boundary

TileMEM provides:

- model spec ingestion and MIR construction;
- HF config topology inference and checkpoint weight-name mapping;
- deployment manifest generation;
- stable tile IDs, byte offsets, scale metadata fields, and fallback metadata;
- backend capability registration and dispatchable-handle checks;
- dry-run KT/SGLang serving command generation;

External developers provide and validate:

- checkpoint provenance and exact tensor loading/repacking;
- FP8, F6, F4, or other packed-weight kernels;
- quantization algorithms and packed storage layouts;
- scale generation, calibration data, and calibration procedure;
- model quality evaluation and admission gates;
- production runtime integration, profiling, and fallback policy validation.

Keep BF16 or another validated fallback path available for unsupported tiles,
unsupported hardware, or quality-sensitive regions.
