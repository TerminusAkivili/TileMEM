---
name: tilemem-environment-setup
description: Use when setting up, diagnosing, or verifying a local TileMEM checkout, including Python, CUDA, CMake, WSL/GPU visibility, offline artifact checks, or first-run customer onboarding.
---

# TileMEM Environment Setup

## Overview

Bring a TileMEM checkout to a known-good state before changing code or running model experiments. Prefer fast offline verification first, then optional CUDA/CMake and full artifact gates.

## Workflow

1. Find the repo root and inspect current state:

```bash
pwd
git status --short --branch
python3 --version
```

2. Use the project entrypoint before ad hoc commands:

```bash
tools/tilemem doctor
tools/tilemem verify --quick
tools/tilemem evidence verify
```

3. If quick verification passes, run the offline SDK/demo path:

```bash
python3 examples/tilemem_industrial_quickstart.py \
  --out-json build/tilemem_industrial_quickstart.json
```

4. Run the full artifact gate only when the release package is present or the user asks for full verification:

```bash
bash scripts/verify_artifact.sh
```

Expected evidence anchors:

```text
Workloads: mixed, long_context
Experts: 2, 4, 6, 8, 10
Policies: kt_expert, tilepo_coarse, tilepo_fine, tilepo_hybrid
Async planning: off, on
Repeats: 3
Request count: 5
Rows: 210 / 210 real success
Gate: PASS
Serving precision: BF16 / KT-native path
```

Expected full-gate anchors include `TilePO V0.1 ablation gate: PASS`, `Rows: 210/210`, and `TileMEM / TilePO artifact verification passed.`

## Optional Native CUDA/CMake Layer

Only run this after Python/offline checks pass and CUDA is visible:

```bash
nvidia-smi || true
nvcc --version || true
cmake -S . -B build/cmake -DTILEMEM_SM=120
cmake --build build/cmake -j
ctest --test-dir build/cmake --output-on-failure
```

Adjust `TILEMEM_SM` for the target GPU. Do not make CMake/CUDA success a prerequisite for basic SDK or artifact inspection unless the task specifically requires native kernels.

## Guardrails

- Do not download large checkpoints or install heavyweight serving stacks unless the user explicitly asks.
- Do not run `--execute` serving paths during environment setup; prefer dry-run command generation.
- Do not hide missing GPU tools. Report `nvidia-smi`, `nvcc`, and CUDA/CMake status separately from Python SDK status.
- If `bash scripts/verify_artifact.sh` fails only because `publish/` artifacts are absent, say quick verification passed and full release verification needs the packaged artifact.

## Quick Diagnosis Table

| Symptom | First Check |
|---|---|
| `tilemem` import fails | `tools/tilemem doctor --json` |
| Python tests fail | `tools/tilemem verify --quick` |
| Full verify missing files | inspect `publish/TileMEM_TilePO_V0_1_20260611` |
| CUDA build fails | check `TILEMEM_SM`, `nvcc --version`, CMake version |
| Customer asks “is it usable?” | run doctor, quick verify, quickstart JSON |
