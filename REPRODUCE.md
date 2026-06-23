# Reproduce TileMEM / TilePO V0.1

This repository is organized so a fresh clone can verify the checked-in V0.1
BF16 / KT-native evidence without downloading models or starting a server.

## One Command

```bash
tools/tilemem evidence verify --json
```

Expected evidence matrix:

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

The command regenerates:

- `build/release_evidence/tilepo_ablation_summary.json`
- `build/release_evidence/tilepo_ablation_report.md`

## Full Offline Gate

```bash
tools/tilemem doctor
tools/tilemem verify --quick
bash scripts/verify_artifact.sh
```

`verify_artifact.sh` checks the SDK, CLI, SKILL files, release package, checksum
manifest, and the same 210-row V0.1 evidence gate.

## Native CMake Gate

CUDA users can also run:

```bash
cmake -S . -B build/cmake -DTILEMEM_SM=120
cmake --build build/cmake -j
ctest --test-dir build/cmake --output-on-failure
```

Adjust `TILEMEM_SM` for the target GPU.
