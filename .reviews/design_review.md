# Design Review

> **Branch:** `main` (uncommitted changes)
> **Compared Against:** `main`
> **Review Date:** 2026-10-21

## Summary
The lifecycle split keeps named pools out of per-child cleanup, which is the right ownership boundary. The direct-reuse policy, however, derives compatibility from a lossy queue listing and does not preserve the allocation’s requested shape. That makes the documented compatibility promise unenforceable for important resource dimensions.

## Per-Group Findings

### Group: Direct reuse validation

**Files:** `src/aiida_flux_scheduler/flux.py`, `tests/test_flux.py`
**Context reviewed:** `aiida-core/src/aiida/schedulers/datastructures.py`, Flux job-list parser and allocation builder in `flux.py`

**Assessment:** The guard is placed at the correct reuse decision, but its input is insufficient to prove compatibility.

#### Queue-list metadata is not an allocation compatibility record

| | |
|---|---|
| **Priority** | 🟠 High |
| **Location** | `src/aiida_flux_scheduler/flux.py:L489-L516` |
| **Impact** | Correctness and operability |

**What the code does now:** It reconstructs a subset of requested shape from `JobInfo` fields returned by `flux jobs`.

**Why this is a concern:** `JobInfo.num_mpiprocs` is contractually total MPI processes in AiiDA, while the parser assigns Flux `ncores` to it (`L1346`). The listing can omit `ncores` before running, and it does not retain GPU or account request information. The guard therefore cannot reliably compare the envelope supplied to `flux alloc`.

**What I'd consider instead:** Store a normalized direct-allocation envelope keyed by the parent allocation ID when it is created, or retrieve and parse the top-level allocation jobspec. Compare that canonical envelope exhaustively and fail closed if it is missing.

## Overall Assessment

⚠️ **Needs revision** — lifecycle ownership is sound, but compatibility needs a reliable source of allocation requirements.
