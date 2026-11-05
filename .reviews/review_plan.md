# Review Plan

## Change Summary
The uncommitted change adds a direct-allocation lifecycle setting. `child` creates an allocation for each CalcJob and attempts to cancel a newly created direct allocation if batch submission errors; `parent` remains the default. It also adds direct allocation shape checks before reuse and replaces transport `chdir` calls with shell `cd` commands.

## Logical Groups

| Group Name | Files | Purpose |
|---|---|---|
| Allocation lifecycle | `src/aiida_flux_scheduler/flux.py`, `tests/test_flux.py` | Create/cancel child-owned allocations and preserve pooled allocation ownership. |
| Direct reuse validation | `src/aiida_flux_scheduler/flux.py`, `tests/test_flux.py` | Reject reusing a parent allocation with incompatible resources. |
| Documentation | `docs/reference/scheduler-options.md`, `docs/usage/basic-submission.md` | Document lifecycle semantics. |

## Context Files to Read

| Group | Context Files | Why Needed |
|---|---|---|
| Allocation lifecycle | `aiida-core/src/aiida/schedulers/scheduler.py`, `.../plugins/bash.py` | Scheduler submission contract. |
| Direct reuse validation | `aiida-core/src/aiida/schedulers/datastructures.py`, parser and allocation command in `flux.py` | Meaning and availability of queried resource fields. |

## Risk Map

| Group | Risk | Why |
|---|---|---|
| Allocation lifecycle | High | A cleanup decision can terminate actual scientific calculations. |
| Direct reuse validation | High | Incorrect reuse can submit jobs into allocations that do not supply requested resources. |
| Documentation | Low | Describes the new behavior. |

## Review Focus Areas

### Correctness Pass Focus
1. Distinguish a confirmed `flux batch` failure from an ambiguous transport/output failure before cancelling an allocation.
2. Ensure reuse validation covers all allocation resources and fails closed when scheduler data is unavailable.
3. Check AiiDA scheduler/transport submission contracts.
