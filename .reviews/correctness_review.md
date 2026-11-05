# Correctness Review

> **Branch:** `main` (uncommitted changes)
> **Compared Against:** `main`
> **Review Date:** 2026-10-21
> **Files Changed:** 4

## Bugs and Logic Errors

### Compatibility validation can reuse an allocation without the requested GPUs or CPUs

| | |
|---|---|
| **File** | `src/aiida_flux_scheduler/flux.py:L489-L516` |
| **Severity** | 🟠 High |

**What's wrong:**
The new validator never compares `num_gpus_per_mpiproc`, although `_start_allocation` uses it to request `flux alloc -g`. It also treats an unavailable `job_info.num_mpiprocs` as compatible (`if requested_slots is not None and job_info.num_mpiprocs is not None`). The plugin itself documents the source field as `ncores` “if already running” (`L115`), so queued allocations commonly lack the value needed to validate CPU shape.

**What will happen:**
A parent allocation created for a CPU-only child can be reused for a later GPU child, or an allocation with a different CPU shape can be reused while it is pending. The subsequent `flux batch` asks the nested instance for the child resources, but the parent allocation cannot supply them; the child will be unschedulable or fail despite the new compatibility guard.

**Suggested fix:**
Persist the requested direct-allocation resource envelope when creating it, or query an authoritative Flux jobspec, and compare nodes, tasks, cores-per-task, GPUs-per-task, queue, and account. If a requested dimension cannot be established, return `False` rather than accepting reuse.

---

### Cleanup cancels a real child job after an ambiguous submission result

| | |
|---|---|
| **File** | `src/aiida_flux_scheduler/flux.py:L1220-L1231` |
| **Severity** | 🟠 High |

**What's wrong:**
The broad `except Exception` treats transport failures exactly like a confirmed non-zero `flux batch` result. A remote command can submit the child successfully and then fail locally while reading stdout (e.g., SSH disconnect or timeout). In that case no child ID is returned, but the allocation already contains a valid child job.

**What will happen:**
The cleanup command cancels the newly created allocation and thereby kills the successfully submitted child calculation. AiiDA receives the transport exception, while the actual calculation is prematurely terminated.

**Suggested fix:**
Only cancel when `flux batch` has returned a confirmed failure (the non-zero result processed by `_parse_submit_output`). For transport/parse ambiguity, retain the allocation and raise an error that explicitly records the uncertain submission, or query the allocation for the expected child before deciding whether cleanup is safe.

## Test Coverage Concerns

The added tests cover only a returned `(1, '', 'batch failed')` result and a manually constructed fully populated `JobInfo`. Add tests for (1) CPU-only allocation followed by a GPU request, (2) pending allocation with missing `ncores`, and (3) an exception raised by `exec_command_wait` after remote dispatch; the last must not blindly cancel a potentially running child.
