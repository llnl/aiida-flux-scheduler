# Scheduler Options

The Flux scheduler plugin reads allocation settings from
`metadata.options.resources`.

## Supported Keys

| Key | Type | Required | Purpose |
| --- | --- | --- | --- |
| `num_machines` | integer | Yes for direct submission and pool creation | Number of nodes requested for the top-level Flux allocation. |
| `num_mpiprocs_per_machine` | integer | Yes for direct submission and pool creation | Number of MPI ranks per node used to compute the `flux alloc -n` value. |
| `num_cores_per_mpiproc` | integer | No | CPU cores per MPI rank/Flux slot. Generates `#flux: -c N`. |
| `num_gpus_per_mpiproc` | integer | No | GPU devices per MPI rank/Flux slot. Generates `#flux: -g N`. Must be at least one when set. |
| `max_wallclock_seconds` | integer | Yes in practice for direct submission and required for pool creation | Wallclock limit for the top-level Flux allocation. |
| `queue_name` | string | No | Flux queue or partition passed to `flux alloc -q`. |
| `account` | string | No | Flux account or bank passed to `flux alloc -B`. |
| `flux_pool` | string | No | Name of a stored pool definition to load instead of per-job allocation settings. |
| `num_mpi_procs_per_machine` | integer | Compatibility alias | Legacy spelling accepted when resolving allocation parameters. |

## Example

```python
builder.metadata.options.resources = {
    'num_machines': 2,
    'num_mpiprocs_per_machine': 64,
    'max_wallclock_seconds': 7200,
    'queue_name': 'batch',
    'account': 'my-project',
}
```

For a four-rank GPU job with one GPU and eight CPU cores per rank:

```python
builder.metadata.options.resources = {
    'flux_pool': 'production',
    'num_machines': 1,
    'num_mpiprocs_per_machine': 4,
    'num_cores_per_mpiproc': 8,
    'num_gpus_per_mpiproc': 1,
}
```

## Notes

- `flux_pool` changes the allocation source: the pool definition supplies the
  allocation resources, timeout, and enablement state.
- `num_cores_per_mpiproc` is used for both child batch requests and pool
  allocation slots.
- Flux `batch` and `alloc` describe GPU resources per slot. This plugin maps an
  AiiDA MPI process to a Flux slot and emits `-g`, whose long form in current
  Flux documentation is `--gpus-per-slot`. It intentionally does not emit an
  assumed generic `--gpus` option.
- `max_memory_kb` is an AiiDA job-template field, but upstream Flux has no
  standard memory request option for `flux batch`/`flux alloc`; this scheduler
  therefore does not generate a memory directive. Use site policy or
  `custom_scheduler_commands` only when the target installation provides a
  documented extension.
- `custom_scheduler_commands` remains available for site-specific constraints
  and attributes. GPU counts should use `num_gpus_per_mpiproc` instead.
