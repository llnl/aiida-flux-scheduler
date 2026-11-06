# Basic Submission

For direct submissions, provide the resource shape through
`metadata.options.resources`, and provide queue, account, and walltime through
the standard `metadata.options` fields. By default, the scheduler submits the
job directly with `flux batch`, without creating an allocation or using
`flux proxy`. These ordinary resource and scheduler options do not enable
allocation management.

## Required Resource Fields

In practice, direct submission should define:

- `num_machines`
- `num_mpiprocs_per_machine`
- `max_wallclock_seconds`

The scheduler also accepts optional `queue_name` and `account` fields.

## Example

```python
builder.metadata.options.resources = {
    'num_machines': 2,
    'num_mpiprocs_per_machine': 32,
}
builder.metadata.options.max_wallclock_seconds = 3600
builder.metadata.options.queue_name = 'debug'
builder.metadata.options.account = 'my-project'
```

## Runtime Behavior

Without `flux_pool` or an explicit `flux_allocation_lifecycle`, the scheduler
submits the job directly with `flux batch` and returns the plain Flux job ID.
It does not create an allocation or use `flux proxy`. Queue, account, and
walltime options do not change this behavior.

To opt into allocation management, set `flux_allocation_lifecycle` to
`child` or `parent` in `resources`. The `child` lifecycle starts a fresh
top-level allocation for each CalcJob and submits into it with
`flux proxy ... flux batch`; if Flux definitively rejects the child
submission, the scheduler cancels that allocation. The `parent` lifecycle
can reuse an allocation across children of the same root WorkChain. It reuses
an allocation only when its recorded node, CPU, and queue shape is compatible
with the child request; an allocation with insufficient remaining walltime is
replaced. The parent lifecycle cannot be combined with `flux_pool`.

Alternatively, set `flux_pool` to use a named shared allocation. Pool
submissions use that allocation regardless of the direct-allocation lifecycle
setting; do not combine `flux_pool` with `flux_allocation_lifecycle`.
