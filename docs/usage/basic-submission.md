# Basic Submission

For direct submissions, provide the scheduler resources through
`metadata.options.resources`. The plugin uses these values to build the Flux
allocation that hosts the submitted batch job.

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
    'max_wallclock_seconds': 3600,
    'queue_name': 'debug',
    'account': 'my-project',
}
```

## Runtime Behavior

When a job is submitted without a pool:

1. the scheduler resolves the root process PK for the current call chain
2. it looks for an existing top-level Flux allocation named `aiida-<root-pk>`
3. if no suitable allocation exists, it starts one with `flux alloc`
4. it submits the job into that allocation with `flux proxy ... flux batch`

If an existing allocation does not have enough remaining walltime for the new
job, the scheduler cancels it and starts a replacement allocation.
