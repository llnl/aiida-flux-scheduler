# Pooled Allocations

Pools allow multiple submissions to share a named Flux allocation that is
tracked in the AiiDA database. A pool definition belongs to a specific user and
computer.

## Create A Pool

```bash
aiida-flux pool create production \
  --computer flux-cluster \
  --num-machines 4 \
  --num-mpiprocs-per-machine 4 \
  --num-cores-per-mpiproc 8 \
  --num-gpus-per-mpiproc 1 \
  --max-wallclock-seconds 14400 \
  --queue-name batch \
  --account my-project
```

Pool records can also be listed, inspected, enabled, disabled, and deleted via
the CLI. See [CLI reference](cli.md).

## Select A Pool In Job Resources

Set `flux_pool` together with each child's own resource shape. The pool values
describe total retained capacity; they do not replace the child request in the
generated batch script.

CPU-only child (no GPU key):

```python
cpu_builder.metadata.options.resources = {
    'flux_pool': 'production',
    'num_machines': 1,
    'num_mpiprocs_per_machine': 8,
    'num_cores_per_mpiproc': 2,
}
```

GPU VASP/LAMMPS/Allegro child:

```python
gpu_builder.metadata.options.resources = {
    'flux_pool': 'production',
    'num_machines': 1,
    'num_mpiprocs_per_machine': 4,
    'num_cores_per_mpiproc': 8,
    'num_gpus_per_mpiproc': 1,
}
```

Both jobs are submitted immediately with `flux proxy <pool-id> flux batch ...`.
The plugin does not reserve resources itself and does not wait for earlier
children. Flux sees the per-child node, slot, core, and GPU requests and decides
when and where each job runs. A CPU-only child can therefore consume cores left
free by a GPU child, subject to the site's Flux scheduler and placement policy.
Requests that can never fit inside the configured pool envelope are rejected at
submission rather than remaining queued forever.

## Allocation Reuse Rules

For a pooled submission, the scheduler:

1. acquires a per-pool remote lock
2. checks the pool runtime state for `current_flux_id`
3. validates that the referenced Flux allocation still exists
4. reuses the allocation if the remaining walltime is sufficient
5. otherwise clears runtime state and starts a replacement allocation while
   leaving the old allocation and all of its children to drain

Disabled pools fail at submission time. Missing pool definitions also fail at
submission time with a scheduler error.

The pool lock only protects allocation discovery and runtime-state updates. It
does not serialize child jobs. Lock names are scoped by user, computer, and pool
to prevent same-name pools from colliding on a shared remote home directory.

## GPU Syntax And Site Verification

Current upstream Flux documents `-g N` / `--gpus-per-slot=N` for `flux batch`
and `flux alloc`. A slot is the plugin's MPI-process resource unit. The target
site was not reachable from the development host, so administrators must verify
that their installed Flux version and resource inventory support this syntax:

```bash
flux --version
flux batch --help | grep -E 'gpus-per-slot|-g'
flux alloc --help | grep -E 'gpus-per-slot|-g'
flux resource list
```

The enclosing Flux instance must advertise GPU resources, and the selected
queue must provide GPU nodes. If the site requires a constraint such as a node
property, add only that site-specific constraint through
`custom_scheduler_commands`; do not duplicate the GPU count there.
