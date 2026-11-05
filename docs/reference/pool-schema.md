# Pool Schema

Pool definitions are stored as AiiDA `Group` records whose extras contain a
plugin-managed payload.

## Group Label

The group label is deterministic:

```text
aiida_flux_scheduler:pool:<user-email>:<computer-label>:<pool-name>
```

## Stored Payload

The plugin stores pool data under the extra key `aiida_flux_scheduler`.

Example payload:

```json
{
  "name": "production",
  "computer": "flux-cluster",
  "user": "user@example.org",
  "enabled": true,
  "timeout": "5m",
  "resources": {
    "num_machines": 4,
    "num_mpiprocs_per_machine": 4,
    "num_cores_per_mpiproc": 8,
    "num_gpus_per_mpiproc": 1,
    "max_wallclock_seconds": 14400,
    "queue_name": "batch",
    "account": "my-project"
  },
  "runtime": {
    "current_flux_id": "f123456",
    "generation": "ad85bb55-7e1e-4f4d-9aae-563045d69fb8",
    "job_name": "aiida-allocation-ad85bb55-7e1e-4f4d-9aae-563045d69fb8",
    "state": "running",
    "last_used_by_parent_pk": 1001,
    "last_validated_at": "2026-03-19T12:00:00+00:00",
    "remaining_seconds": 8200,
    "updated_at": "2026-03-19T12:00:00+00:00"
  }
}
```

The three required resource fields are `num_machines`,
`num_mpiprocs_per_machine`, and `max_wallclock_seconds`. Optional
`num_cores_per_mpiproc` and `num_gpus_per_mpiproc` values must be positive
integers. Together, these fields define the capacity retained by the outer
`flux alloc`; they are not per-child defaults.

Updating a pool's resources or timeout clears its current runtime reference.
The old allocation is not canceled: already submitted children can drain, and
the next submission creates an allocation matching the new definition.

## Runtime Fields

The `runtime` mapping is managed by the scheduler and may be empty for a newly
created pool. Fields currently used by the plugin include:

- `current_flux_id`
- `generation`: identity of this allocation attempt, retained for reconciliation.
- `job_name`: generation-specific Flux name, distinguishing replacement allocations.
- `state`: `creating`, `queued`, or `running` during allocation creation and reuse.
- `last_used_by_parent_pk`
- `last_validated_at`
- `remaining_seconds`
- `updated_at`

The idle watcher counts active jobs using Flux JSON output. A full idle-timeout
grace period starts at broker startup and resets whenever active jobs are seen;
query failures defer shutdown. Timeouts use positive `s`, `m`, `h`, or `d` durations.

Deleting a pool with runtime state still pointing at an allocation requires the
CLI `--force` flag.

## Deferred Submission Records

Named pools submit the outer jobspec with `flux alloc --dry-run | flux job
submit`, publishing a pending allocation ID under a short per-pool lock. They
do not hold that lock while waiting in the queue. The generation and job name
are recorded before submission; an uncertain allocation response must be
reconciled rather than blindly creating another allocation.

Each pooled CalcJob registers an immutable remote intent under
`$HOME/.aiida-flux-scheduler/submissions`. The version-one journal records the
allocation and CalcJob identities, working directory and submit script, plus
submission state, child ID receipt, and cancellation state. These remote
records are separate from the AiiDA Group runtime; preserve them across daemon
restarts and do not edit them to force resubmission.
Allocation submission receipts live under `$HOME/.aiida-flux-scheduler/allocations`.
Both receipt types use exact version-one records and persistent guard files.

The scheduler ID remains `allocation:pending-CalcJobUUID`. Later `get_jobs`
polls submit the child once the allocation and nested broker are ready, then
resolve the same stable alias through its recorded actual child ID. A lost or
ambiguous child response is reconciled by identity, never blindly retried.
Cancellation stops the intent or its child, not the shared allocation. Plain
Flux IDs and existing `allocation:child` IDs remain supported.

## Lock Safety and Rollout

Pool lock metadata uses `schema_version: 1`, a unique acquisition token, and
Linux process identity. Metadata creation, release, and recovery are guarded
by a persistent `flock` guard. Automatic reclamation requires proof that an
aged lock's owner is dead in the same local host/process context; age alone
never justifies removing foreign, unknown, or legacy ownership. There is no
automatic cross-host stale-lock stealing. Investigate unresolved ownership
before manual recovery, and retain the guard file.

Remote execution requires Linux, `flock`, and `flux python`. Stop old daemon
workers before upgrading: mixed lock protocols are unsupported. Use a
controlled rollout to verify queueing, reconciliation, and cancellation at
your site. This describes scheduler lifecycle behavior, not live-cluster
validation or any change in scientific compatibility or results.
