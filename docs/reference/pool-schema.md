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
    "state": "active",
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
- `state`
- `last_used_by_parent_pk`
- `last_validated_at`
- `remaining_seconds`
- `updated_at`

Deleting a pool with runtime state still pointing at an allocation requires the
CLI `--force` flag.
