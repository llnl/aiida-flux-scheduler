# Locking And Recovery

Pool-backed submissions use a remote lock directory so only one process updates
pool runtime state at a time.

## Lock Location

Each pool uses a directory under the remote home directory:

```text
$HOME/.aiida-flux-scheduler/locks/<pool-name>-<scope-hash>.lock
```

Pool names are sanitized before they are used in the path. The hash covers the
AiiDA user, computer, and pool name, preventing collisions between same-name
pools that share a remote home directory.

## Lock Metadata

After a lock is acquired, the scheduler writes:

```text
owner.json
```

inside the lock directory. The metadata currently includes:

- the pool name
- the acquisition epoch timestamp
- schema version and a unique acquisition token
- Linux host, boot, PID namespace, user, PID, and process start-time identity

## Stale-Lock Recovery

Allocation creation returns a pending ID immediately; queueing never holds the
pool lock. Acquisition, release, and recovery use a persistent adjacent `.guard`
file with remote `flock`. Never delete that guard file: replacing its inode can
break serialization.

An aged lock is recovered automatically only when its metadata proves that its
owner is dead in the inspecting process's local host/process context. A live
owner, foreign host or namespace, missing metadata, legacy record, or uncertain
inspection is not reclaimed. Age alone is never proof of abandonment. Recovery
compares the inspected metadata again under the guard before removal; release
likewise checks the acquiring owner's token.

Unverifiable ownership needs manual investigation. Confirm the old creator is
no longer active and reconcile any accepted allocation before removing an
abandoned directory. Do not remove a lock while its owner might still run.

Current default behavior:

- lock wait timeout: 60 seconds
- proven-dead local owner minimum age: 300 seconds

These values are internal defaults in the current implementation and are not
yet exposed through the pool CLI.

Remote Linux and shared-filesystem `flock` support are required. Stop old daemon
workers before upgrading; mixed old and new lock protocols are unsupported.
Preserve the version-one submission and allocation journals under
`$HOME/.aiida-flux-scheduler`. Ambiguous accepted submissions reconcile by identity
and do not blindly retry; clearing journals to force a retry risks duplicates.

Allocation replacement never calls `flux cancel` on a shared pool root. When
remaining walltime is insufficient or a pool definition changes, the previous
root is allowed to drain its queued/running children while a new root becomes
the current target. Forced pool deletion removes only the AiiDA definition and
runtime reference; it does not cancel the remote allocation.
