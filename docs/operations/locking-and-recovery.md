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
- the UTC acquisition timestamp
- the acquisition epoch timestamp

## Stale-Lock Recovery

If lock acquisition fails because the directory already exists, the scheduler
checks the lock age. It prefers the timestamp recorded in `owner.json` and
falls back to the directory modification time for older lock directories.

Automatic stale-lock removal is disabled by default. Creating an outer
allocation can legitimately wait in the site queue for longer than a fixed
age threshold, so treating age alone as proof of a stale owner can create two
roots for one pool. After a submitting process crashes, an administrator must
confirm no allocation creator is still running and remove that pool's remote
lock directory before retrying.

Current default behavior:

- lock wait timeout: 60 seconds
- stale-lock recovery: disabled unless explicitly configured

These values are internal defaults in the current implementation and are not
yet exposed through the pool CLI.

Allocation replacement never calls `flux cancel` on a shared pool root. When
remaining walltime is insufficient or a pool definition changes, the previous
root is allowed to drain its queued/running children while a new root becomes
the current target. Forced pool deletion removes only the AiiDA definition and
runtime reference; it does not cancel the remote allocation.
