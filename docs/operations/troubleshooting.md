# Troubleshooting

## `No Flux pool named ... exists`

The selected pool does not exist for the current AiiDA user and computer.
Confirm the pool name, the computer label, and the active AiiDA profile.

## `Flux pool ... is disabled`

The pool definition exists but is marked disabled. Re-enable it with:

```bash
aiida-flux pool enable <pool-name> --computer <computer-label>
```

## `Timed out waiting for Flux pool lock`

Another process is using the pool, or a stale lock directory has not yet aged
past the scheduler recovery threshold. Retry the submission after a short wait
and inspect the lock path described in
[Locking and recovery](locking-and-recovery.md).

## Allocation Was Replaced Unexpectedly

The scheduler replaces a pooled or parent allocation when:

- the referenced Flux allocation no longer exists
- the remaining walltime is lower than the new job's required walltime

This is expected behavior and is intended to keep submissions from landing in a
dead or undersized allocation.

## Flux Commands Fail On The Remote Computer

The AiiDA computer environment must be able to run `flux alloc`, `flux jobs`,
`flux batch`, and `flux cancel`. Check the remote shell initialization and any
module-loading requirements for the target system.
