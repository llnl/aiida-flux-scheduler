# Pool recovery validation

## Changes

- Publish queued allocation IDs under short pool locks.
- Register durable CalcJob intents with stable pending scheduler aliases.
- Reconcile allocation and child acceptance by immutable identities; never
  blindly resubmit ambiguous requests.
- Guard token-checked lock release and proven-dead local-owner recovery.
- Keep terminal receipts readable after broker shutdown and delay retrieval
  until parent cleanup finishes.
- Use structured idle-job checks with startup grace.

## Validation

- AiiDA 2.9.3 test suite: 122 passed.
- Ruff source/test checks passed.
- All pre-commit hooks passed, including mypy in its isolated environment.
- Package wheel and source build passed; lockfile check passed.
- Strict Sphinx build passed.
- Two isolated real Flux roundtrips passed with the unmodified production helper
  and Flux Python 3.6.8. Three deferred aliases shared a held allocation, stayed
  queued until release, completed without duplicates, and supported child-only
  cancellation. Terminal details remained readable after allocation shutdown.

The repository virtualenv has a pre-existing SQLAlchemy import failure involving
`libfi.so.1`; tests used the site's working AiiDA interpreter. An unrestricted
mypy invocation in an adjacent AiiDA environment reports an existing nullable
Group PK issue in `pools.py`; the configured pre-commit mypy check passes.

## Rollout boundaries

- No HPC jobs were submitted and no existing AiiDA profiles or daemon processes
  were modified during validation. The bounded isolated Flux allocations were
  explicitly cleaned up.
- Stop old workers before upgrade; mixed lock protocols are unsupported.
- Remote Linux, `flock`, and persistent shared-home journals are required.
- Foreign or unverifiable lock owners are not reclaimed automatically.
- Ambiguous acceptance without a discoverable job remains fenced for manual
  investigation rather than risking duplication.
- Avoid concurrent pool-definition edits during submissions; Group-extra writes
  are not database compare-and-set transactions.
- A controlled site campaign remains a deployment gate; isolated plumbing tests
  do not establish scientific model accuracy or convergence.

Existing review artifacts, lockfile, and direct-submission work are included in
the branch at the user's request. Historical review reports describe prior
snapshots, not necessarily unresolved findings in this implementation.
