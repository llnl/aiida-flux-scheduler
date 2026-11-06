# Changelog

All notable changes to AiiDA Flux Scheduler are documented here. This project
follows [Semantic Versioning](https://semver.org/) while it is in the `0.x`
series; minor releases may include user-visible behavior changes.

## [0.2.0] - 2026-11-06

### Added

- Add named shared Flux pools, GPU-aware resource requests, and pool-management
  commands.
- Add durable deferred submission and cancellation reconciliation for pooled
  jobs, including stable pending scheduler IDs.
- Add safe remote pool locks and manual/automatic cleanup of eligible terminal
  submission journals.
- Add explicit `child` and `parent` allocation lifecycles.

### Changed

- Require Python 3.12 or later.
- Submit jobs directly with `flux batch` by default. Set
  `flux_allocation_lifecycle` to `child` or `parent`, or select a named pool
  with `flux_pool`, to use managed allocations.
- Generated walltime limits now include explicit seconds, avoiding Flux's
  interpretation of unitless limits as minutes.
- Apply manual journal cleanup approvals in bounded batches.

### Fixed

- A failed child submission no longer cancels a reusable parent allocation
  that may already serve another calculation.
- Cancellation and polling can resolve deferred jobs after the enclosing pool
  allocation has shut down, without requiring its nested broker.
- Reject a pooled job that no longer fits its refreshed pool definition before
  reusing or creating an allocation.

### Compatibility

- Stop old daemon workers before upgrading; mixed pool-lock protocols are
  unsupported, and older code cannot read compacted version-two terminal
  receipts.
- Preserve journals under `$HOME/.aiida-flux-scheduler`. Do not remove or edit
  them to force a retry; ambiguous submissions require reconciliation.
- Named pools require Linux, `flock`, and `flux python` on the remote system.
- Verify the installed Flux version, queue, and resource inventory at the
  target site. This release has not been validated against a live production
  Flux cluster.
