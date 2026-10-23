# Architecture

The plugin has three main pieces:

- the `FluxScheduler` implementation, which translates AiiDA scheduler calls to
  Flux CLI commands
- the pool registry helpers, which store named pool definitions in AiiDA group
  extras
- the pool CLI, which manages those stored definitions

## Submission Flow

For each submission, the scheduler:

1. parses the generated Flux submit script headers
2. resolves the root AiiDA process PK
3. checks for a named `flux_pool` or an explicit
   `flux_allocation_lifecycle`
4. submits directly with `flux batch` and returns the plain Flux job ID when
   neither is set; ordinary queue, account, and walltime options do not opt in
   to allocation management
5. for explicit `child` or `parent` lifecycle settings, starts or reuses a
   top-level allocation and submits the job through `flux proxy`
6. for `flux_pool`, submits through the pool's named shared allocation

## Pool Runtime Model

Pool definitions are stored in deterministic AiiDA groups. Runtime state is
kept in the same payload and updated when allocations are created, validated,
reused, or cleared.

The lock directory on the remote computer serializes updates to that runtime
state.

## Generated Reference

The docs site includes:

- a generated Click command reference for `aiida-flux`
- a generated Python API reference from `src/aiida_flux_scheduler`

Those pages are intended as implementation reference, while the guides under
`usage/`, `reference/`, and `operations/` should remain the primary user-facing
documentation.
