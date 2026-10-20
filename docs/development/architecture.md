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
3. chooses either direct allocation behavior or pool-backed allocation behavior
4. starts or reuses a Flux allocation
5. submits the actual job into that allocation with `flux proxy`

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
