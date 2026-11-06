# AiiDA Flux Scheduler

`aiida_flux_scheduler` is an AiiDA scheduler plugin for Flux. It supports
direct job submission to Flux and shared pool allocations that can be reused
across related submissions.

The full documentation lives in [docs/index.md](docs/index.md) and is intended
to be published as a GitHub Pages site from this repository.

## Install

Install from source:

```bash
git clone https://github.com/LLNL/aiida-flux-scheduler.git
cd aiida-flux-scheduler
pip install -e .
```

Install the documentation toolchain:

```bash
pip install -e .[docs]
```

## Quickstart

1. Register an AiiDA computer that uses the `flux` scheduler entry point.
2. Submit jobs with `metadata.options.resources` values that include at least:
   `num_machines`, `num_mpiprocs_per_machine`, and
   `max_wallclock_seconds`.
3. Optionally create a shared pool with `aiida-flux pool create` and select it
   from job resources with `flux_pool`.

Without `flux_pool` or an explicit `flux_allocation_lifecycle`, jobs are
submitted directly with `flux batch`; the scheduler returns the Flux job ID
and does not create an allocation or proxy. Queue, account, and walltime
options do not change this behavior. Set `flux_allocation_lifecycle` to
`child` or `parent` to opt into allocation management, or use `flux_pool` to
submit through a named shared allocation.

Named pools publish a queued allocation ID without waiting for resources under
the pool lock. Pooled CalcJobs return stable `allocation:pending-CalcJobUUID`
IDs; later scheduler polls submit children when the allocation and broker are
ready. Remote intents and child receipts persist under
`$HOME/.aiida-flux-scheduler/submissions`. Cancellation affects the intent or
child, never the shared allocation; ambiguous responses are reconciled rather
than blindly retried. Plain Flux IDs and existing `allocation:child` IDs remain
supported.

This path requires Linux, `flock`, and `flux python` remotely. Stop old daemon
workers before upgrading: mixed lock protocols are unsupported, and stale
locks from foreign or unknown owners are not automatically stolen based on
age. Use a controlled site rollout; this is not a claim of live-cluster
validation or changed scientific results. See the
[pool schema and operational limits](docs/reference/pool-schema.md).

Start with:

- [Installation guide](docs/install.md)
- [AiiDA setup](docs/aiida-setup.md)
- [Basic submission](docs/usage/basic-submission.md)
- [Pooled allocations](docs/usage/pooled-allocations.md)

## Build The Docs

```bash
pip install -e .[docs]
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Release Information

`LLNL-CODE-2005941`

See the [changelog](CHANGELOG.md) for project history and the
[0.2.0 release notes](docs/release-notes/v0.2.0.md) for upgrade guidance.

AiiDA-Flux-Scheduler is provided under the MIT license. See
[LICENSE.txt](LICENSE.txt).
