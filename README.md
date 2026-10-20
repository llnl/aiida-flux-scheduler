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

AiiDA-Flux-Scheduler is provided under the MIT license. See
[LICENSE.txt](LICENSE.txt).
