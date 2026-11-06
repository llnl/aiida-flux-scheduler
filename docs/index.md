# AiiDA Flux Scheduler

`aiida_flux_scheduler` integrates AiiDA with Flux and adds optional shared pool
allocations for workflows that benefit from reusing a live Flux allocation.

Use this documentation to:

- install the plugin and docs toolchain
- configure an AiiDA computer to use the `flux` scheduler
- submit jobs directly to Flux
- create and operate reusable pool allocations
- understand runtime state, locking, and recovery behavior
- review changes and upgrade guidance for each release

```{toctree}
:maxdepth: 2
:caption: Getting Started

install
aiida-setup
```

```{toctree}
:maxdepth: 2
:caption: Usage

usage/basic-submission
usage/pooled-allocations
usage/cli
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/scheduler-options
reference/pool-schema
reference/api
```

```{toctree}
:maxdepth: 2
:caption: Operations

operations/troubleshooting
operations/locking-and-recovery
```

```{toctree}
:maxdepth: 1
:caption: Releases

release-notes/v0.2.0
```

```{toctree}
:maxdepth: 2
:caption: Development

development/architecture
```
