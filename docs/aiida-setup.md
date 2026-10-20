# AiiDA Setup

The plugin registers the scheduler entry point `flux`. Configure an AiiDA
computer to use that scheduler and a transport that can reach the Flux login
environment.

## Example Computer Setup

The exact host, work directory, and transport options depend on your site, but
the scheduler should be set to `flux`:

```bash
verdi computer setup \
  --label flux-cluster \
  --hostname flux.example.org \
  --transport core.ssh \
  --scheduler flux \
  --work-dir '/scratch/{username}/aiida-run'
```

Then configure the transport:

```bash
verdi computer configure core.ssh flux-cluster
```

## Environment Expectations

The remote environment used by the computer should provide:

- the `flux` command-line tools
- a shell environment capable of running `flux alloc`, `flux jobs`,
  `flux batch`, and `flux cancel`
- a writable home directory for pool lock directories under
  `$HOME/.aiida-flux-scheduler/locks`
- for GPU pools, a Flux resource inventory that exposes GPUs and supports
  `-g` / `--gpus-per-slot` on both `flux alloc` and `flux batch`

## Verify The Scheduler Entry Point

After installation, confirm that AiiDA can see the scheduler plugin:

```bash
verdi plugin list aiida.schedulers
```

Look for the `flux` entry in the scheduler list.
