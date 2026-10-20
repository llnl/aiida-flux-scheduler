# CLI Reference

The `aiida-flux` command manages Flux pool definitions stored in the AiiDA
database. All commands require an active AiiDA profile.

## Common Tasks

- create or update a pool definition with `aiida-flux pool create`
- inspect configured pools with `list` and `show`
- stop new submissions from using a pool with `disable`
- re-enable a pool with `enable`
- remove a pool definition with `delete`

```{eval-rst}
.. click:: aiida_flux_scheduler.cli:cli
   :prog: aiida-flux
   :nested: full
```
