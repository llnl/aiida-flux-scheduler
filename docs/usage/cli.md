# CLI Reference

The `aiida-flux` command manages Flux pool definitions stored in the AiiDA
database. All commands require an active AiiDA profile.

## Common Tasks

- create or update a pool definition with `aiida-flux pool create`
- inspect configured pools with `list` and `show`
- stop new submissions from using a pool with `disable`
- re-enable a pool with `enable`
- remove a pool definition with `delete`
- preview or compact old terminal submission journals with `journal cleanup`

## Journal retention

Preview cleanup on the selected remote computer (30-day retention by default):

```bash
aiida-flux journal cleanup --computer flux-cluster --retention-days 30
```

Apply after inspecting the JSON report:

```bash
aiida-flux journal cleanup --computer flux-cluster --retention-days 30 --apply
```

Only confirmed terminal journals whose matching CalcJob is sealed in the active
AiiDA profile are eligible. The CalcJob's owner, computer, and recorded scheduler
ID must match. Retention is measured from the journal's last update. Busy,
malformed, unknown-version, active, ambiguous, and incompletely cancelled records
are retained. Missing CalcJobs are retained rather than assumed safe to remove.

Cleanup compacts full version-one intents into version-two terminal receipts;
stable pending aliases still support polling, diagnostics, and duplicate-submit
protection. Working-directory and script details are removed, but identity,
terminal state, child ID, reason, and an intent fingerprint remain. Fingerprints
are rechecked under the persistent guard before compaction, so stale dry-run
approvals cannot compact a changed record.

Allocation receipts and `.guard` files are deliberately retained. Allocation
receipts may still be needed for reconciliation; removing guard inodes breaks
serialization. Cleanup also runs best-effort after accepted pooled submissions:
at most once per day per remote home, with 30-day retention and a rotating batch
of at most 100 journal records. It uses a separate nonblocking maintenance guard,
never the allocation lock. Reservations abandoned by an interrupted submitter
expire after five minutes. Cleanup errors do not fail job submission.
Directory discovery has a time/entry budget, and each automatic helper has a
ten-second deadline. Very large histories that exceed the discovery budget need
the explicit cleanup command rather than delaying submissions.
Computers
without new pooled submissions need explicit maintenance with this command.
The manual command can scan all records and choose a different retention period.

Maintenance does not eliminate the small per-CalcJob alias receipt. Do not manually remove
receipts to force resubmission. Stop old daemon workers before upgrading to
version-two receipt support; older code cannot read compacted receipts.

```{eval-rst}
.. click:: aiida_flux_scheduler.cli:cli
   :prog: aiida-flux
   :nested: full
```
