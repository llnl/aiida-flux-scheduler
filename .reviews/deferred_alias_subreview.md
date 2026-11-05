# Deferred alias focused subreview — findings resolved

Branch: `fix/resilient-flux-pools`; comparison base: `main`. Review and resolution verification: 2026-11-05. Only this report was edited; no source edits, test execution, live Flux operations, or AiiDA runtime mutations were performed.

## Scope

Reviewed allocation generation/receipts, deferred registration, accepted scheduler aliases, monitoring, cancellation, details, and the structured idle watcher in `flux.py` and `deferred.py`. Read related tests and installed AiiDA scheduler, CalcJob update/retrieve tasks, and job-manager consumers. Source/pool lock ownership was assigned to another reviewer. Consulted official Flux jobs/proxy/alloc manuals for state and broker-lifetime semantics.

## Initial snapshot findings — resolved

1. **Pool-definition recursion:** removed; configuration is reread under the allocation lock.
2. **Generation-name protocol mismatch:** names now use `aiida-allocation-<uuid>`, matching helper validation.
3. **Ambiguous allocation direct fallback:** ambiguous/no-ID allocation receipts raise instead of returning a falsy ID to submission dispatch.
4. **Walltime protocol type mismatch:** parsed durations become integer seconds with `math.ceil`, and registration sends an integer.
5. **Incomplete cancellation exposed as terminal:** incomplete cancellation polls raise, and lookup exposes `cancelling` rather than completed cancellation.

## Final implementation recheck findings — resolved

1. **Terminal child polling through a dead pool:** `_get_deferred_job` now short-circuits `done`/`cancelled` to AiiDA DONE before any child proxy query, even when a child ID is retained. `test_terminal_alias_does_not_query_shutdown_pool` covers both terminal states with a child ID.
2. **Parent CLEANUP prematurely considered complete:** helper reports queued for parent CLEANUP and only marks allocation termination at INACTIVE. `test_parent_cleanup_waits_for_inactive` verifies the nonterminal helper response.
3. **Terminal detail querying a dead pool:** terminal detail now returns the durable response envelope and immutable intent instead of proxying to the child. `test_terminal_receipt_details_survive_pool_shutdown` verifies terminal poll/detail with a runner that rejects every broker call.

The watcher uses structured active-job JSON and a full initial idle grace period. The latest compatibility changes embed an AST annotation-stripping bootstrap for older remote Python, quote proxy shell command arguments, use `universal_newlines`, and cancel through `flux cancel`. These were inspected briefly during resolution verification, not subjected to a new full review.

## Current assessment

**No outstanding confirmed defects from this focused review.** All eight reported findings are resolved in the inspected implementation. No additional glaring blocker was noticed during the requested limited verification.

Design: stable UUID aliases, immutable intents, generation-fenced receipts, and nonblocking guards remain appropriate boundaries. Terminal aliases and details no longer require the terminated broker in the reported scenarios. Security: no confirmed exploitable issue was identified within the assigned scope.

## Residual risk / validation handoff

This is static review, not proof of live interoperability. A separate integration agent is validating actual Flux/remote-Python behavior. Real broker readiness, shutdown timing, timeout handling, asynchronous transports, and compatibility behavior should be assessed from that agent's results before merge. Test additions were inspected but not executed by this reviewer.
