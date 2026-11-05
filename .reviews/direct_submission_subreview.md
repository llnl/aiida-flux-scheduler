# Direct submission focused review

Scope: current uncommitted source/tests/docs against `main`; direct-default dispatch, lifecycle omission, plain IDs, queue/bank directives, and previously generated scripts. No source changes made.

## Findings

1. **High: previously uploaded jobs change destination without regenerating their headers.** `src/aiida_flux_scheduler/flux.py:410-412` treats missing lifecycle as direct submission, while `:214-223` only adds queue/bank to newly generated direct scripts. The previous header implementation omitted both. A CalcJob paused after upload and before submission has an old remote script without queue/account, and ordinary existing metadata without a lifecycle. Resuming now submits that script directly to the top-level Flux instance, silently dropping its requested queue/bank. AiiDA's `execmanager.submit_calculation` uses the already uploaded remote script rather than calling presubmit again. Handle this migration explicitly (e.g. provide standard queue/bank as direct batch command arguments, detect/version old scripts, or require re-upload), and add a pre-generated-script resume regression test.

2. **Medium: both new polling test variants deterministically fail.** `tests/test_flux.py:512-514` supplies a single job row without a header. `_parse_joblist_output` (`src/aiida_flux_scheduler/flux.py:1345`) discards the first line as a header, so `get_jobs(..., as_dict=True)` returns `{}` and the membership assertion fails for direct and nested IDs. Use `HEADER\n` plus a valid 13-field row matching the production field order; the current row also has misplaced numeric fields that will fail once the header is fixed. Existing `test_parse_joblist_output_parses_flux_time_fields` demonstrates a valid fixture.

3. **Low: documented pool/lifecycle exclusion is not enforced for parent.** `src/aiida_flux_scheduler/flux.py:660-666` rejects only `flux_allocation_lifecycle='child'`, while `docs/reference/scheduler-options.md` says neither lifecycle may be combined with `flux_pool`. A `parent` + pool request is silently accepted and routed to the shared pool. Reject both explicit lifecycle settings (preferably during resource validation) or document the intentional override, with coverage of both combinations.

## Remaining assessment

The omission/None direct branch, plain-ID query command, and newly generated queue/bank headers are internally consistent. No additional security or architectural blockers found in this narrowly scoped change. Runtime verification was blocked before test collection by the environment's SQLAlchemy extension requiring unavailable `libfi.so.1`; the polling test failures above follow directly from the fixture/parser code.
