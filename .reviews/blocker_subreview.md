# Focused blocker review

Scope: six tracked source, test, and documentation changes initially observed as pending on `main`, subsequently committed as `a10fc15` against `41df8de`. Review limited to direct batch default, explicit allocation lifecycle and ownership, shell `cd` for asynchronous transports, plain/nested monitoring IDs, and queue/bank propagation. No source edits or test runs performed; test execution is delegated to the parent session.

## Verdict

No confirmed blocking defects found in the reviewed changes.

## Passes

- Correctness: traced default direct dispatch, allocation creation/reuse and failure cleanup, command construction, and ID normalization through submit, polling, detailed information, and cancellation. The direct command now propagates recorded queue/bank options for previously uploaded scripts.
- Design: explicit lifecycle selection keeps default direct submissions independent from managed and pooled allocations; pool/lifecycle exclusion is enforced for both explicit settings.
- Security: changed directory, script, queue, bank, and submission allocation arguments are quoted. No confirmed newly introduced exploitable defect found within scope.
- Tests: inspected added fixtures and assertions; polling fixtures include the header expected by the existing parser. No test suite executed here.

Residual risk: no live Flux or real asynchronous transport integration was exercised. Existing allocation lifecycle/concurrency behavior outside changed logic was not audited.
