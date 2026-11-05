# Code Review Summary

> **Branch:** `main` (uncommitted changes)
> **Compared Against:** `main`
> **Review Date:** 2026-10-21
> **Files Changed:** 4

## Overall Verdict

⚠️ **Needs work** — two high-severity correctness issues can respectively run a child in an undersized allocation and terminate a successfully submitted child after an uncertain transport failure.

## Top Priorities

### 1. Make direct-allocation compatibility authoritative and fail closed
**Why it matters:** GPU and pending-allocation CPU mismatches pass the current guard.
**What to do:** Persist/query the full allocation envelope and compare every resource dimension; reject when a dimension cannot be verified.
**Details:** `correctness_review.md`, “Compatibility validation can reuse an allocation without the requested GPUs or CPUs”.

### 2. Do not cancel on an ambiguous `flux batch` transport exception
**Why it matters:** The remote scheduler may have accepted the child before the client loses the response.
**What to do:** Limit cleanup to confirmed batch failures, with reconciliation/query behavior for unknown submission outcomes.
**Details:** `correctness_review.md`, “Cleanup cancels a real child job after an ambiguous submission result”.

## What's Good
- The explicit ownership flag prevents a failed pooled child from cancelling a shared named-pool allocation.
- The new shell `cd` approach avoids relying on a `chdir` method unavailable in asynchronous transports.
