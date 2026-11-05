# Security Review

> **Branch:** `main` (uncommitted changes)
> **Compared Against:** `main`
> **Review Date:** 2026-10-21

## Threat Surface Assessment
The change executes scheduler commands built from AiiDA-managed paths and identifiers. The new path interpolation uses `escape_for_bash`; no concrete injection or authorization regression was found in the reviewed diff.

## Security Posture Summary
No security findings. The substantive risks are scheduler correctness and destructive cleanup after an ambiguous transport failure.
