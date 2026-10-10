# 2026-10-10 — D-XGP-3-BILLABLE: the first Converted claim, and 9 of 12 edge targets unminted

**Status:** MEASURED. Plan: `.claude/plans/cross-glove-business-parity-v1.md` §C.9.2.

- SAP `billing_indicator` -> `billable` is `Converted`. The derived lens `lance-graph-sap::bind::BILLABLE` is proven on real rows against the DTO's documented values; an undocumented value is neither true nor false.
- 9 of `BillableWorkEntry`'s 12 edges point at concepts OGAR has not minted (pinned). Only `project`, `about` and `classified_by` can carry an anchored conversion.
- Disable runs: 5/5 red. One was green at first because the test read its expectation from the table under test; it was fixed.

OPEN: minting `Worker` / `Duration` / `Tenant` and the temporal role (OGAR).
