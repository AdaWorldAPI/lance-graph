# 2026-10-10 — D-XGP-2: SAP/Odoo hot-plugs and the BillableWorkEntry basis, test-pinned

**Status:** MEASURED. Plan: `.claude/plans/cross-glove-business-parity-v1.md` §C.9.1. Crate: `lance-graph-glove-parity::basis`.

- Both plugs activate as `NoCapabilitiesFor(0x0103)` through `OgarAuthority`.
- The canonical basis is `OgarClassView`'s 13 fields for `0x0103`; nothing minted.
- 9 native → canonical claims, all `Hypothesized`; none binds.
- No temporal field on the class; dates stay unmapped (pinned).
- Disable runs: 4/4 guards red.

OPEN: the temporal role (OGAR); the first `Converted` grade (D-XGP-3).
