# INDRA → evidence-wiring convergence: reuse `causal_audit`, no new Proposal/Witness type (2026-10-03)

**Status:** DECISION + TEST-PINNED. No new type; one additive field (`AuditedRelation::id`). Tests:
`lance-graph-contract/src/causal_audit.rs` `indra_*` (5). Reference:
`.claude/harvest/indra-reference-wiring.md`.

- **The socket already exists.** `RelationId` + `AuditedRelation` is a truth-free
  proposition; `SupportReceipt { basis, source: EvidenceSourceId, at, strength }` in an
  append-only `SupportLedger` is the witness; `profile()` counts distinct sources and
  leaves `independent_strength = None`. That covers: claim identity ≠ witness identity,
  two papers → two sources, no truth minted. Identity needed one field: only the
  `Unclassified` variant carries a `RelationId`, so `reclassify` used to drop it and two
  classified relations with equal classification and support compared equal (Codex on
  #1318). `AuditedRelation::id` holds it across reclassification; setting it to a
  constant turns `indra_two_papers_one_relation_two_sources` red. Disable-verified: dropping the dedup in
  `profile()` turns `indra_two_readings_of_one_attestation_are_one_source` red; setting
  `independent_strength` turns `indra_witnesses_mint_no_truth` and
  `indra_review_quoting_primary_is_not_yet_recognised_as_an_echo` red.
- **Convention, now documented on `SupportReceipt::source`:** name who attested, not who
  read it. `EvidenceSourceId` is attribution (paper, sensor, submitter): two parsers of
  one sentence, several `parse_readings` readings of one token, and two sentences of one
  paper are one source. It is NOT an evidence-event id; that separate admission-event
  receipt is parked (`.claude/knowledge/parked-designs-841-856.md` §(a)). An earlier
  revision said "name the observation", which conflicted with that parked design and
  would have counted one paper's sentences as independent sources.
- **Four root regimes, unconnected:** `causal_audit` receipts (`EvidenceSourceId`),
  `revision.rs` mask bits (`independent_roots` / `inherited_roots` / `BasisView`),
  `deepnsm-v2::belief::Stamp` (mod-64, gates NARS revision; planner copy too), CE64
  W-slot (6-bit index into a mailbox-keyed `WitnessTable`). The missing piece between the
  first two is a registry `EvidenceSourceId → mask bit`, which `causal_audit` already
  names as "a registry's job".

**OPEN**
- No polarity on `SupportReceipt` (INDRA case 4, negated evidence).
- No derived-from link (case 3, review quoting primary); pinned as a gap.
- No stable `RelationId` derivation for producers (deepnsm `Spo` is vocab-local).
- `arm_to_truth_u8` counts every row as an independent source.
- tesseract-rs `consistency.rs` mints `triple_nars_truth` at extraction, before any ledger.
