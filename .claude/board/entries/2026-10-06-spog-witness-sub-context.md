# 2026-10-06 — D-SPOG-W-0: the Witness slot as a SPOG sub-context under classid-G

## MEASURED

`crates/cognitive-shader-driver/examples/spog_witness_probe.rs`: 7 tests,
6 disable runs red.

- A SPOG coordinate is read from `(classid, CausalEdge64)` with no stored
  column and no allocation: S/P/O are the resident bytes, G is
  `graph_of(classid)` (the existing "g via classid" rule in `spog_tenants.rs`),
  and the Witness slot is a sub-context inside G.
- The sub-context reading is declared per class. An undeclared class, a class
  declaring another reading of the slot, and v1 / unknown provenance refuse.
  Slot 0 reads as no anchor, as the shipped `w_slot` accessor documents.
- G never moves with the slot; the slot never moves with the classid.
- Against the real `MailboxSoA::apply_edges`: within one graph, accept ⟺ same
  SPOG context on all 64 × 64 slot pairs (64 accepted, 4032 dropped). Across
  graphs they diverge: routing accepts a foreign graph's edge with the same
  slot. Recorded as `ISS-MAILBOX-ROUTES-WITNESS-WITHOUT-GRAPH`.

## OPEN

- Whether the per-class declaration becomes a contract declaration beside
  `band_reading`.
- The cross-graph routing divergence (production change; see ISSUES).
- Sub-context meaning (corpus, frame, schema) stays the consumer's binding.
