# 2026-10-06 — DAV active-observation probe: carrier invariance under Quack (Round 1)

## MEASURED

`crates/jc/examples/dav_active_observation_probe.rs` (`carrier_differential`)
runs the #1344 world through two carriers:

- A: `[Option<bool>; N]` (the #1344 reference);
- B: value lane (`LaneRef::U32`) + validity plane, the plane being the
  horizon's own `independent_roots` word, borrowed with `slice::from_ref`.

Candidate selection in B is one Quack filter, `is_null(VALID) AND
Cmp::Range(SITE)`, lowered by `lance_graph_quack::lower` and executed by
`lance_graph_mask_risc::execute_into` (`Terminal::Keep`). Routes, scoring and
revision are shared code (`trait Observed`).

Equal across 5 apertures × 4 NULL payloads (zero, hidden truth, its
negation, garbage) × 2 enumeration orders: candidate ids, disagreement, EWA
weight, score (bit-exact), order; and, for the DAV and ordinal cycles:
pre-observation disagreement, `RevisionDelta` (kind, effect, resulting),
post-replay disagreement, residual, final observed/value state.

Disable runs, each red: frontier bound to the nullable `VALUE` column (the
`sql_where` silence assert fires); reader passes NULL payload through (A≠B
under the truth payload); leaky twin made non-leaky (the can-fire assert);
ordinal tie-break dropped (the existing #1344 order check fires first);
frontier `hi` off by one.

## OPEN

- `Cmp::Range`'s provenance `Col` (`SITE`) names no resident lane: the
  ordinal axis has no lane, and `Pred::Range` reads none. It is minted by
  hand here, not through `Filter::prefix_facet` + `OrderedLaneWitness`.
- Copies on the B path: the `u32` value lane (N from `projected_claims`),
  the kept mask (`words_for(N)`), `Scratch::for_program`, the set-bit
  `Vec<usize>`, the `Vec<Candidate>`. Round 2 audits these.
