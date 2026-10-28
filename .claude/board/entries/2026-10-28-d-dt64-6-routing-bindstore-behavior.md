# D-DT64-6: DR64 routing, DT64 bind store, per-cell WHEN×ON×DO behaviour, OGIT reference predicates

**Status:** PROPOSED / DOCS ONLY. Plan: [d-dt64-6-routing-bindstore-behavior-v1.md](../../plans/d-dt64-6-routing-bindstore-behavior-v1.md).
No code, ABI, test, runtime or merge authorization. Read at lance-graph `main`
`72697cc31`, odoo-rs `main` `558f12a`, OGAR `main` `6b43a48`; lance-graph #1814
and #1819 have since merged as docs and were read in full on `main`.

## What this records

Eight user decisions from session 2026-10-28, in DECISION / SCOPE / BASIS /
REVISIT WHEN form (plan §1):

1. DT64 has three levels, all 64×64×64, reached by focused global→local:
   DR64³ Identity(+value) × Property(+value); Identity × Property × Predicate;
   Predicate³ = WHEN × ON × DO.
2. The primary view is DR#3 Identity × Properties. Identity × Identity and
   Properties × Properties are optional derived views.
3. WHEN × ON × DO attaches per DR#3 cell as the registry over methods.
4. DR64 = direct routing (mapping table); DT64 = DTO bind store for all
   holographic addresses; behaviour registered as classid → polyfill.
5. classid → behaviour is permanently immutable; versioned under a fixed
   concept via a derived DTO classid; registration via `hotplug.rs`.
6. Global behaviour needs no classid and lives in lance-graph; consumers need
   a classid through the hotplug pattern.
7. OGIT is immutable and read unpinned.
8. DR64 references existing identities and mints none. The global→local
   prefetch amortizes by holding a copy of the register (identity
   references and local positions only, no values), valid while its frame
   is current and rebuilt when the frame check refuses.

## Findings that shaped the plan (VERIFIED-IN-CODE unless marked)

- The three DT64 plans do not name ON / WHEN / DO, but ON×WHEN×DO is
  established in merged lance-graph research docs: the D-CFA series
  (`docs/research/counterfactual-behavior/`, incl. #1814) and the EAW
  series (`docs/research/epistemic-awareness/`), plus #1819. As a registry
  it exists as the odoo-rs lab design `docs/DT64-ON-WHEN-DO-REGISTRY.md`
  (#44 / #45 merged, test-only). #1814 rules out folding WHEN as a
  two-valued OR of atoms; #1819 makes convergence of several mappings
  normal. The plan's dispatch fold and its collision question now follow
  both.
- #1760 (D-SAG-6) is consumed by crai-rs CR-H0 / CR-H1 (crai-rs #1 and #2,
  private), a sidecar retrofit and not a successor: D-SAG-6 is the motor, Ruff + DT64 the transmission,
  #1814's ON×WHEN×DO the guard ledger. The plan places DT64-6 as that
  transmission's behaviour binding, applies D-SAG-6 F06 / F07 / F11 to it,
  and requires a behaviour registration to reference ordered effects,
  since a DO set or deduplicated SPO bag cannot separate `*x = 1; *x += 1`
  from its reverse.
- odoo-rs #46 (Odoo Codegraph harvest schema, docs) is the Odoo intake:
  its `dt64_candidates` array (source nodes → existing `algebra_ref`,
  no seat mint) feeds the per-cell registry, every
  `emitted_by` MethodId is kept (the single-emitter map lost 26 / 37
  edges), and effective behaviour needs a pinned `registry_snapshot`.
  It narrows open question 2 (inheritance before the cell lookup).
  Searched OGAR and ndarray too: no #1760 / D-SAG-6 successor there.
- `lance-graph-ontology::ttl_parse` leaves `ogit:allowed` unexpanded
  (`ttl_parse.rs` module doc, "What this parser handles today"). The typed verb signature table exists only in OGAR
  `OgitTables::Allowed`, and OGAR depends on lance-graph-ontology.
- The two OGIT readers see disjoint verb sets. Of 523 verb files (OGIT
  `b5731a8`, receipt only), `ttl_parse` `classify` accepts the 12
  `rdfs:Class` verbs (all WorkOrder, all with `ogit:from-to`) and OGAR
  `parse_verb` the 511 `owl:ObjectProperty` verbs. `ogit:allowed` sits in
  960 entity-side files and in no verb file, so the `(src, verb, dst)`
  table comes from entity blocks. 518 / 2,756 is OGAR #363's reader output
  (eaw-5), a reader difference, not OGIT drift. Hence D-DT64-6-1a, and
  D-DT64-6-2 reads OGAR's table instead of adding a second parser.
- Ruff has no Rust frontend (`ruff_*_spo`: C++, C#, MeTTa, Python, Ruby,
  SQLAlchemy); the role names #1814 cites are `PredicateRole` bits in
  `ruff_spo_triplet/src/role.rs`.
- "Direct routing" already names position → executable maps in
  `crates/dir-rmap-probe/tests/classview_dr.rs` (doc on `type Route`).
- `D-DT64-Q0`'s "One-sentence contract" forbids "the persistent Binding Store"; decision 4
  needs an amendment there. Requested in plan §4, not applied.

## Open

Plan §6: the immutable unit; inheritance before cell lookup; collision vs
convergence for coexisting methods; WHEN has no attested source; the Q0
contract amendment; the capstone's "targets" axis vs Identity × Property; 64-wide
blocks vs the #1819 256² plane; EXO-46, EXO-148 and NFR-OP-2.
