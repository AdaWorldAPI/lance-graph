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

- The merged DT64 plans use none of the ON / WHEN / DO terms (the three
  plans read in full). The ON×WHEN×DO registry exists only as the odoo-rs lab design
  `docs/DT64-ON-WHEN-DO-REGISTRY.md` (#44 / #45 merged, test-only) and in the
  docs #1814 and #1819 (now merged). #1814 rules out folding WHEN as a
  two-valued OR of atoms; #1819 makes convergence of several mappings
  normal. The plan's dispatch fold and its collision question now follow
  both.
- `lance-graph-ontology::ttl_parse` leaves `ogit:allowed` unexpanded
  (`ttl_parse.rs` module doc, "What this parser handles today"). The typed verb signature table exists only in OGAR
  `OgitTables::Allowed`, and OGAR depends on lance-graph-ontology.
- "Direct routing" already names position → executable maps in
  `crates/dir-rmap-probe/tests/classview_dr.rs` (doc on `type Route`).
- `D-DT64-Q0` line 9 forbids "the persistent Binding Store"; decision 4
  needs an amendment there. Requested in plan §4, not applied.

## Open

Plan §6: the immutable unit; inheritance before cell lookup; collision vs
convergence for coexisting methods; WHEN has no attested source; the Q0
amendment; the capstone's "targets" axis vs Identity × Property; 64-wide
blocks vs the #1819 256² plane; EXO-46, EXO-148 and NFR-OP-2.
