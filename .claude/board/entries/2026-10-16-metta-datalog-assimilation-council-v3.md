# 2026-10-16 — metta-datalog-assimilation: /5plus3 council ratifies the W1–W3 spec (D-ASM-1..4)

STATUS: done | OUTCOME: `.claude/plans/metta-datalog-assimilation-v1-council-v3.md` is the executable spec for D-ASM-1..4; the
plan's decisions A–E and G and its §6 stamp OPEN carry ⊘ council notes; `metta-polyfill-v1.md` §5 "never stored as text" amended.
Nothing is authorized to build; D-ASM-1..4 stay Queued. | OPEN: cross-table projections (two-hop, both ends projected) have no
tuple carrier (W4+); `hyperon-probe/golden` is absent and the 26-script baseline must be measured before `LanceSpace` exists.

The premise gate (premise-auditor) returned PREMISE-SPLIT before any savant was cast, and the stamp sub-premise PREMISE-WRONG:
- a conjunctive body has an EXISTENCE half (one table's row set through fk lanes — quack `Filter::Semijoin`/`EqU32Via`, as D-MTP-1
  shipped) and an ANSWER half (binding tuples with multiplicity) and they are different signatures; "join in relalg" was filing the
  second under the first;
- a bag is a KEYED COUNT (`Terminal::GroupReduce{Count}`, `mask-risc/src/ir.rs:420-446`); `CountKeyRunsU32` is `COUNT(DISTINCT)`
  (`:369-389`) and can only be the dedup falsifier;
- `SpaceMut::add` returns `()` (`hyperon-space/src/lib.rs:237`): there is no refusal channel at `add`, so `LanceSpace` ROUTES at
  `add` (non-row atoms to an inner `GroundingSpace`, counted by kind) and REFUSES at `query` (Hyperon `Error` atoms);
- a stamp is motion, not content: it travels BESIDE the atom as an evidence lane at the same coordinate, never inside it;
- a compound term is interned once, frame-side; a per-row 64-byte tenant files a label as a row value (+12.5 % of the stride).

The five savants (prior-art Opus; iron-rule, code-truth, cascade, different-views Sonnet) returned 49 findings: CONFIRMS 17,
PRIOR-ART-AT 7, GAP 11, RISK 12, VIOLATES 3. Load-bearing: `b4_nondeterm` never reaches the fold (every fact is an `=` rule with a
nested head) so it is the ROUTING fixture and the multiplicity fixture is purpose-built; quack `GroupAddr::Pair` already carries
same-table two-variable projections, so only cross-table ones are refused; "the existing label codebook" was prose — a u16
`LabelCodebook` that mints only at `add` is W1 scope; `BindingsSet` construction is population-proportional by Hyperon's own trait
and becomes the one named exit `materialize_bindings` with a bound; `hyperon-probe/golden` does not exist on disk.

The three reviewers (overclaim, dilution/collapse, firewall) on draft v2: 0 BLOCK, 2 FIX-P1 (the reserved evidence lane must be
written `3` = `NO_PROVENANCE`, not 0; fold-bag and inner-bag are two tagged categories concatenated only at the trait boundary),
16 FIX-P2 (every number sourced or labelled POLICY PIN; fork-relative paths; F6 carries its own amendment; `cargo tree` gate for
the library build), all applied in v3.

Not an Epiphany: every finding is a correction of this plan's own text or a measured property of shipped code; the durable
residue is the ratified spec and the ⊘ notes. MIRROR — BIAS CHECK: the v1 spec filed the join in relalg because the plan said so
(anchoring on the plan's own wording); the premise gate, not the savants, caught it.

**Addendum (2026-10-16, later):** fork PR `AdaWorldAPI/hyperon-experimental#1` merged (`e55dc61`), so both trie fixes are on fork `main`. lance-graph's `hyperon-*` pin is still `3f76dc4`; D-MTP-4's remaining step is the pin bump, OPEN, not authorized here.
