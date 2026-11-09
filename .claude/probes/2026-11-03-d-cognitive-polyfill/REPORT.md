# D-COGNITIVE-POLYFILL P0 — four frontends, one native algebra

**Date:** 2026-11-03. **Status:** LAB PROBE + SOURCE AUDIT. Nothing here changes
a production crate, ships a parser or admits a path. The probe is
`crates/cognitive-polyfill-probe` (own workspace, excluded from the main one).

**Question:** how much of MORK, MeTTa, Narsese and Atomese can lower through the
existing DR / Quack / RelAlg / R2IL / mask-risc / ndarray stack, and where does
each one first need something that stack does not have?

**Short answer:**
- Only the **ground, bounded, conjunctive relational subset** is common to all
  four. It already lowers today:
  - `lance-graph-xlog::support` / `bag_count` answer the chain shape
    `q(Y) :- r(X,Y), s(Y,Z)`;
  - this probe answers the scoped EXISTS/COUNT form in one mask-risc program.
- What each language adds beyond that subset is different semantics, not a
  different engine:
  - MeTTa: binding multisets and evaluation;
  - Narsese: compound terms and truth laws;
  - Atomese: typed nested links and Values;
  - MORK: rewriting into a Set space.
- The first missing piece is the same for MeTTa and MORK: the cold step that
  turns symbols into DR global ids and back. It has no home.

---

## A. Reuse map (verified in source)

| Native capability | Where | Serves |
|---|---|---|
| DR identity: `LocalMap` global↔local, `MapTable` frames, `locate` | `lance-graph-dt64/src/dr.rs:556`, `:805`, `:955` | all four (term ids) |
| Chain support + Bag count over a DR Y map | `lance-graph-xlog/src/lib.rs:255` (`support`), `:280` (`bag_count`) | MeTTa/MORK/Atomese 2-atom chains, XLOG |
| Co-addressed presence AND (leapfrog's first level), bag-count fold | `mork-pathmap-probe/src/join.rs:1-45`, `:512` (`joint`) | MORK-shaped conjunctive bodies (lab, not admitted) |
| Two-hop functional join via Quack `Semijoin` + datafrog oracle | `lance-graph-metta/examples/join_probe.rs` (D-MTP-1) | MeTTa, **functional relations only** |
| α-acyclic vs cyclic body test (GYO) | `lance-graph-relalg/src/join_body.rs:21` | routing any conjunctive body |
| Sealed-order apertures, Question / TruthAtLeast / Choice / Revision as prepared programs | `lance-graph-nal/src/lib.rs`, `src/nal.rs:159-387` | Narsese atomic statements |
| ONNX ALIAS/INLINE over native functions | `lance-graph-onnx/src/lib.rs:1-25` | ONNX numerics only; no symbolic semantics |
| Tract harvest → DR³ + operator registry (lab) | `tract-harvest-probe` (plan status: LAB PROBE IMPLEMENTED) | ONNX graphs; TFLite is PROPOSAL only (`d-tflp-…-capstone-v1/v2`) |

Re-run on this checkout (`origin/main` `ebbdb27`):
- `lance-graph-nal`: 19 passed;
- `lance-graph-metta`: 13 passed;
- `mork-pathmap-probe`: 47 passed (MORK/PathMap as oracles).

## B. Semantic compatibility matrix

Codes: CON = existing and connected · UNW = existing but unwired · LAB = lab only ·
NAT = native composition possible · NEW = a new semantic contract is required ·
OWN = owner decision · UP = upstream only · REF = refuse · NV = not verified.

| Capability | MORK | MeTTa | Narsese | Atomese | Native substrate |
|---|---|---|---|---|---|
| Identity / symbol binding | NEW: symbol → DR id step (`Tag` byte encoding, MORK `expr/src/lib.rs:91`) | NEW: same (D-MTP-2) | LAB: `u16` term addresses in `guid()`, `nal/src/lib.rs:157`; not DR ids | NEW | DR `LocalMap`: CON |
| Atomic ground facts | UP | UP (`SpaceMut::add`, hyperon-space `lib.rs:237`) | LAB (`Beliefs::seal`) | NV (no AtomSpace source in reach) | CON |
| Typed relations | UP | UP (untyped by default) | copula enum only | NEW | ClassView: CON; per-relation class check: LAB (this probe) |
| WHERE / selection | NAT | NAT | LAB (aperture = sealed key prefix) | NAT | Quack `Keep`: CON; cross-frame routing: LAB (this probe, #1943) |
| Shared-variable join (2 atoms) | LAB (D-MPS-8) | LAB (D-MTP-1, functional only) | REF: needs `(*,A,B)` compound | NAT | xlog `support`: CON |
| Repeated-variable equality `R($x,$x)` | UP | UP (`match_atoms`, matcher.rs:1050) | n/a | UP | NAT (diagonal), not built |
| Set semantics | UP (space is `PathMap<()>`, `set_val`, space.rs:1561) | — | — | NV | `Count`/`Any`: CON |
| Bag / multiple bindings | — | UP: `BindingsSet` is a `SmallVec` multiset (matcher.rs:883), GroundingSpace `ALLOW_DUPLICATION`; PeTTa backtracks (bag) | — | NV | xlog `bag_count` + TRACKED `checked_binary_i64` (EXO-108 / MISSING-3) |
| Ordered nested terms | UP | UP | REF in crate (R10) | NEW | REF |
| Hyperedges targeting edges | UP | UP | — | NEW | NEW |
| Structural unification | UP | UP (loops filtered after the match, no occurs check, matcher.rs:1084) | — | UP | REF (P2) |
| Substitution / construction | UP | UP | — | UP | NEW (it creates structure) |
| Nondeterministic evaluation | UP | UP | — | — | REF (no second interpreter) |
| Rewrite rules | UP (`exec`, space.rs:1902) | UP | — | — | PARK |
| Recursive closure | UP (`metta_calculus`) | UP | — | UP | OWN (EXO-73); relalg `μ` not built (`relalg/src/lib.rs:28`) |
| Frequency / confidence | — | — | LAB (`Beliefs` f, c lanes) | NV (TruthValue) | the I32 lanes exist |
| Evidence provenance / revision | — | — | LAB (W3: 4 evidence classes, `Revision`) | — | grouped fold: CON |
| Temporal inference | — | — | NEW | — | NV |
| Type / guard admission | — | UP (type checker) | global, no class check (by design) | NEW | dt64 `Guards`: CON; see W-2 |
| Mutations / frames | UP | UP | `seal` only | UP | `MapTable::advance`: CON |
| Native terminal result | — | — | `Count`, `Max`, revision | — | `Any`/`Count`/`Keep`/`MaskedSum`: CON |

**What is not common** (this keeps the matrix from reading as "all four are
equivalent"):
- **Narsese cannot spell the common test at all.** A binary relation needs
  the product compound `<(*, A, B) --> R>`, which the NAL crate refuses as
  ledger R10 (`nal/src/lib.rs:91`). `<A-->B>, <B-->C> ⊢ <A-->C>` is deduction:
  it carries a truth value, so it is not an existence test.
- **Classic Atomese is not reachable in source here.** No OpenCog AtomSpace
  code exists in the four upstream repos. The only Atomese-shaped text is
  MeTTa-encoded: `hyperon-experimental/python/tests/scripts/a2_opencoggy.metta:8-12`.

## C. P0 falsifier — results

**Fixture:**
- facts `R(A,B) R(B,C) R(C,D) R(A,E)`, with one DR W64 tile `[A,B,C,D,E]`;
- MeTTa `(, (R A $y) (R $y C))`, MORK `(exec P (, (R A $y) (R $y C)) …)` and
  the Atomese projection all hand-lower to one `Pattern`;
- native arm: `Any`/`Count`/`Keep` over `out[a] ∧ inn[c] ∧ W`, one mask-risc
  program over borrowed rows.

**Oracle:** an independent nested loop over the borrowed fact slice.

**Result: 12/12 tests green** (`cargo test --manifest-path
crates/cognitive-polyfill-probe/Cargo.toml`). clippy `-D warnings` and fmt are
clean.

| # | check | result |
|---|---|---|
| 1 | 125 cases (5×5 pairs × 5 WHERE subsets), `Any` and `Count` vs oracle | agree; both answers present |
| 2 | delete `R(B,C)` | A→?→C flips to false |
| 3 | delete `R(C,D)` | A→?→C and A→?→D unchanged; control B→?→D flips |
| 4 | B at local 2 in S1 and local 0 in S2; relation tile has B at 1 | both route to bit 1, answer true; **raw local join gives false** (bit 2 is C there); an unheld id reported as `dropped` |
| 5 | DR round trip `global(local(g)) == g` | holds; an unmapped id is refused |
| 6 | foreign frame on ingest, prepare, route | `Stale` each; control routes |
| 7 | wrong relation; scope via dt64 `Guards` | `WrongRelation`; `ScopeChanged` |
| 8 | `R(A,B)`×2, `R(B,C)`, `R(A,E)`, `R(E,C)` | Set 2, Bag 3: the Set fold cannot answer Bag |
| 9 | NAL beliefs `<A→B> f=1`, `<B→C> f=0 c=0.9` | existence projection says A→?→C true, truth projection false; a confidence-only revision moves no edge |
| 10 | free endpoint, `R($x,$x)`, nested / Narsese product, two relations, one atom | each refused with its own reason |
| 11 | program without `inn[c]` | A→?→D: false becomes true |
| 12 | 1000 hot runs | 0 bytes allocated (per-thread counter) |
| 13 | no copied upstream index | structural: planes are `&relation.out[i..i+1]` / `&relation.inn[..]`; `inn` is a derived index built at ingest (cost noted) |
| 14 | no second evaluator | structural: `execute_into` is the only evaluator called |

**Source disable runs.** Each anchor was asserted present before patching, and
each patch was reverted:

| disable | result |
|---|---|
| join operand | 6 tests red |
| selection routing | c04 red |
| repeated-variable refusal | c10 red |
| Bag multiplicity | c08 red |
| WHERE operand | c01, c04 red |
| frame check in `prepare` | c06 red |

**First failure per language (where the common subset ends):**
- **MeTTa:**
  - D-MTP-1's `Relation2::ingest` refuses this fixture with
    `SecondFact { subject: A }` (`join_probe.rs:61`): the D-MTP-1 path covers
    functional relations only.
  - The general route is `xlog::support`, which takes `&[(u32,u32)]` tuples
    and already-local ids.
  - Missing: the Space adapter that maps `Space::query`'s `BindingsSet`
    multiset onto a Set/Bag choice (D-MTP-2).
- **MORK:** the symbol → DR id step. D-MPS-8's `JoinTable` builds its own
  value-sorted `JoinDomain`; that is a second identity space beside DR, which
  D-RMAP-1 already flags as unadmitted.
- **Narsese:** the product compound term (R10). Atomic inheritance questions
  already run natively.
- **Atomese:** no source to verify against; only a typed-projection proposal
  is possible.

## D. Findings that need an owner

- **W-1 (wire): one term-resolution step for symbol frontends.** MeTTa and MORK
  symbols need a cold, once-per-frame map `symbol → DR global → local`, with
  the inverse for binding emission.
  - Today this is done three different ways: xlog takes pre-localized
    `u32`; D-MPS-8 builds its own `JoinDomain`; NAL uses `u16` terms.
  - Owner: the DR contract (EXO-46) and the GPT-07 proposal (OGAR owns the
    codebook).
- **W-2 (small gap): `Guards::check` needs a `FrameStore`, but `MapTable`'s
  store is private** (`dr.rs:806`). A frame sealed from a `MapTable` therefore
  cannot be guard-checked for Scope against its own store. The probe checks
  scope on a separate store (c07).
- **W-3: routing a selection across frames is per frame, not per query.** This
  is the D-GLOBAL-WHERE finding (#1943); c04 re-confirms it from the
  frontend side.
- **Set vs Bag must be declared per frontend, not inferred:**
  - MORK's space is a Set;
  - Hyperon's default space and PeTTa's `match` are Bags;
  - NAL evidence is neither: it is weighted, with its own duplicate rule.
  - EXO-108 owns the Bag product.
- **NAL → relation projection is an owner decision** (c09). Existence of a
  belief is not an edge; any threshold is policy.

## E. Minimal roadmap (ranked)

1. **WIRE, low risk.** Bind the MeTTa/MORK 2-atom chain to the existing
   `lance_graph_xlog::support` / `bag_count`, after a cold symbol→DR step
   (W-1). Owner: DR + D-MTP-2.
   - Falsifier: c01–c08 rerun against xlog instead of this probe's program,
     plus Hyperon's golden corpus.
2. **PROBE.** The D-MTP-2 `LanceSpace`, read-only and non-recursive: does
   `Space::query` receive whole conjunctions (`complex_query`,
   hyperon-space `lib.rs:340`) or single conjuncts? Report the multiset vs
   Set choice.
3. **PROBE (P2).** Bounded structural matching (`f(X, g(X))`): can a cold
   matcher lower its surviving constraints to DR relations? No native
   unification without an owner decision.
4. **OWNER DECISION.** Narsese compound terms (R10), and the NAL → edge
   projection policy.
5. **PARK.**
   - recursion (EXO-73);
   - rewriting / `exec`;
   - nondeterministic evaluation;
   - Classic AtomSpace;
   - TFLite (PROPOSAL only).

**Verdicts:**
- **REUSE NOW:** DR `LocalMap`/`MapTable`; xlog `support`/`bag_count`; Quack
  `Any`/`Count`/`Keep`; relalg `is_alpha_acyclic`; NAL Question/Choice/Revision;
  ONNX aliases.
- **WIRE:** symbol → DR term resolution (W-1); MeTTa/MORK chain → xlog.
- **PROBE:** `LanceSpace` multiset boundary; bounded structural matching.
- **OWNER DECISION:** EXO-46 selection sharing; EXO-108 Bag product; R10
  compounds; NAL projection; W-2.
- **PARK:** recursion, rewrite, evaluation, AtomSpace, TFLite runtime.
- **SKIP:**
  - a second term store;
  - a copied AtomSpace;
  - a new interpreter;
  - a mask-risc opcode for AND/OR/COUNT, which already exist.
