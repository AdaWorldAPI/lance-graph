# /coresearch extension 2 — Semantic Grammar Recovery: Kythe × Ruff/Charon × MLIR/DialEgg × XLOG/Soufflé → DT64/F### (2026-11-02)

**Status:** EXPLORATION MAP. Ratifies nothing, implements nothing, adopts nothing on its own authority. /5plus3 stays deferred and unsent.
- **Continues:** `2026-11-02-coresearch-xlog-souffle-crepe-dr3-archaeology.md` (merged via #1930).
- **Scratchpad** (`coresearch-xlog/`): brief, crosswalk and raw notes are in `brief-sgr.md`, `crosswalk-sgr.md` and `raw/sgr-*`, `raw/arch7-9-*`.

**The question** (operator-corrected; the earlier "one field reused by several facades" is CLOSED, owned by ClassView/DT64/F###):
> Welche semantischen Informationen liefern Kythe, Ruff/Charon, MLIR/DialEgg und XLOG/Soufflé bereits, welche verlieren wir beim Transcode, und wie lassen sich diese Informationen als typisierte Grammatikrelationen, Heuristiken zur Ambiguitätsauflösung und kontrollierte opaque Reasoning-Grenzen in unserem bestehenden DT64/F###-Modell nutzbar machen?

**Process:**
- Premise gate PREMISE-SPLIT, adopted. Its four corrections:
  1. A gap class and a legitimacy column (admit / refuse(kind) / abstain), instead of one label.
  2. Two grammars (the NL reading vs the program relation) that may share only an abstention contract.
  3. The admission ladder becomes epistemic grade × admitting authority.
  4. Destination = the existing owner; DT64/F### only for behaviour bindings.
- Six scouts, one per surface (Thyme was withdrawn: the source is Kythe).
- Crosswalk of 48 rows.
- Three co-architects.
- Load-bearing citations were re-read by the orchestrator.

**Operator emphasis, carried as a column:**
- A **Kythe edge** is a code FACT (what the compiler knows about code relations).
- An **egglog equivalence** is a JUSTIFIED TRANSFORMATION (what may replace what).
- The allowed flow is FACT → GUARD → TRANSFORMATION. Never the reverse, and never by adjacency or similarity.

## 1. XLOG / Soufflé-derived rules → DR³
- **Variable roles are computed, then discarded (A1, RECOVER).** The roles are head, join key and filter (`VariableEnv`, xlog-logic lower.rs:2819-2867). The promoted plan carries `slot_vars` (promote.rs:1436). The fork's cube matches `MultiWayJoin { fallback, .. }` (xlog-cube lib.rs:558; zero mentions of `slot_vars`) and rebuilds the classes with its own union-find. XLOG recomputes the classes in five places.
- **Polarity is lost at compile (A2, RECOVER).** The reason a stratum boundary exists (negation or aggregate) dies there: compile.rs:921-933 keeps only `stratum.predicates`.
- **Negation is two RIR shapes (A3).** The seam names the wrong one, `UnsupportedJoin(Semi)`, and must re-read the AST to say `NonPositiveBody`.
- **The epistemic literal is XLOG's own controlled opaque boundary (A7, RECONNECT).** It has typed obligations and four solver statuses that must stay distinct. relcomp folds `know p(X)` and `not p(X)` into one refusal, so the operator is lost.
- **Independent ineligibility reasons exist (A8).** They are hypergraph `Boundary` reasons, under the policy "unknown ≠ unsupported". relcomp returns only the first.
- **Aggregate kinds survive (A4, EXTEND).** The seam admits only Count, although Quack has Min, Max and SumI32. Sum must stay refused unless the type bounds it.
- **Linearity is recomputed three or more times (A5).**
- **Rule identity is lost at plan level (A6).**
- **A zero value cannot abstain (A10).** An unstamped composed edge reads as Direct (epi5_reading.rs:87-94, pinned).
- **The only behaviour bindings here are already wired (A9).** They are the `XLOG_LENS` words; `call/N` and `assert` are refused.

## 2. Ruff/Charon → codegraph (loss #1 at harvest, loss #2 at admission)
- **Effects are computed and then dropped (B1, RECOVER).** The counts of asserts and of writes through a dereferenced argument or local are computed (ruff_charon lib.rs:308-313) and printed only in `calls_text` (:352). `relations()` never reads them, and the assert kind (overflow, bounds, div) is not kept.
- **The dispatch reason survives only as label text (B2).** "via Clause/Dyn" sits inside the object string (lib.rs:544), and `dispatch_target` cuts it off (artifact.rs:611-613).
- **The receiver type and the candidate set are not harvested.** The candidate set is recomputed via the IMPLEMENTS join, and IMPLEMENTS is local-crate only while the CRATE row says COMPLETE.
- **Dynamic calls lose their identity (B3).** The operand is discarded (lib.rs:256), the object is the constant `"dynamic"` (:547), and every site collapses onto `?dynamic`.
- **Residuals are written and never read (B4).** "Body not structured", "callee not named" and the rest are written (store.rs:352-356); codegraph reads them zero times. The "dead function" diagnostic therefore ignores them, along with dynamic callers and the local-only IMPLEMENTS join.
- **FUNDECL and provenance lanes are written and verified, but never pushed (B5).** `artifact_bin.rs` already has `fundecl(row)` and `prov()`; only `read_bin_as` does not use them.
- **Types, borrows, control flow and data flow are not harvested (B6).** Closed search found zero hits.
- **Merge hazard B7 (N2).** An unnamed trait or method becomes `"?"` (lib.rs:536-542), so distinct unresolved dispatch targets merge onto `?::?`. An unnamed *impl* goes to residual `TRAIT_METHOD_NOT_FOUND` instead (:476-479). The typed `TraitDeclId` / `TraitMethodId` exist at harvest.
- **Merge hazard B9 (N2 + N3).** The C++ walker emits an unresolved overload set as a bare name, the same as a resolved free function, and free functions carry no namespace (clang_walker.rs:383-397).
- **Python is narrow (B8).** It has a closed set of ORM mutators and no MAY / UNRESOLVED analogue. It carries graded NARS provenance (ruff_spo_triplet triple.rs:1175).

## 3. Semantic grammar, morphology, ambiguity
- **Role vs edge (C1).** A role is a slot position (S/P/O, TEKAMOLO, Wechsel); an edge is a signed i4 stream offset.
- **What is folded away.** `coca::fsm_pos` irreversibly folds conj, prep and pron into Other (C4). Five German case tables have no consumer.
- **Evidence types that transfer to code (C6)**, only as observed facts:
  - argument position;
  - dependency direction;
  - an identity-type constraint.

  Selectional similarity, binding mode and corpus frequency are REFUSE for code.
- **The premise-gate hypothesis is confirmed (C7).** The two grammars share no role vocabulary (closed search: no import either way). They share a nesting contract: deepnsm certain ⊆ certain∪alternative ≅ codegraph RCONF ⊆ RPOSS.
  - The generators differ: deepnsm enumerates configurations exhaustively (capped at 256, overflow reported), while codegraph's possible set is a declared over-approximation.
  - "deepnsm role ≡ codegraph predicate" is REFUSE.
- **Two RECOVERs (C2, C3).**
  - The alternative readings collapse to `alternative_triples: usize` at the paperless seam (tesseract-paperless consistency.rs:621).
  - An unknown token's index is dropped (deepnsm-v2 fsm.rs:922).
- **The richest abstention vocabulary is in a lab crate, joined to nothing (C8):** dt64-admission-probe `UnknownCause` + `EvidenceClass{ProvenPure, WitnessedEffect, Possible, Unknown}`, lib.rs:113-159.

**Abstention found in six typed forms (wider than the brief's three):**
1. Rewrite illegality (`set_rules::Refusal`).
2. Non-commitment (deepnsm `alternative`).
3. Identity failure (DT64 `Refusal`).
4. Seam admission (`LowerRefusal`, cube `Refusal`).
5. Undeclared grade (`Epi5ReadError`).
6. Typed unknown (`UnknownCause`).

**Kinds of opaque:**
- `Measure::Opaque{pure}`;
- an MLIR opaque op;
- a deepnsm unknown token (identity lost);
- an XLOG epistemic literal.

## 4. MLIR / DialEgg / egglog
- **What MLIR keeps (D1).** MLIR keeps op anatomy, SSA, regions, per-resource effects and Pure/Speculatable. We have no MLIR ingestion (closed search), so this is ADD gated on the DialEgg note's own :113 gate.
- **DialEgg's opaque op = `Value(i64 id, Type)` (D2).**
  - It is never merged, and its type is kept.
  - Operands, attributes, regions and effects are NOT in the term. There is no interface support.
  - Opaque types and attributes are carried as strings.
  - So DialEgg alone does not meet N4, and its string carrier fails N8.
- **The only effect bit is `Measure::Opaque{pure}` (D3).** Production keeps FACT → GUARD → TRANSFORMATION (D4: set_rules `Semantics`; laws.rs `admit`).
- **The reversed direction exists only in labs:**
  - (D5) egglog-r2il encodes the premise `a = c·b` straight as `union`, with no domain check, even under `Checked`.
  - (D6) relcomp `shared_relation` `(rule ((edge a b)) ((union a b)))` turns a directed edge into an equality. The 2026-10-31 board sentence "egglog's equality equals the closure of E ∪ Eᵀ" omits that this comes from the probe's authored rule (correction in §8).
- **Distinctions (D7):**
  - term equality and normal form: WIRED;
  - congruence: egglog, offline only;
  - relational closure: WIRED, and not equality;
  - structural similarity: nothing for code; it licenses nothing;
  - logical entailment: no prover.

## 5. Soufflé / XLOG Magic Sets and meta-awareness
- **The MagicSetReport has 4 fields (E1).** They are status, generated predicates, adorned predicates as strings, and declined reasons as free text. It is dropped at compile.rs:1496/1511 and recomputed only by `xlog explain`.
  - For N5 it lacks: seeds; each rule's origin; the replaced originals; other consumers; the SIPS id.
- **Demand restriction is query-relative (E2).** `reach` goes from 6 rows to 2 on the one fixture. It silently omits other consumers of the target predicate, all-free queries and non-target rules (inferred, not run).
- **SIPS is fixed left-to-right (E3).** A recursive call with no bound argument is a hard error (magic_sets.rs:687), although the doc says `auto` leaves the program unchanged.
- **Knowledge kinds (E4).**
  - eqrel and XLOG closure: relational derivations, with no congruence.
  - egglog: the transformation machinery.
  - The magic relation: a derivation.
  - Demand restriction: a transformation, valid only relative to the query.
  - SIPS order: a heuristic.
- **A model for E1 already exists in a lab (E5).** crepe-plan-probe keeps a typed `Rewrite` log with a set/bag gate.
- **Magic Sets is not meta-awareness (E6).** It answers "what must be computed", never "what is undecided". Three things are needed, all EXTEND over existing partitions, not ADD:
  1. a "partial: restricted to demand D" status;
  2. demand formed as the union over every surviving reading;
  3. an evidence-required marker with abstention.

## 6. Kythe (schema SECTION-READ through a summarising fetcher)
- **Anchors and completeness are wired (F1).** Kythe separates anchors (spans) from semantic nodes, and marks `complete` / `completedby`.
- **The export flattens call strength (F2).** Charon exports CALLS_RESOLVED and MAY_DISPATCH both as `ref/call` (lib.rs:687-694) and points the dispatch at the trait. Kythe already expresses "may call" as `ref/call` to the abstract method plus `overrides`. Shape: target the method, and emit `overrides` from IMPLEMENTS. No new edge kind.
- **Kythe has no unresolved-call edge (F3).** codegraph's `UNRESOLVED_CALL` → RPOSS-only is richer than Kythe. Charon drops UNRESOLVED_CALL from the export.
- **`overrides` ↔ IMPLEMENTS narrowing is wired (F4)** (query.rs:200-217).
- **`typed`, `param.N`, `satisfies` and `extends` are not emitted (F5).** `typed` is the primary premise for N2, and Charon does not harvest types today. `aliases` is a name binding and is REFUSE as a rewrite licence.
- **`ref/writes` and `influences` would be taint only (F6).** A write blocks purity, but an absence proves nothing.
- **Kythe has no visibility fact (F7).**
- **Closed-world reading (F8).** "No call/ref rows ⇒ dead" (diag.rs:151-167) is safe as a diagnostic only if its status degrades on dynamic callers, residuals or the local-only IMPLEMENTS join. It is never safe as a guard.

## 7. Semantic Grammar Recovery Matrix
| bucket | rows |
|---|---|
| **1. Recoverable immediately** (value computed, existing lane/owner holds it) | B5 FUNDECL + provenance (`fundecl()`/`prov()` exist; only `read_bin_as` must push them) |
| **2. Small polyfill connection** (wire an already-computed value through an existing boundary) | A1 `slot_vars`; A2 polarity; A3+A8 the full reason set; A5 linearity flag; A6 rule index; A7 the refusal reason (operator kept); B1 effects as a typed per-function mask lane joined by `(artifact, FunDeclId)`, never parsed from text; B2 dispatch reason as a `u8` lane, never parsed from "via …"; B4 residuals read → `Run::Partial`; B7 key by `(TraitDeclId, TraitMethodId)` / do not intern `"?"`; B9 a typed Resolved / LookupSet / Dynamic tag at the AST site; C2 alternatives kept, not counted; C3 unknown-token index kept; C4 PosCode kept as evidence; E1 return the report instead of dropping it; E3 decline instead of error; F2 method target + `overrides` |
| **3. New guarded rules** (FACT → GUARD → TRANSFORMATION, extending an existing refusal family) | B1 taint → `set_rules::Refusal::Impure` / `Checked` (one-way: a set bit refuses; a clear bit proves nothing unless the residual bit is also clear); A4 Min/Max typed, Sum → `Checked`; D5 limit the premise union to Exact, else `Impure` / `Checked` / `Ieee`; E1-extend / E2 decline when the target has consumers outside the demand (XLOG's own `declined` family); B5 → a target-dependent refusal family that must be named before it is built |
| **4. Not yet safely interpretable** (evidence missing) | B2 receiver type, B3 signature, B6, F5, F6: types and write targets are not harvested. A7 meaning: nothing holds a four-valued result. C5: no agreement features. C8: the grade mapping is untested (B1 taint ↔ WitnessedEffect, RPOSS/MAY ↔ Possible, B4/UNRESOLVED ↔ Unknown; nothing produces ProvenPure). D1/D2: ingestion would be a new parser. D3: no path from a code function to a `Measure` operand. E6: no code signature. A10: zero must become refusable. |

**REFUSE (legitimacy, any bucket):**
- deepnsm role ≡ codegraph predicate;
- selectional similarity, binding mode or frequency as code evidence;
- `aliases` as a rewrite licence;
- a directed edge as equality (D6 as a pattern);
- "bare name = resolved" (B9);
- a closed-world "dead" verdict as a guard (F8);
- a singleton devirtualisation without closed-world evidence (B2);
- Sum / LogSumExp without type bounds.

**Order dependency (firewall):**
1. Read the B4 residuals.
2. Then make F8 degrade to Partial.
3. Only then add the B1 effect lanes.

Effects added first would create new closed-world premises.

**Behaviour bindings (DT64/F###):** only A9 (XLOG_LENS, wired) and the per-key count word missing from F000 (A4). Every other row stays with its owner (codegraph lanes/masks, XLOG plan, deepnsm, contract). No row needs a new authority, engine, IR or parser.

## 8. Negative controls, against current code (firewall + falsifier designer)
| control | state today | falsifier home (pre-registered, not run) | expected |
|---|---|---|---|
| N1 equivalent forms normalize | HOLDS in contract/lab domains; untested across surfaces | contract `set_algebra::normalize` triples + Kleene twin | silent |
| N2 no merge of differing forms | VIOLATED: B7 `?::?` (lib.rs:536-542, joined query.rs:196-212), B9 (clang_walker.rs:383-397), D6 pattern | codegraph inline rows with two `?::?` dispatches; Charon half needs an LLBC fixture (not a probe yet) | fires |
| N3 unresolved never becomes CALL | HOLDS for Rust (graph.rs:365-368); VIOLATED for C++ (B9); the Kythe export merges "may" into `ref/call` (F2) | `area_dyn` fixture (silent); export arm; C++ arm (libclang) | export and C++ fire |
| N4 opaque keeps deps/types/effects or refuses | UNTESTED for MLIR (no ingestion); XLOG: refusal floor holds, operator lost (A7) | relcomp: `not` / `know` / `X>2` → one `NonPositiveBody` (check `know` parses at rev bff6da6) | fires |
| N5 Magic Sets keeps query semantics + derivation info | answers HOLD on one fixture; derivation info VIOLATED (report dropped); other consumers untested | reach + second consumer `far`, on/off, run on xlog-cube | report arm fires |
| N6 resolver can abstain | HOLDS in deepnsm (pinned fsm.rs:1174) and codegraph; VIOLATED: epi5 zero (A10), paperless count (C2), relcomp first-reason-only (A8) | per arm | fires on three |
| N7 before/after witness for a lost property | witnessed: B1 (lib.rs:308-313 → absent from `Relation`, :370-382; `util::double` vs `util::unit`), B4 (store.rs:352-356 → never read), C3 (fsm.rs:915 → :922), A1 (promote.rs:1436 → xlog-cube lib.rs:558), B5 | per witness | all fire |
| N8 no string reconstruction / new hot-path engine | HOLDS on the hot query path (integer lanes); VIOLATED at the harvest/intake seam: `dispatch_target` (artifact.rs:611-613), the export's `split(" via ")` (lib.rs:690); DialEgg opaque types are strings | dispatch-reason arm; the counter half needs a test-only hook | fires |

All expected KILLs confirm gaps already named above; none is a new finding.

**The single most discriminating probe** (recover vs add): `ruff_charon/tests/recover_vs_add.rs`, runnable on the committed LLBC fixtures.
1. Take the function pairs that codegraph admission cannot tell apart.
2. Split off those already separable by fields Charon computed and dropped (asserts, deref writes, callee kind). That is the RECOVER set.
3. Check the remainder against the fixture source for a signature or borrow difference. That names each true ADD with its witness pair.

Can-fire twin: `double` vs `unit` must land in RECOVER.

**Board correction proposed (not applied; the entry is append-only):** `2026-10-31-d-reasoning-codegraph-0-xlog-egglog-shared-relation.md:8` should read "under the probe's authored rule `edge ⇒ union`, egglog's equality equals the closure of E ∪ Eᵀ".

**Not an SGR finding (P0 note).** query.rs:203-212 collects and sorts decoded IMPLEMENTS ids into a `Vec<u32>` instead of folding over a mask. It is small per lookup, but it is the shape the zero-copy rule names.

## 9. Highest-leverage recoveries (bridge, re-ordered by the firewall dependency)
1. B4 residuals read path, then the F8 status fix.
2. B1 effects as a typed taint lane, keeping the assert kind. It is the only code-side producer for `Impure` / `Checked`.
3. B7 and B9, the two wired merge hazards. Both are fixable with values the harvesters already hold.
4. E1/E2: return the magic report, and decline when the target has consumers outside the demand.
5. A3+A8: return all ineligibility reasons, not only the first.

## 10. Decisions only the operator can make
- **S1 — Which recoveries to take forward,** and whether in the firewall's order (B4 → F8 → B1).
- **S2 — Where the two merge hazards are fixed:**
  - in the harvesters (ruff: keyed ids, typed call tag), which is preferred because the information exists there;
  - or refused at codegraph admission.
- **S3 — Whether the lab `EvidenceClass` / `UnknownCause` vocabulary leaves the lab** as the shared grade column, gated on the C8 mapping falsifier.
- **S4 — Whether the six refusal kinds stay separate or fold** (seam admission into rewrite illegality, as scout A asked).
- **S5 — Whether to apply the board-sentence correction** to the 2026-10-31 entry (an append-only edit).
- **S6 — Whether the XLOG plan-level recoveries (A1, A2, A6, E1) land in the AdaWorldAPI xlog fork.** relcomp still pins upstream xlog (base decision D4).

## 11. Not searched
- Kythe overview, verifier and indexing docs, beyond the schema page.
- The DialEgg repository code (paper only).
- MLIR internals beyond LangRef / side-effects rationale.
- xlog-runtime, xlog-cuda; cube.rs and wide.rs internals.
- rust-analyzer (no local checkout).
- Whether pyxlog joins rule provenance to plan positions.

No probe was run.

## Closeout
STATUS: measured (archaeology) | OUTCOME: Semantic Grammar Recovery map. 48 crosswalk rows; a 4-bucket recovery matrix; N1–N8 graded against current code (N2, N5-derivation, N6 partly and N8 at the intake seam violated; N3 violated for C++ and the Kythe export). No row needs a new authority or engine. | OPEN: S1–S6; no probe run; /5plus3 deferred and unsent
