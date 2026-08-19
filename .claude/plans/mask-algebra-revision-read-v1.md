# mask-algebra-revision-read-v1 — PLAN (DRAFT)

> **Date:** 2026-08-19 · **Status:** DRAFT, awaiting operator ruling on §5
> **Scope:** ZERO code changes in this document. It is the landing plan for the
> keepers extracted from an in-session review of two externally-authored draft
> modules (`counterfactual.rs` + `revision.rs`). **Both drafts are REJECTED as
> written and are NOT vendored.** They were authored outside this workspace with
> no knowledge of existing surfaces; what survives review is three ideas and one
> substrate gap.
> **D-ids:** D-MAR-1 (substrate gap — mask algebra) · D-MAR-2 (classification read).

---

## §0 — FROZEN DECISIONS

**F1. The drafts are not vendored.** No file, type, or test from either draft
enters the tree. This plan re-derives the keepers against verified surfaces.
Their names are re-used only where the repo-wide grep proves the name free (§1.4).

**F2. No `temporal.rs` changes.** Nothing here touches the sorted-stream
substrate, `QueryReference::at`, or the version-range read. Revision
classification is a read over already-shipped belief state; it does not become a
second episodic ledger.

**F3. `lance-graph-contract` stays zero-dep.** Verified: `crates/lance-graph-contract/Cargo.toml`
carries an empty `[dependencies]` with a doc-comment ruling that even an
*optional path dep* is forbidden (a workspace member resolves path deps at
workspace-load time; a missing sibling breaks every cargo invocation — learned
2026-07-07). D-MAR-1 lands inside that constraint trivially (pure `u64`/`Box<[u64]>`
bit ops). D-MAR-2's module home is constrained by it — see §2.2.

**F4. Substrate-first STOP rule (the headline).** *"A consumer or facade that
needs a capability the substrate lacks does NOT hand-roll it one layer up. STOP;
the capability lands as its own substrate-tier change first"*
(`lance-graph-java/CLAUDE.md` § Missing-capability STOP rule). The rejected
`revision.rs` draft hand-rolled an entire `EvidenceMask` trait **because
`FieldMask`/`WideFieldMask` lack `difference` and `is_subset_of`.** That is a
textbook instance. The fix is the two methods, not the trait.

**F5. D-MAR-2 is a READ, never a pipeline.** `RevisionKind` classifies state the
NARS arena already produced. It mints no tenant, no column, no `ENVELOPE_LAYOUT_VERSION`
bump, no new dispatch. House precedent, quoted verbatim from
`crates/deepnsm-v2/src/basin.rs:57-62`: *"MUL self-measurement: competence ∈ [0,1]
= `1 − width/max_width` … **A derived READ over `max_width`, not a new tenant**"*.

**F6. Order is fixed: D-MAR-1 lands first.** D-MAR-2's inputs are masks; landing
the classifier before the algebra it reads would reproduce the draft's own error.

---

## §1 — INPUT INVENTORY (every line read and verified)

### 1.1 The substrate gap — the two missing methods

| Type | Decl | Present | **Absent** |
|---|---|---|---|
| `FieldMask(pub u64)` | `crates/lance-graph-contract/src/class_view.rs:70` | `EMPTY:74` · `MAX_FIELDS:77` · `from_positions:84` · `with:99` · `has:111` · `count:117` · `is_empty:123` · `FULL:130` · `intersect:134` · `union:143` · `is_disjoint:152` · `inherit:166` | **`difference`, `is_subset_of`** |
| `WideFieldMask(WideRepr)` | `class_view.rs:221` (repr `Small(u64)`/`Wide(Box<[u64]>)` at `:224`) | `EMPTY:253` · `with:259` · `from_positions:288` · `has:295` · `count:306` · `is_empty:315` · `max_fields:326` · `full_for:338` · `intersect:363` · `union:373` | **`difference`, `is_subset_of`** |

Repo-wide grep for `fn (difference|is_subset|is_subset_of|symmetric_difference|complement|without|minus)` across
`crates/lance-graph-contract/src` returns **one** hit — `step_mask.rs:98
`pub const fn without(self, n: u8)`` — a single-bit clear on a *different* type
(`StepMask`), not a set difference. Naming precedent noted, not reusable.

**Honest measurement:** no in-tree consumer currently hand-rolls `FieldMask`
difference. A grep for `& !`-shaped hand-rolls hits only unrelated types
(`SubqueryGraph` `dp_enumerator.rs:118`, `Qualia` `qualia.rs:202`, `StepMask`
`step_mask.rs:100`, `EpisodicEdges` `episodic_edges.rs:145`). The hand-roll was
in the **rejected external draft**, not here. So D-MAR-1 is purely additive with
no fold-in — its justification is the STOP rule (F4) plus named candidate
consumers, not an existing mess.

**Named candidate consumers** (candidates, NOT scope): `standing_mask.rs:126`
`widen(subscriber, key, extra)` is a union with no `narrow` sibling — `difference`
is the missing half. `lance-graph-rbac/src/authorize.rs:199` folds role masks with
`union`; `is_subset_of` is the natural "does this projection stay inside the
permitted surface" assertion. 156 `WideFieldMask` call sites across 17 files.

### 1.2 The shipped machinery D-MAR-2 reads over

- `crates/lance-graph-planner/src/nars/belief.rs:31` — `pub struct Stamp(pub u64)`,
  with `source(id):36` (`1u64 << (id % 64)`), `disjoint(other):42`, `union:48`.
  Module doc `:20-22`: *"Disjoint → NARS revision (evidence pooling, synthesis c
  above both, `|f₁−f₂|` kept); overlap → CHOICE, no double count."*
- `crates/lance-graph-planner/src/nars/belief.rs:108` — `pub enum ReviseOutcome`:
  `Admitted { id }` `:110` · `Revised { id, synthesis_c, depth }` `:114` ·
  `Chosen { id, kept_existing }` `:121`. Produced by `revise_at:191`.
- Mirrored in the workspace-**excluded** `crates/deepnsm-v2/src/belief.rs`:
  `Stamp:33`, `ReviseOutcome:111` (same three variants), `revise_at:187`.
  deepnsm-v2's only dependency is `lance-graph-contract` (its `Cargo.toml:24`).
- `Belief.contradiction: f32` (`belief.rs:107`) — *"Preserved dialectic depth: max
  `|f₁−f₂|` across revisions (the contradiction is committed, not erased)"*.
- Re-exported at `crates/lance-graph-planner/src/nars/mod.rs`:
  `pub use belief::{Belief, BeliefArena, CStmt, Copula, ReviseOutcome, Stamp};`

### 1.3 The withdrawal surface — and a correction to the brief

The brief framed `AssumptionExposed`'s input as a *"withdrawal mask"*. **The
contract already ships this, as a receipt ledger, and explicitly rules a mask
insufficient.** `crates/lance-graph-contract/src/causal_audit.rs:306-315`:

> *"Withdraw every receipt from `source`, returning how many were removed. **This
> is why receipts are canonical and a mask is not: withdrawal requires knowing
> *which* evidence came from whom, and a bitmask cannot answer that.**"*

Surfaces: `SupportLedger:272` · `record:288` · `receipts:295` · `withdraw_source:311`
(→ `usize` removed) · `distinct_sources_for:323` · `EvidenceSourceId(pub u64):240`
· `SupportReceipt:250` · `SupportBasis:186`.

**Consequence, binding on D-MAR-2:** `AssumptionExposed` takes its
source-withdrawal evidence from the ledger (`withdraw_source`'s count /
`distinct_sources_for`), **never** from a newly-minted withdrawal bitmask. A
withdrawal mask would be the same hand-roll the STOP rule forbids, one type over.

### 1.4 Name freedom + collisions to avoid

Repo-wide grep across all `.rs`: `RevisionKind` · `EvidentialEffect` ·
`InterpretiveHorizon` · `EncounterEvidence` · `EvidenceMask` → **0 hits**. All
free.

**Collisions that DO exist**, both in `crates/lance-graph-contract/src/counterfactual.rs`:
- `pub enum CounterfactualError:419` — variants `SwarmNotReady:422`, `NotASplit:424`,
  `MajorityHolds:427`.
- `pub enum RevisionOutcome:432` — variants `Revised:435`, `MajorityHolds:437`.

Both differ from the drafts' same-named types. `RevisionKind` must not collide,
must not be confused with `RevisionOutcome` in prose or `use` lists, and must not
extend either enum (both are append-only surfaces under `I-LEGACY-API-FEATURE-GATED`).

### 1.5 The landing zone for POLICY (not built here)

`counterfactual.rs:379` `pub trait AwarenessRevise` with
`fn revise(&mut self, axis_key: u8, new_evidence: i8) -> Result<(), CounterfactualError>:387`.
Its own doc `:364-378` is explicit that it is a placeholder:

> *"Placeholder trait for the `awareness.revise` surface. # BLOCKED — The
> canonical Rust signature for `awareness.revise` is **BLOCKED** — not confirmed
> on the current contract surface. This trait is a scaffold surface only… Replace
> this trait with a concrete type reference once found."*

Any revision **policy** implements that trait. **This plan does not build the
policy** (§6 D1). Consumer `revise_if_minority_wins:347` is a `todo!()`.

### 1.6 Fragmentation context (deferred — see §6 D5)

19 `fn revise*` sites repo-wide. At least three full re-implementations of the
NARS revision formula, verified by reading each:

| Site | Formula as written |
|---|---|
| `lance-graph-planner/src/nars/truth.rs:57` | `f=(f1·w1+f2·w2)/Σw`, `c=Σw/(Σw+1)` over `evidence_weight()` |
| `lance-graph-planner/src/thinking/sigma_chain.rs:148` | `f=(f1·c1+f2·c2)/(c1+c2)`, `c=(c1+c2)/(c1+c2+1)` |
| `lance-graph-planner/src/physical/accumulate.rs:162` (`fn add`) | identical algebra to sigma_chain, inside `TruthPropagating`'s semiring `add` |

Not this plan's scope; recorded so D-MAR-2 is not mistaken for the dedup.

---

## §2 — PROPOSED RESOLUTION

### 2.1 D-MAR-1 — the mask algebra (lands FIRST)

Purely additive to `crates/lance-graph-contract/src/class_view.rs`. **No layout
change, no repr change, no existing signature touched** — the same acceptance
criterion the `WideFieldMask` widening met (`class_view.rs:186-199`: *"nothing
about `FieldMask` changed"*).

```
FieldMask::difference(self, other) -> Self          // const fn, self.0 & !other.0
FieldMask::is_subset_of(self, other) -> bool        // const fn, self.0 & !other.0 == 0
WideFieldMask::difference(&self, other: &Self) -> Self
WideFieldMask::is_subset_of(&self, other: &Self) -> bool
```

**`FieldMask`** mirrors `intersect:134` / `union:143` exactly: `#[inline] pub const fn`,
`self` by value, `Copy` preserved.

**`WideFieldMask`** reuses the existing `zip_fold:388` unchanged —
`zip_fold(other, |a, b| a & !b)` is already correct across tiers: a missing `b`
chunk reads `0` so `a & !0 = a`; a missing `a` chunk reads `0` so `0 & !b = 0`.
`zip_fold`'s normalization (trailing-zero trim + demote-to-`Small`, `:399-405`)
is **load-bearing**: without it, `Wide([x, 0])` and `Small(x)` would compare
unequal under the hand-written `PartialEq:449`. Add the `(Small, Small)` fast
path for symmetry with `intersect:365` / `union:375`.
`is_subset_of` is specified as `self.difference(other).is_empty()` — one code
path, identity-honest. It allocates on the `Wide` arm; the non-allocating
alternative (`chunk_at`/`canonical_len` loop, both private, both in-module) is a
recorded option if a measurement ever justifies it. **Do not ship both.**

### 2.2 D-MAR-2 — `RevisionKind` as a classification READ

A 9-variant taxonomy over state the arena already holds:

`IndependentConfirmation` · `Reinterpretation` · `HorizonExpansion` ·
`HorizonFusion` · `AssumptionExposed` · `ContradictionPreserved` · `Suspended` ·
`Echo` · `ClosedCycle`

Shape (F5): one classifying function, no state, no trait, no dispatch —

```
fn classify(prior: <prior belief state>,
            outcome: ReviseOutcome,
            stamps: <disjoint | overlap, from Stamp::disjoint>,
            withdrawal: <SupportLedger evidence, §1.3>) -> RevisionKind
```

**What is genuinely new, and it is one variant.** NARS revision cannot
distinguish *"learned more"* from *"was wrong, and the source pushed back"*.
`ReviseOutcome::Revised` covers both: disjoint stamps pool evidence and lift
`synthesis_c` either way. `AssumptionExposed` is the case where the evidence
change is a **withdrawal** — a source retracting a claim — read from the
`SupportLedger` (§1.3), which `ReviseOutcome` alone cannot see. Everything else
in the taxonomy names a state the machinery already computes but does not label.

**`Echo` / `ClosedCycle` = zero-evidential-weight recycling.** `Stamp::disjoint:42`
already detects shared *sources* (overlap → `Chosen`, no double count). It does
**not** detect `A→B→A` cycles across *distinct* sources: each hop carries a
genuinely disjoint stamp, so the arena pools it as new evidence. **`closes_cycle`
stays a NAMED GAP — an INPUT to `classify`, never a mechanism inside it**
(mechanism deferred, §6 D3). `classify` receives the flag; it never derives it.

**Module home — both evaluated:**

| Option | For | Against |
|---|---|---|
| **(A) `lance-graph-planner/src/nars/revision_kind.rs`**, beside `belief.rs` | Pattern-matches `ReviseOutcome`/`Stamp` **directly** — they are planner-local. Planner already deps `lance-graph-contract` (`Cargo.toml:40`), so D-MAR-1's masks and `SupportLedger` are in reach. One new `pub mod` + `pub use` in `nars/mod.rs`. | Invisible to `deepnsm-v2`, whose only dep is the contract — it would keep its own unlabelled `ReviseOutcome:111`. |
| **(B) `lance-graph-contract/src/revision_kind.rs`** | Visible to every consumer incl. `deepnsm-v2`; sits beside `counterfactual.rs`'s `AwarenessRevise` landing zone. | Dependency direction is planner→contract (F3), so the contract **cannot** see `ReviseOutcome`. It would need a mirrored input enum — **a third copy** of a shape already duplicated in planner + deepnsm-v2, plus a mapping shim per call site. That is type duplication minted deliberately. |

**RECOMMENDATION: (A), planner, beside `belief.rs`.** The classifier's whole
value is that it reads the real outcome; (B) buys reach by re-declaring the input
it exists to read. If a second consumer later needs the vocabulary, promote the
*enum only* to the contract then — a 9-variant fieldless enum is zero-dep-clean;
it is the *input* that cannot cross. Operator ruling requested (§5 Q1).

### 2.3 Pre-registered NEGATIVE tests (carried from the draft review)

Two defects found in the rejected drafts become standing gates:

**(a) No match-arm guard that the variant's own construction already implies.**
The draft carried `if has_new_root` as a guard on variants whose construction
*requires* `has_new_root` — an unreachable arm. Guards defeat exhaustiveness
analysis, so the compiler cannot flag it. Gate: **a reachability test per
variant** — every `RevisionKind` constructible from some input (can-fire), and a
paired case where it is *not* returned (can-stay-silent), per the falsifiability
rule's both-halves requirement.

**(b) No error variant without a test that constructs it.** Two draft error
variants (`HopOverflow`, `SequenceOverflow`) were structurally unreachable. Gate:
if `classify` (or anything under D-MAR-2) grows a `Result`, every variant needs a
constructing test or the variant is deleted.

---

## §3 — DELIVERABLES + GATES

| D-id | Deliverable | Gate | Disable-run (must go red) |
|---|---|---|---|
| **D-MAR-1** | `FieldMask::{difference, is_subset_of}` | **G1** `a.difference(b).intersect(b).is_empty()` **PLUS G1-anti**: for a chosen `a ⊄ b`, `a.difference(b)` is **non-empty**. G1 alone passes for `difference ≡ EMPTY` — it is vacuous unsupported. | Return `EMPTY` from `difference` → G1-anti red, G1 still green (proves the pair is needed) |
| | | **G2** `a.is_subset_of(a.union(b))` **PLUS G2-anti**: a chosen pair with `!a.is_subset_of(b)`. G2 alone passes for `is_subset_of ≡ true`. | Return `true` always → G2-anti red |
| **D-MAR-1** | `WideFieldMask::{difference, is_subset_of}` | **G3** G1/G2 identities re-run on the wide tier, **including a Small×Wide cross-width pair in both argument orders** (`Small.difference(Wide)` and `Wide.difference(Small)`) | Drop the `unwrap_or(0)` missing-chunk read → cross-width case red |
| | | **G4** normalization: a `Wide` operand whose difference falls entirely below bit 64 **compares equal to and hashes identically with** the `Small` equivalent (the `PartialEq:449` / V-L P0 contract) | Bypass `zip_fold`'s trim/demote (`:399-405`) → G4 red |
| **D-MAR-2** | `RevisionKind` (9 variants) + `classify()` | **G5** reachability matrix: 9 can-fire + 9 can-stay-silent, non-trivial inputs (an empty-input silence case proves nothing) | Collapse any two variants to one arm → that variant's can-fire red |
| | | **G6** `AssumptionExposed` fires **only** when the ledger reports withdrawal — a `Revised` outcome with no withdrawal must classify otherwise | Ignore the withdrawal input → G6 red (this is the one genuinely new variant; if it cannot discriminate, D-MAR-2 carries nothing new) |
| | | **G7** no guarded arm implied by its own construction (§2.3a); **G8** no error variant without a constructing test (§2.3b) | n/a — review gates, checked by reading the arms |

Gates are run centrally by the orchestrator in the one shared `target/`
(`cargo test -p lance-graph-contract` for D-MAR-1; `-p lance-graph-planner` for
D-MAR-2), plus `cargo clippy -- -D warnings` and `cargo fmt`.

---

## §4 — NON-GOALS

- **No `GadamerRevision` (or any other) revision POLICY.** D-MAR-2 labels; it
  never decides. The policy implements `AwarenessRevise` (§1.5) and is §6 D1.
- **No counterfactual timeline / replay buffer.** See §6 D2 — recorded as a
  REJECTION, not an omission.
- **No revision-formula dedup.** The three re-implementations (§1.6) are
  untouched by this plan.
- **No `closes_cycle` detection mechanism.** Input only (§2.2, §6 D3).
- **No new tenant, column, layout version, or dispatch path** (F5).
- **No `MetaWord` bit allocation.** See §6 D6.

---

## §5 — OPEN OPERATOR DECISIONS

**Q1 — Module home for `RevisionKind`.** §2.2 recommends (A) planner, beside
`belief.rs`. (B) contract is reachable but costs a third copy of the
revise-outcome shape. Ruling requested before D-MAR-2 starts.

**Q2 — Does `AssumptionExposed` feed MUL / Dunning-Kruger as a competence
signal?** The seam exists: `contract/src/mul.rs:50 MulAssessment`, `:100 DkPosition`,
and the house precedent is `deepnsm-v2/src/basin.rs:62 competence(&self, max_width)`
— *"the signal `lance-graph-planner/mul` (Dunning-Kruger / compass) consumes. A
derived READ over `max_width`, not a new tenant"*. A source withdrawing a claim
the system had already pooled is exactly a competence-overestimate signal. **But
wiring it makes `RevisionKind` load-bearing rather than descriptive**, which is a
different risk class and would need its own inertness gate (raising the threshold
must silence something). Recommend: **land D-MAR-2 descriptive-only; decide Q2
after G5/G6 report real firing rates** — a classifier that fires on everything, or
never, carries no information either way and must not be wired before that is
measured.

---

## §6 — DEFERRED — MISSING INTEGRATION

**D1 — The full hermeneutic revision policy.** Implements `AwarenessRevise`
(`counterfactual.rs:379`). Blocked on the same thing that trait's doc is blocked
on: *"the canonical Rust signature for `awareness.revise` is BLOCKED — not
confirmed on the current contract surface"* (`:366-373`). Landing a policy against
a placeholder trait would bake the placeholder in.

**D2 — Bounded counterfactual timeline — REJECTED, recorded so it is not
re-proposed.** The shipped D-ATOM-4 design already prices the road-not-taken at
**4 bits**: `deposit_counterfactual` (`counterfactual.rs:140`) writes
`InferenceType::Counterfactual.to_mantissa() = −6` into the `CausalEdge64`
inference nibble, and `:107-114` rules that the deposit *"must **never** be
written as observed SPO truth… lives in the episodic / ghost tier only"*. A
replay buffer re-materializes as data what the substrate deliberately compressed
to a nibble in a separate lane. Not deferred — **declined**.

**D3 — `closes_cycle` detection mechanism.** A bounded provenance projection that
sees cross-source `A→B→A`, which `Stamp::disjoint:42` structurally cannot (each
hop's stamp is genuinely disjoint). Until it exists, `Echo`/`ClosedCycle` are
classified from a caller-supplied flag (§2.2).

**D4 — `HypothesisReport` / `GroundingRequest` surfaces.** Named in the drafts,
absent here, and out of scope: both are *output* surfaces for a policy that does
not exist (D1).

**D5 — Revision-formula dedup across the 19 sites.** §1.6. Its own plan; needs a
parity oracle across the three algebras before any of them is deleted (the
`truth.rs:57` weight form and the `sigma_chain.rs:148` / `accumulate.rs:162`
confidence form are **not** obviously the same function).

**D6 — `EvidentialEffect` as a `MetaWord` bit read.** `MetaWord(pub u32)` at
`contract/src/cognitive_shader.rs:44`. A bit allocation is a layout decision on a
shipped type and needs its own field-isolation matrix under
`I-LEGACY-API-FEATURE-GATED`; it must not ride in on a classification read.

---

## §7 — SEQUENCING + BOARD HYGIENE

1. Operator rules Q1 (§5).
2. **D-MAR-1** lands alone: `class_view.rs` + G1–G4 with their disable-runs.
   Board: `LATEST_STATE.md` § Contract Inventory (two methods × two types),
   `STATUS_BOARD.md` rows.
3. **D-MAR-2** lands after D-MAR-1 is green, in the home Q1 selected, with G5–G8.
   Board: `STATUS_BOARD.md`, and `EPIPHANIES.md` for the §1.3 correction (a
   withdrawal *mask* is the wrong shape; the ledger already rules why).
4. This plan is indexed in `.claude/board/INTEGRATION_PLANS.md` (PREPEND) in the
   same commit as its first deliverable, per the board-hygiene rule.

**Not written by this plan:** any `.claude/board/*` file. Board updates ride with
the deliverable commits, by the orchestrator, per the one-writer rule.
