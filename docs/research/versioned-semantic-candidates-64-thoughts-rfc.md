# Versioned semantic candidates over 64 thoughts: admission, resolution, and replay

**Status:** DESIGN / FALSIFICATION RFC, 2026-10-18. No new runtime, instruction set, storage writer, or PR/API implementation claimed.
**Scope:** Semantic *candidate identity and evidence-aware resolution* at the seam of language LEGO, the immutable SoA spine, 64 thought tables, RelAlg/Datalog, R2IL, NAL, and versioned sparse writes.
**Not scope:** Rebuild the thought-git API, change CE64 layout, preselect 64 named personas, or import a second reasoning engine.

## 0. Why a separate PR?

- `#1603` is the **linguistic admission** and grammar-to-query research RFC (COCA, DeReKo, UD, Krill, Haspelmath, semantics, R2IL).
- `#1599` already defines **thought-git-api-v1.md** (branch, read, diff, rebase, squash, land, checkout, blame, merge queue), including read/write masks, DRed on stale derived facts, negative-dependency strata, and slot reuse.
- `#1597` maps the 64-thought epistemic frontier, givens, NodeGuid, CE64 and NAL/ONA. `#1602` addresses Cartesian self-model architecture; **do not duplicate their metacognition proposal**.
- **Missing seam:** what exactly is a *candidate*, how is a proposed identity distinguished from a fact, how are incompatible claims preserved across thoughts/versions, what makes a resolution earned, and can a changed grammatical viewpoint converge without equating distinct propositions?

This RFC specifies that missing semantics. It is **not** a competing lifecycle API.

## 1. Thesis: candidates are canonical claims, not merely uncertain values

One sealed reference frame B[v] has an immutable NodeGuid coordinate system. Each thought t in `0..=63` evaluates a question against B[v], selected value lanes and evidence visible at its pinned version. Thought 0 is the givens; thoughts 1..=63 are isolated programs with sparse writes. One thought can apply NAL-1..9 rules, use language grammar, analyze a philosophical frame, or evaluate a query. These are *program choices*, not fixed meanings of thought indices.

A semantic candidate is an explicitly *proposed statement/relationship binding*, with a stable **statement identity** and optional argument-binding alternatives, not a provisional mutation of an already agreed entity.

For the German `es` in `Das Kind nahm das Buch, weil es lernen wollte`:

- candidate C1: `corefers(mention_es, child_entity)`;
- candidate C2: `corefers(mention_es, book_entity)`.

C1 and C2 are **different proposals**; they must never be conflated because both were offered at the same grammatical position. Their *mention* and their set of possible resolutions are the shared **question context**, not proof that C1 and C2 share one factual identity. Morphological agreement can admit both; semantic plausibility can rank them; independently attested coreference can adjudicate them. Rejection, abstention and later reopening preserve their original witnesses.

The principle: **identity is versioned semantic content; membership is a presence mask; a perspective is a thought; support is stamped evidence; a decision is a replayable computation.** These are five different dimensions.

## 2. Existing code/contracts and the boundaries they enforce

| Existing contract | Where | What this RFC must obey |
| --- | --- | --- |
| NodeGuid = canonical 16-byte identity / HHTL coordinate | `crates/lance-graph-contract/src/canonical_node.rs`, `.claude/knowledge/epistemic-frontier-capstone.md` | name a statement once under a frozen codebook; no new row index or label-based hot key |
| Reference ≠ motion | `.claude/knowledge/reference-frame-vs-motion.md` | corpus counts, lemma/POS, calibration live in versioned immutable bakes, never in attention overlays |
| AlphaOverlay | `crates/lance-graph-contract/src/alpha.rs` | current overlay records attention stamps, NOT a production per-cell evidence/value ledger; D-TGA-1 must supply that before this RFC can use it |
| Thought-git proposal | `.claude/plans/thought-git-api-v1.md` (#1599) | branch slot 1..=63, read/write/diff/status/rebase/land, source bits and squash by value are proposed; do not reimplement |
| Same-address operator | `.claude/plans/d-holo-h14c-thought-first-v1.md` | `Thought(t)[T_a[i] : T_b[i]]`; presence selects what exists, never where to chase it |
| Logic before masked payload | `.claude/plans/d-holo-h14d-logic-mask-read-v1.md` | symbolic normalize → address mask → source-masked read → terminal; H14d reports measured probes, not candidate-resolution benchmarks |
| RelAlg IR | `crates/lance-graph-relalg/src/ir.rs` | current `Rel` Boolean selection over **one row universe**; no production general relational μ/fixpoint or multi-variable query compiler claimed |
| NAL with provenance | `crates/lance-graph-nal/src/lib.rs` | stamps and non-overlapping evidential roots govern revision; NAL F/C is not syntactic/grammar confidence |
| 64 thought epistemic decisions | `.claude/knowledge/epistemic-frontier-capstone.md` | one CE64 register layout, 0 givens + 63 hypotheses, W as thought id is a **2026-10-17 decision with code migration OPEN** |

**Do not use a CE64 field to store a candidate enum.** The CE64 payload and upper bits retain their one globally prescribed interpretation. Candidate labels, source-independent identity, grammatical confidence, score distributions, and resolution state belong to **distinct typed semantic/value lanes** or a derived read result, not new meanings for F/C, orientation, W or epistemic bits. The register is an ISA contract, not the full reasoning history.

## 3. Typed proposal and read-result vocabulary (not new storage structs)

These are **logical** terms to lower to the *existing* SoA and thought-git carrier after its gates close. No new allocator or foreign-reference table is authorized.

| Logical term | Role | Separation / refusal |
| --- | --- | --- |
| `ClaimIdentity` | pinned, globally defined semantic proposition at one NodeGuid; can be reified with arguments, predicate and scope | distinct bound arguments or predicate senses ⇒ different identity |
| `QuestionContext` | query, mention or incomplete grammatical binding whose alternatives are compared | **not** a single resolved claim; grouping is selection, not aliasing |
| `CandidateSet(Q, v)` | mask of admitted ClaimIdentity coordinates for question Q at base version v | derived as a read; candidate membership ≠ truth |
| `EvidenceBase` | independently attested source stamps/roots supporting or contesting a claim | not thought count, translation count or audience count |
| `Perspective` | compiled thought program, front end, rules and interpretive lens | a named philosophical lens is not a historical individual's actual judgment |
| `ReadSet(t, v)` | addresses/fields the thought inspected, plus negative dependencies | must support rebase invalidation when prior reads change |
| `CandidateOutcome` | derived result: Absent / Admissible / Contested / Preferred / Established / Abstained / Superseded / Rejected | a view of versioned evidence, NOT a CE64 bit or write-side fact rewrite |
| `ResolutionReceipt` | canonical claim(s), decision policy version, supporting/contesting roots, alternatives, program hash, base and landing version | required to replay and compare, not a bare latest-wins value |

**Names above are proposed logical vocabulary, not committed Rust types or field IDs.** A receipt's provenance may be represented by source bitsets and sealed-version references when the corresponding existing facilities support it; do not import an arbitrary new witness pointer. If it cannot be represented safely, refuse the write pending a narrow ABI decision.

A changed *calibrated* reference value (COCA/DeReKo frequency, UD gold correction, palette mapping) requires a **new bake/version** with pinned manifest/codebook. A revised **candidate claim** is a sparse thought delta under that frame. Never mutate the frame in place, and never AND masks from independently allocated or differently sealed populations without an explicit verified alignment.

## 4. Candidate lifecycle as read algebra, not write-side status flags

**Admission**:
1. Source-specific adapters produce grammar/lexical **alternative readings** and their independent provenance. No first-row-wins.
2. Canonical binder checks source language, scheme, codebook, semantic role constraints and authorized address minting *outside execution*.
3. Each distinct fully specified proposition obtains its own address/claim identity. Unknown referents can remain **unbound**; do not invent the NodeGuid for a proposition whose arguments are unresolved.
4. A candidate-set mask exposes admissible options for a question; a thought proposes sparse values only at canonical admitted coordinates.

**Evaluation**:
1. A thought selects existing lanes and masks at a pinned frame/commit: `Thought(t)[T_a[i] : T_b[i]]`.
2. Same-coordinate conjunction is a mask AND. Relational constraints that span *different* identities must run a real relation operator or **correct, validated coordinate projection**; no gather, foreign-lane index or fake zero-cost join.
3. Multiple perspectives may address the same claim. Their **presence bits** indicate coverage; value and evidence comparisons decide agreement/contradiction. `popcount(written_by[i])` is NOT a count of independent witnesses.
4. For cyclic grammar, Datalog may propagate **monotone admission constraints** to fixpoint. NAL revision consumes stamps and evidence policy separately; a perspective does not become verified simply by surviving many passes.

**Resolution (derived read)**:
- `Admissible`: coherent interpretation, evidence incomplete.
- `Preferred`: supported more strongly under an explicit, versioned decision policy, with rival alternatives retained.
- `Established`: an appropriate *task-specific* oracle/standard establishes the tested property; UD gold can establish POS/dependency matching, **not** worldly truth or causality.
- `Contested`: admissible rivals or conflicting evidence remain, possibly one preferred.
- `Abstained`: insufficient constraints, scope/identity unknown, or source access denied.
- `Superseded`: new evidence or changed context makes an earlier candidate no longer preferred; history and receipt remain.
- `Rejected` (added 2026-10-19, #1604 review): the candidate is refused under an explicit, versioned decision policy (e.g. a hard constraint or a refuting evidence root fails it). Distinct from `Superseded`, which is only for a previously preferred candidate displaced by better evidence. Rejection keeps its witnesses: the refuting evidence roots, the policy version and the program hash stay in the `ResolutionReceipt`, so a later change to that evidence can reopen the candidate.

A query may return many candidates and not a winner. No fabricated single answer. **Missing in an open-world knowledge graph ≠ false**; negation must state its closed-world/stratification scope.

**Commit / review / rebase / squash / land** are inherited from D-TGA (#1599); no rival implementation. Landing changes the next sealed version after the review gates, not the current givens in place. A deleted/changed premise invalidates conclusions that used it (DRed or independent from-scratch reference); a monotone-only "new AND NOT old" is insufficient after deletion.

## 5. Rotation, inversion and perspective are distinct experiments

The four-sentence ring tests *address perspectives*, not just graph-isomorphism:

```text
That key opens a door.
That door hides a garden.
That garden shelters a fox.
That fox carries a key.
```

### Read-angle invariance, if justified

Rotation changes the entry point and the **shape of the needed query plan** (bound variable, aperture, lane selection, inverse predicate, relation order). It should not alter the final **justified** proposition IDs, but might alter intermediate candidate masks, read footprints and unresolved alternatives.

Important: `a key` in the last sentence does NOT force identity with `that key` in the first. **Without a coreference witness**, the graph is an open candidate ring, not a proven closed cycle. Supply explicit identity in the positive oracle, omit it in the negative.

- Active/passive rewrite preserves the canonical relation only with verified roles and correct inverse handling.
- Reversing a predicate direction without forming the inverse is an incorrect semantic query.
- A changed verb (`carries` → `destroys`) creates a different claim, even if the unlabeled graph topology remains a 4-cycle.
- A grammar-authorized candidate that later turns out to be wrong stays in history as a rejected or superseded hypothesis, not a deleted row.
- Different rotations may compile to **different executable plans** but the same *canonical answer set* when the questions are semantically equivalent. Do not require AST byte-equality for different valid physical plans.

### Perspectives as interpretive programs, not historical authorities

Thought 24 "Aesop", thought 27 "PETA", thought 32 "Jesus", thought 48 "Wittgenstein", thought 62 "Hawking-radiation/ABI compression" are **illustrative perspective/program assignments only**; the actual 64 slots are ephemeral and recyclable. Aesop-fable analogies, welfare criteria, moral teaching lenses, use-in-language analysis and information-loss experiments may select distinct lanes and ask different questions of *the same semantic claim*. They may disagree **without being contradictory factual claims**, because they ask different normative, linguistic or metatheoretical questions. Source quotations, teachings or organizational policies would require appropriate attestation; do not simulate authority as evidence.

A "Hawking radiation" thought is an analogy-driven **compression/replay falsifier**, not a physics-based claim about AGI. Require a concrete retained-versus-dropped-information test: can a source-independent consumer recover the **correct canonical claims, alternatives and provenance** from the ABI + receipts + sealed history? A 64-bit CE register alone is not a lossless historical trace.

## 6. Formal read model and invariants

Let `A_v` be the sealed NodeGuid address allocation at reference version `v`; `i` is its canonical row ordinal; `P[t,l,v]` is the presence mask of thought `t` and typed lane `l` at that version; `Q_v` is a question's candidate/admission mask.

```text
R(Q, v, S, L) = Q_v ∧ Aperture_v ∧ RelevantLaneMasks(S, L, v)

Read(t, l, i, v) = Present(value, source) | Absent
Fold(Q, v, S, L) = semantic operation over Read(t, l, i, v)
                    only where i ∈ R
```

Presence answers **who has written at i**, not **which belief is true**. For candidate hypotheses H1/H2 at *different* NodeGuids, same-address reading does not connect them by magic: their mutual-exclusion/coreference constraint is a **typed relation or authorized candidate-group projection** evaluated against real semantic values. If its representation requires cross-address inspection, charge and test that operation.

Versioned merge must preserve **independence of source stamps**:

```text
one source copied into T1, T2     => ONE independent evidence root
two independent sources in T1     => TWO independent evidence roots
T1 agrees with T2                => perspective agreement, NOT proof
T1 conflicts with T2             => value disagreement + scoped reasons
```

No unqualified `latest wins` for truth or identity. Structural last-write-wins may exist for a designated **mutable, non-epistemic field** under an explicit precedence rule; it must not be used for candidate adjudication.

**Roundtrip:** `Replay(sealed_version, deltas, programs, manifests)` must reconstruct identical resolved claim IDs, candidate masks, evidence classes and refusal reasons, subject to explicitly pinned deterministic sources and iteration policy. Byte-identical *nondeterministic trace order* is not promised; require canonical normalized answer/result equality and separately test deterministic trace serialization.

**Security:** Version, class/graph, row, field and temporal read permissions must be applied *before* any payload load. A mask over unauthorized addresses is not safe simply because it is a bitmap.

## 7. Falsification plan: test one semantic invariant against four failure modes

Machine-readable fixture sketch: [`versioned-semantic-candidates-fixtures.json`](versioned-semantic-candidates-fixtures.json).

| Test | Stimulus | Correct result | Kills |
| --- | --- | --- | --- |
| C-01 | 4 rotations, explicit final key identity | same justified claim set, potentially different entry masks / paths | order-dependent referent minting |
| C-02 | same rotations, `a key` without identity evidence | **no fabricated closure**; unresolved ring | structural isomorphism mistaken for coreference |
| C-03 | active ↔ passive with role normalization | same typed relation under valid inverse | surface-subject equals agent |
| C-04 | `carries` → `destroys` | distinct canonical claim/predicate | topology-only "meaning" |
| C-05 | German `es` → Kind/Buch | distinct admitted candidates, no forced truth, provenance retained | same-mention ≡ same-proposition |
| C-06 | one witness copied to 3 thoughts | evidential root count stays 1 | popcount mistaken for evidence |
| C-07 | two independent witnesses in one thought | evidential root count 2 | thought number mistaken for stamp |
| C-08 | rival normative/linguistic lenses | both valid *as perspectives*, no factual conflict unless claims logically oppose | value judgment merged into proposition truth |
| C-09 | premise deleted after derived candidate landed | dependent conclusion retracted/rederived as reference oracle | monotone-only stale rebase |
| C-10 | reused thought slot after squash | `blame` bound to landing version + roots, not reused slot | perspective history corrupted |
| C-11 | contradictory evidence, unsupported cause | contested; no unearned `Causes` | grammar/plausibility promoted to Pearl |
| C-12 | same masks over different frame versions/allocations | typed refusal pending alignment | plausible but wrong bitmap AND |
| C-13 | negative dependency B on A enqueued before A | A lands before B or queue refuses cycle | FIFO mistaken for stratification |
| C-14 | compression test | equivalent claim/alternative/evidence receipt after replay or explicitly `NotRecoverable` | loss hidden by ABI compactness |
| C-15 | 64 thought capacity | 63 simultaneous hypotheses admitted; next refused; slot 0 never assigned | fencepost overwrites givens |

**Test oracles must be independent:** UD held-out grammar for grammatical distinctions, explicitly authored coreference/semantic labels for competing interpretations, Datalog fixed-point set/bag reference for graph patterns, existing NAL/OpenNARS oracles for truth revision, from-scratch rebuild for rebase/deletion, bitwise reference for mask algebra. Don't train on the final answer fixtures.

Reporting per test: proposed claim IDs (symbolic in fixture until canonical binder ready), source-root hashes, version/codebook seal, input question/scope, admission-mask set, result state + refusal reason, alternative-preservation flag, bytes/reads/cpu for same-coordinate portion and **separately** costs of any cross-coordinate joins.

**Kill thresholds**:
- Any false identity, causal certification or overcounted root: **stop**.
- Any mismatch against known oracle after rebase or replay: **stop**.
- Any unproven candidate silently coerced into a unique fact: **stop**.
- A supposed "free" join requiring foreign lookup or cross-address binding without being counted: **withdraw speed claim**.
- No measurable advantage against a simpler stored alternative: **do not promote to production**.

## 8. Integration without new infrastructure

| Phase | Landing zone | Dependencies / admission |
| --- | --- | --- |
| VSC-0 | this RFC + fixture oracle | independent and docs-only |
| VSC-1 | `deepnsm-v2` / grammar DTO consumer of #1603 | lexical/UD alternatives → typed claim *proposals*, refuse unbound identity; preserve foundry evidence |
| VSC-2 | reuse D-TGA-1/2 thought value+read carriers | candidate/query selection through existing presence masks, slot/value semantics established |
| VSC-3 | reuse D-TGA-3/4 rebase/land and NAL stamp gate | conflict and DRed oracles, no cross-version false merge |
| VSC-4 | RelAlg/Datalog AST → R2IL | cyclic ring with explicitly bound coreference, no foreign gather |
| VSC-5 | independent 64-perspective + compress/replay benchmark | no 64 preassigned personas or unmeasured AGI conclusion |

No new crate or opcode is requested. The first implementation should be **test harness + a typed admission/read adapter**, after checking #1599 gates and #1603's grammar ABI. If the existing `Binder`/`RelAlg` interface is insufficient, first demonstrate its exact insufficiency with a failing oracle. Reuse the existing H14c/H14d coordinate read path instead of a fresh trie, join engine or storage layer.

## 9. Non-goals / unresolved decisions

- Whether incomplete grammar bindings get a dedicated canonical *question* address or remain only a query-time selection is **OPEN**. Do not mint a false proposition just to provide a pointer.
- Physical layout of candidate evidence/receipts is **OPEN** pending D-TGA-1 and its read-provenance decision; this RFC chooses **semantics**, not a new byte field.
- W-slot thought identity and bits 46..52 still have code/documentation migration conflicts recorded in `.claude/knowledge/epistemic-frontier-capstone.md`. **Do not write new CE64 fields for this RFC**.
- Separating source *witness* versus *provenance* versus *epistemic confidence* remains mandatory. Witnessed grammatical attachment is not proof a moral/philosophical position is true.
- A fixed set of 64 thought slots imposes an *active concurrent perspective budget*, not a cap on the number of historical commits or possible hypothesis kinds.
- "AGI" is a research direction, not an acceptance metric. This PR's metrics are accuracy, replayability, semantic identity, provenance, and costs.

**Decision sought after review:** Is `ClaimIdentity + QuestionContext + CandidateSet + ResolutionReceipt` the correct **logical vocabulary** for the existing thought-git API, with no new physical carrier until D-TGA-1? Then run C-01/C-02/C-05/C-06/C-09 as the first falsifiers.
