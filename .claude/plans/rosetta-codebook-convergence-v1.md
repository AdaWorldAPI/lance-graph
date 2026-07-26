# rosetta-codebook-convergence v1 — the Bible Rosetta SoA + multi-codebook qualia agreement

> **Status:** PROPOSED (doc-only; D-RCC-1 is the gate, runnable today)
> **Date:** 2026-07-26
> **Author:** main thread, from the operator's convergence arc (this session)
> **Supersedes/absorbs:** task-board W4, W7, W15, W17, W18 (this plan is their
> common substrate); W5-gap-4 and W11 remain in D-SCI-1 but gate on D-RCC-4.
> **Refs:** `E-ROSETTA-IS-A-JOIN-NOT-A-CHOICE-1`,
> `E-HYPERNYM-CLIMB-IS-A-CASCADE-TIER-DELTA-1`,
> `E-CODEBOOK-LICENSE-REGIMES-ONE-ASSET-EACH-1`,
> `E-SCI-1-WITNESS-CONSTRUCTION-LICENSE-1`, `E-WITNESS-SPECIFIC-MEANING`,
> plan `scientific-kg-substrate-v1.md` (D-SCI-1), `I-VSA-IDENTITIES`,
> `I-NOISE-FLOOR-JIRAK`, `I-LEGACY-API-FEATURE-GATED`.

---

## §0 The operator's thesis (canonical, preserved)

1. **The verse address is a frozen external key.** The Bible never changes
   its verse addressing, so the exact sentence in ALL translations lands in
   the SAME SoA row — witnesses become *lanes on one row*, not rows joined by
   an alignment pass. Comparing witnesses is a row-local read.
2. **Language becomes a lane discriminant, not a DTO.** Resolved from the
   classid (the `ClassView::value_schema` door), never carried. The
   per-witness facet already exists: `ClauseSignature { edition, clause_index,
   … }`.
3. **Multiple codebook results are never merged — the qualia of words become
   a VECTOR OF AGREEMENTS.** Components are measured pairwise agreements
   (rank statistics) between codebooks/lanes; contents stay addressable;
   a component with zero variance self-announces a redundant codebook pair.
4. **Sense resolution runs from both ends and meets (the Schnittpunkt):**
   extensional narrowing (cross-language sense-set intersection — polysemy is
   a language-local accident, so 2-3 independently-lexicalized lanes usually
   collapse it) meets intensional refinement (CLAM/HHTL cascade descent —
   senses separate at a measurable tier depth). Their meeting point is
   informative because the two searches have decorrelated failure modes.
5. **Constraint propagation across lanes, never gradient.** Facts resolved
   on one lane (role, reference, aspect) post as constraints to sibling
   lanes at the same `(verse, clause_index)`; form (voice morphology,
   reflexivity) NEVER crosses. Per-feature directed graph: more-marked →
   less-marked. Corpus is closed ⇒ iterate to fixpoint (the Sudoku shape).
   The unresolved residual = the honest FailureTicket tail, and its size is
   the admission criterion for new codebooks.
6. **Missing lanes get filled (e.g. Czech/Kralická) and the whole thing
   bakes into a Bible Rosetta codebook package** — qualia magnitudes are
   aggregated measured agreements ("backprop" = counts propagating into
   codebooks, auditable, never learned weights).

## §1 What exists already (do not reinvent)

| piece | where | state |
|---|---|---|
| Witness facet (`edition`, `clause_index`, voice, relations) | `contract::grammar::witness::ClauseSignature` | SHIPPED (#849/#850 arc) |
| `WitnessDisposition::TextAbsent` (versification/tradition gaps) | same | SHIPPED |
| TEKAMOLO tenant (G4D3), QualiaI4_16D (16×i4) | `contract::canonical_node` / facet | SHIPPED |
| SPO-G named graphs (`Graph::{WordNet, Theographic, Greek, …}`) | `contract` | SHIPPED |
| German codebook generator (frequency+POS+valency+Wechsel) | `planner/examples/data/de/build_de_codebook.py` | SHIPPED |
| WordNet local mirror + hypernym walk | `planner/examples/data/wordnet/` (gitignored) | LOCAL |
| PROIEL Greek treebank + witness probe | `planner/examples/data/proiel/` (gitignored, NC) | LOCAL, oracle-only |
| theographic rails | cloned sibling | LOCAL |
| One-asset-per-regime Release law | `E-CODEBOOK-LICENSE-REGIMES-…` | RULING |

## §2 Deliverables

### D-RCC-1 — lanes-to-singleton probe (CALIBRATOR, run first — not a kill gate)

> **Operator correction (2026-07-26):** originally drafted as a go/no-go on
> the median — WRONG framing. The value of the intersection is per-item, not
> aggregate: every resolved pair (`Schwalbe=swallow`) is a free, permanent,
> deterministic sense anchor, and the benefit of knowing it is OVERWHELMING
> regardless of the distribution's tail. The only true failure mode — a
> false friend / SHARED ambiguity in the same verse — requires both lanes to
> have inherited the same polysemy accident (mostly cognate borrowing),
> which is rare because polysemy accidents are language-local; and it has
> structural escape hatches (add a non-cognate lane, e.g. Czech; fall back
> to the intensional cascade; route to the residual).

> **Second operator correction (same day): the TRUE worst case is not
> "unresolved" — it is a TRANSLATION ERROR producing a confidently wrong
> anchor** (canonical example: `Erbsünde` where the source has `Tod` — a
> doctrinal rendering substituting the interpretive concept for the textual
> one). Naive intersection ingests doctrine as sense evidence; and if the
> substitution travelled the inheritance chain (Vulgate→Luther→KJV), N lanes
> "agree" with ONE upstream cause — correlated error wearing an N-fold
> confirmation costume (the I-NOISE-FLOOR-JIRAK failure, translation form).
> Three defenses, all load-bearing: (1) **source outranks translation** —
> the Greek/Hebrew lane is the text, translation lanes are witnesses ABOUT
> it; an anchor contradicted by the source lane is no anchor, whatever the
> translation head-count (rule-shape of dependency-outranks-case); (2)
> **agreement counts only across inheritance-independent lanes** — the
> witness-independence weight is what distinguishes three witnesses from one
> witness copied thrice; (3) **doctrinal-vocabulary flag, mechanically
> derived from D-RCC-3 itself** — a lemma that never aligns 1:1 to a stable
> source token is translation-layer (interpretive) vocabulary, not
> text-layer rendering; it self-identifies and is excluded from sense
> anchoring (kept as a qualia signal instead: doctrinal load IS
> construction-choice surprisal).

For the vocabulary shared across the lanes on disk (English KJV + German
via UD lexicon + Greek PROIEL as local oracle): per ambiguous English
lemma, lanes-to-singleton distribution + per-item receipts (`swallow`,
`grape` mandatory anchors) + a false-friend/shared-ambiguity census.

What it CALIBRATES (nothing hangs on it): how many lanes are worth carrying
hot; which items route to the intensional end; the size of the residual.
Cost: an example binary over local data; no new deps, no network.

### D-RCC-2 — the Rosetta SoA shape (contract)

Row = versification-normalized verse address (3-byte b/c/v core; scheme map
Masoretic/LXX/Vulgate as a lane property). Lanes = witnesses. Within-row
ordinal = `clause_index` (verses ≠ sentences). Absence = `TextAbsent`, never
zero. Language = lane discriminant resolved from classid — **no LanguageDto**.
~31,102 rows; whole canon in memory.

- V3-conform: content-blind facet; the qualia agreement vector is a READ of
  the existing QualiaI4_16D carving, not a new column.
- Codebook-SET identity is part of the schema resolution and version-gated
  (`I-LEGACY-API-FEATURE-GATED`): change the participating codebooks and
  stored agreement components would silently change basis.

### D-RCC-3 — word alignment derived from the corpus itself

Verse co-presence ≠ token correspondence. Derive the per-pair bilingual
lexicon by deterministic co-occurrence alignment over the ~31k aligned
verses (no external lexicon licence inherited; CILI demoted to cross-check).
Bootstrap order: verse align (free) → word align (derived) → sense
intersection (D-RCC-1 machinery, corpus-wide) → qualia components.
**Status 2026-07-26:** IN FLIGHT — `rosetta/build_alignment.py` (PMI baseline
from D-RCC-1 §C + a stronger association score, en→de and en→el). Its output
also becomes the successor to the failed monolingual closed-class detector
(`E-DISPERSION-CLOSED-CLASS-DETECTION-FAILS-1`): closed-class labels TRANSFER
through alignment from English (which has UD POS + WordNet) to Czech/Greek
(which have neither), instead of being detected per-language.

Side product (load-bearing, see D-RCC-1 second correction): the
**doctrinal-vocabulary flag** — lemmas with no stable 1:1 source-token
alignment are interpretive vocabulary (`Erbsünde` class), excluded from
sense anchoring, retained as doctrinal-load qualia. Anchoring precedence:
source lane > independent translation agreement > inherited agreement.

### D-RCC-4 — qualia-agreement vector (hydration by POS routing)

Components (all deterministic, each with a support mask — absent ≠ zero,
zero-fallback ladder):

1. WordNet depth (tier delta) 2. polysemy count (the `swallow` flag)
3. sibling density 4. subtree size 5. multi-inheritance junction flag
6. WordNet↔CLAM discrepancy (do co-hyponyms cluster in centroid space?)
7. cross-lane translation agreement (from D-RCC-3)
8. …budget: K codebooks → K leave-one-out components, ≤16 at i4.

POS is the ROUTER: open-class → ladder+metric components; closed-class →
construction/position statistics (TEKAMOLO, Satzklammer, case — the German
lane's home turf). i4 width justified: step 0.125 < sampling SE (~0.23 at
n=20) — the sample, not the nibble, is the precision limit. Qualia attach to
`(word, sense/context)`, NEVER the lemma (else the `swallow → consumption`
bug recurs one level up). Support = NARS frequency/confidence, no second
confidence notion.

**Unblocks:** W5-gap-4 (keep-first polysemy) via component 2 + D-RCC-1 sense
index → un-gates W11 (Aesop probe).

> **Third operator correction (same day): "search for translation errors"
> IS CHAODA.** No bespoke error-hunter module — over the aligned-lane SoA,
> a doctrinal substitution (`Erbsünde` vs the θάνατος/mors/death/smrt
> cluster) is a high anomaly score in a sparse manifold region: CHAODA's
> native read of the CLAM tree. Consequences: (1) **outlier ≠ fork by
> cluster shape** — one deviant lane = a far point off a tight cluster; an
> inherited substitution = a BIMODAL split (two internally-tight
> sub-clusters), so single-lane error vs tradition fork is distinguished
> structurally; (2) **the witness-independence weights become MEASURED, not
> asserted** — lanes repeatedly co-clustering against the source lane over
> 31k rows recover the stemma empirically (Lachmannian textual criticism as
> a substrate side effect); documented history (Vulgate→Luther→KJV) demotes
> to a cross-check of the measurement; (3) D-RCC-5 and the error search are
> TWO READS OF ONE TREE — cluster ancestry read paradigmatically = sense
> tiers; read row-locally across lanes = anomaly/fork detection.

> **Refinement (operator): WordNet + qualia magnitude complete the CHAODA
> read.** (a) Synset mapping gives CHAODA a LANGUAGE-NEUTRAL coordinate
> system — Tod/death/mors/smrt co-locate in concept space with no shared
> embedding, so the taxonomic anomaly read runs BEFORE/independent of
> D-RCC-3 alignment; (b) the anomaly MAGNITUDE = the hypernym tier delta
> (sibling synsets = translational freedom/Compatible; LCA-near-root, the
> Erbsünde-vs-Tod case = doctrinal substitution) — deterministic and
> auditable, never a learned weight; (c) polysemy count triages "our
> sense-lookup erred" from "the translator substituted", and the
> doctrinal-vocab flag routes interpretive vocabulary to qualia instead of
> error; (d) metric×taxonomy conjunction is the verdict — both anomalous =
> substitution/error; metric-only = register drift; taxonomy-only =
> idiom/metaphor; the off-diagonals are information. The detector is
> itself an agreement vector (component 6 doing double duty), never a
> merged score.

### D-RCC-5 — CLAM/HHTL ↔ WordNet alignment probe + CHAODA lane-anomaly read

Does common-prefix-length in the (hierarchical-4⁴) centroid address track
WordNet LCA depth? Spearman ρ vs a flat-256 null, and — the sharper form —
does adding the vertical lane SHRINK the D-RCC-6 unresolved residual for
open-class words? Feeds component 6. (This is the previously-specified,
never-run probe; it lives here now.)

### D-RCC-6 — cross-lane constraint propagation to fixpoint

Post role/reference/aspect facts across lanes at `(verse, clause_index)`;
form never crosses (the `VoiceClass` refusal, one level up). Per-feature
directed marked→unmarked graph derived from treebank feature inventories
(case/voice: el→en; clause-bracket: de→el; definiteness: el→la; aspect:
cs→all — the Czech justification). **Provenance-separated:** propagated
values are derived, excluded from agreement scoring (else the qualia vector
self-fulfills toward agreement). Residual after fixpoint = FailureTicket
tail = codebook admission criterion.

### D-RCC-7 — Czech lane (Kralická 1613)

West-Slavic lexicalization (decorrelated polysemy accidents) + obligatory
aspect (the marked source lane for the temporal/reference pointers).
Verify: PD status of the specific digital transcription (1613 text is PD;
a modernized edition may carry editorial claims).

### D-RCC-8 — the Bible Rosetta package (Release)

PD texts are ingredients; NC treebanks are the ORACLE only (validate the
derived layer locally, never enter the artifact — the libtesseract pattern).
Package: verse table + N PD text lanes + derived (alignment, sense index,
lanes-to-singleton, agreement components, support masks) + MANIFEST per
lane per regime (one-asset-per-regime law). Witness-independence weights
recorded (Vulgate→Luther→KJV inheritance discount; el↔de is the strong
pair). Domain-bias limit (biblical senses only) travels WITH the codebook.

## §3 Order & gates

```
D-RCC-1 (calibrator, cheap, local — informs lane count + routing, blocks nothing)
D-RCC-2 (contract shape)
   → D-RCC-3 (alignment) → D-RCC-4 (qualia vector) → un-gate W11
   → D-RCC-5 (CLAM probe, parallel)   → component 6
   → D-RCC-6 (propagation fixpoint)   → residual criterion
   → D-RCC-7 (Czech lane)             → aspect source + false-friend hatch
   → D-RCC-8 (package/Release)        — needs licence re-verify pass
```

## §4 Licence posture + open items

> **Operator ruling (2026-07-26): this is self-funded personal RESEARCH.**
> NC/academic licences restrict redistribution and commercial use, not
> research use — so nothing below blocks the research deliverables
> (D-RCC-1..7). The licence table's job is documentation-in-place: if
> monetization ever happens, the decision is made against the documented
> regimes, not reconstructed. The one-asset-per-regime law gates ONLY the
> public Release path (D-RCC-8). GermaNet (academic licence) joins the
> usable-for-research, documented-for-later set alongside PROIEL.

1. Per-treebank licence re-verification for any lane leaving local disk
   (only the already-probed set is verified; Latin UD family assumed BY-SA,
   NOT yet re-read).
2. Kralická digital-edition provenance (D-RCC-7).
3. Versification scheme map source (Masoretic/LXX/Vulgate offsets) —
   standard tables exist, none vendored yet.
4. ~~Greek PD edition choice~~ — **DISCHARGED 2026-07-26**
   (`E-PD-GREEK-LANE-ACQUIRED-TISCHENDORF-1`). **Tischendorf 8th ed. is stated
   `Public Domain`; Textus Receptus and Westcott-Hort are BOTH
   `CC BY-NC-SA 4.0`** — the assumption that a pre-1929 Greek edition is
   automatically redistributable is FALSE, because the digital transcription
   carries its own claim. Acquired: 27 NT books, 7,895 verses, full row-key
   containment in KJV, 62 KJV verses `TextAbsent` (critical-text omissions,
   not errors). The PROIEL annotation stays oracle-only regardless.
   Fetcher: `rosetta/fetch_greek_lane.py`.
