## 2026-08-23 — E-CONTENT-NEVER-TRAVELS-IN-CLASSID-1 — "not rail-expressible" never meant "therefore classid"; and the copula already had a shipped home

**Status:** ROOT LAW (operator-issued) + FINDING — [MEASURED]
(`PROBE-COPULA-GROUP-MASK-1` 9/9, `PROBE-COPULA-DISTRIBUTION-1` 5/5).
Retracts the Step 2 ruling request's Item 1 recommendation in place.
**Confidence:** High for the law and for every falsifier listed.
Whole-corpus SCALE is explicitly OPEN and reported as blocked.

### The retraction

The Step 2 ruling request recommended `Copula → relation concept →
classid reference`. **Withdrawn.** C1–C4 established only
`COPULA ≠ RAIL PLACEMENT`; that does NOT establish `COPULA = CLASSID`.
Unrelated conclusions — and the leap between them was content drifting
into the reading selector, the exact smuggle the dock/route separation
exists to prevent.

### The law (operator, 2026-08-23)

```
  CONTENT NEVER TRAVELS IN CLASSID.
  CLASSID SELECTS THE READING.

  classid = HOW these bytes may be read
  HHTL    = WHERE the resident thing lives
  mask    = WHAT part / group / region conducts
  edges   = HOW addressed things relate
```

No per-copula classids; no predicates, relation identity, group identity
or belief identity smuggled into classid — for copula or anything else.
Companion laws: SHARE THE HIERARCHY, NOT NECESSARILY THE PAYLOAD; AN INDEX
OR MASK MAY ACCELERATE THE ABI, IT MUST NEVER BECOME A SECOND ABI;
MEASURE THE DISTRIBUTION BEFORE BUYING THE REPRESENTATION.

### ⊘ The answer was already shipped — and this arc failed to check first

**`nars::facet_fold` (ENTROPY-MILESTONES M26) already carries the copula,
losslessly, in the resident M20 register, with ZERO classid involvement:**

```
  CStmt {s, cop, p}  ⟷  SpoFacet (12-byte content-blind register)
    rail 0  subject     s as (lo, hi)
    rail 1  predicate   (copula TAG, Rel lo)   ← the copula lives HERE
    rail 2  object      p as (lo, hi)
    rail 3  ew_subject  (Rel hi, spare)        ← Rel's u16 completes here
```

Re-verified over the measured corpus rather than trusted from its unit
tests: **16/16 statements round-trip byte-exact**, `Rel(u16)` payloads
included; five copulas on one `(s,p)` produce five DISTINCT registers, so
the discriminating information is resident bytes and nothing upstream is
consulted. **0 extra bytes** — it relabels a register the awareness plane
already holds.

The intermediate `RelRow` hypothesis was therefore a proposal for shipped
code — precisely the rediscovery tax `CLAUDE.md` names: *"Proposing a type
that already exists is a 30-turn rediscovery tax — check first."*

### The Active-Directory shape (the general topology finding, which stands)

A DN homes an object; `member`/`memberOf` are NOT ancestry — inverse VIEWS
over ONE many-to-many relation between already-addressed objects. Measured
on shipped operators: copulas reconstruct exactly from resident row
content while the group reading is lossy BY DESIGN (`Rel(7)`/`Rel(12)`
share a group, stay distinct) (G-F1); members/memberOf are inverse views
with no duplicated canonical state (G-F2); a cross-subtree Sim pair is
expressible ONLY as a row — the hierarchy homes both ends, it does not
pretend to BE the relation (G-F4); ONE classid spans four differing
copulas (G-F5); regrouping is view-only (G-F6); truth/provenance ride the
CLAIM, never the classification (G-F8); and `group ∩ HHTL region ∩
truth-condition` composes in one pass over borrowed rows (G-COMPOSE).

**The demarcation this settles:** applicability and scope inherit up/down
HHTL; **the pairwise relation itself never does.** The Sim row is the
standing witness.

### ⊘ A prediction of this arc's own, REFUTED by measuring it

The addendum named the KJV corpus as the **Rel-heavy** regime that would
contrast with the Inh-dominated closure fixture. Measured, through the
REAL `stance::stream` producer on REAL KJV Genesis 2–3:

| | Inh | Rel | Impl | Sim |
|---|---|---|---|---|
| closure fixture | 10 | 2 | 1 | 1 |
| **real KJV narrative** | **13** | **2** | **1** | **0** |

**Inh 6× Rel.** Both corpora now measured lean the SAME way, so a
Rel-heavy regime is **UNDEMONSTRATED, not merely unmeasured** — a
materially different status. Relatedly, *"tactics Impl"* was a phantom:
`tactics` emits only `Inh`/`Sim`; the real `Impl` producer is `stance`.

### Cost, measured and extended

| representation | at measured shape | at t=10k |
|---|---|---|
| **`facet_fold`** | **0 extra bytes** | **0** |
| a `RelRow`-style row | 896 B (n=16) | grows with relations |
| dense 4-group bitmap | 164 B (t=18) | **50 MB** regardless of content |

The fixture-scale surprise (dense beating sparse) **inverts** at real term
counts — and both lose to a fold that allocates nothing.

### Blocked, and not fabricated

The whole-KJV **scale** measurement cannot run: `data/coca/lexicon.tsv`
(Release `coca-codebook-v2`) and `pg10.txt → kjv_spo.tsv` are absent by
design. A hand-written corpus would be a fabricated measurement, so none
was produced. **Scale stays open; shape is measured.** Because the
recommended carrier is already shipped and costs nothing, the open scale
question does not gate adopting it — it gates only any future proposal to
replace it.

