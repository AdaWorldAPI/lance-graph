# BELIEF-ABI-RESTORATION-1 — Step 2 ADDENDUM: the copula correction

> Status: ADDENDUM to `.claude/plans/belief-abi-step2-ruling-request-v1.md`,
> operator-directed 2026-08-23. **Proposes NO mint.** Supersedes the ruling
> request's Item 1 recommendation, which is retracted below.

## 1. The retraction

The ruling request's Item 1 recommended:

> ~~`Copula` → relation concept → classid reference~~ **RETRACTED.**

The C1–C4 probe result established only:

```
  COPULA ≠ RAIL PLACEMENT
```

It did **not** establish:

```
  COPULA = CLASSID
```

Those are unrelated conclusions, and the leap between them was the drift.
Routing relation content through classid would smuggle instance/content
semantics into the reading selector — the exact move the dock/route
separation exists to prevent. The prior "a relation is a class" ruling
addresses how OGAR *mints concepts*; it is not a licence to make `classid`
a semantic payload field on relation rows.

## 2. The law

```
  CONTENT NEVER TRAVELS IN CLASSID.
  CLASSID SELECTS THE READING.

  classid = HOW these bytes may be read
  HHTL    = WHERE the resident thing lives
  mask    = WHAT part / group / region conducts
  edges   = HOW addressed things relate

  THE POPULATION DOES NOT MOVE. THE VIEW DOES.
  SHARE THE HIERARCHY, NOT NECESSARILY THE PAYLOAD.
  AN INDEX OR MASK MAY ACCELERATE THE ABI.
  IT MUST NEVER BECOME A SECOND ABI.
  MEASURE THE DISTRIBUTION BEFORE BUYING THE REPRESENTATION.
```

Do not mint one classid for Inh, another for Sim, another for Impl. Do not
smuggle predicates, relation identity, group identity, or belief identity
into classid — for copula or for anything else.

## 3. The machinery audit (what already exists for this job)

| machinery | what it is | role in the copula question |
|---|---|---|
| `WideFieldMask` (`class_view.rs`) | up-to-64+ field/group bit selection, `intersect`/`union`/`is_disjoint`, fail-closed `EMPTY` | broad group CLASSIFICATION of terms/rows — cannot carry pairwise topology |
| `RowFocusMask` / `AttentionFocusFacet` | antichain of HHTL regions; `covers`/`common_prefix`/`intersect`/absorbing `union`/conservative `difference` | hierarchical region selection — WHERE a group applies, never WHAT relates to what |
| relation rows (SPO store, `graph/spo/`; `EdgeBlock` per `ClassView::edge_codec_flavor`) | resident many-to-many topology between addressed things | the natural carrier for arbitrary relation topology |
| `spo::truth::TruthValue` | per-edge (frequency, confidence), revision shipped | truth rides the CLAIM row, never the group |
| CE64 / band readings | causal topology + reasoning lens registers | orthogonal planes; not copula carriers |

The rigid distinction, kept: **HHTL/V3 mask = hierarchical region
selection; `WideFieldMask` = broad field/group selection; relation rows =
arbitrary many-to-many topology.** A non-hierarchical relation is never
forced into HHTL ancestry merely because HHTL is available.

## 4. What was measured (`PROBE-COPULA-GROUP-MASK-1`, 9/9)

Corpus: arena-closure output (Inh chain 1→5, 10 rows after closure) plus
hand rows for Sim (cross-subtree), Impl, and two Rel verbs — 14 rows over
8 terms across two HHTL subtrees.

**Distribution (G-DIST):** Inh=10, Sim=1, Impl=1, Rel=2; max fan-out 5,
max fan-in 4; occupancy 14/256 possible cells = **5.5% — sparse**.
*Fixture bias, stated:* closure only derives Inh/Sim, so this corpus is
Inh-dominated. The KJV right-corner corpus (Rel-heavy) and tactics output
(Impl) are the distributions a workload-scale measurement still needs.

**The Active-Directory shape holds** on shipped operators:

- **G-F1** — every copula reconstructs EXACTLY from resident row content
  `(tag, verb)`; the group reading is lossy BY DESIGN (`Rel(7)` and
  `Rel(12)` share one group, stay distinct copulas). Content lives in the
  row; the group is ergonomics.
- **G-F2** — `members(g)` and `memberOf(row)` are inverse VIEWS over one
  relation; the four group views partition all rows; resident bytes are
  untouched by both lookups. No duplicated canonical state.
- **G-F4** — the cross-subtree Sim pair (0x40.\* ↔ 0x50.\*): neither
  address covers the other; the relation exists ONLY as a row. The
  hierarchy homes both ends; it does not pretend to BE the relation.
- **G-F5** — ONE classid across all rows while four copulas differ;
  reconstruction never reads a classid.
- **G-F6** — a 4-group and a 2-group reading coexist over the same bytes;
  insertion leaves prior rows byte-identical. Regrouping is view-only.
- **G-F8** — reclassifying a row's group leaves its truth and stamp
  untouched: truth/provenance are properties of the CLAIM, never the
  classification.
- **G-COMPOSE** — `group ∩ HHTL region ∩ truth-condition` runs as chained
  predicates over borrowed rows in one pass; nothing materialized. The
  "brutal mask" composition works.

**G-F10 — the cost comparison, with its honest surprise.** At THIS
fixture's scale the dense per-group t×t bitmap (32 B) is *cheaper* than
sparse rows (784 B) — because t=8 is tiny. The scaling arithmetic inverts
hard: dense grows as `groups × t²/8` (t=10⁴ ⇒ ~50 MB per group family
regardless of content), sparse rows grow with actual relations. **Which
wins is a property of the measured workload, not of the design** — which
is exactly why the law says measure first. On these fixture numbers the
addendum buys NOTHING.

## 5. Falsifier status (operator's F1–F10)

| falsifier | status |
|---|---|
| F1 exact copula reconstruction | **held** (G-F1) |
| F2 member/memberOf without duplicated truth | **held** (G-F2) |
| F3 mask forcing materialization sparse rows avoid | not triggered at fixture scale; re-test at workload scale |
| F4 HHTL faking a many-to-many | **held** (G-F4 — the row carries it) |
| F5 classid carrying content | **held** (G-F5) |
| F6 group updates repacking the population | **held** (G-F6) |
| F7 sidecar becoming a second object universe | not exercised (no sidecar built); fence stands |
| F8 truth/provenance on the group instead of the claim | **held** (G-F8) |
| F9 exact inverse lookup + provenance under grouping | **held** (G-F1+G-F2+G-F8 jointly) |
| F10 mask denser than sparse rows for the workload | **measured both ways at fixture scale**; workload-scale open |

## 6. Up/down inheritance vs relation topology (deliverable point 8)

What CAN ride HHTL inheritance without confusing hierarchy with relation
topology: **applicability and scope** — where a group's classification
applies, where support generalizes (`common_prefix` up), where a falsifier
propagates (`covers` down). What CANNOT: the pairwise relation itself.
The Sim row is the measured witness: its endpoints share only the class
root, and any attempt to express it as ancestry would misplace it. The
hierarchy is shared; the payload is not necessarily.

## 7. V4 / BPE / OGAR-loco (deliverable point 7)

Left as MEASURED ALTERNATIVES, not adopted: a recurring
`group ∩ region ∩ condition → behaviour` selection that survives
falsification is a candidate for a learned routing particle. Per the
standing law they remain addressed views/operators/sidecars over the same
resident ABI — never another population owner. Nothing here builds one;
recurrence has not been measured.

## 7b. ⊘ MEASURED — the three corrections that close this addendum

`PROBE-COPULA-DISTRIBUTION-1` (5/5) ran the two distributions §4 named as
open. All three findings correct THIS document.

### Correction A — the shipped fold this addendum failed to audit

**`nars::facet_fold` (ENTROPY-MILESTONES M26) already carries the copula,
losslessly, in the resident M20 register — with zero classid involvement.**
§8 below listed "sparse relation rows with copula content resident in the
row" as candidate C, a hypothesis to measure. It is not a hypothesis. A
strictly cheaper form is shipped, tested, and green:

```
  CStmt {s, cop, p}  ⟷  SpoFacet (the 12-byte content-blind register)
    rail 0  subject     s as (lo, hi)
    rail 1  predicate   (copula TAG, Rel lo)   ← the copula lives HERE
    rail 2  object      p as (lo, hi)
    rail 3  ew_subject  (Rel hi, spare)        ← Rel's u16 completes here
```

Re-verified over the measured corpus, not trusted from unit tests: **16/16
statements round-trip byte-exact** (D1), including `Rel(u16)` payloads
spanning rails 1+3. Five copulas on one `(s,p)` yield five DISTINCT
registers (D2) — the discriminating information is resident bytes, and
nothing upstream is consulted.

This is the **"consult before you guess" tax** the repo's own CLAUDE.md
warns about, paid in full: *"Proposing a type that already exists is a
30-turn rediscovery tax — check first."* The probe-local `RelRow` in
§4 was exactly that.

### Correction B — "KJV Rel-heavy" was REFUTED, not merely unmeasured

§4 named the KJV corpus as the **Rel-heavy** regime that would contrast
with the Inh-dominated closure fixture. Measured, through the REAL
`stance::stream` producer on REAL KJV Genesis 2–3:

| | Inh | Rel | Impl | Sim |
|---|---|---|---|---|
| closure fixture (§4) | 10 | 2 | 1 | 1 |
| **real KJV narrative** | **13** | **2** | **1** | **0** |

**Inh 6× Rel.** The prediction is refuted. Both corpora now measured lean
the SAME way, so **a Rel-heavy regime is UNDEMONSTRATED, not merely
unmeasured** — a materially different status, and the reason for naming
predictions in advance.

### Correction C — "tactics Impl" was a phantom

`nars::tactics` emits **only `Inh` and `Sim`** (every `Copula::` site
verified). There is no tactics Impl distribution. The real producers are
`nars::stance` (both `Impl` and `Rel(verb)`) and `reason_whole_book`
(`Rel(pid)`).

### Cost, at the measured shape and extended

| representation | at measured shape | at t=10k |
|---|---|---|
| **`facet_fold`** | **0 extra bytes** (relabels an existing register) | **0** |
| §4's `RelRow` | 896 B (n=16) | grows with relations |
| dense 4-group bitmap | 164 B (t=18) | **50 MB** regardless of content |

§4's fixture-scale surprise (dense beating sparse) **inverts** at real term
counts — and both lose to a fold that allocates nothing.

### Still BLOCKED, and not fabricated

The whole-KJV **scale** measurement cannot run: `data/coca/lexicon.tsv`
(Release `coca-codebook-v2`) and `pg10.txt → kjv_spo.tsv` are both absent
by design. A hand-written corpus would be a fabricated measurement, so none
was produced. **Scale stays open; shape is now measured.**

## 8. What Step 2 now asks about `cop` (replacing Item 1's question)

Not *"where do we encode Copula?"* but:

> **What is the cheapest lawful resident relation + selection geometry
> from which Copula is merely an ergonomic reading?**

**The answer is E, and it is already shipped.** Re-graded after §7b:

- **E. existing-tenant composition — `nars::facet_fold` → `SpoFacet`.**
  **RECOMMENDED.** The copula is a 2-bit tag on rail 1 (plus `Rel`'s u16
  across rails 1+3) of a 12-byte content-blind register the awareness
  plane already holds. Lossless, round-trip-gated, **0 extra bytes**, zero
  classid involvement. Verified on the measured corpus (D1/D2), not merely
  on its own unit tests.
- **C/D. sparse relation rows (± group masks)** — SUPERSEDED by E for the
  copula question. The `PROBE-COPULA-GROUP-MASK-1` results still stand as
  the general **many-to-many topology** finding (G-F1/F2/F4/F5/F6/F8 held),
  and the group-mask ergonomics remain available as a SELECTION layer over
  whatever carries the relation — but the copula itself needs no new row.
- **A/B. masks alone** — REJECTED as sole carriers: classification cannot
  carry pairwise topology; HHTL must not fake many-to-many (G-F4).
- **F. a new tenant** — NOT proposed, and now clearly unnecessary.

**What the ruling can now decide, and what it cannot.** SHAPE is measured
and points at E. SCALE is still open (whole-KJV blocked on uncommitted
Release data). Since E allocates nothing and is already shipped, the scale
question does not gate adopting it — it gates only any FUTURE proposal to
replace it. If the ruling accepts E, `cop` leaves the residue list
entirely: it has a home, and that home costs nothing.
