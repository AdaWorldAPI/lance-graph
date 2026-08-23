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

## 8. What Step 2 now asks about `cop` (replacing Item 1's question)

Not *"where do we encode Copula?"* but:

> **What is the cheapest lawful resident relation + selection geometry
> from which Copula is merely an ergonomic reading?**

Candidate homes, in the order the measurements currently favour:

- **C. sparse many-to-many relation rows** (copula content resident in the
  row) — held every falsifier at fixture scale; the natural topology
  carrier.
- **D. relation rows + group masks** — C plus lossy-by-design selection
  ergonomics; the composition measured green (G-COMPOSE).
- **A/B. WideFieldMask / HHTL-region masks alone** — REJECTED as sole
  carriers: classification cannot carry pairwise topology (G-F10 note),
  and HHTL must not fake many-to-many (G-F4).
- **E. existing-tenant composition** — the SPO store IS candidate C's
  shipped ancestor; whether `Copula` maps onto its edge reading without
  loss is the remaining wiring question.
- **F. a new tenant** — NOT proposed. Nothing measured requires it.

**No mint until the workload-scale distributions (KJV Rel-heavy, tactics
Impl) are measured and composition is ruled out.**
