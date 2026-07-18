# D-TRI-1 classid-half — batched OGAR mint spec — v1

> **Status:** SPEC (ratifiable), 2026-07-17; **corrected 2026-07-18**
> (`E-COGNITIVE-ATOMS-ALREADY-FROZEN`). The value-tenant half of D-TRI-1
> shipped (#717, autopoiesis triangle). This doc specs the **classid
> half** — the remaining batched mint — to the **handoff boundary**.
>
> **CORRECTION (2026-07-18):** the original v1 proposed a **Cognition
> `0x03` ConceptDomain** for cognitive task types (inference / deduction /
> induction / abduction / synthesis / extrapolation / syllogism / fan-out).
> That was a **rediscovery-tax error** — those are **already global frozen
> atoms** in `holograph::dntree` (epistemic verbs 72–95: INFERS=74,
> DEDUCES=82, INDUCES=83, ABDUCES=84, …; frameworks 0x80–0x8F:
> COUNTERFACTUAL=0x84, ABDUCTION=0x85, CAUSALITY=0x83; NARS `syllogize()`
> is a shipped operation). §2 is rewritten to **reference** those atoms,
> never mint them. The **only real concept mint** left in this batch is
> **chess `0x06`**; BoardAggregates is a **value tenant** (§3), not a
> concept domain.
>
> One knob remains open for the operator/council: the **BoardAggregates
> lane width** (§3). Everything else (chess `0x06`, the read-mode
> contract, the execution sequence) is execution-ready. **No bytes land
> with this doc.** Every id below is a PROPOSAL for council/operator
> ratification (RESERVE-DON'T-RECLAIM = permanent once minted).
> **Plan:** `triangle-tenants-gestalt-separation-v1.md` §5 (mint discipline).
> **Index:** `.claude/board/INTEGRATION_PLANS.md` (prepended same commit).

---

## §0 The handoff boundary (what this session prepped vs what the council/other session executes)

This session owns the D-TRI triangle arc; the graphrag arc + the **S1
probe** (`P-COMMUNITY-BASIN-AGREE`) are the other session's. Per the
2026-07-17 synergy demarcation (S5 = one mint train):

- **Prepped to the boundary (this doc):** the byte-precise mint spec for
  MY inputs — **chess `0x06`** (the only concept mint) + **BoardAggregates**
  (a value tenant, §3). Cognitive task types are NOT minted — they are
  existing frozen atoms (§2).
- **Gate (other session, before execution):** **S1** decides whether a
  `community-id` discriminant joins the batch. If S1 measures *identity*
  (`part_of:is_a ≡ Leiden ≡ episodic basin`, the operator's ruling),
  community-id NEVER mints and the batch is exactly chess `0x06` +
  BoardAggregates + the graphrag `D-GR-5` doc-seam classid. If S1 measures
  *distinct*, one more concept row folds in. (S1 was subsequently retracted
  by #722 — structure≠similarity — so community-id does not mint; see §6.)
- **Execution (council, never solo):** OGAR originates → lance-graph
  mirrors with parity tests → read-modes land. One PR pair, audited by
  `baton-handoff-auditor` at the OGAR↔lance-graph seam.

**One knob still open for the operator/council** (flagged, not assumed):
the **BoardAggregates lane width** (§3). (The former Cognition-domain-byte
knob is **removed** — see the correction header; cognitive task types are
frozen atoms, not a domain to mint.)

---

## §0c Concept-id vs classid — the two are NOT the same (CodeRabbit #719)

The `0xDDCC` values in §1–§2 are **u16 CANON concept ids** (the codebook
key). A **classid** is the 32-bit node key = `compose_classid(canon,
appid)` = canon-high `(canon << 16) | appid`. Internal cognitive/chess
concepts carry **no per-app render skin**, so propose **appid `0x0000`**
(canon-only; a specific app — e.g. q2 `0x01` — stamps its appid at its own
render membrane, NEVER here). Read-mode = `{tail_variant, value_schema,
edge_codec}` from `BUILTIN_READ_MODES`. **This table is authoritative — every
parity test + the OGAR and lance-graph impls reference it; no impl guesses a
classid or a read-mode.**

| concept (u16) | classid (`canon<<16 \| 0x0000`) | read-mode `{tail, value_schema, edge_codec}` |
|---|---|---|
| `chess_episode` `0x0601` | `0x0601_0000` | `{V3, Compressed (Fingerprint+residues: FEN print + move edges), CoarseOnly}` — quarantined corpus |
| `chess_candidate`/`iteration`/`event` `0x0602..0x0604` | `0x0602_0000`..`0x0604_0000` | `{V3, Compressed, CoarseOnly}` |

**No `cognitive_*` classid rows** (correction 2026-07-18). Cognitive task
types are frozen atoms (§2), not concepts — a thinking-row does NOT carry a
`cognitive_<verb>` classid. The task-TYPE is an **atom index** stored in a
row lane (`MetaColumn` / triangle lane), and the row's **classid is its
domain** (chess `0x06`, OSINT `0x07`, …). Whether that row materialises the
triangle lanes is a **read-mode property** — a domain class carrying
thinking lanes routes to `ValueSchema::Thinking` (§5), entity classes keep
`Cognitive`. No new domain is needed to express "this row is a thinking
task."

**BoardAggregates is a value tenant, not a classid** (correction
2026-07-18). The lane @ row_offset 188 (§3) lands as an additive
`ValueTenant` — exactly the recipe #717 shipped for the triangle. The
earlier "board-row `cognitive_board 0x0309`" proposal lived in the deleted
Cognition domain and is **de-scoped**: a board-materialising read-mode
attaches to an existing mailbox/kanban class when one is wired, not to a
newly-minted Cognition concept. Only the chess block above is a real
concept mint this batch; `appid 0x0000` (canon-only, no per-app skin) and
the chess `Compressed` schema are the proposals for council ratification.

## §1 Chess domain `0x06` (operator-ruled exact byte)

Operator ruling "Chess0x06". The `DecisionEpisodeV1` corpus (plan §5,
task DECISION-EPISODE-V1) decomposes into four concepts. Chess is a
**quarantined corpus** (plan §5): its rows are chess-domain rows, NEVER
thinking-row lanes, until `D-TRI-4` measures chess↔thinking transfer.
The 0x06 domain byte does NOT lift that quarantine.

**`ogar-vocab` `ConceptDomain`:** add `Chess` variant;
`canonical_concept_domain`: `0x06 => ConceptDomain::Chess`.

**CODEBOOK block** (pattern = the Geo `0x0FXX` block):

| concept | id | role |
|---|---|---|
| `chess_episode` | `0x0601` | a decision episode (one position-to-decide) |
| `chess_candidate` | `0x0602` | a candidate move under evaluation |
| `chess_iteration` | `0x0603` | one search/eval iteration over candidates |
| `chess_event` | `0x0604` | a played move / outcome event |

Feeds **P5** (chess-as-teacher via temporal.rs) once minted; consumers
pull via `contract::ogar_codebook::canonical_concept_id`, never a local copy.

---

## §2 Cognitive task types — ALREADY FROZEN ATOMS (do NOT mint a domain)

> **Correction 2026-07-18 (`E-COGNITIVE-ATOMS-ALREADY-FROZEN`).** The v1
> draft proposed a `Cognition 0x03` `ConceptDomain` with a `cognitive_<verb>`
> CODEBOOK block. **DELETED.** Operator ruling: *"Cognitive atoms and
> induction abduction synthesis inference deduction extrapolation syllogism
> are global part of the frozen. They were part of the previous lance-graph
> PR of the other session if you read them."* They exist today in
> `crates/holograph/src/dntree.rs` as frozen atoms — minting a parallel
> `0x03` concept domain would be the exact 30-turn rediscovery tax the
> workspace rules warn against (LATEST_STATE § "Proposing a type that
> already exists").

**Where each task type already lives** (frozen — reference by index, never
re-mint):

| task type | frozen atom | home in `dntree.rs` |
|---|---|---|
| inference | `DnVerb::INFERS` = **74** | Epistemic verbs (72–95) |
| deduction | `DnVerb::DEDUCES` = **82** | Epistemic verbs |
| induction | `DnVerb::INDUCES` = **83** | Epistemic verbs |
| abduction | `DnVerb::ABDUCES` = **84** (verb) / `Framework::ABDUCTION` = **0x85** | Epistemic verbs / frameworks 0x80–0x8F |
| extrapolation | `DnVerb::PREDICTS` = **89** / `EXPECTS` = **90** | Epistemic verbs |
| hypothesis | `DnVerb::HYPOTHESIZES` = **81** | Epistemic verbs |
| counterfactual | `Framework::COUNTERFACTUAL` = **0x84** (+ `CausalEdge64` counterfactual inference type) | frameworks 0x80–0x8F |
| synthesis | an **orchestration/compose op** (bundle/merge across sources) — not a single atom; the OrchestrationBridge composition path | — |
| syllogism | a **shipped NARS operation** — `syllogize()` (`lance-graph-osint` `p5_syllogize`) | NARS = 0x80 framework |
| fan-out | an **orchestration op** (parallel dispatch), not a concept | — |

**Consequence for the Tasks-SoA row.** A thinking/task row does NOT carry a
`cognitive_<verb>` classid. It carries:
- **classid = its domain** (chess `0x06`, OSINT `0x07`, ProjectMgmt `0x01`,
  …) — the row is *about* something in a domain;
- **task-type = a frozen atom index** stored in a row lane (`MetaColumn` /
  the triangle lanes from #717) — read the atom, not a classid;
- **thinking-lane materialisation = a read-mode property** — a domain class
  that carries thinking lanes routes to `ValueSchema::Thinking` (§5); the
  atom index rides those lanes. No new domain expresses "this is a task."

**No `ConceptDomain::Cognition`, no `canonical_concept_domain` `0x03` arm,
no `cognitive_*` CODEBOOK block.** Byte `0x03` stays **unassigned/reserved**
(RESERVE-DON'T-RECLAIM — do not claim it as a side effect of this
correction). The verb catalogue (rung-2 144-verb atoms,
`persona-vs-rung-ladder.md`) is the storyline these atoms belong to; this
batch references it, never re-mints it.

## §2a Persona modeling — per-consumer opt-in + its own mint (operator ruling 2026-07-18)

Operator ruling: *"persona modeling when framed with chess is probably fine
but should require per consumer opt-in and minting. For chess related play
style is probably nice to have; for smb woa etc persona modeling is not
business logic conform."*

- **Persona / play-style is NOT part of this batch and NOT a global mint.**
  It is a **per-consumer opt-in** capability with its **own** mint, decided
  by that consumer.
- **Chess:** play-style persona is a **nice-to-have** (a chess-consumer may
  opt in and mint its own persona concept for `D-TRI-4/5` play-style
  transfer). Not required for the chess `0x06` corpus mint (§1).
- **Business consumers (smb-office, woa, medcare, …):** persona modeling is
  **NOT business-logic-conform** — do NOT wire persona lanes into their
  thinking rows. Their rows stay strictly domain + task-atom; no persona.
- This keeps persona off the shared spine (per `I-VSA-IDENTITIES` Layer-2:
  persona is a role catalogue a consumer opts into, not a substrate concept)
  and out of the D-TRI-1 batch entirely.

---

## §3 BoardAggregates value tenant @ row_offset 188 (W2a)

NOT a classid — a value-slab lane (the W2a deliverable, Addendum-12a).
The triangle occupies `[152,188)`; BoardAggregates takes the next
contiguous offset **188** (operator ruling put the triangle first). It is
`ValueTenant` index **13**, added to `ValueSchema::Full` (same additive /
zero-version-bump discipline as the triangle, gated by
`v3-envelope-auditor`).

**Width — the open knob (§0).** Addendum-12a left it "waits on the mint."
Per-mailbox board aggregates (kanban column counts, WIP, cycle stats) fit
a compact fixed lane; **propose `U8 × 8`** (`[188,196)`) — 8 KanbanColumn
counters as saturating u8, matching the KanbanColumn DAG width. Ends at
196 ≤ 480. Council/W2a-owner confirms the field set before mint. This lane
lands as a pure **additive `ValueTenant`** (like the triangle in #717) —
**no new board-row classid** rides this batch (the former `cognitive_board
0x0309` proposal lived in the now-deleted Cognition domain and is
de-scoped, §0c). When a board-materialising read-mode is later wired, it
attaches to an existing mailbox/kanban class — and, per Addendum-12a T3,
must NOT fall through to `ReadMode::DEFAULT`.

---

## §4 Execution sequence (council, post-S1 — never solo)

1. **OGAR originates** (`ogar-vocab/src/lib.rs`): add the **`Chess`**
   `ConceptDomain` variant, the `canonical_concept_domain` arm (`0x06`),
   the chess CODEBOOK block (§1), and `class_ids` module constants. One
   OGAR PR. **No `Cognition` variant, no `0x03` arm** (§2 — frozen atoms).
2. **lance-graph mirrors** (`lance-graph-contract::ogar_codebook`):
   wire-compatible copy of the `Chess` variant + arm + ids, with the
   two-sided **parity tests** (the drift fuse — `canonical_concept_domain`
   agreement, id-uniqueness, `0x06→Chess` pin).
3. **Read-modes** (`canonical_node::BUILTIN_READ_MODES`): a
   `ReadMode` entry for the four chess-episode classids
   (`0x0601..0x0604`), each with its `{tail_variant, value_schema,
   edge_codec}`. **No new classid falls through to
   `ReadMode::DEFAULT=Full`** (T3 gate). (No cognitive/board classid — §2,
   §3.)
4. **BoardAggregates tenant** (`canonical_node`): `ValueTenant::BoardAggregates
   = 13` @ row_offset 188, added to `Full`, field-isolation matrix test —
   the same recipe #717 shipped for the triangle. Value-tenant only; no
   classid.
5. **Auditors:** `v3-envelope-auditor` (the BoardAggregates layout),
   `baton-handoff-auditor` (the OGAR↔lance-graph mirror seam),
   `firewall-warden` (no PII, no model id).

---

## §5 P4 thinking-row ValueSchema (TD-TRI-1-P4 obligation #2)

The triangle is in `Full` only (#717). P4 wires thinking rows to read the
triangle lanes; those rows need a schema that materialises Frozen/Learned/
Explore. **Proposal:** a NEW `ValueSchema::Thinking` with the EXACT field
mask = the `Cognitive` hot set **∪** the three triangle lanes — the
`ValueTenant` set:

```
{ Meta, Qualia, Fingerprint, Energy, Plasticity, EntityType, Kanban,
  FrozenStyle, LearnedStyle, ExploreStyle }   // 10 tenants
```

(NOT `MaterializedEdges` / `HelixResidue` / `TurbovecResidue`).
`tenant_bytes()` = Cognitive's 66 + 3×12 = **102 B** (≤ 480, layout-
preserving, zero `ENVELOPE_LAYOUT_VERSION` bump).

**Routing requirement (mandatory):** a domain class that carries thinking
lanes (its rows hold task-atom + triangle lanes) resolves to `Thinking` in
`BUILTIN_READ_MODES`, **NEVER** to `Cognitive`. Entity classes
(OSINT/PROJECT/ERP/Commerce) that do NOT carry thinking lanes keep plain
`Cognitive` (no triangle). This is the concrete form of `TD-TRI-1-P4`
obligation #2. (There is no separate `cognitive_*` classid — the task-TYPE
is a frozen atom index in the row, §2; the routing discriminant is whether
the class carries thinking lanes, resolved per-class in `BUILTIN_READ_MODES`,
not a dedicated cognitive domain.)

**Persist-side test (mandatory):** a round-trip test that populates the
triangle lanes on a `Thinking`-schema row, bakes it through the
`field_mask`-driven persist, and asserts the triangle bytes **survive** (are
not dropped) — the exact failure obligation #2 names (a `Cognitive`-resolving
thinking row would drop them).

Lands when a thinking-lane-carrying class first routes to it (chess `0x06`
thinking rows are the first candidate, once P4 wires the lanes) — a schema
with no class routing to it would be dead, so it is NOT minted ahead of its
first consumer. Gated by `v3-envelope-auditor`.

---

## §6 Coordination seams (with the graphrag arc — record, don't duplicate)

- **S5 mint train:** this batch = {chess `0x06`, BoardAggregates@188
  (value tenant, no classid)} + graphrag `D-GR-5` doc-seam classid.
  Cognition `0x03` is **removed** (frozen atoms, §2); community-id does
  NOT mint (S1 retracted by #722). One OGAR mint. The other session owns
  D-GR-5; this doc owns chess `0x06` + BoardAggregates.
- **S3 atom-0 / label-0:** #717 locked palette256 atom/index 0 = null. If
  a graphrag community label (dense u32 from 0) is ever persisted into a
  palette256 lane, it must be `label+1` (0 = unassigned). Same family as
  `TD-TRI-1-P4` obligation #1 (codebook idx-0 = null).
- **S1 gate:** community-id is the other session's call; this batch does
  not pre-mint it.

---

## §7 What this spec deliberately does NOT do

No byte minted (proposals only); no Cognition domain (frozen atoms, §2);
no persona mint (per-consumer opt-in, §2a); no solo mint (the council
convenes the batch); no verb-catalogue touch (rung-2 144 verbs are a
separate storyline — the atoms §2 references, never re-mints); no
quarantine lift (chess `0x06` ≠ chess↔thinking merge — `D-TRI-4` still
gates that). The one open knob (BoardAggregates width `U8×8`) is flagged
for ratification, not assumed.
