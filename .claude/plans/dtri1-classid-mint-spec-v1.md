# D-TRI-1 classid-half — batched OGAR mint spec — v1

> **Status:** SPEC (ratifiable), 2026-07-17. The value-tenant half of
> D-TRI-1 shipped (#717, autopoiesis triangle). This doc specs the
> **classid half** — the remaining batched mint — to the **handoff
> boundary**: everything is design-complete and byte-precise so the
> doc-W4 council can execute it mechanically once the gate clears. **No
> bytes land with this doc.** Every id below is a PROPOSAL for
> council/operator ratification (RESERVE-DON'T-RECLAIM = permanent once
> minted).
> **Plan:** `triangle-tenants-gestalt-separation-v1.md` §5 (mint discipline).
> **Index:** `.claude/board/INTEGRATION_PLANS.md` (prepended same commit).

---

## §0 The handoff boundary (what this session prepped vs what the council/other session executes)

This session owns the D-TRI triangle arc; the graphrag arc + the **S1
probe** (`P-COMMUNITY-BASIN-AGREE`) are the other session's. Per the
2026-07-17 synergy demarcation (S5 = one mint train):

- **Prepped to the boundary (this doc):** the byte-precise mint spec for
  MY inputs — chess `0x06`, the Cognition task domain, BoardAggregates.
- **Gate (other session, before execution):** **S1** decides whether a
  `community-id` discriminant joins the batch. If S1 measures *identity*
  (`part_of:is_a ≡ Leiden ≡ episodic basin`, the operator's ruling),
  community-id NEVER mints and the batch is exactly the three groups
  below + the graphrag `D-GR-5` doc-seam classid. If S1 measures
  *distinct*, one more concept row folds in.
- **Execution (council, never solo):** OGAR originates → lance-graph
  mirrors with parity tests → read-modes land. One PR pair, audited by
  `baton-handoff-auditor` at the OGAR↔lance-graph seam.

**Two knobs still open for the operator/council** (flagged, not
assumed): (a) the **Cognition domain byte** — `0x03` proposed (next free;
Unassigned spans `0x03`–`0x06`, `0x10`+), the operator named this the one
they may want to set; (b) the **BoardAggregates lane width** (§3).

---

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

## §2 Cognition task domain `0x03` (proposed byte; operator knob)

Operator ruling: *"Tasks like fan-out counterfactual synthesis inference
deduction extrapolation syllogism etc are canonical."* These are cognitive
**task TYPES** — canonical OGAR concepts (domain content like "Queen's
gambit" is NOT). They are the Tasks-SoA task-row classids' concepts.

**`ogar-vocab` `ConceptDomain`:** add `Cognition` variant;
`canonical_concept_domain`: `0x03 => ConceptDomain::Cognition` (**pending
operator confirm of the byte**).

**CODEBOOK block:**

| concept | id | verb |
|---|---|---|
| _(reserved)_ | `0x0300` | **Cognition domain root — `CC==0x00`, NOT a CODEBOOK entry** (`ogar_codebook.rs:4-6` convention: `CC==0x00` is the domain root; `canonical_concept_domain` returns the tag). Every promoted concept starts at `0x??01`. |
| `cognitive_fanout` | `0x0301` | fan-out |
| `cognitive_counterfactual` | `0x0302` | counterfactual |
| `cognitive_synthesis` | `0x0303` | synthesis |
| `cognitive_inference` | `0x0304` | inference |
| `cognitive_deduction` | `0x0305` | deduction |
| `cognitive_extrapolation` | `0x0306` | extrapolation |
| `cognitive_syllogism` | `0x0307` | syllogism |
| `cognitive_task` | `0x0308` | the generic Tasks-SoA task-row concept (an explicit slot, never the reserved root) |

**Codebook-root convention (Codex P2 on #719, applied):** slot `CC==0x00`
is the reserved domain root in every `0xDDCC` block — never a concept. The
chess block (§1) already starts at `0x0601`; this Cognition block now does
too (`0x0301`), with the generic task-row concept at `0x0308`. The council
must NOT emit a `0x??00` CODEBOOK row for any domain.

**Note on the noun/verb axis:** these are verb-shaped, but they enter the
codebook as concept **nouns** (`cognitive_<verb>`) — the OGAR codebook is
a concept registry keyed by `u16`; the verb catalogue (rung-2 144-verb
atoms, `persona-vs-rung-ladder.md`) is a SEPARATE storyline and is NOT
what this mint touches. The task-row classid resolves to the SoA row
(topic/angle/thinking/planner columns), not to a verb atom. Open item: if
the council wants the tasks aligned to the 144-verb rung-2 atoms instead
of standalone concepts, that's a design fork to settle before mint (record
on the board, do not guess).

---

## §3 BoardAggregates value tenant @ row_offset 188 (W2a)

NOT a classid — a value-slab lane (the W2a deliverable, Addendum-12a).
The triangle occupies `[152,188)`; BoardAggregates takes the next
contiguous offset **188** (operator ruling put the triangle first). It is
`ValueTenant` index **13**, added to `ValueSchema::Full` (same additive /
zero-version-bump discipline as the triangle, gated by
`v3-envelope-auditor`).

**Width — the open knob (§0b).** Addendum-12a left it "waits on the mint."
Per-mailbox board aggregates (kanban column counts, WIP, cycle stats) fit
a compact fixed lane; **propose `U8 × 8`** (`[188,196)`) — 8 KanbanColumn
counters as saturating u8, matching the KanbanColumn DAG width. Ends at
196 ≤ 480. Council/W2a-owner confirms the field set before mint. Its
classid + `BUILTIN_READ_MODES` board-row entry ride this batch (fall-
through to `ReadMode::DEFAULT` is FORBIDDEN — Addendum-12a T-gate).

---

## §4 Execution sequence (council, post-S1 — never solo)

1. **OGAR originates** (`ogar-vocab/src/lib.rs`): add the `Chess` +
   `Cognition` `ConceptDomain` variants, the `canonical_concept_domain`
   arms (`0x06`, `0x03`), the CODEBOOK blocks (§1, §2), and `class_ids`
   module constants. One OGAR PR.
2. **lance-graph mirrors** (`lance-graph-contract::ogar_codebook`):
   wire-compatible copy of the same variants + arms + ids, with the
   two-sided **parity tests** (the drift fuse — `canonical_concept_domain`
   agreement, id-uniqueness, `0x06→Chess`/`0x03→Cognition` pins).
3. **Read-modes** (`canonical_node::BUILTIN_READ_MODES`): a
   `ReadMode` entry for the chess-episode classids, the Tasks-SoA task-row
   classid, and the BoardAggregates board classid — each with its
   `{tail_variant, value_schema, edge_codec}`. **No new classid falls
   through to `ReadMode::DEFAULT=Full`** (T3 gate).
4. **BoardAggregates tenant** (`canonical_node`): `ValueTenant::BoardAggregates
   = 13` @ row_offset 188, added to `Full`, field-isolation matrix test —
   the same recipe #717 shipped for the triangle.
5. **Auditors:** `v3-envelope-auditor` (the BoardAggregates layout),
   `baton-handoff-auditor` (the OGAR↔lance-graph mirror seam),
   `firewall-warden` (no PII, no model id).

---

## §5 P4 thinking-row ValueSchema (TD-TRI-1-P4 obligation #2)

The triangle is in `Full` only (#717). P4 wires thinking rows to read the
triangle lanes; those rows need a schema that materialises Frozen/Learned/
Explore. **Proposal:** a NEW `ValueSchema::Thinking` = the `Cognitive` hot
set ∪ the three triangle lanes — so entity classes keep plain `Cognitive`
(no triangle) and thinking/task classids resolve to `Thinking`. Lands
**with** this batch (the Tasks-SoA task classid routes to `Thinking`), not
before — a schema with no classid routing to it would be dead until the
mint. Gated by `v3-envelope-auditor`.

---

## §6 Coordination seams (with the graphrag arc — record, don't duplicate)

- **S5 mint train:** this batch = {chess `0x06`, Cognition `0x03` +
  Tasks-SoA, BoardAggregates@188} + graphrag `D-GR-5` doc-seam classid
  (+ community-id iff S1 says distinct). One OGAR mint. The other session
  owns D-GR-5 + community-id; this doc owns the three groups above.
- **S3 atom-0 / label-0:** #717 locked palette256 atom/index 0 = null. If
  a graphrag community label (dense u32 from 0) is ever persisted into a
  palette256 lane, it must be `label+1` (0 = unassigned). Same family as
  `TD-TRI-1-P4` obligation #1 (codebook idx-0 = null).
- **S1 gate:** community-id is the other session's call; this batch does
  not pre-mint it.

---

## §7 What this spec deliberately does NOT do

No byte minted (proposals only); no S1 run (theirs); no solo mint (the
council convenes the batch); no verb-catalogue touch (rung-2 144 verbs are
a separate storyline); no quarantine lift (chess `0x06` ≠ chess↔thinking
merge — `D-TRI-4` still gates that). The two open knobs (Cognition byte
`0x03`, BoardAggregates width `U8×8`) are flagged for ratification, not
assumed.
