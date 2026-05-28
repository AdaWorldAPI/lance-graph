# 2026-05-28-1200 — PR #418/#419 review + surreal/mailbox/Baton/SoA-as-BindSpace-surrogate plan map

> **What this is.** A read-only synthesis surfacing (a) the substantive content of
> open PRs #418 and #419 against `adaworldapi/lance-graph`, (b) the in-session
> plan corpus around the owned little-endian Baton contract / mailbox-as-owner /
> SoA-as-BindSpace-surrogate / SurrealDB-as-view, and (c) one navigability
> meta-finding (the older `.claude/surreal/` POC docs are not annotated as
> superseded by `E-RUBICON-RACTOR`). **No code or behavioral change in this
> handover; no board file edited (only appended).**
>
> **Session:** 2026-05-28 review pass on branch
> `claude/lance-graph-ontology-review-Pyry3`. Provenance for the underlying
> rulings is in the PR bodies + the cited epiphanies; this doc is the index.

---

## 1. PR #418 — `docs: BindSpace-singleton → mailbox-owned SoA migration spec (one LE contract end-to-end)`

**State.** Open, mergeable-clean, base `main`, head
`claude/splat3d-cpu-simd-renderer-MAOO0` (the D-PERSONA-1 splat-renderer branch
— the spec rides on it; minor branch/content mismatch, not a blocker). +478
across 6 files. Single code touch: a `//!` module doc-note on
`crates/cognitive-shader-driver/src/mailbox_soa.rs`. Everything else is plan +
board.

**Verdict: sound, merge-ready as a spec.** Exemplary CCA2A hygiene — one PR
adds the plan (`.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`), the
`INTEGRATION_PLANS` entry, the `STATUS_BOARD` rows (D-MBX-1..6 + a TD row), two
`EPIPHANIES` entries (append, prior content intact), and the `TECH_DEBT` entry
— exactly as the CLAUDE.md Board-Hygiene Rule requires. All D-MBX-1..6 are
**Queued**; nothing in this PR executes.

**What it rules (substance, in three lines):**

1. **SoA-as-BindSpace-surrogate.** `MailboxSoA<N>` *becomes* the BindSpace — it
   is **not** a per-mailbox copy of a singleton (that's still a singleton = the
   `E-CE64-MB-4` aliasing problem). The shared `Arc<BindSpace>` (driver.rs:56,
   `BindSpace::zeros(4096)` in serve.rs:29) **dissolves**. Each mailbox owns its
   rows cradle-to-grave.
2. **The owned little-endian contract = the Baton.** One SoA layout runs the
   whole vertical with **no boundary re-encode** — shader-driver `MailboxSoA`
   → `lance-graph-contract` LE types
   (`CausalEdge64`/`QualiaI4_16D`/`MetaWord`/`entity_type`) → Lance storage.
   `ShaderCrystal.persisted_row` is a *pointer to the same row*, not a
   serialized copy. The only cross-boundary state is the owned baton
   `(u16, CausalEdge64)` (`CollapseGateEmission`; wire cost
   `13 + 10·baton_count`).
3. **Column map.** Drop the 64 KB `Vsa16kF32` `cycle` plane; own
   `edges`/`qualia`/`meta`/`entity_type`; *reference* (not own) content via
   CAM-PQ; ontology stays a shared `Arc`, out of the SoA.

### Substantive review notes (no blockers, but worth flagging for readers)

1. **Two footprint figures must not be conflated** — and the plan now catches
   this itself: bare migrated columns ≈ **24–50 B/row**, but the *full hot
   thought* ≈ **6 KB** because the content/topic/angle Hamming planes stay hot
   (that's what sets the **64k–256k thought ≈ 300–600 MB ceiling**). The win is
   dropping the 64 KB f32 plane (71.6 KB → ~6 KB), not shrinking to 30 B.
   `E-MAILBOX-IS-BINDSPACE` quotes the "24–30 B" figure and the plan §2.7
   quotes "6 KB" — both correct, different scopes; OQ-1 is marked resolved by
   the capacity math. A careless reader can still conflate them — preserve the
   "bare columns vs full hot thought" distinction when citing.
2. **`E-RUBICON-RACTOR` is honest CONJECTURE.** Its own provenance note says
   *"Libet and Heckhausen appear nowhere in the board/code/transcripts."* It's
   a post-hoc psychological framing over the **already-shipped** Σ10
   (`SigmaTierRouter`, D-CSV-10, #388). Nothing to *implement* from
   `E-RUBICON-RACTOR` itself — it's a mapping/naming:
   - ractor START = crossing the Σ10 Rubicon (commit);
   - Libet "free won't" = `CollapseGate` pre-commit veto;
   - ractor STOP = post-actional eval → tombstone-witness.
3. **The doctrinal contradiction is correctly gated, not silently resolved.**
   Migration step S5 (delete the `cycle` plane) is blocked on **OQ-4**:
   `CLAUDE.md` "The Click" is still written on `Vsa16kF32`. The 2026-05-26
   Baton-scoping note already scopes `Vsa16kF32` out *as a carrier* (keeping
   the bundle math) — half-laid. The full doctrinal rewrite must land before
   S5.

---

## 2. The SurrealDB role correction — the crux

The single most important shift this session is a **correction to SurrealDB's
role**. Older docs still carry the superseded framing; the supersedure is
recorded only in `E-RUBICON-RACTOR` and is not annotated at the POC docs
themselves (see §5 below for the navigability finding).

| Framing | Where it appears | Status |
|---|---|---|
| "SurrealDB-on-Lance persistence = Zone 2 cold store" | `.claude/surreal/RECONCILIATION_with_canonical_plan.md:20`; `causaledge64-mailbox-rename-soa-v1` Zone 2; surreal POC docs (`01_deps_substrate.md`…`12_clean_writer_invariants.md`, `cognitive-substrate.md`) | **SUPERSEDED** |
| **"LanceDB is the *leading* store / source of truth (append-only, versioned). SurrealDB is one *VIEW* over it (the Rubicon kanban), never a store."** | `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md` §2.7 + `EPIPHANIES.md` `E-RUBICON-RACTOR` (PR #418) | **CURRENT RULING** |

**Consequence.** `surreal_container` (SurrealDB-on-`kv-lance`) is **optional**:

- D-MBX-6's hot/cold transparent-view wiring uses the **LanceDB cold tier
  directly** as the source of truth + the GoBD-by-construction audit trail
  (`E-LADDER-SERVES-MAILBOX` §6 tombstone-witness).
- The SurrealDB kanban is a *second* view over the same LanceDB rows — the
  Rubicon action-phase board (deliberation | crossed/intention | actional |
  evaluated). Moving a thought across columns = `ractor` start/stop transitions.
- This matches `RECONCILIATION_with_canonical_plan.md` §"Course-correction"
  already saying *"fold the `surreal_container` scaffold into the canonical
  crates rather than ship separately."* The blockers on `surreal_container`
  (fork dep + Lance 6 pin, marked BLOCKED A/B/C/D) therefore do not block
  D-MBX-6.

---

## 3. The plan corpus map (the answer to "what additional plans this session")

PR #418 sits at the apex of a stack of plans on the mailbox-owned-SoA / Baton
/ hot-cold theme. The list, with role in the arc and current status:

| Plan / surface | Role in the arc | Status |
|---|---|---|
| **`.claude/plans/bindspace-singleton-to-mailbox-soa-v1`** (PR #418) | The column-level migration map: dissolve `Arc<BindSpace>` → `MailboxSoA<N>` ephemeral thoughtspace; D-MBX-1..6 | Proposal (#418, open) |
| **`.claude/plans/causaledge64-mailbox-rename-soa-v1`** | The **driver/parent** plan #418 subsumes under (§0 zones, §1 ownership-typed compartments, §5 `MailboxSoA<N>`, §6 5-crate inventory, §7 7-PR sequence) | Canonical |
| **`.claude/plans/cognitive-substrate-convergence-v1/v2/v3`** | D-CSV series. **D-CSV-7 SHIPPED** the `MailboxSoA` accumulator. **D-CSV-10 SHIPPED** the Σ10 Rubicon-resonance `SigmaTierRouter` (#388, sprint-12 Wave F). v3 is the latest version | v3 active; parts shipped |
| **`.claude/plans/atom-mailbox-substrate-v1`** | atoms (bipolar I4-32D) → styles → personas, quorum, signed-mantissa counterfactual, AriGraph hot/cold/tombstone. The locked 33-TSV scaffold (recent commits: `19a9a3a`, `cb7882d`, `f9c3f1d`, `0a2b4f5`, `00a4cc3` — D-ATOM-1..5 `///`-surface) | Active scaffold |
| **`.claude/plans/rung-persona-orchestration-v1`** + **`rung-ladder-grounding-v1`** + **`rung-mul-grounding-v1`** | The escalation→epiphany→ghost **ladder reframed as mailbox machinery** (`E-LADDER-SERVES-MAILBOX`): *persona = dispatch policy, not container.* ractor outer-swarm (D-PERSONA-5); ghost-preempt = the Libet veto; death → tombstone-witness | Grounding |
| **`.claude/surreal/`** (15 docs) | The SurrealDB-container-on-`kv-lance` POC (SoA container type, codec, write/read, read-time fold, catalog KV, moka L2, epoch-tick harvest, **lockfree handoff ring**, log compaction, clean-writer invariants, PHASE2 adjacent crates) — a *narrower parallel re-derivation* of the canonical plan, to be folded in | POC; **fold-in pending**; **framing partially superseded** (see §2) |
| **`.claude/plans/3DGS-Cesium-BindSpace4-headstone-exploration`** | The tombstone/"headstone" witness visualization exploration | Exploration |

### Anchoring epiphanies

| Tag | Status | What it ratifies |
|---|---|---|
| **`E-BATON-1`** | FINDING (`dec049b`) | "Baton" = the workspace's native term for the LE contract. Ratifies deprecating the singleton BindSpace + `Vsa16kF32`-as-carrier. Sea-star topology. Inter-mailbox state = owned `(u16, CausalEdge64)` handoffs (`CollapseGateEmission`). `wire_cost_bytes() = 13 + 10·baton_count`, **not** `16384·4`. |
| **`E-CE64-MB-4`** | FINDING | Ownership-typed compartments make UB a compile error. Move/ownership semantics prove no-alias / no-race / no-UAF at compile time. |
| **`E-MAILBOX-IS-BINDSPACE`** | CONJECTURE (NEW in #418) | `MailboxSoA<N>` *becomes* the BindSpace surrogate (not copied per mailbox). The singleton `Arc<BindSpace>` dissolves; drop the 64 KB `Vsa16kF32` cycle plane. |
| **`E-RUBICON-RACTOR`** | CONJECTURE (NEW in #418) | Σ10 Rubicon = Heckhausen action-phase crossing. ractor start/stop = crossing/closing the Rubicon. Libet "free won't" = `CollapseGate` pre-commit veto. Kanban = a SurrealDB **view** over leading **LanceDB**. |
| **`E-LADDER-SERVES-MAILBOX`** | CONJECTURE | The ladder is mailbox machinery (persona is a *policy* over the substrate, not a container). ractor outer-swarm sync/async (§1); ghost preempt = veto (§5); tombstone-witness (§6). |
| **`I-VSA-IDENTITIES`** | FINDING (iron rule) | Bundle *identities*, not content. The dense planes do not get copied per mailbox (that would be N× the carrier being deleted). |
| **`I-LEGACY-API-FEATURE-GATED`** | FINDING (iron rule) | Feature-gate v1 `BindSpace` accessors during the S1–S4 staged migration; never silently diverge semantics. |
| **`E-CONTRACT-NO-SERIALIZE`** | FINDING | Contracts compile types; membranes serialize. Same byte layout disk-to-RAM ⇒ no membrane re-encode for the thoughtspace. (The *ontology* path legitimately re-encodes `MappingRow ↔ RecordBatch` and is the documented carve-out — see plan §4.) |

### Open gates / dependency chain (nothing executes before these)

```
PR-NDARRAY-MIRI-COMPLETE      ── close U16x32/U32x16/U64x8 + i-word simd_nightly method gaps;
   │                              route simd::* through simd_nightly under cfg(miri); delete
   │                              simd_nightly/_original_draft.rs
   ▼
D-CE64-MB-1-impl              ── par-tile Mailbox<T> apex
   │
   ▼
D-MBX-1   add migrated columns to MailboxSoA<N> behind `mailbox-thoughtspace` (gate alongside)
D-MBX-2   move engine_bridge per-row surface onto mailbox rows; cycle plane → transient local
D-MBX-3   ShaderDriver holds sea-star of mailboxes; kill BindSpace::zeros(4096) in serve.rs:29
D-MBX-4   death → SPO-G + Lance tombstone-witness
D-MBX-5   delete BindSpace + cycle plane; remove gate     ── BLOCKS ON OQ-4 (CLAUDE.md
                                                                  "The Click" doctrinal rewrite
                                                                  off Vsa16kF32 — half-laid by
                                                                  the 2026-05-26 Baton-scoping
                                                                  note; full rewrite still TBD)
D-MBX-6   ThoughtStruct = transparent hot/cold view over LanceDB; SurrealDB kanban as second
          view (optional; surreal_container BLOCKED A/B/C/D is NOT on the critical path)
```

### Cross-cutting tech debt named here

- **`TD-RESONANCEDTO-DUP-1`** (new in #418, **Deferred** per user) — two
  distinct `ResonanceDto` structs under the same name in
  `crates/thinking-engine`: `dto.rs:59` (ripple-field shape) vs
  `awareness_dto.rs:21` (multi-perspective S/P/O shape). Fold into D-MBX-2 (the
  `engine_bridge` re-encode-seam collapse).

---

## 4. PR #419 — `docs(odoo-savants): 25 AXIS-B evidence contracts (carve-out) + dispatch decision`

**Flagged explicitly because the question grouped it with #418 but it is
unrelated to surreal/mailbox.** Docs-only (+2567 across 26 files), the
odoo-savant evidence-contract carve-out feeding
`lance-graph-contract::reasoning`. Verdict: low merge risk; two things worth
recording:

- **Dispatch decision** (resolves the scaffold's open question): **one
  `Reasoner` impl per `ReasoningKind`**, not a data-driven registry —
  `CustomerCategoryReasoner`×4 / `PostingAnomalyReasoner`×3 /
  `NextBestActionReasoner`×12 / `OtherReasoner`×6. `Other(RECONCILE_MATCH)` is
  correctly disambiguated by `ReasoningContext.namespace`
  (`erp.k3.reconcile_match` vs `erp.k3.payment_reconcile`), not by code.
- **The real gate is the 14 `NEEDS-INPUT` blockers** for D-ODOO-SAV-4 (impl),
  which tie back to the cross-repo story:
  - **woa-rs feeds**: supplier lead/reliability/cost (L13 procurement),
    demand/movement history, per-move `date_due`/`paid_date` lateness
    (PartnerTrustAdvisor), RBAC role/company evidence
    (UserCompanyAccessAdvisor).
  - **lance Layer-2 alignment axioms** (candidates currently family `None`):
    `account.fiscal.position`, `product.pricelist`,
    `account.analytic.distribution.model`, `stock.*`, SKR03/04 exchange
    gain/loss codes (ExchangeAccountSelector).

One folded-in correction: **ProductCatalog family = `0x64`** (not the stale
`0x63` = `ogit:MRORepair`), per
`lance-graph-callcenter/src/odoo_alignment.rs:47-54`.

---

## 5. Navigability meta-finding — the surreal POC docs lack supersedure annotation

The supersedure of *"SurrealDB-on-Lance = the Zone-2 cold store"* by
*"LanceDB-leading; SurrealDB-as-view"* is **recorded only in `EPIPHANIES.md`
`E-RUBICON-RACTOR`** and the PR #418 plan §2.7. The following surfaces still
carry (or read as) the older framing without a supersedure pointer:

- `.claude/surreal/RECONCILIATION_with_canonical_plan.md:20` — table row
  *"SurrealDB-on-Lance persistence | Zone 2 `lance-graph-callcenter` +
  AriGraph SPO-G quads"*.
- `.claude/surreal/cognitive-substrate.md` — the substrate framing predating
  the LanceDB-leading ruling.
- `.claude/surreal/01_deps_substrate.md` … `12_clean_writer_invariants.md` —
  the 12-task POC reads as if SurrealDB is the persistence target.

A reader landing in `.claude/surreal/` first will get the superseded framing.
The board governance is append-only so the existing lines stay; a
**non-mutating pointer file** (e.g.
`.claude/surreal/SUPERSEDURE_NOTE_2026-05-27.md`) that names
`E-RUBICON-RACTOR` + plan §2.7 as the current ruling is the lowest-risk fix —
**explicitly NOT done in this handover** (out of scope: read-only synthesis +
appends only).

Recorded as `E-SURREAL-POC-UNANNOTATED-SUPERSEDURE` appended to `EPIPHANIES.md`
in the same commit as this handover.

---

## 6. Action surface (what a next session would do, in order)

1. **Merge PR #418** when ready — it is design-only, no behavioral risk, and
   the cited gates already prevent execution-before-readiness.
2. **Optionally add the surreal supersedure pointer** (§5) so future sessions
   landing in `.claude/surreal/` find the current SurrealDB-as-view ruling
   without first reading `EPIPHANIES.md`.
3. **Land `PR-NDARRAY-MIRI-COMPLETE`** in `adaworldapi/ndarray` — the literal
   unblock for everything downstream.
4. **Land `D-CE64-MB-1-impl`** (par-tile `Mailbox<T>` apex) — the literal
   unblock for D-MBX-1..6.
5. **Begin S1** (D-MBX-1: add migrated columns behind `mailbox-thoughtspace`
   feature), with feature-gated parallel `Arc<BindSpace>` per
   `I-LEGACY-API-FEATURE-GATED`.
6. **Doctrinal sweep for S5** — update `CLAUDE.md` "The Click" off
   `Vsa16kF32` (the 2026-05-26 Baton-scoping note is half the work; the rest
   is the body of "The Click" still written on the f32 carrier). Without this,
   S5 (delete the cycle plane) is blocked.

PR #419 is on its own track: low-risk merge; the 14 `NEEDS-INPUT` blockers
gate D-ODOO-SAV-4 (impl).

---

## 7. Cross-refs

**PRs:** [#418](https://github.com/AdaWorldAPI/lance-graph/pull/418) ·
[#419](https://github.com/AdaWorldAPI/lance-graph/pull/419)

**Plans (in `.claude/plans/`):** `bindspace-singleton-to-mailbox-soa-v1` (in
#418, not on this branch) · `causaledge64-mailbox-rename-soa-v1` ·
`cognitive-substrate-convergence-v1/v2/v3` · `atom-mailbox-substrate-v1` ·
`rung-persona-orchestration-v1` · `rung-ladder-grounding-v1` ·
`rung-mul-grounding-v1` · `3DGS-Cesium-BindSpace4-headstone-exploration`.

**Surreal POC (in `.claude/surreal/`):** 15 docs +
`RECONCILIATION_with_canonical_plan.md`. Framing partially superseded — see §2
+ §5.

**Epiphanies (in `EPIPHANIES.md`):** `E-BATON-1`, `E-CE64-MB-4`,
`E-MAILBOX-IS-BINDSPACE` (new #418), `E-RUBICON-RACTOR` (new #418),
`E-LADDER-SERVES-MAILBOX`, `I-VSA-IDENTITIES`, `I-LEGACY-API-FEATURE-GATED`,
`E-CONTRACT-NO-SERIALIZE`, `E-SURREAL-POC-UNANNOTATED-SUPERSEDURE` (new, this
handover).

**Source code anchors (on `main`):**
`crates/cognitive-shader-driver/src/{bindspace.rs:234,mailbox_soa.rs,driver.rs:55-116,engine_bridge.rs,bin/serve.rs:29}`,
`crates/lance-graph-contract/src/{collapse_gate.rs,cognitive_shader.rs:382,counterfactual.rs}`,
`crates/causal-edge/src/edge.rs`, `crates/p64-bridge/src/lib.rs` (the storage-ward conformance template).
