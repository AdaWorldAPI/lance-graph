# SoA Value-Tenant Migration — Plan v1 (harvest brief + 5+3 sign-off)

> **Status:** BRIEF (2026-06-24). This is NOT the migration — it is the
> orientation + harvest brief + sign-off protocol that *produces* the
> migration. The plan body (§5) is a skeleton filled only after the harvest
> session returns the inventory (§4) and both 5+3 panels (§6) sign off.
>
> **Authoring honesty:** every file/type pointer in §2 is named from
> in-context canon (CLAUDE.md, `substrate-unification-thesis.md`, the node
> canon, the iron rules) — **not** from a verified read this session. The
> harvest session's first job is to **Read each one fully and confirm or
> correct the pointer.** A pointer here is a lead, never a fact.

---

## 0. Why a brief instead of a plan

You cannot migrate value tenants you haven't *read*. The failure mode this
brief exists to prevent: a session greps `ValueTenant`, sees 9 hits, writes a
plan against the 9 hits, and misses the tenant that's constructed through a
`From` impl in a consumer crate, or the one whose bytes are reclaimed under a
feature flag. The migration's correctness is bounded by the inventory's
completeness, and the inventory's completeness is bounded by **read-depth**.

So the work is **4 sessions, 3 phases**:

| Phase | Session(s) | Output |
|---|---|---|
| 0 — orientation | this doc | the brief (§2–§4 + §6) |
| 1 — harvest | **1 session** | the inventory (§4 schema), filled |
| 2 — sign-off | **2 sessions**, each a 5+3 | two independent LAND/HOLD/REJECT verdicts on the filled plan |

The two sign-off panels are run **independently** (diverse-redundancy, the
medcare MySQL-witness pattern): convergence between them is signal, divergence
names the real seam. The main thread reconciles and only then writes §5 for
real.

---

## 1. What a "value tenant" is (the object of the migration)

The canonical node is `key(16) | edges(16) | value(480)` = 512 B
(`canonical_node.rs`, operator-LOCKED, RESERVE-DON'T-RECLAIM). A **value
tenant** is a typed claim on some of those 480 value bytes — the `ValueSchema`
says how the slab is carved, a `ValueTenant` is one carve. The §8 strong form
(`substrate-unification-thesis.md`) proposes an *additive* `ValueSchema::
Homogeneous` (N × 16-byte `(part_of:is_a)` facets) **alongside** the existing
tenants. The migration question for each existing tenant is therefore:
**KEEP as-is / homogenize to a facet / PQ-code it / deprecate** — and whether
that move is layout-preserving.

---

## 2. Where to look — READ these, do not grep-to-conclude

> **Read-discipline (WoA L40 / lance-graph reading-ladder):** `grep`/`sed`/
> `tail`/`head` are **locators**, never comprehension. Use Grep to *find* the
> type; use **Read on the whole file** to understand it. For any file >2000
> lines, multiple Reads with offset/limit covering the *entire* relevant
> region — never a single snippet. Declare `depth=full` only with a real read
> behind it (proof-of-read: file + ~3 section names you can cite).

The harvest must Read, fully, in this order:

1. **`canonical_node.rs`** (lance-graph core) — the authoritative layout.
   `NodeGuid` / `EdgeBlock` / `NodeRow`, the `const _` 16/16/512 size asserts,
   and **every** `ValueTenant` / `ValueSchema` definition + its byte offsets.
   This file is the ground truth; everything else is a consumer of it.

2. **`lance-graph-contract`** — the zero-dep type crate. The four BindSpace SoA
   columns (`FingerprintColumns` / `QualiaColumn` / `MetaColumn` / `EdgeColumn`,
   PR #223) and any tenant/schema enums that consumers build against. The
   `SoaEnvelope` + the three-tier model (`docs/architecture/soa-three-tier-
   model.md`, PR #477 tombstone — zero-copy creation→tombstone, no inter-mailbox
   serialization). `last_active_cycle` (the renamed consumption stamp).

3. **The dynamics tenant** — `INERTIA_SLOT` and the #509/#511/#513 arc
   (`SoaMemberSpec` calibration). The BF16 buffer / impulse-permeability axis
   (`perturbation-sim/src/place_buffer.rs`, #607).

4. **The structure + identity tenants** — `perturbation-sim/src/cascade_key.rs`
   (#605): V1/V2 spatial + V3 `(part_of:is_a)` 8:8 tile; and `helix_place`
   (identity, #607). These are the *prototype* tenants the §8 facet generalizes.

5. **Every producer/consumer that constructs or reads a tenant** — this is the
   part a grep-only pass misses. Read (or have an Explore sub-agent map, then
   Read the hits):
   - `lance-graph-supervisor`, `symbiont`, `lance-graph-planner` (in-workspace).
   - The BBB consumers that pull tenants by `*Port::class_id` /
     `canonical_concept_id`: **smb-office-rs**, **medcare-rs**, **woa-rs**.
     (Per the BBB barrier — they must *pull*, never construct a `*Bridge` or
     copy the codebook; the harvest confirms they don't.)
   - The q2 / OGAR consumers flagged in cont.³⁹ (e.g. q2 `osint-bake/fma.rs`
     `NodeGuid::new_v2(...)` — a 7-group API that does **not** exist in
     `canonical_node`; an `I-LEGACY-API-FEATURE-GATED` live blocker to record,
     not silently fix).

6. **The board registry** — `.claude/board/LATEST_STATE.md` § *Current Contract
   Inventory*. This is the workspace's own list of what types exist; a tenant in
   the code but not on the board (or vice-versa) is itself a finding.

---

## 3. What to ALWAYS pay attention to (the gates every tenant must clear)

For each tenant the harvest touches, hold these in mind — they are the
acceptance criteria the §6 panels will check:

- **Layout class.** Is the proposed move *layout-preserving* (a new
  `ValueSchema` variant alongside; offsets unchanged) or *layout-breaking* (the
  480 bytes re-carved)? Breaking ⇒ canon-level ⇒ needs `ENVELOPE_LAYOUT_VERSION`
  bump **and** the operator's nod. The 16/16/480 split and the `const _` asserts
  do not move without that. (RESERVE-DON'T-RECLAIM: a zeroed/unused region is
  *not consulted*, never *compacted away*.)
- **`I-LEGACY-API-FEATURE-GATED`.** The same accessor name must never mean two
  things under two feature flags. Any v1 accessor over bytes a v2 layout
  reclaims ⇒ route through the canonical mapping OR feature-gate to a documented
  no-op + migration pointer, **and** ship a *field-isolation matrix test* (write
  each field, assert all others unchanged). Sprint-11 caught this 5×; expect
  codex P1 to flag it.
- **The conflation trap** (`substrate-unification-thesis.md` §1 / §8.1). Does
  the tenant fuse two axes that should be orthogonal? `part_of` ⊥ `is_a`;
  location ⊥ impulse-permeability; **scalars are not hierarchical** — a
  susceptance / price / timestamp forced into an 8:8 `(part_of:is_a)` tile is
  the split-error run in reverse. Scalar tenants → PQ-code facet, **gated on
  F-1** (codebook fidelity) + F-code (lossless containment), never a raw
  homogenize.
- **Substrate iron rules.** `I-SUBSTRATE-MARKOV` (transition paths bundle, never
  raw XOR); `I-VSA-IDENTITIES` (bundle *identities*, never content/CAM-PQ codes;
  CAM-PQ is for search, VSA for bundling — separate tools). A homogenize that
  superposes content violates this.
- **No serialization in the hot path** (ADR-022 / three-tier model). The tenant
  is zero-copy from creation to Lance tombstone; the migration must not
  introduce a serialize/deserialize step to change a carve.
- **Producers ⊥ consumers.** Every tenant has a write side and N read sides;
  the migration is incomplete until *both* are accounted for. This is the
  baton-handoff surface (cross-crate DTO match).

---

## 4. What the harvest returns — the inventory schema

One row per value tenant. The harvest fills this table and **nothing else**
(no migration code in Phase 1 — inventory only):

| field | meaning |
|---|---|
| `name` | the tenant / `ValueSchema` variant |
| `def_site` | `file:line` of the definition (confirmed by Read, not grep) |
| `offset/width` | byte range within the 480 value bytes |
| `axis` | which §2 basis axis it serves — identity / structure / dynamics / truth / composition |
| `codec` | how the bytes decode — place⊕residue / PQ code / BF16 / raw / VSA |
| `producers` | crates that *write* it (`file:line`) |
| `consumers` | crates that *read* it (`file:line`), incl. BBB pull-sites |
| `migration_class` | **KEEP** / **homogenize-to-facet** / **PQ-code** / **deprecate** |
| `layout` | preserving / breaking (and the version-bump cost if breaking) |
| `conflation_risk` | does it fuse two axes? (yes → must split first) |
| `gate` | which falsifier must be green first — F-1 / F-code / F-collapse / operator-nod / none |
| `board_status` | present in LATEST_STATE Contract Inventory? (drift if not) |

Plus a short prose **completeness note**: what the harvest could *not* confirm
by read (disk-walled crate, missing source, ambiguous From-impl) — named, not
hidden. Silent truncation reads as "covered everything" when it didn't.

---

## 5. The migration body (SKELETON — written for real only after §4 + §6)

Direction, fixed; specifics, deferred to the filled inventory:

1. **Additive contract first.** Land `ValueSchema::Homogeneous` + a
   `FacetCascade` type (`facet_classid(4) | 6×(8:8)=12` = 16 B) as a *new
   variant alongside* `ValueTenant` — layout-preserving, no version bump. Behind
   a feature flag. Field-isolation matrix tests from day one.
2. **Per-tenant, in migration_class order:**
   - `homogenize-to-facet` tenants (those that already ARE `part_of:is_a`) move
     first — lowest risk, prototype is `cascade_key` V3.
   - `PQ-code` tenants (scalars) move only after **F-1 is green** (ndarray-side).
     Until then they stay `KEEP`.
   - `KEEP` tenants that are irreducibly heterogeneous stay `ValueTenant` —
     this is expected and healthy (the §8.5 *homogeneity-non-closure* bound:
     if everything resists homogenizing, §8 reduces to "key is a schema
     pointer" and that's the honest finding, not a failure to force).
   - `deprecate` tenants get the `I-LEGACY-API-FEATURE-GATED` treatment
     (no-op + migration pointer + paired "corruption is observable" test).
3. **Consumer bump last**, per BBB barrier: consumers re-point to the new pull
   API; no `*Bridge` construction. The q2 `new_v2` blocker is resolved *with the
   operator*, not silently.
4. Every PR is doc-+-board-hygiene complete in the same commit (LATEST_STATE
   Contract Inventory row, PR_ARC entry).

---

## 6. The 5+3 sign-off (two independent sessions)

Each sign-off session runs the OGAR 5+3 hardening pattern over the **filled**
plan (§4 inventory + §5 body): **5 research savants** produce object-level
findings, **3 brutally-honest reviewers** gate. Run the two sessions
independently; reconcile on the main thread.

**5 research savants** (from the ensemble — match the axes the inventory touches):

1. **`truth-architect`** — measurement-before-synthesis; is every migration_class
   backed by a read/number, or is it a projection? Flags F-1-gated rows that
   sneak forward.
2. **`iron-rule-savant`** — binary YIELDS/VIOLATES against the four iron rules +
   AP catalogue. Any VIOLATES = auto-REJECT.
3. **`dto-soa-savant`** — does any tenant's move smuggle in a *new struct/trait/
   bridge* instead of a new SoA column/variant? (PR #223: capability lands as a
   column, not a layer.)
4. **`baton-handoff-auditor`** — the producers⊥consumers surface: does each
   tenant survive the cross-crate roundtrip after the carve change? CATCH-CRITICAL
   / CATCH-LATENT / CLEAN.
5. **`container-architect`** (or `core-first-architect` for the OGAR/ClassView
   seam) — the 480-byte layout itself: offsets, asserts, reserve-don't-reclaim,
   version-bump accounting.

**3 brutally-honest reviewers** (the gate):

1. **`brutally-honest-tester`** — P0/P1/P2 ledger + binary LAND/HOLD/REJECT;
   field-isolation matrix coverage is mandatory for any reclaim.
2. **`overclaim-auditor`** — every grade vs its evidence; "layout-preserving"
   claimed without the asserts checked = flagged.
3. **`firewall-warden`** (or `dilution-collapse-sentinel`) — non-negotiables
   (no hot-path serialization, no PII labels, BBB barrier) **and** that the
   homogenize doesn't *dilute* a sharp tenant or *collapse* a distinct one into
   a facet that loses its meaning.

**Reconciliation rule:** the two panels' convergence is the strongest evidence;
where they disagree is the real seam → that tenant's row is re-opened, not
averaged. No tenant migrates on a single panel's say-so.

---

## Cross-references

- `substrate-unification-thesis.md` §1 (split-conflated-axes), §8.1
  (homogeneous facet, the conflation trap), §8.5 (homogeneity-non-closure KILL).
- `canonical_node.rs` — the 16/16/480 canon + `const _` asserts.
- `perturbation-sim/src/{cascade_key.rs (#605), place_buffer.rs (#607)}` — the
  prototype identity/structure/dynamics tenants.
- Iron rules: `I-LEGACY-API-FEATURE-GATED` (the reclaim discipline + 5-instance
  catalogue), `I-SUBSTRATE-MARKOV`, `I-VSA-IDENTITIES`.
- `docs/architecture/soa-three-tier-model.md` (PR #477) — zero-copy, no
  inter-mailbox serialization.
- `.claude/agents/BOOT.md` — the savant ensemble + 5+3 pattern.
- `.claude/board/LATEST_STATE.md` § Contract Inventory — the type registry the
  harvest cross-checks.
