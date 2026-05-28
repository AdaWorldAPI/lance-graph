# OQ Catalogue — `bindspace-singleton-to-mailbox-soa-v1` follow-up (PR #418)

**Status:** FINDINGS / INPUT — parallel-session review of PR #418's plan
(`.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`) and supporting board
entries (`E-MAILBOX-IS-BINDSPACE`, `E-RUBICON-RACTOR`). Each entry is a
**question** for the owning sessions (the plan-author session for plan-§ updates;
the D-MBX-1..6 worker sessions for impl detail); answers route via this file or
the relevant board entry. **No code is proposed and no plan-doc is edited by this
PR.**

**Authored:** 2026-05-27, branch `claude/418-followup-findings`. Survey gate run
before authorship: open-PR overlap (clean — only #261 draft blackboard);
prior-art grep across `origin/main` for each candidate question (results
cited inline below); **Libet wall-clock budget framing explicitly rejected from
scope** because `.claude/surreal/cognitive-substrate.md:49-51` already rules
*"The numbers are operational, not Libet's 550/200 ms… Keep the phase structure;
reject the neuro-numerology."*

**Convention:** Mirrors `.claude/board/sprint-log-13/oq-catalog.md` shape.
`OQ-MBX-N` numbering continues the plan's own §8 (`OQ-1..4` inline) so that
plan-§8 and this file form one logical catalogue. Three valid resolutions per
entry, each useful as calibration: **(a)** "covered in `$foo.md` / `$session_id`"
→ close & cross-ref; **(b)** "real gap → queue as `D-MBX-7+` / plan §X update";
**(c)** "wrong framing because `$reason`" → close with the reason.

**Out of scope (deferred to separate proposals):**
- Libet wall-clock budget — contradicts `.claude/surreal/cognitive-substrate.md:49-51`.
- Cross-server baton replication — additive design, separate conversation
  (the existing `.claude/plans/multi-server-cognition-expansion-v1.md` is the
  starting point).

---

## Index

| # | Topic | Severity | Owner / Blocks |
|---|---|---|---|
| `OQ-MBX-7`  | `#[repr(C)]` byte-pin on `MailboxSoA<N>` itself? | P0 | plan §2 / D-MBX-1 |
| `OQ-MBX-8`  | `persisted_row: u32` semantics under Lance versioning / compaction | P0 | plan §2.5 / D-MBX-4 |
| `OQ-MBX-9`  | `ThinkingStyleI4_32D` placement in `MailboxSoA` | P1 | plan §3 / D-MBX-1 |
| `OQ-MBX-10` | Sea-star concurrency model post-S3 (`Send`, container, access) | P1 | plan §6 / D-MBX-3 |
| `OQ-MBX-11` | Mailbox-level eviction at the 64k–256k thought ceiling | P1 | plan §2.7 / D-MBX-3 |
| `OQ-MBX-12` | S1 dual-path discipline + field-isolation-matrix tests | P1 | plan §6 / D-MBX-1..2 |
| `OQ-MBX-13` | BE-endianness policy for the LE contract | P2 | plan §2.5 |
| `OQ-MBX-14` | Baton `u16` ↔ HHTL cascade-address encoding relationship? | P3 | PP-15 audit input |

---

## OQ-MBX-7 — `#[repr(C)]` byte-pin on `MailboxSoA<N>` itself?

**Question.** Plan §2.5 asserts THE little-endian contract is "one SoA layout
end-to-end, same bytes hot→disk, no re-encode at any boundary." How is the byte
layout of `MailboxSoA<N>` *enforced* — via `#[repr(C)]` on the container plus
per-column type pins, via inheritance from `SoaContainerHeader<N>`-style
descriptors, or via some other mechanism the plan doesn't surface?

**Why it matters.** The "same bytes hot→disk" claim is a hard invariant: any
field reordering, padding shift, or alignment surprise breaks `persisted_row`
(OQ-MBX-8) and the GoBD tombstone-witness audit trail. Today the pattern is
established for the *primitives* — `SoaContainerHeader<N>` ships as LE
`#[repr(C)]` in ndarray (commit `547824bc`, per `.claude/board/AGENT_LOG.md:186`),
`RungState` is a 16-B `#[repr(C)]` Pod
(`.claude/board/INTEGRATION_PLANS.md:148`), `OntologyDelta` is 32-B `#[repr(C)]`
(`.claude/plans/bindspace-columns-v1.md:262`), and `EPIPHANIES.md:257` already
treats `SoaContainerHeader` + `SoaColumns` as the LE-contract realisation of
`CANONICAL_DIMENSION_ALLOCATION.md`. But plan §2 defines `MailboxSoA<N>` as a
*container* of `[CausalEdge64; N]` / `[QualiaI4_16D; N]` / `[MetaWord; N]` /
`[u16; N]` arrays plus accumulator scalars, and the migration map in §3 does not
pin the container's `repr` or assert its `size_of`.

**Cross-ref / prior art.** `.claude/board/EPIPHANIES.md:257`
(dimension-allocation-IS-LE-contract);
`.claude/plans/cognitive-substrate-convergence-v3.md:578`
(`I-FEATURE-GATE-FIELD-ISOLATION` candidate iron rule — every v2-style layout
change requires field-isolation-matrix tests at the bit boundary; promotes
naturally to enforcement of the pin on every layout-bearing struct);
`.claude/specs/pr-ce64-mb-3-bindspace-efgh.md:102` (`#[repr(C)]` already specced
for bindspace columns).

**Blocks.** Plan §2.5 grounding; D-MBX-1 column-addition shape.

---

## OQ-MBX-8 — `persisted_row: u32` semantics under Lance versioning / compaction

**Question.** Plan §2.5 says *"`ShaderCrystal.persisted_row` is a pointer to the
same SoA row laid down in Lance, not a serialized copy in a different shape."*
What does the `u32` actually denote under Lance's versioned/compacting storage
model — a Lance `_rowid`, a (fragment-id, fragment-offset) pair, a stable
RecordBatch row key, or something else? What happens to the value when a Lance
compaction rewrites the fragment, when the dataset rolls a new version, or when
the row is rewritten by an MVCC update?

**Why it matters.** This is the load-bearing finding of the survey, because the
write path **isn't built yet**:
`crates/cognitive-shader-driver/REFACTOR_NOTES.md:129` says verbatim
*"BindSpace is in-memory only. `EmitMode::Persist` sets `persisted_row` in the
crystal but doesn't write to disk."* — and indeed
`crates/cognitive-shader-driver/src/driver.rs:458` does
`EmitMode::Persist => Some(resonance_dto.top_k[0].row)`, i.e. records the
in-RAM top-k row index rather than a Lance row identity. The §2.5 "pointer to
the same SoA row" claim is therefore forward-looking; D-MBX-4 (death → SPO-G
quad + Lance tombstone-witness) is the deliverable that has to make it true,
and the version/compaction semantics need to be pinned **before** that
deliverable picks a row-identity type, or the tombstone-witness back-pointer
will silently break across the first compaction.

**Cross-ref / prior art.** Lance versioning is extensively discussed at the
*session* level (`.claude/SESSION_LANCE_ECOSYSTEM_INVENTORY.md:50-56`,
`.claude/adr/0001-archetype-transcode-stack.md:62-113`,
`.claude/FINAL_STACK.md:41-86`, `.claude/VISION_ORCHESTRATED_THINKING.md:35-179`),
all treating Lance's MVCC + time-travel as the persistence model — but **none**
pin what the `u32` row identity means under that model. The plan's hot/cold
view (§2.7) and tombstone-witness invariant (`E-LADDER-SERVES-MAILBOX` §6)
both depend on the answer.

**Blocks.** Plan §2.5 / §2.7 grounding; D-MBX-4 row-identity type; D-MBX-6
hot/cold transparent-view contract.

---

## OQ-MBX-9 — `ThinkingStyleI4_32D` placement in `MailboxSoA`

**Question.** `ThinkingStyleI4_32D` (i4 × 33 = 3 Pearl + 9 Rung + 5 Σ + 8 Ops + 4
Presence + 4 Meta, riding the shipped ndarray i4-32 unpack) is a fully designed
basis (`EPIPHANIES.md:234..287`, resolved out of `E-I4-META-1` /
`.claude/knowledge/atom-basis-inventory.md`). Plan §3's column migration map
does not mention it. Is it (a) an owned per-row SoA column
(`[ThinkingStyleI4_32D; N]` at ~16 B/row), (b) a dispatch-time projection of
`MetaWord` (already in §3), or (c) only a p64-bridge parameter (so it never
materialises as a mailbox column)?

**Why it matters.** Parallel to OQ-2 (temporal/expert fold) but for a type the
plan didn't flag. If (a), the hot footprint goes from ~6 KB/thought (§2.7) to
~6 KB + 16 B/thought — negligible for the 64k–256k ceiling but still a budget
line. If (b), the projection algorithm needs to be specified (which `MetaWord`
bits encode the 33 lanes? Or is it a separate read off `meta` + `edges`?). If
(c), `MailboxSoA` never holds it and the call sites that need a style read it
fresh from p64-bridge each cycle.

**Cross-ref / prior art.** `.claude/board/EPIPHANIES.md:234, 253, 271, 287, 293,
392` (`ThinkingStyleI4_32D` basis decision);
`.claude/knowledge/atom-basis-inventory.md:3-54` (lanes-as-atoms, i4×33 = the
contract); `.claude/plans/rung-ladder-grounding-v1.md:94` (RungLevel = R1-R9
dim-group of `ThinkingStyleI4_32D`); `crates/p64-bridge/src/lib.rs`
(`ThinkingStyle → layer_mask + combine + contra` — already shipped, no
re-encode).

**Blocks.** Plan §3 migration map; D-MBX-1 column-addition shape.

---

## OQ-MBX-10 — Sea-star concurrency model post-S3

**Question.** Plan §6 step S3 says *"`ShaderDriver` holds a sea-star of
mailboxes (kill the 4096 singleton in `serve.rs`)."* What is the *shape* of the
container — `Vec<MailboxSoA>`, `HashMap<MailboxId, MailboxSoA>`, a pool with
slot reuse, a sharded set keyed by `w_slot`? Is `MailboxSoA<N>` `Send`/`Sync`
today (and if not, what's the access discipline — a per-mailbox actor in the
ractor outer-swarm, or driver-thread-pinned)? What's the supervisor's failure
model when one mailbox in the sea-star panics?

**Why it matters.** S3 is the deliverable where the singleton actually dies;
the container shape and the access discipline determine the whole sea-star
runtime contract (ractor child set, supervision tree depth, par-tile mapping).
`E-CE64-MB-4`'s ownership guarantee is per-mailbox; the cross-mailbox layer
needs its own specification.

**Cross-ref / prior art.** `.claude/specs/pr-ce64-mb-1-par-tile-crate.md`
(par-tile `Mailbox<T>` apex — InMemory / Tokio / SupabaseSubMailbox backings;
gate D-CE64-MB-1-impl); `D-PERSONA-5` (ractor outer-swarm runtime); plan §5
lifecycle ("spawn → think → emit → die → tombstone") implies the actor model
but does not commit on the container/scheduler.

**Blocks.** D-MBX-3 (driver-singleton kill); D-MBX-4 (death → tombstone path
needs a "from where" answer).

---

## OQ-MBX-11 — Mailbox-level eviction at the 64k–256k thought ceiling

**Question.** Plan §2.7 pins a hot-tier ceiling of **~64k–256k thoughts**
(~6 KB/thought ≈ 300–600 MB at 64k). What signal reclaims a mailbox when the
ceiling is approached — global LRU on mailbox last-touch, supervisor
back-pressure from a working-set sensor, ractor mailbox idle-timeout, an
energy-decay threshold (`MailboxSoA.energy` integrated below floor for K
cycles)? Slot-level LRU on `AttentionMask` slots
(`pr-ce64-mb-5-mailbox-soa-attentionmask.md §2.2`) handles *within-mailbox*
slot reuse, not *between-mailbox* reclaim — those are different layers.

**Why it matters.** Without a named eviction signal, the sea-star can't be
sized — it either over-allocates (OOM at the ceiling) or starves
(over-eager reclaim that thrashes warm mailboxes). The §2.7 ceiling is a
capacity *claim*; this OQ asks for the *policy* that enforces it.

**Cross-ref / prior art.** `pr-ce64-mb-5-mailbox-soa-attentionmask.md §2.2`
(slot-level LRU — wrong layer); `E-LADDER-SERVES-MAILBOX` §5 (ghost-tier
preemptible to zero — the *capability*, but not the *trigger*); D-CSV-7
(MailboxSoA accumulator, shipped — has `energy` and `last_emission_cycle` that
could feed a decay signal).

**Blocks.** D-MBX-3 supervisor wiring; D-MBX-6 hot/cold spill discipline.

---

## OQ-MBX-12 — S1 dual-path discipline + field-isolation-matrix tests

**Question.** Plan §6 says S1–S4 run with both paths live behind the
`mailbox-thoughtspace` feature, with the v1 `Arc<BindSpace>` accessors
*"feature-gated to a documented no-op with a migration pointer — never silently
change semantics."* During S1, when does a non-feature-gated caller (e.g. the
`engine_bridge` per-row read/write surface on the legacy path) get a
no-op-vs-route-through-mailbox answer? What test artefact gates each of S1, S2,
S3 — specifically, where is the **field-isolation-matrix test** required by
`I-FEATURE-GATE-FIELD-ISOLATION` (candidate iron rule, CSV-v3:578) queued?

**Why it matters.** `.claude/board/sprint-log-11/meta-review-opus.md:133` records
that codex catches the v1-API-under-v2 anti-pattern repeatedly (4 instances in
PR #383 alone) — *"backward-compat shims for layout-breaking feature flags
require systematic test coverage of the bit boundary."* This migration is
load-bearingly that pattern (S1 adds columns under a feature; S2 routes through
them; S5 deletes the old carrier). Without the field-isolation matrix on the
critical bit boundaries (`MetaWord`, `CausalEdge64` v2 reclaimed bits per
`I-LEGACY-API-FEATURE-GATED`), the migration will hit the same codex catches.

**Cross-ref / prior art.** `.claude/plans/cognitive-substrate-convergence-v3.md:406,578`
(I-FEATURE-GATE-FIELD-ISOLATION candidate);
`.claude/board/sprint-log-11/meta-review-opus.md:133` (the 4-instance evidence
trail); plan §7 invariant #5 (I-SUBSTRATE-MARKOV preserved); plan §6 S1–S4
gating ladder.

**Blocks.** D-MBX-1 acceptance criteria; D-MBX-2 acceptance criteria.

---

## OQ-MBX-13 — BE-endianness policy for the LE contract

**Question.** The LE-contract framing (plan §2.5; `E-CONTRACT-NO-SERIALIZE`)
assumes a little-endian host throughout the vertical (shader-driver →
`lance-graph-contract` → Lance storage). What is the policy on big-endian
targets — panic at startup, transparent byteswap at the I/O membrane,
feature-gate the LE contract behind `cfg(target_endian = "little")`, or declare
BE explicitly unsupported?

**Why it matters.** Lance's on-disk format is LE; cross-architecture is mostly
theoretical for the current deploy targets, but a stated policy is cheaper now
than a surprise CI failure on a future aarch64-BE / s390x experiment. Also
relevant if the multi-server cognition plan
(`.claude/plans/multi-server-cognition-expansion-v1.md` §4 "deterministic
apply") ever crosses arch boundaries in the Raft log.

**Cross-ref / prior art.** `.claude/plans/multi-server-cognition-expansion-v1.md`
§4 (deterministic apply across machines = precondition for replication);
`E-CONTRACT-NO-SERIALIZE` (compile-time handshake, no membrane serialize —
implicitly LE).

**Blocks.** Plan §2.5 wording precision; LE-contract crate-level docs.

---

## OQ-MBX-14 — Baton `u16` ↔ HHTL cascade-address encoding relationship?

**Question.** The LE baton is `(u16 target, CausalEdge64)` (per `E-BATON-1`,
plan §7 invariant #2). HHTL is discussed as a 4-nibble cascade-level encoding
in transcripts (`SESSION_*`, `IDEA_JOURNAL_*`). Is the baton's `u16` target
meant to encode (or be derivable from) the HHTL cascade-address scheme — so
that "where the baton lands" carries cascade-level information — or is the
`u16` strictly a target *row index* into the receiving mailbox with no HHTL
structure?

**Why it matters.** PP-15 (`baton-handoff-auditor`,
`EPIPHANIES.md:466,498`) is the meta-review savant whose remit covers exactly
this seam; getting the question in writing before PP-15's first scheduled pass
on #418 means PP-15 has the question, not the rediscovery, on input. Either
answer is fine — *"no, baton `u16` is a flat row index, cascade addressing
lives elsewhere"* is a valid closure.

**Cross-ref / prior art.** `crates/lance-graph-contract/src/collapse_gate.rs`
(`CollapseGateEmission` / `push_baton` / `wire_cost_bytes`);
`.claude/board/EPIPHANIES.md:466,498` (PP-15 baton-handoff-auditor scope);
HHTL discussion across `.claude/SESSION_LANCE_ECOSYSTEM_INVENTORY.md`,
`.claude/IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md`, etc.

**Blocks.** Nothing in the immediate D-MBX-1..6 critical path; PP-15 audit
input.

---

## Resolved / not-filed (survey closed these before authorship)

- **TD-RESONANCEDTO-DUP-1 severity re-rate** — verified P3 is correct.
  `crates/thinking-engine/src/dto.rs:59` and
  `crates/thinking-engine/src/awareness_dto.rs:21` define `ResonanceDto` in
  **separate modules** (`pub mod dto;` + `pub mod awareness_dto;` in
  `lib.rs:17,32`), so there is no crate-root name collision and the crate
  compiles fine. All 6 consumers (`bf16_engine.rs`, `composite_engine.rs`,
  `dual_engine.rs`, `engine.rs`, `f32_engine.rs`, `signed_engine.rs`) import
  only `crate::dto::ResonanceDto`; nobody currently imports the
  `awareness_dto::` variant by name, so even cross-import confusion is latent
  rather than active. P3 (cosmetic type-dup) stands; folding into D-MBX-2 per
  the existing TD entry remains the right disposition.

- **Libet wall-clock budget** — closed before authorship per
  `.claude/surreal/cognitive-substrate.md:49-51`; see preamble.
