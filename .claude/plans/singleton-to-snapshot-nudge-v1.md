# Plan: Singleton → Snapshot Nudge — every shared-mutable singleton becomes a per-owner SoA with Arc-swap COW snapshot

**Version:** v1
**Date:** 2026-06-07
**Status:** PROPOSAL
**D-ids:** D-SNGL-1 through D-SNGL-7
**Branch:** `claude/stoic-turing-M0Eiq`

---

## The thesis

The workspace has one architectural direction for shared state:

> **No shared mutable singleton. State is owned per-mailbox as a `MailboxSoA<N>`,
> read via cycle-coherent Arc-swap COW snapshots, and calcified to SPO + Lance
> tombstone. Cross-boundary state is the discrete LE baton, never a materialized
> singleton.**

This is already ratified across three epiphanies and two plans:

- `E-MAILBOX-IS-BINDSPACE` — `MailboxSoA<N>` *is* the per-mailbox thoughtspace;
  the singleton `Arc<BindSpace>` is dissolved, not copied.
- `E-BATON-1` — inter-mailbox state is the `(u16 target, CausalEdge64)` baton;
  no persisted/transmitted singleton.
- `E-DEINTERLACE-TWO-SCALES` — deinterlace is one operation at two scales;
  byte-scale is the SoA Arc-swap snapshot at `cycle()` granularity.
- Plan `bindspace-singleton-to-mailbox-soa-v1` — dissolves the `ShaderDriver`
  `Arc<BindSpace>` singleton specifically.
- Plan `cycle-coherent-soa-snapshot-v1` — the Arc-swap COW snapshot mechanism
  (no-cross-cycle-lag guarantee).

**What is missing:** a single, enumerated audit that nudges *every*
singleton-shaped construct in the workspace onto this architecture, so the
migration is exhaustive rather than ad-hoc. This plan is that audit.

---

## Two kinds of "singleton" — only one is a target

The grep for `LazyLock` / `OnceLock` / `static` / `Arc<…>` returns two
fundamentally different shapes. The distinction is the whole game:

### NOT a target — read-only immutable codebooks (LEAVE AS-IS)

These are built once and never mutated. They are the I-VSA-IDENTITIES
Layer-2 role catalogues and Layer-1 codebooks. A `LazyLock` here is correct
and idiomatic — it is a const table, not shared mutable state.

| Construct | Home | Verdict |
|---|---|---|
| `SUBJECT_KEY` / `PREDICATE_KEY` / … role keys | `contract::grammar::role_keys` | ✅ Keep — immutable role identity codebook |
| `FINNISH_KEYS` / `TENSE_KEYS` / `NARS_KEYS` | `contract::grammar::role_keys` | ✅ Keep — immutable codebook |
| `KUNDE_KEY` / `RECHNUNG_KEY` / … callcenter keys | `contract::grammar::role_keys` | ✅ Keep — immutable codebook |
| `VECTOR_DISTANCE_*_UDF` / `HAMMING_*_UDF` | `lance-graph::datafusion_planner::udf` | ✅ Keep — DataFusion UDF registration, immutable |
| `simd_caps()` singleton | `ndarray::simd_caps` | ✅ Keep — hardware capability probe, immutable |

**Rule:** a `LazyLock<T>` where `T` is never mutated after init is a codebook,
not a singleton. The data-flow invariant (`ndarray/.claude/rules/data-flow.md`
§2) already blesses these: "Caches use interior mutability (`RwLock`,
`LazyLock`) or are built once."

### IS a target — shared mutable runtime state (NUDGE)

These hold mutable runtime state behind a shared handle. They are the
singletons the architecture dissolves into per-owner SoA + snapshot.

| Construct | Home | Nudge | Owning plan |
|---|---|---|---|
| `ShaderDriver.bindspace: Arc<BindSpace>` | cognitive-shader-driver | Dissolve into per-mailbox `MailboxSoA<N>`; driver holds a sea-star of mailboxes | `bindspace-singleton-to-mailbox-soa-v1` (D-MBX-3/5) |
| `BindSpace::zeros(4096)` in `bin/serve.rs` | cognitive-shader-driver | Delete; mailboxes allocate their own SoA | `bindspace-singleton-to-mailbox-soa-v1` (D-MBX-5) |
| `AttentionMatrix.gestalt` (shared mutable summary, drifting via `unbundle_from`) | `lance-graph-planner::cache::kv_bundle` | Either rebuild-from-scratch or raw-sum+count; the gestalt is a snapshot read, not an incrementally-mutated singleton | THIS PLAN (D-SNGL-3) + TD-UNBUNDLE-FROM-1 |
| `ATTENTION_CACHE` / `LINEAR_CACHE` `LazyLock<RwLock<…>>` | `ndarray/crates/burn::ops::matmul` | Audit: is this a JIT-kernel cache (keep, like UDFs) or runtime belief state (nudge)? | THIS PLAN (D-SNGL-4) |
| any `Arc<RwLock<…>>` / `Arc<Mutex<…>>` runtime caches surfaced by the audit | workspace-wide | Classify codebook-vs-singleton; nudge only the latter | THIS PLAN (D-SNGL-2) |

---

## Deliverables

### D-SNGL-1 — Workspace-wide singleton census

Grep every crate for `LazyLock` / `OnceLock` / `OnceCell` / `Lazy` / `static …` /
`Arc<RwLock` / `Arc<Mutex`. Classify each hit into **codebook** (immutable, keep)
or **singleton** (shared-mutable, nudge). Output: a census table appended to
`docs/architecture/soa-three-tier-model.md` § "Singleton census" so the
codebook-vs-singleton verdict is recorded once and not re-litigated.

### D-SNGL-2 — Classification gate (the one rule, codified)

A doc-level decision procedure (mirrors the lab-vs-canonical decision procedure):
> Is the static ever mutated after init? **No → codebook, keep.** **Yes → is it
> per-owner runtime state? Yes → nudge to `MailboxSoA<N>` + snapshot. No (truly
> process-global, e.g. a JIT-kernel cache) → keep behind `RwLock` per data-flow
> §2, but document why it is not per-owner.**

### D-SNGL-3 — `AttentionMatrix.gestalt` correctness + snapshot shape

Fix TD-UNBUNDLE-FROM-1: the gestalt is a derived snapshot, not an
incrementally-subtracted singleton. Switch to raw-sum + count so the gestalt is
exactly `(sum[d] / count).round()`, OR rebuild on read. Remove the deprecated
`unbundle_from` once no caller remains. The gestalt then matches the snapshot
doctrine: a coherent read over the heads at a cycle stamp, not a drifting
accumulator.

### D-SNGL-4 — burn matmul cache audit

Classify `ATTENTION_CACHE` / `LINEAR_CACHE` in `ndarray/crates/burn`. If they
cache *compiled kernels* keyed by shape, they are JIT-kernel caches (keep,
document as process-global per D-SNGL-2). If they cache *runtime activations /
beliefs*, they are singletons and must move to the SoA. Record the verdict.

### D-SNGL-5 — Snapshot trait adoption checklist

For each nudged singleton, the migration target is the same trait surface from
`cycle-coherent-soa-snapshot-v1`: implement `SnapshotProvider` (D-SOA-SNAP-2),
return a `MailboxSoaSnapshot` (D-SOA-SNAP-1) under a cycle stamp. This deliverable
is the per-crate checklist binding each nudge to the snapshot contract so the
migration is uniform.

### D-SNGL-6 — No-cross-cycle-lag falsification per nudged crate

Each nudged crate inherits the D-SOA-SNAP-5 test shape: writer thread advancing
cycles + N reader threads snapshotting; assert every snapshot is single-cycle.
The test is the merge gate for that crate's nudge.

### D-SNGL-7 — Board hygiene + EPIPHANIES

This plan + INTEGRATION_PLANS prepend + STATUS_BOARD rows + (if the census
surfaces a genuinely new finding) an EPIPHANIES entry. The candidate epiphany:
`E-SINGLETON-IS-CODEBOOK-OR-SOA` — every static is exactly one of two things,
and the test (mutated-after-init?) is mechanical.

---

## Execution ordering

1. **D-SNGL-1 + D-SNGL-2 first** (census + gate) — settles codebook-vs-singleton
   verdicts before any code moves. Pure doc/audit work.
2. **D-SNGL-3** (AttentionMatrix) — the one concrete correctness bug; ships with
   the kv_bundle deprecation already landed this session.
3. **D-SNGL-4** (burn audit) — independent; classify, then keep-or-nudge.
4. **D-SNGL-5/6** per nudged crate — gated on `cycle-coherent-soa-snapshot-v1`
   D-SOA-SNAP-1/2 landing first (the trait surface must exist to adopt it).
5. **D-SNGL-7** lands with D-SNGL-1 (board hygiene is same-commit, per CLAUDE.md).

---

## Non-goals

- Not touching read-only codebooks (role keys, UDFs, simd_caps). Those are
  correct as-is.
- Not re-deriving the BindSpace dissolution — that is owned by
  `bindspace-singleton-to-mailbox-soa-v1`. This plan references it, does not
  duplicate it.
- No new snapshot mechanism — reuses `cycle-coherent-soa-snapshot-v1`'s trait.

---

## Cross-references

- `bindspace-singleton-to-mailbox-soa-v1` — BindSpace singleton dissolution
- `cycle-coherent-soa-snapshot-v1` — the Arc-swap COW snapshot mechanism reused here
- `EPIPHANIES.md` E-MAILBOX-IS-BINDSPACE, E-BATON-1, E-DEINTERLACE-TWO-SCALES
- `TECH_DEBT.md` TD-UNBUNDLE-FROM-1 — the AttentionMatrix gestalt drift
- `ndarray/.claude/rules/data-flow.md` §2 — the "caches built once" invariant
  that keeps codebooks legal
- `docs/architecture/soa-three-tier-model.md` — the zero-copy lifecycle target
