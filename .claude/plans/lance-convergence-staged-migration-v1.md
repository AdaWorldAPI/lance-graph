# lance convergence — the STAGED migration plan (v1)

> **Status:** MEASURED / ready-to-execute for stages 0–1; stage 3 is
> PROBE-GATED. Every claim carries file:line from a read of the CURRENT tree
> (origin/main `afeb0458`) or a crates.io / GitHub release query dated
> 2026-09-05. Where a claim was corrected mid-pass, the correction is kept
> in §7 rather than silently overwritten.
>
> **READ BY:** anyone proposing a lance / lancedb bump, touching
> `LanceCycleWriter` / `VersionedGraph` / `persist_sink`, wiring lgj `Mask`
> to storage, or citing "time travel" / "WAL" / "ACID" / "zero-copy" as a
> property of this stack.
>
> **Confidence:** HIGH on the exposure map (API surface enumerated by grep,
> then each hit READ — three loose-grep false positives were caught and are
> listed in §7). HIGH on the ceiling analysis (four crates.io dependency
> queries, reproducible). MEDIUM on the convergence seams (§4) — they are
> designs, each with a named probe; none is measured yet.
>
> **Operator rulings this plan rests on (2026-09-05):**
> - **R2** — everything wires into the SoA V3 substrate, no exceptions, except
>   CausalEdge64-adjacent as ALU legacy substrate. Recorded on EPIPHANIES as
>   `E-EVERYTHING-WIRES-TO-SOA-V3-CE64-IS-ALU-LEGACY-1`.
> - **R3** — no locks; pins only for lance / lancedb / arrow 58 / datafusion 54,
>   at the MAJOR ("we aim at latest compatibility, still open to 58.x").
>   Landed as the pin rule in `CLAUDE.md` via #1182.

---

## 0. Headline

**The arrow / datafusion ceiling is not ours and does not move with lance.**
lance 9, 10 and 11 all require `arrow ^58.0.0` / `datafusion ^54.0.0`; so do
lancedb 0.33.0, 0.37.1 and 0.38.0. arrow 59.x and datafusion 55 exist and are
unreachable from ANY published lance or lancedb. **58 / 54 is the latest
compatible pair, not a cautious one.** A lancedb bump therefore buys a newer
`lance` major and nothing else — which turns "should we bump" into a pure
lance-major question, and that question has a measurable answer (§3).

**This repo's lance surface is narrow, and neither bump breaks it.** The nine
things we call (§2) are untouched by all nine breaking changes across lance 10
and 11. The exposure that IS real is *semantic*, not API: lance 11's breaking
set clusters on **row identity** (fragment-id reuse, stable-row-id migration,
row-id high-water marks), which is the axis this substrate addresses rows by.

**Time travel, WAL, ACID and zero-copy are all REAL in this tree — and each
has one seam where lance 10/11 now offers natively what we built by hand.**
Those seams are the convergence candidates (§4). Each is a "one truth"
decision: adopt lance's, keep ours as a projection of it, or keep ours and
never enable lance's. Running both is the failure mode.

---

## 1. The pin rule, as landed (#1182)

**Exact-pin only where an upstream crate itself demands exact-equals;
everywhere else the major.** Measured, lancedb 0.33.0 requires:

| it requires | operator | consequence for us |
|---|---|---|
| `lance` + 12 siblings | `=9.0.0` | **forced** — not our choice; also why the published `lance 9.0.1` is unreachable |
| `arrow` ×9 | `^58.0.0` | ours was `=58.4.0` — strictly narrower than the family asks; relaxed to `58` |
| `datafusion` ×9 | `^54.0.0` | ours was `=54.1.0`; relaxed to `54` |

Resolution verified (`cargo generate-lockfile`, probe lock deleted): identical
versions today — arrow 58.4.0 / datafusion 54.1.0 / lance 9.0.0 / lancedb
0.33.0. Zero movement; 58.5.0 is taken automatically when it lands. The scar
that motivates the rule is already in-tree: `Cargo.toml:141-144` records
cargo unable to satisfy our `lance = "=6.0.1"` against lancedb's transitive
`=6.0.0`.

Two canon corrections rode along: `CLAUDE.md:36` and `:1200` cited
`Cargo.lock` as verification authority for a file the project rule removed
(`.gitignore:2-3`, commit `6190c803`, `ISS-STALE-AUTHORITY-LOCKS` RESOLVED).
An unfalsifiable citation that also invited a future session to re-create
the lock. **Direction of that drift is worth noting:** `CLAUDE.md:1201`
already said `arrow = "58"` — the DOC was right and the MANIFEST had drifted.

**Version-number trap, recorded because it caught a real person today:** the
Rust crate line is 0.33.0 → 0.37.1 → 0.38.0. **0.34 / 0.35 / 0.36 do not exist
as Rust crates** — those numbers belong to the independently-versioned PyPI
package. The lance9 probe plan §1 recorded the same trap for 0.36 last month.

## 2. Exposure map — what we call vs what broke

**The lance surface this repo actually calls** (grep over `crates/*/src`, each
hit read; local modules that merely *sound* like lance — `lance_cache`,
`lance_sink`, `lance_membrane` — are feature names / local files and are
excluded):

`Dataset::open` · `Dataset::write` · `InsertBuilder` (`WriteMode::{Create,Append}`
on the cycle sink — **⊘ 2026-09-05: `VersionedGraph::write_batch` uses
`WriteMode::Overwrite` for every commit after the first; the "never `Overwrite`"
claim that stood here was wrong, see §7.6**) · `checkout_version` · `version()` · `versions()`
· `tags().create` · `scan()` · `LanceTableProvider` · `lance_linalg::distance`
· `LanceNamespace` / `DescribeTableRequest` · `Error`

**Not used anywhere:** compaction, blob, cache backend, fragment API,
manifest config, index build / HNSW, **MemWAL**, **stable row ids**.

### lance 9 → 10 (lancedb 0.33 → 0.37.1) — 3 breaking

| breaking | our exposure |
|---|---|
| `perf(compaction)!` skip row-address maps when no index remap | none — no compaction calls |
| `perf(cache)!` fixed-size cache keys | none — the `lance-cache` here is a local feature/module |
| `fix(blob)!` preserve null selections across blob APIs | none — no blob APIs |

42 feat/perf. On-thesis: zone-map seeds in data-file footers; DataOverlay
commit + index masking; cross-store `deep_clone`; `merge_insert` accepting a
`TableProvider`; `perf(dataset): read transactions by version without
populating session caches`; `fix(mem_wal): read-your-writes via dual
visibility cursors` (§4C).

**Verdict: a cheap hop.** Zero measured exposure, 4 weeks old, 79 571
downloads.

### lance 10 → 11 (lancedb 0.37.1 → 0.38.0) — 6 breaking, clustered on ROW IDENTITY

| breaking | our exposure |
|---|---|
| `fix(core)!` stop reusing fragment ids across an **overwrite** | API: none. **⊘ 2026-09-05: exposure is REAL** — `VersionedGraph::write_batch` overwrites on every commit after the first (§7.6). Measured under lance 10 by `tests/lance_row_identity_probe.rs`: fragment id `0` is reused on BOTH overwrites and two `_rowaddr`s alias to a DIFFERENT `node_id` across consecutive versions. Semantic: we sit on **physical** row addresses (`fragment_id<<32 \| offset`) since stable row ids are off — any future overwrite path inherits the new semantics |
| `feat` `Dataset::migrate_to_stable_row_ids` | none until enabled; **this is the migration lance is steering everyone toward** |
| `fix(table)!` restore `migration_next_row_id` on `ManifestBuildConfig` | none |
| `refactor!` compose exact current-format readers | `Dataset::open` + `scan()` only — low, but it is the reader we use |
| `fix(dataset)!` conflict a NOT NULL tightening with concurrent writes | none — no schema tightening, sole-writer |
| `feat(compaction)!` max_source_rows / bytes | none |

79 feat/perf. Three unusually on-thesis (§4): `feat(scanner): add external
row-address mask prefilter`; `feat(index): zone map support for all data
types`; `feat: per value support for fixed-length packed structs`. Plus:
`feat: report the row ids deleted between two versions`; `feat(dataset):
lightweight dataset version references`; `feat(java): efficient dataset
version count`; `feat: support shared session for fragment API`; typed
`CommitConflictError` + `fix: bound commit conflict retry backoff` + `fix:
recognize own commit on conflict`; `fix: preserve the row id high-water mark
across restore`; `fix: delete data files when cleanup retains an older tag`.

**Verdict: real upside on exactly our axis, breaking changes on exactly our
axis, and the least proven** — 6 days old, 3 853 downloads (lancedb 0.38.0:
**210**), 22 betas + 2 RCs before GA.

---

## 3. The staged migration

| D-id | stage | gate | status |
|---|---|---|---|
| **D-LNC-0** | pin rule: `arrow`/`datafusion` to the major; canon `Cargo.lock` citations fixed | `cargo generate-lockfile` resolves to identical versions; probe lock deleted; CI green | **SHIPPED** — #1182 merged `a738e4ae` |
| **D-LNC-1** | lance 9 → 10 / lancedb 0.33 → 0.37.1 | (a) §2 exposure table re-verified against the 10.0.0 release body on the day of the bump; (b) workspace `cargo test` green; (c) `LanceCycleWriter` commit-cycle tests + `VersionedGraph` tests green; (d) lance9-probe §7's two "looks like Lance fallout and is not" failures re-checked first — **plus the third shape (§7.5): a stale path-dep sibling** | In PR — #1187 (`claude/lance-10-bump`, d1fc58c1); gate (d) fired for real on the sibling shape |
| **D-LNC-2** | **row-identity probe** on lance 11 (pre-registered, §5) | a real dataset written under 10, opened under 11: `checkout_version(v)` for every `v` in `versions()` returns byte-identical `(node_id, seal)` sets; the physical row address of every row is unchanged; a delete between two versions is reported by lance's new delta exactly as `VersionedGraph::diff` reports it | **SHIPPED** #1189 `afce6815` (⊘ 2026-09-05). Green under BOTH majors, all disable arms two-sided across the bump; cross-major fixture byte-identical (§5a). Delete-delta arm RE-SCOPED (§7.9) |
| **D-LNC-3** | lance 10 → 11 / lancedb 0.37.1 → 0.38.0 | D-LNC-2 GREEN **and** an adoption floor: lancedb 0.38.x ≥ 30 days old with a patch release or ≥ 5 000 downloads, whichever first. Not a technical gate — a "22 betas" signal | **SHIPPED** #1190 `8761eea1` (⊘ 2026-09-05). The operator waived the adoption floor and merged both stage PRs before their gates finished; the gates ran anyway and all four are green (§5a). |
| **D-LNC-4** | mask convergence probe: SoA row index ≡ Lance row address (§4A) | measured on a real mailbox dataset; the bitmap identity holds for a single-fragment append-only dataset AND is shown to break on the first compaction/delete — both halves, or the probe is vacuous | Queued — independent of the bump |
| **D-LNC-5** | replace `VersionedGraph::diff`'s two-version full materialization with lance's native `row ids deleted between two versions` (§4E) | zero-copy-warden verdict LENS-CLEAN; output byte-identical to the materializing path on the same versions | blocked on D-LNC-3 (the API is lance 11) |
| **D-LNC-6** | session reuse: stop `Dataset::open` per call (17 sites in `lance-graph` alone) (§4E) | same results; measured open-count strictly decreases | Queued — lance 10 already ships `shared session`; independent of D-LNC-3 |
| **D-LNC-7** | **one-WAL decision** (§4C): MemWAL vs `LanceCycleWriter` | a ruling, then a disable-run showing the losing WAL is provably unreachable | **Operator decision** |

**Sequencing that is not negotiable:** D-LNC-2 before D-LNC-3; D-LNC-4 before
any lgj wiring to a lance prefilter; D-LNC-7 before enabling MemWAL anywhere.
D-LNC-1, 4, 6 are mutually independent and can land in any order.

## 4. Hardening vs wiring-to-benefit — the five seams

Each seam is stated as: **what is real today (file:line) → what lance 10/11
offers → the one-truth question → HARDEN or WIRE.** "Both" is the failure
mode: two mechanisms for one property is two truths, and the second one is
the one nobody tests.

### 4A. Rows vs region mask — the convergence the operator asked about

**Real today.** lgj's `Mask` is `MaskWords { words: Box<[u64]> }`
(`native/lgj-abi/src/registry.rs:56-58`) — a **row bitmap over the envelope's
rows `0..N`**, the currency of `Mask × ClassView/WideFieldMask → Mask`. lgj
has **zero** `row_address` / `prefilter` vocabulary. The mailbox side is
`MailboxSoA<N>` with fixed-`N` column arrays (`mailbox_soa.rs:58-196`).

**What lance 11 offers.** `feat(scanner): add external row-address mask
prefilter` — a bitmap over **physical row addresses** (`fragment_id << 32 |
offset`) pushed into the scan. That is the same *shape* as lgj's `Mask`,
one layer down.

**The one-truth question.** The two bitmaps are the same object **iff the SoA
row index equals the Lance row address**. That holds for exactly one dataset
shape: single fragment, append-only, in row order, no deletes, no compaction.
Every one of those conditions is violated by something lance 10/11 ships or
we already do: `persist_sink.rs:12` seals one 64k-row cycle per version
(multi-fragment by construction), compaction is a first-class lance 10/11
feature, and deletes are how tombstones work (`soa-three-tier-model.md:43`).
**So the identity is a coincidence of the empty dataset, not an invariant.**

**Verdict: HARDEN before WIRE.** The convergence is real and worth having —
a mask computed in the envelope, evaluated by the scanner, no
materialization — but only through a **translation lens** `SoaRow → RowAddress`
that is itself a ClassView projection (zero-copy law: "the array itself is a
ClassView projection"). Never by asserting the two indices coincide.
**D-LNC-4 is the falsifier**, and it must have both halves: the identity
holds on the append-only fixture AND breaks on the first delete. A probe that
only shows it holding is vacuous (`E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`).

What this does NOT change: lgj's mask-native invariant ("WHERE MAY LOOK LIKE
WHERE. IT MUST EXECUTE LIKE MASK") is *strengthened* — the storage layer now
speaks mask too. What crosses the membrane is still a name (a mask handle),
never a byte position (`bbb-warden`). The row-address bitmap lives below the
T1/T2 line; lgj never sees an address.

### 4B. Time travel / versioning

**Real today — corrected mid-pass (§7).** `at_version` (`crates/lance-graph/src/graph/versioned.rs:499`, inside
`impl VersionedGraph`) calls `checkout_version`;
`:514` `versions()`; `:508` `version().version`; tags via `tags().create`
(`:525-535`); `diff(from, to)` at `:543-544`. The archetype `World::fork` /
`at_tick` (`world.rs:82-118`) are **deliberately lance-free** — they hand a
tick down to `VersionedGraph::at_version`. `hydrate/dirty.rs:41-42` reads
`version_id()` for the dirty check. `temporal.rs` builds the epistemic
projection (`QueryReference::at(v, rung)` + `deinterlace`) on top.

**What lance 10/11 offers.** lance 10: `perf(dataset): read transactions by
version without populating session caches`. lance 11: `lightweight dataset
version references`; `efficient dataset version count`; `report the row ids
deleted between two versions`; `fix: delete data files when cleanup retains
an older tag`; `fix(java): preserve sessions on dataset checkout`.

**Verdict: WIRE, two items, and one HARDEN.**
- WIRE (D-LNC-5): `diff()` today materializes BOTH versions via
  `read_all_batches` = `ds.scan().try_into_stream()` (`:645-651`) to compare
  node ids and seals. lance 11's version delta is the lens that replaces it.
  Note `diff()` currently has **zero callers** outside its file — so this is
  a public-surface correctness item, not a hot-path one.
- WIRE (D-LNC-6): every `VersionedGraph` method does a fresh `Dataset::open`
  (`:500, :506, :512, :523, :530, :541` …) — 17 sites in `lance-graph`. lance
  10's shared-session path and lance 11's "preserve sessions on checkout"
  make the fix cheap and measurable (open-count strictly decreases).
- HARDEN: **tags are load-bearing** (`tag("epoch-42")`) and lance 11 changed
  cleanup-vs-tag semantics. D-LNC-2 must include a tagged-version retention
  check after a cleanup, not just `checkout_version` equality.

### 4C. WAL

**Real today.** An in-house WAL: `WalSink` trait (`persist_sink.rs:534`)
whose production impl is **`LanceCycleWriter`** (`cycle_sink.rs:731`) — so the
WAL *is on Lance*; the other two impls are test fakes. Replay is
watermark-idempotent over `LandedSlot`s sorted by `(cycle, stream_position)`
(`persist_sink.rs:677-700`), with `applied_through` as the idempotence key.
One 64k-row cycle seals into exactly one `DatasetVersion` (`:12`), and "no
artifact-backed semantic change → no write → no new version" (`:75`).

**What lance 10/11 offers.** A full MemWAL: `read-your-writes via split
index-apply and dual visibility cursors` (10); `index catch-up positions …
withdrawn on index change`, `derive index catch-up from the version a commit
read`, `frozen memtable and backpressure stats`, `force_seal_active` returning
the sealed generation (11).

**The one-truth question — and it is the sharpest in the plan.** Our
`applied_through` watermark and MemWAL's `index catch-up position` are the
**same concept**. Enabling MemWAL under `LanceCycleWriter` would put two
write-ahead logs under one commit, each with its own notion of "landed".
Crash recovery would then have two answers.

**Verdict: HARDEN — D-LNC-7 is an operator ruling, not an engineering call.**
Three shapes are legitimate; "both" is not:
1. **Keep ours, never enable MemWAL** — zero change, MemWAL's features stay
   unreachable. Cheapest; forgoes read-your-writes.
2. **Adopt MemWAL, retire `LanceCycleWriter`'s replay** — our
   `stream_position` becomes a projection of lance's catch-up position.
   Largest change; the reconciliation-first design (`cycle_sink.rs:732-760`)
   would be re-derived from lance's `recognize own commit on conflict`.
3. **Keep ours as the semantic WAL, use MemWAL only as a buffer** —
   requires proving the two positions are monotone-equivalent. This is the
   "both" that looks like a compromise; it needs the strongest falsifier.
This plan recommends **1 until D-LNC-3 is green, then re-decide** — because
MemWAL's value proposition (read-your-writes) is precisely what §4D's
reconciliation read already pays for.

### 4D. ACID / commit conflict

**Real today.** `LanceCycleWriter::commit_cycle` (`cycle_sink.rs:734+`) is an
optimistic-concurrency commit with a **base-version fence**:
`batch.frame.base_version != head → CommitError::Fenced { current_head }`
(`:812-813`), plus `HashConflict` (`:804`), `Ambiguous` (`:759`), and a
`committed_through` watermark. Design is **append-first, zero reads on the
normal path; reconciliation read only on ambiguity** (`:778-786`). This is a
CAS over Lance versions implemented one layer above Lance. Sole-writer today
(`lance_membrane.rs:295` "action-commit sole-writer").

**What lance 11 offers.** `fix: raise typed CommitConflictError instead of
bare OSError`; `fix: bound commit conflict retry backoff`; `fix: recognize
own commit on conflict and avoid deleting committed artifacts`;
`fix(dataset)!: conflict a NOT NULL tightening with concurrent writes`; `fix:
conflict schema metadata updates with merges`.

**Verdict: WIRE the error type, HARDEN the fence.** `CommitError::Io(e)`
(`:821`) currently swallows whatever lance raised; after D-LNC-3 it can
match `CommitConflictError` and map it to `Fenced` *without* the
reconciliation read — lance now answers "did my own commit land" natively.
But the fence itself must stay ours: lance's retry-with-backoff is a
liveness mechanism, ours is a **semantic** one (the frame's `base_version` is
part of the cycle's identity). Letting lance retry underneath our fence would
commit a cycle against a head our frame never saw. **Disable-run for
D-LNC-3:** with lance's retry left on and our fence removed, a forced
concurrent write must produce a wrong-head commit; with the fence on, it
must produce `Fenced`.

### 4E. Zero-copy

**The law.** "Zero copy is a law without escape hatches; the array itself is
a ClassView projection" — with one carve-out, a value at a STRICTLY HIGHER
awareness rung (`zero-copy-warden`). Three-tier model: "every SoA envelope is
zero-copy from creation to Lance tombstone; Lance writes LE bytes from the
in-place backing store" (`soa-three-tier-model.md:11-14, :165`).

**Violations found, on our side, by this pass:**
- `VersionedGraph::diff` materializes two full versions (§4B → D-LNC-5).
- `read_all_batches` is a general full-scan-to-`Vec<RecordBatch>` helper
  (`versioned.rs:645`) — any new caller inherits the violation.

**What lance 10/11 offers that serves the law:** `per value support for
fixed-length packed structs` (11) — the 512-byte fixed row / 4+12 facet is
exactly a fixed-length packed struct; today it is carried as `Box<[u64]>`
planes. `zone map support for all data types` (11) — pruning before read is
the storage-side twin of the mask prefilter. `shared session` (10/11) — the
open-per-call pattern (§4B) re-reads manifests we already hold.

**Verdict: WIRE D-LNC-5 and D-LNC-6 now; PROBE packed structs later.** The
packed-struct path is a layout question (`v3-envelope-auditor` territory —
any change to how the 16-byte facet is stored is a `LAYOUT-GATED` change
needing the field-isolation matrix), not a bump question. It gets its own
plan when D-LNC-3 lands; it is named here so it is not forgotten.

## 5. Falsifiers — pre-registered, with their disable runs

| probe | can-it-fire | can-it-stay-silent | disable run |
|---|---|---|---|
| D-LNC-2 row identity (⊘ re-pinned 2026-09-05 against the probe as built) | (a) every `versions()` entry is byte-identical on `(node_id, seal)` and `_rowaddr` across a re-open; (b) a real `Dataset::delete` is `Staunen` in `graph_seal_check`; (c) fragment-id reuse across an overwrite is MEASURED — `LNC2_FRAGMENT_REUSE=expected` green under lance 10, `=forbidden` green under lance 11 | the same version is `Wisdom`; after `cleanup_old_versions` the tagged version survives byte-identical while an untagged old version is GONE (`at_version` → Err) | under lance 10 `=forbidden` → RED (verified 2026-09-05, `probe.rs:355`); under lance 11 `=expected` must go RED. The original "delta == `diff()`" arm is UNREACHABLE: lance 11's `get_deleted_row_ids` requires stable row ids at both endpoints (we are on physical addresses) and `GraphDiff` has no removed-nodes field (§7.7–7.9) |
| D-LNC-4 mask identity | append-only single-fragment fixture: `MaskWords` bitmap == row-address bitmap, word for word | — | delete one row, re-scan → the identity MUST break; if it does not, the fixture never left the coincidence regime and the probe is vacuous |
| D-LNC-5 native delta | output byte-identical to `read_all_batches` diff on ≥ 3 version pairs incl. one with deletes | — | feed the lens the wrong `to_version` → must differ |
| D-LNC-6 session reuse | open-count (instrumented) strictly decreases on the `VersionedGraph` test suite | results byte-identical | remove the shared session → count returns to baseline |
| D-LNC-3 fence | forced concurrent write with our fence ON → `Fenced`; lance retry left enabled must NOT produce a commit | sole-writer path unchanged | fence OFF, lance retry ON → a wrong-head commit must be observable |
| tag retention (D-LNC-2 sub-arm) | after a cleanup that retains a tagged older version, `checkout_version(tagged)` still opens | — | drop the tag first → cleanup must remove it |

Every threshold above is a `==` or a strict inequality, never `>=` on a
count, per the falsifiability rule ("prefer `== N` over `>= N`").

---

## 5a. Measured — the probe's results under both majors (2026-09-05)

Run at `CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0`, lance 10.0.0 then
lance 11.0.0, same probe binary, same fixture.

| observation | lance 10.0.0 | lance 11.0.0 |
|---|---|---|
| fragment ids across the two overwrites | `{0} → {0} → {0}` | `{0} → {1} → {2}` |
| ids shared between consecutive versions | 2 | **0** |
| `_rowaddr` aliased to a DIFFERENT `node_id` | 2 | **0** |
| `LNC2_FRAGMENT_REUSE=expected` | green | **RED** (`probe.rs:372`) |
| `LNC2_FRAGMENT_REUSE=forbidden` | RED (`probe.rs:355`) | **green** |
| `cleanup_old_versions(TimeDelta::zero())` | 2 old versions removed, tagged survives byte-identical, untagged gone | identical |
| workspace `cargo test --no-fail-fast` | exit 0 | **exit 0**, 2317 passed, 0 failed, 68 suites |

Both policy arms are two-sided ACROSS the bump, which is the strongest form
this falsifier can take: the arm that must fail on one major is the arm that
must pass on the other, so neither result can be a vacuous pass.

**The finding that is not in the release notes: lance#8206 is not
retroactive.** The cross-major arm opened the lance-10-written fixture under
lance 11 and verified it byte-identical against `reference.tsv` (16 lines,
four versions) — and that same read still reports **2 ids reused and 2
`_rowaddr`s aliased**, because fragment ids are written into the manifest by
the writer and are simply read back. So:

- the bump stops NEW overwrites from minting a colliding address space;
- it repairs nothing in data that already has one;
- **§4A's verdict is unchanged by the bump.** "A row-address-keyed reader is
  sound only WITHIN one Lance version" holds under both majors, and the
  `SoaRow → RowAddress` lens must stay scoped per `DatasetVersion` for
  pre-existing datasets no matter which major reads them.

Migrating an existing dataset out of that state is `migrate_to_stable_row_ids`
(lance 11, one `Merge` commit, quiesced writers), which is D-LNC-5's decision,
not this stage's.

## 6. Relationship to prior plans

- **`lance9-datafusion54-upgrade-probe-v1`** — the last bump. Its §1 recorded
  the 0.36-does-not-exist trap this plan re-hits at 0.34; its §7 names two
  failures that look like Lance fallout and are not (D-LNC-1 gate (d)); its §9
  "still open" items are inherited unchanged — the unchecked crates
  (`-benches`, `symbiont`, `cognitive-stack`, `surreal_container`,
  `-python`) are still unchecked, and this plan does not claim them.
- **`bindspace-mailbox-soa-wiring-v1`** (D-BSW-0..4) — orthogonal and
  unblocked by this plan. One interaction: **ruling R2 retires D-BSW-2's
  `dispatch_busdto` exclusion.** The standing ruling on that symbol is
  `.claude/v3/COMPONENT-MAP.md` §6 — `engine_bridge.rs::dispatch_busdto` +
  `persist_cycle`: *BLOCKED→W4a, "correctly grandfathered until the batch
  writer (W1) exists"* — and R2 supersedes the grandfathering, not the
  BLOCKED verdict: it still lands behind the batch writer, it just no longer
  has an exit door. `MailboxSoA` carries an f32 *scalar* tenant
  (`energy: [f32; N]`, `mailbox_soa.rs:66`) but no 16-dim f32 *vector*
  tenant; the wiring target for `qualia_f32` is the i4 register — `atoms.rs`
  `I4x32/I4x64` ("byte-compatible with QualiaI4_16D and CausalEdge64
  mantissa") or `QualiaI4_16D` directly — gated on a quantization-equivalence
  probe using the D-MTS-6 proxies (`|ΔE|`, surprise agreement, descent ρ).
  That probe belongs to the BSW plan; it is named here so R2's consequence is
  not lost.
- **#1182** (D-LNC-0) — the pin rule this plan stands on.
- **`ISS-STALE-AUTHORITY-LOCKS`** — the no-lock rule; this plan's resolution
  probes generate and delete a lock, never track one.

---

## 7. Corrections made during this pass (kept, not overwritten)

The falsifiability rule applies to the author too. Four claims were wrong
when first written and were corrected against the tree before landing; a
fifth false-fallout shape was found while executing D-LNC-1, and four more
were found by BUILDING the D-LNC-2 probe against the tree (6–9):

1. **"Time travel is claimed, not wired."** WRONG. My grep was for
   `lance::checkout`; the real call is `checkout_version`
   (`versioned.rs:501, 543, 544`). Only the archetype `World` layer is
   lance-free by design. Corrected in §4B.
2. **"No f32 tenant exists on `MailboxSoA`."** IMPRECISE. `energy: [f32; N]`
   exists (`:66`); what is missing is a 16-dim f32 *vector* tenant. My
   earlier field grep matched `Box<[..]>`/`Vec<..>` and missed `[T; N]`
   arrays. Corrected in §6.
3. **"Nothing checks that `DatasetVersion` is 1:1 with the Lance version."**
   WRONG for the production path: `cycle_sink.rs:437` and
   `graph/scheduler.rs:168` read it FROM `ds.version().version`. It is only
   the in-process `LanceVersionWatcher` (`version_watcher.rs:57`) that is a
   `watch` channel rather than a Lance observer. Corrected in §4C.
4. **Three loose-grep false positives**, each caught by reading the hit:
   `lance-cache`/`lance-sink` matched `[features]` lines (they are local
   modules); "compaction" matched our own `compact_bytes` /
   `compact_byte_size`; and an earlier `^name = "(lance` without a closing
   quote matched a path crate in a lock. The rule that survives all three:
   **grep locates; read comprehends.** A count is not a finding.
5. **A THIRD "looks like Lance fallout and is not" shape (2026-09-05,
   D-LNC-1 execution).** The first local `cargo test --workspace` after the
   10.0.0 bump failed with `error[E0432]: unresolved import
   ndarray::simd::ternlog / mask_ternlog` in `lance-graph-planner`
   (`nested_bands.rs:31-32`). Not the bump: the local `ndarray` path-dep
   sibling sat 49 commits behind `origin/master` (the branch is `master`,
   not `main`) and predated ndarray commit `1493a10`, which added both
   symbols. Fast-forwarding the sibling cleared it. lance9-probe §7 named
   two shapes (missing `protoc`; `--all-targets` example gating); this is
   the third, and it is the one CI cannot produce — CI checks out the
   sibling's default branch fresh. **Rule: before attributing a compile
   error to a lance bump, `git -C ../ndarray status -sb` and count
   `behind`.** Every sibling path-dep is a hidden version pin that the
   no-lock policy does not cover.
6. **"We never `Overwrite`" (§2, twice).** WRONG. `VersionedGraph::write_batch`
   (`crates/lance-graph/src/graph/versioned.rs`, the `WriteMode` match right
   after `Dataset::open`) picks `Overwrite` whenever the dataset already
   exists — every `commit_encounter_round` after the first REPLACES the whole
   node/edge/fingerprint tables, while its own doc comment says "appended".
   The `Append`-only claim was true of `LanceCycleWriter`, not of the graph.
   Consequence: lance#8206 (fragment ids across an overwrite) is a change we
   are exposed to, and the probe measured what it protects against: under
   lance 10, fragment id `0` is reused on every overwrite, so two
   `_rowaddr`s (`0<<32|1`, `0<<32|2`) name node 2/3 at one version and node
   3/4 at the next. **A row-address-keyed cache is only sound WITHIN one
   version** — this is the §4A one-truth verdict, now measured, and it holds
   under 11 too (fresh ids do not make old addresses comparable, they only
   stop them colliding).
7. **`GraphDiff` cannot see a removal.** It has `new_nodes` / `modified_nodes`
   / `new_edges` and nothing else; B→C (node 2 dropped by the overwrite) is an
   EMPTY diff. Only `graph_seal_check` reports it (`Staunen`). Pinned in the
   probe so that a `removed_nodes` field forces a deliberate re-pin.
8. **`diff()` assumes nodes/edges/fingerprints share version numbers.** It
   checks out the EDGES dataset at the NODES version number. A direct
   `Dataset::delete` on nodes (the probe's D) advances nodes to a version
   edges never had, and `diff(C, D)` is an `Err(DatasetNotFound
   …/edges.lance/_versions/4.manifest)`, not a diff. The lockstep holds only
   while every write goes through `commit_encounter_round`. Pinned as
   measured; `TECH_DEBT` `TD-VERSIONED-GRAPH-DIFF-LOCKSTEP-AND-NO-REMOVALS-1`.
9. **The pre-registered delete-delta arm was unreachable as written.**
   lance 11's `DatasetDelta::get_deleted_row_ids` (lance#8589) documents
   *"Requires stable row ids at both endpoints"*; our datasets are written
   with `enable_stable_row_ids: false` (the default), on physical row
   addresses. So "lance's delta and `diff()` report the delete identically"
   could never have run — and per 7, `diff()` reports NO delete anyway. The
   probe replaces it with `graph_seal_check` as the one truth for removals,
   and D-LNC-5 (native delta) is where stable row ids are decided, not here.
   Note the corollary for the alpha overlay: an EPHEMERAL dataset can turn
   stable row ids on at creation for free (no migration), which is why the
   delta API fits the alpha layer before it fits the graph spine.
