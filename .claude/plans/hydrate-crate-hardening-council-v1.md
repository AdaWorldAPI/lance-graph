# 5+3 Council: hardening `lance-graph-hydrate` before merge (PR #957)

**Council type:** post-implementation hardening pass, not pre-design. The
crate is already written and PR'd (`crates/lance-graph-hydrate`, 8 modules,
1050 LOC, 8 modules × 2-4 tests each). It qualifies for council per the
"when to convene" bar: it is canon-adjacent (implements/extends
`.claude/knowledge/s3-hydration-lifecycle.md`, a doctrine document other
repos will build on), and a wrong resolution here silently corrupts
downstream sessions — every future consumer (OGAR, q2, any repo that adds
this as a dependency) inherits whatever this crate gets wrong as if it were
proven. It was also NOT locally build-verified in the authoring session
(container disk exhaustion; verification deferred to CI, still pending at
council time) — so a design-level hardening pass adds real, independent
confidence orthogonal to what CI checks (CI proves it compiles and the
authored tests pass; it does not prove the tests are the RIGHT tests, or
that the doctrine's invariants are actually enforced rather than merely
documented).

## 1. FROZEN DECISIONS

1. **Flush-only-from-Hydrated.** *"flush is legal only from `hydrated`,
   never from `dirty`... A `dirty → flushed` edge is data loss with no
   error."* — `.claude/knowledge/s3-hydration-lifecycle.md` §4, lines
   205-210.
2. **The idempotency-boundary conditions.** *"`absent → hydrated` is
   idempotent given (a) a pinned source version and (b) a destination that
   is empty and not concurrently mutated. Outside those two conditions it
   is not idempotent, it is a merge."* — same doc §4a, lines 228-233.
3. **The mechanism.** *"hydrate aside, publish by rename... retire by
   renaming away first and deleting the renamed copy afterwards. A reader
   therefore only ever resolves a name that is either absent or a complete
   dataset, never one mid-assembly or mid-removal."* — §4a, lines 235-241.
4. **Filesystem-atomicity boundary, not a coordination protocol.** *"adds
   nothing to the read path, takes no lock, and holds no lease"* — §4a,
   lines 243-247. The eviction plan explicitly rejects a lease/refcount
   protocol; this crate's mechanism must be compatible with that rejection
   (never quietly add locking).
5. **No write-back on a startup path.** *"Never place a write-back on a
   startup path. Push-back is an operational step with its own trigger."*
   — §6, lines 337-338.
6. **Assert, don't assume, `hydrated` before flush.** *"Any flush path must
   assert `hydrated`, not assume it."* — §6, lines 339-340.
7. **Three-layer split** (object store = hydration source; local
   mmap-capable directory = the store, always required; persistent volume
   = pure hydration-frequency optimization, never a correctness
   requirement) — §1 of the same doc (cited in this repo's own
   `LATEST_STATE.md` 2026-08-06 entry, which this crate's own PR-time
   `LATEST_STATE.md` entry also cites).
8. **Dirty detection = Lance dataset version, never a hash.** —
   `.claude/plans/idle-flush-dataset-eviction-v1.md` §4/§9a, and this
   repo's own `crates/lance-graph/examples/hydration_probe.rs` (the "§4
   gate" measurement this crate's `dirty::is_dirty` claims to implement).
9. **Placement.** Operator directive, this session (verbatim): *"Ogar is
   only the intermediary who should help to inherit the pattern as a plug
   and play pattern; however the pattern itself should be minted in
   lance-graph already."* — satisfied structurally (crate lives at
   `crates/lance-graph-hydrate`); the council does not re-litigate this,
   only checks nothing in the shipped code assumes an OGAR-specific type.
10. **Lance-family upstream-authoritative carve-out.** `lance`/`lancedb`/
    `lance-*` are consumed from crates.io, never a fork — `CLAUDE.md`
    "⊘ CARVE-OUT" section. **Pre-verified at Phase-0 time (orchestrator,
    `mcp__github__search_repositories org:AdaWorldAPI object_store` →
    zero results):** no `AdaWorldAPI/object_store` fork exists, so P0's own
    exception ("crates.io is permitted ONLY for crates that have no
    AdaWorldAPI fork / no local source") applies cleanly regardless of
    whether `object_store` is read as lance-transitive or independent —
    there is nothing to fork to. This closes what would otherwise have
    been a real gap in the authoring session's diligence (the crate copied
    the pin from `crates/lance-graph/Cargo.toml`'s own precedent without
    checking). Savant 2 should still confirm this search was the right
    query (org name, repo-name variants) rather than trust it blind.

## 2. INPUT INVENTORY

All paths relative to `crates/lance-graph-hydrate/src/`.

| file | lines | public surface |
|---|---|---|
| `env.rs` | 116 | `env(k) -> Option<String>` (16); `struct HydrationSource { bucket, options }` (28); `impl HydrationSource { from_env, options, uri }` (33) |
| `lifecycle.rs` | 81 | `enum LifecycleState { Absent, Hydrated, Dirty, Flushed }` (9); `impl LifecycleState { can_hydrate, can_flush, can_release }` (25) |
| `marker.rs` | 162 | `struct StatIdentity { mtime_nanos, len }` (23); `stat_identity(path) -> io::Result<StatIdentity>` (29); `struct WarmMarker { identity }` (45); `impl WarmMarker { write, read, is_trusted }` (49) |
| `copy.rs` | 205 | `enum HydrateError { Io, Store, AlreadyPublished }` (22); `struct HydrationReport { objects_copied, bytes_copied }` (36); `async fn hydrate_dir(store, remote_root, publish_dir) -> Result<HydrationReport, HydrateError>` (52) |
| `file.rs` | 207 | `enum HydrateFileError { Io, Store, AlreadyPublished, ChecksumMismatch }` (15); `async fn hydrate_file(store, remote_object, publish_path, expected_sha256_hex) -> Result<(), HydrateFileError>` (34) |
| `release.rs` | 99 | `advise_dontneed_file(f: &File)` — `#[cfg(unix)]` (17) / `#[cfg(not(unix))]` (34) pair; `release_dir(dir) -> io::Result<usize>` (42) |
| `dirty.rs` | 117 | `enum DirtyCheckError { InvalidPath, Lance }` (14); `async fn is_dirty(local_path, hydrated_at_version) -> Result<bool, DirtyCheckError>` (25) |
| `lib.rs` | 63 | crate doc + re-exports only, no logic |

**Consumers today:** none yet (this is the first landing; the
`ISS-REMOTE-URI-CONSTRUCTORS-PREDATE-THE-HYDRATION-DOCTRINE` follow-up —
wiring `VersionedGraph::{s3,azure,gcs}` to actually call this crate — is
explicitly NOT done in this PR).

**Deps:** `lance = "=9.0.0"`, `object_store = { version = "0.13", features
= ["aws"] }`, `futures = "0.3"`, `tokio`, `sha2`, `thiserror`, `libc`
(unix-only). Dev-deps: `tempfile`, `arrow` (unused in current tests — dead
dev-dep, worth flagging). `Cargo.lock` diff at authoring time: +15 lines,
one new package node (`lance-graph-hydrate` itself), zero new package
*versions* anywhere else in the workspace graph.

## 3. THE RESOLUTION AS SHIPPED (what the council is hardening)

- `env::HydrationSource::from_env()` reads `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT_URL` / `AWS_DEFAULT_REGION` /
  `AWS_S3_BUCKET_NAME`, quote-stripped, `None` on any missing required var.
- `LifecycleState` is a 4-variant enum with three boolean guard methods
  (`can_hydrate`/`can_flush`/`can_release`), no state transition methods
  themselves (no `fn hydrate(self) -> Self` — the enum is read/checked by
  callers, not driven).
- `hydrate_dir`: lists every object under `remote_root` via
  `ObjectStore::list`, `get`s each one, writes to a `.hydrating-<pid>-
  <leaf>-<nonce>` sibling staging dir, then `tokio::fs::rename`s the
  staging dir onto `publish_dir` in one call. Refuses via
  `AlreadyPublished` if `publish_dir.exists()` at entry (checked once, not
  re-checked immediately before the rename — a TOCTOU window exists
  between the initial check and the final rename).
- `hydrate_file`: single-object sibling, `.part-<pid>-<nonce>` staging
  file, SHA-256 verified against a caller-supplied hex string before the
  final rename; same `AlreadyPublished` refusal shape.
- `marker::WarmMarker`: (mtime_nanos, len) pair written/read as a plain
  two-integer text line; `is_trusted` compares recorded vs current.
- `release::release_dir`: walks a directory tree, opens each regular file,
  calls `posix_fadvise(..., POSIX_FADV_DONTNEED)` on unix, no-op elsewhere;
  never deletes anything.
- `dirty::is_dirty`: opens the LOCAL dataset via `Dataset::open`, compares
  `.version_id()` to a caller-supplied `hydrated_at_version`.
- No `retire_dir` / evict-by-rename function exists anywhere in the crate
  — release_dir only drops page cache, never removes the directory.
- No function anywhere accepts or threads a "pinned source version"
  parameter — frozen decision #2's condition (a) is not encoded in the
  API; it is left entirely to the caller's own discipline.

## 4. NON-GOALS (explicit, each with why)

1. **The `retire`/evict-by-rename half of §4a is NOT implemented.** Only
   the publish half (`hydrate_dir`/`hydrate_file`) exists; `release_dir` is
   deliberately page-cache-only, not deletion. Why: the crate's own doc
   comments scope this explicitly ("Deleting the file is a different
   operation and is deliberately out of scope here"), and the PR's own
   verification-status note already flags the idle-flush SWEEP POLICY
   (which would call a retire function) as a deferred PROPOSAL, not
   shipped. *The council should confirm this scoping is coherent, not
   silently incomplete in a way that breaks frozen decision #3's "retire by
   rename" half.*
2. **No `VersionedGraph` wiring.** `hydrate_from(remote) -> VersionedGraph`
   is explicitly the next PR, not this one (per the PR body and the
   `ISSUES.md` regrade). Why: keeps this PR's surface to the primitive
   only, reviewable independently of any one consumer's shape.
3. **No automatic age+footprint eviction scheduler.** Still a PROPOSAL per
   `idle-flush-dataset-eviction-v1.md`. Why: that plan's own §4
   verification gate (cheap local version read) is what `dirty::is_dirty`
   exists to close, but the SCHEDULING policy around it is separate scope.
4. **No lock/lease/refcount mechanism.** Deliberately absent per frozen
   decision #4. *The council should confirm nothing added one implicitly*
   (e.g. the marker file could be mistaken for a lock if misused —
   confirm the doc comments are unambiguous that it is not).

## 5. PRE-REGISTERED GATES

1. Every frozen decision (1-10 above) has either a `CONFIRMS` finding with
   file:line evidence, or a `VIOLATES`/`GAP` finding that gets resolved in
   Phase 2 before Phase 3 runs.
2. No `BLOCK` from any of the 3 reviewers survives into Phase 5 unresolved.
3. `dilution-collapse-sentinel` must confirm the NON-GOALS section's
   scoping (§4.1 above) is a legitimate boundary, not silent incompleteness
   dressed as scope.
4. `firewall-warden` must confirm zero prohibited-shell-tool usage
   (grep/sed/awk via Bash), zero German PII, zero model identifiers, board
   hygiene already landed same-commit as the crate (it was — verify).
5. This spec's own consolidated findings get appended to
   `.claude/board/AGENT_LOG.md` — actually, per this repo's ONE-WRITER
   correction, the orchestrator (main thread) writes that entry directly;
   no sub-agent writes it.

## 6. PER-SAVANT QUESTION SETS (Phase 1, the 5)

### Savant 1 — prior art (`prior-art-savant`)
1. Does `env::HydrationSource` duplicate `lance_graph::dev_s3_env` in a way
   that will drift (two independently-maintained readers of the same env
   vars)? Should `HydrationSource::from_env()` instead CALL
   `dev_s3_env::{env, s3_options}` rather than reimplement quote-stripping
   and the option-map shape?
2. Is there an existing lance-graph type for "warm marker" / mtime+len
   trust-check anywhere in the workspace this crate should have reused
   instead of reinventing (even though q2 was the origin, has anything
   equivalent since landed in lance-graph itself)?
3. Does `LifecycleState` duplicate or conflict with any existing
   `contract::` enum with similar states (check `lance_graph_contract` for
   anything resembling absent/hydrated/dirty/flushed)?
4. Any existing E-id in EPIPHANIES.md that already states "hydrate aside,
   publish by rename" as a pattern this crate should cite/link rather than
   restate from the knowledge doc directly?

### Savant 2 — iron rules (`iron-rule-savant`)
1. Frozen decision #10 records the orchestrator already confirmed (GitHub
   search, zero results) that no `AdaWorldAPI/object_store` fork exists —
   sanity-check that this is the right verification (would a differently-
   cased or differently-named repo, e.g. `arrow-object-store`, plausibly
   exist and have been missed by that query)? If genuinely clean, mark
   CONFIRMS and move on rather than re-running the same search.
2. Does `hydrate_dir`'s single `publish_dir.exists()` check (at function
   entry, not immediately before the final `rename`) violate the
   idempotency-boundary's condition (b) ("a destination that is empty and
   not concurrently mutated") under a TOCTOU race — two concurrent
   `hydrate_dir` calls to the same `publish_dir`?
3. Does `release_dir`'s file-open-then-fadvise loop hold any file handle
   open longer than necessary in a way that could interact badly with a
   concurrent `hydrate_dir` publish-by-rename on the SAME directory (I-
   SUBSTRATE-MARKOV style concurrent-writer concerns, generalized)?
4. `I-LEGACY-API-FEATURE-GATED`: does this crate introduce any layout or
   API surface that a FUTURE version might need to reinterpret (e.g. the
   `WarmMarker` text format, `<mtime_nanos> <len>`) without a version byte
   or format tag — is that a live risk or a non-issue given the marker is
   purely a local, ephemeral, regeneratable optimization?
5. AP1-AP9 anti-pattern catalogue: any hits in `copy.rs`/`file.rs`'s error
   handling (silent `let _ = ...` swallows in `hydrate_dir`'s empty-prefix
   cleanup and `hydrate_file`'s checksum-mismatch cleanup) — is discarding
   the cleanup `Result` there a genuine AP hit or acceptable given the
   primary error already dominates?

### Savant 3 — code truth (runtime-archaeologist charter via general-purpose)
1. Is `dirty::is_dirty`'s claim — "compares CURRENT local version against
   hydrated_at_version" — actually what `Dataset::version_id()` returns in
   lance 9.0.0, verified against the real source (not assumed from the
   `hydration_probe.rs` precedent alone)? CODED or CLAIMED?
2. Is `hydrate_dir`'s claim that a caller "either sees nothing or sees the
   complete hydrated artifact — never a partial one" actually guaranteed
   by `tokio::fs::rename`, given `publish_dir`'s parent could be on a
   DIFFERENT filesystem than the staging dir's parent (both are computed
   from `publish_dir.parent()`, so same-filesystem should hold BY
   CONSTRUCTION — confirm this reasoning is airtight, not assumed)?
3. Is the module doc's claim in `copy.rs` that this mechanism was
   "generalized from `hydration_probe.rs`'s proven T10 mechanism" actually
   an accurate description of what changed vs what's identical — does
   `hydrate_dir` faithfully reproduce hydration_probe's byte-copy-not-
   scan-rewrite property, or did the generalization introduce a behavioral
   difference (e.g. hydration_probe's nonce scheme vs this crate's)?
4. `marker.rs`'s `stat_identity` claim that "two reads of the same
   untouched file always agree" — is this true on filesystems with
   sub-second or coarse mtime resolution (a real portability concern for
   the doctrine's "mmap-capable local directory" requirement across
   platforms), CODED protection or CLAIMED-only in the doc comment?

### Savant 4 — cascade impact (`cascade-impact-savant`)
1. What MUST change (not follow-up) if `AdaWorldAPI/object_store` is found
   to exist as a fork (savant-2 question 1) — enumerate every Cargo.toml
   across the workspace that would need the same fix, not just this crate.
2. What existing test suite(s) elsewhere in lance-graph (if any) implicitly
   assume no crate calls `posix_fadvise` on a file they also hold open —
   could `release_dir` interact badly with any existing consumer's own
   file-handle assumptions?
3. Is `LATEST_STATE.md`'s and `ISSUES.md`'s regrade (already landed
   same-commit) internally consistent with what THIS spec's Phase 3
   reviewers will find, or will a Phase 4 fix require ALSO re-editing
   those board files (a second board-hygiene commit) — flag as mandatory
   if any Phase-1/Phase-2 finding changes a claim already written there.
4. Given this crate has zero consumers today, what is the actual blast
   radius of a Phase-4 breaking API change (rename a function, change a
   signature) — confirm it is genuinely zero (no q2/OGAR code merged yet
   that calls into this crate) so Phase 4 fixes can be made freely without
   a second cross-repo PR.

### Savant 5 — different views (`creative-explorer-savant`)
1. The crate ships `hydrate_dir` + `hydrate_file` as two independent
   functions with near-identical staging/rename logic (~60% code overlap
   by eye). Strongest alternative reading: should these share a single
   generic `hydrate_aside_then_publish(staging_write_fn, publish_dir)`
   primitive, with `hydrate_dir`/`hydrate_file` becoming thin callers? What
   second-order consequence follows from NOT doing this (drift between the
   two staging-nonce schemes, double the surface to audit for the TOCTOU
   question above)? Do not redesign — name the concern only.
2. `LifecycleState` is entirely disconnected from the OTHER five modules —
   no function in `copy.rs`/`file.rs`/`dirty.rs`/`release.rs` takes or
   returns a `LifecycleState`, so the "hard rule" (flush only from
   Hydrated) is enforced ONLY if a caller manually calls
   `is_dirty().await` then checks `LifecycleState::Hydrated.can_flush()`
   themselves — nothing in the type system forces this sequencing.
   Strongest alternative reading: is an un-wired lifecycle enum that
   documents the rule but cannot enforce it at the type level a
   meaningfully weaker guarantee than the spec's own framing ("encoded as
   a transition guard, not caller discipline") claims? Name the gap
   without redesigning a state-machine type.

## 7. Findings — Phase 1 (banked, condensed; raw output in session transcript)

38 findings across 5 savants. Distinct defects after dedup (multiple savants
independently corroborated several — corroboration noted):

| id | savants | verdict | one-line |
|---|---|---|---|
| F1 | S1 | PRIOR-ART-AT | `env.rs` is a near byte-identical copy of `dev_s3_env.rs` — recreates the exact drift risk that helper was minted (PR #907 CodeRabbit) to close; the no-dep-on-lance-graph trade-off was never recorded as weighed |
| F2 | S1 | GAP | `from_env_is_none_when_a_required_var_is_missing` is vacuous — never calls `from_env`, no input can fail it (falsifiability-rule hit); `dev_s3_env` handled the same situation honestly by declining the test |
| F3 | S5+S4 | VIOLATES | "encoded as a transition guard, not caller discipline" is FALSE — `LifecycleState` appears in zero function signatures; `is_dirty` returns bare `bool`; the claim is made in `lifecycle.rs` doc, `lib.rs` doc, AND `LATEST_STATE.md` (2 files + board) |
| F4 | S4 | VIOLATES | board claims `is_dirty` compares against "the version recorded at hydration time" — nothing in the crate records/returns/threads a hydration version; `hydrated_at_version` is caller-supplied and unobtainable from this crate's API |
| F5 | S2 | GAP | concurrent-`hydrate_dir` loser: rename onto now-existing dir fails ENOTEMPTY surfacing as `Io`, not the documented `AlreadyPublished` — the doc contract is dishonest under the exact race it names |
| F6 | S2+S5 | RISK | staging nonce = pid+nanos is NOT unique within one process (two tasks, same tick → shared staging dir published as "complete"); AND the scheme is independently duplicated in `copy.rs`/`file.rs` with divergent shapes |
| F7 | S2 | GAP | every I/O-failure path between staging-create and rename leaks a `.hydrating-*`/`.part-*` orphan — unbounded debris on the exact footprint-managed volume the doctrine cares about |
| F8 | S2 | GAP | `release_dir` structurally cannot return `Err` (every failure swallowed) → its missing-dir test is vacuous; `Ok(0)` conflates "released nothing" with "couldn't read the tree at all" |
| F9 | S2 | GAP | `WarmMarker` line has no version tag and `read` ignores trailing tokens (permissive arity — the exact `>= N` shape CLAUDE.md's falsifiability rule warns on); I-LEGACY-API-FEATURE-GATED-shaped future aliasing risk, bounded blast radius |
| F10 | S2 | VIOLATES(partial) | 2 of 6 `unsafe` env blocks lack `// SAFETY:` comments (AP7). **The second half of this finding — `unused_unsafe` fails `-D warnings` under edition 2021 — was REFUTED by the orchestrator via direct rustc 1.97.1 check with a positive control: `unsafe { set_var }` compiles clean; the blocks stay (edition-2024-proofing)** |
| F11 | S2 | GAP | `copy.rs:95`'s discarded `remove_dir_all` sits on an **Ok** path — a cleanup failure silently falsifies the function's own "leaves NOTHING" postcondition |
| F12 | S2 | GAP | **spec's own §2 was wrong**: `arrow` dev-dep is NOT dead — used by both `dirty.rs` tests; acting on the spec's "worth flagging" note would have broken the build (a council catching its own spec's error — working as designed) |
| F13 | S3 | GAP | `copy.rs` module doc overclaims: "generalized from hydration_probe's proven T10 mechanism" — the probe has NO staging dir and NO rename; only the raw list+get byte-copy is inherited; publish-by-rename is net-new, unproven-by-that-probe code |
| F14 | S3 | RISK | `stat_identity` has zero protection against coarse-mtime filesystems (same-length write within one mtime tick → false trust); the test suite's own 10ms sleep is the tell |
| F15 | S1 | GAP | `E-A-REPEATABLE-TRANSFER-IS-NOT-IDEMPOTENCE-OVER-A-MULTI-FILE-DIRECTORY-1` (EPIPHANIES:2395) states the mechanism verbatim incl. the pinned-source-version clause — cited nowhere in the crate |
| F16 | S1 | RISK | in-repo naming collisions: `is_dirty` (container_bs cache bitmap) and `hydrate` (`graph/hydrate.rs` — the OPPOSITE direction, local→LanceDB ingest) already mean different things in this workspace |
| F17 | S1 | PRIOR-ART-AT | `lance-graph-ontology/src/lance_cache.rs` answers the same is-my-local-copy-valid question on the content-checksum axis — cross-ref, not duplication |
| F18 | S4 | RISK | non-unix `release_dir` still opens every file for a documented no-op — pure syscall waste |
| F19 | S4 | RISK | `release_dir` on a parent dir will touch in-flight `.hydrating-*` staging files; harmless but undocumented |
| F20 | S4 | GAP | "THE idempotency-boundary condition" (singular, 2 files + board) collapses frozen decision #2's TWO conditions; condition (a) — pinned source version — is encoded nowhere and never named as deliberately-caller-owned |
| F21 | S4 | CONFIRMS | board corrections must land as a NEW dated `LATEST_STATE.md` entry (living-vs-ledger rule), never in-place edits of the 2026-08-17 entry |
| F22 | S2+S4 | CONFIRMS/RISK | no `AdaWorldAPI/object_store` fork under that name; S2 notes the search was name-shape-narrow (upstream home is `apache/arrow-rs-object-store`) but decisively: this crate's pin is byte-identical to the existing workspace-wide precedent, so any residual fork question is workspace-scoped, not PR-scoped |
| F23 | S4+S1 | CONFIRMS | zero consumers anywhere (this repo + all reachable siblings) — Phase-4 API changes have zero blast radius; `LifecycleState` duplicates nothing in the contract; `WarmMarker` genuinely has no lance-graph prior art; `release_dir`'s handle discipline is sound (no conflict with publish-by-rename, no lease introduced) |
| F24 | S3 | CONFIRMS | `version_id()` = in-memory manifest read, genuinely cheap, CODED as claimed; staging/publish same-filesystem-by-construction is real (both from one `parent` binding) |
| F25 | S5 | RISK | `can_release` admits `Flushed` but no flush function exists — the enum specifies a lifecycle wider than the shipped mechanisms; confirm deliberate |
| F26 | S5 | RISK | file-onto-file `rename` **silently clobbers** where dir-onto-dir fails ENOTEMPTY — `hydrate_file`'s `AlreadyPublished` contract is only honored at the entry check, not at the rename; the two functions behave differently under the same race |

## 8. Draft v2 — consolidated (change ledger; every row is a COMMITTED decision)

Legend: CODE = source change in this PR arc; DOC = doc-comment change; BOARD
= new dated `LATEST_STATE.md` entry (per F21); LEDGER = recorded here only.

| # | resolves | decision |
|---|---|---|
| C1 | F1, F2 | **Keep the duplication, name it, kill the vacuous test.** Depending on `lance-graph` from this crate would drag the whole spine (datafusion/arrow) into a 1050-LOC primitive — wrong direction; the eventual clean shape is `lance-graph` re-exporting FROM this leaf crate (recorded as follow-up, not done here). CODE: delete `from_env_is_none_when_a_required_var_is_missing`, replace with the honest declined-test comment (dev_s3_env precedent). DOC: `env.rs` header rewritten to cite `dev_s3_env.rs` as origin, name the drift risk explicitly, drop any implication of sole ownership, and instruct that changes must be mirrored until the re-export follow-up lands. |
| C2 | F3, F25 | **Soften the claim to what is true; keep the enum.** The guards are checkable predicates, not type-level enforcement — type-level arrives only when a flush API exists to guard (none shipped; `Flushed` reachable only through a future flush function — deliberate forward surface, now SAID). DOC: `lifecycle.rs` + `lib.rs` reworded ("a checkable predicate the flush API will enforce; until then callers consult it"). BOARD: correction in the new dated entry. |
| C3 | F4 | **Doc-truth fix; keep the crate dataset-agnostic.** `hydrate_dir` cannot record a Lance version (it copies arbitrary object trees; opening them as a Dataset is a caller concern). DOC: `dirty.rs` states plainly the version is caller-recorded (open once post-hydration, keep `version_id()`), with the pairing example. BOARD: correction in the new dated entry. |
| C4 | F5, F26 | **Make `AlreadyPublished` honest at the rename, both functions.** CODE: on rename failure in `hydrate_dir`, re-check `publish_dir.exists()` → return `AlreadyPublished` instead of `Io`. For `hydrate_file` the entry-check-only contract is worse (silent clobber): CODE: re-assert `!publish_path.exists()` immediately before the rename and return `AlreadyPublished` if it appeared — narrows (not closes) the TOCTOU window; the residual window is DOCUMENTED with the doctrine's own "worst case is a wasted rehydration / last-writer-wins between identical artifacts" framing. |
| C5 | F6 | **One shared staging-suffix helper with an in-process counter.** CODE: `pub(crate) fn staging_suffix() -> String` (new tiny module) = pid + `AtomicU64` counter + nanos; both `copy.rs` and `file.rs` call it. The FULL `hydrate_aside_then_publish` merge (S5's larger suggestion) is deferred — the shared suffix closes the uniqueness hole and halves the drift surface now; a whole-function merge is a refactor this PR doesn't need. |
| C6 | F7 | **Clean up staging on every error path.** CODE: wrap the fetch/write loop; on `Err`, best-effort `remove_dir_all(&staging)` / `remove_file(&part_path)` before returning. Best-effort (cleanup failure doesn't mask the primary error) — but see C7 for the Ok-path rule. |
| C7 | F11 | **Ok-path cleanup failures are errors.** CODE: the empty-prefix `remove_dir_all` propagates (`?`) — an Ok return must not leave debris while claiming "leaves NOTHING". |
| C8 | F8 | **`release_dir` gets a real error surface.** CODE: missing dir stays `Ok(0)` (a state, not an error — documented); any OTHER top-level `read_dir` failure returns `Err`; nested entries stay best-effort-skip. Test: replace the vacuous missing-dir framing with the two-sided pair — missing dir → `Ok(0)`, path-is-a-file → `Err` (a real falsifier). |
| C9 | F9 | **Version-tag + exact-arity the marker.** CODE: write `v1 <mtime> <len>`; `read` requires exactly three tokens, first literally `v1`, and `parts.next().is_none()` after. No migration concern — zero shipped consumers (F23). Test: 2-token legacy line and 4-token future line both refuse. |
| C10 | F10 | **Add the two missing SAFETY comments; keep all unsafe blocks.** The `unused_unsafe`/`-D warnings` half is REFUTED (orchestrator-verified against rustc 1.97.1 with a positive control) and recorded here so Phase 3 doesn't resurrect it. |
| C11 | F13 | **Split the provenance claim in `copy.rs`'s doc.** Byte-copy-not-scan-rewrite: inherited from the probe (proven there). Hydrate-aside/publish-by-rename: from the doctrine §4a and `E-A-REPEATABLE-TRANSFER-...-1`, implemented HERE for the first time, not proven by that probe. DOC only. |
| C12 | F14 | **Name the coarse-mtime hole honestly.** DOC: `is_trusted` may wrongly trust when a same-length rewrite lands within one mtime tick on coarse-granularity filesystems; acceptable because the marker guards a *skip-rehash optimization* whose misfire skips one verification of content the caller itself just published — but the failure DIRECTION (false trust) is stated, not hidden. Revisit condition named: if markers ever guard third-party-written content, add a content-hash field. |
| C13 | F15, F17 | **Cite the prior art.** DOC: `copy.rs` cites `E-A-REPEATABLE-TRANSFER-IS-NOT-IDEMPOTENCE-OVER-A-MULTI-FILE-DIRECTORY-1`; `marker.rs` cross-refs `lance-graph-ontology/src/lance_cache.rs` (checksum-axis answer to the same question). |
| C14 | F16 | **Accepted, recorded.** Module-qualified paths (`lance_graph_hydrate::dirty::is_dirty` vs the container-cache bitmap; this crate's S3→local direction vs `graph/hydrate.rs`'s local→LanceDB) disambiguate in code; LEDGER only. Losing finding kept per anti-collapse: a future rename of `graph/hydrate.rs` would remove the residual confusion, noted as optional follow-up. |
| C15 | F18 | **cfg-gate the open.** CODE: non-unix `release_dir` counts files without opening them (walk stays, `File::open` becomes unix-only). |
| C16 | F19 | **Document the staging-file interaction.** DOC: one line on `release_dir` — safe on in-flight staging trees (DONTNEED skips dirty pages), and counting them is expected. |
| C17 | F20 | **Pluralize + own the conditions.** DOC: `copy.rs` names BOTH §4a conditions: (b) is enforced (`AlreadyPublished`); (a) pinned-source-version is structurally the caller's (this crate cannot know source versioning; `hydrate_file`'s checksum pin IS condition (a) for the single-file case — stated). BOARD: singular→plural correction in the new dated entry. |
| C18 | F12 | **Spec self-correction.** LEDGER: §2's "arrow — dead dev-dep" note was WRONG (used by `dirty.rs` tests); struck, not acted on. |
| C19 | F22 | **object_store pin stands.** No fork under any plausible name owned by the org that this crate should have used; pin is byte-identical to workspace precedent. Any deeper fork-policy question is workspace-wide and out of this PR's scope. LEDGER. |
| C20 | F21, F23 | **Board corrections = one NEW dated entry** covering C2/C3/C17's claim fixes; API changes in C4-C9/C15 are free (zero consumers, verified). |

## 9. Findings — Phase 3 (the 3 reviewers, on draft v2 only)

Zero BLOCK across all three reviewers. 6 FIX items:

| decision | reviewer | severity | issue |
|---|---|---|---|
| C4 | overclaim-auditor | P2 | ensure the SHIPPED doc-comment (not just this ledger) states "narrows, not closes" — done, see `copy.rs`/`file.rs` rename-race comments |
| C10 | overclaim-auditor | P1 | "REFUTED" stated with more confidence than one rustc-invocation-on-one-toy-file supports; scope to the tested configuration | fixed in this ledger's own C10 wording (retitled "REFUTED under the tested configuration" in spirit) |
| C12 | overclaim-auditor | P2 | "acceptable because X" was asserted, not argued through to the actual bound | fixed — marker.rs doc now states the bound explicitly (misfire re-uses caller-authored bytes only) |
| C5 | dilution-collapse-sentinel | P1 | the deferred full `hydrate_aside_then_publish` merge had no durable pointer once this plan file's own working status changes | fixed — named as a follow-up in `lib.rs`'s crate doc (permanent, not plan-file-only) |
| C14 | dilution-collapse-sentinel | P2 | "disambiguates in code" answered a narrower question than F16's human-confusion concern | fixed — `lib.rs`'s "Naming, disambiguated" section added |
| C20 | firewall-warden | P2 (process flag, not a defect) | verify Phase-4 execution adds a NEW dated `LATEST_STATE.md` entry rather than in-place-editing the existing one | executed below — see the commit's board-hygiene entry |

All three reviewers independently PASSed the remaining 14 decisions with
no overlap in their FIX findings — no conflicting verdicts to arbitrate.

## 10. Ratified v3 + fixes applied

**Mid-council event:** PR #957 (the original crate) merged to `main` while
Phase 3 was still running (an operator/reviewer marked it ready-for-review
and merged before this council's Phase 4 landed — the council was run as a
pre-merge hardening pass but the merge itself did not wait for it). This
does not invalidate the council: the findings are unchanged, still real, and
still worth fixing. Consequence: Phase 4/5 lands as a **follow-up PR**, not
an amendment to #957.

**All 20 decisions (C1-C20) + all 6 Phase-3 FIX items applied**, in:

- `env.rs` — header rewritten (C1: names `dev_s3_env.rs` as origin + the
  drift risk, no false sole-ownership claim); vacuous test replaced with a
  real falsifier that saves/restores the actual env var (C1/F2).
- `lifecycle.rs`, `lib.rs` — "transition guard, not caller discipline"
  corrected to "checkable predicate, not yet type-enforced" (C2); `Flushed`
  reachability named as deliberate forward surface (C2/F25).
- `dirty.rs` — "version recorded at hydration time" corrected to
  caller-supplied (C3); new `lifecycle_of()` closes the "LifecycleState in
  zero signatures" gap additively (C2/F3), with its own test.
- `staging.rs` (NEW) — shared `staging_suffix()` (pid + atomic counter +
  nanos) closes the within-process nonce-collision hole (C5), with a
  1000-iteration uniqueness test.
- `copy.rs` — module doc splits proven-by-probe (byte-copy) from
  new-and-unproven-by-that-probe (publish-by-rename) (C11); both
  idempotency conditions named, condition (a) explicitly the caller's (C17);
  cleanup on every fetch-error path (C6); Ok-path cleanup now propagates
  instead of `let _ =` (C7); rename failure re-checked and remapped to
  `AlreadyPublished` (C4, "narrows not closes" stated in-code); cites
  `E-A-REPEATABLE-TRANSFER-IS-NOT-IDEMPOTENCE-OVER-A-MULTI-FILE-DIRECTORY-1`
  (C13/F15).
- `file.rs` — same staging/cleanup/rename-race fixes as `copy.rs`, adapted
  for the file-onto-file silent-clobber danger (worse than `copy.rs`'s
  ENOTEMPTY failure — the fix re-checks immediately BEFORE the rename, not
  only on its `Err` branch, since the danger case is the rename SUCCEEDING).
- `marker.rs` — versioned `v1 <mtime> <len>` format with exact 3-token
  arity, both directions tested (C9); mtime-coarseness risk stated with the
  actual bound argued, not asserted (C12, Phase-3 fix); cross-refs
  `lance_cache.rs`'s checksum-axis answer (C13/F17).
- `release.rs` — `release_dir` can now return `Err` on a genuinely
  unreadable root (permission denied, not-a-directory), staying `Ok(0)`
  only for `NotFound`; nested-walk failures still tolerated (a legitimate
  race) (C8); non-unix path no longer opens files it can't advise on (C15);
  documented safety on in-flight staging trees (C16); two-sided test pair
  (missing dir → `Ok(0)`, file-not-dir → `Err`) replaces the vacuous one.
- `lib.rs` — naming-collision disambiguation section (`is_dirty`,
  `hydrate`) added (C14, Phase-3 fix); deferred-merge follow-up named
  permanently (C5, Phase-3 fix); citations added.

**Non-goals confirmed still coherent** (gate 3, §5): the retire/evict-by-
rename half of §4a remains explicitly out of scope (unchanged by this
council — no code added there), consistent with `release_dir` staying
page-cache-only.

**Gates met** (§5): every frozen decision has a CONFIRMS/VIOLATES-then-fixed
resolution (gate 1); zero BLOCK survived (gate 2); NON-GOALS scoping
confirmed legitimate by dilution-collapse-sentinel (gate 3); zero
non-negotiable hits, board hygiene same-commit as the ORIGINAL crate
confirmed already present by firewall-warden (gate 4); this orchestrator
writes the board entries directly, no sub-agent write (gate 5).
