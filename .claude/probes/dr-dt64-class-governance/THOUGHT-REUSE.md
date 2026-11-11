# D-DRDT-CLASS-GOVERNANCE-1 — Thought-git reuse for versioned class and method state

Status: census, docs only. Read at main 238b6d1e4 (2026-11-11).

Question: what could be reused from the thought-git work for versioned replay of class and method state, and what a "Thought-0/63 reuse probe" would have to show.

Facts and a probe outline only. No architecture decision. Paths are under `/home/user/lance-graph` unless stated.

## 1. What the plan defines

Source: `.claude/plans/thought-git-api-v1.md`, status line: "PROPOSAL, 2026-10-17. Test-first; nothing built" (line 3). It has later dated decisions and a correction section (lines 267-309).

### Objects

- **Thought.** A number `0..=63`, "64 in all" (line 17). Slot 0 is main, the givens of the current sealed version. Slots 1..=63 are allocated on demand (lines 6-10, 273).
- **Version.** A Lance version. A sealed version is "a PR entry": one `land` produces one version that records what changed, which sources it carried, what the gate decided, which conflicts were recorded and which version it was rebased onto (lines 141-148).
- **Cell.** A coordinate. Names are labels resolved once through a label codebook (lines 17-18).
- **Delta.** The alpha overlay, "the record of where attention went" (lines 35-40).
- **Receipt and PR record.** A separate table keyed by (data dataset, data version, thought id) holding PR records, receipts and relations. Decided 2026-10-20 (lines 172-175).
- **Slot number.** CE64 bits 53..58 (`w_slot`), 6 bits, names the thought. The plan says it names a lane and points nowhere (lines 269, 274). Slot numbers recycle, so "no reader of a sealed row reads `w_slot`" (lines 288-291).
- **`ThoughtCommitId`.** Proposed as a thought identity that survives slot reuse and is never reused (lines 248-251, 285). Status: candidate.

### Refs and commits

- **Refs.** `Main` (sealed version v), `Thought` (slot 1..=63 with a `merge_base`), `MergeQueue` (lines 79-98).
- **Commit.** `land` is the only commit: an up-to-date check, a gate, then a squash into the next Lance version (lines 97, 179-194).
- A version number is never reused. A thought number is reused. Blame therefore returns `(landing version, source bitset)` (lines 156-160, 195-197).

### API verbs

From the surface listing (lines 78-98): `branch`, `version`, `checkout(v)`, `log`, `blame`, `write`, `read`, `diff`, `status`, `rebase`, `squash_into`, `discard`, `enqueue`, `land`.

Rules (lines 177-200): up to date before landing; rebase moves no data; landing writes the next version; blame survives the squash; strata survive.

### Work items

D-TGA-0 to D-TGA-6, all unchecked (lines 202-231). The plan points to `thought-git-basics-v1.md` (D-TGB-0..6) for the basics (line 309). That file was not read.

### Where "64 layers" and "0..63" appear in the plan

Lines 6, 17, 80, 86, 208-213, 273-276, 280-286. "Layer 0" is the givens (D-HL givens layer, line 30). The plan does not say "64 thought layers" literally. It says 64 thought slots (line 273) and "W is the thought layer, 0..63" (line 276).

## 2. What `crates/thought-git-probe` implements

Files under `crates/thought-git-probe/src/` (Glob): `lib.rs`, `fixtures.rs`, `oracle.rs`, `slots.rs`, `receipt.rs`, `squash.rs`. Tests under `tests/`: `oracle.rs`, `signals.rs`, `slots.rs`, `protected_readers.rs`, `squash_gate.rs`, `receipt.rs`.

`lib.rs:4-6` states the scope: the full API "is still a proposal. This probe holds the pre-registration and the first lab piece, the slot pool; it has no production land, rebase or persistence pipeline."

| piece | where | what it does |
|---|---|---|
| Fixtures | `fixtures.rs:12-31` | 12 rows, 4 lanes. `Outcome` enum (four Rubicon outcomes: Commit, Plan, Prune, Veto, per `receipt.rs:637-644`) |
| Oracle | `oracle.rs:1-108` | `next_version(givens, prior, landing)`: only `Commit` lands; a cell with a given is never overwritten (recorded as `Contradiction`); two thoughts offering different values for one new cell give a `Divergence` and nothing is stored; unwritten cells read as `None` |
| Naive squash | `oracle.rs:110-122` | union of writes, later over earlier |
| Slot pool | `slots.rs:27-136` | `LIVE_SLOTS = 63`; `Slot { index 1..=63, generation: u32 }`; `branch` with FIFO or least-used eviction; a full pool squashes into a live parent or child, else evicts; a stale handle is refused after free, squash or eviction (`slots.rs:1-25`) |
| Squash gate | `squash.rs:13-110` | `Lane` ids asserted in `1..=63` (`squash.rs:22`); strict contradiction and divergence checks over final states; blame as a lane bitset; ordered history and a seal |
| Receipt | `receipt.rs:366-387` | thought, base, program, inputs, seed, config, `captured_in` slot, source fold, roots, output fold, outcome |
| Replay | `receipt.rs:564-597` | refuses in order: `MissingSeed`, `ConfigChanged`, `UnknownBase`, `SourceChanged`, `OutputMismatch`. Never reads a slot |
| Version store | `receipt.rs:53-68`, `153-184` | `BaseRef { dataset, branch, version }`. `VersionStore` is "a stand-in for Lance". `commit_cycle` seals a version with a `CommitWitness` (cycle, cohort, batch fold) |
| Metadata table | `receipt.rs:695-847` | `MetaKey { data: BaseRef, thought }`; `stage` then `seal`; a sealed row is immutable; `landed` checks seal and witness; `incomplete` lists committed thoughts with no sealed row; `replay_landed` |

What the probe states it is not (`receipt.rs:39-45`): lab only, `BaseRef` stands in for D-CPG's `DurableRef`, `VersionStore` stands in for Lance, wiring the witness into the real writer is D-TGB-7's admission decision.

Not implemented in the probe, per the files read: `checkout`, `log`, `blame` by version, `rebase`, `status`, `MergeQueue`, any Lance storage, any class or method type. Receipt programs are one hard-coded program, `DERIVE` (`receipt.rs:207-219`).

### Layers 0 to 63 in the probe

- Slot 0 is "the givens (main)"; slots 1..=63 are live thoughts (`slots.rs:3`, `27-28`, `42`).
- "Layer 0" is used for givens in `oracle.rs:9`, `13`, `43`.
- `squash.rs:13` and `22` bound lane ids to 1..=63.

## 3. Counterparts in DT64 today

grep for `thought` / `Thought` in `crates/lance-graph-dt64/src` and `crates/lance-graph-dt64-lance/src`: no matches. DT64 has no thought concept. The counterparts below are by role only.

| thought-git piece | closest DT64 piece | how close |
|---|---|---|
| Version number never reused | `VersionClock` only increases (`dr.rs:165`) | same property. DT64's counter is per clock, and can be forked (`dr.rs:184-187`) |
| Append-only version log | `RouteLog` events per position (`dr.rs:209-215`); `TileLog` events per tile (`dr.rs:713-757`) | same shape. In memory |
| `checkout(v)` read at a version | `routed_at(version, position)` (`dr.rs:296`); `MapTable::diagnose_at(version, tile)` (`dr.rs:1075`) | read side only. Not a branch |
| Receipt and replay without the slot | `dt64-lance` `restore`: re-applies each cycle at its own version, refusing invalid logs (`dt64-lance/src/lib.rs:513-547`, `561-630`) | same idea for map tables only. No program, no seed |
| Commit witness naming cycle and cohort | marker cast per advance: kind, version, source, relation_map (`dt64-lance/src/lib.rs:413-420`); `ForeignOwner` check (`522-527`) | partial. One owner mailbox, no cohort |
| Data first, metadata second, `incomplete` | `claim` then `advance_at` after the seal ticket; failed seal leaves a gap (`dr.rs:981-1014`; `dt64-lance/src/lib.rs:18-22`) | similar discipline. Different failure vocabulary |
| Slot generation, stale handle refused | `Frame` identity by `Weak` pointers; `FrameStore::check` refuses a re-sealed frame (`dr.rs:462-479`) | similar. A frame is not a slot |
| Schema witness on a basis | `BasisDigest` (`lib.rs:287-329`) | different thing. Not a receipt |
| Contradiction record for a write to a given | `Refusal::BehaviourImmutable` for a re-registered class (`lib.rs:904-906`) | refuses, records nothing |
| 64 slots | `Width::W64`, the predicate axis of 64 in `DR256_SQ_X64` (`dr.rs:113-117`) | the number 64 only. Different quantities. No code links them |
| PR record table keyed by (dataset, version, thought) | none | UNKNOWN. No class or method version table exists (see `SNAPSHOT-VERSIONING.md`) |
| `MergeQueue`, `rebase`, `status`, `blame` | none in DT64 or in the probe | none |

## 4. A reuse probe would need

Each item names an independent oracle and a negative control. This is an outline, not a decision to build.

1. **Class state as a version-addressed cell set.** Model a class's schema rev and method bindings as cells in the probe's `VersionStore`, with a `BaseRef` per sealed version. Oracle: the from-scratch `next_version` in `oracle.rs`, fed the same landings. Negative control: a landing that changes a bound method under the same `MethodId` and `SigRev` must be refused, as `BindStore::register` refuses it (`lib.rs:904-906`).
2. **Replay a class at version N without the live store.** Rebuild the `BindStore` content at N from receipts and compare its binding digests with the digests recorded at N. Oracle: `binding_digest` computed by dt64 on an independently constructed store (`lib.rs:411`). Negative control: change one effect argument in the receipt and require `OutputMismatch`-style refusal.
3. **Version identity across processes.** Show a `MethodRef` or its stamp binding digest verified in a second process using only the persisted receipt. Oracle: `Stamp::binding()` of a store built from scratch from the same inputs (`lib.rs:380`). Negative control: a reference whose `issuer` is a different store instance still fails `check_issuer` in-process (`lib.rs:1161-1167`), so the probe must not weaken that check.
4. **Slot reuse does not alias a class.** Reuse a slot index for a different class thought and require the old handle refused. Oracle: `SlotPool` generation check (`slots.rs:23-25`). Negative control: skip the generation check and require the old handle to be accepted, so the test can fail.
5. **Persisted cohort per advance.** Write route events, map advances and class registrations through one cycle id and read them back from Lance with the cycle store schema (`cycle_sink.rs:127-147`). Oracle: `restore` plus `scan_image` on a reopened store. Negative control: a cycle written on behalf of another mailbox is refused (`dt64-lance/src/lib.rs:522-527`).
6. **Routes and maps on one clock.** Share one `VersionClock` between a route log and a `MapTable` and show one version names one point in both. Oracle: `routed_at(v)` and `diagnose_at(v)` at the same `v` against a hand-built timeline. Negative control: two independent clocks must produce a mismatch the test detects (`BindStore` holds its own route log, `lib.rs:777`).
7. **Gap handling.** Fail one seal and require the claimed version to be a gap in both log and store, with later versions still answering. Oracle: `dt64-lance` replay of the surviving cycles (`dt64-lance/src/lib.rs:561-630`). Negative control: an out-of-order `advance_at` is refused (`NotAfterLast`, `dr.rs:1009-1011`).
8. **Thought 0 and 63 edges.** Run the 62-open, 63-open edges the plan pre-registers: the 63rd `branch()` succeeds, the 64th is refused, slot 0 is never handed out (plan lines 211-215). Oracle: `SlotPool::live()` count (`slots.rs:100-102`). Negative control: allow slot 0 and require a test to fail.
9. **Stale frame replay.** Show whether a `Frame` sealed before a restore can be re-issued at a past version, which `dt64-lance` marks OPEN (`dt64-lance/src/lib.rs:40-41`). Oracle: `MapTable::map` on the issuing table before restore. Negative control: a frame of another table is `Stale(ForeignFrame)` (`dr.rs:1057-1060`).

## 5. Not checked

- `thought-git-basics-v1.md`, `PREREGISTRATION.md`, `docs/research/versioned-semantic-candidates-64-thoughts-rfc.md` were not read.
- The `tests/` of the probe were not read beyond their file names.
- Whether `.claude/knowledge/epistemic-frontier-capstone.md` agrees with the plan's `w_slot` statements: not checked.
