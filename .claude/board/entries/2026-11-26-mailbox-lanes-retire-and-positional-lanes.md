# Retire lance-graph-mailbox-lanes; DT64 lanes keyed by position, not by string

**Status:** OPEN. Inventory complete; no code changed. The DT64 change is the next step.

DECISION: EXO-469 (lance-graph · Decisions).

## How this was found

`a_read_only_act_allocates_what_was_measured` failed on main (43 allocations, pinned 41). The two extra allocations are the `label` String of `META`, cloned by `LaneBinding` in the guard's and the operands' `bind_lanes`. Binding a `FieldRef` with an empty label brought the count back to 41. #2211 re-pinned to 43 and called the allocations intended. That was wrong: no string may be touched on a per-call path, so the strings must go, not the pin.

## What is wrong

1. **A second, imaginary owner over the mailbox.** `lance-graph-mailbox-lanes` (#2066) rebuilds what the contract already has:
   - `MailboxLanes` is a second view over the columns `contract::soa_view::MailboxSoaView` (`soa_view.rs:67`) already lends as borrows.
   - `static NEXT_SOCKET: AtomicU64` (`lib.rs:157`) mints socket identities beside `NodeGuid`, which is the only identity of a mailbox.
   - `Column::from_iri(&str)` (`lib.rs:103`) finds a column by string compare, where its position on the facet stride already says which column it is.
   - `act` / `cascade` / `apply` with `Intent` is an actor-style read → intent → write-back loop. Ownership is `SoaEnvelope::mailbox_owner()`'s guarantee, never an actor.
   - Its only outside user says so itself: `virtual-address-lab/tests/mailbox_method.rs` doc, *"A `NodeGuid` is not one of them (the mailbox stores no key)."*
2. **Strings on the bind and check path.** DT64's `LaneSource::lane_of(…, basis: &FieldRef, …)` (#1970) hands every source a field's string identity. `LaneBinding` stored `predicates: Vec<Box<str>>` (#2021), then `refs: Vec<FieldRef>` (#2153, from this branch): one, then two, heap strings per bound field per bind.

## Inventory (all hits read or recounted on the main thread)

**Dependents of `lance-graph-mailbox-lanes`:** the root `members` entry (`Cargo.toml:43`) and `virtual-address-lab` `[dev-dependencies]` (`Cargo.toml:25`), one test file (`tests/mailbox_method.rs`, 4 `use` lines, 5 `MailboxLanes::new`, 1 `act`). Nothing else: not quack (mailbox-lanes depends on quack, not the reverse), not xlog, not report, not r2il, not dt64, not report-rs. Board and plan text: 46 lines in `.claude/`, mostly `plans/d-vas-implementation-v1.md` and the D-VAS rows of `STATUS_BOARD.md`.

**`impl LaneSource`:** 32.
- 18 key by position (`lane = field`, `_basis` unused), for example `lance-graph-report/tests/bound_sel.rs:152`, `lance-graph-pg-row/tests/sink.rs:209`, `lance-graph-vrl/tests/common/mod.rs:174`, `foundry-parity-lab/src/source.rs:192`. Positional binding already works; the `&FieldRef` parameter is unused by them.
- 14 compare `basis.predicate_iri`. Outside tests:
  - `lance-graph-mailbox-lanes/src/lib.rs:312` (`Column::from_iri`)
  - `lance-graph-report/src/lane_source.rs:159` → `Catalog::field` (`boundary.rs:264`, linear name search; its doc defends "per bind and per check, never per row")
  - `lance-graph-xlog/src/cube/lend.rs:102` (`predicates: [&str; 2]`)
  - `lance-graph-intake/src/lib.rs:395`, plus `code_for(&str)` at `:874-880` (written by this branch)
  - `lance-graph-dir-sim/src/dt64.rs:253`, `:576` (its doc, `:42`: matched by `predicate_iri`, "never by position")
  - `virtual-address-lab/src/grid.rs:257`, `quack-boundary-lab/src/lib.rs:349`, `pm-tile-fold-probe/src/fixture.rs:118`
  - report-rs (`origin/main` `323be53`): `src/xverweis.rs:399-401`, `src/workbook_delta/dt64/p21_lane_socket_lab.rs:188`

**The target shape already exists:** #2160 (`2eb04a69`, EXO-332) — a row is `key : value`; a value that follows from its key is a law lane (`contract::lane_law`, closed affine op set), read at the key's position; `borrow_all_at` reads at caller-held positions. `LaneBinding` already carries `laws: Vec<LawLane>`.

**Same rule, second list (not mailbox-lanes):** `lance-graph-xlog/src/wide.rs:478-578` folds into `BTreeMap<u64, u32>` / `BTreeSet<u64>`; `lance-graph-xlog/tests/body_pair.rs` checks answers by decoding into `BTreeSet<(u32, u32)>`.

**Same rule, the xlog repository (`AdaWorldAPI/xlog`, `origin/main` `d6de529`, the rev `crates/xlog-coverage-probe` pins):** `xlog-cube` executes xlog's compiled plans on `lance_graph_xlog::cube::Cube`, so it sits on the fabric. In `crates/xlog-cube/src/lib.rs`:
- intake keyed by string, as tuples: `base: BTreeMap<String, Vec<(u32, u32)>>` (`:251`), `sets: BTreeMap<String, Vec<u32>>` (`:253`), `empty_sets` / `empty: BTreeSet<String>` (`:255`, `:263`);
- output decoded into owned rows: `out: BTreeMap<Label, Rows>` with `type Rows = BTreeSet<Vec<u32>>` (`:264`, `:291`);
- its module doc says *"no string reaches execution"*; intake by `String` and tuple-set output contradict it.

`BTreeMap`/`BTreeSet` across the repo: 1,712 uses in 22 crates; `xlog-cube` 45 in `src/`, 44 in tests. The rest (xlog-logic, -cuda, -gpu, -prob, -runtime, pyxlog) is xlog's own engine, off the fabric; whether it migrates is that repo's decision under its `ENGINEERING.md` (not read here). This session has no push access to the xlog repo.

## Next

1. DT64: `LaneBinding` drops `refs`; `lane_of` takes the field position; field → lane resolved once at class registration; `check_bound` compares rev, basis digest and each `LaneAddr`.
2. Move the 14 string-keyed sources to position tables; the 18 lose an unused parameter.
3. Retire `lance-graph-mailbox-lanes`; virtual-address-lab's test reads `MailboxSoaView` directly.
4. Re-pin the allocation count to what remains, and add a test that fails if bind or check compares or allocates a string.
5. xlog-cube (xlog repo): intake by codebook label, relations as cube lanes, results as masks read through the cube; no `String` key, no `Vec<(u32, u32)>`, no `BTreeSet<Vec<u32>>`.

## Agent runs

Five Haiku runs (harvest-scout × 4, substrate-mapper × 1). Re-checked on the main thread: manifests 3/3, Rust uses 4/4, `LaneSource` impls 32/32, two doc hits, two map items. Wrong hits: 0. Misses: one read another repo's `AGENT_LOG.md`; two read the 678 KB log only in part; one capped on bare `apply(` (no mailbox-lanes hit among the 250 lines returned). Retired: none.
