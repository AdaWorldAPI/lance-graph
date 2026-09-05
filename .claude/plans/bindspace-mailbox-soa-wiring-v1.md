# BindSpace → MailboxSoA — the WIRING plan (v1)

> **Status:** MEASURED / ready-to-execute. Every claim below carries file:line
> from a read of the CURRENT tree (origin/main `9f8aa779`, #1174), not from a
> prior plan. Three prior plans exist and all three are design-stage; this one
> supersedes their *sequencing* only — see §6.
>
> **READ BY:** anyone touching `cognitive-shader-driver`, the mailbox cutover,
> or proposing BindSpace retirement.
>
> **Confidence:** HIGH on the census (three independent Sonnet censuses, each
> naming its exhaustive search per guardrails §1 rule 10, cross-checked against
> the orchestrator's own reads). NOT COMPILED, NOT RUN — no `cargo` was invoked
> at any point (guardrails §1 rule 7); every "exists / is wired / is gated"
> statement is structural.

## 0. The headline correction

**The migration is far more built than the board says, and is blocked on
nothing.** `COMPONENT-MAP.md:108` rules `bindspace.rs::BindSpace` *"RETIRE
(W7)"*. Measured against the tree:

| board claim | measured |
|---|---|
| retirement is wave **W7** | **no W7 exists** — `INTEGRATION-PLAN.md` runs W0–W6; W6 is classid-adoption/legacy-alias retirement, not this. The retirement has **no wave and no D-id** (`STATUS_BOARD.md`: zero `D-V3-W7` rows) |
| parity gate at `mailbox_soa.rs:1145` | it is at **:1361** (216-line drift), live, not `#[ignore]`d |
| one parity test | **two** — `:1361` (7 columns) and `:1480` `test_mailbox_soa_dense_planes_parity_with_bindspace`, which compares `content`, the heaviest plane |
| successor column: `—` | `COMPONENT-MAP.md:109` names `MailboxSoA<N>` plainly; the generated index just fails to carry it |

## 1. What is actually built

- **Read shim — WIRED.** `BackingStore` (`backing.rs:55-149`): 6 read methods,
  both `Singleton` and `Mailbox` arms real. Called throughout
  `driver.rs::run` (`:252, :257, :283, :285, :323, :369, :437, :533`).
- **Write shim — BUILT, COMPLETE, UNWIRED.** `BackingStoreWrite`
  (`backing.rs:164-314`): 9 write methods, **both arms real, zero
  `todo!()`/`unimplemented!()`/no-op**. Its `Mailbox` arm already delegates to
  the nine `MailboxSoA` setters (`:417, :594, :606, :618, :630, :646, :658,
  :670, :686`). **Zero callers outside `backing.rs`'s own `#[cfg(test)]`
  module** — `driver.rs` contains the string `BackingStoreWrite` 0 times.
- **Equivalence harness — REAL.** `tests/w2_differential.rs`: 4 tests asserting
  **bit-identical `ShaderCrystal`** between a singleton-backed and a
  mailbox-backed driver (full window, offset window, meta-prefilter, alpha
  merge), each with non-vacuity checks. No `#[ignore]`.
- **The routing mechanism — PRESENT AND UNCONDITIONAL.**
  `ShaderDriver.mailboxes: HashMap<MailboxId, MailboxSoA<1024>>`
  (`driver.rs:99`) is **not** feature-gated; `backing()` (`:200-223`) returns
  `BackingStore::Mailbox(mb)` when populated.

## 2. What is not wired (the whole gap)

1. **`BackingStoreWrite` has no caller.** The write path bypasses the shim.
2. **Five writers still hardcode `&mut BindSpace`** — `engine_bridge.rs`
   `ingest_codebook_indices:58`, `dispatch_busdto:281`,
   `write_qualia_observed:490`, `write_qualia_17d:548`, `persist_cycle:784` —
   plus three `serve.rs` handlers reaching through `Arc::get_mut`
   (`:139, :519, :639`).
3. **`mailboxes` is only ever populated by one caller, a test**
   (`tests/w2_differential.rs:277` via the builder `with_mailbox`,
   `driver.rs:890`). Production drivers hold an empty map, so even a
   feature-on build takes the singleton fallback (`driver.rs:217`).
4. **CI never builds the feature.** `mailbox-thoughtspace = []`
   (`Cargo.toml:82`), `default = []`. Zero `--features mailbox-thoughtspace`
   anywhere in `.github/workflows/`. `tests/w2_differential.rs:20` is
   `#![cfg(feature = "mailbox-thoughtspace")]` — **the entire file**. So all 4
   equivalence tests and every `Mailbox` write arm have **zero CI coverage**
   and can rot silently.

## 3. What is NOT a blocker (settled, not open)

The three `&mut self` methods on `impl BindSpace` (the only such block,
`bindspace.rs:318-418`) have no `MailboxSoA` counterpart — and all three are
**absent by design, already decided**:

| method | why absent |
|---|---|
| `write_cycle_fingerprint:415` | the `Vsa16kF32` cycle plane is *"NEVER migrated"* (`mailbox_soa.rs:140`) |
| `set_qualia_f32:354` | `with-engine` lab-only tenant; production + MailboxSoA carry the i4 column only (`bindspace.rs:271`) |
| `set_ontology:364` | the registry is shared/immutable, cold Zone-2, external to any mailbox (`mailbox_soa.rs:106`) |

**The cycle plane is the one that looked load-bearing and is not.** Measured:
`driver.rs`'s `cycle_fp` is computed **transiently** (`:367` zero-init, `:372`
XOR-accumulate) and never read from storage. The stored plane has exactly
**one** production reader — `unbind_busdto` (`engine_bridge.rs:412`) — and that
block is **already** `#[cfg(not(feature = "mailbox-thoughtspace"))]` (`:409`)
with the trade recorded in-source: *"Under `mailbox-thoughtspace` this block is
gated out and the non-headline indices stay 0 (documented loss)."* Every other
reader is a `#[cfg(test)]` assertion.

## 4. The plan

Deliverables are `D-BSW-0..4` (BindSpace→SoA Wiring); rows in
`.claude/board/STATUS_BOARD.md`.

Ordered by dependency. Each step names its falsifier. No step deletes anything.

### D-BSW-0 (M0) — put the feature under CI (do this first; it is the cheap one)

Add one workflow line building/testing `cognitive-shader-driver` with
`--features mailbox-thoughtspace`.

**Exact precedent, same repo, same shape:** `rust-test.yml:158-173` records a
feature gate hiding tests from CI, diagnosed by counting
(*"`--features supervisor` runs 13 tests; `--features supervisor,cycle-driver`
runs 43"*) and closed by one added feature, justified as *"a strict superset of
what this step ran before, so it cannot lose coverage."* This is that, for
`mailbox-thoughtspace`.

- **Falsifier:** test count strictly increases, and the 4 `w2_differential`
  tests appear by name in the run. If the count is unchanged, the feature did
  not reach the crate and the step is wrong.
- **Risk:** the 4 tests may be red — nobody has run them in CI, ever. That is
  the point of running them. A red result is a finding, not a failure of M0.

### D-BSW-1 (M1) — wire `BackingStoreWrite` into the driver

Give the driver a write entry point that returns `BackingStoreWrite` the way
`backing()` returns `BackingStore`, and route `driver.rs`'s write path through
it. The shim already handles both arms; this adds a caller, not a capability.

- **Falsifier:** a test that writes through the driver under both feature
  states and asserts the same observable row state (the `w2_differential`
  pattern, extended from read to write).
- **Guardrail:** M1 adds **no** new writer to `BindSpace` (guardrails §2:
  *"add new writers to it"* is the named footgun) — it moves an existing write
  behind the existing shim.

### D-BSW-2 (M2) — route the engine_bridge writers through the shim

Take them in the order the census gives, easiest first:

| writer | route |
|---|---|
| `write_qualia_observed:490`, `write_qualia_17d:548` | direct — `BackingStoreWrite::set_qualia` / `MailboxSoA::set_qualia:606` |
| `persist_cycle:784` | its `edge` + `meta` writes map to `set_edge:594` / `set_meta:618`; the cycle write is the §3 documented loss under the feature |
| `dispatch_busdto:281` | same shape; already `#[cfg(with-engine)]` |
| `ingest_codebook_indices:58` | no bundled equivalent — compose from `set_content:686` + `set_meta:618` + `set_temporal:646`, or leave last |

- **Falsifier per writer:** the existing parity tests must stay green, and the
  writer's own test must pass under **both** feature states.

### D-BSW-3 (M3) — populate `mailboxes` in a production path

Until a non-test caller calls `with_mailbox`, the feature-on build still takes
the singleton fallback (`driver.rs:217`). This is the step that makes the
migration observable in production rather than only in tests.

- **Gate:** `driver.rs:209`'s `debug_assert!(self.mailboxes.len() <= 1)` — W5
  multi-mailbox routing is out of scope; exactly one designated mailbox.

### D-BSW-4 (M4) — retirement — NOT NOW

Deleting `BindSpace` is **explicitly forbidden at this stage**. Guardrails §2
names both directions as footguns: *"add new writers to it; remove it."* §1
rule 8: *"Retirement is proof-gated and never a worker task."* M4 opens only
after M0–M3 are green and a corpus/adoption proof exists — and it needs a
**wave and a D-id first**, since neither exists today.

## 5. Board corrections this plan owes

1. `COMPONENT-MAP.md:108` — "W7" names a nonexistent wave; the retirement has
   no D-id. Either mint one under W6 or state plainly that it is unscheduled.
2. `COMPONENT-MAP.md:108` — parity-gate pointer `:1145` → `:1361`, and it is
   **two** tests, not one (add `:1480`).
3. The supersession index's successor column for `BindSpace` is `—` while
   `COMPONENT-MAP:109` names `MailboxSoA`. Generator gap, not a data gap.

## 6. Relationship to the three prior plans

`bindspace-mailbox-soa-dependency-map-v1` (*"MAP / preflight. No source wired
yet"*), `bindspace-singleton-to-mailbox-soa-v1` (*"CONJECTURE / design … NOT yet
implemented"*), `bindspace-mailbox-soa-w3-w4a-impl-v1` (council-reviewed spec).
All three are design-stage and predate `backing.rs` (landed 2026-07-24, #844).
**None is superseded in content** — this plan supersedes only their sequencing,
because the thing they sequence toward (the dual-armed shim) now exists.

Their two cited in-flight rows are stale: `STATUS_BOARD` marks D-V3-W3a and
D-V3-W4a *"In PR (2026-07-10, branch `claude/review-claude-board-files-nhqgx1`)"*
— that branch is **0 ahead / 1204 behind** origin/main, last commit a merged
medcare PR, touching zero relevant files. Neither is in flight.

## 7. The stranded predecessor — found late, and it changes §6

**A plan with this exact filename already exists, council-hardened, and never
merged.** `origin/claude/bindspace-mailbox-soa-wiring-plan` carries
`.claude/plans/bindspace-mailbox-soa-wiring-v1.md` (273 lines, dated
**2026-06-17**, *"CONSOLIDATED + 3-BRUTAL-CRITIC PASS APPLIED; v1 verdict was
HOLD; v2 integrates every P0/P1"*). The branch is **2770 commits divergent** from
today's main and the file is **not on main** — the document stranded; the work it
specified largely did not.

Found only because pushing this plan hit a non-fast-forward on the same branch
name. Nothing in the board points at it.

**Its binding operator constraints, carried forward verbatim** (they govern M1–M4
and I did not have them when §4 was written):

> two paths step by step; never delete the old before the new is tested;
> CausalEdge64 dedup precise; delete BindSpace LAST.

**Its P0/P1 findings are all RESOLVED in today's tree** — checked, not assumed:

| its finding (2026-06-17) | status today |
|---|---|
| *"W2 is BLOCKED on #518 — `MailboxSoA::content_row` does not exist until #518 merges"* | **CLOSED** — `mailbox_soa.rs:680` `pub fn content_row` |
| *"prefilter off-by-(N−len): a zeroed `MetaWord` passes `accepts`, so a 1024-row mailbox returns 1024 phantom rows; `MailboxSoA` has no populated-count → add a high-water-mark (W1c)"* | **CLOSED** — `mailbox_soa.rs:217` `populated: usize` + `set_populated`, and `backing.rs:79-80` clamps the mailbox prefilter to `mb.populated()` |
| *"DROPPED the W2.5 mutability gate — there is no `Arc<ShaderDriver>`; the Mutex guard already yields `&mut ShaderDriver`. Do NOT add a `RwLock`"* | still correct — `driver.rs:99` `mailboxes` is a plain `HashMap`; no `RwLock` present |

So the 2026-06-17 council was right, its blockers were closed by later PRs
(#518 content planes, W1c populated-count, #844 `backing.rs`), and **the only
thing that never landed is the document**. That is the same failure mode this
plan's §5 records for `COMPONENT-MAP`: the code moved and the board did not.

**Consequence for §6:** the supersession claim there is now four plans, not
three, and this one is the closest ancestor — §4's M0–M4 is a re-sequencing of
its W2→W7 against a tree where its blockers are gone, not a fresh design. Its
architecture verdict (*"mailbox-as-owner, zero-copy, CausalEdge64 firewall,
cycle-drop confirmed sound by all 3 critics"*) is inherited, not re-litigated.

**Open question for the operator, not resolved here:** whether that branch
should be revived, cherry-picked for its remaining W-step detail, or left as a
citation. It is 2770 commits divergent; a merge is not the obvious answer.
