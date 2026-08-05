# D-BLW-1 — `examples/blw_tenant.rs` (Opus filigree agent, 2026-08-04)

**Branch:** `claude/x265-x266-plans-review-h9osnl`. **Status: EDIT-ONLY —
NOT COMPILED, NOT LINTED, NOT RUN, NOT MEASURED.** The cargo grant was
withdrawn mid-task by operator directive; the orchestrator gates centrally.
Nothing below is a measurement. Every "must", "asserts", "proves" refers to
*code that was written*, never to an observed run.

## Files touched

| path | change |
|---|---|
| `crates/lance-graph-planner/examples/blw_tenant.rs` | **new**, 1,056 lines |
| `crates/lance-graph-planner/Cargo.toml` | `[dev-dependencies] cognitive-shader-driver = { path = "../cognitive-shader-driver" }` + an 18-line rationale block |
| `Cargo.lock` | 1 line, written by the orchestrator's own cargo run, not by me |

## Preflight grep (the §7 gate) — **27**, not 0

`batch_writer|BatchWriter|KanbanStep|KanbanMove|kanban|owner_adapter|MailboxSoA|SoaEnvelope`
→ **27 matching lines**. The two deleted harnesses scored 0, which is why they
could not support a substrate claim.

Per symbol: `MailboxSoA` 11 · `BatchWriter` 7 · `write_row` 7 ·
`try_advance_phase` 5 · `NextPhaseScheduler` 4 · `KanbanMove` 4 ·
`emit_bootstrap_intent` 3 · `persist_cycle` 3 · `recover_and_apply` 3 ·
`identity_plane_at` 3.

## Substrate symbols consumed, with `file:line`

| surface | definition |
|---|---|
| `MailboxSoA<N>` — the tenant | `crates/cognitive-shader-driver/src/mailbox_soa.rs:58` |
| `MailboxSoA::new` (asserts `w_slot < 64`) | `mailbox_soa.rs:292` |
| `MailboxSoA::write_row` — the ONE cycle-aware row mutator | `mailbox_soa.rs:417` |
| `MailboxSoA::set_populated` / `populated` | `mailbox_soa.rs:495` / `:486` |
| `impl MailboxSoaView for MailboxSoA<N>` (zero-copy column borrows) | `mailbox_soa.rs:852` |
| `identity_plane_at` override (real planes, `populated`-guarded) | `mailbox_soa.rs:886` |
| `impl MailboxSoaOwner for MailboxSoA<N>` → `advance_phase` | `mailbox_soa.rs:949`, `:953` |
| `MailboxSoaOwner::try_advance_phase` (checked, Rubicon DAG) | `crates/lance-graph-contract/src/soa_view.rs:311` |
| `KanbanColumn::{next_phases, can_transition_to}` | `contract/src/kanban.rs:101`, `:113` |
| `KanbanMove` (+ derived `libet_window_us`) | `contract/src/kanban.rs:172`, `:206` |
| `NextPhaseScheduler::on_version` — the thing that is NOT the applier | `contract/src/scheduler.rs:81` |
| `owner_adapter::emit_bootstrap_intent` → `rebind_bootstrap` | `planner/src/owner_adapter.rs:92`, `:68` |
| `BatchWriter::{cast, on_behalf_of, intent_moves, casts}` | `planner/src/batch_writer.rs:104`, `:128`, `:120`, `:114` |
| `persist_sink::persist_cycle` (owner-mismatch + cycle-mismatch guards) | `planner/src/persist_sink.rs:335` |
| `persist_sink::recover_and_apply` → `try_advance_phase(mv.to)` | `planner/src/persist_sink.rs:396`, applier at `:430` |
| `WalSink` / `SweepSlot` / `LandedSlot` / `DetachedCycleBatch` | `planner/src/persist_sink.rs:303`, `:128`, `:157`, `:180` |
| `StrategyOutcome` (the D-MBX-A6 carrier) | `planner/src/traits.rs:182` |

## The finding that shaped the design — `recover_and_apply` IS the paired-move applier

`.claude/knowledge/batchwriter-kanbanstep-wiring.md` §5.1 lists "the
version-completion seam" as *to build*. **Reading the source, the applier
already exists**: `persist_sink::recover_and_apply` (`persist_sink.rs:396`)
walks SEALED landings in canonical `stream_position` order and, for each,
applies **`ls.slot.paired_move`** — the move that was cast — via
`owner.try_advance_phase(mv.to)` (`:430`), with an `OwnerMismatch` guard
(`:413`) and a `StalePhase` corruption guard (`:421`). It never consults
`NextPhaseScheduler`. So §4's trap is structurally avoided by the shipped
function, not by my harness's discipline.

What is genuinely missing is narrower than "the seam": **a concrete `WalSink`**
(that module's own header: *"This module builds NO concrete Lance sink"*), and
the `BatchWriter`-cast → `Vec<SweepSlot>` glue. The harness supplies both
locally and says so.

**Second structural finding, from the constraint itself.** One tenant = one
kanban board ⇒ **2,000 rows cannot each cast a lifecycle move**: the second
one would hit `StalePhase` (`persist_sink.rs:421`), because the first already
advanced the tenant. So the harness emits *N row landings with
`paired_move: None`* (the module's documented "no-step landing",
`persist_sink.rs:145`) plus **exactly ONE** landing carrying the tenant's
paired move. The lifecycle belongs to the tenant; the payload belongs to the
rows. That falls out of "an owner is a tenant" rather than being decoration on
top of it.

## The central falsifier, as written (NOT as measured)

`snapshot(&Tenant) -> Vec<u8>` builds a **complete** LE byte image: 7 tenant
scalars + **every** per-row column of **every capacity row** `0..N_CAP`
(not `0..populated`) — energy, plasticity, both cycle stamps, edge, qualia,
meta, entity_type, temporal, expert, sigma, all three style lanes, and all
three identity planes. `IMAGE_LEN` is const-computed and asserted at runtime,
so a column silently dropped from the snapshot cannot pass as "byte-identical"
— the exact defect the previous arm shipped (a 6-column snapshot calling
itself a full comparison).

- **Silent half** (per cycle, ×3): snapshot → `sweep_rows` over every row →
  snapshot → `first_diff` must be `None`, else `panic!` naming the offset and
  its column. The sweep takes `&V: MailboxSoaView` — *no `&mut self` during
  computation* is structural, not a comment.
- **Can-fire twin** (×2, both non-trivial): (a) a gated `write_row` of one
  fixed column on one row of 2,000 — asserted to be detected AND located in
  "fixed columns"; (b) a **one-bit** write into the **ANGLE plane**, which is
  all-zero for the entire run — asserted detected AND located in "ANGLE plane".
  (b) is deliberate: a snapshot that never reads the identity planes would
  report "identical", which is precisely the prior defect.
- **Anti-vacuity on the write-back** (§12.3): after stamping ONLY the fired
  rows, every row is compared row-image-wise; `differs != is_dirty` for any row
  is a failure, plus `changed == fired.len()` and `changed > 0`.
- **Guard probe:** `try_advance_phase(Evaluation)` from `Planning` must `Err`
  **and** leave the image byte-identical.
- **Trap probe (cycle 3):** `NextPhaseScheduler` is asked what it *would* have
  proposed from `Evaluation` (`Commit`); the cast says `Plan`; the applied move
  must equal the cast and must differ from the scheduler's default.
- **Lens anti-vacuity:** probe "god" must fire on some rows and not others;
  an absent term must fire nothing. The stronger *sparseness* assert is gated
  on `scanned >= 512`, and the gate is **stated rather than tuned** — a short
  prefix of Genesis 1 is ~90 % "God", so asserting sparseness there would be an
  assertion about Genesis 1, not about the filter.
- **Image coverage gate:** `nonzero > seated * 8` — scale-free, so it does not
  silently pass at one corpus size and fail at another.

## The seam I stopped at

**Durability observation.** `MemWal` is in-process `Mutex`/`Vec` (mirroring
`persist_sink`'s own `FakeWalSink`, fence included). It proves the *contract*
— one append per cycle, stored order, sealed read horizon — and **not
durability**. `deinterlace` still has no production caller and there is still
no production `DeinterlaceRow` implementor (`batch_writer.rs` module doc), so
the "durability is observed via `QueryReference::at` + deinterlace" contract is
**not** exercised here. I did not build a `DeinterlaceRow`: that is D-BLW-3's
read surface, and faking it would have been the substitution this deliverable
exists to prevent.

## What I did NOT do

- Did not run cargo in any form (grant withdrawn mid-task).
- Did not build a stance instrument (§12.3c/§12.7 killed it; D-BLW-2's problem).
- Did not touch `persist_sink.rs`, `temporal.rs`, `batch_writer.rs`,
  `owner_adapter.rs`, `soa_view.rs`, `mailbox_soa.rs`, or `crates/jc` (§12.5).
- Did not re-add the `jc` dev-dep.
- Did not enable `cognitive-shader-driver`'s `with-planner` feature (which
  would reach `MailboxSoA::cast_on_behalf`, `mailbox_soa.rs:757`) — that would
  activate the reverse edge, and `cast_on_behalf` does not perform the
  bootstrap rebind that `emit_bootstrap_intent` does.
- Did not write `AGENT_LOG.md` or any shared board file.

## Open questions I could not close without a compiler

1. **Dev-dep cycle.** `lance-graph-planner --dev--> cognitive-shader-driver`,
   whose own planner dep is optional and off by default. *Partially resolved
   by evidence I did not produce*: the orchestrator's `Cargo.lock` now lists
   `cognitive-shader-driver` under the planner, so resolution succeeded.
   Build/link is still unverified.
2. **Clippy under `-D warnings`.** The `snapshot` loop was rewritten to iterate
   `energy.iter().enumerate()` specifically to remove any `needless_range_loop`
   exposure; a `match`→`if let` rewrite removed `single_match` exposure. Other
   default lints unverified.
3. **Deref coercion sites.** `WriteCell.content/topic/angle` are
   `Option<&[u64]>`; I pass `Some(v.as_slice())` everywhere rather than
   `Some(&v)` so no coercion is relied on.
4. **`o.phase() as u8`** on the `#[repr(u8)]` fieldless `KanbanColumn` — should
   be legal; unverified.
5. **Runtime cost.** ~12.16 MiB per snapshot × ~12 snapshots, built
   byte-by-byte in a debug build. Untimed. If the orchestrator finds it slow,
   the corpus bound is the first knob (`-- 512`), not the snapshot's coverage.
6. **`ISS-MAILBOXSOA-ROW-COST-VS-512B-CANON` is NOT resolved here.** The image
   is a figure of `MailboxSoA<2048>` (6,144 B/row of identity planes alone),
   **not** of the 512 B canonical `NodeRow`. Named in the file header so no
   number in the output is read as a canon-row figure.

---

## ORCHESTRATOR GATE RESULT (main thread, 2026-08-04) — closes the open questions above

The agent wrote this harness under the no-cargo rule and correctly reported it
as NOT COMPILED / NOT LINTED / NOT RUN. Gated centrally in the single shared
`target/`. Outcome:

| gate | result |
|---|---|
| `cargo build -p lance-graph-planner --example blw_tenant` | **PASS** (one round-trip: the first attempt failed `E0433 cannot find crate cognitive_shader_driver`; the agent had already added the dev-dep, my build simply predated that edit) |
| `cargo clippy -p lance-graph-planner --example blw_tenant` | **PASS** — 0 warnings citing this file, after 2 fixes (below) |
| `cargo fmt -p lance-graph-planner --check` | **PASS** |
| run, 2,000-verse bound | **PASS**, 3 s wall, exit 0 |

**Substrate check:** grep for
`batch_writer|BatchWriter|KanbanStep|KanbanMove|kanban|owner_adapter|MailboxSoA|SoaEnvelope`
returns **27**. The two deleted harnesses returned **0**.

**Two clippy findings, and one of them clippy was WRONG about:**
1. `needless_range_loop` — genuine. Fixed by enumerating the borrowed `energy`
   slice (`energy.iter().enumerate().take(n_rows)`).
2. `explicit_counter_loop` on `stream_position` — **false positive, suggestion
   rejected.** That variable advances once per FIRED ROW inside the landing
   `map` (hundreds per cycle) *plus* once for the tenant-level landing, so it is
   a witness-stream position, not a loop counter. Clippy's
   `(0_u64..).zip(plan.iter())` rewrite would have silently redefined it as the
   cycle index. Suppressed with `#[expect(..., reason = ...)]`.

**Measured falsifier outcomes** (all from the real run, not restated from the
source): PROBE-GUARD illegal edge refused + tenant byte-identical; PROBE-LENS
255/2000 fire and 0 for an absent term (discriminating, not degenerate);
PROBE-MUT-a gated column write caught at byte 6226039; PROBE-MUT-b **one-bit**
ANGLE-plane flip caught at byte 6232248; PROBE-TRAP — *scheduler proposed
Commit, cast said Plan, applied Plan, the paired move won*.

**The boundaries the harness declares are accepted as-is and NOT upgraded:**
durability unproven (MemWal is in-process), `deinterlace`/`DeinterlaceRow`
unexercised (no production implementor exists), no stance or semantic claim.

Landed in `cbca9e6`.
