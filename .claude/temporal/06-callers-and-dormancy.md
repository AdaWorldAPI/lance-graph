# 06 — callers, dependencies, and the dormancy census

**The arc's headline finding.** Nearly every mechanism examined is built,
tested, doctrinally justified, and connected to nothing. The stack's problem is
not missing mechanism.

## The dormancy census

| mechanism | where | production consumers |
|---|---|---|
| `SpogTenants` (the octopus) | `contract/src/spog_tenants.rs`, 246 lines, 4 tests | **ZERO**, in any repo. Only `pub mod spog_tenants;` at `lib.rs:184` |
| `AlphaTunnel` | `contract/src/alpha_tunnel.rs` | via `dispatch_thought` only; `run_wave_parallel` has ZERO callers outside its own test; `lane_mut` has one, in `rung_horizon`'s tests |
| `rung_horizon` (planner V2) | `planner/src/rung_horizon.rs` | `pub mod` only; `run_wave_with_horizon` exercised solely by its own tests |
| `AlphaMask` | `contract/src/alpha.rs` | **ZERO production hits anywhere** |
| `rung_schedule` | `contract/src/rung_schedule.rs` | `pub mod` + ONE example (`rung_waves.rs`) |
| `MailboxSoA::consume_firing` | `cognitive-shader-driver/src/mailbox_soa.rs:380` | **ZERO in production**; every call site is a test or example. Both BLW harnesses say so outright: *"`consume_firing` is not exercised — the harness never delivers batons"* |
| ndarray `ternlog` / `mask_ternlog` | `ndarray/src/simd*` | lance-graph and OGAR: **NO caller**. Only `lance-graph-java`'s lgj-abi consumes it |
| `residue_band` (the pothole) | `OGAR/crates/ogar-dismech/src/lib.rs:417` | **ZERO outside its own crate** |
| surrealdb kv-lance `Timeline` | `surrealdb/core/src/kvs/lance/timeline.rs` | `#![allow(dead_code)] // unwired`, 1,662 lines of tests |
| `batch_writer::cast()` | `planner/src/batch_writer.rs` | its own module doc says ZERO production call sites; ledger `TD-DOC-COMMENTS-CLAIM-UNWIRED-BEHAVIOUR` |
| `witness_tombstone.rs` | `lance-graph/src/graph/` | all `todo!()` |

### The two exceptions

| mechanism | consumers |
|---|---|
| **`TemporalPov`** (`contract/src/temporal_pov.rs`, 314 lines, zero-dep) | `deepnsm-v2` (whole KJV, 23,145 verses) AND `stockfish-rs` (real lichess games). Two repos, two independent real corpora |
| `recipe_dispatch` | `lance-graph-ogar/src/recipe_vocab.rs` — a real consumer |

`EvidenceMask` is a partial exception: one real consumer,
`planner/src/dismech_candidates.rs`.

## The alpha dependency graph, measured

```
dispatch_thought (contract/wave_dispatch.rs:62)
  ├── AlphaAllocation::over(base)         contract/alpha.rs
  ├── schedule_for(seed)                  contract/rung_schedule.rs
  ├── AlphaTunnel::over(&alloc, cycle)    contract/alpha_tunnel.rs   [10 lanes]
  │     └── AlphaOverlay::over_shared     one allocation, ten shadows
  └── run_recipes(&mut ctx, ids)          contract/recipe_dispatch.rs
        └── result DISCARDED (`_steps`)

  the ONE production caller:
    MedCare-rs medcare-nodesoa/src/frontier_dispatch.rs:81

  the terminal consumer of the scanpath:
    FrontierDispatch.scanpath -> PatientShadow.scanpath
      -> scanpath_identities() -> medcare-server/src/views/reasoning_debugger.rs:896
      = a debug HTML view.
```

**Nothing re-reads the scanpath to rank, prune, prioritise, or seed a next
cycle.**

`MailboxSoA` (energy/threshold/firing) and `dispatch_thought` (rung/alpha)
share NO type, NO call and NO data path. **Two disjoint cognition substrates.**

## The migration ledger

| commit / PR | repo | what arrived | what did not |
|---|---|---|---|
| **#1112** merged `59980e17`, 2026-08-31 14:00, +2027/−1, 7 files | lance-graph | `alpha.rs` (938), `alpha_tunnel.rs` (402), `rung_schedule.rs` (372), `rung_horizon.rs` (213), `recipe_dispatch.rs` (+99) — each headed `⚠ MIGRATED FROM medcare-rs/crates/medcare-nodesoa/…` | the Arrow/Lance STORAGE half stayed in medcare deliberately (Commitment #4) |
| `d7bb9a98` | lance-graph | `wave_dispatch::{dispatch_thought, touched_indices, WaveDispatchOutcome}` + `spog_tenants::SpogTenants` + the `rung_waves` example | `SpogTenants` arrived with NO caller and still has none |
| `ef32b76` | MedCare-rs | *"M1: Rung stirbt sichtbar in medcare"* — local tunnel + scheduler + mirror deleted; `wave_dispatch.rs` → `frontier_dispatch.rs` = adaptation + ONE call | — |
| `5f63f95` | MedCare-rs | pin advanced to `59980e17`; all mirrors deleted | — |

**Board hygiene gap:** `PR_ARC_INVENTORY.md` (6383 lines, newest #1195) has NO
entry for #1112 or `d7bb9a98`; `AGENT_LOG.md` carries nothing on alpha or
`wave_dispatch`. The only board trace is one incidental clause inside a
different epiphany (`EPIPHANIES.md:1586`). Against this workspace's own
Mandatory Board-Hygiene Rule, the migration that made rung/alpha the substrate
default is unrecorded.

## Migration plan status (MedCare `docs/MIGRATION_PLAN_AGNOSTIC_THINKING_2026-08-31.md`)

Waves M0–M5, each with falsifier, disable run, gates and a one-commit rollback.
Its mechanical test is worth keeping: *would this line still make sense if the
consumer were a chess repo?* Yes means lance-graph, no means medcare.

| wave | status |
|---|---|
| M0 upstream substrate | **LANDED** |
| M1 rung dies visibly in medcare | **LANDED**, with an honest acceptance note: the word-boundary grep is NOT empty, 4 hits remain in `alpha.rs`'s tests |
| M2 octopus re-scope onto `spog_tenants` | **NOT LANDED** — zero grep hits |
| M3 first-thought walk audit | **NOT LANDED** — zero grep hits |
| M4 views as masks (`AlphaMask`) | **NOT LANDED** — zero grep hits |
| M5 loco atoms | GATED on an operator mint |

## The rung 0..9 schedule, and why it is flat

`LEVELS = 10`, lane 0 = "before any derivation", 34 NARS recipes over rungs
1..=9. `schedule(initial_known)` admits a recipe when
`requires().covered_by(known)`, then grows `known |= writes` per wave;
unreached recipes are NAMED in `Schedule::unreachable`.

**Pinned FLAT in the migrated copy** — `contract/src/rung_schedule.rs:321`,
`der_plan_ist_flach_weil_kein_rezept_fuer_ein_anderes_produziert`,
`assert_eq!(s.depth(), 1)` over four seed masks including EMPTY and 0xFF.

The reason, in the test's own words: `writes()` is a READ-MODIFY-WRITE
vocabulary, not a producer/consumer vocabulary — a recipe changes a field of
the `ThoughtCtx`, it does not make that field available to another. And:
*"keine Menge Schlussfolgern erzeugt eine Messung"* (no amount of inference
produces a measurement). Corroborated in the consumer:
`frontier_dispatch.rs:134` pins `waves == 1`.

## `seed.clone()` — RULED #2 INTENTIONAL, high confidence

`contract/src/wave_dispatch.rs:73`. Option 3 (half-migrated alpha path) is
FALSIFIED on four axes:

1. `git log --all -S"&mut seed"` returns EMPTY across the ENTIRE history of
   BOTH repos.
2. The direct ancestor already had the identical shape — MedCare `e789f37`'s
   `dispatch_frontier` called `thought_ctx_from(frontier)` per lane, i.e. it
   REBUILT the seed. `seed.clone()` replaced a CONSTRUCTOR CALL, not a
   write-back.
3. There is no earlier form to have degraded from: `e789f37` is the first
   dispatcher ever written and it already did not carry.
4. The justification travelled WITH the code — the "readonly base + owned
   microcopies" paragraph appears in the MedCare ancestor and survives
   verbatim-in-English at `wave_dispatch.rs:17`.

Positive case: it IS `ndarray/.claude/rules/data-flow.md`'s codified law; a
wave is DEFINED as recipes that cannot read each other's writes; and
`run_wave_parallel` requires `F: Fn + Sync` under `thread::scope`, where a
shared `&mut ThoughtCtx` would not compile, breaking the module's own
byte-identical parallel≡sequential falsifier.

**Strongest evidence AGAINST the ruling, recorded:** `recipe_dispatch.rs:292-295`
states as a positive design property that *"later recipes see earlier writes …
the Think-loop semantics, not a defect"*, and the dispatcher denies that
property ACROSS rungs — a rung-9 lane runs its deep recipes at `ctx.rung = 1`
with unspent `free_energy`. What holds the ruling at #2 is that no version of
this code, in either repo, ever did otherwise.

**Two real defects sit NEXT TO the clone and are not the clone:**
- the clone becomes a bug the instant the DAG grows an edge (tripwire: the
  flat-plan test goes red first);
- `ctx.rung` is never set to the lane's rung, so two independent notions of
  "rung" never meet.

## An unrecorded semantic collision

`AlphaStamp.rung` has TWO production writers with incompatible meanings:
- `wave_dispatch.rs` writes `recipe_dispatch::rung(id)` — an attention-ladder
  step 1..=9;
- MedCare `attention.rs:128` writes `domain_rung(classid)` = `Domain::of_classid(..) + 1`
  — a STATIC property of the address, 1..=8.

The reader `reflection()` (`attention.rs:153`) interprets it as DOMAIN. The
repo's own `REASONING_FABRIC_CROSSMAP.md` §6 enumerates three carriers of
"cognitive rung"; `domain_rung` is a fourth, absent from that table. They do not
collide today only because they drive separate overlays.

## The cycle loop is ABSENT

`git grep -E 'cycle \+ 1|cycle\+\+|for cycle in|next_cycle'` over all `.rs` at
`ef32b76^`: **zero hits.** `cycle: u32` is a caller-supplied constant (`7`, `3`
in tests). Nothing advances it. There is no cycle loop anywhere in MedCare's
history.
