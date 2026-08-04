# PROBE-IGNITION — design note (Opus filigree lane, design-only, no code)

> **Scope:** the falsifier for *"inject corpus X into thinking style Z"* — the
> first DRIVEN traversal of the built-but-undriven write path
> (`batchwriter-kanbanstep-wiring.md` §0: *"the machinery exists and is
> undriven"*). This note decides placement, the ignition grammar's lowering,
> the pre-registered assertion table, and the silence-honesty design. It writes
> no code and proposes no change to `crates/jc`, `temporal.rs`, or
> `persist_sink.rs`.
>
> **Operator ruling folded in (mid-flight, 2026-08-04):** *"I don't want any
> messaging in the common sense, only casting and eventually 'looking into the
> kanban'. In theory it could be as simple as setting a start bit in a kanban
> tenant."* Every message-shaped element of the earlier draft was **rewritten,
> not annotated** — §1b carries the (a)/(b) decision the ruling demands, §2 is
> a scan-and-cast loop with no queue, and §3's G-table gained the
> discovered-by-reading twin.

---

## §0 — Headline: what this probe proves, and what it deliberately cannot

**Proves (can-fire).** A fleet of real `MailboxSoA` tenants, seeded from a real
corpus and armed with a thinking style by a **write**, is discovered by a
**scan of the kanban board alone**; the armed style's own `StrategyOutcome`
(minted by `StyleStrategy::plan`, `style_strategy.rs:270-289`) is **cast**
write-on-behalf through `emit_bootstrap_intent` (`owner_adapter.rs:92`); the
cast is drained, sealed into ONE WAL write / ONE version, and applied
(`run_cycle`, `cycle_driver.rs:446`) — and **only then** does any phase change.
The MUL gate (`shade_owner` → `gate_decision_i4`, `cycle_driver.rs:615` /
`mul.rs:575`) sustains the arc from `CognitiveWork` onward. Ignition is the
first `run_cycle` in the tree with a real owner, a real corpus, and a real
style behind it.

**Proves (can-stay-silent).** The same loop, on the same fleet, **rests** —
and rests for five structurally distinct reasons, each with a named cause:
work exhausted (Hold on a **would-be-Flow qualia**), lifecycle absorbed
(Commit), pruned (Block), unarmed (no style bits), and out of scope (address).
On the final cycle the scan yields zero casts and **no version is sealed at
all** — `wal_writes` does not move. A brain that cannot rest is the 150/150
defect wearing a crown; this probe's rest is *earned by the loop*, not
constructed by the fixture.

**Deliberately cannot (see §5 for the full list).** No durability, no
parallelism, no scale, no timing, no multi-writer, no validity, no
GUID-prefix routing, no 36-style claim, no `deinterlace` claim. It is
**ignition mechanics only**.

---

## §1 — Placement

**File:** `crates/lance-graph-supervisor/tests/probe_ignition.rs`

**Run:** `cargo test -p lance-graph-supervisor --features cycle-driver -- --nocapture`

Reasoning, in the order the constraints bind:

1. `run_cycle` / `collect_casts` / `shade_owner` live in **supervisor**
   (`cycle_driver.rs`, behind `#[cfg(feature = "cycle-driver")]`).
   `StyleStrategy` lives in **planner**. The dep edge is one-way
   (`supervisor → planner`, supervisor `Cargo.toml`: *"Planner does NOT dep
   supervisor — no cycle"*), so **only the supervisor side can see both**. The
   probe cannot live in the planner.
2. The real owner is `cognitive_shader_driver::mailbox_soa::MailboxSoA`, which
   supervisor already carries as a **dev-dependency** (`Cargo.toml`, added for
   `tests/w2b_real_owner_probe.rs`) — available to `tests/` and `examples/`,
   never in the runtime dep graph. No manifest change is needed.
3. `tests/` over `examples/`: an example's `main()` is **never executed by
   `cargo test`** (the codex P2 that forced the Babel probes into the workflow
   by hand). A `#[tokio::test]` runs under the ordinary crate test command.
4. Feature gating follows the established in-repo pattern —
   `w2b_real_owner_probe.rs:19` wraps its whole body in
   `#[cfg(feature = "supervisor")] mod w2b_real_owner_probe { … }`. This probe
   uses `#[cfg(feature = "cycle-driver")]` the same way, so the file compiles
   to nothing without the feature and needs no `required-features` entry.
5. **CI is a gate requirement, not a nicety.** The supervisor CI step passes
   `--features supervisor` only; the entire P4 falsifier suite had therefore
   never run in CI (`E-A-PER-FEATURE-CI-STEP-NAMED-LIKE-PER-CRATE-COVERAGE-1`,
   found 2026-08-04). **This probe MUST land with the `--features
   cycle-driver` invocation added to the workflow in the same PR**, or it
   inherits that blind gate on arrival.

Gate lines print as `probe.ignition.G<n> …` via `eprintln!` — the convention
`cycle_driver.rs:1657`'s `perf.p4f` line already uses. The probe **reports its
counts**; an `expect()` would say only "not found".

---

## §1b — The ignition ruling: two verbs, and which realization

Two verbs exist in this design and no others: **CAST** (a write through
`BatchWriter`, write-on-behalf, `batch_writer.rs:132`) and **LOOK INTO THE
KANBAN** (a read of `MailboxSoaView::phase()` over the scoped owner set). There
is no endpoint, no actor, no RPC, no queue, and no command-shaped method.

### The decision: **(a) — the start bit IS the cast. No new bit is minted.**

`start()::where()` lowers to exactly this, all of it shipped:

| grammar axis | lowers to | shipped surface |
|---|---|---|
| `table($x)` | seed rows: content plane + energy | `MailboxSoA::write_row` (`mailbox_soa.rs:417`), `apply_edges` (`:348`) |
| `ThinkingStyle($z)` | **a write of `MetaWord` bits** into the owner's MetaColumn | `MailboxSoA::set_meta` (`:618`) / `write_row`'s `cell.meta` |
| `where(prefix)` | the **scan's iteration scope** — a `MailboxId` range | read-only over `MailboxFleet::owner` (`cycle_driver.rs:183`) |
| `start()` | **nothing extra.** An armed owner in a non-absorbing column IS started; the next scan finds it | `MailboxSoaView::phase()` |
| `MUL(true)` | `gate_decision_i4` decides every subsequent cycle | `mul.rs:575` → `KanbanColumn::advance_on_gate` (`kanban.rs:146`) |

**Why (a) suffices, including for "armed but not yet cycled".** That state is
expressible without a bit: `where()` scopes the **scan**, not the arming. An
owner can be armed (MetaWord bits written) and sit outside the scanned range
forever — armed, never started. The probe asserts exactly this with the
OUTSIDE cohort (§3 G7): 32 owners armed and firing, byte-identical to the
firing cohort, that never cast because **the address is the only difference**.

**What (a) genuinely cannot do — stated so (b) is not needed on a guess.**
Reading the board alone, (a) cannot distinguish *"ignited, evaluated, and
Holding"* from *"never scanned"*: both show a non-absorbing phase and no cast.
`current_cycle` separates them only once an advance has landed, so an owner
that Held on its very first evaluation is indistinguishable from an owner
nothing ever looked at. **Nothing in PROBE-IGNITION needs that distinction**
(the probe knows its own scope and asserts the scan set directly), so (b) is
**not proposed**.

**If (b) is ever wanted, here is its home and its trap.** The kanban×Rubicon
value tenant (`ValueTenant::Kanban`, `canonical_node.rs`, 8 bytes at value-slab
`[112,120)`, LE `phase(u8) | exec(u8) | reserved(u16) | cycle(u32)`) has **two
reserved bytes at `[2..4]`** — a documented bit there needs **no
`ENVELOPE_LAYOUT_VERSION` bump**, satisfying the envelope-auditor shape. **But
that tenant is PER-NODE**, read via `NodeRow::kanban()`, while the board this
loop drives is **per-mailbox** (`MailboxSoA::phase`, one field, no per-row
kanban column exists on the SoA at all). Setting a start bit "at the addressed
rows" would therefore create N per-row boards inside one tenant — the same
category error as the deleted tiling harness
(`E-AN-OWNER-IS-A-TENANT-NOT-A-SHARD-1`: an owner is a tenant, not a shard).
**If (b) is ever taken, it must be a per-MAILBOX start state, not a per-row
one**, and the two carriers must be reconciled first. Recorded here as an open
question (§6 Q3), not as a proposal.

### The driver receives nothing (operator point 3, made structural)

The probe's per-cycle input is `scan_board(&fleet, SCOPE_IDS)` where
`SCOPE_IDS` is a **compile-time constant range**, recomputed from nothing and
derived from no previous cycle's output. Three structural consequences the
probe asserts:

- **No carry-over list.** `CognitiveWorkOutcome::held_owners`
  (`cycle_driver.rs:483`) is **discarded every cycle**. A Held owner is
  re-found by the next scan because its phase still shows it in a
  non-absorbing column. This closes the #879 open item *"held_owners
  accumulation becomes the driver's job with a strand falsifier"* in the
  ruling-compliant direction: **there is no accumulation because there is no
  list.** Falsifier: the REST cohort wakes at cycle 4 having been re-found by
  scan alone (G5).
- **The scope slice is byte-identical every cycle** — asserted, so no
  cycle-to-cycle channel can hide in it.
- **The only path from the harness to the loop is owner state**: seeding,
  arming, and the wake are all writes into `MailboxSoA` columns; nothing else
  is passed.

---

## §2 — The ignition sequence, step by step

Every function named is shipped. The probe composes; it mints no type.

### Pre-loop (fixture construction — writes only)

1. **Corpus.** `load_verses(path, limit)` shape from `blw_fusion.rs:478`
   (`index\ttext` TSV, `BLW_KJV_TSV`, default `/tmp/kjv_verses.tsv`). If the
   file is absent, a deterministic synthetic corpus is generated and the
   fixture provenance is **printed**. The corpus is a **non-degeneracy
   fixture, not a semantic instrument** — no assertion in this probe depends on
   the text being scripture. A guard asserts the seeded content planes are
   non-zero and pairwise distinct across a sample, so a degenerate corpus
   cannot make the byte-identity claims trivially true.
2. **Fleet.** `HashMap<MailboxId, MailboxSoA<64>>` — the blanket `MailboxFleet`
   impl (`cycle_driver.rs:190`). Construction `MailboxSoA::new(id, w_slot,
   threshold)` (`mailbox_soa.rs:292`), `set_populated(48)` (`:495`) per the W1c
   discipline, `tick()` (`:399`) to cycle 1 exactly as `blw_fusion.rs:729`.
3. **Seed (`table($x)`).** Per row: `write_row(row, cycle, &WriteCell{ content,
   entity_type, temporal, meta, qualia, .. })` (`:417`). `content` via the
   `encode_plane` bloom shape (`blw_tenant.rs:248-294` provenance).
4. **Arm (`ThinkingStyle($z)`) — a write.** `cell.meta = Some(MetaWord::new(z,
   …))` where `z ∈ {0 unarmed, 1 Analytical, 2 Creative, 3 Reflective}`
   (`cognitive_shader.rs`, `MetaWord::new` / `::thinking`). See §6 Q1 for why
   the vocabulary is three values and not thirty-six.
5. **Qualia — DECLARED, copied from the shipped test fixtures.**
   `flow_qualia()` (`cycle_driver.rs:1669`: warmth=4, groundedness=3,
   coherence=4, valence=2) for every cohort except CONTRA, which gets
   `block_qualia()` (`:1675`: coherence=−3, tension=3). Written via
   `cell.qualia`. **This is not an invented corpus→qualia encoder** — it is the
   same construction `cycle_driver`'s own gate falsifiers use, reused verbatim.
6. **Energize.** `apply_edges(&[(row, CausalEdge64)])` (`:348`) — the shipped
   energy-delivery path (`energy[row] += mantissa/8 * confidence`), which also
   enforces `edge.w_slot() == owner.w_slot`. Energy above `threshold` is what
   makes a row *firing*.

### Per cycle `c = 1..=6`

7. **LOOK INTO THE KANBAN.** `scan_board(&fleet, SCOPE_IDS)` reads
   `owner.phase()` for each id in the constant scope and partitions into
   `planning / cognitive / evaluation / absorbed`, plus `missing` for a scoped
   id with no registered owner. Pure reads through `MailboxFleet::owner`
   (`cycle_driver.rs:183`).
8. **Planning column → the STYLE mints.** Probe-local `column_pass` (see §6 Q2
   — the shipped `cognitive_pass` cannot reach this column):
   - `armed = MetaWord::thinking(owner.meta_at(0))`; `0` ⇒ skip (unarmed).
   - `gate = gate_decision_i4(&owner.qualia_at(0), mantissa_of(owner))`
     (`mul.rs:575`).
   - `target = owner.phase().advance_on_gate(&gate)` (`kanban.rs:146`);
     `None` (Hold) ⇒ **rest, cast nothing**.
   - `out = StyleStrategy.plan(PlanInput{ context: ctx_of(owner, armed), plan:
     None, outcome: None }, &mut Arena::new())` (`style_strategy.rs:270`) →
     `StrategyOutcome{ reliability, intended_move: Planning→CognitiveWork,
     exec: Elixir }`.
   - If `out.intended_move.to != target` (i.e. the gate said Prune), replace
     with `shade_owner(owner, &qualia, mantissa, out.reliability)`
     (`cycle_driver.rs:615`) — the gate's move, still carrying the **style's**
     reliability.
   - `emit_bootstrap_intent(&out, owner.mailbox_id(), owner.current_cycle(),
     &mut writer, payload)` (`owner_adapter.rs:92`) → `rebind_bootstrap`
     (`:68`) binds mailbox 0 → live owner, no-theft guarded → `BatchWriter::cast`
     (`batch_writer.rs:132`).
9. **CognitiveWork column → the SHIPPED seam, the GATE mints.**
   `run_cognitive_work_gated_over(&fleet, &scan.cognitive, &mut writer,
   read_gate)` (`cycle_driver.rs:662`), `read_gate` returning
   `(qualia_at(0), mantissa_of(owner), StyleStrategy::reliability_for(style,
   &ctx), payload)`. `shade_owner` mints (`exec: Native`).
   **`held_owners` is discarded.**
10. **Evaluation column → probe-local `column_pass`** (same shape as step 8;
    the style's `intended_move` cannot express this edge, so the gate mints).
11. **REST BRANCH.** If `writer` staged **zero** casts this cycle: record the
    cycle as a rest, **do not seal**, do not call `run_cycle`. No version, no
    WAL write. (`persist_cycle` has no empty-batch guard — an empty cycle would
    still commit and burn a version, i.e. a heartbeat. Not resting is a write;
    resting must be no write.)
12. **Otherwise `run_cycle(&sink, &mut fleet, &mut writer, CycleFrame::new(
    CycleId(c), base), position_base, &mut watermarks, |_| 0u64)`**
    (`cycle_driver.rs:446`) — which is `collect_casts` (`:220`) →
    `seal_cycle` (`:280`) → `persist_cycle` → `apply_sealed_transitions`
    (`:338`) → `MailboxSoaOwner::try_advance_phase` (`soa_view.rs:295-322`).
    `position_base = max(prev_base, sealed.next_position_base)` — the
    restart-stable contract (`collect_casts` doc, `:203-210`).
13. **Write-back pass (`&mut`, after apply — never during compute).** For each
    owner in `applied.applied`: `consume_firing(row)` on ONE firing row
    (`mailbox_soa.rs:380` — stamps `last_active_cycle`, resets `energy[row]`,
    same-cycle idempotency guarded). This is what makes `mantissa` fall.
14. **Scheduled wake (cycle 4 only).** `apply_edges` re-energizes one row of
    each REST-cohort owner. **A write, not a message** — and the same verb the
    initial energizing used.

**Sink.** `MemWal` copied with provenance from `blw_fusion.rs:396-475` /
`blw_tenant.rs:405-501` (`Mutex<Vec<SealedCycle>>`, `AtomicU64` version, base
fence, `wal_writes` counter). Contract only; **not durability**.
**Payload `P = Vec<u8>`** (forced: `run_cycle` takes `BatchWriter<Vec<u8>>` and
`SweepSlot::payload` is `Vec<u8>`) carrying `RowSpanDescriptor{row_lo, row_hi,
cycle}.to_le_bytes()` (`blw_fusion.rs:365-381`) — a **descriptor**, never owned
delta bytes.

### The pinned run shape (PRE-REGISTERED — fixed before any number exists, NOT adjustable after a run)

| constant | value | reason it is this number |
|---|---|---|
| `FLEET_OWNERS` | **64** | Smallest power of two that leaves the where()-excluded set (32) a **majority** of the fleet while keeping 64 real `MailboxSoA<64>` allocations bounded: 3 identity planes × 64 rows × 256 words × 8 B = **384 KB/owner ≈ 24 MB**, lazily mapped. NOT 64k: the sparse-vs-fleet property is already proven at 64k over `FakeOwner` (`cycle_driver.rs:1098`); this probe re-anchors the **driven** loop over the real owner. |
| `ROWS_PER_OWNER` (`N`) | **64** | Capacity. |
| `POPULATED_ROWS` | **48** | `< N`, so the `n_rows()`-vs-capacity distinction is live and zero-padding rows are never read (the W1c phantom-row discipline, `mailbox_soa.rs:852-864`). |
| `CORPUS_VERSES` | **3072** | `64 × 48` exactly — every owner gets a full slice; no owner is a short tail. |
| `SCOPE` | **`0..32`** | Half the fleet. The complement is armed, firing, and identical — so the where() axis has a non-trivial exclusion set (32), not a token one. |
| `CYCLES` | **6** | The minimum exhibiting all five cohort behaviours: 3 cycles complete the Flow arc `Planning→CognitiveWork→Evaluation→Commit` (exactly 3 DAG edges, `kanban.rs:101-107`), +1 to observe an **earned** Hold at a non-absorbing column, +1 for the wake write to take effect, +1 for the woken owner to advance again. |
| `FIRING_ROWS` | **3** (IGNITE/CONTRA/UNARMED/OUTSIDE), **1** (REST) | 3 = one per advance across the 3-edge arc; 1 = exhausts after the first advance, which is what produces the Hold that is not death. |
| `CONSUME_PER_ADVANCE` | **1** | Ties work consumption to lifecycle steps 1:1 so the mantissa's decay is legible. |
| `WAKE_CYCLE` | **4** | The first cycle after two full resting cycles (2 and 3) — two, so "rests" is not a single-sample claim. |

### Cohorts (all inside `FLEET_OWNERS = 64`; in-scope ids sum to exactly 32)

| cohort | ids | armed | firing | expected arc |
|---|---|---|---|---|
| `IGNITE_A` | 0..6 | Analytical | 3 | c1 →CognitiveWork, c2 →Evaluation, c3 →**Commit**; c4-6 absorbed, silent |
| `IGNITE_C` | 6..12 | Creative | 3 | identical arc, **different reliability** (G2iii) |
| `REST` | 12..20 | Analytical | 1 | c1 →CognitiveWork, exhausts; **c2,c3 Hold**; wake at c4 →Evaluation; c5,c6 Hold |
| `CONTRA` | 20..24 | Analytical | 3 | c1 Block → **Prune** (absorbing); c2-6 silent |
| `UNARMED` | 24..31 | **none (bits 0)** | 3 | never planned; **zero casts, byte-identical, 6 cycles** |
| `ORPHAN` | 31 | — | — | in scope, **no owner registered** (the #879 missing-owner caveat, G10) |
| `OUTSIDE` | 32..64 | Analytical | 3 | constructed byte-for-byte like `IGNITE_A`; **zero casts** — only the address differs |

Cycle-1 sparse set = `IGNITE(12) + REST(8) + CONTRA(4) = 24` advanced,
**40 untouched**, and every untouched owner has a **named** cause: 32
out-of-scope, 7 unarmed, 1 orphan. (The `kept*3 < total` filter form is **not**
claimed here — 24×3 > 64. The honest anti-vacuity is the exact decomposition:
`untouched == 40`, `untouched > advanced`, and each subset accounted for.)

---

## §3 — The pre-registered assertion table

Every row has both halves. "Can-fire" and "can-stay-silent" inputs are
**non-trivial on both sides** — no empty-input silences.

| id | assertion | can-fire input | can-stay-silent input |
|---|---|---|---|
| **G1** | Ignition advances exactly the scanned-armed-and-Flowing set | c1: 24 owners advance; `wal_writes == 1`; `sealed.transitions.len() == 24` | c1: 40 owners byte-identical (phase + `current_cycle` + `energy` + `meta` + `qualia` + `content_row`), decomposed 32/7/1 |
| **G2a** | The cast is a shipped minter's, never the harness's | compile-time self-scan (`include_str!("probe_ignition.rs")`) asserts the source contains **no `KanbanMove {` struct literal** | the same scan asserts the source **does** contain `emit_bootstrap_intent` — a scan that finds nothing is not evidence |
| **G2b** | `Planning` casts are the STYLE's, later casts are the GATE's | every sealed move with `from == Planning` has `exec == Elixir` (`StyleStrategy::intended_move`'s signature, `style_strategy.rs:391-399`) and `to == CognitiveWork` | every sealed move with `from ∈ {CognitiveWork, Evaluation}` has `exec == Native` (`shade_owner`'s, `cycle_driver.rs:632`) — the discriminator is not constant |
| **G2c** | The armed bits reached the plan and changed something | `reliability(IGNITE_A) != reliability(IGNITE_C)` bit-for-bit (the R-GATE property, `style_strategy.rs:486-508`) | two owners armed with the SAME style produce **bit-identical** reliability — a random or style-blind reliability fails one half or the other |
| **G3a** | Casting mutates nothing | snapshot `(phase, current_cycle)` fleet-wide before the passes; assert unchanged **after** all casts are staged and **before** the seal | on a rest cycle the same snapshot is unchanged across the whole cycle |
| **G3b** | Phases advance ONLY via seal→apply | self-scan asserts the source contains no `.advance_phase(` and no `.try_advance_phase(` — the probe has no path to mutate a phase | after apply, the changed set is **exactly** `sealed.transitions`' owners; `sink.reads() == 0` during apply (P4b reads no dataset) |
| **G4** | **The gate discriminates on ONE axis over identical, non-trivial qualia** | REST owner at c1: `qualia == flow_qualia()`, `mantissa == 1` ⇒ `Flow` ⇒ casts | REST owner at c2: `qualia` **byte-identical to c1**, `mantissa == 0` ⇒ `Hold` ⇒ casts nothing. Anti-rig asserts: `qualia != QualiaI4_16D::ZERO`, `trust_texture_i4(qualia) == Calibrated`, and `warmth + groundedness − tension == 7 ≥ 4` — i.e. **a would-be-Flow qualia that nonetheless rests** |
| **G5** | Rest is a reschedule; Prune is not | REST is re-found by the scan at c2 and c3 (`scan.cognitive` contains all 8) and **advances at c4** after the wake write | CONTRA appears in **no** scan set after c1 (absorbing) and never casts again. `rediscovered(REST) == 8`, `rediscovered(CONTRA) == 0` — a scan returning everything or nothing fails |
| **G6** | The fleet can rest completely | c1 seals: 24 casts, `wal_writes` 0→1 | c6: scan yields **zero** casts, **no seal happens**, `wal_writes` unchanged from c5, and every owner is byte-identical to its post-c5 state. Same code path, different board state |
| **G7** | The `where()` axis is load-bearing | epilogue (after all other assertions): widen the scope by one OUTSIDE id, run the Planning pass onto a **throwaway** writer, assert it stages a cast | main run: OUTSIDE (32) is byte-for-byte equal to `IGNITE_A`'s construction (`energy`, `content_row`, `meta_at`, `qualia_at`) and casts **zero** times across 6 cycles |
| **G8** | The style-arming axis is load-bearing | epilogue: write non-zero thinking bits into one UNARMED owner, run the Planning pass onto a throwaway writer, assert it stages a cast | main run: UNARMED (7) casts zero times and stays byte-identical, with a corpus identical to IGNITE's |
| **G9** | #879 OPEN — the drained-writer retry footgun is **observable** | side fixture (2 owners, own `MemWal`): inject one WAL failure ⇒ `CycleError::Seal`, `failure.casts` is the byte-identical frozen set, no owner mutated; retry via `seal_cycle(sink, failure.frame, failure.casts)` lands it | on the SAME writer, a fresh `collect_casts` yields **zero** slots — the footgun made visible: a naive `run_cycle` retry would seal an empty cycle and silently "succeed". Comment pins this as the falsifier a future guard must flip |
| **G10** | #879 OPEN — the silently-skipped missing owner | the probe-local `column_pass` **counts** ORPHAN (`missing == 1`) | the shipped `run_cognitive_work_gated_over` handed the same scope list reports **neither** a cast nor a held owner for it, and `CognitiveWorkOutcome` has no field that could. Assert the two passes differ by **exactly** 1. Comment: when the upstream counter lands, this becomes `missing == 1` on both |
| **G11** | Ordering is not a write-side concern | `sealed.transitions` is sorted by `stream_position` and `position_base` is monotone across cycles incl. the skipped rest cycles | no confirmation ledger exists: self-scan asserts the source contains no `ack`/`confirm` identifier (`E-ACK-ELIMINATED-1`) |

**Anti-vacuity note on G2a/G3b/G11 (the self-scans).** A compile-time
`include_str!` scan is only as wide as the file. It proves the *probe* did not
fabricate a move or touch a phase; it does not prove that of a helper in
another module. The probe therefore imports only `lance_graph_planner::*`,
`lance_graph_supervisor::cycle_driver::*`, `lance_graph_contract::*`, and
`cognitive_shader_driver::mailbox_soa::*` — all shipped — and the import list
is itself part of the scanned text.

---

## §4 — Silence-honesty: the qualia extractor question, answered

`shade_owner` (`cycle_driver.rs:615`) takes `(qualia, mantissa, reliability)`
from a **caller-supplied extractor** — the honesty ledger's own named gap
(`cycle_driver.rs:57-61`: *"its qualia/mantissa inputs come from a
caller-supplied extractor, NOT from a live MailboxSoA / shader-driver
dispatch"*). That is exactly where a rigged silence would hide, so the design
is explicit about which input is declared and which is derived.

**The split:**

- **`qualia` is DECLARED, and copied from the shipped gate falsifiers'
  own fixtures** — `flow_qualia()` (`cycle_driver.rs:1669`) and
  `block_qualia()` (`:1675`), written into the real `MailboxSoA::qualia`
  column at seed time and **never mutated by the loop**. It is therefore
  provably **constant** for every owner across the whole run.
- **`mantissa` is DERIVED from live owner state**: `min(7, count of populated
  rows with |energy| ≥ threshold) as i8`. Two shipped fields
  (`MailboxSoA::energy`, `::threshold`), no encoder. It falls only because
  `consume_firing` (`:380`) reset an energy cell — a real state change made by
  the shipped consumption primitive.

**Why this makes the silence honest rather than rigged.** Trace the gate
(`mul.rs`, read line by line):

- `flow_state_i4`: `flow_proxy = warmth + groundedness − tension = 4 + 3 − 0 =
  **7**`. With `mantissa > 0` ⇒ `flow_proxy ≥ 4` ⇒ **`Flow`**. With `mantissa ==
  0` ⇒ not Flow (needs `> 0`), not Transition (same), not Anxiety
  (`flow_proxy` is 7, not `≤ −2`, and `mantissa < 0` is false) ⇒ **`Boredom`**.
- `trust_texture_i4`: coherence 4, valence 2, tension 0 ⇒ **`Calibrated`** in
  both cases.
- `gate_decision_i4`: `(Calibrated, Flow)` ⇒ **`Flow`**; `(Calibrated,
  Boredom)` ⇒ the `_ =>` arm ⇒ **`Hold`**.

So the resting owner's gate input is **the shipped test suite's own Flow
fixture, unchanged, at a flow_proxy of 7** — the maximum this fixture family
reaches. The probe asserts that. An all-zeros qualia would also produce Hold
(`flow_proxy 0`, mantissa 0), and that is precisely the trivial silence the
falsifiability rule forbids as the only case; this design's silence is the
opposite — **the gate is looking at a state that says "go" on four of five
channels and rests anyway, because there is no work left**. One axis varies;
it is derived from real state; and the constant axis is asserted non-trivial
rather than merely asserted equal (`E-THE-EQUALITY-PASSED-WHILE-AN-AXIS-WAS-CONSTANT-1`).

**A third, differently-caused silence keeps the detector honest.** CONTRA's
`block_qualia()` drives `Uncertain ⇒ Block ⇒ Prune` — an *absorbing* silence,
not a resting one. G5 asserts the two are distinguishable by the scan
(`rediscovered(REST) == 8` vs `rediscovered(CONTRA) == 0`). A probe that could
not tell rest from death would be measuring nothing.

**The `&mut` discipline.** All gate inputs are read through `&owner`
(`MailboxFleet::owner`); consumption happens in a **separate `&mut` pass after
apply** (step 13). No `&mut self` during computation
(`.claude/rules/borrow-strategy.md`); the mutation is a gated write-back, not a
side effect of the read.

---

## §5 — Not claimed

Printed as a block at the end of the run, in the `blw_fusion.rs` §6 style.

1. **No durability.** `MemWal` is an in-process `Mutex`/`Vec`; its "versions"
   are sequence numbers, **not Lance versions**. `LanceShardSink` does not
   exist (`persistence-cycle-wal-bootstrap-v1.md`).
2. **No parallelism.** The loop is synchronous — #879's own honesty ledger.
   Only D-KIA-A2's pre-registered protocol (median-of-5 after one discarded
   warm-up, ≥2× at ≥4,096 owners with ≥100 µs thought bodies; stay-silent
   within ±10 % on trivial bodies) can convert "parallel" from doctrine to
   measurement. This probe makes no timing measurement of any kind.
3. **No scale claim.** 64 owners. The 64k sparse property is separately proven
   over `FakeOwner` (`cycle_driver.rs:1098`) and is not re-asserted here.
4. **No multi-writer claim.** Single-writer `MemWal`;
   `TD-RECOVERY-HASH-PARTITION-UNCERTIFIED` (`recover_fleet`'s hash partition
   vs `temporal.rs::local_trajectories`) is untouched — the probe adds no
   evidence for or against it, and its in-order log certifies nothing.
5. **No `deinterlace` / temporal claim.** The probe does not read through
   `deinterlace` and implements no `DeinterlaceRow`. D-BLW-3 owns that seam.
6. **No validity claim.** `reliability` is settledness, not ground-truth
   correspondence (`E-RELIABILITY-NOT-VALIDITY`).
7. **No GUID-prefix routing claim.** `where()` is a contiguous `MailboxId`
   range — an honest **stand-in**. `MailboxId` is a bare `u32` with no
   classid/HEEL/HIP/TWIG structure in this fleet. The claim is only *"an
   address-shaped scope excludes owners that would otherwise fire."*
8. **No 36-style claim.** Three styles are reachable (§6 Q1).
9. **No semantic claim about the corpus.** The corpus makes the columns
   non-degenerate; nothing asserted depends on its meaning. Qualia are declared
   fixtures, not encoded from text.
10. **No zero-copy claim.** `SweepSlot::payload` is `Vec<u8>` by the shipped
    signature; the descriptor discipline is honoured (a row span, not delta
    bytes) but no zero-copy property is measured.
11. **No claim that the loop can re-enter `Planning`.** See §6 Q4 — under the
    shipped gate it structurally cannot.
12. **No recovery claim.** `recover_fleet` (`cycle_driver.rs:700`) is not
    exercised; G9's retry is the WAL-failure path only.

---

## §6 — Open questions for the orchestrator

**Q1 — The arming vocabulary is three values, not thirty-six, and nothing
bridges the two surfaces.** `MetaWord::thinking()` is 6 bits (a 36-style
space). `StyleStrategy::resolve_style` (`style_strategy.rs:231-251`) reads
`PlanContext.thinking_style: Option<Vec<f64>>` — a **23D f64 vector** — and by
dominant axis can return only **`Analytical` / `Creative` / `Reflective`**.
There is **no shipped function** mapping `MetaWord` → `ThinkingStyle` or →
the 23D vector. The probe therefore arms with `z ∈ {0,1,2,3}` and maps to the
three vectors `resolve_style` can actually decode, and says so. *Should a
`MetaWord → PlanContext` bridge be a deliverable, and is the 23D vector or the
6-bit field the canonical arming surface?* (This is persona-vs-rung-ladder
territory — `.claude/v3/knowledge/persona-vs-rung-ladder.md` is the mandatory
read before answering.)

**Q2 — The shipped P4c seam can drive exactly ONE of the five kanban
columns.** `cognitive_pass` (`cycle_driver.rs:490`) hard-filters
`if owner.phase() != KanbanColumn::CognitiveWork { continue; }` (`:505`), and
both public entry points (`run_cognitive_work` `:555`,
`run_cognitive_work_over` `:577`) route through it. **Ignition (`Planning`)
and completion (`Evaluation`) have no shipped driver at all** — which is why
`BatchWriter::cast` has no production caller. The probe writes a local
`column_pass(fleet, ids, column, writer, think)`; the natural upstream fix is
the **one-line generalization** (parameterize the column;
`run_cognitive_work` becomes `column_pass(…, CognitiveWork, …)`). The probe
uses the shipped seam for the `CognitiveWork` column specifically and asserts
its local pass never handles a `CognitiveWork` owner, so there is no
divergence risk by construction. *Approve the generalization as a follow-up
deliverable?*

**Q3 — `MailboxSoA::phase` (per-mailbox) vs `ValueTenant::Kanban`
(per-node) are two carriers of the same concept at two granularities**, and
only the first is driven. The operator's "(b) a start bit in the kanban
tenant" lands in the second. `KanbanTenant`'s `reserved: u16` at bytes `[2..4]`
is the layout-bump-free home if it is ever wanted, but a **per-row** start bit
would fabricate N boards inside one tenant
(`E-AN-OWNER-IS-A-TENANT-NOT-A-SHARD-1`). *Should the two carriers be
reconciled — e.g. `MailboxSoA::phase` documented as the authoritative board and
`NodeRow::kanban().phase` as its per-row projection?* Not proposed here.

**Q4 — The shipped gate cannot express "loop back for another round."**
`Evaluation.next_phases() == [Commit, Plan, Prune]` (`kanban.rs:105`) and
`advance_on_gate(Flow)` takes the **first non-Prune** = **`Commit`**
(`kanban.rs:146-152`). `Commit` is absorbing. So a Flowing owner terminates in
three advances and the `Evaluation → Plan → Planning` re-entry edge — which
exists in the DAG and which `blw_fusion.rs` hand-drives by fabricating the
move — is **unreachable through the MUL gate**. Sustained multi-loop cognition
therefore needs either a policy above the gate (new machinery — rejected in a
probe) or a contract change. The probe's arc stops at `Commit` and says so.
*Is `Evaluation → Plan` meant to be gate-selectable, and if so on what
signal?*

**Q5 — The sealed record carries no style identity.** `KanbanMove` has
`{mailbox, from, to, witness_chain_position, exec}`; the style-conditioned
`reliability` lives on `StrategyOutcome` and is **not** cast. A reader of the
sealed log can tell *which minter* produced a transition (`exec`: Elixir =
`StyleStrategy`, Native = `shade_owner`) but **not which thinking style**.
G2c therefore asserts style-provenance at the cast site, not from the log.
*Is that a gap worth closing? No new field is proposed here.*

**Q6 — Corpus hard-requirement.** The probe prefers the `BLW_KJV_TSV` corpus
and falls back to a deterministic synthetic one so it can run in CI, with the
provenance printed and a distinctness guard. *Should the TSV be hard-required
instead (matching `blw_fusion.rs:709-717`'s pre-registration assert), with the
corpus fetched in CI?*

**Q7 — CI gating.** This probe is inert unless the workflow runs
`cargo test -p lance-graph-supervisor --features cycle-driver`. That step does
not exist today. It must land in the same PR.

---

## Read-provenance (what backs this note, and what does not)

**Read in full:** `.claude/knowledge/batchwriter-kanbanstep-wiring.md`;
`crates/lance-graph-supervisor/src/cycle_driver.rs` (1811 lines, two calls);
`crates/lance-graph-planner/src/strategy/style_strategy.rs`;
`crates/lance-graph-planner/src/owner_adapter.rs`;
`.claude/board/exec-runs/dblw3-api-inventory-sonnet.md`;
`.claude/board/AGENT_LOG.md` first 120 lines.

**Read in the specific regions cited (not in full):** `batch_writer.rs`
(module doc + all methods + tests); `persist_sink.rs` (`persist_cycle`,
`SweepSlot`, `CycleFrame` only); `mailbox_soa.rs` (struct fields,
`new`/`write_row`/`apply_edges`/`consume_firing`/`tick`/`set_populated`, the
`MailboxSoaView`/`MailboxSoaOwner` impls, qualia/meta accessors);
`contract/mul.rs` (`gate_decision_i4`, `trust_texture_i4`, `flow_state_i4`);
`contract/kanban.rs` (`next_phases`, `can_transition_to`, `advance_on_gate`);
`contract/canonical_node.rs` (`KanbanTenant` + `NodeRow::kanban`);
`contract/cognitive_shader.rs` (`MetaWord`); `traits.rs` lines 85-205;
`blw_fusion.rs` (module doc, constants, `MemWal`, loaders, the main seal loop);
`.claude/plans/kanban-64k-inverted-awareness-v1.md` §0-§4.

**NOT read — treat any claim depending on these as UNVERIFIED:**
`persist_sink.rs` in full (line numbers for `persist_cycle` / `WalSink` are
cited by name only, not by line — `recover_and_apply:396` and its guards at
`:410/412/421/430` are taken from the knowledge doc's receipts, not re-read);
`temporal.rs` (relied entirely on the Sonnet inventory);
`contract/soa_view.rs` (`try_advance_phase` at `:295-322` taken from the
knowledge doc); `contract/qualia.rs` beyond `ZERO`/`get`/`with`;
`kanban_actor.rs`; `blw_tenant.rs` (cited only through `blw_fusion.rs`'s
provenance comments and the Sonnet inventory §C); `crates/jc`.

**Not run:** no cargo command of any kind was executed by this lane.
