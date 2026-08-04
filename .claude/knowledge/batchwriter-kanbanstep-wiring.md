# BatchWriter → KanbanStep — what is wired, what is hand-rolled, what is left

> **READ BY:** any session wiring a consumer onto the post-#879 write path
> (`batch_writer` / `owner_adapter` / `KanbanMove` / `try_advance_phase`), and
> **mandatory** before writing a harness that claims to run "on the substrate".
> Born 2026-08-04 from Arm BLW, where a harness was built that touched **none**
> of this and therefore could not be evidence for any substrate claim.
>
> **Every "unwired" claim below was verified by reading the source named beside
> it, not inferred from a name.** Where a module's own doc-comment states the
> gap, that is cited — those comments are the authority and they are honest.

---

## 0. The one-paragraph answer

The write path is **ahead-firing**: a thought announces where it intends the
mailbox to go, casts that intent *before* the write lands, and resumes
immediately. The lifecycle **step** is applied *after* Lance accepts the write.
The pre-write half is **built and tested**. The post-write half — the seam that
actually applies the move — **is not built**, and `BatchWriter::cast()` has
**zero production call sites** today.

---

## 1. The chain, end to end

```
  StyleStrategy                        (plan time)
    └─ StrategyOutcome::intended_move  = BOOTSTRAP SENTINEL
                                         mailbox 0, witness_chain_position 0
                                         — surfaced, NEVER emitted
                 │
                 ▼
  owner_adapter::emit_bootstrap_intent(outcome, owner, owner_cycle, &mut writer, payload)
    ├─ rebind_bootstrap(mv, owner, owner_cycle) -> Option<KanbanMove>
    │    · 0 → live owner, 0 → live cycle
    │    · every other field (from/to/exec) preserved BIT-FOR-BIT
    │    · returns None if the move ALREADY names a live owner  ← no-theft guard
    └─ BatchWriter::cast(owner, vec![rebound], payload) -> CastId
                 │                                    ▲
                 │                        AHEAD of the write, on purpose
                 ▼
  BatchWriter (staging, ephemeral — NOT a WAL)
    · board: BTreeMap<CastId, (MailboxId, Vec<KanbanMove>)>   (cast order for free)
    · resolve_owner(on_behalf, resolver) -> (owner, was_cache_hit)   W1c delegation cache
    · drain_pending_payloads() -> impl Iterator<Item = (CastId, P)>  eager drain
                 │
                 ▼
  sink  · P is a DESCRIPTOR (mailbox, dirty row-range, cycle) — never owned bytes
        · reads the LIVE backing store at flush via NodeRowPacket::as_le_bytes
        · one physical flush coalesces all earlier intents for a row
                 │
                 ▼
  Lance accepts the write → a successful LanceVersion
                 │
                 ▼
  ✗ THE GAP ✗   the version-completion seam: apply THE PAIRED MOVE
                 MailboxSoaOwner::try_advance_phase(to)
                   · checks KanbanColumn::can_transition_to (the Rubicon DAG)
                   · Ok(KanbanMove) on a legal edge, Err(RubiconTransitionError) on an
                     illegal one — NO mutation on error
```

**"No successful write ⇒ no applied step."** (`owner_adapter.rs` module doc.)

---

## 2. What is BUILT and TESTED

| surface | file | what it gives you |
|---|---|---|
| `BatchWriter<P>` | `lance-graph-planner/src/batch_writer.rs` | `cast` / `casts` / `intent_moves` / `on_behalf_of` / `resolve_owner` / `drain_pending_payloads`. 4 unit tests. |
| `rebind_bootstrap`, `emit_bootstrap_intent` | `lance-graph-planner/src/owner_adapter.rs` | the pre-write cast half, incl. the **no-theft** guard. 5 unit tests, incl. anti-vacuity (asserts the sentinel fields *actually changed*, not merely `is_some`). |
| `MailboxSoaOwner::{advance_phase, try_advance_phase}` | `lance-graph-contract/src/soa_view.rs:295-322` | the SOLE mutation surface. `try_advance_phase` is the checked one and should be preferred — an illegal edge becomes a typed error rather than silent corruption. |
| ~~`VersionScheduler::on_version`, `NextPhaseScheduler`~~ | `lance-graph-contract/src/scheduler.rs:46-95` | ⊘ **BELONGS TO A DIFFERENT ARM — see the correction below. Not part of this write path.** |

> **⊘ CORRECTION (2026-08-04, operator-challenged: "what did you zombie a
> scheduler from — we have batchwriter, kanbanstep, thinking").** Fair
> challenge; I measured it rather than defending it.
>
> **It is NOT a zombie.** `VersionScheduler` / `NextPhaseScheduler` has real
> production consumers in eight crates, not merely its own definition:
> `lance-graph/src/graph/scheduler.rs` (16 refs),
> `lance-graph-supervisor/src/kanban_actor.rs` (16), `symbiont/src/kanban_loop.rs`
> (14), `surreal_container/src/view.rs` (8),
> `cognitive-shader-driver/src/mailbox_soa.rs` (7),
> `lance-graph-planner/src/elevation/cycle.rs` (6).
>
> **But it does not belong in THIS table**, because it is a different arm with a
> different trigger. Its own doc (`scheduler.rs:42-45`) says so: a
> `VersionScheduler` is *"what a `surreal_container` `LIVE` query (or the
> callcenter `LanceVersionWatcher`) calls per `versions()` tick"*. That is the
> **version-tick / LIVE-query arm** — something outside observes a new dataset
> version and asks "should this mailbox advance?". The batchwriter path is the
> opposite direction: a thought *announces* where it intends to go, casts that
> intent, and the paired move is applied after the write lands.
>
> **Where I picked it up, and why that was not good enough.**
> `batch_writer.rs`'s module doc (lines 41-43) states: *"The kanban advance is
> the in-stream synchronous kanbanstep (`VersionScheduler::on_version →
> try_advance_phase(&mut)`), fired inline by whoever already holds the
> version."* I took that at face value. **The code disagrees with it:**
> `persist_sink::recover_and_apply` applies `slot.paired_move` via
> `try_advance_phase` and never consults a scheduler. When a doc-comment and the
> function that actually runs disagree, the function wins — and I propagated the
> comment into this table instead of checking.
>
> **The resulting incoherence was visible inside this very document:** §2 listed
> the scheduler as part of the write path while §4 warned never to let the
> scheduler drive the write path. Both cannot be right. §4 is the correct one.
>
> **The write path is: thinking → cast (`BatchWriter`) → write → paired move
> applied (`try_advance_phase`).** No scheduler in it. The scheduler is
> legitimate, live, and belongs to the tick-driven arm; it is a *contrast* here,
> not a component — which is exactly the role `blw_tenant.rs`'s `PROBE-TRAP`
> gives it ("scheduler would have said Commit; the cast said Plan; Plan was
> applied").
| `KanbanColumn`, `KanbanMove`, `ExecTarget` | `lance-graph-contract/src/kanban.rs` | the shipped lifecycle types. **Do not mint a parallel `KanbanMove`** — `batch_writer`'s own doc says so. |

Live `advance_phase` implementors (i.e. real owners, not test fakes):
`cognitive-shader-driver/src/mailbox_soa.rs:953`, `symbiont/src/kanban_loop.rs:180`,
`onebrc-probe/src/lane_e.rs:114`. Test fakes exist in `persist_sink.rs`,
`soa_view.rs`, `kanban_actor.rs`, `cycle_driver.rs` — do not mistake those for
production wiring.

---

## 3. What is NOT wired (verified, with the source that says so)

| gap | evidence |
|---|---|
| **`BatchWriter::cast()` has ZERO production call sites** | `batch_writer.rs` module doc, "STATUS: DECLARED", verified 2026-07-27 |
| ~~**The post-write apply seam does not exist**~~ | ⊘ **WRONG — see the correction below.** |
| **`deinterlace` has no production caller** | `batch_writer.rs` doc: all call sites are in `temporal.rs`'s own `#[cfg(test)]` module |
| **No production `DeinterlaceRow` implementor** | same doc; the trait is at `temporal.rs:318` |

> **⊘ CORRECTION (2026-08-04, hours after this doc was written — the row above
> was mine and it was wrong).** I built the "does not exist" claim from
> `owner_adapter.rs`'s statement that it "owns only the pre-write cast half"
> and inferred the other half was unbuilt. **It is built.** Found by the
> D-BLW-1 agent reading the source I had not; re-verified by me line by line
> before writing this note.
>
> **`persist_sink::recover_and_apply` (`persist_sink.rs:396`) IS the
> version-completion applier.** It walks sealed landings in canonical stream
> order, filters to `ls.slot.owner == owner.mailbox_id()`, skips anything at or
> below the `applied_through` watermark, and for a landing carrying
> `Some(paired_move)` it applies **that move** via
> `owner.try_advance_phase(mv.to)` (`:430`) behind two guards:
> `PersistError::OwnerMismatch` when `mv.mailbox != me` (`:412`) and
> `PersistError::StalePhase` when `mv.from != owner.phase()` (`:421`). A landing
> with `paired_move: None` only advances the watermark (`:410`).
>
> **It never consults `NextPhaseScheduler`** — so §4's trap is avoided *by the
> shipped function*, not by the discipline of whoever calls it. §4 still stands
> as a warning for anyone writing a NEW applier; it is no longer a live hazard
> in this path.
>
> **What is actually missing is much narrower than "the seam":** a concrete
> `WalSink` (that module's own header says it "builds NO concrete Lance sink")
> and the glue that turns a `cast` into a `SweepSlot`. Both are small next to
> "build the applier", which is what my §5.1 sent a reader off to do.
>
> **The lesson, since it is the same one twice today:** I derived a negative
> from *one* module's self-description instead of reading the module it pointed
> at. A doc saying "X is a separate seam" tells you where X is **not**, never
> whether X exists — exactly the search-boundary defect recorded in
> `E-A-NEGATIVE-EXISTENCE-CLAIM-IS-ONLY-AS-WIDE-AS-ITS-SEARCH-1`, committed by
> me in the very doc that cites it.

Ledger: `.claude/board/TECH_DEBT.md` `TD-DOC-COMMENTS-CLAIM-UNWIRED-BEHAVIOUR`.

**So a doc-comment describing how durability is *observed* is the intended
contract, not a running path.** Read those paragraphs as specification.

---

## 4. The trap, stated because it is one line away from being fallen into

The post-write step must apply **the paired move** — *the one that was cast*.

> "never manufacture a generic `next_phases().first()` transition merely because
> some version appeared" — `owner_adapter.rs` module doc

`NextPhaseScheduler` is *right there* and looks like the thing to call after a
write. Calling it as the post-write applier would fabricate a lifecycle
transition **decoupled from what the thought actually intended**, and every
downstream reading of the kanban column would then be a reading of the
scheduler's default rather than of a decision. `on_version` is for deciding
whether to advance *on a version tick*; it is not the applier for a cast.

---

## 5. What is left to build (D-BLW-1's actual scope)

1. ~~**The version-completion seam.**~~ ⊘ **ALREADY BUILT — see the correction
   in §3.** `persist_sink::recover_and_apply` (`persist_sink.rs:396`) applies the
   paired move via `try_advance_phase` behind `OwnerMismatch` / `StalePhase`
   guards. Do **not** write a second applier. What remains is only: **a concrete
   `WalSink`** (the module builds none) and **the cast → `SweepSlot` glue**.
   There is still deliberately **no confirmation ledger**
   (`E-ACK-ELIMINATED-1`), so durability is a *read* of what Lance holds, never
   a replay from `BatchWriter`.

   **Shape constraint discovered while building D-BLW-1, and it falls out of
   "an owner is a tenant" rather than being pasted on:** with ONE tenant, rows
   cannot each cast a lifecycle move — the second row's move would hit
   `StalePhase`, because the first already advanced the board. So a cycle emits
   N row landings with `paired_move: None` plus **exactly ONE** landing carrying
   the tenant's move. One mailbox = one kanban board, so one step per cycle.
2. **A production `DeinterlaceRow` implementor** + a production caller of
   `deinterlace`, so durability is observed the way the contract says.
3. **A `P` descriptor type** — `(mailbox, dirty row-range, cycle)`. It must stay
   a descriptor: the zero-copy ruling forbids owned delta bytes here.
4. **Then, and only then**, a harness may claim to run "on the substrate".

**The falsifier for step 4 is the iron rule**: evaluating all rows mutates
nothing — snapshot the tenant's backing bytes, evaluate every row, assert
byte-identical, **and** prove the comparison can detect a deliberately
introduced mutation (a guard that cannot bark is the defect one level up).

---

## 6. Hand-rolled vs reused — provenance for Arm BLW's session (2026-08-04)

### Reused, and from where

| thing | source | note |
|---|---|---|
| `stance::{stream, stance_panel, contradiction_ranking, Interner, ReadOut, FlipKind, Provenance, RungLift}` | lifted from `examples/probe_eyes_opened.rs` into `lance_graph_planner::nars::stance` | behaviour-preserving lift; examples cannot be imported, so nothing outside one example could reach the four stances |
| `CausalWitnessFacet`, `Locus` (24), `agreement_count` | `lance-graph-contract/src/causal_witness.rs` | shipped W1–W7; `#[repr(transparent)]` over `[u8; 12]` = 24 × i4 |
| `BeliefArena`, `CStmt`, `TruthValue`, `Stamp` | `lance_graph_planner::nars` | unmodified |
| `jc::stats` (`cohen_kappa`, `phi`, `binary_association`) | `crates/jc` — workspace-EXCLUDED, dev-only | the **independent oracle**; a measure may not be its own oracle. Dev-dep **removed** when the κ harness was deleted; re-add dev-only if the rebuild needs it, and never modify `jc` while using it as the oracle |
| lane corpora + `versification_map.tsv` + lane codebooks | GitHub Release `v0.1.0-codebooks-2026-07-26`, generated by `examples/data/rosetta/*.py` | the fetch followed the shipped `fetch_greek_lane.py` pattern, incl. its verbatim licence gate |

### Hand-rolled, and why that was the right call

| thing | why hand-rolled |
|---|---|
| `deepnsm_v2::corpus` — `split_verses`, `split_verses_detailed`, `CorpusSplit`, `announces_new_testament`, `is_verse_marker` | the **inbound leg owns all text handling** (`E-DEEPNSM-V2-IS-INBOUND-LEG-REASONING-LIVES-IN-LANCE-GRAPH-1`). This replaced an inline copy inside an example that carried an OT truncation for its entire life; moving it into the library is what lets `cargo test` gate it. |
| the four PD lane fetches (Vulgate / Peshitta / Aleppo / Leningrad) | no shipped fetcher covered them; the licence gate was copied from `fetch_greek_lane.py` rather than reinvented |
| the A1/A2/A3 anchors + the three control pairs | hand-picked **from the text** and pre-registered in plan §12.6 *before* an instrument exists, so they grade an instrument instead of being fitted by one |

### Hand-rolled and DELETED as failed attempts

| thing | verdict |
|---|---|
| `blw_bible_lens_wave.rs` (tiled 64 owners) | category error — an owner is a **tenant**, not a shard; it fabricated 63 tenants. Deleted while **green**, because it was green on a fabricated shape. |
| a KJV parser written into `nars/stance.rs` | layer violation — the inbound leg already had one. Reverted. |
| `blw_lens_twin.rs` (the κ instrument) | κ retired as the instrument (§12.3c): it measures *coincidence* and discards what a stance is. Nihilism and sarcasm are both negative, so no sign/boolean separates them. |
| `blw_texture.rs` (the texture instrument) | **measured KILL** (§12.7): used the 24-locus register and wrote **3 loci**, only 1 shared, so `agreement_count` was capped at 1 **before any verse was read**. The carrier changed; the instrument did not. |

**The findings from all four survive** in `.claude/plans/cycle-loop-closure-driver-v1.md`
§12.1a′ / §12.3a′ / §12.3c / §12.7 and in `.claude/board/EPIPHANIES.md`. Only the
dead code went. That is the intended shape: *keep the record, delete the residue.*

---

## 7. The 90-second preflight before claiming "on the substrate"

Run this grep against your harness:

```
batch_writer|BatchWriter|KanbanStep|KanbanMove|kanban|owner_adapter|MailboxSoA|SoaEnvelope
```

**A count of 0 means your harness is a free-standing loop** and cannot support a
substrate claim, however green it is. That grep returned `0` for `blw_texture.rs`,
which is how D-BLW-1 was found to be unbuilt while a harness stood in for it.
