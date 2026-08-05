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
The pre-write half is **built and tested**. ~~The post-write half — the seam
that actually applies the move — **is not built**~~, and `BatchWriter::cast()`
has **zero production call sites** today.

> **⊘ REGRADED IN PLACE (2026-08-04) — the struck clause was wrong, three
> times over.** The post-write half **is built, end to end**:
> `cast` → `cycle_driver::collect_casts` (reads the intents back, seals ≤1 move
> per owner per cycle as `SweepSlot::paired_move`) →
> `persist_sink::recover_and_apply` → `MailboxSoaOwner::try_advance_phase`.
> Corrections 1–3 in §3 walk each link with line numbers.
>
> **The true statement is the surviving half of the sentence: nothing CALLS
> it.** `BatchWriter::cast` has zero production call sites, and so does
> `collect_casts`. Read this document as *"the machinery exists and is
> undriven"*, never as *"the machinery is missing"* — the two lead to opposite
> next actions, and this doc sent readers toward the wrong one for a day.

---

## 1. The chain, end to end

```text
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
  cycle_driver::collect_casts(writer, cycle, position_base, row_of)
    · drain_pending_payloads() FIRST (ends the &mut borrow)
    · on_behalf_of(cast) -> owner
    · intent_moves(cast)  <- THE READ-BACK of what cast() recorded
    · FIRST move per owner (cast order) -> SweepSlot::paired_move
    · every further move for that owner -> CollectedCasts::held
        -> restage_held() re-casts it NEXT cycle (never dropped)
                 ▼
  persist_sink::recover_and_apply(...)   the version-completion applier
                 apply THE PAIRED MOVE
                 MailboxSoaOwner::try_advance_phase(to)
                   · checks KanbanColumn::can_transition_to (the Rubicon DAG)
                   · Ok(KanbanMove) on a legal edge, Err(RubiconTransitionError) on an
                     illegal one — NO mutation on error
                   · guards: OwnerMismatch (mv.mailbox != me), StalePhase (mv.from != phase)
                 ▼
  ✗ THE GAP ✗   NOT a missing component — a missing CALLER.
                 Every box above is built and tested. Nothing in production
                 invokes cast() or collect_casts(). (Corrected 2026-08-04;
                 the ✗ used to sit on the applier box, which does exist.)
```

**"No successful write ⇒ no applied step."** (`owner_adapter.rs` module doc.)

---

## 2. What is BUILT and TESTED

| surface | file | what it gives you |
|---|---|---|
| `BatchWriter<P>` | `lance-graph-planner/src/batch_writer.rs` | `cast` / `casts` / `intent_moves` / `on_behalf_of` / `resolve_owner` / `drain_pending_payloads`. 4 unit tests. |
| `rebind_bootstrap`, `emit_bootstrap_intent` | `lance-graph-planner/src/owner_adapter.rs` | the pre-write cast half, incl. the **no-theft** guard. 5 unit tests, incl. anti-vacuity (asserts the sentinel fields *actually changed*, not merely `is_some`). |
| `MailboxSoaOwner::{advance_phase, try_advance_phase}` | `lance-graph-contract/src/soa_view.rs:295-322` | the SOLE mutation surface. `try_advance_phase` is the checked one and should be preferred — an illegal edge becomes a typed error rather than silent corruption. |
| `KanbanColumn`, `KanbanMove`, `ExecTarget` | `lance-graph-contract/src/kanban.rs` | the shipped lifecycle types. **Do not mint a parallel `KanbanMove`** — `batch_writer`'s own doc says so. |
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
> and ~~the glue that turns a `cast` into a `SweepSlot`~~ — ⊘ **that glue exists
> too; see the third correction below.** Both are small next to "build the
> applier", which is what my §5.1 sent a reader off to do.
>
> **The lesson, since it is the same one twice today:** I derived a negative
> from *one* module's self-description instead of reading the module it pointed
> at. A doc saying "X is a separate seam" tells you where X is **not**, never
> whether X exists — exactly the search-boundary defect recorded in
> `E-A-NEGATIVE-EXISTENCE-CLAIM-IS-ONLY-AS-WIDE-AS-ITS-SEARCH-1`, committed by
> me in the very doc that cites it.

> **⊘ CORRECTION 3 (2026-08-04, operator-pointed: "what about batchwriter line
> 104").** Line 104 is `BatchWriter::cast(on_behalf, moves, payload)`. The
> pointed question is the right one: **this document never named who reads the
> `moves` argument back.** §2's own correction ended with *"thinking → cast →
> write → paired move applied"*, which leaves `paired_move` arriving from
> nowhere — and a reader who needs one would then hand-roll the pairing, which
> is the "do not mint a parallel `KanbanMove`" failure one level up.
>
> **The reader exists: `cycle_driver::collect_casts`
> (`lance-graph-supervisor/src/cycle_driver.rs:220-256`).** It is the
> `cast → SweepSlot` glue this doc called missing two paragraphs above. Measured:
>
> 1. `drain_pending_payloads()` first (`:227`) — ends the `&mut` borrow so the
>    intents can then be read immutably.
> 2. per drained cast: `writer.on_behalf_of(cast)` → the owner (`:232`).
> 3. `writer.intent_moves(cast)` (`:236`) — **the read-back of line 104's
>    `moves`.** The FIRST move per owner in cast order becomes that owner's
>    `SweepSlot::paired_move` (`:238`, `:251`).
> 4. every further move for an owner already paired this cycle goes to
>    `CollectedCasts::held` (`:243`), and `restage_held` (`:261`) re-casts it
>    into the NEXT cycle — **not dropped, not sealed-then-ignored, not
>    truncated.**
>
> So the full chain is `cast` → `collect_casts` → `SweepSlot::paired_move` →
> `persist_sink::recover_and_apply` → `try_advance_phase`. Every link is built.
>
> **The constraint a consumer must know, which only appears here:** a cast may
> carry any number of moves, but **at most one move per owner per cycle is
> sealed**. Casting three transitions for one mailbox does not perform three
> steps in one cycle — it performs one and defers two. Code that assumes
> otherwise is wrong in a way nothing will report, because the held moves *do*
> eventually apply, just later.
>
> **Still unwired, and this is the honest residue:** `collect_casts` has **no
> production caller** — only `cycle_driver.rs:459` (its own `seal`-side
> companion) and its own test at `:933`. That is consistent with the earlier
> finding that `cycle_driver` itself is unwired; the two facts are not in
> tension, and the distinction is the whole point of this document: **the seam
> is BUILT and the seam is UNDRIVEN.** "Build the glue" was the wrong next
> action; "call it" is the right one.
>
> **Third instance of one defect in one day.** Correction 1: the applier
> "does not exist" (it did). Correction 2: the scheduler is in the write path
> (it is a different arm). Correction 3: the cast→slot glue is missing (it
> exists). All three are negatives I inferred from one module's self-description
> without reading the module it pointed at
> (`E-A-NEGATIVE-EXISTENCE-CLAIM-IS-ONLY-AS-WIDE-AS-ITS-SEARCH-1`). The pattern
> is now strong enough to state as a rule for this document's successors:
> **before writing "X is not built", grep for X's consumers, not for X's
> description.**

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
| a KJV parser written into `nars/stance.rs` | layer violation — the inbound leg already had one. Reverted. **(Clarified 2026-08-05 after an external reader took this row to mean stance.rs itself was reverted:** what was reverted is the duplicate INGESTION parser — Gutenberg/verse splitting, which `deepnsm-v2::corpus` owns. The stance MACHINERY (clause→belief→panel) was deliberately LIFTED from the probe example into `lance_graph_planner::nars::stance` at 4a74d69 and is live — a promotion, not a revert. Two different objects.) |
| `blw_lens_twin.rs` (the κ instrument) | κ retired as the instrument (§12.3c): it measures *coincidence* and discards what a stance is. Nihilism and sarcasm are both negative, so no sign/boolean separates them. |
| `blw_texture.rs` (the texture instrument) | **measured KILL** (§12.7): used the 24-locus register and wrote **3 loci**, only 1 shared, so `agreement_count` was capped at 1 **before any verse was read**. The carrier changed; the instrument did not. |

**The findings from all four survive** in `.claude/plans/cycle-loop-closure-driver-v1.md`
§12.1a′ / §12.3a′ / §12.3c / §12.7 and in `.claude/board/EPIPHANIES.md`. Only the
dead code went. That is the intended shape: *keep the record, delete the residue.*

---

## 7. The 90-second preflight before claiming "on the substrate"

Run this against your harness (the pattern must stay quoted or the shell eats
the alternation), and read the printed count:

```sh
rg -c 'batch_writer|BatchWriter|KanbanStep|KanbanMove|kanban|owner_adapter|MailboxSoA|SoaEnvelope' \
   crates/lance-graph-planner/examples/<your_harness>.rs \
  || { s=$?; [ "$s" -eq 1 ] && echo 0 || echo "rg FAILED (exit $s) — not a count"; }
```

(`rg` exits 1 for genuinely-zero matches and 2 for a real failure — wrong
path, bad pattern. A bare `|| echo 0` would launder a failure into "zero
matches, free-standing harness"; the exit-code split above keeps the two
distinguishable.)

**A count of 0 means your harness is a free-standing loop** and cannot support a
substrate claim, however green it is. That grep returned `0` for `blw_texture.rs`,
which is how D-BLW-1 was found to be unbuilt while a harness stood in for it.

---

## 8. ⊘ OPERATOR RULING (2026-08-04) — the two #879 invariants that must NOT be reversed, and the hash-partition caveat

Verbatim intent, mapped to source the same day:

**Invariant 1 — the batchwriter amortizes ONLY CHANGED.** N casts → ONE WAL
write per cycle; only the sealed sparse set advances; the untouched remainder
is byte-identical (the #879 anti-vacuity falsifier, green at 64k/17). Any
change that widens the write back toward dense/full-image, or adds a per-cast
physical write, reverses #879 and is rejected on sight.

**Invariant 2 — arrival order is never a write-side concern; canonical order
is established by deinterlace BEFORE the seal.** (Rewritten 2026-08-05 — the
earlier one-sentence form read as "read-time only," contradicting §8's
operator-sharpened deinterlace-before-write ruling. Two distinct claims,
both required, never conflated:)

- **(a) Cross-MAILBOX arrival:** the writer fires ahead; no ack exists
  (`E-ACK-ELIMINATED-1`); nothing at the write site synchronizes mailboxes
  against each other. Re-introducing write-side cross-mailbox ordering,
  synchronization, or a confirmation ledger reverses #879 and is rejected
  on sight.
- **(b) Per-MAILBOX canonicalization:** casts never arrive in the same
  order, and the SEAL takes deinterlaced input — the caller canonicalizes
  each mailbox's casts before sealing, via `temporal.rs` (`deinterlace` /
  layer-1 `local_trajectories`, sort key `cast_seq` /
  `(hlc ?? version, version)`) or via the known-order hash helper ONLY once
  certified equally exact on the out-of-order regime (§8;
  `TD-RECOVERY-HASH-PARTITION-UNCERTIFIED`). A caller sealing raw arrival
  order violates (b) without violating (a) — the seal must never ingest
  raw arrival order.

For STORED logs, `temporal.rs` remains the canonical recovery surface at
read time; (b) governs the write path's input, not a new cross-mailbox
synchronization.

**The caveat the ruling names — a hash partition stands where temporal.rs is
preferred.** `cycle_driver::recover_fleet` (P4e, `cycle_driver.rs:700-746`)
partitions the sealed log per owner via
`HashMap<MailboxId, Vec<LandedSlot>>` **in stored order** (`:713-716`), and
`persist_sink::scan_sealed` is EXPLICIT that it does not repair order:
*"in the STORED canonical order — this seam does NOT sort"*
(`persist_sink.rs:315-317`), with the test
`scan_sealed_does_not_repair_order_on_read` (`:711-715`) proving it returns
AS-STORED even for a deliberately out-of-order cycle. The canonical
deinterlacer for exactly this job exists one crate over:
`temporal.rs::local_trajectories` (`:424-434`) partitions AND re-sorts each
owner chain by `cast_seq`, with
`layer1_orders_one_owners_chain_by_cast_seq_not_log_order` proving it repairs
out-of-order storage.

**Status per the ruling: the hash partition is TEMPORARILY ACCEPTABLE, for
performance only.** The two are equally exact **iff** stored order per owner
== `cast_seq` order per owner — which holds today (single-writer `MemWal`
appends in seal order) but is **UNCERTIFIED**, and is exactly the assumption
that breaks under multi-writer / HLC-interleaved / compacted logs — the case
temporal.rs was built for. `temporal.rs` is preferred; the hash path may
remain ONLY once certified equally exact. Certification falsifier defined at
`TD-RECOVERY-HASH-PARTITION-UNCERTIFIED` (TECH_DEBT). Until one of the two
happens (certify or migrate), no new caller may copy `recover_fleet`'s
partition shape — route new recovery reads through layer-1.

> **⊘ §8 AMENDED (operator, 2026-08-04, two directives read together):**
> (1) *"We ALWAYS want 64k thoughts concurrency which never arrive in same
> order, period"* — arrival order is never deterministic, by design. (2)
> *"Before writing they need to be deinterlaced, either by temporal.rs, or
> previously-known-order hash as a helper to be certified."* The deinterlace
> obligation therefore sits **BEFORE the write** — the seal consumes a
> canonically-ordered stream, never raw arrival order — and the hash helper
> is a legitimate known-order FAST PATH, admissible **only once certified
> equally exact against temporal.rs** (the canonical deinterlacer) on the
> out-of-order regime the design actually runs in. Certification against
> in-order-only inputs certifies nothing. Ledger:
> `TD-RECOVERY-HASH-PARTITION-UNCERTIFIED` (sharpened in place, same day).
>
> **Grades, stated explicitly (per knowledge-doc discipline):**
> - "Every link `cast → collect_casts → paired_move → recover_and_apply →
>   try_advance_phase` is BUILT with zero production callers" — **FINDING**.
>   Probe = consumer grep per symbol; run 2026-08-04; result recorded in §2/§3
>   with file:line receipts (`intent_moves` callers, `collect_casts` callers,
>   `shade_owner` callers — each grep listed in the corrections).
> - "Hash-partition apply order == layer-1 apply order under today's
>   single-writer MemWal" — **CONJECTURE** (source-reasoned from
>   `scan_sealed`'s as-stored contract + MemWal's append order; the property
>   test has NOT run). Under the ruling above this conjecture, even if true,
>   certifies nothing — it survives only as the migration's regression
>   falsifier on in-order logs.

## 9. Consumer orientation (operator, 2026-08-04)

- **`deepnsm-v2` is the intended FIRST CONSUMER of this write path.** The
  driver that Addendum-15 (V3 plan) names as the genuinely open item lands as
  deepnsm-v2 consuming `cast()` — not as a free-standing harness promoted to
  production. Sessions scoping "the driver PR" start there.
- **`lance-graph-callcenter` is the blood-brain barrier** between this hot
  path and EXTERNAL consumers, built for the regime where consumers run
  10^4–10^7× slower than the substrate. Current work is hot-path only; the
  BBB membrane is out of scope until an external consumer is, and nothing in
  this document licenses routing a hot-path write through it.

### §9a — what `kanban_actor.rs` IS, then (operator reading, verified 2026-08-04)

**"Consumer-facing: prepare decision, wait for tick."** Verified against the
message surface: `KanbanMsg::MulAdvance` is the atomic PREPARE-DECISION shape
(the MUL gate runs against the owner's CURRENT phase and the transition
applies in the SAME serialized message, so no sender can make the phase read
stale between decision and mutation — `kanban_actor.rs:101-119`, the codex
#578 fix); `KanbanMsg::Tick` is the WAIT-FOR-TICK shape (a substrate version
tick lowers to `next_phases().first()` — the in-actor realization of
`NextPhaseScheduler`'s policy, `:120-130`). That is why the file consumes
`VersionScheduler` and why its one live library consumer drives it via
`drive_version_tick`: it is the **tick-arm's consumer-facing surface**, the
shape a 10^4–10^7×-slower consumer (behind the callcenter BBB) interacts
with. The #879 boundary the file itself now states stays in force: *"a
version tick is global knowledge, never permission to advance"* — so the
tick-driven ADVANCE half is what was demoted, not the prepare-decision
pattern, and no new production architecture depends on the actor.

### §10 — the concurrency doctrine between batchwriter phases (operator, 2026-08-04)

> *"Between the batchwriter phases, each mailbox concurrently needs to decide
> its kanban update or process the already active. Never linear, always ≤64k
> in parallel."*

Between cast and seal, every mailbox is concurrently in exactly one of two
states: **deciding** its next kanban update (the MUL gate: advance / hold /
prune) or **processing** its already-active phase work. The fleet is never a
linear sweep — up to 64k mailboxes run this decide-or-continue choice in
parallel, and their casts arrive in no deterministic order (§8's ruling).
The SEAL stays single-writer and sparse (Invariant 1); parallelism lives in
the thought/decide phase, never in the seal — which restates the kanban-64k
plan's design constraint with the per-mailbox choice made explicit.

**Honesty note, unchanged by the doctrine:** today's `run_cycle` /
`cognitive_pass` iterate owners in a synchronous loop (#879's own honesty
ledger). That linearity is an implementation placeholder inside a correct
ownership model, not the model itself; D-KIA-A2's pre-registered falsifier
(median-of-5, ≥2× at ≥4,096 owners, ±10 % stay-silent) is what converts
"parallel" from doctrine to measurement, and the claim ladder holds until it
runs.

### §9b — the ignition API grammar (operator, 2026-08-04, three messages read together)

> `cognitive-shader-driver::table($x)::ThinkingStyle($z)` — and even the pure
> Friston framing (`free_energy::content()::start()::MUL(true)`) still needs a
> `start()::where()`.

The verb decomposes into four axes, every one already carried by shipped
machinery — the API is composition, not invention:

| axis | verb | lowers to |
|---|---|---|
| WHAT (the brain) | `table($x)` | seed the tenant rows + stream the arena (`seed_tenant` shape; deepnsm-v2 is the first feeder) |
| HOW (the lens) | `ThinkingStyle($z)` | a `MetaWord` bits write to `MetaColumn` (contract-typed; never a new trait) — `StyleStrategy::plan` then mints the bootstrap intents FROM the armed style (`style_strategy.rs:270-289` → `emit_bootstrap_intent`) |
| WHERE (the aim) | `start()::where(prefix)` | **the GUID prefix IS the where** — classid/HEEL/HIP/TWIG prefix routing selects the row range/basin the loop drives; the key was designed to prerender exactly this ("the key prerenders nodes… before ever fetching a value") |
| ARM (the sustainer) | `MUL(true)` | the gate decides Flow/Hold per owner per cycle (`shade_owner` = `gate_decision_i4`); active inference SUSTAINS — it never STARTS. `start()` fires the first `run_cycle`; the MUL decides every subsequent one, and Hold everywhere is the legitimate resting state |

Placement per the BBB/consumer rulings: the API face lives in
cognitive-shader-driver (contract types only at the seams — its planner dep
stays optional); the cast minting stays planner-side; the metronome
supervisor-side. Its falsifier is PROBE-IGNITION, both halves pre-registered:
can-fire — a fresh corpus yields Flow fleet-wide and the STYLE's casts (not
the harness's) advance phases; can-stay-silent — a fully-reconciled corpus
yields Hold everywhere and casts nothing. A brain that cannot rest is the
150/150 defect wearing a crown.

> **⊘ §9b LOWERING CORRECTED (operator ruling, 2026-08-04, same hour):** *"I
> don't want any messaging in the common sense, only casting and eventually
> 'looking into the kanban'. In theory it could be as simple as setting a
> start bit in a kanban tenant."*
>
> The grammar's four axes stand; the LOWERING is not methods, not messages,
> not an endpoint. There are exactly **two verbs** in the entire ignition
> story: **CAST** (write intent through the BatchWriter, write-on-behalf —
> "Melden macht frei") and **LOOK INTO THE KANBAN** (read the board state).
> `start()::where()` = a cast-shaped write that sets the start state in the
> kanban value tenant (`ValueTenant::Kanban`, the per-node 8-byte cursor) at
> the addressed rows — in the simplest honest form, the sealed
> Planning→CognitiveWork intent ITSELF is the start bit, and no new bit is
> minted unless "armed but not yet cycled" is measurably inexpressible that
> way. The driver never receives anything: its input is a SCAN of the board
> it owns — the same shape as the version-tick arm's LIVE reads. Style
> arming is likewise a write (MetaWord bits), never a call.
>
> This SUPERSEDES the earlier custody-free "control-plane endpoint" advice in
> the same session's chat: there is no endpoint. Consumers cast; the board is
> looked into; everything else was ceremony. PROBE-IGNITION's design lane was
> corrected mid-flight and its can-fire assertion gains the twin: *the driver
> discovered the work by reading the board, and nothing else could have told
> it* — no side channel may exist in the probe.


---

## ⊘ Sharpening (2026-08-05): what "zero production callers" means — and what changed

An external review read "zero production callers" as a claim that the
library chain itself is unwired. It is not, and never was: the
library-internal edges exist and are documented above
(`emit_bootstrap_intent` → `BatchWriter::cast`; `run_cycle` →
`collect_casts` → `seal_cycle` → `apply_sealed_transitions`). The claim was
always about the ROOT: **no production runtime invokes that chain.**

Status change (2026-08-05): `tests/probe_ignition.rs` (GREEN, 2/2) now
drives the complete chain — arm → scan → `emit_bootstrap_intent` → `cast` →
`run_cycle` → seal → apply — as a test-rooted driver. The remaining honest
gap is narrower and is stated as such: **no live, externally-rooted runtime
invokes the chain repeatedly over the intended real owner population against
a durable sink.** That is an integration/rooting slice (the deepnsm-v2
consumer direction), not another driver.

Separately: `blw_fusion.rs` predates the probe and produces its sealed
SERIES by calling `persist_cycle`/`recover_and_apply` directly with a
hand-built `SweepSlot` — it does NOT exercise `collect_casts`/`seal_cycle`/
`run_cycle`, and its records never claimed it did (its permitted claims are
the `DeinterlaceRow`/`deinterlace` firsts and the rank-criterion finding).
Rebasing its seal loop onto `run_cycle` is tracked as
`TD-BLW-FUSION-MANUAL-SEAL`.


---

## ⊘ OPERATOR ORDER (2026-08-05): §10's doctrine is THE MAIN MODEL

The decide-or-continue ≤64k doctrine in §10 is not one reading among
several — by operator order it is **the main model**: 64k 1:1 owners,
compile-time mutation-exclusive, independent thought bodies, one
deterministic seal boundary. Any text in this doc or elsewhere that reads
the one-tenant benchmark configuration as the architecture is subordinate
to this order. Canonical entry: `EPIPHANIES.md`
E-64K-1TO1-OWNERS-IS-THE-MAIN-MODEL-1. The outer-level parallelism claim
is gated by D-KIA-A2's pre-registered falsifier; the GREEN probes
(probe_ignition, d_ign_b_lenses) already drive the 1:1 topology, 64 owners,
synchronously.
