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
| `VersionScheduler::on_version`, `NextPhaseScheduler` | `lance-graph-contract/src/scheduler.rs:46-95` | decides *whether and how* to advance on a version tick. `NextPhaseScheduler` = forward arc (`next_phases().first()`), stamps the Libet anchor `-550_000 µs` on the `Planning → CognitiveWork` Σ-commit crossing. |
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
| **The post-write apply seam does not exist** | `owner_adapter.rs`: "That post-write application is a SEPARATE seam (the version-completion path)" — the adapter "owns only the pre-write cast half" |
| **`deinterlace` has no production caller** | `batch_writer.rs` doc: all call sites are in `temporal.rs`'s own `#[cfg(test)]` module |
| **No production `DeinterlaceRow` implementor** | same doc; the trait is at `temporal.rs:318` |

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

1. **The version-completion seam.** On a successful `LanceVersion`, look up the
   casts whose writes that version covers, and for each apply its paired
   `KanbanMove` via `try_advance_phase(mv.to)` on the owner. Requires deciding
   how a `LanceVersion` maps back to `CastId`s — note there is deliberately **no
   confirmation ledger** (`E-ACK-ELIMINATED-1`), so this is a *read* of what
   Lance holds, never a replay from `BatchWriter`.
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
