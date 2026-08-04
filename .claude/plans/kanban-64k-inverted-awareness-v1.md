# kanban-64k-inverted-awareness v1 — parallel thinking + the inverted-awareness witness

> **Status:** PLANNED / CONJECTURE-graded per section (2026-08-02, main thread)
> **Operator anchors (verbatim):** (a) *"real thinking 64k via kanban as
> orchestration is possible in parallel"*; (b) *"for the private consumer arc we inverted
> awareness — the ontologies are the frozen cathedral the LTM mind is looking
> into, and the subject's STM gets the reflection of the volatile observations
> as a domain subset via rails-shaped RO wiring into ontologies, using all
> catalog criteria as binary ranges … *(anchor paraphrased: private-consumer
> names elided — public-repo separation of concerns)* … the statistics
> correlation is then a witness … the standing wave can sit in observer and
> observed in 2 categories — that allows for measurable Horizontverschmelzung
> of ontologies (Gadamer)."*
> **Companion plans:** `cycle-loop-closure-driver-v1.md` (P4a–P4f SHIPPED,
> PR #879), `epistemic-quadrant-materialization-v1.md` §4c (cross-term rule +
> PROBE-REFLEXIVE-POLICY), `persistence-cycle-wal-bootstrap-v1.md` (§2 sparse
> ruling; LanceShardSink DEFERRED).
> **Review basis:** the R1–R15 review list (this session, 2026-08-02) — each
> wave below names which R-items it settles.

---

## 0. Ground state (verified, with receipts — do not re-derive)

| Fact | Receipt |
|---|---|
| Cycle driver seal/apply shipped; 64k/17 sparse falsifier green incl. anti-vacuity untouched-count, 1 WAL write, 0 dataset reads | PR #879, `lance-graph-supervisor/src/cycle_driver.rs`; second-opinion review confirmed all 8 self-audited findings |
| Execution is a **synchronous loop**; "wait-free" scoped to the cast/cycle boundary only; `MailboxFleet` = blanket impl over `HashMap<MailboxId, O>` | #879 module doc (its own honesty ledger); review §D |
| `KanbanActor` (real ractor, serialized single-writer, S2 MUL gate + S3 version tick + S4 registry delivery) exists and is **not** among `BatchWriter::cast`'s callers | `lance-graph-supervisor/src/kanban_actor.rs`; grep receipt 2026-07-31 |
| `owner_adapter::emit_bootstrap_intent` = the write-on-behalf cast consumer; its one caller (`cycle_driver.rs:516`, `cognitive_pass`) is driven by the **HashMap probe fleet**, not by actors — no ACTOR-OWNED caller exists (#879's own honesty-ledger scope) | `cycle_driver.rs:516`; `write-on-behalf.md` §Interim reality (D-MBX-A6-P3c) |
| Cross-mailbox ordering = `temporal.rs` HLC deinterlace, recovered at READ time — this is *why* ahead-firing needs no ack | operator ruling, `lance-graph-planner/src/temporal.rs` module doc |
| `LanceShardSink` does not exist; durability leg is a test-only in-RAM `FakeWalSink` | #879 review §D; `persistence-cycle-wal-bootstrap-v1.md` status table |
| Rung-3 recipe substrate: 34 NARS recipes catalogued, 29/34 primitive modules shipped; **O1 gap: no rung dispatches to recipes/verbs/StyleFamily yet** | `.claude/v3/knowledge/persona-vs-rung-ladder.md` |
| Consumer-side physical bake does not exist yet (in-RAM buffering sinks only; writer not built); a criteria-catalog/migration drift was found by the private consumer repo's own audit | private consumer board (details stay there) |
| `InferenceType::Synthesis` has **no producer over derived corpus beliefs** — only over Cypher query strings + the cache layer | session task #65 gate-1 correction (SESSION-LOCAL task list, not a GitHub number; category error, self-caught, recorded in the task's metadata) |
| Statistic-as-witness = the zero-copy law's ELEVATED carve-out (cross-input computation of a different KIND), same precedent as `Locus::Quorum` | zero-copy-warden verdict vocabulary; `zero-copy-lens-law.md` |

---

## 1. The two claims, with their unproven words named

**Claim (a)** — *64k parallel thinking via kanban.* Everything is shipped
except the word **"parallel"**: the driver loop is synchronous, the fleet is a
HashMap, and the real actors are unwired. This is the operator-named
"KanbanStep was rewired and the refactor not completed" gap. Arm A closes it.

**Claim (b)** — *inverted awareness.* The inversion: ontology = immutable LTM
(the cathedral the mind looks INTO — read-only, cacheable, public); subject
STM = volatile observations REFLECTED onto the cathedral via rails-shaped
read-only wiring; awareness becomes **measurable** when a view-2 observer
computes cohort statistics over the reflection and those statistics act as a
**witness** (higher-rung derivation, legitimately stored). Fusion of two
ontology horizons (Gadamer) becomes a *measured* quantity. Arms B–D build the
lance-graph side; the private consumer repo consumes (separation of concerns
— its bake, its migrations, its sensitive-data handling stay in its own repo
and PRs, and are never named here).

---

## 2. Arm A — make "parallel" true (settles R1)

**A1 — the actor-fleet driver seam (the incomplete refactor).** Two
structural facts bound this wave (codex P1 on this plan + operator
clarification, 2026-08-03):
(i) `MailboxFleet`'s synchronous `owner()`/`owner_mut()` borrows
(`cycle_driver.rs:183-190`) CANNOT be implemented over the ractor registry —
`where_is` returns an `ActorRef`, and `KanbanActor` deliberately keeps its
owner private behind async messages; holding a second owner to satisfy the
trait would break single-writer. The earlier "implement `MailboxFleet` over
the registry, wiring-only" spec is **withdrawn as structurally impossible**.
(ii) The `HashMap` fleet is NOT a placeholder awaiting actor replacement —
it is the deliberate cheap keyed store whose job is ORDER-FREE access:
cross-mailbox ordering is `temporal.rs`' read-time job (HLC deinterlace,
operator ruling), so the apply side never needs sorted or synchronized
writes. W1 is therefore an Opus DESIGN gate first, choosing between two
seams that both preserve the pre-registered invariant — *one writer per
mailbox-phase state, no second owner, no ack*:
  - **Guarantee-dummy owner** (operator model): ONE supervisor actor owns
    the keyed fleet store — its serialized message loop is the sole
    mutator; the thought phase fans out over read-only owner views, casts
    ahead-fire into the single `BatchWriter`, seal/apply stay
    single-writer inside the owning actor.
  - **Per-mailbox actors** (codex variant): the sealed sparse set is
    applied by delivering each owner's transition through its own mailbox
    (`KanbanMsg::Advance`, the shipped S4 edge).
Either seam lands the first **actor-owned** `emit_bootstrap_intent` caller
(the existing `cognitive_pass` caller is HashMap-fleet-driven).

> **⊘ CORRECTION (operator ruling, 2026-08-04) — A1 above is WRONG as written;
> there is no design gate and no actor seam to choose.**
>
> **#879 is the complete and independent production phase-progression path.**
>
> **KanbanActor has no assigned architectural responsibility. It is legacy
> experimental compatibility code retained only because existing probes or
> consumers still reference it. No new production architecture may depend on
> it. Its presence does not designate it as the future home of an ownership,
> planning-initiation, concurrency, cognition, reasoning, or lifecycle
> mechanism.**
>
> The production path, complete and standalone in #879:
> `plan evaluation → KanbanMove intent → BatchWriter → sparse seal →
> one WAL/version → inline apply`.
> **No actor bridge, actor fleet, actor-owned driver, or actor custody model is
> required.** #879 is not being redesigned by this correction.
>
> 1. Both A1 seams are struck — the per-mailbox `KanbanMsg` apply (the message
>    bus #879's writer-fires-inline ruling already excluded) *and* the
>    guarantee-dummy owner framing, which invented an ownership architecture the
>    ruling does not call for.
> 2. **"First ACTOR-OWNED caller of `emit_bootstrap_intent`" is withdrawn** as a
>    milestone. Corrected W1 ledger (statuses fixed 2026-08-04): **SHIPPED** —
>    a held owner is rescheduled, re-polled, wakes, and advances later (#879's
>    own falsifiers). **OPEN** — protect callers from retrying `run_cycle` with
>    the drained writer instead of retrying `SealFailure.casts`. **OPEN** —
>    surface/count a missing owner in `cognitive_pass` instead of silently
>    skipping.
> 3. `KanbanMsg::{Advance, MulAdvance, Tick}` and the five re-exported driver
>    helpers are marked **LEGACY** in source (disclosure in the first five header
>    lines of `kanban_actor.rs`). Marked, not deleted: `onebrc-probe`'s Lane E is
>    a live consumer via `drive_version_tick`. No runtime behaviour changed.
> 4. **Caller/spawn migration inventory (corrected).** Kept strictly as
>    evidence for why immediate deletion would break current consumers and as
>    the removal work-list — it confers no architectural legitimacy. `KanbanActor` is spawned in three places, none
>    of them the supervisor tree: `kanban_actor.rs`'s own `#[cfg(test)]` tests;
>    `tests/w2b_real_owner_probe.rs` (60/103/144); and
>    `onebrc-probe/src/lane_e.rs:170` — **library source, not a test**. An earlier
>    version of this line claimed every spawn was in one file: a single-file check
>    written up as a repository-wide census, caught by external review. Third
>    absence-claim of this arc to rot; the operational fix is to re-run the search
>    at write-time and keep the command with the claim.
>
> Board: `EPIPHANIES.md` `E-ACTOR-IS-NOT-THE-PHASE-PATH-1`.
- Design constraint: the seal/collect side stays single-writer (one
  `BatchWriter`); parallelism lives in the **thought phase** (owners think
  concurrently, cast ahead-fire), never in the seal. Ordering is already the
  read side's job (HLC deinterlace), so no ack machinery may appear — a
  confirmation ledger anywhere in this arm is an automatic reject
  (`E-KANBANSTEP-IS-THE-TRIGGER-1`).
- Carries #879 review caveats as requirements, not notes: the `run_cycle`
  retry footgun gets a doc-comment + a `debug_assert`-style guard or typestate
  (drained writer must not silently "succeed" a retried cycle); `held_owners`
  accumulation becomes the driver's job with a strand falsifier;
  `cognitive_pass`'s silently-dropped missing owner gets a `missing` counter
  (symmetry with `apply_sealed_transitions`).

**A2 — the parallelism falsifier.** The claim is only honest if measured:
N owners thinking concurrently (tokio joinset over cycle-driver MUL-gated
`CognitiveWork` — NOT the deprecated `MulAdvance` actor arm; corrected
2026-08-04) vs. the same N sequentially, same corpus, same seals.
- **Can-fire:** concurrent wall-clock materially below sequential at 4k+
  owners with non-trivial per-thought work.
- **Stay-silent:** with trivial thought bodies the two must converge (else the
  harness measures its own overhead).
- **Pre-registered measurement protocol** (hand-set a priori per the
  threshold-honesty rule; codex P2 on this plan; NOT adjustable after the
  measured run — a miss is a miss): statistic = median wall-clock over ≥5
  measured runs after 1 discarded warm-up, identical corpus and seals.
  Can-fire = at ≥4,096 owners with per-thought busy-work ≥100 µs, concurrent
  median ≤ ½ × sequential median (≥2× speedup). Stay-silent = with trivial
  thought bodies (<1 µs), medians within ±10 %.
- **Kill condition:** if seal-side contention serializes end-to-end throughput
  regardless of fleet size, claim (a) is regraded to "64k-scale sequential
  sparse cycles" — still true, different claim, board-recorded as such.

**A3 — `LanceShardSink` (real durability).** Stays DEFERRED behind its own
crash falsifiers per the persistence plan. Arm A does not pretend it exists;
A1/A2 run on the WAL contract only. (R6's consumer bake has the same shape on
the private-consumer side.)

## 3. Arm B — the cathedral/reflection contract surface (settles R5, R8; lance-graph side only)

**B1 — catalog binary-range criteria as an L-plane reading.** The rails already
exist (`le-contract.md` L1–L3, `part_of:is_a`); what's missing is the
*criterion* reading: per-criterion `(range, in/out)` as bit-positions over a
facet — content-blind bytes the ClassView projects, per V3 doctrine. No new
key layout, no new tenant until `v3-envelope-auditor` gates it. Deliverable is
the contract type + field-isolation tests, NOT any consumer's data.
- Includes the **catalog-mirror guard** shape (generic: contract criteria set
  ↔ consumer migration must not drift — the drift-audit lesson as a reusable check, so
  the missing-catalog-entries class of bug dies once).

**B2 — RO-wiring direction proof.** The inversion's invariant: observation →
ontology binding is **read-only into the cathedral** — a subject row *points
at* ontology addresses (classid via `HealthcarePort::class_id`, never a local
codebook copy per `ogar-consumer-preflight.md`); nothing ever writes the
ontology. Falsifier: the contract surface offers no `&mut` path from an
observation to an ontology row — checked structurally (API audit), not by
convention.

## 4. Arm C — the statistics witness (settles R2, R3, R4, R7, R9, R13)

**C1 — jc crate audit first** (R7 — one read, no build): what does `jc`
actually provide toward ICC/α/ρ with variance components? Output: a one-page
capability map. Everything below adjusts to what's found.


> **C1 result (2026-08-04, read-only audit).** `jc` is IN-TREE at `crates/jc/`.
> `reliability.rs` ships `pearson`, `spearman`, `cronbach_alpha`, and
> `icc(ratings, IccForm)` with `Icc2_1` (two-way random, absolute agreement)
> and `Icc3_1` (two-way mixed, consistency). There is also a `jirak.rs`, so
> C4's noise-floor requirement has a local implementation to cite rather than a
> paper to hand-derive from.
>
> **Two of C2's renames are the same computation; one is a real gap:**
> - **φ = Pearson on two binary variables** → `pearson` already computes it;
>   the work is *reporting* it as φ plus the marginal-capped ceiling caveat.
> - **KR-20 = Cronbach's α on dichotomous items** → `cronbach_alpha` is the
>   right function; naming + caveat only.
> - **κ is NOT ICC under another name** — a different estimator. No `kappa`,
>   `kr20`, `phi_coef` or `tetrachoric` anywhere in `jc`. **This is the gap**,
>   and D3's fusion falsifier cannot run until it closes.

**C1b — the additive-only jc extension (operator ruling, 2026-08-04).**
Needed: **κ** (agreement), **McDonald's ω** (the coefficient α is routinely
mis-substituted for), and **Effektstärke / effect size** (also assessed
missing). **Hard constraint: ADDITIVE ONLY — do not modify or refactor any
existing `jc` function.** `pearson` / `spearman` / `cronbach_alpha` / `icc`
are load-bearing and stay untouched; new estimators land as new items beside
them. **Any diff that edits an existing `jc` statistic is an automatic
reject, independent of merit.** Blocks D3.

**C2 — name the dichotomous statistics correctly.** Over binary catalog
criteria: Pearson→**φ** (report the marginal-capped ceiling), Cronbach's
α→**KR-20**; **κ is a SEPARATE estimator, not a renamed ICC** — where a
continuous workflow would reach for ICC on binary criteria, compute **κ**
instead, and keep **ICC as ICC** for the non-binary jc escalation only;
Spearman **degenerates and is dropped** at view 2 (it returns only in jc's
non-binary escalation). The implementation and every doc name the dichotomous
forms; reporting "Pearson" while computing φ is the defect class this arm
exists to prevent.

**C3 — reliability vs validity split (hard gate).** α/KR-20/ICC/κ = 
**reliability**, claimable from the cohort alone. **Validity requires an
external criterion** (an external gold-standard criterion, defined on the private consumer board) and
is NOT claimed until one is wired. The plan's public claim ceiling until then:
*"measurable reliability as a first step toward measurable awareness."*

**C4 — Jirak noise floors.** Binary criteria within one catalog panel are
domain-correlated — weak dependence *by construction*, so every
significance statement cites Jirak 2016 rates per `I-NOISE-FLOOR-JIRAK`;
classical IID Berry-Esseen is forbidden here exactly as for fingerprints.

**C5 — witness storage under the ELEVATED carve-out.** The cohort statistic
is a cross-input derivation of a different KIND than any observation → it may
be stored; the ruling names the rung explicitly in the type's doc. i4
quantization (~0.13 resolution over [−1,1]) is sufficient for a *witness*
(tap/signal); the full-precision value lives in jc's output, not the lane.

**C6 — anti-circularity gate.** The witness may gate admission ONLY when
computed on a prior/held-out cohort slice — never the slice it gates (the
M-GATE self-proving-loop lesson from session task #65 — session-local task
list, not a GitHub number — promoted to a rule here).
Falsifier: same cohort, gate on/off, admitted-set must differ only via the
held-out statistic.

## 5. Arm D — measurable Horizontverschmelzung (settles R10, R11, R14; feeds #65)

**D1 — observer/observed as two Locus categories, cheapest formulation
first.** Try expressing view-1 (patient-in-cathedral) and view-2
(cohort-observer) as two `Locus` values over ONE arena, resolved by the
shipped `standing_wave_grounded_lens` — no new machinery. Only if the
bipartite read genuinely cannot be expressed does a structural change get
designed (and then as a ClassView election, not a new layer).

**D2 — reflexivity stays escalate.** Observer-observing-itself is
unrepresentable as routing (offset 0 = unbound) and the shipped policy
escalates. Arm D changes nothing here; any minting proposal routes through
`PROBE-REFLEXIVE-POLICY` first (plan §4c), full stop.

**D3 — the fusion falsifier, middle band pre-registered NOW.** Two ontology
projections of one cohort (e.g. two catalog-derived criteria
views); fusion measured as their **κ-family agreement** (the projections are
binary criteria views, so C2's dichotomous rule applies here too; ICC returns
only if the comparison runs on jc's non-binary escalation):
- **κ ≈ 1.0 ⇒ redundancy** — two names for one horizon, no fusion.
- **κ ≈ 0 ⇒ no shared horizon** — nothing to fuse.
- **Fusion lives in the pre-registered middle band** (band fixed from C1's
  capability map + a pilot slice *before* the measured run; recorded on the
  board before results exist).
- Connection to session task #65 (PROBE-FREE-ENERGY-DESCENT, session-local
  task list — not a GitHub number): a genuine fusion event is Synthesis-*shaped*
  (cross-domain closure). D3's machinery is the corpus-side Synthesis
  producer the M-GATE was missing — landing it un-blocks that task's gate 1
  without the query-string classifier category error.

## 6. Wave order, D-ids, gates

| Wave | D-id | Deliverable | Gate to pass | Model |
|---|---|---|---|---|
| W0 | D-KIA-0 | jc capability map (C1) + dichotomous-statistics decision note (C2 naming) | read-only; note on board | main thread |
| W1 | D-KIA-A1 | ⊘ RESCOPED 2026-08-04 — no design gate, no actor seam choice, no actor-owned `emit_bootstrap_intent` milestone (all withdrawn; actors do not drive). #879 is the canonical phase-progression path and is not redesigned. SHIPPED: held-owner reschedule/wake (#879 falsifiers). OPEN: run_cycle drained-writer retry guard (retry `SealFailure.casts`, not the writer); missing-owner counter in `cognitive_pass` | existing 19 falsifiers stay green; strand falsifier; no-ack audit clean | Opus design → Sonnet impl |
| W2 | D-KIA-A2 | parallelism falsifier (protocol pre-registered in §2 A2: median-of-5, ≥2× at ≥4k owners, ±10 % stay-silent) | can-fire + stay-silent both green, else regrade claim (a) | Opus |
| W3 | D-KIA-B1 | catalog criterion contract type + catalog-mirror drift guard | field-isolation matrix; `v3-envelope-auditor` verdict LAYOUT-CLEAN/GATED | Sonnet impl, Opus gate |
| W4 | D-KIA-C5 | witness type under ELEVATED ruling + C6 held-out gate | zero-copy verdict ELEVATED recorded; anti-circularity falsifier | Opus |
| W5 | D-KIA-D1 | observer/observed two-Locus read | expressible-with-shipped-machinery answer (either way, recorded) | Opus |
| W6 | D-KIA-D3 | fusion falsifier over two projections | middle band pre-registered BEFORE run; result vs band | Opus |

Blocked-on external (not this repo's waves): the consumer physical bake
(private repo, R6), validity criterion selection (C3), `LanceShardSink` (A3).

## 7. Kill conditions (pre-registered)

1. A2 fails both directions → claim (a) regraded, not massaged.
2. jc lacks variance-component machinery (C1) → witness ships as κ/KR-20 only;
   ICC deferred, stated plainly.
3. D3 lands outside the pre-registered band → "no measured fusion at this
   granularity" is the recorded result; the band is not moved post hoc.
4. D1 needs new machinery → it stops and reports; no new layer without the
   ClassView-election route.

## 8. Discipline carried from this session

- Falsifiability rule in full: every gate has a can-fire AND a stay-silent
  half on non-trivial input; thresholds get inertness tests; no doc claim
  without an exercising test or a *claimed, unverified* label.
- Statistics honesty: reliability ≠ validity; dichotomous forms named as
  such; Jirak everywhere significance is claimed.
- No confirmation/ack state anywhere in Arm A (the transition is the event).
- helix API untouched (standing ruling). No model identifier in artifacts.
- Board hygiene in the same commit as any wave landing.

**Honesty ledger:** nothing in this plan is measured yet except the §0 ground
state. Claim (a) is CONJECTURE until W2; claim (b)'s "measurable awareness" is
capped at *reliability* until a validity criterion exists; Horizontverschmelzung
is a designed falsifier, not a finding.
