# D-IGN-B — DESIGN (Opus design lane, 2026-08-05)

**Scope:** design only. No code, no cargo, one file (this one). Realizes plan
`.claude/plans/cycle-loop-closure-driver-v1.md` §12.11 on the scaffolding
PROBE-IGNITION left green (`crates/lance-graph-supervisor/tests/probe_ignition.rs`,
1,389 lines, 2/2 tests, G1–G11 both halves — AGENT_LOG 2026-08-05).

Every claim below is anchored at `file:line` or marked **UNVERIFIED**. Nothing
here was compiled or run.

---

## 0. Headline, and the two findings that shape it

D-IGN-B swaps the fixture thought body for the **shipped** four-stance panel
(`lance_graph_planner::nars::stance`) driven by the armed ordinal in
`MetaWord::thinking()`. Two findings constrain that, both verified in source:

**F0 — there is NO per-stance dispatch, and the note must not pretend there
is.** `stance_panel(arena, intern, out)` (`stance.rs:469-478`) returns all four
stances as ONE 4-tuple. There is no stance enum, and no way to compute one
stance alone without refactoring the shipped function. **Consequence, stated in
the required words: arming selects WHAT IS READ, not what is computed.** Every
armed owner runs the same panel over its own arena; the ordinal picks the tuple
element. §12.11's phrase "z=1..4 dispatch to the shared nars stance bodies" is
therefore realizable only as *selection*, not dispatch, and the file's prose,
its gate names, and its printed output must all say selection.

**Does the honest framing kill the deliverable?** No — and this is a judgment,
so it is argued rather than asserted. The pre-registered observable in §12.11 is
"different lenses over byte-identical rows produce non-identical readouts; the
same lens produces bit-identical ones". A selection axis satisfies that
non-vacuously: the four tuple elements are four different types over four
different derivations (`stance.rs:479-532`), so the choice is load-bearing and
falsifiable, and L4 (§4) can still fail. What the framing DOES kill is any
claim about *compute* being lens-dependent — no CPU path differs, cost is
identical across z, and a "the armed bits steer the computation" sentence would
be false. That claim is not in §12.11's observable, so the deliverable stands,
one size smaller than its name suggests. **Recommendation: rename the axis in
the file from "lens dispatch" to "lens selection" so the next reader inherits
the correct size.**

**F1 — the lens cannot read the row bytes; it reads the owner's own verses,
selected by the owner's address.** `stance::stream` takes
`verses: &[(String, String)]` — labelled verse TEXT (`stance.rs:161-167`), and
its whole machine is a token walk over that text (`stance.rs:190-406`). What a
row actually carries is a **bloom plane**: 4 bits per token OR'd into a
`WORDS_PER_FP`-word plane (`probe_ignition.rs:232-257`). That is one-way — no
inverse exists, and none is proposed. So the lens body takes the owner's
`mailbox_id` + `populated()` and re-reads the SAME corpus slice the owner was
seeded from (`probe_ignition.rs:426-429`, `:444-461`). **This must be stated in
the file and in the not-claimed block: D-IGN-B does not decode SoA row bytes
into text.** It is still address-driven cognition — which owner thinks, over
which verses, in which lens, is decided entirely by the owner's id, span and
armed bits — but it is not row-byte decoding.

**F1b — this is the §12.7 defect shape (a harness reading past the substrate),
and it must be named in those words.** §12.7 killed `blw_texture.rs` partly
because its grep count for `batch_writer|BatchWriter|KanbanStep|kanban|
owner_adapter|MailboxSoA|SoaEnvelope` was **0** — "a free-standing loop over a
TSV … therefore cannot be evidence for any substrate claim, only for the stance
functions". D-IGN-B takes its text from the same place such a harness would.

**My judgment: acceptable HERE, on one condition, and the condition is
testable.** In `blw_texture` the substrate governed *nothing* — no owner, no
phase, no cast, no seal. Here it governs **selection end-to-end**: which owners
are in scope (`where()`), which are armed (`meta_at(0).thinking()`), which have
reached `CognitiveWork` (only via an applied, sealed transition), and which
verse span is read (`populated()` + the owner's id). Remove the substrate and
nothing runs; gates L2/L5/L6/L7 each falsify one leg of that. What the
substrate does NOT govern is the *content* of the read.

**The condition:** the file never claims the stance readout is evidence about
the substrate's data path, and the not-claimed block carries F1/F1b verbatim
(item 1 in §6). If a future reader wants "the lens read the rows", the honest
route is an instrument over the encoded planes, not a better sentence about
this one. **If the orchestrator judges that a probe whose cognition reads past
the substrate is a corpus harness wearing a substrate costume regardless of the
selection argument, that is a defensible call and D-IGN-B should be re-scoped
or killed — the above is the strongest honest case, not a guarantee.**

**F2 — z=5 (Fusion) is BLOCKED in-cycle. See §3. That finding is the
deliverable, per the brief; the honest reduction is R2 below.**

---

## 1. (a) Where the lens SELECTION attaches (not dispatch — see F0)

**Decision: `run_cognitive_work_over`, NOT `run_cognitive_work_gated_over`.**

- `run_cognitive_work_gated_over` (`cycle_driver.rs:663-676`) takes
  `read_gate: FnMut(&F::Owner) -> Option<(QualiaI4_16D, i8, f32, Vec<u8>)>` and
  then *fixes* the outcome to `shade_owner(owner, &qualia, mantissa, reliability)`
  (`:672-674`). Its return channel is `(qualia, mantissa, reliability, payload)`
  — there is no slot a lens readout can leave through. This is the closure the
  probe currently uses (`probe_ignition.rs:807-816`); it is the wrong carrier
  for D-IGN-B.
- `run_cognitive_work_over` (`cycle_driver.rs:577-587`) takes the **general**
  seam `think: FnMut(&F::Owner) -> Option<(StrategyOutcome, Vec<u8>)>` and
  forwards to `cognitive_pass` (`:490-531`), which filters to
  `KanbanColumn::CognitiveWork` (`:505-507`) and casts via
  `emit_bootstrap_intent` (`:512-519`). **This is the seam.**

**Shape of the lens closure** (probe-local; `think` is `FnMut`, so it may hold
`&mut` state — that is how the readout escapes without changing any signature):

```
|owner: &Tenant| -> Option<(StrategyOutcome, Vec<u8>)> {
    let z = owner.meta_at(0).thinking();           // the ordinal, read here
    if z == 0 { return None; }                     // unarmed: no lens, no cast
    let readout = run_lens(z, owner, corpus);      // §2 — the shipped panel
    readouts.insert((owner.mailbox_id(), owner.cycle()), readout);   // &mut capture
    let qualia = owner.qualia_at(0);
    let mantissa = mantissa_of(owner);
    let reliability = StyleStrategy::reliability_for(thinking_style_for(z), &ctx);
    let outcome = shade_owner(owner, &qualia, mantissa, reliability)?;
    Some((outcome, row_span_payload(owner)))
}
```

**Why the out-of-band `&mut` capture rather than a second pass — and the
honest cost.** The shipped seam has **no readout slot** in either variant: the
gated closure returns `(QualiaI4_16D, i8, f32, Vec<u8>)` (`:663-676`), the
general one returns `(StrategyOutcome, Vec<u8>)` (`:577-587`). Both are
read-only-owner-in, decision-out. Three options, judged:

1. **`&mut` collector captured by the `FnMut`** (chosen). The lens runs at the
   exact moment the substrate says this owner thinks, so the readout is keyed
   by a phase the harness did not choose. Cost: the readout escapes
   *out-of-band* — it is invisible to the seam's own types, so nothing in
   `cycle_driver` can ever check it. That is a real weakness and it is why L6
   exists (every readout key must be justified by a prior applied
   `to == CognitiveWork` move) — the ordering claim is asserted by the probe,
   not guaranteed by the signature.
2. **A separate probe-local pass** (the `column_pass` precedent,
   `probe_ignition.rs:559-598`). Rejected as the primary: a second pass
   re-derives "who is in CognitiveWork" itself, which is exactly the coupling
   the ignition property exists to demonstrate — the lens would then run because
   the *harness* decided to, one step removed from the seal.
3. **Change the shipped seam to carry a readout.** Rejected outright: it widens
   a shipped signature for a probe's benefit, and `cycle_driver` has production
   callers of both variants.

**Recommendation to the orchestrator:** option 1, with L6 as the compensating
gate, and the file stating in-source that the readout is out-of-band and
therefore probe-asserted rather than type-enforced.

**How the ordinal reaches it:** `owner.meta_at(0).thinking()` — the same read
the probe already performs at `probe_ignition.rs:809` and `:606`. No
MetaWord→PlanContext bridge is introduced; §12.11's Q1 non-goal is preserved
verbatim (persona-vs-rung-ladder §"four spaces" is the mandatory read before
any such bridge, and this design does not open it — the six ordinals are a
probe-local arming vocabulary, **not** the persona-36 and **not** rung-3
runbooks).

**What does NOT change:** the gate, the DAG move, the seal/apply, the
write-back. The lens changes the READOUT only; the transition is still minted
by `shade_owner` (`cycle_driver.rs:615-635`). Gate L5 (§4) pins exactly that.

---

## 2. (b) The readout type

**Decision: mint nothing in any shipped crate. The readout is a probe-local
test type over the four shapes `stance_panel` already returns.**

`stance_panel` (`stance.rs:469-478`) returns a 4-tuple of four *different*
types in one call over one arena:

| z | lens | shipped return shape (`stance.rs:474-477`) |
|---|---|---|
| 1 | Hegel | `Vec<(CStmt, f32)>` — Aufhebung ranking |
| 2 | Nietzsche | `Vec<(CStmt, FlipKind)>` — genealogy partition |
| 3 | Kant | `Vec<(String, f32, f32)>` — (label, graded quale, ablated quale) |
| 4 | Wittgenstein | `Vec<(u16, usize)>` — (concept, distinct games) |

Consequences, both load-bearing:

1. **The panel computes all four; the lens SELECTS one.** That is the honest
   description and must be written as such — the harness does not run four
   different algorithms, it takes the z-th projection of one shipped read over
   one arena. Cost is therefore identical across lenses (see §5).
2. **No shipped type unifies them, and none should be invented.** A probe-local
   `enum LensReadout { Hegel(..), Nietzsche(..), Kant(..), Wittgenstein(..) }`
   plus a `digest(&self) -> u64` (a stable order-preserving fold over the
   variant's contents) is the whole surface. Probe-local test types have
   precedent in the same file (`OwnerFingerprint` `:642-649`, `ScanResult`
   `:518-525`, `RowSpanDescriptor` `:296-311`).

**`ReadOut` evaluated as the readout type — REJECTED, with reasons.**
`stance::ReadOut` (`stance.rs:132-145`) is a shipped struct with Vec fields
(`provenance`, `lifts`, `impls`, `pass2_admitted`, `pass2_revised`) and is the
obvious candidate, so it gets an explicit verdict rather than silence:

1. **It is an INPUT to the panel, not its output.** `stream` fills it
   (`stance.rs:161-167`, `:291-295`, `:346-361`) and `stance_panel` then
   *consumes* it (`:472`, `:500-510`, `:522-525`). Landing `ReadOut` as the
   D-IGN-B observable would report what the parser saw, **identically for every
   z** — the lens axis would vanish and L1 would fail by construction. That
   makes it the single most dangerous wrong choice available here.
2. **It is lens-independent by definition** — one `ReadOut` per owner, four
   stances read from it. Using it as the readout would be the §12.7 error in a
   new costume: an instrument that cannot see the distinction it exists to make.
3. **It is, however, the right thing to keep and print as CONTEXT** — per-owner
   `provenance.len()` / `lifts.len()` / `impls.len()` explain *why* a lens came
   back empty (L3), and Kant's readout is derived from `out.lifts` directly
   (`:500-510`), so an owner with zero lifts has an empty Kant readout for a
   legible reason. **Recommendation: print `ReadOut` cardinalities alongside
   every readout; never use it AS the readout.**

**Explicit anti-decision: do NOT route the readout through
`CausalWitnessFacet`.** That is the carrier §12.7 KILLED — the 24-locus
register into which only 3 loci were ever written, bounding `agreement_count`
at 1 of 24 by construction (plan §12.7, "the register was necessary and is not
sufficient"). Reusing it here would rebuild the same defect one level up. The
four heterogeneous shapes above are *why* the §12.7 collapse does not
mechanically bind D-IGN-B — but that must still be TESTED, not assumed: gate L4.

**Where it lands:** a `HashMap<(MailboxId, u32), LensReadout>` keyed by (owner,
cycle) in the test body. Nothing is written back into the SoA, nothing is cast,
nothing is persisted. A readout is an observation the harness makes, not state
the substrate carries — consistent with §12.5 "a lens is a read".

---

## 3. (c) z=5 Fusion — BLOCKED in-cycle. The honest reduction.

**Verified premises:**

- `blw_fusion` needs the ranking **pool to grow**: "the ranking POOL must grow
  — this is what makes a verdict horizon-dependent at all"
  (`blw_fusion.rs:494`, `seed_slice` seats one incremental slice per cycle
  `:500-528`).
- It needs **many horizons**: `S_CYCLES = 8` (`:123`), `SLICE = 250` (`:126`),
  `V_PIN_CYCLE = 4` (`:131`), and the Δκ table in plan §12.8 is over eight of
  them.
- The two projections are `QueryReference::at(v_pin, RUNG_STRICT=0)` vs
  `at(v_pin, RUNG_AWARE=5)` (`blw_fusion.rs:133-136`, `:983-996`) over
  `VerdictRow`s emitted per (subject, horizon) (`:216-241`, `:900-926`).

**Against the probe's shape:**

- an owner is seeded ONCE, all 48 rows, before the loop
  (`probe_ignition.rs:444-461`) — the pool never grows;
- the measured run seals **once** (c1: `wal_writes == 1`, `:952-956`; c5/c6 are
  rest cycles, `:839-847`) — so an owner has ~1 horizon, not 8;
- with one horizon, Strict and Aware admit the same rows, the folded verdict per
  subject is identical, and any Δ is **0 by construction** — a vacuous readout,
  the exact failure class this workspace's falsifiability rule exists to reject;
- `jc` is a dev-dep of `lance-graph-planner` only (`lance-graph-planner/Cargo.toml:77`);
  `lance-graph-supervisor/Cargo.toml` has **no** `jc` — so κ additionally needs a
  manifest change (orchestrator decision, not a worker's).

**Two reductions; I recommend R2.**

- **R1 (rejected as the default):** change the run shape — seat 8 rows/cycle,
  force every cycle non-resting so each owner accrues ≥4 sealed horizons, then
  report Strict/Aware admitted-row counts. Rejected because it entangles
  D-IGN-B's z=1..4 headline with a run-shape rewrite, and it breaks the
  inherited G4/G6 rest gates the probe pinned.
- **R2 (recommended):** **z=5 is NOT in the main fleet.** Ordinal 5 stays
  *reserved* in the arming vocabulary and is exercised by a SEPARATE
  `#[tokio::test]` in the same file, with **one** owner (an owner is a tenant —
  `E-AN-OWNER-IS-A-TENANT-NOT-A-SHARD-1`), incremental seating, and its own
  MemWal, mirroring `blw_fusion`'s shape at reduced scale. Its permitted
  observable is the **admitted-row-count gap** Strict vs Aware at the owner's
  own pin plus the folded verdict-set difference — **no κ, no fusion verdict,
  no jc dep**. If the orchestrator wants κ, that is a manifest change and a
  separate deliverable.

If even R2 is deemed out of scope for this stage, the correct outcome is: z=5
is declared reserved-unimplemented in the file and in the not-claimed block. It
is never silently mapped to one of z=1..4.

---

## 4. (d) Pre-registered gate table

Pinned BEFORE any run. Every gate has both halves on non-trivial inputs.

**Precondition for the headline pair — TWIN SLICES.** The probe's owners get
*disjoint* verse slices (`owner_verses`, `probe_ignition.rs:426-429`), so
"byte-identical rows" is unreachable there. D-IGN-B re-carves the in-scope
cohorts (§5): owners `0..8` are all seeded from the SAME 48-verse slice, armed
`z = 1,1,2,2,3,3,4,4`. Gate L0 asserts that identity before L1 is read.

| # | Gate | can-FIRE (non-trivial) | can-STAY-SILENT (non-trivial) |
|---|---|---|---|
| **L0** | twin premise | the 8 twin owners' content planes are pairwise **byte-identical** across all populated rows | the twin planes are **non-zero** and differ from a non-twin owner's plane (else "identical" is the trivial all-zero case) |
| **L1** | **the lens axis is load-bearing** | owners 0 (z=1) and 6 (z=4), byte-identical rows, same cycle ⇒ `digest` values **differ** | owners 0 and 1 (both z=1), byte-identical rows, same cycle ⇒ digests **bit-identical** |
| **L2** | arming | the unarmed owner has **no** entry in the readout map for any cycle; after arming it (the G8 pattern, `probe_ignition.rs:1125-1143`) a readout appears | an armed owner in a cohort that never enters CognitiveWork produces no readout either — absence is not proof of the arming axis on its own |
| **L3** | **per-lens non-emptiness (measured, not assumed)** | each of z=1..4 yields a NON-EMPTY readout on ≥1 owner | per-lens empty counts are printed for all 32 in-scope owners; a lens empty on **every** owner is a loud FINDING, not a silent pass |
| **L4** | **anti-degeneracy / anti-collapse (§12.7's shape)** | over ONE owner, the four lens digests are **not all equal** (≥3 distinct of 4) | each non-empty lens yields **≥2 distinct digests across the 32 in-scope owners** — a lens whose readout is constant over every owner carries no information (the 99.61 % / 150-of-150 shape, plan §12.3a″ + CLAUDE.md falsifiability rule) |
| **L5** | mechanics unchanged by the swap | c1 still seals the pinned decomposition (20 Flow `Planning→CognitiveWork` + 4 Block `Planning→Prune`, `probe_ignition.rs:977-984`) with the lens body in place | the readout map is **empty** at c1 (nothing has entered CognitiveWork yet) — the lens cannot precede the seal |
| **L6** | seal→apply ordering | every readout key `(id, c)` has `id` in the set of owners whose applied move had `to == CognitiveWork` in a PRIOR cycle | an owner that never received such an applied move has no readout at any cycle |
| **L7** | address axis (inherits G7) | an OUTSIDE owner, run through the lens body directly, produces a readout | the OUTSIDE cohort has **no** readout from the main loop (never scanned) |

**Pre-registered risk, recorded so the fallback is not post-hoc.** §12.3a″
measured Hegel **constant-false** on the SPO path (uniform frequency ⟹ zero
contradiction depth), and `contradiction_ranking` filters `> 0.05`
(`stance.rs:418-427`) while `stream` emits `f = 0.9` / `f = 0.05` under
negation (`stance.rs:274`). A Hegel readout is therefore non-empty only if some
statement is observed **both** negated and affirmed within one owner's 48
verses. **If Hegel is empty on all 32 owners, that is a measured result and L3
reports it as such; L1's can-fire pair then MUST be witnessed by a pair not
involving the empty lens** (e.g. z=3 vs z=4). Pre-registering this now means a
post-hoc pair swap cannot be mistaken for fitting. The same caveat applies to
Nietzsche (it consumes Hegel's ranking, `stance.rs:483-496` — if Hegel is
empty, Nietzsche is empty too; **the two are not independent**, which is itself
worth printing).

Kant is *not* eligible for the old tautology: `quale > ablated` reduces to
`modal > 0.5` (plan §12.3a, verified against `stance.rs:499-510`) and is
forbidden as an assertion. The Kant readout is the (label, graded, ablated)
triple digest, never a comparison verdict.

---

## 5. (e) Pinned run shape

Reuse the probe's constants where they still hold; ONE change, justified.

| constant | value | status |
|---|---|---|
| `FLEET_OWNERS` | 64 | unchanged (`probe_ignition.rs:108`) |
| `ROWS_PER_OWNER` / `POPULATED_ROWS` | 64 / 48 | unchanged (`:109-110`) |
| `CORPUS_VERSES` | 3072 | unchanged (`:111`) — the twin owners reuse slice 0; unused slices stay loaded, keeping the pinned constant honest |
| `SCOPE` | 0..32 | unchanged (`:112-113`) |
| `CYCLES` | 6 | unchanged (`:114`) |
| **cohorts** | **re-carved** | **CHANGED — see below** |

**Cohort re-carve (the one change).** In-scope 0..32:
`0..8` = TWIN BLOCK, one shared verse slice, armed `z = 1,1,2,2,3,3,4,4`,
`flow_qualia()`, 3 firing rows; `8..30` = SPREAD BLOCK, distinct slices, armed
`z` cycling 1..4, `flow_qualia()`, 3 firing rows (this block feeds L4's
non-constancy half); `30` = UNARMED (`z=0`); `31` = ORPHAN (not inserted —
keeps G10's #879 accounting caveat alive); `32..64` = OUTSIDE, armed, never
scanned (L7).

**Justification:** the headline gate is "different lenses over **byte-identical
rows**". Disjoint slices make that sentence untestable — any digest difference
would be confounded by content. The twin block is the minimum change that makes
L1 falsifiable. The rest/CONTRA cohorts drop out because their gates (G4/G5)
are PROBE-IGNITION's, already green, and re-litigating them here would dilute
this stage's single fact.

**Cost, bounded and stated.** §12.7 measured `stance::stream` superlinear —
`staunen(Snapshot::of(arena, 0.0))` is an O(arena) scan **per rung lift**
(`stance.rs:324-328`), which is why a 31,102-verse single-arena run blew a
10-minute budget. That does **not** bind here: each owner gets its **own**
arena over **48** verses, and at most one lens pass per owner per cycle. Order
of magnitude: ~30 armed owners × ≤5 cycles × 48 verses. **UNVERIFIED as a wall
time** — no run has happened; the build lane should print the elapsed lens time
and the orchestrator should treat >30 s as a signal to cut the spread block.

---

## 6. (f) NOT claimed

1. **No row-byte decoding, and the cognition reads past the substrate.** The
   lens takes its text from the corpus slice the rows were seeded from,
   selected by the owner's address and span; bloom planes are one-way (F1).
   This is the §12.7 defect shape; the substrate governs selection, never
   content (F1b). No substrate-data-path claim follows from any readout.
2. **No per-lens dispatch.** `stance_panel` computes all four stances in one
   call; arming selects **what is read**, not what is computed (F0). No claim
   that the armed bits steer any computation, and no cost difference by z.
3. **No independence between lenses.** Nietzsche is computed from Hegel's
   output (`stance.rs:483-496`); Kant is derived from `ReadOut::lifts`.
4. **No fusion verdict, no κ.** z=5 is reserved (§3).
5. **No stance validity.** A readout is a read, never evidence that a stance is
   right about anything.
4. **No parallelism, no durability, no scale, no multi-writer, no recovery** —
   PROBE-IGNITION's not-claimed items 1–4, 12 carry over verbatim.
5. **No 36-style claim.** Six probe-local ordinals; the persona-36 bridge stays
   an open non-goal (persona-vs-rung-ladder O1/O3).
6. **No rung-3 / runbook claim.** The four stances are not the 34 NARS tactics
   and are not `StyleFamily` macros.
7. **No deinterlace/temporal claim in the main loop** (only the separate R2
   test would touch `QueryReference`, and then only for row admission).
8. **No claim that the four lenses are independent instruments** — Nietzsche is
   computed from Hegel's output (`stance.rs:483-496`).
9. **No zero-copy claim.** `SweepSlot::payload` is `Vec<u8>` by the shipped
   signature.
10. **No semantic claim about the corpus.** Qualia remain declared fixtures
    (`probe_ignition.rs:201-210`); nothing is encoded from text into qualia.

---

## 7. (g) Open questions + blockers for the orchestrator

- **B1 (BLOCKER, and the deliverable per the brief): z=5 cannot run per-owner
  in-cycle** on the probe's run shape — one horizon, no pool growth, no `jc`
  dep. §3. **Decision needed: R2 (separate reduced test, no κ), or defer z=5
  entirely.** Either is honest; silently mapping z=5 onto another lens is not.
- **Q1 — is the cohort re-carve (§5) acceptable?** It is the only way L1 is
  falsifiable, but it *replaces* the probe's REST/CONTRA cohorts inside SCOPE.
  Those gates stay green in `probe_ignition.rs`, which is untouched; confirm
  that is the intended split (two files, two facts) rather than one growing file.
- **Q2 — verse labels.** `stance::stream` wants `(label, text)`. Proposal:
  `"kjv:{global_index:05}"`, matching `blw_fusion`'s subject format
  (`blw_fusion.rs:220`). Confirm, since a label change later invalidates
  digests.
- **Q3 — digest definition is load-bearing and must be pinned in the build
  brief**, not left to the builder: a stable fold over the variant's contents in
  the shipped iteration order, floats hashed via `to_bits`, no `HashMap`
  iteration anywhere in it (`stance_panel`'s Wittgenstein arm sorts before
  returning, `stance.rs:530-532` — that is the only reason it is
  order-deterministic; the Kant and Hegel arms are already vector-ordered).
- **Q4 — if L3 measures Hegel AND Nietzsche empty on all 32 owners**, D-IGN-B's
  effective lens axis is 2-valued (Kant vs Wittgenstein). That is still a
  passing probe under §4's pre-registered fallback, but the orchestrator should
  decide in advance whether a 2-valued axis is worth the build, or whether the
  corpus slice should first be checked for a polarity flip.
- **Q6 — the two framing calls are yours to ratify, not mine.** (i) F0: is a
  *selection* axis (not dispatch) still worth building? My answer is yes and the
  argument is in F0, but the deliverable's name promises more than it delivers,
  so the rename should be explicit. (ii) F1b: is reading text past the substrate
  acceptable given that selection is fully substrate-governed? My answer is yes
  under the stated condition; a "no" here kills or re-scopes D-IGN-B and is
  defensible. **Neither should be settled by the build lane.**
- **Q7 — L1's silent twin is where hidden nondeterminism surfaces.** Two owners,
  same z, byte-identical rows ⇒ bit-identical digest is only non-trivial because
  the panel's Wittgenstein arm builds a `HashMap` before sorting
  (`stance.rs:513-532`) and the interner assigns ids in first-sight order
  (`:63-82`). If ids differ between two owners with identical text the digest
  must still match — worth confirming in the build brief that per-owner
  interners start empty (they will, if each owner gets a fresh
  `Interner::new()`).
- **Q5 — CI.** `cycle-driver` is not in the supervisor's CI feature list (the
  fifth blind gate, AGENT_LOG 2026-08-04); this test inherits that inertness.
  Workflow edits are operator-approved only — recorded, not changed.

---

## 8. Reads performed for this note

`sonnet-worker-guardrails.md` (full); `AGENT_LOG.md` (first 130 lines);
plan §12.3a″ / §12.3c / §12.5 / §12.7 / §12.8 / §12.11; `persona-vs-rung-ladder.md`
(full); `probe_ignition.rs` (full, 1,389 lines); `probe-ignition-design-opus.md`
(cited via the probe's own in-file §-references and the AGENT_LOG entry —
**partial**, flagged here rather than implied); `stance.rs` (full);
`blw_fusion.rs` (targeted: constants `:100-260`, seeding/analysis `:480-600`,
deinterlace `:960-996` — **not** read end-to-end, so its gate list is cited only
where quoted); `cycle_driver.rs` (targeted: `:220-260`, `:338-360`, `:483-680`);
both Cargo manifests. No cargo command was run; nothing here is compiled or
measured.

**Coordinator's three constraints (relayed mid-lane) — all three had been
derived independently in this lane before the message arrived, and the note now
states them in the coordinator's required terms:** no per-stance dispatch (F0 +
§1 heading + not-claimed 2); no readout slot in the shipped seam (§1's
three-option judgment, out-of-band `&mut` chosen with L6 as compensation); the
stance bodies consume text, not planes, which is the §12.7 defect shape (F1b +
not-claimed 1). The additional instruction — evaluate `ReadOut` as the readout
type — produced a **rejection with reasons** (§2): it is the panel's INPUT, is
lens-independent, and using it would make L1 fail by construction; it is
retained as printed context only.

---

## ⊘ Orchestrator ratification of Q6 + Q7 (2026-08-05)

The design lane correctly refused to settle two framing calls itself and
refused to let the build lane settle them. Both are ruled here, before the
build lands, so the record shows the decision preceded the result.

**Q6(i) — F0, a SELECTION axis rather than dispatch: BUILD IT, renamed.**
§12.11's observable is *"different lenses over byte-identical rows produce
different readouts"*, and a selection axis satisfies that non-vacuously —
four different return types, four different derivations, and L4 can still
fail. What the finding kills is any **compute-steering** claim: no assertion
that the armed bits change what runs, and no cost difference by z. The axis
is therefore named **lens selection** in the file, in the printed banner, and
in the plan row; "dispatch" is not used for it anywhere. A deliverable whose
name promises more than it delivers is the failure mode this ruling exists to
prevent.

**Q6(ii) — F1b, reading text past the substrate: ACCEPTABLE HERE, on the
stated condition.** The §12.7 KILL was a harness where the substrate governed
**nothing**. Here it governs **selection end-to-end** — which owner, which
span, which arming, and a phase reachable only through a sealed transition —
and L2/L5/L6/L7 each falsify one leg of that. The condition is binding and
already met in the note: **no substrate-data-path claim may follow from any
readout**, and F1/F1b appear verbatim in the not-claimed list. If a future
reader finds a readout being cited as evidence about the substrate's data
path, this ruling is void and the probe is a corpus harness in a substrate
costume.

**Q7 — per-owner interners: relayed to the build lane as a requirement**, not
left to chance. L1's silent twin (same z, byte-identical rows ⇒ bit-identical
digest) is only non-trivial because the Wittgenstein arm builds a `HashMap`
before sorting and the interner assigns ids in first-sight order. Each owner's
lens run therefore constructs a **fresh** `Interner`/arena, and the digest must
be computed over sorted, id-independent content. If the build cannot satisfy
that, L1 must be reported as unbuildable rather than passed on lucky ids.

**`ReadOut`-as-readout rejection: upheld.** It is the panel's INPUT, it is
lens-independent, so L1 would pass by construction — which is precisely the
vacuous-assertion shape the house rule forbids. Keeping it as printed context
(its cardinalities explain an empty lens) is the right use.
