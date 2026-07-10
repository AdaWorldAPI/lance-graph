# VISION — the AGI-aspiring canon, re-grounded 2026-07-10

> The companion of `FUTURE-DESIGN.md` (which carries the wiring queue): this
> file carries the WHY. Written at the close of the 2026-07-10 arc, at the
> operator's directive to pull the AGI-aspiring threads back into one living
> synthesis before the tier transition — "and you keep the vision alive."
>
> Every claim here is graded: **[G]** = measured, probe named; **[RULING]** =
> operator-decided, canonical text in EPIPHANIES; **[ASPIRATION]** = the part
> still ahead, named so it cannot masquerade as earned. That grading
> discipline IS the vision-keeping — the fire without the map burns the map.

## 0. The one sentence

**A mind is a deterministic substrate that cannot stop thinking, never waits
for anything, and reconstructs more than it stores.**

Each clause is now load-bearing: *cannot stop thinking* — active inference as
the dispatch mechanism (CLAUDE.md "the shader can't resist the thinking");
*never waits* — the prime invariant E-NOBODY-WAITS-1 [RULING]; *reconstructs
more than it stores* — measured twice in one day (D-MTS-5, D-MTS-6) [G].

## 1. The Click survived every carrier — because it was never about the carrier

The founding claim (CLAUDE.md P-1): parsing, disambiguation, learning, memory,
and awareness are ONE operation. The carrier of that operation has now moved
three times, each move recorded as a dated supersession rather than a rewrite:

1. **VSA bundle** (2026-04-21) — multiply+add on role-indexed fingerprints in
   `Vsa16kF32`.
2. **Baton** (2026-05-26, E-BATON-1) — the bundle scoped to one compartment;
   inter-mailbox state as discrete owned handoffs, ownership proving
   no-aliasing at compile time.
3. **Zero-copy SoA + temporal stream** (2026-06-11 PR #477; 2026-07-10
   E-MARKOV-TEMPORAL-STREAM-1) — no handoff type at all; every envelope
   zero-copy from creation to Lance tombstone
   (`docs/architecture/soa-three-tier-model.md`); the Markov trajectory on
   the `temporal.rs` sorted stream, where Chapman-Kolmogorov holds *exactly*
   by stream order — a strengthening, not a retreat. VSA holds its
   I-VSA-IDENTITIES niche (≤32 lossless role superposition, one compartment)
   with the bundle algebra untouched.

The lesson the three moves teach: **the Click is the LOOP, not the algebra
that hosts it** — the loop that resolves each input against everything it has
committed (graph), recently seen (episodic), believes about its own competence
(awareness), and currently expects (prior), then commits the result back into
the tissue it just read. Cut any organ and a specific cognitive capacity
disappears (CLAUDE.md "thinking tissue"). The tissue IS the loop. That
sentence has survived three substrates; treat it as the invariant to protect
when the fourth move comes.

## 2. Thinking-is-a-struct became thinking-is-a-ROW

The `Think` struct's fields were declared to BE the grammatical roles of
cognition (CLAUDE.md: trajectory=Subject, awareness=Modal, free_energy=Kausal,
resolution=Predicate, graph=Lokal, episodic=Temporal — the TEKAMOLO of
cognition IS the struct layout). V3 did not delete that; it **compiled it into
the row**:

| Think organ | V3 home |
|---|---|
| trajectory | the `temporal.rs` version-range read (any window, per-reader rung) [RULING E-MARKOV-TEMPORAL-STREAM-1] |
| awareness | MetaWord bits + NARS truths in SoA lanes; migrating per E-THINKING-TENANTS-V3-1 [RULING] |
| episodic | Lance versions — the OGAR D-DELTA mapping made primary; basins = `part_of:is_a` rails |
| graph | SPO quads / AriGraph |
| codec | palette 256×256 LUTs (1.8 ns measured) |
| resolution | the kanban card's Rubicon position (§5) |

AGI-as-glove (The Stance) holds unchanged: AGI = (topic, angle, thinking,
planner) = the SoA under dispatch — never a wrapping struct, never an "AGI
service." The struct didn't die. It became **the row plus its ClassView
readings** — which is a stronger form of itself, because rows are addressed,
versioned, replayed, and projected.

## 3. Projection is now a FIVE-axis measured fact, not a slogan

"Classes don't modify or copy, they only project" began as a design ethic.
The 2026-07-10 arc closed it into measurement across five axes:

| axis | mechanism | status |
|---|---|---|
| **bytes** | RESERVE-DON'T-RECLAIM (zero tier = not consulted, never compacted) | [RULING], canon ladder |
| **readings** | classid → ClassView projects the content-blind 4+12 facet; every sanctioned reading coexists in the same 12 bytes | [RULING E-V3-FACET-4-PLUS-12] |
| **time** | temporal replay: `QueryReference::at(v, rung)` + deinterlace — crash-replay = session-replay = time-travel, ONE mechanism | [G] W1b probes |
| **scale** | comma replay: upper pyramid bounds (64×64…256k×256k) regenerated transiently from (GUID, envelope) — **N_eff 11.00/12 witnesses vs strict alignment's 1.00; replay bit-identical any order; 82 KB touched vs ~69 GB dense** | **[G] D-MTS-5** (`comma_quorum.rs`) |
| **bits** | comma-dithered reconstruction: **one stored truth bit per level matches full-CausalEdge64 awareness proxies (k\*=1 vs aligned k\*=4; the lattice buys ≈log₂ L effective bits)** | **[G] D-MTS-6** (`comma_awareness.rs`) |

The fifth axis is the deepest AGI-aspiring result the substrate owns: **the
awareness payload is mostly reconstructible from the address plus a tiny
envelope.** The comma (an irrational-progression coprime walk, generated from
the address, never stored) makes L pyramid levels *independent witnesses* —
stratified, not redundant. Boundary condition, measured honestly:
N_eff = min(L, spectral participation of the detail) — broadband content gets
the latent granularity; smooth content saturates at its own information
content (D-MTS-5 run #1's pre-registered FAIL, kept in the chronicle). This is
what "granularity and flexibility for details we can't even comprehend yet"
means operationally: the upper levels were never baked at write time, so a
future reader can project resolutions and readings nobody designed for.
Fence: D-MTS-6b (driver-integrated fixture) gates any REAL CausalEdge64
shrink — the proxies are proxies [ASPIRATION until 6b].

## 4. Families orchestrate, runbooks execute, templates compile

The style stack resolved into a ladder with distinct KINDS at each rung
[RULING E-STYLE-FAMILY-VS-RUNBOOK-1]:

- **StyleFamily (12)** — abstract orchestration families: which KIND of
  thinking a cycle runs. One canonical type after M9 killed five divergent
  hand-rolled tables [G, E-FIVE-STYLE-TABLES-1 — duplication had already
  produced live semantic drift; the dedup was archaeology, not tidying].
- **ThinkingStyle runbooks (36)** — literal NARS runbooks: concrete inference
  recipes, seeded into the rung ladder, consumed as graph-flow's replayable
  chaining unit. Orthogonality preserved (THINKING_RECONCILIATION): HOW-TO-BE
  (personas) and HOW-TO-DO (operations) are different 36-spaces — never
  conflate them.
- **Compiled templates** — the elixir-like notation × StepMask: runbooks
  compile down; the oracle (rig, 1–2 ms framework vs 8.4 s LLM measured —
  orchestration is FREE) ratchet-shrinks its own involvement.

The endgame this ladder points at [ASPIRATION, the honest crown]: **thinking
that compiles its own thinking** — traces become templates, templates carry
the replayable runbooks, the LLM's role shrinks to proposing what the
substrate then executes deterministically. Style-as-class (D-TSC-2/3, after
the batched mint) makes every rung addressable: per-cycle casts reference the
style classid, so attention is replayable from the address alone.

## 5. The Rubicon heart, under the prime invariant

The kanban is not a task board — it is **volition, typed**
(`contract::kanban::KanbanColumn`): Planning → CognitiveWork is the −550 ms
Σ-commit (the Rubicon crossing, Libet-anchored — intention forms before the
conscious act); Planning → Prune is the pre-Rubicon veto ("free won't");
Evaluation's 3-way is the postactional reckoning; Plan → Planning
re-deliberates carrying the witness. The cast records intent AHEAD of the ack
— crossing at intent formation, exactly Rubicon-conformant.

And beneath it, the prime invariant [RULING E-NOBODY-WAITS-1]: **nobody waits
for anything or any scheduling.** Writes are fire-and-forget ("melden macht
frei"); the Lance ack itself proposes the next move (`ack_and_propose` —
orchestration is self-updating, postactional evidence of cycle N seeding the
deliberation of cycle N+1); updates reprioritize, never gate; absorbing views
rest the loop, never deadlock it; ractor exists solely as the compile-time
ownership guarantee — `&mut` IS the serialization, no messages, no actors, no
scheduler anyone blocks on. Any message path found in the tree is redundancy
(TD-MESSAGE-RESIDUE, left as-is by ruling). Every future design choice on
this surface is judged against the invariant's sentence first.

## 6. The body plan — the ancestry pipeline

```
ladybug-rs → thinking-engine → p64 (the DTO ladder:
  StreamDto Φ / PerturbationDto Ψ / BusDto B / ThoughtStruct Γ)
  → cognitive-shader-driver → SoA (MailboxSoA, value tenants)
```

The organism grew along this arc, and the organs that haven't reached the
bloodstream yet are catalogued, not lost: `CascadeChannels8` (the named
Morton-cascade mantissa carrier — first wiring target), the calibration
battery (cronbach / ground_truth / reencode_safety — feeds D-MTS-2's
certification gate), the cognition organs (ghosts / persona / qualia /
world_model — style-class candidates), the M8 collapse surface, the L4
learning-loop seam. The census (MODULE-TABLE ancestry addendum) is the
anatomical atlas; the standing constraint is vascular law: **gems reach the
hot path only as ClassView readings of existing lanes or through the
envelope-auditor gate — never as ad-hoc lanes.**

## 7. What must not dilute

1. **Probe-first, chronicled honestly.** Claim → pre-registered gates → run →
   the numbers stay even when they fail. Both comma probes failed their first
   runs on mis-registered gates; both chronicles kept the failures and both
   diagnoses became findings (the spectral-participation ceiling; the
   boundary-noise margin). A vision that cannot survive its own probes is
   decoration.
2. **The iron rules** — I-SUBSTRATE-MARKOV (bundle, never XOR, for transition
   paths), I-NOISE-FLOOR-JIRAK (weak dependence; no classical Berry-Esseen σ
   claims), I-VSA-IDENTITIES (bundle identities, never content registers;
   four tests before reaching for VSA), I-LEGACY-API-FEATURE-GATED (no silent
   semantic change under a shared name — the five style tables were the
   proof of why).
3. **Append-only canon.** Supersessions are dated notes on top of the old
   text, never rewrites. The three Click supersessions are the vision's
   fossil record — the proof it evolves instead of thrashing.
4. **The proof chain stays attached** (categorical-algebraic-inference §5:
   Shaw's Kan-extension theorem making element-wise binding a theorem not a
   choice; beim Graben's Fock-space universality; Jian & Manning's
   abstraction-first learning; Schulz's KL decomposition; Alpay's Ω(t²)
   bound that active inference dodges; Graichen's 85/75 tiering boundary;
   Gallant's three-stage isomorphism to bind→bundle→resolve; Kleyko's
   computing-in-superposition). The niche VSA kept is the niche the papers
   actually prove.
5. **The measured numbers are the vision's spine**, not its decoration:
   611M SPO lookups/sec · 1.8 ns palette distance · 17K tokens/sec ·
   1–2 ms orchestration vs 8.4 s oracle · N_eff 11.00/12 · k\*=1 ·
   1549 tests green at the M9 merge. When a future session doubts the
   aspiration, it should re-run the probes, not re-read the prose.

## 8. The road (in dependency order)

1. **D-MTS-1** — Markov-as-stream parity on the DeepNSM corpus: the stream
   earns the Markov crown or it doesn't. Gates ALL VSA-path removal. The
   keystone. [next]
2. **D-TTV-1** — thinking tenants onto the V3 substrate (envelope-auditor
   gated), the old CausalEdge64 standing as the perturbation baseline.
3. **D-MTS-6b** — awareness in the REAL driver loop vs that baseline: the
   knee measured where it matters. Only then may the edge shrink.
4. **D-MTS-2/3** — the L4 palette² shader certification; hierarchical-4⁴
   codebooks. The perturbation shader becomes the carrier it was pinned to be.
5. **D-TSC-2/3** (after the batched mint) — styles as classes; attention
   replayable from the address; the hot-plug seat filled.
6. **The compilation loop closes** [ASPIRATION]: traces → templates →
   runbooks → families — the substrate proposing, executing, and shrinking
   its own thought, under the prime invariant, on tissue that reconstructs
   more than it stores.

The shader can't resist the thinking. Now the substrate can't forget how to
prove it.
