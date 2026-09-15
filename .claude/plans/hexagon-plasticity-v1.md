# hexagon-plasticity-v1 — can six learned rails discover local adjacency, and can ARM promote a cue to a path?

**Status:** ACTIVE — landed 2026-09-14 on operator go. **W-1 and W0 RAN BEFORE
LANDING**; their results and the corrections they forced are §9, and the gate that
replaced the struck W0 signal gate is §10. Waves W1–W5 are not started.
**D-ids:** D-HXP-0 … **D-HXP-7** (rows on `STATUS_BOARD.md`; D-HXP-7 added §10).
**Ruled symbols named, with their rulings:** `MergeMode::Xor` — forbidden on the
magnitude side by `I-SUBSTRATE-MARKOV` (breaks the Chapman-Kolmogorov semigroup);
the sign/phase side may XOR. `CausalEdge64` — named only to state it is NOT touched
(operator, 2026-09-13).
**READ BY:** truth-architect · falsifier-auditor · measurement-skeptic · iron-rule-savant ·
dto-soa-savant · zero-copy-warden · kernel-membrane-warden · heuristic-gate-warden.
**Extends (does not supersede):** `HEXAGON-MQ-FEASIBILITY.md` §H (ruled the register an
*address*, and only for the `6×(8:8)` rail); `E-HEX-TENANT-RAIL-IS-DIRECTION-CHAIN-IS-FREE-
SHIFT-IS-THE-COST-1` + its same-day STORNO (put the grey side in the loop for the first time);
PR #1226 (a validated executor a rail program could run on).
**Not touched, by rule:** `CausalEdge64` (operator, 2026-09-13: *never*); the 512-byte row
stride; the 16-byte edge block; the zero-dep contract's dependency set.

---

## 0. The claim, in the operator's words (2026-09-14, three statements)

> *"The whole idea of hexagon is, that there's enough muscle memory to follow Mississippi
> Queen masking being a possible learned shortcut, a new synapse not ordered from HHTL top
> down."*

> *"BPE is a bag of pairs thrown at a GPU. Our Substrat should be able to 'get the cue' and
> recognize it akin to a synapse and with enough frequency and confidence becoming a path
> after validation."*

> *"The whole point is whether hexagon can provide plasticity and if arm discovery can
> discover edges — that's a breakthrough. But imagine the potential: you have intermediate
> known unknowns and suddenly a new thought is discovered and reinforced by flow
> (Csíkszentmihályi)."*

Three claims fall out, tested in that order because each is the next one's precondition:

- **C1 — plasticity.** Six `(cue : strength)` rails per node can *learn* which neighbours
  recur, by cue → bundle → evict — and at least some learned rails point **off-trie**: to a
  node the HHTL prefix would not reach at that depth. A rail that never leaves the trie is a
  cache of descent, not a synapse.
- **C2 — discovery → path.** `lance-graph-arm-discovery` (Aerial+ transcode), fed candidate
  rails as transactions, promotes a subset to **validated paths** carrying `⟨f, c⟩`, and the
  promoted set beats a same-size shuffled promotion on held-out futures.
- **C3 — flow.** Promotion yield is highest where challenge ≈ skill (the escape rate is
  neither ~0 nor ~1), and `FlowState` — already a `#[repr(u8)]` contract enum — is the
  readout, not a metaphor.

## 1. What is settled — do not re-derive

Every number below is measured and cited in §8. The scoping error the plan corrects is
stated first, because it is why the arc is open at all.

**The MQ arc fed the register top-down material in all four hexagon tests.** H5a: a
refinement-tree address. H5b: nearest-over-addresses. H5c: `common_prefix` bags. H5d — the
one arm that asked the lateral question (*"6 rails of `(label:target)`"*) — never ran:
*"the register was never in the loop, so the information is identical by construction."*
A sweep of both reports for `learned shortcut | lateral | bypass | non-hierarch` returns
**zero** hits. So *"the hexagon adds no computational material"* is true of **the hexagon as a
host for the quotient's own addresses** — and says nothing about six learned lateral rails.

| verdict | claim | evidence |
|---|---|---|
| **broke** | the six as a recall topology | Q6/Q7/Q8; `E-Q8` |
| **broke** | id composition carries meaning | H5c: prefix-bag R@10 **0.48 %** vs permuted **0.77 %** |
| **broke** | one byte per refinement round | H5a: fan-out 25 / **463 / 958 / 427** / 176 … rounds 2–4 exceed 255 |
| **broke** | two eyes reach equivalent states | R3: 3.1 % entry equality, 86 % eye-identifiable |
| **survived** | the `6×(u8:u8)` register as an *address* | untouched by any falsifier |
| **survived** | depth-≤3 address in **3 bytes**, mixed-radix | 25·463·958 = 11.1 M < 2²⁴ (H5a's own escape) |
| **survived** | the rail-index-is-direction reading | hex-tenant FINDING, 3 gates 32/32, 0 heap B/step |
| **finding** | R2 normalization law transfers, with negative control | 0.7–1.1 bits vs 1.3–3.1; **nothing** on random graphs / unseen boards |
| **finding** | the trie-node dividend, three independent times | range write 49–99 ns vs 22.5 µs (228–462×); shift over node span −66 % vs field −14 %; mixed-radix fit |

**The division of labour the operator's BPE remark names is already measured (F-MQ8 + H5b):**

| | recognition ("get the cue") | transfer (the law) |
|---|---|---|
| BPE bag of pairs | **strong** — override-twin R@10 **18.1 %** | **fails** — worse than raw, **5 / 5** foreign corpora |
| hydrated quotient | weak — R@10 **1.5 %** (null 0.15 %) | **transfers** — **12 / 12** foreign corpora |

BPE bags are addressed by *content*; the quotient by *future structure*. Content-addressing
is what makes a cue fire; future-structure is what makes it generalize. The plan wants both,
and **the cue side is the measured weak link** — that ordering is why W1 precedes W2.

**The river board is the fairest ground for C1 and the least fair for the quotient.** Same
board, other entries, **0 % escape** (nothing left to generalize): the local table C2p wins
at **3.75 bits** against the futures quotient's 4.68, and the report's own reason is *"the
river's law is tile-local and a futures quotient is the wrong tool for it."* Tile-local is
lateral. A six-slot learned adjacency *is* a local table.

## 2. The loop, mapped onto shipped symbols

Nothing in the loop needs a new type. Three of its pieces turned out to exist under their
own names already; the plan's work is wiring and measurement.

| step (operator's words) | mechanism | symbol | status |
|---|---|---|---|
| "get the cue" | 1-byte content-addressed pre-filter | `Scent: u8` — byte 0 of ZeckF64, ρ 0.937, *"heuristic pre-filter"*; container slot W176-191 *"scent/palette neighbor indices"* | **shipped** |
| "recognize it" | mask predicate / resonance, T1 population algebra | `*_to_mask_under` (ndarray #307), `MaskOp` executor (#1226) | **shipped** |
| "akin to a synapse" | `(existence : weight)` per rail | hex-tenant `6×(permeability : strength)`; strength bumped **Hebbian, saturating, on the delta frontier** (`scratch & !state`) — *"one owner, no second structure"* | **shipped** (probe) |
| "frequency" | co-occurrence count | `CandidateRule::cooccur` — *"the NARS evidential mass `m`"*; rail `strength` | **shipped** |
| "confidence" | evidence-discounted truth | `arm_to_truth_u8(rule, k) -> TruthU8` (`translator.rs` at `pub fn arm_to_truth_u8`); `NarsTruth` — three producers, one carrier | **shipped** |
| "after validation" | held-out rule mining | `lance-graph-arm-discovery`: `Dataset { spec: FeatureSpec, rows: Vec<Vec<u32>> }`, `CandidateRule { antecedent, consequent, cooccur, antecedent_count, window }` | **built — zero production callers** |
| "becoming a path" | promotion of a rail to a committed edge | CollapseGate FLOW; rail write-back by gated XOR (sign) / bundle (magnitude) | **partial** |
| "known unknowns" | the escape set | held-out positions with no training future: **51.1 %** at depth 12 (58 % → 51 % as the corpus grows) | **measured** |
| "flow (Csíkszentmihályi)" | challenge/skill readout | `mul::FlowState { Flow=0, Boredom=1, Transition=2, Anxiety=3 }` — *"Challenge ≈ Skill → flow"*, `#[repr(u8)]`, D-CSV-13b | **shipped** |

**The one byte that changes meaning.** Today rail `d` is a fixed hex direction and its first
byte is *permeability of d*. C1 reinterprets that byte as `scent(target)`: six fixed
directions become **six learned pointers**, `6×(scent : strength)`. Layout, stride, edge
block, `ENVELOPE_LAYOUT_VERSION`: unchanged. Per `I-LEGACY-API-FEATURE-GATED` the reading is
selected per class (`ClassView`), never inferred from bytes.

**The engine for the hardest step is idle.** `lance-graph-arm-discovery` is a workspace
member, declared by `osint` under `[dev-dependencies]` only, with **no `src/` caller
anywhere** (measured this session; `markov_soa.rs` (a doc comment, not a call site) is a doc comment). Under C2 it is
not an orphan — it is the promotion gate waiting for candidates.

## 3. Waves, gates, kill conditions

Every gate is **two-sided**, has a named **disable run** (red-then-green or it is not
evidence), and an **anti-vacuity** check. A wave that cannot fail is not a wave. Numbers
with a `?` are thresholds set here and are **policy pins until W0/W-1 measure them**; the
doc-comment on each constant must say so.

### W-1 — reproduce the baselines (D-HXP-0, no new code)

*Why first:* `git status` reports clean, not current — and data ages the same way. Every
threshold below is anchored to a number from a prior container.

- Build the `mqprobe` crate from the scratchpad plateau; re-emit the successor table for
  both eyes and the river fixture (`pybaseline_river.py` for the C2p arm).
- **Gate:** C2p on river same-board reproduces **3.75 bits ± 0.05**; code source k=3
  reproduces H **1.72** / C2p **1.95**; H5b R@10 reproduces **1.5 %**.
- **Kill:** any anchor fails to reproduce → re-pin every threshold in this plan to the
  fresh number *before* W0, and record the drift on the board. No wave runs against a
  number that did not reproduce.

### W0 — the histogram that decides feasibility (D-HXP-1, no new code)

Six slots against measured refinement fan-out **463 / 958 / 427** is 0.6 % of the branching.
Fan-out counts *distinct* children and says nothing about how often each is taken.

- **Procedure:** over the W-1 successor table, per state, sort successors by transition
  count; report the mass carried by the top-k for k ∈ {1, 2, 3, 6, 12, 32}, on both eyes
  and on the river same-board row. Emit the full top-k curve, not one number.
- **Gate (PASS):** top-6 mass **≥ 60 %?** on both code eyes.
- **Kill:** top-6 mass **< 40 %?** on both eyes → six slots cannot be adjacency; the arc
  stops, and *"the k at which mass reaches 60 %"* is the finding filed instead.
- **Between:** proceed, with `k` at 60 % recorded as the slot count the substrate actually
  needs; C1's "six" becomes a measured parameter, never a constant.
- **Anti-vacuity:** the same histogram on the synthetic *unstructured* graph must be flat
  (top-6 ≈ 6/|succ|). If it is not, the histogram is measuring the generator, not the law.

### W1 — the cue benchmark (D-HXP-2)

The measured weak link. Incumbent: **BPE motif bags, R@10 18.1 %** on override-twins.
Quotient: **1.5 %**. Null: **0.15 %**.

- **Procedure:** the identical override-twin task from H5b, with the query addressed by a
  scent/codebook cue instead of a hydrated bag. Two arms: (a) `Scent` byte alone;
  (b) palette256 code (ρ 0.9973 on *structured* patterns — **not** single words, where the
  `TD-BASE17-FOLD-CEILING` caps ρ at 0.26; the fixture must be SPO/motif-shaped, and the
  plan says so because it is the one place the codebook is known to fail).
- **Gate (PASS):** R@10 **≥ 18.1 %** — a content-addressed cue matches the dictionary's
  recognition *without the dictionary's corpus lock* (F-MQ8's 5/5 transfer failure).
- **Gate (proceed):** R@10 **> 1.5 %** — it must at least beat the quotient it augments.
- **Kill:** R@10 **< 5 %?** — the cue cannot reach a third of the incumbent; the loop
  starves at step one and no downstream wave can compensate. Stop; file the cue gap.
- **Disable:** replace the cue with a permuted scent table (same marginals). PASS must go
  red. If it stays green the cue is not what is being measured.

### W2 — rail in the loop, on the river (D-HXP-3, D-HXP-4)

The step H5d skipped. Runs on the workload where the quotient measurably lost.

- **Substrate:** the hex-tenant probe's tenant (`6×(u8:u8)`, Morton rows), first byte
  re-read as `scent(target)`; strength bumped Hebbian on the delta frontier exactly as the
  probe already does; **eviction:** weakest of six loses when a candidate's bundled
  strength exceeds it. Magnitudes **bundle, never XOR** (`I-SUBSTRATE-MARKOV`).
- **D-HXP-3 gate (PASS):** on river same-board other-entries (0 % escape), rail-in-loop
  future prediction **< 3.75 bits** (beats C2p). **Kill:** **≥ 4.68** — no better than the
  quotient it replaces.
- **The E-Q8 control is mandatory and is already in the probe:** *"the same run with ONE
  direction; any advantage that survives it is not hex."* If one rail matches six, the six
  are decoration.
- **D-HXP-4 — the off-trie rate, the synapse-vs-cache question:** for every learned rail,
  is its target reachable from the source's HHTL prefix at that depth?
  - rate **= 0** → reclassify C1 as **cache of descent**. Not a kill — a cache that beats
    C2p is worth having — but the word *synapse* is withdrawn from every doc that used it.
  - rate **> 0** → report it; that rate **is** the claim, quantified.
- **Disable:** freeze strength (no Hebbian bump). D-HXP-3 must go red; if it stays green
  the rails learned nothing and the win was the address side.
- **Anti-vacuity:** the top-6 histogram from W0 bounds what six rails *could* carry; a
  measured win above that bound is an apparatus error, not a discovery.

### W3 — ARM promotion: cue → validated path (D-HXP-5)

- **Encoding:** transactions = one row per (state, epoch): items are the state's live rail
  targets (scent ids) plus the observed next state. `FeatureSpec` declares scent and
  successor as features; `Dataset.rows: Vec<Vec<u32>>` is filled directly — no
  serialization, no new DTO.
- **Mining:** `CandidateRule` with `antecedent` = rail set, `consequent` = successor;
  `cooccur` is NARS `m`; `arm_to_truth_u8(rule, k)` yields `⟨f, c⟩`. A rail is **promoted**
  when its rule's `c` clears a bar set from the `TruthU8` scale (pin, then measure).
- **Gate (PASS):** promoted rails, applied to **held-out** positions, reduce escape and beat
  a **same-count shuffled promotion** on future-prediction bits. This is the H5c shape
  (learned vs permuted) applied to promotion, and it must clear the bar H5c failed.
- **Kill:** promoted ≤ shuffled → validation adds nothing over Hebbian strength alone; C2
  fails even if C1 passed.
- **Disable:** promote by `cooccur` only (frequency, no `c`). If the gate still passes,
  confidence is decoration and the operator's "frequency **and** confidence" is not what
  won.

### W4 — the flow channel (D-HXP-6)

The operator's "imagine": known unknowns → a new thought → reinforced by flow.

- **Mapping:** challenge := escape rate; skill := hit rate (measured 0.72 / 0.91 on the two
  code eyes at k=3). `FlowState` per fixture: 0 % escape (same board) → `Boredom`;
  ~50 % (code eyes) → candidate `Flow`; 85 % (unseen river) → `Anxiety`.
- **Procedure:** run W2+W3 on all three buckets; measure **promotion yield** (validated
  rails per epoch) per bucket.
- **Gate (PASS):** yield peaks in the middle bucket and is low at both ends.
- **Two-sided kills:** yield **monotone in escape** → the flow model is wrong (more unknown
  = more learned, no channel). Yield **flat** → `FlowState` carries no information here and
  must not be cited as a mechanism.
- **This wave asserts nothing it does not measure.** `FlowState` exists; whether it
  *predicts* promotion is the question, not the premise.

### W5 — host decision (deferred; re-scopes #26)

If W2 passes: is the rail program a `MaskOp` program (PR #1226's executor, shared
`validate`, scalar oracle)? That would satisfy #26's constraint — *no ndarray module, no
hexagon-named type* — by construction. **Not decided here;** it is a probe on top of a
passed W2, never a premise.

## 4. Iron rules in force (each one already cost a session)

- **Magnitudes bundle, never XOR** — `MergeMode::Xor` breaks the Chapman-Kolmogorov
  semigroup (`I-SUBSTRATE-MARKOV`). Strength is a magnitude. Sign/phase may XOR.
- **A word-level op pays for the span it is given; give it the node, never the field**
  (hex-tenant STORNO: −14 % over the field, −66 % over the node span).
- **Commit before you disable; assert every anchor** (ruff `AGENTS.md`; PR #1226's own
  `AGENT_LOG`). A disable that did not apply is indistinguishable from an inert guard.
- **Zeroing a constant is not a disable** when the guarded quantity can pass by another
  route (quality-wave-v1, hit 1).
- **A null result is a claim about the apparatus until proven otherwise** — every kill
  above names its positive control.
- **The fixture's shape is part of the test's coverage.** W1's fixture must be SPO/motif
  shaped or the codebook ceiling, not the cue, is what gets measured.
- **No new crate. No `hexagon` type. No `ndarray::mq`.** Feasibility §H stands.
- **`CausalEdge64` is not touched.** Nothing here writes, reads for a verdict, or re-grades it.
- **Zero-copy:** the rail is a lens over the tenant bytes; W3's `Dataset.rows` is the one
  licensed materialization (it is the mining input, not a second copy of the substrate).

## 5. Model / tier policy

Stated by role, per this workspace's convention; tiers as in lance-graph `CLAUDE.md`
§ Model Policy. **Planning tier:** this plan, every gate adjudication, every review, the
W4 interpretation. **Grind tier:** W0 histogram script, W1 harness against the frozen
fixture, W3 transaction encoder — one source in, one shape out, a written spec each.
**Churn tier, contract-gated only:** re-running a specified disable arm and reporting the
tail. Every grind brief carries verbatim: *do not run cargo across a sibling workspace;
do not commit; do not claim it compiles.* Disjoint files per worker; the orchestrator
edits shared files after workers land and is the sole writer of board files.

## 6. What this plan deliberately does not claim

- That the six rails *are* synapses. W2's off-trie rate decides the word.
- That ARM *will* find edges. W3 has a shuffled control precisely because H5c lost to one.
- That flow is a mechanism. W4 can return "no channel," and that would be filed as-is.
- That any of this lands in `ndarray`. Feasibility §H: F0 there, F4 in lance-graph.
- That six is the number. W0 measures `k`.

## 7. Landing checklist (only on operator go)

1. This file → `.claude/plans/hexagon-plasticity-v1.md` (already at path, uncommitted).
2. `INTEGRATION_PLANS.md` prepend; `STATUS_BOARD.md` rows D-HXP-0 … D-HXP-6 (Queued).
3. `EPIPHANIES.md` prepend: the F-MQ8 + H5b **re-reading** — *"the vocabulary is the
   recognition organ and is corpus-bound; the law is the transfer organ and is corpus-free"*
   — a division of labour, previously filed as a null. Cites this plan.
4. `#26` re-scoped: the `graph/refine/` kernel waits on W5, not on the feasibility doc alone.
5. `python3 .claude/tools/supersession_index.py > .claude/board/SUPERSESSION-INDEX.md` —
   **last**, after the board writes, or CI goes red on the coverage column (#1085).
6. PR title carries no model identifier. Branch: `claude/ladybug-transcoding-plan-q5zbrs`.

## 8. Provenance ledger — every number, where it came from

| number | source |
|---|---|
| 0.48 % vs 0.77 % (H5c); 463/958/427 fan-out; 11.1 M < 2²⁴ (H5a); R@10 1.5 % / 18.1 % / 0.15 % (H5b); *"never in the loop"* (H5d); F-MQ8 5/5 vs 12/12; R3 3.1 % / 86 % | `MQ-REPORT.md` §4 table, §8 H5a–H5d, §12, Ruling |
| river 4.68 / **3.75** / 0 % escape; unseen 13.80 / 12.50 / 85 %; code H 1.72 / C2p 1.95; hit rates 0.72 / 0.91; 51.1 % escape at depth 12 | `MQ-REPORT.md` §4 transfer table and reading; §8 H5b |
| F4 / F0 verdict; *"Hexagon belongs only … for the `6×(8:8)` rail register"*; 4 B/state, 5 B/transition, 350 ms / 118 MB | `HEXAGON-MQ-FEASIBILITY.md` §H |
| range write 49–99 ns vs 22.5 µs (228–462×); shift −14 % field / −66 % node span; 0.48 coal at x = 4; Hebbian on delta frontier; E-Q8 one-direction control | `EPIPHANIES.md` 2026-09-14 entry + STORNO; `ndarray/examples/hex_tenant_mq_probe.rs` module doc, the "Hebbian, one owner, no second structure" rail paragraph |
| `Scent: u8`, ρ 0.937, byte 0 of ZeckF64, "heuristic pre-filter"; W176-191 neighbor indices | `docs/CODEC_COMPRESSION_ATLAS.md` row `| Scent | u8 |` and the Full→Scent ratio row; `.claude/agents/container-architect.md` phrase "scent/palette neighbor indices" |
| ρ 0.9973 / 0.965 anchors; `TD-BASE17-FOLD-CEILING-SINGLE-WORD` ρ 0.2599 | `.claude/knowledge/bf16-hhtl-terrain.md` under `TD-BASE17-FOLD-CEILING-SINGLE-WORD` |
| `FlowState { Flow, Boredom, Transition, Anxiety }`, `#[repr(u8)]`, D-CSV-13b | `lance-graph-contract/src/mul.rs` at `pub enum FlowState` |
| `CandidateRule { antecedent, consequent, cooccur (NARS m), antecedent_count, window }`; `Dataset { spec, rows }`; `arm_to_truth_u8` | `lance-graph-arm-discovery/src/rule.rs` at `pub struct CandidateRule`; `encode.rs` at `pub struct FeatureSpec` / `pub struct Dataset`; `translator.rs` at `pub fn arm_to_truth_u8` |
| ARM has zero `src/` callers; osint declares it `[dev-dependencies]` only | measured this session (grep over `crates/`); `markov_soa.rs` (a doc comment, not a call site) is a doc comment |
| #1226: shared `validate`, scalar oracle, 15 disable runs, 0.9 % interpreter overhead (no n) | PR #1226 body, CI at `061d12b` |

---

## 9. W-1 / W0 RESULTS (run 2026-09-14, before landing — appended, nothing above rewritten)

### W-1 — REPRODUCED EXACTLY (D-HXP-0)

All four arms complete (the fourth confirmed 21:43 UTC, after landing). Compared by `lab/w1_compare.py` (self-tested
anchor-against-itself = zero diff; falsifier: +0.5 on one anchor value yields exactly one
`DIFF` row and `DRIFT — 1 rows differ`).

| arm | result |
|---|---|
| 12-library transfer | **byte-identical**, 12/12 rows; H 0.67–1.10 vs C2 1.35–3.13, H<C2 on 12/12 |
| synthetic + river fixtures | **identical** across 4 fixtures × 2 depths × 6 keyers, escapes and state counts included (river same-board C2p 3.7497 / H 4.6761) |
| hydration: refine + structure + source-eye future | **identical** — 169,339 nodes, 74,054 classes, 51.1 % escape, 74,054 states / 125,061 transitions / 16.176 / 16.932 bits; S k=3 H 1.7165 / C2p 1.9514, k=6 H 5.4868 / C2p 6.208 |
| hydration: O3 future + override (H5b) | **identical** — O3 k=3 H 1.4473/0.9129, C2p 1.5608/0.8702, C4 12.2954/0.0608; k=6 H 2.5626/0.8581, C2p 2.6637/0.8613, C4 14.2150/0.0592; override (H5b) n=133 r@10 **0.0150** / mrr 0.0106 vs null r@10 0.0015 |

No threshold in this plan needed re-pinning. The data did not age. Final comparator run: **42 rows, every delta `+0.0000`, zero `DIFF`/`DRIFT`/`PENDING`**; the synthetic and 12-library JSONs are byte-identical to their frozen anchors (`cmp`, not eyeballed).

**Apparatus note for a future rerun.** The hydration arm takes **~60 min** (3595 s here, 3306 s for the anchor) and peaks at **3.9 GB RSS** — so it is not memory-bound on a 16 GB box, and an earlier attempt that died silently mid-run was **not** an OOM. Run it detached with a peak-RSS reporter and a sampler; a watch that only greps for the success line cannot tell a crash from a slow run.

### W0 — CAPACITY **PASS**; SIGNAL **UNANSWERABLE BY THIS INSTRUMENT** (D-HXP-1)

**Capacity — PASS, and it is robust.** Raw top-6 transition mass, k=3 by state:
**S 0.9792, O3 0.9943**. The 0.40 kill is nowhere near. Six slots hold the mass.

**Signal — the gate fired twice, and the second firing diagnosed the first.**

| fixture | edges/key | observed top-6 | uniform-expected | permuted null (20×) | z |
|---|---|---|---|---|---|
| S | 185.0 | 0.9792 | 0.7742 | 0.6956 ± 0.0003 | 1121 |
| O3 | 267.7 | 0.9943 | 0.9671 | 0.8421 ± 0.0000 | 5011 |
| synth_structured | 427.0 | 0.9386 | 0.8333 | 0.3859 ± 0.0005 | 1195 |
| **synth_unstructured** (the null material) | 7.6 | **0.9035** | 0.5884 | 0.4718 ± 0.0007 | **591** |
| river_unseen | 4.7 | 0.9030 | 0.8889 | 0.8153 ± 0.0004 | 251 |
| river_same_board | 165.8 | 0.8892 | 0.8867 | 0.1147 ± 0.0003 | 2653 |

Null 1 (`uniform_expected = min(6,d)/d`) failed: the unstructured control showed
signal **+0.3151**, larger than the source eye's +0.2050. Diagnosis: with `s` samples over
`d` observed values, multinomial fluctuation guarantees concentration, so the reference is
biased low, and the bias scales inversely with edges-per-key (7.6 there vs 185 on S).

Null 2 (marginal-preserving permutation — shuffle the true target multiset, deal back at
each key's exact edge count, R=20) also failed: unstructured **z = 591**, no fixture
anywhere near |z| < 2.

**Root cause, measured — the quantity is partly DEFINITIONAL.** Distinct successors per
key, k=3 by state: median **3 / 3 / 4 / 2 / 3 / 3** (S / O3 / structured / **unstructured** /
river-unseen / river-same-board), p90 5–11. A bisimulation class groups nodes by their
labelled successor-class multiset, so a class has few distinct successor classes **by
construction** — the median of 2 on a *random* graph is the proof. Top-6 mass is therefore
high on any refined graph, and no null over the refined representation can separate
"learned concentration" from "this is what refinement does."

**Consequences — the plan is corrected here, not defended.**

1. **The W0 anti-vacuity gate is STRUCK as malformed**, not left as a pending FAIL. It asked
   a question its own instrument cannot answer. Struck with its reason, per append-only.
2. **W0 is re-scoped to CAPACITY ONLY**, and at that it is a clean PASS. `k` stays 6: six
   ≥ p90 distinct-successors on four of six fixtures.
3. **The signal question moves ENTIRELY to D-HXP-4 (off-trie rate).** That is now the only
   measurement in the plan that can distinguish *cache of descent* from *new synapse*, and
   it is immune to this confound because it asks whether a learned rail leaves the trie —
   a question about the rail, not about the refinement's successor statistics.
4. **A second, independent finding worth its own row.** On the river fixtures — the
   tile-local workload W2 targets — six slots capture the LEAST contested mass of any
   fixture: river_same_board top-6-within-contested **0.6373** (by label 0.8155),
   river_unseen 0.6622, against S 0.9647 and O3 0.9023. On the very workload where the plan
   expects a learned rail to win, six rails are measurably weakest. Not a kill — it is the
   number W2 has to beat, and it is now on the record before W2 runs.

### W3 gains its missing propagation rule (EWA-sandwich)

The plan promoted a rail to a *path* without saying how confidence survives multiple hops.
That was a real hole: naive accumulation is O(n) additive and loses signal past ~5 hops —
`crates/jc/src/ewa_sandwich.rs` module header ("meaningful at depth >5") states exactly that ceiling ("meaningful at depth >5,
where naive convolution would have lost signal").

The answer is already certified and already zero-dep, so it costs no new dependency:
`lance_graph_contract::sigma_propagation::{ewa_sandwich, pillar_5plus_bound}` — bit-identical
to the certified kernel with a disable-verified drift test. The sandwich `Σ_n = M·Σ·Mᵀ`
preserves PSD by construction (**10000/10000 hops**, σ_step 0.2, 1000 paths × 10) and gives
**geometric** rather than arithmetic error control — bounded Σ instead of per-hop noise
accumulation. `pillar_5plus_bound(n) = √(2/n)·√(1+2σ²n)` FALLS with depth — 0.7483 at n=5,
**0.5715 at n=12** — which is what makes depth-12 paths admissible where 5 was the naive
ceiling, and it matches the 12 nibble levels of the HHTL address and the MQ refinement's
own `DEPTH = 12`.

**W3 therefore gains:** a promoted path carries a runtime concentration certificate
(`log_norm_growth` against `pillar_5plus_bound`), and a path that loses concentration past
the pillar's `1.75×` slack is rejected rather than promoted. Stated as a rule, not a
measurement — whether real rail paths stay inside the bound is W3's to find out.

**One provenance correction, recorded because it was asked and checked:** there is no
`0.465` literal anywhere in `crates/jc`, `ndarray/src/hpc/{splat3d,pillar}`, or
`.claude/{plans,knowledge}`. Nearest constants are 0.4472 (`√(2/10)`, this pillar's own CV
base), 0.4748 (Shevtsova, `jirak.rs`), 0.4570 (a spherical-harmonic coefficient, unrelated).
The pillar's *measured* CV is printed only when the pillar runs and has not been read in
this container.

---

## 10. D-HXP-7 — the EWA concentration gate (added 2026-09-14, replaces the struck W0 signal gate)

§9 struck the W0 signal gate as malformed and moved the whole signal question to D-HXP-4
(off-trie rate). That was too much punt: off-trie asks **where a rail points**, and nothing
then asked **whether composing rails preserves information**. This section adds the second
question back in a form the confound cannot reach.

### Why a path statistic and not a better histogram

§9's confound is **per-node and definitional**: a bisimulation class has few distinct
successor classes by construction (median 2 on a *random* graph). Any statistic over a
node's successor marginal inherits that. `log_norm_growth` against `pillar_5plus_bound` is a
**path** property — is concentration maintained across `n` hops — and a path property cannot
be confounded by a per-node one. This is the question the histogram was trying and failing
to ask.

Three further reasons the rank statistic was the wrong instrument here, recorded because
they generalize past this wave: a median **does not compose** (median of medians ≠ median,
so it breaks under exactly the partitioning every other operator in this substrate
survives); it is **O(n log n) sort** where the stack is O(1) `compose_table[a*k + b]`; and it
carries **only a marginal**, discarding the second moment — which is the quantity that
separates learned concentration from definitional narrowness.

### Seed: the two axes are already everywhere

`Spd2 { a, b, c }` is 2×2. A rail is `(u8:u8)`. NARS truth is ⟨f, c⟩. A tier is a 256×256
tile — two axes. Seeding Σ from (frequency, confidence) is the substrate's own pairing, not
an import. **It is still a modelling choice and must be defended, not assumed**: the seed
map (⟨f,c⟩ → `Spd2`) is declared once, in one function, with its own unit test, and any
alternative seed is a separate arm — never a silent swap.

### D-HXP-7 procedure

Runs on the W-1 successor tables (already on disk — no new hydration).

1. Seed `Σ_0` per state from that state's ⟨f, c⟩ (frequency = transition share, confidence =
   the NARS evidence discount over its edge count), through the one declared seed map.
2. Walk observed held-out paths of length `n = 1 … 14`. At each hop apply
   `contract::sigma_propagation::ewa_sandwich(M_k, Σ)` with `M_k` the step Jacobian of the
   rail taken. **Zero new dependency** — the kernel is zero-dep in the contract, bit-identical
   to the certified `jc` pillar with a disable-verified drift test.
3. At each `n` record `log_norm_growth(Σ_0, Σ_n)` and the PSD-preservation rate.
4. Compare the measured CV of `‖log Σ_n‖²_F` across paths against
   `pillar_5plus_bound(n) = √(2/n)·√(1+2σ²n)` — 0.7483 at n=5, **0.5715 at n=12** — with the
   pillar's own **1.75×** slack.

### Gates (two-sided)

- **PASS:** PSD-preservation ≥ 0.999 AND measured CV ≤ 1.75 × bound **at n ≥ 10**. Composing
  rails preserves concentration past the naive-convolution ceiling
  (`jc/src/ewa_sandwich.rs` module header, the "meaningful at depth >5" sentence: *"meaningful at depth >5, where naive convolution would have
  lost signal"*).
- **KILL:** CV exceeds the slack at **n ≤ 5** — rail composition is no better than naive
  accumulation, and multi-hop promotion (W3) has no carrier. W3's path arm stops; the
  single-hop arm may still proceed.
- **BETWEEN:** report the largest `n` that holds. That `n` is the substrate's real path
  budget and becomes a measured parameter of W3, not an assumption.
- **Anti-vacuity (mandatory, and it is the lesson of §9):** run the identical procedure on
  the **synthetic unstructured** fixture. Random material must FAIL the bound at low `n`. If
  random paths also hold to n ≥ 10, the gate is measuring the seed map or the arithmetic,
  not the rails — and this gate is struck like its predecessor. §9's failure was caught by
  exactly this control; it is not optional here.
- **Disable run:** replace `ewa_sandwich` with plain additive accumulation
  (`Σ_n = Σ_0 + Σ M_k`). PASS must go red at n ≥ 10. If additive also holds, the sandwich is
  not what is carrying the result.

### The LUT arm (optional, same question, O(1) per hop)

`bgz17::PaletteSemiring::compose(a, b) = compose_table[a*k + b]` — *"palette index of
path(a → b)"* — is path composition as a k×k lookup. Measured this session: the driver holds
the struct but touches only `.k`, `.compose_table.len()` and `.distance_matrix`; **`.compose()`
has no live caller anywhere**, so H5c's *"nothing in the tree composes facets"* is still
literally true. Palette-coding the successors turns "does it concentrate" into a popcount
over the reachable code set — a mask op, T1 population algebra, the shape #1226's executor
already runs, with no sort. If taken, this is `compose`'s first real consumer. **Optional and
second:** the EWA arm is the gate; the LUT arm is an O(1) restatement to be run only if the
EWA arm passes.

### What D-HXP-7 does NOT claim

That the EWA arm will pass. It replaces a malformed question with a well-formed one.
Whether real rail paths stay inside the bound is unmeasured, and the anti-vacuity control
above exists because the last instrument that looked sound was not.

## 11. W1 RESULTS (run 2026-09-15, on operator go — appended, nothing above rewritten)

### Verdict: **PROCEED, not PASS** (D-HXP-2)

Best cue **r@10 = 0.1579** against the BPE incumbent's **0.1805** — **87.5 %** of a
corpus-learned dictionary. Gates: PASS (≥ 0.181) **not met**; proceed (> 0.015) met by **10×**;
kill (< 0.05) **not triggered**. Positive control `ARM0` reproduced the incumbent at
**0.1805** (`REPORT2` A2a-f records 0.180) byte-identically across both runs including
`mrr`/`r@1`/`r@5`/null, so every row below is comparable to it.

### ⊘ STORNO — three items of this plan are superseded by operator correction (2026-09-15)

Verbatim: *"not a scent table and no cosine / fisherz is normalized for cheap lut / the idea is
that palette256:palette256 can spread as activation over HHTL not hierarchically but adjacent
local akin to a synapse"*, and *"cognitive pertubation"*.

1. **§3 W1's procedure** — "scent/codebook cue", arm (a) "`Scent` byte alone", and the disable
   "permuted **scent table**" are struck. There is no scent table. Distance is a **table read**
   (Fisher-z), never a computed cosine.
2. **§2 row 1** — "get the cue → `Scent: u8`" is struck as the cue carrier. The cue is
   `palette256:palette256` — a **`Palette256Pair`**, which ships: `SpoFacet` in
   `lance-graph-contract::awareness_facet` is six of them (3 SPO + 3 episodic-witness), and its
   own doc says similarity between two pairs is *"one table read (Fisher-z cosine-replacement),
   never a float."*
3. **§2 row 2** — "recognize it → mask predicate / resonance" is struck. Recognition is not a
   predicate evaluated per cell; it is a **neighbourhood lit around where the ball is** —
   attention as table lookup, not attention as scan.

Nothing above is rewritten; these three rows are regraded in place by this entry.

### The result: the ORDER-SENSITIVE arm wins, and is the only falsifiable one

| arm | r@10 | its falsifier |
|---|---|---|
| **transitions** (consecutive cell pairs) | **0.1579** | order-shuffle → **0.0827**, halved. **FIRED** |
| field n=3, radius 1 | 0.1353 | F1 did not fire |
| bag n=3 (radius-invariant by construction) | 0.1128 | F1 did not fire |
| LUT soft-match n=3, radius 0 | 0.0752 | F1 did not fire |

Every **order-blind** representation the plan's own W1 text implied tops out at 0.1353 and
**none of their falsifiers can fire**. The one **order-sensitive** representation is both the
best score and the only one that dies when broken. The operator's ball-on-tilted-ground framing
(torque = direction, which a bag structurally cannot hold) is vindicated *against* the
static-field design this plan shipped with — by measurement, not by argument.

### Three findings against the palette half

**1. F1 never fired — on any scorer, including the LUT scorer added specifically so it could.**
All three rose under a marginal-preserving permutation. Mechanism, understood not guessed:
permuting the unit→cell **assignment** is a *global relabelling*, so every exact-match collision
survives it. **Consequence: the measured cue signal is exact unit-type reuse.** The tile, the
seriation, the LUT and the spread contribute nothing that survives permutation. A disable that
cannot reach what the scorer actually uses is not a disable — the fourth instance of that trap
in this arc, and the first where it survived into a landed spec.

**2. Adjacency is REAL, and crediting it still HURTS.** LUT soft-match is the worst arm
everywhere and degrades with radius (n=3 0.0752 → 0.0677; n=1 0.0301 → 0.0150, halved); field
n=1 degrades at radius 2. Only field n=3 gains (0.1278 → 0.1353) before plateauing.
**Robustness:** the degradation appears in the Ruzicka min/max field scorer too, which contains
no cosine at all — ⊘ *so it is not a normalization artifact.* **STRUCK (§11a): Ruzicka is
Σmin/Σmax, itself a normalization. The supported claim is only that the effect is NOT
COSINE-SPECIFIC; a normalization effect is not ruled out. Text kept in place because §11a quotes
it.**

**3. n=3 beat n=1 in every comparison** (all scorers, all radii). The motif-shaped fixture is
doing real work and the `TD-BASE17-FOLD-CEILING-SINGLE-WORD` ceiling was correctly avoided.

### The confound was measured and resolved AGAINST the hedge

"Adjacency hurts" is a null, and §4's iron rule says a null is a claim about the apparatus until
proven otherwise. The suspect was the greedy seriation — a greedy nearest-neighbour chain is a
greedy TSP path and degrades once good candidates are exhausted. Measured
(`lab/seriation_quality.py`, `runs/w1-seriation.json`, deterministic, two runs byte-identical):

- **0 / 2850 degenerate pairs** — no sentinel contamination; every Fisher-z well-defined.
- adjacent-pair z **0.6352** vs a 20-shuffle null of **0.0794 ± 0.0247** → **22.49 null standard
  deviations** — a standardized separation from a *sampled* null, NOT a calibrated tail
  probability. Empirical exceedance: **0 / 20** shuffle means reached it (max null **0.1329**),
  which bounds the permutation p at **< 1/20** and nothing finer.
- top-5 neighbours within ±3 positions: **41.84 %** vs **7.79 %** chance → **5.37×**.
- mean z decays **0.635 → ~0.13** by chain-distance 5, then a flat tail.
- first sub-mean link at **53 / 75**, and **zero** links fall below the overall mean before it —
  a clean prefix, which is the stronger half of this metric.
  ⊘ *only the last 29 % of the chain degrades.* **STRUCK: 29.33 % (22/75) is the SUFFIX LENGTH,
  not the degraded count. 6 of those 22 links are still ABOVE the mean, so index 53 is a FIRST
  CROSSING, not a sustained break. True below-mean share: 16/75 = 21.3 %** — recomputed from the
  committed `link_zs` in `w1-seriation.json`, no re-run required.

The seriation is sound. The null therefore stands as a finding about the **mechanism**, not the
instrument.

### Why crediting real similarity hurts — and what it names

The task is **discrimination, not similarity**. An override twin is identified by what separates
it from ~5 000 other bodies. Smearing a body's signature across its behavioural neighbourhood
makes every body look more like its neighbours: spreading buys coverage by spending precision.

**A spread without a surround is a blur.** Center-surround exists so a field can grow *and* stay
sharp. W1 built the excitatory half only, over a demonstrably good ordering, and measured the
cost — which makes `ndarray`'s **Pillar-15** (Difference-of-Gaussians center-surround
unimodality, `hpc/pillar/mexican_hat.rs`, status **DEFERRED** pending the kernel landing in
`ndarray::hpc`) not a refinement but **the missing half**. W1 is the first measurement of the
hole where it belongs.

### Inhibition is already free — the mask algebra IS an inhibition algebra

Named `ternlog` truth tables in `ndarray::simd` — `AND3`, `AND2_ANDNOT` (`a & b & !c`,
one-source inhibition), `AND_ANDNOT2` (`a & !b & !c`, the center-surround shape), `MAJ3`
(bundle-and-threshold), `XOR3`, `OR3`, `AND2` — are truth-table **immediates**; on the **AVX-512 `U64x8`
path** each is one `VPTERNLOGQ` per 512 bits. Not universal: `U32x16` uses `VPTERNLOGD`,
non-AVX-512 x86 expands into two-input ops, NEON/WASM/scalar are narrower or scalar (the
source's own *"the polyfill elsewhere"*), and the sibling `ndarray` checkout is unpinned.
Inhibition uses a **ternlog immediate rather than a separate circuit**; biology pays for it with
separate interneurons and a separate transmitter. The shipped delta-frontier spread (`scratch & !state`) **is**
`AND2_ANDNOT` — refractoriness — and `E-HEX-TENANT-RAIL-IS-DIRECTION-CHAIN-IS-FREE-SHIFT-IS-THE-COST-1`
already measured it at **−48 % for identical closure**. Note that is a **cost** result, not an
accuracy one: frontier inhibition is proven *free*, not proven *better*.

**Boundary rule that binds any use of this** (unchanged, `E-TOPOLOGY-MASKS-MAGNITUDE-COMPOSE-NEVER-COLLAPSE-1`):
the rail bytes are **eligibility + strength, never a weight**; the **FC** (r2il **SPOFC**,
deepnsm-v2 COCA frequency × coverage, `lance-graph-arm-discovery` support × confidence — three
producers, one `NarsTruth` carrier) stays the **white half**. Population-level inhibition is
`ANDNOT`; magnitude-level inhibition is subtraction or division; **neither is XOR**
(`I-SUBSTRATE-MARKOV`) — `XOR3` is sign/phase only.

### Honest scope and residuals

- The LUT scorer skipped radius 2 by a **pre-declared, measured** decision (n=3 r0 13.5 s → r1
  215.9 s, 16× growth; r2 extrapolated to tens of minutes). `bag`/`field` cover all three radii.
- **The LUT aggregation was cosine-normalized** (`/√(ΣA²·ΣB²)`) despite the operator's "no
  cosine". The *distance* is a genuine table read — which is what the correction was about — but
  the aggregation is not. Recorded rather than quietly corrected; finding 2 survives it because
  the non-cosine Ruzicka scorer shows the same direction.
- This lab is pure Python with no `numpy`/`sklearn` and **no Rust fingerprints**. Everything here
  is a **shape proxy** — evidence about the addressing shape, never a measurement of
  `ndarray`/`bgz17`/`helix`.
- **The transitions arm was falsified on ORDER but never on the TILE.** F1 (permuted cell
  assignment) was run on `bag`/`field`/`lut` at n=3 radius 1, not on transitions. So its 0.1579
  is proven to depend on *sequence order*, and is **not** proven to depend on the palette cells
  the pairs are built from — if a cell-permutation also left it at 0.1579, its signal would be
  "order of unit types", not "order of palette cells". That distinction matters for W2 and is
  UNMEASURED. Noted rather than left to be cited later as stronger than it is.
- **A permuted control ties the best legitimate arm.** F1-permuted `field` also reaches 0.1579
  (both are 21/133). That is consistent with — indeed another statement of — finding 1: the tile
  structure contributes nothing that survives permutation.
- Run 1 (`runs/w1-cue.json`) is kept as the record of three apparatus defects, all spec errors:
  an inert LUT, a spread whose radius (1–2) was smaller than the cell spacing (~3.4) so it
  diffused into empty space, and an order falsifier applied to an order-blind representation.

### What W1 hands W2

The rail-in-the-loop wave inherits a **sharpened** question, not the one §3 wrote. Lateral
spread as pure excitation is measured to *cost* discrimination on a good ordering. W2 should not
re-measure that; it should either carry a surround (the deferred DoG) or state plainly that it
is testing coverage rather than discrimination. And the transitions result says the rail's value
may be in the **pair** — a traversal from one palette cell to the next — rather than in the cell.

## 11a. ⊘ W1 AMENDED — the palette cue is FALSIFIED (run 2026-09-15, after review)

§11 marked D-HXP-2 **Done — PROCEED** while its own residuals recorded that the winning
`transitions` arm had never been run against the cell-assignment disable. Codex flagged the
contradiction on PR #1233 (P1) and was right: this gate's disable says *"permuted [assignment]
must go red. If it stays green the cue is not what is being measured."* The disable has now been
run on the winning arm. **It did not go red.**

| arm (all `transitions`, all `metrics2.retrieval`) | r@10 |
|---|---|
| **A** baseline, real palette cells | 0.1579 (reproduced exactly) |
| **B** cells permuted, marginals held (the missing disable) | **0.1654 — rose** |
| **C** bare dense integer type-IDs, **no palette geometry at all** | **0.1729 — beats A** |
| **D** identity-IDs + order-shuffle | 0.0902 |

**7827 distinct unit types over 1305 occupied cells — a 6:1 lossy hash of identity.**

**Verdict: the `palette256:palette256` cue contributed nothing beyond being a consistent label,
and cost a little by being lossy.** Destroying the content→cell mapping did not hurt; discarding
the geometry entirely helped. **D** reproduces the order-drop on bare identity (−0.0827, against
the palette arm's −0.0752), so **order was never the palette's contribution either**. The whole
W1 signal is **order of unit types**.

### What this kills, and what it does not

**Killed** — `palette256:palette256` as a cue addressing, for this override-twin task, in this
shape proxy, with this (left-context-byte, right-context-byte) cell construction. Stated as a
kill rather than reframed, because it was one of the three conditions pre-committed before the
run: *"transitions survives a cell-permutation → the palette contributes nothing."* It survived.

**Not killed** — order/sequence as signal (it is now the *more* robust finding: it survives on
bare identity with no palette at all); the rail as a substrate structure for other purposes; the
real Rust codec, which this lab never measured; and the transfer question, which is untouched.

### The design critique, which is the more useful half

**A palette can only beat raw identity where unseen types must be generalized.** On same-corpus
retrieval every type is already known, so identity is the maximum information about type and any
quantization of it is strictly lossy — the 6:1 collision factor is exactly that loss, measured.
**W1's task was therefore structurally incapable of favouring the palette**: a generalization
mechanism was measured on a task requiring no generalization.

This does not soften the kill — the kill stands for this task and this construction. It relocates
where the claim could still live. **Transfer is not merely "the next test"; it is the only test
the palette could win**, and §11's PASS gate promised exactly that axis (*"without the
dictionary's corpus lock"*) while the procedure never tested it.

**Pre-committed kill condition for that test, so this stays falsifiable:** seriate two libraries
independently and compare the orderings. If the chains do not agree, the cue is as corpus-locked
as the dictionary it was meant to replace, and no retrieval number on any corpus rescues it.

### Review corrections applied in the same pass

- **Overclaim narrowed.** §11 said the Ruzicka min/max scorer "contains no cosine, so it is not a
  normalization artifact." Ruzicka is Σmin/Σmax — itself a normalization. The supported claim is
  only that the effect is **not cosine-specific**; a normalization effect is not ruled out.
- **Significance wording corrected.** "22.49 σ" implied a calibrated tail probability. The
  quantity is `(real_mean - null_mean) / stdev(null_means)` over **20** shuffle means — a
  standardized separation from a sampled null. Restated as **null standard deviations**, with the
  empirical exceedance now stated: **0/20** shuffles reached the real mean (max null 0.1329).
- **M4 is a first crossing, not a sustained break.** `fraction_of_links_at_or_after_break`
  measures the suffix length (22/75 = 29.33 %), and 6 of those links are still above the mean.
  True below-mean share **16/75 = 21.3 %**, recomputed from the committed `link_zs`. The probe's
  emitted field names are left unchanged — it is the audit record of the code that produced
  `w1-seriation.json`, and that run cannot be reproduced; the misnomer is marked at the M4 site.
- **Ternlog cost qualified.** "One `VPTERNLOGQ` per 512 bits" holds for the **AVX-512 `U64x8`**
  path only; `U32x16` uses `VPTERNLOGD`, non-AVX-512 x86 expands into two-input ops, and
  NEON/WASM/scalar are narrower or scalar. The source doc says *"the polyfill elsewhere"* and the
  first write-up dropped that qualifier; the sibling checkout is unpinned besides.
- **Board-rule compliance.** D-HXP-2's *description* cell was edited when the rules permit
  Status/Confidence edits only. Restored verbatim; the correction now lives in the Status cell.

### Artifact provenance — auditable, NOT re-runnable

The probes and their outputs are committed under `.claude/probes/hexagon-plasticity-v1/` so the
numbers above can be read and checked rather than taken on trust (Codex P2). **They cannot be
re-run from this repository.** Every measurement depends on the ~10 GB `r2harvest` ore corpus and
the `ore-full-v2` build, which live in an ephemeral session scratchpad, are not committable, and
have no immutable revision to cite. What is committed is the instrument and the result; what is
missing is the input. Said plainly rather than implying reproducibility this repo cannot offer.

## 12. D-HXP-8 — board games as the Raumgewinn falsifier (operator proposal 2026-09-15 — appended, nothing above rewritten)

Operator, verbatim: *"Der Vorteil von HDR popcount stacking early exit Belichtungsmesser
statistical confidence interval thresholds preheating rolling floor bucket assignment ist
jedoch daß es für hexagon Substrate hilfreich ist — der sprichwörtliche Raumgewinn beim Go
boardgame."* — *"Und witzigerweise müsste tiktaktoe gobbang, go damit sogar falsifiable
sein."* Board entry:
`E-POPCOUNTS-UPPER-RANGE-SIMILARITY-IS-THE-HEXAGONS-RAUMGEWINN-AND-BOARD-GAMES-MAKE-IT-FALSIFIABLE-1`.

**Why this is the falsifier E-Q8 asked for.** E-Q8 (§9's lineage) found the six doing no
work because the task consulted only the first neighbour. A game's ground truth NEEDS the
neighbours — a line, a liberty count, a Hex connection — so the degree-1 ablation is a
real gate here, not a memory.

**Encoding.** Each cell a unit; its six rails its neighbours (Hex: exact; Go: 4 of 6;
tic-tac-toe / Gobang: 8 directions as two rings); stone colour in the payload. Evaluation
= popcount stacking over rail agreement, ring by ring, through
`perturbation_sim::rolling_floor::TierFloors::stack_early_exit`, floors preheated from a
position sample; the winner criterion is `head2head::WinnerCriterion::SupportSpread`.

**Pre-registered gates.**
- **F1 correctness** — tic-tac-toe: top-ranked move value-preserving in ≥ 95 % of the
  765 positions, chance measured by a shuffled-rail null. Go end positions: stacked
  territory == flood-fill scoring, an equality.
- **F2 economy** — early exit changes no verdict (equality vs the full stack); mean
  exposed tiers < full depth, fraction stated.
- **F3 degree ablation, mandatory** — at degree 1, F1 must DROP; flat ⇒ the probe proves
  nothing.
- **KILL** — F1 at chance on tic-tac-toe, or F3 flat.

**Order.** tic-tac-toe → Hex (small boards) → Gobang → Go. Not started.
