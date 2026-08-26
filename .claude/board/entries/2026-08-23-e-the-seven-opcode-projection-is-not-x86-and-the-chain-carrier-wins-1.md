## 2026-08-23 — E-THE-SEVEN-OPCODE-PROJECTION-IS-NOT-X86-AND-THE-CHAIN-CARRIER-WINS-1 — four wave probes: the chain carrier confirmed, the vocabulary survives optimization, and the boundary that qualifies all of it

**Status:** FINDING — [MEASURED] × 4 (`PROBE-R2IL-OPTIMIZATION-TRANSFER-1` 5/5,
`PROBE-R2IL-DEFUSE-MACROS-1` 6/6, `PROBE-R2IL-SLAG-BOUNDARY-1` 5/5,
`PROBE-STAMP-CAPACITY-1` 6/6). Autoattended wave: 6 slices — 4 Sonnet probe
workers, 1 Sonnet scribe, 1 Opus synthesis worker, 1 Haiku guarded executor for
the gate sweep. Orchestrator compiled centrally, adjudicated every gate, fixed
the P0s, and is the sole writer of this entry.
**Confidence:** High for these two binaries at the pass-1 convention. The
fourth finding is precisely the reason that qualifier is not boilerplate.

**F-10 — the def-use chain carrier CONFIRMS its own prescription.** #1014
concluded "the macro carrier must be the def-use chain, never the linear opcode
window." Run as code on the same 143 episodes: 1,872 length-3 def-use chains
collapse to **27 distinct signatures** against the window's **97** (3.6× tighter),
and the chain top-10 carries **0.887** of all occurrences against the window's
**0.505**. Decisively: **95.9%** of the top chain's occurrences skip at least one
intervening op (median span 6, max 148) — they are invisible to a window matcher
at any width the data would justify. The prescription was not merely reasonable;
it is measurably the better carrier.

**F-9 — the idiom vocabulary survives optimization COMPLETELY (pre-registration
refuted in place).** TRAIN `stress_test` (71 fns / 3040 ops) vs held-out TEST
`stress_test_opt` (72 / 2300), same source, disjoint address keys. Predicted
PARTIAL survival on the theory that some top idioms are unoptimized-compilation
artifacts. Measured: **10/10** top-K transfer; of 64 TRAIN trigram types **61
survive** (3 TRAIN-only) while TEST carries **94**, of which **33 are TEST-ONLY**.
The optimizer did not prune the vocabulary — it ADDED to it, while cutting
ops/function 42.82 → 31.94. The density half of the prediction held; the pruning
half was wrong and is recorded in the probe's own docs, not adjusted away.

**F-11 — THE BOUNDARY, and it qualifies every finding in this arc.** The
harvest's addressed residual is **larger than its classified output**:
classified 17,557 rows vs residual count 36,747 (**ratio 0.478**), of which
**88.1%** is the single named reason `opcode_not_in_convention`; every
`by_address` shape sums EXACTLY to its `grouped` count. The furnace is behaving
correctly — the residual is convention-bounded and named, never dropped. But the
consequence is sharp: **F-1's 99.7%, F-2's type-collapse and F-10's chain
vocabulary are all measured over the seven-opcode projection, with roughly twice
that volume sitting outside it.** These are not claims about x86-64; they are
claims about a projection of two binaries. Stating that plainly is the finding.

**F-12 — the stamp loss curve.** Through the shipped `Stamp`: loss is exactly 0
for every N ≤ 64 and strictly positive past it — 33.3% at 96, **55.2% at 143**,
87.5% at 512. The 55.2% is the upper bound (every source hitting one macro);
#1014's measured ~26% is one real idiom appearing in fewer than all episodes.
Modelled 128/256-bit registers recover 15/0 dropped at N=143 — **modelled only,
not a proposal, not implemented, memory/cache/wire costs unmeasured.** Any width
change is the operator's ruling; this is input to it.

**Wave-process notes worth keeping:**
- A worker caught an error in ITS OWN BRIEF: I specified the slag `by_address`
  section as 5 columns; the file declares 4. It parsed what the file declares and
  said so, rather than reconciling silently. That is the brief being wrong and
  the guardrail working.
- The Haiku executor STOPPED at command 1 on a formatting failure (a worker file
  landed after my format pass), wrote its receipt, attempted no fix, and escalated
  — exactly its contract. Supervisor fixed and re-carded.
- Two pre-registrations were refuted across the wave (F-9 here, E3 in #1014). Both
  recorded in place. A wave that never refutes a prediction is not measuring.

**The meta-review changed the findings, not just the code.** A read-only Opus
reviewer returned FIX-THEN-LAND with 4 P0s: three gates that no input could fail
(one of which LABELLED a true-by-construction identity "the falsifier for this
whole section"), and this arc's own knowledge-doc header stating F-1 as an
unqualified claim about "real machine code" that the same document later
contradicts. Two findings were WEAKENED as a result and are recorded here in
their weakened form: **F-10 no longer claims "the prescription is confirmed"** —
the chain carrier's over-admission is 0 *by construction*, so what remains is an
uncontrolled concentration comparison (1,872 chain occurrences vs ~5,054 window
occurrences, no occurrence-matched control run); and **F-11's ratio compares ore
ROWS to residual COUNT UNITS**, which no gate establishes are the same unit, so
it is a magnitude comparison and is now labelled one. Every P0 and the
load-bearing P1s were fixed, every gate re-run green with strictly harder
assertions. A review that only confirms is not a review.

**Files:** `probe_r2il_optimization_transfer.rs`, `probe_r2il_defuse_macros.rs`,
`probe_r2il_slag_boundary.rs`, `probe_stamp_capacity.rs`,
`.claude/knowledge/r2il-behavioral-carrier.md` (the consolidated reference,
F-1..F-12).

