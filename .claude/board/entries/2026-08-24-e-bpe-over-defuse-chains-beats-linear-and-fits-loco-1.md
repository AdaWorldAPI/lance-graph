## 2026-08-24 — E-BPE-OVER-DEFUSE-CHAINS-BEATS-LINEAR-AND-FITS-LOCO-1 — R2IL x BPE POC, corrected twice (codex review + architecture review): real per-occurrence immediates, per-merge-id amortization, and the 90-slot number demoted from "ceiling" to "one encoding's headroom"

**Status:** FINDING — [MEASURED] (`PROBE-BPE-R2IL-LOCO-MICROCODE-1`, 10/10,
**re-measured after two review passes**). Originally landed with a
companion `PROBE-STAMP-MORTON-CASCADE-1` (7/7); the stamp-cascade design
fork was SPLIT OUT to its own PR per architecture review (orthogonal to
R2IL x BPE, muddied this entry's receipt) — see its own entry.
**Confidence:** High for the mechanism on this corpus (2 binaries, 143
episodes, pass-1 seven-opcode convention); exploratory, not architectural,
for the three cross-codec invariants (INV1-3, explicitly labelled as such).

**The architecture under test, operator-stated:** R2IL is the opcode — the
atomic, FAITHFUL vocabulary (thinking atoms + NARS; not P-code, not
SLEIGH). IR lands in V4, physically identical to V3 (16-byte content-blind
dock, classid selects the reading, no ENVELOPE_LAYOUT_VERSION bump). BPE
merges become candidate microcode macros — efficient recombination OVER
the faithful substrate. **R2IL carries faithfulness; BPE never needs to,
because it never has to be a second truth — B4's byte-exact round-trip is
the gate that keeps that boundary real.**

**Headline (B2/B3), relabeled — NOT unconditional compression
superiority (architecture review):** 33 BPE merges over 1,872 real
def-use chain occurrences achieve **113.4 tokens saved per merge**, vs
**50.1** for the same BPE algorithm run over the LINEAR opcode stream of
the same episodes. The prior "2.3x denser compression per
domain-vocabulary slot" framing OVERCLAIMED: the chain-occurrence input
(5,616 atom-slots) OVERLAPS — the same `Op` row can sit in more than one
def-use chain and gets counted more than once — while the linear input
(5,340 atoms) counts each row exactly once. Normalizing by merge count
does not remove that exposure-multiplicity confound. Corrected reading:
**"the def-use-chain carrier saves more tokens per merge on its own
(overlap-inflated) occurrence stream"** — an equal-source-budget control
would be needed before calling this true compression superiority.

**★ The `FnIndex` "90 free slots" number is DEMOTED — it is the headroom
of ONE ENCODING CHOICE (one `FnIndex` minted per macro), NOT the
autopoiesis vocabulary ceiling (architecture review, the single most
important correction of this pass).** `StyleLane::{Learned,Explore}` are
`[u8; 12]` — 12 palette256-indexed slots, one per `StyleFamily` ordinal
(`lance-graph-contract/src/soa_view.rs:41`), each byte selecting one of
**256** entries in that lane's OWN palette — a separate 256-wide address
space per lane, unrelated to `FnIndex`'s domain band. `ogar-loco` ROUTES
R2IL x BPE microcode into those planes; it does not have to OWN one
`FnIndex` per macro to do so. The corrected architecture:
`System[256, native Rust, fixed] | Learned[≤256, R2IL x BPE] |
Explore[≤256, R2IL x BPE]` — `ogar-loco`'s domain band is one possible
ADDRESSING scheme over that space, not its size. B1 still correctly
measures 7 R2IL atoms + B2's 33 merges = 40 of 90 slots under THAT ONE
encoding; read every "90"/"50 free" number in the probe file as scoped to
it, never as a vocabulary bound.

**B5 — the loco fit, RE-MEASURED (codex review, P2: `n_atoms - 1` is the
merge-tree's internal edge count, a structural constant, NOT the real
external `Call.values` arity, which comes from `Episode::ins` at the
opcode's actual operand sites):** of 33 learned macros, real-immediates
classification is 7 fit `Pairs` (≤1 real imm), **13 fit `Triples`** (was
26 under the wrong metric), **11 fit `Quads`** (was 0), **2 fit NONE**
(need >3 real immediates — was 0; ids 18 and 38). The "0 fit NONE"
framing was too triumphant, exactly as flagged — some real macros need
MORE than this pass-1 chain-length-3 extractor's span can address.
Immediates measured content-matched per occurrence (`macro_span_in_
occurrence`, order-independent — a merge id need not survive to an
occurrence's FINAL encoding, so a final-stream walk silently undercounts;
first fix attempt hit exactly this and was corrected before landing);
worst-case taken per macro (conservative for a `Call`); 0 of 33 macros
showed variable real arity across their occurrences on this corpus.

**B6 is candidate ranking, never admission.** MUL
(`contract::mul::GateDecision`/`Homeostasis`/`FlowState`) is the real
gate; the autopoiesis triangle (`ValueTenant::{FrozenStyle,LearnedStyle,
ExploreStyle}`) is **resonance-based thinking**, not an RL policy —
`PROBE-METACOGNITIVE-TRIANGLE-1` (#998) already proved `RungReceipt`-only
judging, `FreeEnergyComparison::minority_wins()`, and Explore running in a
**counterfactual lane** (`deposit_counterfactual` stamping `RawEdge -6`,
never observed truth). B6 ranks via shipped `TruthValue::revise`+`Stamp`
only; freeze/admit/promote/gate belong to MUL and the triangle, never
this probe. **Relabeled (codex review, P2):** B6's 27 groups are DISTINCT
FINAL-ENCODING SIGNATURES, not one-per-learned-macro (a group can bundle
several merge ids, or a merge id plus a leftover atom) — the earlier
label implied a 1:1 mint correspondence that doesn't hold.

**Three exploratory cross-checks against shipped precedent (INV1-3):**
- **INV1 (HighHeelBGZ's stride-as-role):** 33 macros collapse into 21
  atom-multiset role classes, largest holds 4.
- **INV2 (bgz17 `LayeredScope`'s scent-prune-then-escalate):** of 5,054
  linear-window candidates, a cheap opcode-multiset check prunes 4,053
  (80.2%) before the expensive exact def-use walk runs on the 1,001
  survivors (414 truly chained, 587 coincidental).
- **INV3 (BGZ-HHTL-D's shared-palette amortization), RE-MEASURED (codex
  review, P2: was grouped by B6's final-encoding signature — 27 groups,
  not 33 macros — the wrong unit for "episodes-per-mint"):** now counted
  per LEARNED MERGE ID directly (content-matched, same fix as B5): **33
  macros measured**, top-5 by occurrence carry episodes-per-mint
  **[86, 90, 63, 80, 56]** (was [63,80,74,47,30] under the old grouping) —
  real cross-episode amortization, now on the correct unit.

**Process notes (autoattended, auto-resolve, two review passes):**
- Container restart lost the first BPE-probe worker's entire in-progress
  write — re-dispatched from scratch.
- The re-dispatched worker introduced 2 real defects (an atom-count
  conflation; several raw `Option`/tuple Debug-format leaks) — hand-fixed
  by the orchestrator, reverified by recompiling rather than trusting
  self-report.
- Meta-review (Opus) failed **four consecutive times** to a server-side
  API 529 Overloaded outage — the PR was opened without it per operator
  direction rather than holding indefinitely.
- **A live `codex` GitHub review on the opened PR caught 3 real P2
  defects** (the two re-measurements above, plus the same
  final-encoding-vs-merge-id conflation appearing in both B6 and INV3) —
  all three fixed and re-verified by full recompile + rerun against the
  real corpus, not accepted on the reviewer's or a worker's word alone.
- **A separate architecture review caught the 90-slot framing
  over-claim** (routing capacity mistaken for vocabulary ceiling) before
  merge — fixed in both the probe's doc comments and this entry.

**Fences:** no mint actually performed anywhere (no classid, no
vocabulary table, no learner subsystem, no write to any
`ValueTenant`/MUL/triangle type); `ogar-loco`, bgz17's cascade shape,
`highheelbgz`'s `SpiralAddress`, `BGZ-HHTL-D`, and `ruff_r2il` are all
MIRRORED/CITED, never imported; corpus is 2 binaries at the pass-1
seven-opcode convention — nothing here is a claim about x86-64 in general;
V4 is physically V3 (no layout bump); this probe writes no bytes at rest;
B6 ranks candidates only, admission is MUL's/the triangle's.

**Files:** `probe_bpe_r2il_loco_microcode.rs`.
