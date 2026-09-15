# The causal-plane inventory — what ships, what is wired, what is prose

> **READ BY:** any session touching `CausalEdge64` / `CausalEdgeV3`, the A9 locus
> register, `ValueTenant`, the four-plane convergence, plasticity/promotion,
> σ-propagation, truth propagation, or the hexagon hop. Read **before** proposing
> a new tenant, a new lane shape, a new register, or a new reading of an existing
> one.
>
> **Companion rulings:** `EPIPHANIES.md` `E-SIX-SEMANTIC-FAMILIES-MUST-NOT-IMPERSONATE-EACH-OTHER-1`
> (2026-09-02, operator) · `.claude/v3/soa_layout/le-contract.md` §3 ·
> `.claude/v3/soa_layout/witness-nibble-lane.md` (why sub-byte is a LANE, not an L9).

## Why this exists

The recurring failure in this area is **not** a missing carrier. It is reading a
shipped shape as a shipped capability. Across one session, six distinct
"the substrate should do X" proposals each showed a missing production connection:
*X's selector or kernel is represented or documented, but the distinction is not
consumed at the step where it would do work.* **Validation and production status
differ by row** — see §3; rows 5 and 6 in particular are neither proven nor in `src/`. This file is the inventory that makes that checkable in one read instead of
six greps.

---

## §1 — The four planes, and where each one lives

The convergence is operator-dated 2026-08-23 and documented in the header of
`crates/lance-graph-planner/examples/probe_four_plane_causal_medium.rs`:

| plane | question | carrier |
|---|---|---|
| HHTL / Attention V3 | WHERE is relevant? | `AttentionFocusFacet` scope |
| `CausalEdge64` 59..60 | WHAT causal topology? | `CausalTopology` |
| `CausalEdge64` 61..63 | THROUGH WHAT lens? | `ReasoningBand` |
| Tarski G24N4 | WHY supportable? | `+n` support / `0` unresolved / `−n` falsifier |
| R2IL V4 | WHAT did we DO? | typed intervention row + observed consequence |

**Iron rule from that header:** *"the planes are jointly readable, never derived
from one another."* A negative support nibble must not auto-flip the band or
rewrite the topology — **and the inverse also holds**: a positive support lane
must not auto-determine the act. That inverse is the formal content of
*"I experienced the meta, it was worthy, now nature wins."*

### `CausalEdge64` v2 is 100 % allocated (`crates/causal-edge/src/layout.rs`)

```
 0- 7 S        8-15 P       16-23 O       24-31 FREQ    32-39 CONF
40-42 CAUSAL (why-planes)   43-45 DIR     46-49 INFER (4-bit signed i4 mantissa)
50-52 PLAST   53-58 W       59-60 TRUTH/topology        61-63 SPARE/ReasoningBand
```

**Zero free bits.** `CausalEdgeV3` (96-bit) has room: byte `[9]` reserves 5 bits,
`[10..12]` are reserved-dormant.

Bits 59–60 carry **two ordinal-identical readings** — `TrustTexture`
{Crystalline, Solid, Fuzzy, Murky} and `CausalTopology` {Direct,
IndirectKnownIntermediates, IndirectUnknownIntermediates, Unknown}. Its doc is a
model of how to fuse two readings honestly: wire and ordinal compatibility exact,
legacy projection intentional, and **historical factual provenance explicitly NOT
guaranteed** — it refuses to backfill meaning onto old rows.

---

## §2 — The six semantic families (ratified; do not add a seventh from a shape)

| # | family | carrier |
|---|---|---|
| 1 | episodic / Markov **loci** — sign = orientation, pointers only | `CausalWitnessFacet`, tenant 14 |
| 2 | epistemic **qualia** — signed proprioceptive magnitude of current state | `QualiaI4_16D`, tenant 1 |
| 3 | epistemic **population basins** — the mammal/whale/wombat/elephant case | **accepted VACANCY** |
| 4 | epistemic causality **trajectory** — ordered evolution | `TrajectorySignature` / `RevisionTrajectory` |
| 5 | the **causal graph** | `CausalEdge64` / `CausalEdgeV3` |
| 6 | the **epistemic / knowledge graph** | — |

**Invariant:** `locus ≠ magnitude ≠ population basin ≠ causal graph`;
`trajectory ≠ causal graph`. Same physical shape ≠ same semantics; same codec ≠
same ClassView.

`ValueTenant` (`crates/lance-graph-contract/src/canonical_node.rs:859`), 0..15:
`Meta · Qualia · MaterializedEdges · Fingerprint · HelixResidue · TurbovecResidue ·
Energy · Plasticity · EntityType · Kanban · FrozenStyle · LearnedStyle ·
ExploreStyle · Tekamolo(13) · CausalWitness(14) · EpisodicBasin(15)`.
Value slab: **220 of 480 B used** — space has never been the constraint.

**A9 / `G24N4`** (`causal_witness.rs`): the 12-byte content-blind register read as
**24 signed i4 loci**, `WITNESS_LOCI=24`, `NAMED_LOCI=16`, 8 reserved-empty. Each
nibble is a **context pointer** into the ±8 window (`0` = unbound, sign =
orientation), never a strength. Status: **EXPERIMENTAL**, deliberately not a §3
layout — sub-byte's sanctioned home is a **lane**, and it needs no catalogue
petition.

**The autopoiesis triangle** (tenants 10/11/12) is three perspectives on one
policy in one row — `Frozen` = the disposition the dispatch runs off *with no
lookup beyond the row*; `Learned` = what deliberation produced; `Explore` =
deterministic address-derived jitter (D-QUANTGATE, never RNG, replay holds).
The documented gate — *"`learned[f]` promotes to `frozen[f]` only after winning
the held-out arm"* — is "nature wins" as a shipped rule.

---

## §3 — The six unwired seams (the load-bearing section)

Every row: a selector and a kernel exist as named surfaces, and nothing consumes the
distinction where it would matter. **Their maturity is NOT uniform** — rows 1–4 have
kernels that ship and are proven; row 5's kernel returns a placeholder; row 6's
promoter exists only in a probe.

| # | selector (ships) | kernel (ships) | state |
|---|---|---|---|
| 1 | `CausalTopology` 59–60 | `AND3` hop, `lgj-abi/exports.rs:1816` | the hop's **third mask slot carries `struct_f`**, not the causal plane |
| 2 | `CausalTopology` 59–60 | `sigma_propagation::ewa_sandwich(m, sigma)` — PSD-proven 10000/10000 | **M is caller-supplied**; only `jc`'s own proof calls it |
| 3 | `InferenceType` / `ReasoningBand` | NARS revision, truth semirings | `adjacent_truth_propagate` takes **bare `TruthValue`**; the plane selects a semiring once **per plan** (`orchestration_impl.rs:146`), never per edge |
| 4 | `CausalTopology` / `InferenceType` | `PlasticityState` (s/p/o hot/frozen) | **no update rule reads across them**; the only co-occurrences are field-isolation tests asserting they don't touch |
| 5 | `AND_ANDNOT2` = `domain ∧ ¬result` (the surround) | ternlog immediate, ships in ndarray | **not wired at the hop** |
| 6 | `LearnedStyle` → `FrozenStyle` promotion | documented in the tenant | implemented **only in `probe_sudoku_teacher.rs`**; no promoter in any `src/` |

Rows 1–4 are substitutions between things that already ship and are already
proven. **Row 6 is the same shape but is NOT in that group** — its promoter ships
only as a probe (`probe_sudoku_teacher.rs`), never from any `src/`, exactly as its
row says. Row 5 additionally needs a **certified shape**: Pillar-15
(`ndarray/src/hpc/pillar/mexican_hat.rs`) is DEFERRED returning placeholder
`passed=true`, and `hdr_cascade.rs:128-141` is a piecewise-linear ramp.

### Separate defect, same file, not a wire

`graph/blasgraph/typed_graph.rs:72` documents `traverse` as **"Single-hop
traversal"**, carries the inline comment `// A × A under the given semiring = one
hop`, and computes `matrix.mxm(matrix, …)` — which is **two** hops. One hop is `A`.
`multi_hop(&["r"])` returns `A` itself, so the two provably disagree by a hop.

**`masked_traverse` has the identical defect** (`:132`, `matrix.mxm(matrix, …)`
before the label filter) — confirmed by reading, and NOT named by the review that
raised `traverse`. It additionally rebuilds a COO per call to filter target columns.

The disagreement is structural rather than suspected: `A` and `A × A` differ by
construction, so no test is needed to establish it (one is still needed to pin the
fix). **Not fixed in this PR** — it is a behaviour change to a shipped semiring
primitive, and this PR is docs-only. `TypedGraph::traverse` has no caller outside
its own test module; `masked_traverse` is called by `graph_router.rs:332` and
`examples/sigma_probe_masked_traverse.rs:132`, so a fix has to re-pin those.
Filed as `ISS-TYPEDGRAPH-TRAVERSE-HOP-COUNT`.

---

## §4 — The composition (the thing the planes are for)

```
hexagon (WHERE)  ×  the causal planes (WHAT KIND)  ×  SPOFC (HOW STRONG)
```

None stores another's meaning. Mechanically it is two steps, both shipped:

1. **narrow** — `AND3`, one `VPTERNLOGQ` per 512 bits on the AVX-512 `U64x8` path
   (the polyfill elsewhere; `U32x16` uses `VPTERNLOGD`). `exports.rs:1756` spells
   it `selected_f = ternlog<AND3>(class_f, src, struct_f)` — **three mask slots**.
2. **weight** — semiring `mxv` / `masked_traverse` over the survivors.

**Vacuous activation is `AND3` with only geometry in its inputs.** *Activation over
what exists* is the same instruction with existence and qualification in the other
two slots. The instruction was never the problem.

White matter is shipped and was not used by the W1 arc: `GrBMatrix::{mxv, vxm}`,
`TypedGraph::{traverse, multi_hop, masked_traverse}`, 7 semirings. W1 spread with a
*radius over palette address geometry* instead — and the field grew 25 → 123 → 311
cells while r@10 stayed flat, because it was diffusing into empty address space
(1305 occupied of 65536).

---

## §5 — Three discipline rules, each earned

1. **Consume, never impersonate.** Tarski *uses* truth-texture and is not it;
   SPOFC *reads* causality and stays FC; a ClassView *reads* a register and is not
   its meaning. Four independent September mistakes were all one move: shape
   equivalence read as semantic equivalence.
2. **Two algebras at the learning layer** (mirroring sign-vs-magnitude / XOR-vs-
   bundle): **structure** changes by intervention — discrete, licensed, topology
   and loci; **strength** changes by evidence — graded, in the semiring. Neither
   may silently become the other. This is why the substrate can represent an agent
   whose correct reasoning does not determine its act, and a correlational model
   cannot: one number collapses "reasoned well" and "acts accordingly".
3. **A correction is not applied until every site states it.** Prose describing a fix
   and the fix itself get written in the same breath, and only the prose is checked.
   Measured six times in one session — a §11a amendment that left the sentence it
   *quotes* unqualified; a commit message describing a vacuous-test repair the commit
   did not contain; one overclaim corrected in three files while the PR body and a
   `STATUS_BOARD` Status cell kept it. **The mechanical check is to grep the STRUCK
   PHRASE**, not to re-read the amendment — and to include the PR body and board cells
   in the sweep, because they are the most-read sites and are not in the diff you just
   reviewed.
4. **Dialogue is branch diff, not messages.** Perspectives reason with each other
   by being diffed over deterministic forks — `scenario-world`'s branch diff /
   deterministic replay / time-travel reads, with three already-distinguished
   carriers. Per-agent RPCs are the actor/message shape lance-graph **deleted**
   (*progression is existence, not command*). `ExploreStyle`'s replay-identity is
   the precondition: without it a diff is noise.

---

## §6 — The instrument: one measurement, several boards

RPS, the scorpion fable, and MQ are **not three probes**. They are branch diff over
deterministic forks on three geometries:

| board | what the diff does | role |
|---|---|---|
| RPS | cycles — no branch dominates | the **unforced null**; a fixed disposition always loses |
| scorpion | two unprivileged horizon-branches | fusion = does a branch exist containing both (**satisfiability** — the Tarski plane's own question) |
| MQ | Raumgewinn vs infight over one position | "nature wins" = the **coincidence rate** of two evaluation branches — a property of board geometry |

Gadamer's *Horizontverschmelzung* is the third row's design constraint: a fused
horizon is **a third reading under which both remain recoverable**, never a merge
producing one register. The 59–60 pair is the shipped worked example.

---

## §7 — Measured: the ±8 window is not a discount

`.claude/probes/horizon-window-v1/` (re-runnable — no corpus, no fetch, no RNG).

- exponential control **0/19** reversals on all three fixtures (provable zero holds → harness valid)
- boxcar band matches the derived `[T2−T1+1, T2−1]` on all three
- **direction is OPPOSITE to hyperbolic on all three**: hyperbolic `later→sooner`
  (impatience rises with proximity — akrasia); boxcar `sooner→later` (blindness
  *falls* with proximity)
- **no indifference region, analytically**: the boxcar flip margin is exactly
  `V2 − V1` and cannot be small; hyperbolic crosses continuously, so some `k` is
  arbitrarily close to indifference (an earlier **41–206×** ratio was a
  sampling-grid artifact and is withdrawn)
- at the representable forward bound `w=7` (i4 is `[−8, +7]` and `+` = consequent),
  **all three fixtures** read all-`sooner` from every vantage — **uninformed, not
  impatient, and unable to report the difference**. The window is **asymmetric**:
  8 steps backward, 7 forward — one more step of cause than of consequence, which
  is the wrong way round for a causal agent

**Falsified by this:** *"an A9-locus agent is constitutionally a scorpion."* The
scorpion's signature is hyperbolic **curvature**; the substrate's pathology is
horizon **width**, and they point opposite ways.

**Consequence:** this is a measured argument for the `EMPTY, −7..+7` nibble the
six-families ruling already deferred (`EMPTY` = no valid observation, `0` =
observed neutral). Without that distinction a hard-horizon agent cannot report
that it is blind — which is exactly the state the sweep shows it spends most of
its time in.

---

## §8 — Instrument-availability ledger

This workspace documents its **measurements** far better than the **instruments**
that produced them. Check here before citing a number as reproducible.

| instrument | state |
|---|---|
| `.claude/probes/horizon-window-v1/` | **re-runnable from the repo** |
| `.claude/probes/hexagon-plasticity-v1/` | **auditable, not re-runnable** — the ~10 GB `r2harvest` corpus was ephemeral, no revision to cite |
| KJV ±8 histogram | **not run** — needs the `v0.1.0-cam96-data` release + Gutenberg #10; both tag-addressable, so it *would* be reproducible. Run `bible_wave --export` once and commit the TSV, and the histogram becomes arithmetic over a committed artifact |
| MQ / Mississippi Queen | **no implementation** — `mississippi` appears only in prose (plans + board). The W-1 river baseline came from the same ephemeral scratchpad |
| `probe_four_plane_causal_medium` | in-tree, runnable; tests ABI **representation**, explicitly not causal discovery |
| `probe_sudoku_teacher` | in-tree; carries the only promotion implementation, with **both** a promote and a refuse row |
