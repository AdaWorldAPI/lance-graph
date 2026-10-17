# 2026-10-17 — CE64 one layout; the Moore tenant as synapse, membrane and local NARS

**Status:** DECISIONS recorded from the 2026-10-17 session (user statements quoted as
BASIS). Code changes named here are NOT made; each waits for its own go.
**Touches:** `causal-edge` (`layout.rs`, `isa.rs`, `edge.rs`),
`lance-graph-contract/src/moore_tenant.rs`, `perturbation-sim` (Moore probes),
ndarray `hpc/{causal_diff,p64_bridge,jina/causal,entropy_ladder}.rs`.

## Decisions (answered; do not re-ask)

| # | DECISION | BASIS (user, 2026-10-17) | consequence |
|---|---|---|---|
| D1 | CE64 is an ISA contract, a lever read like BF16 mantissa bits; it is not storage and is not written back into | "Causaledge64 is not storage it's the 'TCP IP' ISA contract for AGI, a contract lever"; "Akin to BF16 mantissa bits" | no "result lands in 59..63" designs; no retiring of contract bits as storage |
| D2 | CE64 stays 64 bits; no CE128 user asked "Would it improve anything to switch to causaledge128 or would it dillute the mantissa reading as a wire format"; the session answered stay at 64 and the user did not reopen it | widening would dilute the mantissa reading of the wire |
| D3 | There is ONE CE64 layout. Align every reader on it, migrate, catalog. No renames | "There is no other layouts / We just need to aligh on one / And migrate and catalog"; "Don't rename, use what exists and harden the bit layout to be canonical" | ndarray's three divergent layouts are migration targets, not alternatives |
| D4 | The layout is three regions: bits 0..23 S/P/O payload (read as palette256 codes or as an ISA register), bits 24..39 F/C (palette256 or ISA register), bits 40..63 a 24-bit fixed contract with one meaning for every reader | "causaledge64 is 24 bit spo payload that can be read either as palette256 or ISA register / And FC that can be read either as palette256 or ISA register / And 24 bit fixed contract" | `causal_edge::layout` v2 is the canon; domains choose only what payload palette codes mean |
| D5 | NARS16x8 / the Moore tenant reuses the same layout and omits the bits locality already answers | "NARS16X8 reuses it for Moore tenant / By omitting 8 Bits when locality already answers the 8 Bits" | `MooreNars16` = contract 24 − DIR 3 − W 6 + polarity 1 = 16 bits (`moore_tenant.rs:107-219`) |
| D6 | The Moore witness is Cartesian addressing: a lane's partner is the neighbour's own Moore tenant at the opposite slot, found by coordinate, never stored | "In NARS16 the Moore tenant connects with the Moore tenant of the neighbors / So the witness is Cartesian addressing" | the module header's "one witness covers all eight lanes, held by the owner" and `lift_ce64`'s `MixedWitness` refusal encode the superseded reading |
| D7 | The unit of Moore reasoning is the membrane: two neighbours agreeing on their shared boundary. Agreement IS the witness | "2 neighbors agreeing on a cognitive membrane" | a cell has 4 membrane orientations (8 slots / 2); agreement is a side-by-side compare of a plane with its shifted plane (the pair sits at one coordinate; no gather) |
| D8 | Purpose: local reasoning akin to synapses, with excitation and inhibition | "The whole point of Moore tenant+ NARS16x8 is locality reasoning Akin to synapses and exhibition inhibition" | cell = neuron, lane = synapse, slot = which neighbour, polarity = which end |
| D9 | Goal: a cheap local perturbation field with cross-node, Bayesian and Hebbian learning | "We want local cheap pertubationsfeld cross node learning Bayesian learning hebbian learning etc" | builds on `perturbation-sim` D-MOORE-PERT-AB-0 (#1427) and D-MOORE-OBSERVABLE-FIRST-0 |
| D10 | Hebbian wiring IS SPOFC with local NARS reasoning: what fires together wires together as evidence | "The idea is that what fires together wires together SPOFC adding NARS reasoning locally" | co-firing = positive evidence, a lone firing = negative evidence (inhibition on the F side); strength is NAL truth `f = w⁺/w`, `c = w/(w+1)`, updated by `isa::truth::revision` |
| D11 | Simpson's paradox is what SPOFC suffers under naive NARS accumulation; the architecture prevents it by SPOFC per SoA SPO, fine identity splitting and CE64, with 59..63 and revision as the backstop | "The Simpsons paradox is what spoFC suffers if you accumulate it as NARS usually does, we put it in every SoA SPO and differentiate a lot not to mention CE64 / so it can't dillute or spill and if it does we have 59..63 and revision" | on the Moore lattice, evidence crosses only an agreed membrane; the 8 synapses are never pooled blind. P-pool test on branch `ccr-9f3b8075-74u47l-simpson-coresearch` (`simpson_revision_pooling.rs`) pins the naive case |

## WORKING-MODEL (follows from the decisions; not yet confirmed or built)

- **SPOFC per synapse fills the three Moore tenants.** Per lane:
  `MoorePalettePairs` (u8:u8) = P and O palette codes; `Nars16x8` (u8:u8) = F and C;
  `MooreNars16` = the contract part; locality answers S (the centre cell), DIR (the
  slot) and W (the membrane). 48 stored + 16 answered = 64, a full CE64 per synapse.
  This would give the CANDIDATE `Nars16x8` (`moore_tenant.rs:8`) its reading.
- **Local NARS is triangle closure inside the 3×3 neighbourhood.** Deduction A→B, B→C
  gives A→C when `offset(s)+offset(t)` is a Moore offset; induction/abduction over a
  shared neighbour. The valid `(s,t)→u` compositions are a fixed compile-time table;
  each is a shifted-plane operation.
- **PLAST (50..52) is consolidation, the weight is the F/C evidence.** Refines (does not
  contradict) the canon row E-BASIN-NOT-EDGE-PLASTICITY: the Hebbian state stays on the
  per-edge register, never basin-level.

## CE64 reader catalog (migration targets under D3/D4)

| site | reads CE64 as | migration |
|---|---|---|
| `causal_edge::layout` + `isa::Field` | v2, three regions | canonical; none |
| contract `MooreNars16` | contract minus locality-answered bits | catalog only (D5) |
| `causal-edge/src/edge.rs:139-151` header + v1 consts 183-199 | v1 (temporal at 52..63) | mark V1-legacy, point at `layout` |
| ndarray `hpc/causal_diff.rs` `CausalEdge64` | own split (block/projection/verb/row/l1/f10/c10) | block/projection/verb → S/P/O palette codes; f/c → u8 F/C; row, L1 stay in `WeightEdge`; upper 24 bits contract default |
| ndarray `p64_bridge::causal_edge_compat` | SRC 58 / TGT 52 / FREQ 44 / CONF 36 / LAYER 33; doc falsely claims compatibility | read v2 regions |
| ndarray `jina/causal.rs` `pack_edge` | v1 (temporal 12 at 52) | v2 contract |
| ndarray `entropy_ladder` | "spare bits [63:61]" (now Epi5 certification) | move off contract bits |
| ndarray `ogit_bridge/.../CognitiveCell.ttl` | "per causal_diff.rs" | point at `causal_edge::layout` |
| lance-graph (new) | — | test pinning ndarray's v2 constants against `causal_edge::layout` |

Not CE64, out of scope: holograph `pack_edge`, contract `dawg::pack_edge`, ogar-obo
`pack_edges`, `CausalEdgeV3`.

## OPEN

- Remove `MixedWitness` from `lift_ce64` and reword the module header (D6), or keep
  CE64 W as the corpus root that Moore never reads. Waits for a go.
- `Nars16x8` = F:C per lane: confirm and ratify (contract PR).
- What counts as "fired" per cell per step (the perturbation field crossing a
  threshold?).
- Evidence stamp per membrane hop so loops and triangle deduction do not double count;
  where its bits live.
- Hebbian saturation: inhibition on disagreement plus a decay rate (policy pin until
  measured).
- Palette index range in `causal_edge_compat` (6-bit `Palette64` vs palette256 codes).
- PAL8 version gate when `causal_diff` moves to v2.
- `moore_nars16_probe` / `moore_plasticity_probe` (PR #1406) not yet read; read before
  the learning probe.
- Promotion path: `EPIPHANIES.md` is generated from `epiphanies-canon.json`, whose row
  sources must resolve to archive lines, so a post-baseline Eureka has no route into the
  projection today. These decisions live here until that path exists.
