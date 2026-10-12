# 2026-10-12 — CE64 hypothesis layers: plateau and plan (D-HL-0)

**Status:** WORKING-MODEL. Documentation only; no code changed, no stored
meaning changed, no decision taken.

- Plateau: `.claude/knowledge/ce64-reasoning-substrate-plateau.md`
- Plan: `.claude/plans/ce64-hypothesis-layers-v1.md` (D-HL-0..9)
- Research map: `.arxiv/2026-10_ce64_hypothesis_layers_research_map.md`

## What the plateau records

- **The operator's design** (2026-10-12): opaque little-endian bytes with an
  entry tax and no later conversion; an immutable `classid × NodeGuid / row`
  address; up to 64 parallel hypothesis layers as sparse deltas merged on
  read; a read mask × thinking mask; a 64-slot budget for hydration then
  reasoning; hydration as a global sudoku over bits 50..52 and 59..63;
  validation by running literal Narsese in parallel and comparing with
  `jc::reliability`.
- **Where each idea already lives**, with status: the ISA register contract,
  the Alpha overlay and 10-lane tunnel, `SpogTenants`, the split-tunnel
  append, `RowAddrMask`, D-GSO, D-RPF, the Staunen/cycle/loop probes,
  tenants 18..20, the NAL audit oracles, `jc::reliability`.
- **Eight open conflicts between design and code** (O1..O8). The sharpest:
  bits 46..49 are a signed opcode in code and an activation in the design;
  bits 50..52 are plasticity that `learn` gates on, so Staunen cannot be
  written there under the current reading; the tunnel has 10 rung lanes, not
  a 64-lane budget; overlay rows are whole and ephemeral, not persisted deltas.
- **Nine unification hypotheses** (U1..U9). The two with the most leverage:
  U1, declare the reading of bits 40..63 per classid through the ClassView,
  so the ISA reading and the design reading coexist behind a refusal; and
  U2, merge-on-read is the wide accumulator that removes D-RPF-P5's c = 244
  stall by construction.

## Not new here

The c = 244 stall (D-RPF-P5 E2), duplicate inflation (E3), the BF16
analogy's scope (coresearch 2026-10-08) and the cost numbers are cited, not
re-measured.

## Closeout

```text
PR #<tbd> | STATUS: open | OUTCOME: plateau + plan + research map;
design-vs-code conflicts listed with the decision each needs | OPEN:
D-HL-1 decisions (O1, O2, O3, O5, O6); D-HL-2/3 are the first probes
```
