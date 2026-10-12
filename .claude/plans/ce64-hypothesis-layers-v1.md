# CE64 hypothesis layers — v1

**Status:** PROPOSAL (D-HL-0..9). D-HL-0 is this documentation. Nothing else
is authorized: each later item needs its decision (D-HL-1) or its probe to
pass first. No CE64 bit, tenant, opcode or ABI change is authorized by this
plan.
**Date:** 2026-10-12.
**Plateau (read first):** `.claude/knowledge/ce64-reasoning-substrate-plateau.md`.
**Research map:** `.arxiv/2026-10_ce64_hypothesis_layers_research_map.md`.
**Parents:** D-GSO (global sudoku), D-RPF (resident projection fold),
D-CE64-STAUNEN/CYCLE/LOOP, D-ALPHA-G-0, D-NAL-AUDIT-0, coresearch
"CE64 as BF16 for thinking" (2026-10-08).

## Overview

The operator's design (plateau §1) treats reasoning as up to 64 parallel
hypothesis layers over an immutable address space. Each layer is a sparse
delta of opaque little-endian bytes keyed by `NodeGuid` / row id, merged on
read. A layer that clears a validation ladder is composited into the base
by one fold and one Lance append. Literal Narsese, run by an independent
engine, is the yardstick the binary encoding must agree with.

Most pieces exist (plateau §2). This plan connects them, in an order where
every step is a measurement that can kill the next.

## Checklist

- [x] **D-HL-0** Plateau, plan, research map, board entry (this PR).
- [ ] **D-HL-1** Decisions O1, O2, O3, O5, O6 (plateau §4). Operator.
- [ ] **D-HL-2** Merge-on-read vs in-place revision (U2).
- [ ] **D-HL-3** A per-delta evidential base as the independence test (U6).
- [ ] **D-HL-4** Narsese parallel calibration harness.
- [ ] **D-HL-5** Hypothesis lanes: `AlphaTunnel` with a lane budget.
- [ ] **D-HL-6** Two masks before decode (U5).
- [ ] **D-HL-7** SparseDelta probe record, persisted through the split tunnel (U4).
- [ ] **D-HL-8** Validation ladder: when a layer may commit.
- [ ] **D-HL-9** Hydration budget split sweep.

## Order

```text
D-HL-2 ─┐
D-HL-3 ─┼─► D-HL-4 ─► D-HL-9
        │       ▲
D-HL-5 ─┴─► D-HL-6 ─► D-HL-7 ─► D-HL-8
D-HL-1 gates any stored-meaning change; D-HL-2..6 are probes and need no decision.
```

## Details

### D-HL-1 — decisions (operator)

Plateau §4, rows O1 (bits 46..49), O2 (bits 50..52), O3 (W ownership, the
coresearch G6), O5 (10 rung lanes vs a 64-lane budget), O6 (what persists).
The recommendation in plateau §5 U1 is to declare readings per classid
through the ClassView, so the ISA reading and the design reading coexist
behind a refusal for undeclared readings. Any stored-meaning change ships
behind a version gate (`I-LEGACY-API-FEATURE-GATED`).

### D-HL-2 — merge-on-read vs in-place revision (S)

- **Where:** `crates/causal-edge/tests/`, next to `d_rpf_p5_beta.rs`.
- **Arms:** (a) revise into the u8 byte per observation (today);
  (b) accumulate `Σw`, `Σw·f` in integers, round once; (c) exact rational.
- **Grid:** unit-evidence chains N = 1..1000 at observation strengths
  c_obs ∈ {1, 64, 128, 240, 250} (the E2b table).
- **Measure:** final code error vs (c); the N at which (a) stalls.
- **Kill:** (b) within 1 code of (c) on every cell and (a) not ⇒ U2 holds.
  (b) as far from (c) as (a) ⇒ U2 is cosmetic; record and stop.
- **Anti-vacuity:** at least one cell where (a) and (c) differ by ≥ 2 codes.

### D-HL-3 — a per-delta evidential base as the independence test (S)

- **Where:** same test file family.
- **Setup:** D-RPF-P5 E3's duplicate fixture (4 and 8 copies). Each delta
  carries `(layer, seq, evidential base)`, the base being a source bitset
  (`nars::belief::Stamp` shape). Fusion is cumulative only between disjoint
  bases; overlapping bases are not counted again. Layer membership plays no
  part in the decision.
- **The fixture must separate layer from provenance**, or it encodes the
  answer: it includes (a) one observation copied into two layers, which must
  count once, and (b) two independent observations inside one layer, which
  must count twice; plus the plain cases (duplicates in one layer,
  independents in distinct layers).
- **Kill:** the sd ratio does not return to within 10 % of 1.0 on all four
  cases ⇒ U6 is wrong. A rule keyed on the layer id instead of the base must
  fail cases (a) and (b); that arm is run as the disable.
- **Silence twin:** independent observations must still narrow the posterior
  exactly as today.

### D-HL-4 — Narsese parallel calibration harness (M)

- **Arm L:** `.nal` files run by the pinned OpenNARS for Research build
  (`.claude/knowledge/nal-recipe-audit/oracles.md`, `RunNal.java`).
- **Arm B:** the same evidence as CE64 deltas, merged on read (D-HL-2 (b)).
- **Output:** one row per derived statement `(statement, f_L, c_L, f_B, c_B)`.
- **Statistics:** `jc::reliability::{icc, spearman, cronbach_alpha}`, plus
  absolute code error and its sign (bias).
- **Encodings swept:** linear u8 (today); log-evidence u8; the Moore pair
  reading of tenants 19/20, if D-HL-1 allows it.
- **Kill:** an encoding that does not beat linear ICC by the declared margin
  (set before the run, in the test file) is dropped.
- **Fixtures:** start from the upstream NAL-1/2 fixtures the audit already
  ran (nal1.*, nal2.*); add `revision_points.nal`.

### D-HL-5 — hypothesis lanes (M)

- **Where:** probe first (`cognitive-shader-driver/examples/`), not
  `contract/src/alpha_tunnel.rs`.
- **Shape:** a tunnel whose lane count is a parameter up to 64 over one
  `AlphaAllocation`; one writer per lane; `run_wave` and `run_wave_parallel`
  byte-identical (the existing tunnel's falsifier, kept).
- **Measure:** reserve cost per lane (must stay an empty `Vec`), claim cost,
  merge cost per touched row.
- **Kill:** reserve cost grows with lane count ⇒ the "64 for nothing" premise
  fails; report the slope.

### D-HL-6 — two masks before decode (M)

- **Where:** `lance-graph-benches/examples/` beside `lance_source_probe.rs`.
- **Projection first:** `AlphaMask` bits are `AlphaAllocation` ordinals and
  `RowAddrMask` holds Lance row addresses (fragment id << 32 | offset). Build
  an explicit `AlphaAddr` / `NodeGuid` → row-address map for the fixture and
  project the thinking mask through it before intersecting. A test pins the
  projection: every projected address resolves to the row whose key equals
  the alpha address, and a fragment boundary inside the selection is covered.
- **Arms:** unmasked scan; `RowAddrMask` only; `RowAddrMask ∧ project(AlphaMask)`.
- **Measure:** rows decoded, bytes read, wall time; answers must be equal
  across arms.
- **Kill:** the joint mask decodes no fewer rows than the read mask alone on
  the fixture ⇒ the thinking mask adds nothing at the storage layer.

### D-HL-7 — SparseDelta, persisted (M)

- **Record:** `(NodeGuid or row id, layer, seq, field mask, dirty bytes)`;
  the open vocabulary item in `entries/2026-10-05-quack-storage-portability.md`.
- **Path:** the split-tunnel append (one Lance version per batch, never
  updated in place), replay once, fold resident.
- **Falsifiers:** replay composite == in-memory composite, byte for byte;
  dropping one delta changes exactly the rows it touched; a delta with a
  field mask outside the declared reading is refused.

### D-HL-8 — validation ladder (M)

- **Candidate rungs:** hydrated (59..63 set) → witnessed (W breadcrumb) →
  settled (Staunen below a threshold, under whatever home D-HL-1 gives it) →
  committed.
- **Measure:** for each commit rung, agreement with the oracle (D-HL-4) and
  Lance writes per 10⁶ deltas.
- **Kill:** no rung beats "commit everything" on agreement ⇒ the ladder is
  decoration.

### D-HL-9 — hydration budget split (M)

- **Sweep:** k = lanes "basic" hydration may use, k ∈ {4, 8, 16, 32, 48, 60}.
- **Measure:** oracle agreement (D-HL-4) and lanes left for reasoning.
- **Output:** the curve; the knee is the recommended split.
- **Kill:** agreement is flat in k ⇒ the budget is not the binding constraint.

## Related fix-now items outside this plan

From the NAL ledger (`D-NAL-AUDIT-0` follow-ups), each S and independent of
D-HL-*:
R1 RCR abduction direction; R2 commit the tactic oracle test; R4 retire the
two non-NARS revision laws (needs coresearch G2); R16 AriGraph evidence stamp
(U6's evidential base is the general form); R17 AriGraph deduction and contradiction semantics.

## Boundaries

- mask-risc never executes CE64 instructions (D-RPF-8).
- No VSA on this path.
- No new tenant until D-HL-4 picks an encoding.
- No opcode added to the strict decoder by this plan.
- No `.0.0` pins or new dependencies; probes use existing crates.
