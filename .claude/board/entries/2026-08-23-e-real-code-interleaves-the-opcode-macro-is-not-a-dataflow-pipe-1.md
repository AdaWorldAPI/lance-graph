## 2026-08-23 — E-REAL-CODE-INTERLEAVES-THE-OPCODE-MACRO-IS-NOT-A-DATAFLOW-PIPE-1 — the real FunctionBehavior episode measurement: 99.7% over-admission by opcode matching, and three structural facts the toy could not show

**Status:** FINDING — [MEASURED] (`PROBE-R2IL-REAL-EPISODES-1`, 5/5, run
against the REAL ruff-side R2IL pass-1 harvest: 143 functions from real
x86-64 ELF64 binaries, 17,557 typed FlatFact rows, provenance FNV-pinned in
ruff's committed `.claude/harvest/r2il/PROVENANCE.md`; bulk stream from the
ruff GitHub Release `r2il-harvest-pass1`).
**This closes the named-not-built item** from
`E-HAPPY-PATH-RL-WOULD-HAVE-LEARNED-THE-CLOBBER-1` — the corpus was fetched,
never fabricated; the probe is env-gated (`R2IL_ORE_TSV`) and exits CORPUS
ABSENT rather than synthesize.
**Confidence:** High for these binaries at the pass-1 seven-opcode
convention; a 2-binary/143-function corpus is not "all real code."

**Headline (E5):** of 380 real occurrences of the top recurring opcode
trigram `(int_add, copy, store)`, exactly **1 is dataflow-chained** and 379
are NOT — a happy-path opcode matcher over-admits **99.7%** on real code.
Context base rate: 31.5% of ALL adjacent op pairs are def-use linked (SSA
coverage is total — 0 of 11,653 operand rows lack a ValueId — so this is a
dataflow fact, not missing coverage). The top idiom chains far BELOW base
rate: it is an ADDRESSING idiom (address computed, value staged, store
issued to memory), not a dataflow pipe. Real code interleaves independent
chains; **sequential adjacency is not composition**. The Phase-2 R6 toy
result (happy-path RL learns the clobber) is thus confirmed at real scale
and sharpened: on real machine code the macro carrier must be the DEF-USE
CHAIN, never the linear opcode window.

**Refuted pre-registration, recorded (E3):** "real top-1 trigram count >
shuffled" FAILED (380 vs 387) — shuffling a copy/int_add-dominated marginal
CREATES monotone `(copy,copy,copy)` runs. The real structure is
TYPE-COLLAPSE: 97 distinct trigram types vs 264 under shuffle (>2.7x), and
the top-10 types carry 50.5% of real occurrences vs 33.6% shuffled.
Recurring idioms are real; top-1 occurrence count was the wrong statistic.

**The stamp ceiling BINDS at real scale (E4):** with 143 episodes and
`Stamp`'s 64 bits, every widely-recurring macro measurably drops evidence —
`(int_add,copy,store)`: 64 revised, 23 CHOICE-dropped (e=0.999). The
modulo-64 conservatism is no longer a footnote: at real corpus size it
discards ~26% of the evidence for the widest idiom. Conservative (never
double-counts), but a real capacity note for Step-2's stamp residue item.

**Reader validity (E1/E2):** the probe-local TSV reader is cross-validated
against ruff's INDEPENDENTLY COMMITTED census — 17,557 rows, 5 kinds, 9
opcodes, all counts exact; 143 episodes across 2 binaries, 5,340 ops
partitioned. One misread caught by the gate: the TSV's kind column carries
CamelCase variant names, not Census's snake_case (the first E1 run failed —
the gate working).

**Fences:** no `ruff_r2il` import (separate cargo workspace; schema cited,
E1 catches misreads); reading the evidence TSV here is a measurement, not a
re-ingest into ruff's pipeline; no mint, no BPE, no learner subsystem; the
R2IL x BPE / OGAR-loco / V4 synthesis remains a three-IF hypothesis.

**Files:** `crates/lance-graph-planner/examples/probe_r2il_real_episodes.rs`.

