## 2026-08-19 — E-RP-SEAL-PASS1-THE-MAXIM-WORKED-1

**Status:** FINDING (consolidated), first independent pass complete.

**The event.** The RP-SEAL 15-researcher independent pass (run
`wf_ca974718-1b4`, 15/15 cells, 0 errors, no cross-talk, blind to boards)
completed and was consolidated per charter §12–§13:
`docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md`, with all 15 reports + the
orchestrator pre-pass committed as Appendix H under `docs/lotus/rp-seal-v1/`.

**What died (charter §17 executed):** the phase/comma coefficient schedule
(C2 adversary: 0/6 conditions, syndrome separation impossible under
permutation; C1+C3 independently bound cascades ≥33% overhead / priced in
distance — three disjoint routes, one kill); unconditional Morton/SFC
default (B2 measured + B3 literature: identical page partition to Hilbert
at 4^j pages, 20–45× quantizer hot-spots, three proven negative results);
"compaction = one manifest publication" (reserve_fragment_ids is itself a
commit); the naive query+repair one-grouping hope (Morton locality
MANUFACTURES the correlated failure that kills locality-aligned parity).

**What was found:** the coalescing path writes MORE bytes than no
coalescing — physical amplification = b+1 (65× at b=64) from 512 B null
padding per landing row, and 52% of seal CPU is a non-amortizing FNV hash
(E2); 11 arrival/physical-order leaks into durable coordinates, worst
L2: durable semantic replay order IS Lance physical scan order, so
compaction here MUTATES semantics (D2) — this gates all layout work;
five shipped seal paths with defective/false-accept behavior wearing
strong names (C2 §B1–B5, firefly independently by C3); Lance 9.0.0
already ships the whole compaction/remap/stable-row-id/per-row-version
apparatus unused by our append-only path (5 cells concordant).

**What survived, sharpened:** boring hash + row/column P+Q over the native
64×64 (6.25%, 63× repair amplification vs flat-RS 4095×) as the seal
baseline; the caller-supplied compaction-rewrite ordering key as the ONE
open precedented upstream seam (in-repo R-tree Hilbert leaves + Delta
Z-order as precedent; IndexRemapper usable today); two genuine
NOVEL-CANDIDATES with no prior art found by independent searches — the
joint query-scatter × repair-scatter address-order question (refined by
C2's anti-synergy into "query-aligned order, ANTI-aligned parity groups")
and the STRICT/AWARE/RETRO reader-rung admission tier (D3: no literature
counterpart; D1: currently inert in code, T_now missing as a type,
hlc_tick not actually HLC).

**Scope-pivot filter:** PASSED — zero findings sourced from rustynum or
symbiont; the two mentions cite the exclusion ruling, and D1 converts it
into a finding (the version→kanban OUT-bridge lost its only caller).

**Tier 0 before anything else:** pin D2's leaks (F-PHYS-ORDER first),
re-verify E2, X-C2-1 injection harness over the five seal paths, E3's
locality-debt trigger metric, perf-event or strike the cache-miss metric.
Full matrix M1–M15 + tiering in the consolidation doc.

