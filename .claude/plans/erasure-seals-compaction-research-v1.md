# erasure-seals-compaction-research-v1 — RP-SEAL (operator charter, 2026-08-18)

> **Status: DISPATCHED 2026-08-18** — independent pass running as a 15-agent
> background workflow (run `wf_ca974718-1b4`; reports land under the session
> scratchpad `rp-seal-v1/`, committed at consolidation as Appendix H).
>
> **PASS 1 COMPLETE + CONSOLIDATED (2026-08-19).** 15/15 cells, 0 errors.
> Consolidation: `docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md` (M1–M15,
> DELETED list, Tier 0–4). Appendix H committed: `docs/lotus/rp-seal-v1/`.
> The comma schedule is DELETED per §17; Tier 0 probes are the next phase.
> **§0 STORNO (operator, same day):** canonical replay coordinates are
> the core fix (physical iteration order must not define semantic replay
> order); retention/tombstone for indefinite historical reconstructibility
> is core; compaction = optional storage economics, never semantic repair,
> never a cognition prerequisite. See the consolidation doc §0.
> Tier allocation per the operator's "Opus for filigrane, Sonnet for
> grindwork": the five ADVERSARY cells (A2/B2/C2/D2/E2) = strongest tier;
> the five BUILDER and five SCOUT cells = grindwork tier. Consolidation
> (§12) and the final attack pass (§13) run on the main thread AFTER all 15
> reports exist. This plan file is the operator's program, recorded as the
> **⊘ SCOPE PIVOT (operator, 2026-08-18, mid-dispatch — BINDING):**
> **rustynum is struck from the source-of-truth list — everything ndarray.**
> rustynum is the historical donor already ported into AdaWorldAPI/ndarray;
> every primitive/capability/SIMD question routes to `/home/user/ndarray`.
> **crates/symbiont is DEPRECATED** (superseded by the supervisor/persistence
> arc, lance-graph PRs #879/#911/#912/#913) and is not live surface.
> Application: the workflow script was corrected in place (preamble source
> map + the two cell briefs that named rustynum); the in-flight first pass
> could not be force-stopped in this harness build, so the ruling ALSO binds
> consolidation as a hard filter — any first-pass finding citing rustynum or
> symbiont is re-anchored to ndarray / marked deprecated-source, or its cell
> is surgically re-run with the corrected brief. No second-pass or attack
> work may touch either. **Deltalake removal ratified** ("we don't need
> deltalake") — see E-PIN-LANCE9-LANCEDB033-DF541-ARROW58-NO-DF53-1.
>
> arc's source of truth. It SUPERSEDES the sequencing of the earlier lotus
> charter's phases 2–8 (docs/lotus/ deliverables 3–9 now route through this
> program); the landed Phase 0/1 deliverables (audit + F-ORD-REAL falsifier,
> PR #961) stand as prior session output that the researchers deliberately
> do NOT read during the first pass (§11 independence).

## Title

An Experimental Study of Bidirectional Erasure-Coded Seals and
Locality-Aware Compaction in Versioned Structure-of-Arrays Grid Stores

Subtitle: Morton-Cascade Amortization, Temporal Deinterlacing, Reed–Solomon
Repair Locality, and Index-Remapped Rewrite Policies in Lance

## 0. Posture

This is a research program, NOT an implementation sprint. Do not begin by
implementing the proposed architecture. Fifteen independent researchers:
5 research domains × 3 independent agents per domain. They FIRST investigate
independently, THEN consolidate evidence. Only AFTER consolidation may they
attack implementation / upstream candidate work.

The objective is not to vindicate the proposed idea. The objective is to
determine which parts survive contact with: exact Lance source; measurements;
coding theory; storage-system literature; temporal/database literature;
cache/SoA behavior; adversarial falsification. Unexpected negative results
are high-value results. Actively search for previously unnoticed structural
equivalences, cross-domain correspondences, and counterexamples. Do NOT claim
novelty merely because no one on this team remembers prior art. Every
suspected novel pattern requires a prior-art search.

## 1. Source of truth

Repositories: AdaWorldAPI/lance-graph, AdaWorldAPI/OGAR, AdaWorldAPI/ndarray,
lance upstream. (AdaWorldAPI/rustynum appeared in the original program text
and was STRUCK by the same-day scope pivot — see the header note.) Temporal reference: OGAR commit
`386a6fd848334b1d880c8408b3810f045d135cfe`, `docs/TEMPORAL-TIME-TRAVEL.md` —
read it literally. Temporal concepts: current tick / current Lance version;
last awareness tick / reference horizon; write tick; hindsight knowledge;
STRICT / AWARE / RETRO access; causal vs wall-clock order; possible HLC
coordinate. Do not reduce temporal.rs to an ordering utility: the research
hypothesis treats it as temporal deinterlacing `T_now × A_last × W_write`
with an interlacing window resembling `A_last < W_write <= T_now` —
investigate the actual implementation before relying on this shorthand.

## 2. Strict Lance source discipline

TWO evidence columns: (A) EXACT PINNED LANCE 9.0.0; (B) CURRENT LANCE
UPSTREAM. Never infer 9.0.0 capability from current upstream. Record per
claim: exact crate/version, Cargo.lock checksum, file, line, feature gate,
API stability, public API vs internal mechanism. Upstream appears to contain
machinery around: file compaction; reencode/binary-copy modes; incremental
compaction limits; excluded fragment boundaries; deferred index remapping;
fragment reuse indices; old-row-address → new-row-address remapping —
VERIFY all independently.

Session anchors (dispatch note): column A at `/tmp/sources/lance-9`
(upstream tag v9.0.0; workspace Cargo.lock checksum
`23d04bed056e254bc6e31264b031c8492507ca57939586f016924081dcf221a9`);
column B at `/tmp/sources/lance-main` (cloned 2026-08-18).

## 3. Baseline data geometry (hypotheses, not frozen architecture)

One V3-style logical row = 512 B. Small work field = 4×4 rows = 16 rows =
8 KiB. Cycle = 4096 fields × 8 KiB = 32 MiB = 65,536 rows. 2-D hierarchy:
4=2×2, 16=4×4, 64=8×8, 256=16×16, 1024=32×32, 4096=64×64. Inverse
reduction: 4096→1024→256→64→16→4→1.

## 4. Core distinctions

Never conflate: semantic identity; logical grid locus; worker completion
order; semantic/write time; awareness time; Lance DatasetVersion; physical
fragment; physical row address; secondary-index row address; compaction
order; parity group. Arrival may determine WHEN work executes; it must not
accidentally determine semantic identity. Physical relocation must not change
logical identity. Reed–Solomon protection, if used, is initially defined over
canonical LOGICAL bytes/symbols, not encoded Lance page bytes.

## 5. The 5×3 research matrix

- **A — Lance storage/compaction**: A1 builder (map exact 9.0.0:
  transactions, fragments, writer path, compaction planner, fragment IDs,
  row IDs, stable IDs, index remapping, version publication, GC, prepared
  writes); A2 adversary (destroy the thesis that random first-come chunk
  placement is cheap: fragmentation, index-remap, page-locality, point-read,
  range-read, long-retention failures); A3 upstream scout (map evolution
  beyond 9.0.0; identify generic extension seams proposable upstream without
  GridLake semantics).
- **B — Spatial layout / Morton cascade**: B1 builder (locality-key
  generators: Morton, Hilbert, raster, random, project-native Morton/HHTL);
  B2 adversary (adversarial query distributions: stripes, checkerboards,
  diagonals, sparse hotspots, moving windows, skewed clusters, temporal
  bands); B3 literature scout (SFCs, clustering metrics, cache-oblivious
  layout, learned SFCs — do not assume Morton wins).
- **C — Erasure coding / seals**: C1 builder (checksum/hash; XOR P; RAID6/RS
  P+Q; 2-D product codes; hierarchical cascade parity); C2 adversary (inject:
  single/double erasure, silent corruption, wrong-slot substitution, stale
  chunk, duplicate chunk, correlated/boundary/phase-aligned failures; measure
  detect/localize/repair/falsely-accept); C3 coding-theory scout (RS repair,
  MDS, LRC, regenerating codes, product/array codes, repair bandwidth,
  subpacketization, repair locality — no home-grown ECC accepted merely
  because tests pass).
- **D — Temporal deinterlacing**: D1 builder (map temporal.rs + OGAR
  time-travel to versioned reads; model T_now × A_last × W_write; test
  STRICT/AWARE/RETRO where implemented); D2 adversary (late facts, reordered
  workers, same semantic write under different scheduler timing,
  cross-version reads, stale awareness, clock skew, restart, cross-server
  causal ambiguity; prove scheduler chronology does not become semantic
  chronology); D3 literature scout (bitemporal DBs, transaction vs valid
  time, snapshot isolation, causal consistency, HLC, temporal provenance,
  retroactive queries).
- **E — SoA / cache / amortization**: E1 builder (benchmark harnesses:
  512 B rows, 8 KiB fields, cascade thresholds, first-come vs whole-64K,
  canonical sort vs scatter vs index-remapped random placement); E2
  adversary (allocations, resident bytes, cache misses, bytes
  copied/hashed/rewritten, fragment count, versions, index-remap cost,
  read/write amplification, recovery cost; one experiment with compaction
  disabled); E3 systems scout (LSM compaction design space, write
  amplification, columnar reorganization, data skipping, clustering,
  cache-oblivious structures, tiered/leveled, adaptive physical design).

## 6. Required prior-art starting packet

arXiv:2202.04522 (LSM compaction design space → trigger/layout/granularity/
data-movement policy); arXiv:1206.3804 (LRC → repair locality, distance,
storage-rate tradeoff); arXiv:1509.04764 (Repairing RS → exact repair,
repair bandwidth, MDS constraints); arXiv:1806.04437 (EC for distributed
storage overview → LRC, regenerating, RS repair, access cost);
arXiv:1801.07399 (Onion Curve → SFC clustering metric, query-shape
dependence, counterexamples to Hilbert/Morton optimality); arXiv:2008.01684
(SFCs for high-performance data mining → locality preservation,
cache-oblivious applications, transformation cost); arXiv:2009.06309
(Data-Driven SFCs → adaptive ordering, multiscale/quadtree);
arXiv:1812.07123 (CausalSpartanX + HLC literature → causal timestamping,
visibility latency, clock anomalies, snapshot semantics). Find more; do not
stop at this packet.

## 7. The phase/comma hypothesis

"Pythagorean comma is ECC" is FORBIDDEN as a claim. The admissible
hypothesis: a deterministic incommensurate / phase-progressive schedule may
be useful for choosing an independent RS/product-code coefficient orientation
across spatial cascade levels. Candidate sources: comma-like phase
progression, QuintenZirkel, BGZ17/highheelBGZ arithmetic, φX, 11/17
residues, CurveRuler. FIRST locate and characterize actual implementations;
then determine whether any candidate schedule (1) preserves matrix rank,
(2) preserves MDS/correction properties, (3) reduces cross-level syndrome
collisions, (4) improves localization, (5) costs ≤ SIMD work, (6) can be
reconstructed from locus/level instead of stored. Controls: standard RS
coefficients; sequential finite-field powers; random-but-seeded valid
coefficients; no modulation. If the project schedule loses to boring RS:
DELETE IT.

## 8. GridLake / bidirectional code hypothesis

Evaluate (not assume): A flat RS(k+2,k); B row P/Q + column P/Q; C product
code; D local P/Q + cascade parity; E hash-only syndromes; F payload parity;
G hybrids (strong per-chunk hashes + erasure parity). Report actual
redundancy; never call 12.5% "free"; distinguish CPU from storage cost.

## 9. Compaction research questions (upstream must stay general)

Locality-key ordered compaction; layout-aware rewrite grouping; optional
group boundaries; caller-provided sortable layout key; page/block permutation
when reencode is unnecessary; post-compaction permutation receipts; locality
statistics in CompactionMetrics; locality-debt trigger; stable logical row
IDs across physical moves; deferred physical clustering; incremental
locality repair. Do NOT upstream Morton/GridLake/Lotus/comma/temporal.rs/
graph semantics where a generic u64 / expression / layout policy suffices.

## 10. Cross-domain discovery mandate

Every researcher looks for equivalences (questions, not conclusions):
compaction as locality "repair"; shared grouping for erasure repair and
query locality; temporal closure as compaction trigger; cascade completion
as spatial carry chain; fragment-reuse maps as permutation witnesses;
parity-group boundaries as compaction boundaries; locality debt from
index-remap entropy; one SFC order minimizing both query scatter and RS
repair scatter; prepared fragments as immutable code symbols before one
manifest publication; multi-scale syndromes identifying stale-version
contamination; temporal delta altering only a secondary syndrome
orientation; identity-derived layout keys making canonicalization
constructive. Classify every surprise: KNOWN / TRANSFER / NOVEL-CANDIDATE
(documented search required) / DISPROVED.

## 11. Independence rule

The three researchers inside one domain must NOT share intermediate
conclusions during the first pass. Each returns independently: SOURCE
ARCHAEOLOGY / MECHANISM / EXPECTED BENEFIT / EXPECTED FAILURE / EXPERIMENT /
PRIOR ART / SURPRISES / VERDICT. Only after all 15 reports exist do
consolidation agents see them. (Dispatch note: the first pass is also blind
to `docs/lotus/**` and all board files — prior sessions' conclusions would
synchronize the researchers.)

## 12. Consolidation pass

One evidence matrix: claim / source evidence / measurement evidence /
literature evidence / supporting agents / dissenting agents / confidence /
falsifier / next experiment. Then a cross-domain relationship graph.
Independently rediscovered equivalences are high-value; unanimity with
shared ancestry is NOT independent confirmation.

## 13. Final attack pass (tiering)

TIER 0 measurement/falsifiers only; TIER 1 lance-graph internal experiment;
TIER 2 generic Lance prototype; TIER 3 credible Lance upstream RFC/PR;
TIER 4 paper-worthy result. Every upstream candidate answers: general user
benefit; no lance-graph semantics in the API; backward compatibility;
effects on fragment format / indexes / compaction / binary-copy / object
storage / version retention; failure behavior; the benchmark that convinces
an upstream maintainer.

## 14. Primary metrics

logical bytes produced; physical bytes written; physical bytes rewritten;
write amplification; read amplification; point-read latency;
grid-neighborhood latency; random take latency; fragment count; files
touched/query; pages touched/query; index remap bytes; index remap CPU;
compaction CPU; compaction wall time; repair read bytes; repair write bytes;
repair helpers/chunks touched; seal CPU; seal metadata bytes; cache misses;
peak RSS; restart recovery work; version count; stale-version retention
bytes. Do not optimize a metric that was not measured.

## 15. Falsifiers

F-ARRIVAL (same semantic input, adversarial completion order → same
canonical result); F-LAYOUT (random physical placement → correct reads and
seals); F-COMPACT (before/after compaction → same logical payload, identity,
logical RS validity); F-RS (all promised erasure patterns reconstruct
exactly); F-SILENT (silent corruption detected before reconstruction);
F-PHASE (project phase progression performs no worse than standard baseline,
retains rank); F-NOPHASE (no measurable advantage → remove it); F-SFC
(Morton must defeat or justify itself against raster/Hilbert/Onion);
F-TEMPORAL (scheduler timing never changes semantic time); F-HINDSIGHT
(STRICT/AWARE/RETRO return the expected epistemic projections); F-NOCOMPACT
(healthy with compaction disabled for the soak window); F-AMPLIFY (locality
improvement justifies bytes rewritten); F-UPSTREAM (generic proposal useful
with GridLake completely absent).

## 16. Final paper-shaped deliverable

Title / Abstract / 1 Problem / 2 Background / 3 Lance archaeology /
4 Temporal model / 5 Grid-SoA model / 6 Erasure-code model / 7
Locality-layout model / 8 Methodology / 9 Results / 10 Negative results /
11 Cross-domain discoveries / 12 Prior art / 13 Novelty audit / 14 Candidate
Lance extensions / 15 Threats to validity / 16 Conclusion. Appendices:
A source provenance; B falsifiers; C benchmarks; D coding matrices;
E locality curves; F compaction traces; G temporal traces; H agent
disagreement record. Dry systems-paper prose. Avoid "Lotus magic" /
"holographic" / "quantum" / "time machine" / "perfect random access" /
"self-healing" unless quantitatively demonstrated. Preferred vocabulary:
multi-scale, locality-preserving, version-aware, temporal-reference horizon,
incremental closure, logical-to-physical remapping, erasure-coded seal,
product-code, Reed–Solomon, space-filling curve, rewrite amplification,
repair locality, layout debt, compaction policy, structure-of-arrays,
prepared publication, causal timestamp, bidirectional syndrome, hierarchical
locality.

## 17. Research maxim

DO NOT MAKE THE BEAUTIFUL IDEA TRUE. MAKE IT SURVIVE FIFTEEN PEOPLE TRYING
TO MAKE IT FALSE.
