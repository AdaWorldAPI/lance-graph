# Co-research: the D-STAUNEN arc (#1885–#1889) against code and outside sources — 2026-10-31

**Status: exploration map. Ratifies nothing.** The operator chooses which probes run.

## Question
The arc is five docs-only maps (D-STAUNEN-0..4): plasticity as a DT64 capability over Atlas SNN events, HDC signatures, novelty-gated eligibility, XLOG/NSL hypotheses in sparse THOUGHT deltas, and 64-thought metaplasticity with root-disjoint evidence. Which of its gates have their own falsifier run, and what outside knowledge changes which gate to run first?

Premise gate (premise-auditor): PREMISE-SPLIT. A PR that overlaps a gate is candidate evidence, not admission. X6 splits into X6a (lineage and independent support) and X6b (attribution). HDC and SNN map only onto candidate discovery and novelty modulation, never onto evidence.

## Gate ledger (code cartographer; file:line in the scratch crosswalk)
- **Falsified by its own gate:** X3 only (lance-graph-nsl oracles, #1846).
- **Partial:** X0, X1, X2 (#1884 bag counts; no forced-overflow test on the checked multiply), X4 (no gradient check), X6 (test-level, #1896), X7, X8.
- **Candidate evidence:** C1, M2.
- **Absent:** C0, C2, C6, X5, M0, M1, M3–M8, A0–A7. For A0–A7 the closed search space was `crates/**/*.rs,*.toml` for brian2, stdp, atlasir, three-factor and eligibility traces.

## Fixed during the run
#1896's `derivation_multiplicity_is_not_independent_evidence` counted **lineage** (every root any derivation touches) and called it independent evidence. With A reporting 1→2→4→5 and B only 1→3→4, the derivations rest on {A} and {A,B}. Absorption leaves {A}: one independent support, where the test asserted two.

The test now computes the minimal-environment label (ATMS / PosBool, arXiv 2202.10766 §2–3) for three cases, and keeps lineage as a separate count. Four disables are red, each at a distinct assertion. Commits `9007724`, `da0feb5`, `c295d25`; the last scopes the label as a cold test-only oracle.

## Verdicts
| # | idea | verdict | why / probe |
|---|---|---|---|
| 1 | Independent support = root-disjoint minimal environments (X6a) | **PROBE** P-X6A-LEDGER | Already in production as `DerivationLedger` (`lance-graph-planner/src/nars/support.rs:308-323`, D-EPI-COT C1/C2). Probe: a plain-text fixture exported by xlog-relcomp-probe, replayed by a planner dev test into the ledger. `pooled` must agree with the probe's label on {A}/{A,B} and on disjoint {A},{B}. On the triangle {A,B},{B,C},{A,C} they **diverge by design**: the label has 3 environments, `pooled` selects 1. The test pins that divergence. The probe crate does not depend on the planner: the probe is a separate workspace because of arrow 60 vs 58, and the planner takes a path ndarray. |
| 1b | Label stable under cycle unrolling | **PROBE** P-X6A-CYCLE (run first) | `derivations()` lists simple paths only, so recursion is unexercised. Fixture: 1→2, 2→3, 3→2, 3→4; A = {1→2, 2→3, 3→4}, B = {3→2}. The label must equal {{A}} for k = 0..3 unrollings. Twin: a bypass 1→5→4 reported by C must give {{A},{C}}, disjoint. Disable: drop minimality. |
| 2 | N[X] is infinite under recursion | **PROBE** P-REC-NO-COUNT | Pin that no API returns a multiplicity for a recursive fact. Twin: the acyclic count is admitted. |
| 3 | One witness is not lineage | **PROBE** P-TRACE-NOT-DISJOINT | On the diamond, `trace` roots == 1 and the `support` label == 2. Twin: a chain. |
| 4 | Truncated proof sets prove overlap only | **PARK** | One-way rule; matters only once a k-truncated proof set exists. The ledger horizon is k = 1. |
| 5 | Retraction under recursion (DRed^c) | **PARK** | Stale frame refused plus recompute is enough until incremental retraction is asked for; a counter store would be a new store. |
| 6 | PROV attribution (X6b) | **ADOPT-NOW (vocabulary)** | Root, version and program hash stay on the test record and fixture header, not a new type. `reading` is an attribute of the result, not provenance. CE64 W never carries it (#1847 :405). |
| 7 | Root = agent or report? | **OPEN, operator** | `root()` hashes observer and facts, so one observer filing two reports counts twice. P-AGENT-VS-REPORT pins both readings once ruled. |
| 8 | Three senses of "provenance" | ADOPT-NOW (vocabulary only) | Attribution (PROV) / lineage-support (semiring) / encoding generation (`EdgeProvenance`). Rename nothing. |
| 9 | Cross-THOUGHT root-disjointness | SKIP (duplicate) | Already D-HL-3 (Queued, falsifier pre-registered; capstone :150). |
| 10 | Replay must not reinforce | SKIP (duplicate) | 2026-10-22 E2/E3/P-M1 and #1847 E1–E7 already cover it. |
| 11 | Novelty gating | PARK | Must inherit D-CE64-STAUNEN-0's kill condition: raw surprise lost to round-robin. |
| 12 | Eligibility traces / delayed credit | PARK | H3 inconclusive under Holm. Traces would make W carry evidence. |
| 13 | HDC/SNN for discovery only | ADOPT-NOW (ruling) | I-VSA-IDENTITIES; ScentLearner #1844 is the surface. Nengo is non-commercial: never a dependency. |
| 14 | Mask-union why-provenance holds only for monotone queries | **PROBE** (cheap) | X-gates over NOT/minus refuse; a test pins the refusal. |
| 15 | Stored N[X]/Why(X) carrier | SKIP (anchor) | 2026-10-21 PARK/SKIP; #1888 forbids a provenance store. |
| 16 | Signed diffs as multiplicity | PARK (lens) | A DD/DDlog engine would be a new engine. The concrete X2 gap is a forced-overflow test on `native.rs:45`: PROBE, cheap. |
| 17 | Kythe ref/call | PROBE (P-KYTHE-CALL) | #1894 emits `ref` only. The probe must start red; call facts need a source (Charon not installed). |
| 18 | egglog proofs | SKIP (resolved) | The PLDI'23 paper calls proofs future work; the fork has `new_with_proofs` and #1896 uses it. |
| 19 | Atlas Definition/Instance/Run identity | PROBE (document check, needs network) | The fetched Atlas README does not mention it. A0–A2 stay parked until it is found at file:line. Brian2 CeCILL-2.1: external oracle only. |
| 20 | NSL neighbour packing | SKIP (anchor) | Copies neighbour features per example (zero copy). Oracle use (#1846) stays. |

## Recommended order
1. P-X6A-CYCLE
2. P-REC-NO-COUNT and P-TRACE-NOT-DISJOINT (all offline, test-only)
3. P-X6A-LEDGER, which decides whether X6a is already met by the ledger
4. The X2 overflow test

A-gates wait for P-ATLAS-IDENTITY. M-gates cite D-HL-3 and the 2026-10-22 probes rather than re-plan them.

## Not searched
- Atlas source tree: only the README was fetched.
- Green et al. 2007, de Kleer 1986, Nieuwenhuis–Oliveras and the Soufflé TOPLAS paper are cited secondhand.
- Miconi 2018's arXiv id is unverified.
- Scallop's licence file was not read.
- The arc's ACT-R, PointCare, qualia and Brain-SoC context was not researched, by design.
- `holograph`, `cognitive-shader-driver/src/serve.rs` and `learning/src/cam_ops.rs` matched eligibility/STDP terms and were not read.

PR #1896 | STATUS: open | OUTCOME: co-research map for the D-STAUNEN arc; lineage-as-evidence defect fixed | OPEN: root = agent or report; which probes to run

MIRROR
- BLIND SPOT: #1896's provenance test was reviewed for multiplicity versus roots, but never against a case where a second observer depends on the first.
