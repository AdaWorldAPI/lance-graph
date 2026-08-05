# D-BLW-5 — design note (MAIN-THREAD authored, 2026-08-05)

> **Provenance:** the Opus design LANE for this deliverable was stopped by
> the operator mid-run and is not relaunched. This note is authored by the
> orchestrating main thread itself (Opus-class filigree per the model
> policy), which respects the agent stop while completing the synthesis the
> plan requires. It incorporates three inputs the stopped lane never saw:
> the operator's 64k main-model ORDER (E-64K-1TO1-OWNERS-IS-THE-MAIN-MODEL-1),
> the completed Sonnet inventory (`d-blw-5-api-inventory-sonnet.md`), and
> the orchestrator's placement ruling.
> **The BUILD is gated on the operator's word.** Nothing here dispatches.

Binding doctrine: `.claude/knowledge/observer-effect-tfpn-doctrine.md` +
plan §12.9/§12.9a. Foundation: `tests/d_ign_b_lenses.rs` (GREEN) — the
per-owner arena runs in-cycle, selected by arming.

---

## (a) Placement — supervisor tests, with the pre-authorized jc edge

`crates/lance-graph-supervisor/tests/d_blw_5_observer.rs`, feature-gated
`#[cfg(feature = "cycle-driver")]` (the sibling pattern). Forced by the
single-system requirement of the single-measurement law: S₀ → inject → S₁
must happen inside ONE evolving arena+loop process, and only the supervisor
sees `run_cycle` (inventory §C). The supervisor cannot reach `jc` today;
the ORCHESTRATOR-RATIFIED manifest change (the one exception): add
`jc = { path = "../jc" }` to lance-graph-supervisor `[dev-dependencies]`,
under the four D-BLW-3 constraints (dev-only; never production; never
modify `crates/jc`; never invert — jc stays zero-dep). `ndarray` stays
unreachable supervisor-side by design; the shape census is probe-local
(below) with a provenance comment citing `ndarray::simd::cascade` (on
ndarray master via merged PR #273) as the machinery it stands in for.

## (b) Awareness coupling, the S readers, and the C6 firewall

- **Coupling:** each armed owner runs the D-IGN-B lens body — `stream`
  fills a per-owner `BeliefArena` from the owner's corpus slice (fresh
  `Interner` per owner, the L1 id-independence discipline).
- **Injection is arena-native:** `BeliefArena::observe()` accepts
  hand-built statements (inventory §A — no text path needed). The payload
  is a small belief family, every member stamped with a probe-reserved
  provenance marker (`Stamp`): 16 bucket-beliefs (census masses as truth
  frequency) + 1 rank-belief (rank₀ as frequency). ELEVATED-rung semantics
  are carried by the stamp, not a new field.
- **Propagation channel (what makes T non-vacuous):** after injection, one
  post-injection reasoning pass runs (the same pass2/fixed-point step the
  lens body already uses), so the injected testimony CAN interact via NARS
  revision and derived closure. Without this pass, T-silence would be
  structural, not measured.
- **The S readers (pre-registered):** two arena-derived binary readers
  over the owner's verse-subjects —
  A: "subject participates in ≥1 contradiction-ranked statement";
  B: "subject has a rung-lift record".
  Pre-registered FALLBACK (the D-IGN-B pattern, pinned NOW, never chosen
  after output): if either marginal is degenerate (0 or 1) on the UNARMED
  control at V₀, substitute B′: "subject appears in ≥2 distinct
  Wittgenstein games". S = `jc::stats::binary_association(A, B)` pooled
  per arm-cohort. Full tables always (C2); κ and φ reported, never bare.
- **The C6 firewall, precisely:** (1) the readers EXCLUDE the verbatim
  injected statements (matched by the probe-reserved stamp) — everything
  downstream of revision is fair game, because that propagation IS the
  measurand; (2) S₀/S₁ are computed in a measurement block AFTER the final
  seal, and a compile-time self-scan (the sibling probes' `include_str!`
  pattern, self-match-guarded) asserts the S identifiers appear nowhere
  before the measurement marker — nothing in the loop can branch on them;
  (3) jc is called only inside the measurement block — the one-way oracle.

## (c) The payload, concretely

At V₀ (per arm-cohort): compute each owner's per-subject association
contribution pool; census it into a probe-local 16-bucket histogram in
Fisher-2z space (equal-width in 2z ≈ equal-information; provenance comment
cites the Belichtungsmesser); rank₀ = the observed cohort statistic's
Prozentrang within the pooled prior distribution. shape₀ × rank₀ is ALL
that is injected — never a raw κ (the payload law). The ledger:
`BTreeMap<(StatId, Arm, Cohort, Metric, DatasetVersion), Sealed>` —
append-only; a second write at a sealed key ERRORS (the remeasure guard).

## (d) The arms — cohorts of INDEPENDENT 1:1 owners (the 64k order)

Per E-64K-1TO1-OWNERS-IS-THE-MAIN-MODEL-1: arms are COHORTS OF OWNERS,
1:1, mutation-exclusive — never row-partitions of one owner. Pinned:

| cohort | owners | injection |
|---|---|---|
| T | 8 | shape₀ × TRUE rank₀ |
| F+ | 8 | shape₀ × rank shifted +1.5 logit units |
| F− | 8 | shape₀ × rank shifted −1.5 logit units (equal magnitude, opposite sign; anchors with true rank outside [0.05, 0.95] are EXCLUDED, never clipped) |
| P | 8 | structurally identical belief family, permuted-pool shape, zero information |
| N | 4 | the T-injection observed through the awareness-free bloom criterion (must stay frozen BY CONSTRUCTION; doubles as pool-drift baseline) |
| UNARMED | 4 | none (the degeneracy control for the reader fallback) |

Cycle plan (pinned): c1 seed + lens pass → seal → **V₀** → measure S₀ per
arm (ledger-sealed) → c2 inject (arena writes, stamped; a WRITE, the
two-verbs doctrine) → post-injection pass → seal → **V₁** → measure S₁
(fresh one-shots; S₀ is never recomputed — the single-measurement law).
Synchronous loop; fleet-level parallel stays A2-gated (not-claimed).

## (e) Pre-registered gates (every one twinned, all numbers final NOW)

- **O1 remeasure guard:** recompute at a sealed key ERRORS (can-fire);
  fresh `(id, scope, V₁)` passes AND a different arm at the same
  `(id, V)` passes (can-stay-silent).
- **O2 placebo:** |ΔS(P)| < 0.10 required — twin: P's injection mechanics
  DID execute (stamped beliefs present in every P arena), so the silence
  is informational, not absence.
- **O3 null instrument:** N's criterion byte-identical V₀→V₁ (frozen by
  construction) — twin: N's own pool-drift measured and reported as the
  baseline.
- **O4 the observable:** ΔS(T) = S₁−S₀ fires iff |Δκ| ≥ 0.10 (two-sided,
  the corrected convention). BOTH outcomes are named results: fire = the
  effect; silent at every floor = the honest null ("awareness does not
  reflect this statistic").
- **O5 direction test:** both F arms run; S₁ tracking the INJECTED rank =
  anchoring/Goodhart (a finding even if T is silent); correcting TOWARD
  truth = evidence-dominance; value-invariant movement = perturbation.
- **O6 firewall self-scan:** S identifiers absent before the measurement
  marker (can-fire) + the scan finds the marker and the jc call (a scan
  that finds nothing is not evidence).
- **O7 exclusion is load-bearing:** a SHADOW reader including the
  injected statements yields a different S than the firewalled reader on
  T-cohort arenas (can-fire) — and the firewalled reader is non-empty
  (can-stay-silent).
- **DROP:** |Δκ| < 0.01 with zero reader-Hamming across BOTH cycles for
  the arm — all-horizon Hamming, the C7-corrected form.

## (f) Kill conditions (pre-accepted) + not-claimed

Kills: P moves ⇒ instrument invalid (reported, not tuned); N moves ⇒
plumbing leak, run void; T silent everywhere ⇒ honest null; F tracks
injected rank ⇒ the anchoring finding stands alone. Not-claimed: no
validity, no parallelism (A2-gated), no durability (MemWal), no fusion
verdict, no generalization past this corpus/instrument, no per-stance
dispatch (selection), no substrate-data-path claim from any readout
(F1b carries over verbatim).

## (g) Open items for the operator

1. **The build itself** — gated on your word (this note + the inventory
   are the complete build inputs).
2. The supervisor+jc dev-dep lands with the build commit (pre-ratified).
3. If O4 fires, D-BLW-3b's arm C inherits this instrument as designed
   (TD-BLW3B re-route).
