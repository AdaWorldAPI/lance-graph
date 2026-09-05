# Observer-Effect Measurement Doctrine — the TFPN design (D-BLW-5)

> **READ BY:** truth-architect, certification-officer, integration-lead, and
> any session building D-BLW-5 or ANY probe that feeds a measured statistic
> back into the awareness lane (jc loops, MUL-coupled criteria, NARS
> belief-injection experiments).
>
> **Status:** design doctrine, operator-ruled 2026-08-04. The *machinery*
> references are FINDING (every cited surface exists in source, anchors
> below). The *effect* is CONJECTURE — D-BLW-5 is unmeasured, queued behind
> PROBE-IGNITION. Plan: `.claude/plans/cycle-loop-closure-driver-v1.md`
> §12.9 + §12.9a.

---

## 1. The claim under test

D-BLW-3 (§12.8, SHIPPED + MEASURED) measured **first-order fusion**: horizons
merge by sharing *data* (pool growth; Δκ −0.485 → 0 over 8 horizons).
D-BLW-5 measures **second-order fusion**: horizons merge by sharing the
**measurement of each other**. Information about the correlation of a
dataset, when it enters the awareness, influences the correlation — the
observer effect, run deliberately and instrumented instead of avoided.

The Click's own arrow is the hook: `awareness.revise(key, outcome)` →
`global_context += fact` → reshapes the NEXT cycle's F landscape. Here the
injected fact is *about the cohort's own statistics*.

---

## 2. The payload law — distribution × Prozentrang, NEVER the raw statistic

**What is injected is not the correlation.** Injecting the raw scalar (κ, φ,
or the full `BinaryAssociation` as a value) builds the Goodhart collapse into
the instrument: a scalar is trivially echoable, so the anchoring fixed point
(awareness parrots the number back) is available by construction and the
F-arms cannot distinguish anchoring from reflection.

**What is injected — the preserving payload:**

1. **The distribution SHAPE** of the statistic over the *prior* pool — a
   palette256/HDR-bucketed census (banded exposure with popcount-stacking
   early exit, statistical confidence-interval thresholds, preheating +
   rolling floor bucket — the Belichtungsmesser reading).
2. **The Prozentrang** — the percentile rank of the observed association
   *within that prior distribution*. A rank-within-a-shape says where the
   observation SITS without handing the awareness a value to parrot.

**Machinery anchors (FINDING — these exist):**
- `ndarray::hpc::cascade` — `expose(distance) → Band`
  (Foveal/Near/Good/Weak/Reject, `cascade.rs:162-175`) +
  `recalibrate(&mut self, alert: &ShiftAlert)` (`cascade.rs:211`): the
  banded exposure meter with recalibration.
- `ndarray::hpc::statistics` — `percentile(&self, p)` (`statistics.rs:41`).
- Exact wiring of shape-census → injection payload is pinned at build time;
  the doctrine binds the SHAPE of the payload, not an API.

---

## 3. The single-measurement law — once measured, never remeasured

**A measurement burns the state it measured.**

- S₀ is measured **once**, at version V₀, and sealed (version-stamped).
- After injection, the system that produced S₀ no longer exists. Running the
  instrument again yields **S₁ at V₁ — a NEW one-shot measurement of a
  DIFFERENT (post-injection) system** — never a "remeasure of S₀". A
  remeasure of S₀ does not exist even in principle.
- The only thing that carries forward from V₀: **shape₀ × rank₀**, frozen.

**Enforcement = temporal.rs hindsight blindness × the shape sensor:**

- `temporal.rs` supplies the blindness: rung-gated version-range reads — a
  Strict-rung reader at V sees only ≤ V. The D-BLW-3 fusion example is the
  proven precedent (`no_hindsight_streamed_known_game`; `QueryReference::at`
  + `deinterlace`).
- The shape sensor's output rides as **META only**: rung-marked ELEVATED
  (statistic-as-witness, higher-rung derivation — the zero-copy carve-out),
  never as corpus, and **never recomputed over post-injection data and
  back-dated to V₀**.
- This combination is what makes the probe viable *without* remeasurement:
  every arm's injection payload derives from the V₀-sealed shape; every
  arm's observable is a fresh V₁ one-shot; the comparison is between two
  version-stamped one-shots, each blind to what came after it.

---

## 4. The TFPN arms, with their philosophical readings

| arm | injection (per §2) | reading | pre-registered expectation |
|---|---|---|---|
| **T** (true) | shape₀ × true rank₀, ELEVATED-rung | **Gadamer, Wirkungsgeschichte**: understanding is conditioned by what tradition hands it — the fused horizon is *wirkungsgeschichtliches Bewusstsein* made measurable | the observable: S₁ − S₀ (two one-shots, V₁ vs V₀). Fires iff it clears the pre-pinned floor. |
| **F+ / F−** (false) | true shape₀ × FALSE rank — **bounded-safe payload**: equal-magnitude opposite shifts applied in **logit(rank) space**, which is unbounded and symmetric by construction, so no boundary clipping can occur; anchors whose true rank falls outside a pinned eligibility band [δ, 1−δ] are **excluded from the F-arms entirely, never clipped** (clipping would silently unequalize the ± magnitudes and confound the direction test) | **the direction test.** S₁ tracking the injected rank = **Gadamer's Vorurteil as anchoring** (prejudice-structure dominating evidence) = **Goodhart realized** (a measure that becomes a belief ceases to be a good measure). S₁ correcting TOWARD truth against the injection = evidence-dominance (the hermeneutic circle revising its prejudice). Movement invariant to the injected value = mere perturbation, no verdict. | both directions MUST run — a one-sided F is not a direction test. |
| **P** (placebo) | structurally identical shape-and-rank envelope, zero information (choice pinned at build time; note: "uniform shape + median rank" is NOT empty — it asserts *nothing unusual*, a real statement) | instrument validity | **must not move.** If P moves S₁, the instrument measures injection mechanics, not information. |
| **N** (null instrument, free) | the T-injection observed through the §12.8 bloom-rank criterion | the awareness-free control | **must stay frozen BY CONSTRUCTION** — that criterion has no awareness term. Second duty: N's own V₀-vs-V₁ shape drift measures pool drift absent awareness coupling — the baseline the observer effect is read against. |

**Mechanical prerequisite (honest):** the §12.8 popcount-rank instrument
CANNOT exhibit the effect — no awareness input — which is exactly what makes
it arm N. The observed reader must be awareness-coupled: NARS-revision side
(the injected shape×rank participates as a belief, interacting via
support/contradiction) or a MUL-qualia-coupled criterion. Choosing which is
the probe's first build decision.

---

## 5. The falsification regimen

Pre-registration and guards, all binding BEFORE any run:

1. **Pre-registered, non-adjustable numbers.** Every floor, band, and rank
   shift is pinned at build time, before the first run. Hand-tuned values
   are labeled as such per I-NOISE-FLOOR-JIRAK. Post-hoc adjustment of any
   threshold voids the run.
2. **Kill conditions, pre-accepted:**
   - **P moves** ⇒ instrument invalid. Reported, not tuned away.
   - **N moves** ⇒ plumbing leak; the run is void (the G2 pattern one level
     up: an awareness-free criterion that responds to awareness input is a
     defect in the harness, not a discovery).
   - **T silent at every floor** ⇒ the finding is the honest null:
     "awareness does not reflect this statistic." True and useful.
   - **F tracks injected rank** ⇒ the anchoring/testimony-dominance finding
     stands even if T is silent — Goodhart-vulnerability is itself the
     discovery.
3. **Guard twins (house falsifiability rule):** every gate carries a
   can-FIRE test and a can-STAY-SILENT test, both on non-trivial inputs.
4. **The remeasure guard (new, from the single-measurement law):** the
   measurement ledger is append-only, keyed
   `(statistic-id, arm, cohort, metric, version)` — scope-qualified, so
   independent arms/cohorts/metrics legitimately writing at the same
   version never collide with each other; only a true recompute of the SAME
   scoped one-shot hits a sealed key. A second computation attempt at a
   sealed key must ERROR.
   - can-fire: a test attempts the recompute and proves the guard barks;
   - can-stay-silent: a fresh `(id, scope, V+1)` one-shot passes untouched,
     AND a different arm's write at the same `(id, version)` passes.
5. **Direction-test symmetry:** F+ and F− both run, same magnitude of
   shift in logit(rank) units, opposite signs — the equal-magnitude
   requirement is defined in the space where it cannot be broken by the
   rank bounds (see the F± payload rule in §4).
6. **No p-values** (C4). The paired contrasts + placebo + null-instrument
   arms ARE the inference. Full tables, never bare κ (C2 naming).
7. **Anti-circularity, instrumented not violated (C6):** C6 forbids a
   witness gating the slice it was computed on because that is a
   self-proving loop. This probe deliberately CLOSES that loop and MEASURES
   it — therefore **nothing downstream may gate on S₁**, ever. The loop is
   an observable, never an admission criterion.
8. **jc stays the one-way oracle:** `crates/jc` measures S₀ and S₁, is
   never modified, and is never fed its own output as input. The loop runs
   through the system's awareness, not through jc.

---

## 6. Cross-references

- Plan: `.claude/plans/cycle-loop-closure-driver-v1.md` §12.8 (D-BLW-3
  first-order result), §12.9 (D-BLW-5 arms), §12.9a (payload refinement —
  this doctrine's plan-side mirror).
- Board: `EPIPHANIES.md` `E-HORIZONTVERSCHMELZUNG-GAP-CLOSES-1` (first-order
  gap closure), `E-MEASUREMENT-BURNS-THE-STATE-1` (the single-measurement
  law).
- Iron rules: I-NOISE-FLOOR-JIRAK (threshold labeling);
  falsifiability rule (CLAUDE.md P0 — can-fire/can-stay-silent twins).
- Precedent code: `crates/lance-graph-planner/examples/blw_fusion.rs`
  (version-gated hindsight-blind reads, the G-gate discipline);
  `crates/lance-graph-planner/src/temporal.rs` (`QueryReference::at`,
  rung admission, `deinterlace`).

---

## 7. Status note (2026-09-05) — first measurement landed

D-BLW-5 was resumed by operator ruling with the belief-arena reader and measured
(`crates/lance-graph-supervisor/tests/d_blw_5_observer.rs`; plan §12.9b;
E-BLW5-FIRST-MEASUREMENT-1). Every gate in §5 held. O4/O5 read SILENT at the κ floor while
the reader marginals saturated in proportion to the injected typicality — the effect the
doctrine names (§4 "S₁ tracking the injected rank") is visible in the marginals and
invisible to κ. Two amendments to the machinery this doctrine only sketched: (a) the rank
enters awareness as TYPICALITY (mass at rank) carried in the confidence of
`subject Inh prior` at frequency 1 — a frequency below 0.5 is discarded by the arena's
expectation-CHOICE (E-NARS-EXPECTATION-CHOICE-PREFERS-IGNORANCE-TO-A-CONFIDENT-NEGATIVE-1);
(b) the awareness-coupled reader must read the DERIVED layer — observed truth never moves
from testimony, by the arena's own ground-protection rule. The observer EFFECT stays
CONJECTURE beyond this corpus/instrument; the next instrument (D-BLW-5b) pins marginal floors.
