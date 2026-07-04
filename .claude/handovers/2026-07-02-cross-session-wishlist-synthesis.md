# Cross-session wishlist synthesis — 2026-07-02 (post-#630/#631)

> From: the medcare-bridge session (branch `claude/medcare-bridge-lance-graph-wmx76z`).
> Inputs: this session's E1–E10 sweep (OGAR@19373a2 + ruff@b459ec3 review fan-out),
> the coverage session's grounded wishlist (ruff/OGAR/V3 items with acceptance
> tests), op-nexgen's R/L/O/X wishlist (2026-07-02). Operator relayed all three.
> APPEND-ONLY once committed; corrections cite their pass.

## Ratified-by-convergence (≥2 sessions independently)

1. **Two-sided fuses** — OGAR-side flip/COUNT_FUSE pinning test; "a dependency
   contract named on one side is a slogan." (this session #1 ≡ coverage #7)
2. **Counts pinned by tests, never prose** — predicate-count pin (62), MethodSig
   existence/parity pin, post-flip prose sweep. (E8 ≡ coverage #1/#2)
3. **Governance metrics = zero-dep contract folds** — classid_scan (#630) →
   emission_scan (op-nexgen L2) is a design language now; name it in EPIPHANIES
   so the next metric follows it (never a grep).
4. **F17 body triage** — unblocked since 06-30, consumed by nothing; flagged by
   all three sessions. Most-agreed next move on the DO path. (mine #9 ≡ R6)
5. **Board prepend-collision fix** — per-entry board files (coverage #8) +
   COORDINATION.md per repo (op-nexgen X1). Measured cost datapoint: 3 of this
   session's rebases conflicted ONLY on board prepends.

## Dedup decisions (avoid parallel work)

- **E2-ruby DROPPED in favor of op-nexgen R1** (D-AR-3.5 column stratum from
  the migration DSL — better Rails truth source than model-body extract_fields;
  measured 99/65/27-typed on the real corpus). E2-python (`inherits_from`
  emission in ruff_python_spo) STANDS — uncovered elsewhere; gates
  mint_factored's is_a axis on Odoo.
- **"Disposition ledger" (coverage #5) ≡ the 3-bucket DO triage**
  (`do-arm-triage-3-bucket.md`, #625): buckets = the routing decision,
  disposition ledger = its conservation accounting. Cross-reference as ONE
  doctrine; do not let two vocabularies grow.

## ruff needs ONE sync arc (and minimal governance)

Pending ruff work from three sessions: mint_factored split-brain rebase (E1),
coverage items 1–5, nexgen R1–R8 (incl. a vendored diff targeting "pristine
main" and R7's already-failing branch fixture). Proposed integration order:
(1) mint_factored rebase (smallest, oldest, ruff_spo_address only) →
(2) R1 D-AR-3.5 while main is pristine for the diff →
(3) R2 curated fixes + R7 fixture reconciliation →
(4) predicate-manifest parity test LAST (pins the union, guards forever).
Plus: give ruff a COORDINATION.md + 20-line LATEST_STATE before the arc starts.

## One operator ruling covers two byte-truth conflicts

E7 (hi-u16: `domain:appid` in le-contract vs `domain:concept-slot` in OGAR
canon) and coverage #9 (EdgeBlock: OGAR `key16+value496` vs lance-graph CANON
`key16|edges16|value480`) are the same defect class: byte layout stated
divergently in two append-only ledgers. Settle both in one ruling; the form of
the settlement: one set of consts + size asserts both repos PIN, both
CLAUDE.mds POINTING, not restating (#630 M7 is the in-repo precedent).

## Columnar-interchange guard

op-nexgen L3 (Arrow triples, 5 columns s p o f c) and this session's E5 (Mint →
Arrow facets batch) MUST share one `ruff-interchange` record-batch schema
family + one provenance header (R5), and the provenance stamp must include
`minter@sha` — E1's split-brain proved an artifact that doesn't name its
minter is unverifiable. Two adjacent Arrow schemas defined independently would
recreate the field_type three-dialect problem at the columnar layer.

## Gap no list claimed

**Probe-corpus archival rule, operationally:** F17, count_adoption-vs-bake, and
the medcare probe re-run will each produce a number quoted for months — archive
the reproduction (input + generation recipe + hash) WITH the run, or we
re-create "last measured, can't re-verify."

## Kept by this session

E1 mint_factored rebase (step 1 of the ruff arc), E2-python inherits_from,
C#-golden vs Predicate::ALL, E5 Mint emission (pending the shared interchange
schema decision), predicate-manifest test (step 4).
