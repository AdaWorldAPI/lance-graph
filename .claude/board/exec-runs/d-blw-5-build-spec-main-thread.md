# D-BLW-5 — BUILD SPEC (main-thread authored, 2026-09-05; operator: "resume the D-BLW-5 loop with the belief arena reader")

> Binding inputs, in precedence order: `.claude/knowledge/observer-effect-tfpn-doctrine.md`
> (doctrine) → `.claude/board/exec-runs/d-blw-5-design-main-thread.md` (design (a)–(g)) →
> `.claude/board/exec-runs/d-blw-5-api-inventory-sonnet.md` (exact signatures; wins on any
> conflict) → THIS file (the pinned build). Every number below is FINAL before the first run
> (doctrine §5.1). Hand-tuned values are labelled `[hand-tuned]` per I-NOISE-FLOOR-JIRAK.

## 0. Two corrections to the 2026-08-05 design note (recorded, not silent)

C1. **The note's readers A/B were injection-invariant by construction.** Both read only
fields that `stance::stream` writes (`contradiction` moves only through `revise_at` on an
OBSERVED statement; `lifts` only through `stream`). `admit_derived` never overwrites a
grounded belief (the Codex stamp fix). So with those readers T-silence would have been
STRUCTURAL, not measured — the vacuity the falsifiability rule forbids. Reader B is
therefore re-pinned to read the arena's DERIVED layer (§4), which the tactics DO move.

C2. **The rank enters awareness as typicality, not as a bucket index.** Encoding the rank as
`(subject Inh band_r)` at f=1 makes `rcr_abduce` link every subject to every other through
the shared reserved predicate `band_r` at c≈0.447 REGARDLESS of the shape — pure injection
mechanics, which the P arm would rightly kill. The composable encoding is the one the doctrine
already names as the meaning of Prozentrang: "where the observation SITS in the prior". The
payload's 16 shape beliefs carry the masses; the rank binds the cohort to the prior at
frequency = the prior's mass at the observed rank (§5). Two ranks with EQUAL mass are
indistinguishable to awareness under this encoding — a stated limitation, not a bug.

## 1. Placement (design (a), unchanged)

- New file: `crates/lance-graph-supervisor/tests/d_blw_5_observer.rs`, whole file wrapped
  in `#[cfg(feature = "cycle-driver")] mod d_blw_5_observer { ... }` exactly like
  `d_ign_b_lenses.rs`. One `#[tokio::test] async fn d_blw_5_observer_effect_belief_arena()`.
- Manifest: `crates/lance-graph-supervisor/Cargo.toml` `[dev-dependencies]` gains
  `jc = { path = "../jc" }` with the comment: "D-BLW-5 oracle ONLY (dev-dep; never production;
  jc is never modified and never fed its own output — doctrine §5.8). Pre-ratified in
  exec-runs/d-blw-5-design-main-thread.md (a)."
- CI: `rust-test.yml:173` already runs every supervisor test binary under
  `--features supervisor,cycle-driver`; nothing to add.
- Probe-local private structs/enums/fns inside the test module ARE permitted (precedent:
  `LensReadout`, `MemWal`, `ScanResult` in `d_ign_b_lenses.rs`). Nothing shipped is minted.

## 2. Scaffold — copy from `d_ign_b_lenses.rs`, provenance-commented at each site

Copy VERBATIM (rename nothing): `fnv1a`, `bloom_add`, `tokens`, `encode_plane`, `BLOOM_K`,
`flow_qualia`, `thinking_style_for`, `style_vector_for`, `plan_context_for`, `mantissa_of`,
`RowSpanDescriptor`, `row_span_payload`, `SealedCycle`, `MemWal` (+ its `WalSink` impl),
`build_owner`, `ScanResult`, `scan_board`, `ColumnPassOutcome`, `column_pass`,
`plan_or_evaluate_think`, `owner_verses`, `labelled_verses`. Same imports as that file PLUS:
`use jc::stats::{binary_association, fisher_2z, BinaryAssociation};`
`use lance_graph_contract::shape_rank::{RemeasureError, RemeasureKey, RemeasureLedger, ShapeRankPayload, SHAPE_BUCKETS};`
`use lance_graph_planner::nested_bands::{quantize_2z, NestedBandsBuilder};`
`use lance_graph_planner::nars::stance::{stream, Interner, ReadOut};`
`use lance_graph_planner::nars::tactics::{rcr_abduce, Throttle};`
`use lance_graph_planner::nars::{BeliefArena, CStmt, Copula, Stamp, TruthValue};`
Drop what is unused (clippy `-D warnings` is the gate; the orchestrator runs it).

Pre-registered run shape:
```
FLEET_OWNERS = 40 (MailboxId 0..40), ROWS_PER_OWNER = 64, POPULATED_ROWS = 48,
CORPUS_VERSES = 40*48 = 1920, SCOPE = 0..40, CYCLES = 2, every owner armed z=1
(Analytical), content salt = u64::from(id), flow_qualia(), firing_rows = 3.
Cohorts by id (design (d), 1:1 owners, never row-partitions):
  T   =  0..8    inject shape0 × TRUE rank0
  FP  =  8..16   inject shape0 × rank shifted +1.5 logit
  FM  = 16..24   inject shape0 × rank shifted −1.5 logit
  P   = 24..32   inject uniform shape × median rank (8)   [zero-information envelope, pinned]
  N   = 32..36   inject the T payload; OBSERVED through the awareness-free bloom criterion
  CTRL= 36..40   no injection (degeneracy control + DROP twin)
```

## 3. Corpus — deterministic synthetic ONLY (no env var, no KJV path)

`fn window(w: usize) -> Vec<String>` (exactly POPULATED_ROWS lines), predicate terms via
`synth_term(w, n)` copied from d_ign_b (`SYNTH_STEMS`), subject terms
`fn subj(w, i) -> String = format!("sub{w:02}{i:02}")` (7 chars, alphanumeric, in no
catalogue; a bare content word becomes the clause subject because no pronoun precedes it —
`stance.rs` "Bare content word: subject anchoring"). Copula "was" arms the predicate; "was
not" negates. NEVER use pronouns, "because", perception verbs, or words ending in "ed".
```
n_subj   = 5 + w % 4                      // 5..8 subjects
shared:  "sub_w_0 was T(0)." , "sub_w_1 was T(0)."                       // s0,s1 share T(0)
         if w % 2 == 0 also "sub_w_2 was T(1)." , "sub_w_3 was T(1)."     // s2,s3 share T(1)
own:     for i in 0..n_subj, for j in 0..(2 + (i + w) % 3): "sub_w_i was T(10 + 10*i + j)."
contra:  for i in 0..(2 + w % 3): "sub_w_i was T(80 + i)."  then LATER (after all own lines)
                                  "sub_w_i was not T(80 + i)."            // revision → contradiction 0.85
pad:     "sub_w_{j % n_subj} was T(100 + j)." until 48 lines
```
Assert `out.len() <= POPULATED_ROWS` before padding (max 4 + 8*4 + 8 = 44). Corpus =
`(0..40).flat_map(window)`. Labels: `labelled_verses` ("kjv:{:05}" — keep the format; it is
only a label).

## 4. The owner's mind, the reasoning pass, the readers (C6 firewall)

```rust
struct Mind { arena: BeliefArena, intern: Interner, out: ReadOut,
              subjects: Vec<u16>,          // distinct `p.stmt.s` over out.provenance, sorted
              reserved: HashSet<u16> }     // empty until injection
const THROTTLE: Throttle = Throttle { c_min: 0.0, budget: 65_536, hub_indegree: usize::MAX }; // permissive [pinned]
const MAX_PASSES: u32 = 64;
fn build_mind(verses: &[(String,String)]) -> Mind   // stream(.., pass2=false), then reason()
fn reason(arena: &mut BeliefArena)                   // f = rcr_abduce(arena,&THROTTLE); for c in f.candidates { arena.admit_derived(c.stmt, c.truth, &c.premises, c.rung); } arena.close_transitive(MAX_PASSES);
```
`reason` is the ONLY propagation channel and runs at BOTH versions (V0 pre-injection, V1
post-injection) for EVERY owner including CTRL — so V1−V0 on CTRL measures pass idempotence,
not injection.

Readers, per verse label `v` with provenance (verses with no provenance are skipped in BOTH
vectors, and the skip count is printed):
- **A (evidence, injection-invariant by construction — documented as such):**
  `∃ p ∈ out.provenance: p.verse == v && arena.get(p.stmt).contradiction > 0.05`.
- **B (awareness-coupled, "inferential corroboration"):** let `(s, p)` be the verse's FIRST
  provenance stmt. `∃ b ∈ arena.entries(): b.stmt.cop == Inh && b.stmt.p == p && b.stmt.s != s
  && subjects.contains(&b.stmt.s) && !reserved.contains(&b.stmt.s) && b.stamp == Stamp::default()
  && b.rung >= 1 && b.truth.confidence >= C_MIN`, `C_MIN = 0.01` [hand-tuned; derivation: the
  injected chain reaches c ≈ m·0.9·c_ab·0.9 with c_ab = 0.81m/(0.81m+1); P's m = 1/16 gives
  0.0024 (silent), m ≥ 0.13 gives ≥ 0.012 (fires)].
- **B_shadow (O7 only):** B without the `!reserved.contains(..)` and `subjects.contains(..)`
  clauses (any s' ≠ s).
- **B′ (pre-registered FALLBACK, decided at V0 on CTRL only, never after output):** if A or B
  is degenerate (rate 0 or 1) on the pooled CTRL cohort at V0, print the fallback line and use
  B′ = "the verse's subject participates in ≥ 2 distinct Wittgenstein games" (`stance_panel`
  4th element, games count for `s`) for ALL cohorts at both versions.
- `S(cohort, version) = binary_association(&a, &b)` over the concatenation of the cohort's
  owners' verse vectors (owner order ascending). Print the FULL table (copy
  `print_association_table` from `blw_fusion.rs:689-705`, cite it). Never a bare κ.

## 5. Payload, prior pool, arms, ledger, injection

At V0 (after the c1 seal), over ALL 40 owners: `phi_owner = S(owner-as-cohort-of-one).phi`.
Pool = `quantize_2z(fisher_2z(phi))` for every `Some(phi)`; PRECONDITION `pool.len() >= 16`
else panic "PRECONDITION: prior pool too thin ({n} < 16) — a corpus defect, not a finding".
`let nb = NestedBandsBuilder::new(SHAPE_BUCKETS).calibrate_equal_width(&pool, v0.0);`
Per injected arm cohort X ∈ {T, FP, FM, P, N}: `phi_c = S(cohort X, V0).phi`; if None → the
arm is DEGENERATE-AT-V0 (printed, excluded from every gate, named result). Else
`obs = quantize_2z(fisher_2z(phi_c))`, `payload_true = nb.shape_rank(obs, v0.0)`.
- T, N: `payload_true` (N seals under its own arm id).
- FP/FM: `rf = payload_true.rank_fraction() + 1.0/32.0` (bucket midpoint). If
  `rf < 0.05 || rf > 0.95` → arm EXCLUDED (printed, never clipped). Else
  `l = ln(rf/(1−rf)) ± 1.5`, `rf2 = 1/(1+exp(−l))`, `rank2 = min(15, floor(rf2·16))`,
  `ShapeRankPayload::new(payload_true.shape, rank2, v0.0)`.
- P: `mass = pool.len()`, uniform shape `q = mass/16, r = mass%16`, buckets `0..r` get `q+1`,
  the rest `q`; rank 8. `ShapeRankPayload::new(shape_uniform, 8, v0.0)`.
Ledger (`RemeasureLedger::new()` in the test body): key
`RemeasureKey { stat_id: 1, arm: {T=1,FP=2,FM=3,P=4,N=5}, cohort: <cohort lo id>, metric: 1 (=phi), dataset_version: v0.0 }`
sealed with the arm's payload BEFORE any injection.

`fn inject(mind: &mut Mind, payload: &ShapeRankPayload)`:
```
RESERVED_STAMP = Stamp::source(63)   [pinned; corpus src ids fold mod 64 — a reserved stmt is new, so Admitted regardless]
prior = intern.id("blw5:prior"); band_k = intern.id(&format!("blw5:band:{k:02}")) for k in 0..16
mass = payload.mass() as f32
for k: arena.observe(CStmt{s: prior, cop: Inh, p: band_k}, TruthValue::new(shape[k] as f32 / mass, 0.9), RESERVED_STAMP)
typ = shape[rank] as f32 / mass                     // the prior's mass at the observed rank
for s in subjects: arena.observe(CStmt{s, cop: Inh, p: prior}, TruthValue::new(typ, 0.9), RESERVED_STAMP)
reserved = {prior} ∪ {band_k}
```
then `reason(&mut arena)`. The 16 + n_subj statements are the verbatim-injected set.

## 6. Cycle plan (design (d)) — the SoA loop supplies the sealed versions

Fleet, `MemWal`, `BatchWriter`, `scan_board`, `column_pass(plan_or_evaluate_think)`,
`run_cognitive_work_gated_over` (closure: NO lens; returns the same
`(qualia, mantissa, reliability, row_span_payload)` tuple as d_ign_b) and `run_cycle` exactly
as d_ign_b's loop. Minds live in `HashMap<MailboxId, Mind>` beside the fleet.
- c=1: build every mind (`build_mind`) BEFORE the casts; casts; `run_cycle` → **V0 =
  sink.head()**; then the V0 measurement (§5 + S(cohort,V0) for all six cohorts + N's bloom
  verdicts) via the `on_sealed` callback.
- c=2: inject per cohort (T/FP/FM/N with their payloads, P with the uniform payload, CTRL none),
  `reason()` on EVERY mind (CTRL included); casts; `run_cycle` → **V1 = sink.head()**;
  `on_sealed` → S(cohort,V1), N bloom verdicts at V1.
The loop lives in `fn run_loop(corpus, minds, on_sealed: &mut dyn FnMut(u32 /*cycle*/, DatasetVersion, &HashMap<MailboxId,Mind>, &Fleet))`
defined BEFORE the measurement marker and it never reads what `on_sealed` computes (O6).
S0 is computed exactly once and stored; it is NEVER recomputed at V1 (single-measurement law).

N's awareness-free criterion (copy `score_row`/`rank_verdicts` shape from
`blw_fusion.rs:314-375`, cite): seed plane = `encode_plane(<owner's verse 0 text>, salt)`;
`score(row) = popcount(content_row(row) & seed)`; verdict = score ≥ the 75th-percentile score
over the owner's 48 rows (q = 0.25, the D-BLW-3 pin); `Vec<bool>` per N owner at V0 and V1.

## 7. Gates (all pre-registered, each twinned; movement floor 0.10 on κ, the D-BLW-3 pin)

Write a marker line `// ── MEASUREMENT BLOCK (O6 marker) ──` ; readers, `measure_cohort`,
the ledger and the gates are defined AFTER it; `run_loop` and everything it calls BEFORE it.
- **O1 remeasure guard** — can-fire: re-`seal` at T's V0 key → `Err(RemeasureError::AlreadySealed{..})`;
  can-stay-silent: `seal` at `(1, 1, T-lo, 1, v1)` with `nb.shape_rank(obs_T, v1.0)`… NO — that
  would be a remeasure of S0's input. Use a payload built with version v1 from the SAME
  `payload_true.shape/rank` (`ShapeRankPayload::new(shape, rank, v1.0)`) → `Ok`; and arm 2 at
  `(1, 2, T-lo, 1, v0)` → `Ok`.
- **O2 placebo** — assert `|Δκ(P)| < 0.10` (κ at both versions must be Some, else the arm is
  DEGENERATE and the assert is replaced by the printed named result); twin: every P mind has
  `arena.get(prior Inh band_00).is_some()`.
- **O3 null instrument** — assert N bloom verdicts byte-identical V0→V1 per owner; print the
  Hamming (expected 0 — rows unchanged — and SAY so: "frozen by construction; pool drift 0").
- **O4 the observable** — PRINT `Δκ(T) = κ1 − κ0`; `fires = |Δκ(T)| >= 0.10`. BOTH outcomes are
  named results, NOT asserted: print "O4 FIRES — awareness reflects the statistic" or
  "O4 SILENT — the honest null: awareness does not reflect this statistic (at floor 0.10)".
  If κ1 is None while κ0 is Some: print "O4 SATURATED — reader B degenerate at V1" (named).
- **O5 direction** — with `d = Δκ(FP) − Δκ(FM)` (both arms eligible and non-degenerate):
  `d >= 0.10` → "ANCHORING (testimony-dominance, Goodhart realised)"; `d <= −0.10` →
  "EVIDENCE-DOMINANCE"; both |Δκ| ≥ 0.10 with |d| < 0.10 → "PERTURBATION (value-invariant)";
  else "SILENT". Print; assert only that a classification string was produced.
- **O6 firewall self-scan** — `const SRC: &str = include_str!("d_blw_5_observer.rs");` split at
  the marker; assert the pre-marker half contains none of `"fn reader_a"`, `"fn reader_b"`,
  `"binary_association("`, `"kappa"` (self-match guard: skip lines containing `"O6"`); assert
  the post-marker half contains the marker AND `"binary_association("` (a scan that finds
  nothing is not evidence).
- **O7 exclusion is load-bearing** — on the T cohort at V1: `S_shadow` (B_shadow) ≠ `S` (compare
  the n00/n01/n10/n11 tuples) can-fire; and B's positive count > 0 can-stay-silent.
- **CTRL** — assert `Δκ(CTRL) == 0.0` exactly (pass idempotence) AND print reader Hamming 0.
- **DROP (design (e))** — per injected arm: `|Δκ| < 0.01` AND reader-B Hamming V0→V1 == 0 →
  print "DROP fires for <arm>". Printed, not asserted.
- **Preconditions (panic = corpus defect, not finding):** CTRL A-rate and B-rate ∈ (0,1) at V0
  (else the B′ fallback line applies to B; A degenerate → panic); pool ≥ 16.

## 8. Output discipline

Print every association table in full (C2), every payload as `shape=[..] rank=r
prozentrang=.. version=v`, the pool size and ladder boundaries, each arm's typicality
`typ`, and end with a "== D-BLW-5 — what this test does NOT claim ==" block: no validity of
the observer effect beyond this corpus/instrument; no parallelism; no durability; no fusion
verdict; no per-stance dispatch; jc untouched and one-way; the rank enters as typicality
(C2 above) so equal-mass ranks are indistinguishable; N frozen BY CONSTRUCTION (rows do not
change), its pool-drift duty reads 0 here; the synthetic corpus is symmetric by design and
the V0 reader rates are what the generator makes them.

## 9. Worker deliverables

1. `crates/lance-graph-supervisor/tests/d_blw_5_observer.rs` (new).
2. `crates/lance-graph-supervisor/Cargo.toml` (one dev-dep line + comment).
3. Tag-file `.claude/board/exec-runs/d-blw-5-build-sonnet.md`: files touched, every signature
   read (file:line), what could not be verified ("not compiled, not run — orchestrator gates").
No cargo. No other file. No board file.

## 10. Addendum after the dry runs (2026-09-05, main thread) — instrument fixes, NOT threshold changes

Every floor, C_MIN, band and shift above is unchanged. Three instrument defects were found by
dry runs 1–2 and fixed BEFORE the recorded run; the numbers of the dry runs are in the board
entry, not hidden:

- **`reason()` is now a bounded RCR+closure fixed point.** One round is not idempotent (the
  derived layer feeds RCR new premises); CTRL moved by Δκ = −0.0028 in dry run 1 with no
  injection. A pass that runs at V0 and V1 must be a fixed point or V1−V0 measures the pass.
- **The corpus counts are a splitmix64 fold of the window index**, not `w % {2,3,4}` (period
  12 → the 40 owner φ values were 12 atoms and the prior had empty buckets between them; T's
  pooled statistic landed in one, typicality 0).
- **C2 amended: typicality rides in the CONFIDENCE of `subject Inh prior`, frequency 1.** With
  f = typicality (< 0.5) the arena reads a confident NEGATION and `admit_derived`'s
  expectation-CHOICE replaced it with a vacuous closure path (c = 3.5e-11, expectation ≈ 0.5
  beats 0.462). Measured in dry run 2. This is an arena property worth its own board entry.
- **O7 restated to what can fire.** Under RCR-only reasoning no reserved-SUBJECT belief ever
  acquires a corpus predicate, so `B_shadow ≡ B` and the pinned twin could not fire. Restated:
  can-fire = the T arena at V1 holds derived beliefs carrying a reserved term (the payload DID
  propagate); can-stay-silent = none of the beliefs reader B accepted at V1 carries a reserved
  term (the firewall holds by the reader's shape).
