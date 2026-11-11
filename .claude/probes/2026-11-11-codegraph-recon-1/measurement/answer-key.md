# D-CODEGRAPH-RECON-1 held-out answer key

Revision: lance-graph working tree = origin/main 9f4563b for crates/ (plus the cg_query example only).
Built by the orchestrator from source reads, the Charon store (gateB) and the rust-analyzer GRAPH run,
each fact re-read in source. Coverage of each tool is stated, because several answers depend on scope.

Scoring per question: CORRECT (all key items, no false items), PARTIAL (subset, no false items),
WRONG (any false item stated as fact), plus COVERAGE-HONEST yes/no (did the answer say what it did not see).

## Q1 Where is ClassView defined, and what implements it?
- Definition: `crates/lance-graph-contract/src/class_view.rs:1057` `pub trait ClassView`.
- Implementations: 35 `impl … ClassView for …` sites in 30 files (syntactic, every crate dir incl. excluded
  crates, tests, examples). rust-analyzer resolves 13 of them in 9 files of loaded workspace code; the other 21
  files are excluded crates, examples, cfg(test) unit modules, or test modules inside class_view.rs.
- Full credit needs the definition, an impl count with its scope, and the note that excluded crates / test
  modules are outside a compiler index.

## Q2 Which crates depend on DT64?
- 16 crates declare `lance-graph-dt64 =` in Cargo.toml: cognitive-polyfill-probe, dr-isa-stream-lab,
  drisa-register-probe, dt64-admission-probe, foundry-parity-lab, grammar-constraints-probe,
  lance-graph-codegraph, lance-graph-dir-sim, lance-graph-dt64-lance, lance-graph-r2il, lance-graph-report,
  lance-graph-xlog, math-polyfill-probe, ogar-behavior-probe, soa64-u64-lab, xlog-relcomp-probe.
- NOT a dependent: lance-graph-quack (its `.admit(` calls are its own test-only typed code space).

## Q3 Who uses LaneBinding::check, including tests?
- In-crate: `lane.rs:360` (`LaneBinding::lane`), `lane.rs:386` (`LaneBinding::borrow_all`). CONFIRMED by Charon and RA.
- Tests: `crates/lance-graph-dt64/tests/lane_socket.rs` 227, 277, 314, 373, 403, 406(-407), 423(-424) — all inside
  macro arguments; RA emits no call row for any of them (it marks the macro region); Charon (lib target) does
  not see test targets at all.
- `crates/lance-graph-report/src/lane_source/tests.rs` 320, 341, 418 (+1 in the same file) — cfg(test) unit module,
  not indexed by RA, not in the Charon lib artifact.
- `crates/lance-graph-dir-sim/tests/ordinal_chain_lab.rs:1328` — lab test.
- False friends: `dir-rmap-probe` `.check(` calls and dt64 `tests/dr.rs` `g.check` / `store.check` are other types.

## Q4 What is reachable from BindStore::method_ref / BindStore::admit?
- admit: callees `check_issuer`, `Guard::witness`, `GuardWitness::eq` (CONFIRMED, Charon). Callers observed:
  `lance_graph_r2il::binding::prepare` (lib), plus r2il integration tests (RA: polyfill_views::prep,
  polyfill::mref, binding_native::mref, binding::mref) and ~140 call sites in ~20 other files (tests, probes,
  labs) visible only to source search.
- method_ref: no lib caller in the four compiled crates; 4 r2il test callers (RA); the rest in tests/probes.
- Full credit needs "callers beyond the loaded crates are unobserved", not "none".

## Q5 How does a MethodBinding get from DT64 to R2IL preparation?
- `lance_graph_r2il::binding::prepare` (`crates/lance-graph-r2il/src/binding.rs:321`): guards.check → check_route →
  store.resolve → `store.admit(r.method)` (line 334) returns `&MethodBinding` → refusals on `when` (336) and
  three-valued truth (339) → lens check (342) → binds `binding.effects` into `Prepared.steps`.
- `prepare_native` (binding_native.rs) takes a `Prepared`, so it inherits every check above.

## Q6 Which WHEN guards lower, and where does execution refuse them?
- None lower into R2IL. `prepare` returns `ExecRefusal::GuardNotExecutable` when `binding.when.is_some()`
  (`binding.rs:336`). WHEN evaluation exists only in grammar-constraints-probe (lab), not on the R2IL path.

## Q7 Can a ScopedDecision reach the DT64 lane path?
- No compiled path. `ScopedDecision` is defined in `lance-graph-rbac/src/authorize.rs:170`. dt64, quack, r2il and
  report do not depend on lance-graph-rbac. lance-graph-dir-sim depends on both but uses LaneBinding only in a
  lab test and never consumes a ScopedDecision. `walk_rails_authorized` takes a caller-supplied `allowed`
  closure, not a ScopedDecision. This is a MISSING integration (corpus R326, R417, R422, R430–R435), never an edge.

## Q8 What changes if WideFieldMask or MethodBinding changes?
- WideFieldMask: RA resolves 259 references in 4 crates (contract 179, dt64 52, rbac 17, r2il 11) of loaded,
  non-test code; source search adds cfg(test) modules (e.g. report lane_source/tests.rs, 10 mentions), excluded
  crates and doc-only mentions (lance-graph soa_config.rs, lance-graph-ogar recipe_vocab.rs — docs, not code).
- MethodBinding: RA resolves 46 references in 3 crates (dt64 27, r2il 17, xlog 2).
- Full credit needs both counts with their scope and the split between code, test-only and doc-only mentions.

## Q9 For LaneBinding::check: what is known, possible, unknown?
- Known: 2 in-crate callers; callees admit_lane, source_owns, current_route, FrameStore::check,
  BindStore::resolve_counted, LaneAddr::eq.
- Possible: trait-dispatched calls on its generic `V: ClassView` / `S: LaneSource` (Charon MAY_DISPATCH:
  `LaneSource::borrow`, `LaneSource::lane_of`, `LaneSource::frames`, `ClassView::fields_at` …).
- Unknown: callers in other crates and in test targets (it is PUBLIC); calls inside macro arguments;
  cfg(test) unit modules; excluded crates.

## Q10 Which Mermaid edges in 03/05/06 cannot be recovered from compiler facts?
- The 24 GAP rows (corpus R261–R265, R276, R322, R323, R326, R426–R438, R442, R444): ExecPermit issuer, one
  executor for WHEN and DO, per-row scope lane, visible_fields limiting bound fields, one record per answer.
- The drawn solid edge `ScopedDecision --> walk_rails_authorized` (R417/R422): no call site exists.
- The 61 INTERPRETATION rows (status/plan claims, "only way" claims) are not compiler facts either.

## Corrections after grading (verified in source by the orchestrator)
- Q4: the four r2il integration-test callers (polyfill_views::prep, polyfill::mref, binding_native::mref,
  binding::mref) call `method_ref`, not `admit`. Direct `admit` calls in r2il tests: binding.rs:536, :641.
- Q6: WHEN guards DO lower outside R2IL in the lab: foundry-parity-lab/src/method.rs:218-266 reads
  `binding.when` and lowers it via `rel_from_set` + `run_with_program` into a mask-RISC program (LAB).
  The key's main claim stands: none lowers into R2IL; `prepare` refuses at binding.rs:336.
