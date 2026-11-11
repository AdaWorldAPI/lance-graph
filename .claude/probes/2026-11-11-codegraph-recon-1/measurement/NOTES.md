# Grading notes (D-CODEGRAPH-RECON-1, blind)

Source checked: /home/user/lance-graph working tree (HEAD 40c138ca1), read only.

## Per-file totals

| file | correct | partial | wrong | coverage-honest |
|---|---|---|---|---|
| X30 | 7 | 3 | 0 | 10/10 |
| X44 | 7 | 3 | 0 | 10/10 |
| X50 | 8 | 2 | 0 | 10/10 |
| X54 | 6 | 4 | 0 | 10/10 |
| X66 | 8 | 2 | 0 | 10/10 |
| X68 | 7 | 3 | 0 | 10/10 |
| X84 | 9 | 1 | 0 | 10/10 |
| X93 | 7 | 3 | 0 | 10/10 |

## How the rules were applied

- WRONG was kept for a false member of the answer set (a non-caller listed as a caller, a non-dependent, a non-impl), a false count of the answer set, or "none" where items exist. No cell met that bar.
- A peripheral false statement was marked PARTIAL and named in the reason. Examples: dependency kind (X30 Q2, X66 Q7, X68 Q7), a test/lib bucket (X54 Q1, X54 Q4), a supporting count (X30 Q1, X44 Q9), a workspace-membership aside (X93 Q2), a tool-output claim (X84 Q9).
- Line-number citation slips were noted and not penalised. Examples: visible_fields :184 (actual 193), grammar when.rs:424 (actual 421), RouteLog::check placed at lib.rs:1045 (actual dr.rs:268).
- Q1, Q4: the key's full-credit rule accepts a scoped count. A compiler-only answer that gives the RA 13 impls / the RA callers and states that excluded crates and cfg(test) code were unobserved was graded CORRECT. A text-search answer that gives the full 35-site list with a lib/test split was also graded CORRECT.
- Q8: CORRECT needs compiler-resolved counts with their scope, plus a code / test / doc split. Text-only counts are true in every file, but those answers were graded PARTIAL. Of the RA-count answers, only X84 also gives the text superset with its doc caveat.
- Q10: no answer reproduces the corpus row counts (24 GAP / 61 INTERPRETATION). Full credit was given for naming the gap nodes (PERMIT, ONE, G2A, G2B, ISS, JOIN), the solid SD --> walk_rails_authorized edge as not a call, and the status/intent claims.

## Key gaps (true facts in answers that the key does not cover)

1. `BindStore::register_derived` (dt64 lib.rs:826) also calls `insert` (:840), so `register` is not the only entry (X50, X54).
2. `lance_graph_xlog::bag_count_admitted` (xlog lib.rs:384) takes `&MethodBinding` and never calls `admit`. Its admit happens in the caller, xlog-relcomp-probe/tests/dto_binding.rs:244.
3. Q2 dependency kinds: lance-graph-dir-sim (:51) and math-polyfill-probe (:30) are `[dev-dependencies]`; the other 14 are normal. Five `.claude/probes` manifests also depend on dt64.
4. Q3 transitive callers through `lane` / `borrow_all`: foundry-parity-lab arm_b.rs:155, arm_c.rs:163, method.rs:229; soa64-u64-lab socket.rs:206/207/227; dr-isa-stream-lab gemm/int8/shape/cube.
5. Q9: seven `LaneSource` impls exist in source (dt64 tests lane_socket.rs:91, foundry source.rs:178, soa64 socket.rs:136, dr-isa source.rs:225 and int8.rs:159, report lane_source.rs:143, dir-sim ordinal_chain_lab.rs:1109). RA sees two.
6. Q7/Q10: `walk_rails_authorized` (contract selection.rs:335) has exactly one caller, its own test at selection.rs:689.
7. Q1: lance-graph-codegraph publication.rs:65 `PubView` is library code, not test code. Two further impls sit in .claude/probes (main.rs:72, v2_compose.rs:90). `AiwarClassView` has only an inherent impl.
8. Q4: r2il tests hold 7 `.method_ref(` call sites in 4 files (binding.rs 312/680/721/864, binding_native.rs 313, polyfill.rs 182, polyfill_views.rs 208). The key's "4 r2il test callers" counts the helper functions RA resolves. dt64-admission-probe/src/fixture.rs:221 is a free fn named `method_ref`, and its ~15 test call sites are not direct BindStore::method_ref calls.
9. Q8 text counts (crates/, whole word, includes comments): WideFieldMask 505 lines in 49 files across 23 crates, 430 lines in 42 files excluding `//` lines. MethodBinding 101 lines in 22 files across 11 crates, 79 in 17 excluding `//`. Including .claude, the counts are 513/51 and 116/24.
10. ogar-behavior-probe declares the dt64 dependency but no .rs file names `lance_graph_dt64`.

## Key errors

1. **Q4, admit callers.** The key lists "r2il integration tests (RA: polyfill_views::prep, polyfill::mref, binding_native::mref, binding::mref)" as callers of `admit`. Those four helpers call `method_ref`, not `admit`:
   - crates/lance-graph-r2il/tests/integration/binding.rs:311-312 `fn mref` → `s.method_ref(...)`
   - binding_native.rs:312-313
   - polyfill.rs:181-182
   - polyfill_views.rs:200,208
   The direct `admit` call sites in r2il tests are binding.rs:536 and :641. The key repeats the same four helpers correctly under method_ref, which suggests a copy error. No answer was penalised for it: the compiler-only answers (X50, X68) list admit <- prepare and treat the helpers as transitive callers, which is correct.
2. **Q6, "WHEN evaluation exists only in grammar-constraints-probe (lab)".** This is false. crates/foundry-parity-lab/src/method.rs:217-280 (`PreparedMethod::prepare`) admits the binding, takes `binding.when` (:218), renames `guard.expr` onto bound planes, and lowers it with `rel_from_set` + `run_with_program` into one mask-risc `Program`. It refuses with `Symbolic` or `Folds` otherwise, and the module doc (:9-11) says so. The key's main claim stands: no WHEN guard lowers into R2IL, and binding.rs:335-337 refuses it. Answers that cite foundry lowering were not penalised.

Imprecision, not counted as an error: Q7 says lance-graph-dir-sim "depends on both". dir-sim has no direct lance-graph-rbac dependency. It names it only in `[patch."https://github.com/AdaWorldAPI/lance-graph"]` (Cargo.toml:76-78), which redirects the transitive dependency of the optional OGAR crate `ogar-rbac` (:44, feature `rbac`). The statement holds transitively. Answers that say "patch only, not a dependency" are also true.
