# D-DRDT-CLASS-GOVERNANCE-1 — Falsifiers

Status: census, docs only. Read at `main` `238b6d1e4` (2026-11-11).

The brief lists 25 falsifiers. This table says, for each, whether a test on
main already covers it. "Test" names are functions in
`crates/lance-graph-dt64/tests/bind_store.rs` unless another file is given.
They were located by name; the orchestrator read the lens tests at
`:1468-1560` and the stamp code, not every test body. A row marked COVERED
therefore means "a test with this intent exists", and its body still has to
be read before the row is cited as proof.

Legend: COVERED / PARTIAL / MISSING / NEEDS-DECISION (cannot be written
until a council decision fixes what the right answer is).

| # | falsifier (brief §12) | today | existing test or gap | oracle, negative control, failure mode for the missing part |
|---|---|---|---|---|
| 1 | Same right `FTMM` resolved in the global polyfill and in two class contexts | PARTIAL | `two_behaviour_lenses_under_one_left_half_resolve_independently`, `a_data_class_and_a_behaviour_lens_sharing_a_left_half_do_not_cross_read`. No test has ONE right half under THREE left halves | Oracle: three hand-written bases of different lengths. Control: swap two left halves, the bases must swap. Failure: two contexts return the same basis |
| 2 | Without a class context the global or local technology meaning applies | NEEDS-DECISION | No code path resolves a bare right half. Every lookup takes a full `u32` | Needs crosswalk X2 decided first |
| 3 | `WHERE` inside Quack with no external classid resolves in the Quack context | NEEDS-DECISION | Quack lowers a `Query` to a program with no classid at all (BYPASS-MATRIX) | Same as 2 |
| 4 | An explicit classid overrides that context only through registration | NEEDS-DECISION | depends on 2 and 3 | — |
| 5 | Unregistered technology/method code is refused | COVERED | `an_unknown_behaviour_lens_fails_closed_instead_of_inheriting_its_concept`, `method_ref_refuses_what_is_not_bound` | — |
| 6 | A right-reserved `F` address without a valid `MethodBinding` is refused | PARTIAL | Refused at `resolve`/`admit`. Not refused at `execute_into`, which takes no address | Oracle: a census of call sites (see plan, cut 4). Failure: a new direct call site compiles and runs |
| 7 | A left-reserved `F` address is not read as an ordinary classid | PARTIAL | OGAR guard test for `0xFFFF` (`ogar_codebook.rs:926-931`). Nothing covers left `0xF000..=0xFFFE`, which the D-DT64-7 plan lists as unallocated (O1) | Oracle: `canonical_concept_name(x)` is `None` for every `x` in `0xF000..=0xFFFF`. Control: a known concept still resolves |
| 8 | Two `MethodId`s must not collide through an 8-bit address code | NEEDS-DECISION | No 8-bit method code exists. `MethodId` is `u32`; `AlgebraRef.word` is `u8` and is a different thing | Needs crosswalk X1/X4 |
| 9 | Same method at another schema revision is re-admitted or refused | COVERED | `a_class_is_never_resolved_at_another_revision_even_when_history_has_it`, `schema_revision_and_field_basis_are_checked` | — |
| 10 | Legacy classids keep their historical reading | COVERED | `classid_canon_compat` tests in `ogar_codebook.rs`; plan item D-DT64-7-1 is still Queued for the `0x0000_F123` case | Write D-DT64-7-1 |
| 11 | Same field count in another order is detected | COVERED | `an_equal_length_reorder_is_refused`, `a_same_count_reorder_across_revisions_replays_only_through_history` | — |
| 12 | A changed WHEN guard is detected | COVERED | `a_guard_added_to_an_unguarded_method_refuses`, `same_read_write_sets_different_effect_order_stay_distinct_and_a_dropped_guard_refuses` | — |
| 13 | A changed DO effect order gives another binding contract | COVERED | `a_reference_to_reversed_effects_is_refused_as_another_binding` | — |
| 14 | An invalid ClassView basis does not fall back to the concept basis | COVERED | `an_unknown_behaviour_lens_fails_closed_...`, `a_lens_classid_cannot_register_through_its_concepts_basis` | — |
| 15 | A foreign `BindStore` issuer is refused | COVERED | `a_same_binding_from_another_store_is_refused_as_foreign`, `a_cloned_store_refuses_the_originals_reference` | Note: the same mechanism also refuses a legitimate replay in another process (SNAPSHOT-VERSIONING) |
| 16 | IAM refuses an otherwise valid method | MISSING | `admit` has no authorization input (METHOD-RESOLUTION). No test joins RBAC and admission | Oracle: a role table written by hand. Control: the same call with the role granted succeeds. Failure: execution without the grant |
| 17 | Replay with an old schema/method binding gives the old effect or a clear refusal | PARTIAL | Field-basis history is covered (`a_history_keeping_source_replays_each_revision_against_its_own_basis`). The store itself is not persisted, so no replay crosses a process | Oracle: effect recorded at version N. Control: a changed binding at N+1 must not change the replay of N |
| 18 | A failed or ambiguous seal produces no silently valid selection | UNKNOWN | belongs to `lance-graph-dt64-lance` and the seal work of PR 1993; not read by the orchestrator | Read `crates/lance-graph-dt64-lance/tests` first |
| 19 | A G2L reorder invalidates prepared operands | PARTIAL | `LaneBinding::check` refuses a non-current frame (`lane.rs`). In the lab on PR 2010, `a_stale_frame_is_refused_and_a_fresh_preparation_answers`. Nothing on main for prepared programs | — |
| 20 | Different source/destination coordinates are not gathered implicitly | PARTIAL | `a_plane_from_another_row_order_is_refused` (lab, PR 2010). Admission route R3 is not admitted (EXO-121) | — |
| 21 | A 37-property dimension stays physically under 64 | COVERED | `a_consumer_reserves_its_extent_and_is_not_capped_at_64` covers the reservation side; `Axis::for_survivors` picks DR64 at 64 or fewer | A direct test at exactly 37 is one line |
| 22 | A 200-method dimension is addressable without 256 business-value lanes | MISSING | No test registers 200 methods and checks what is materialized | Oracle: count lanes borrowed. Control: 65 methods must also not widen the property axis |
| 23 | 63 thought slots share one valid base selection and keep their own deltas | MISSING | `thought-git-probe` has `LIVE_SLOTS = 63` but no shared prepared selection (THOUGHT-REUSE) | see THOUGHT-REUSE |
| 24 | A changed classid, method, scope or snapshot invalidates reuse | PARTIAL | Each witness is checked separately on main; no test changes one of four and asserts the other three stay valid | A 4×4 matrix test: change one, expect exactly that refusal |
| 25 | Codegraph detects a synthetically introduced semantic bypass | MISSING | `lance-graph-codegraph` is an excluded crate; no test plants a direct `execute_into` call and expects a finding | Oracle: the planted call's file:line. Control: an admitted call must not be reported |

Counts: COVERED 9, PARTIAL 8, MISSING 4, NEEDS-DECISION 4 (2, 3, 4, 8), UNKNOWN 1 (18).

The four NEEDS-DECISION rows are the same question as crosswalk X1/X2: what a
right half means with no left context, and what the low byte is.
