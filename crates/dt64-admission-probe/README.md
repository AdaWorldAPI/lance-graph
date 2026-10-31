# dt64-admission-probe (LAB)

`BindStore::admit` authenticates a router's method reference: method
identity, signature revision, guard witness. It is unchanged here. This probe
adds a typed **execution-admission policy** after it:
`ExecutionPolicy::decide(store, ref, evidence, request)` runs `admit`, then
reads the method's D-SAG-6 evidence rows and decides whether the method may
execute.

Standalone: own `[workspace]`, excluded from the lance-graph workspace,
nothing depends on it. Depends only on `lance-graph-dt64` and
`lance-graph-contract`, by path, read-only.

```
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  cargo test --manifest-path crates/dt64-admission-probe/Cargo.toml
```

## The outcomes

| Evidence of the method's source | Result |
|---|---|
| 1. proven pure | `Ok(Admitted::Pure { binding, assumed })` |
| 2. WITNESSED effect, DO grant covers every effect row | `Ok(Admitted::UnderDoGrant { binding, effects })` |
| 3. any POSSIBLE row | `Err(ExecRefusal::Possible { rows })` |
| 4. any UNKNOWN fact | `Err(ExecRefusal::Unknown { causes })` |
| 5. WITNESSED effect, no covering grant | `Err(ExecRefusal::RequiresDoGate { effects })`, `GrantDoesNotCover`, `GrantForOtherMethod` |
| `admit` refused | `Err(ExecRefusal::Binding(refusal))`, unchanged |

UNKNOWN dominates POSSIBLE, which dominates a witnessed effect. A DO grant
authorizes witnessed effects only; it never lifts POSSIBLE or UNKNOWN.

Rules the tests pin:
- **Absence of rows is not purity.** Pure requires an attachment and a
  `DEFINES` row for the subject.
- **Evidence attaches to `(class, MethodId, SigRev)`**, never to the method's
  name. Evidence for another signature revision does not attach.
- **Scope** = the subject, nested subjects (`subject::…`, which covers the
  closures), and local callees (a `CALLS` target that has a `DEFINES` row).
- **A non-local callee is UNKNOWN** unless the policy trusts its prefix. A
  pure verdict returns the trusted callees it relied on (`assumed`).

The evidence vocabulary is `project_rows.py`'s (`WITNESSED` / `POSSIBLE` /
`UNKNOWN`), mirrored as a crate-local enum. It is not a global evidence type,
and `set_rules::Truth` (row logic) is not used as one.

## Fixture outcomes (D-SAG-6 `sag_fixture`)

| Method | Result |
|---|---|
| `count_below` | `Pure`, assuming 3 `core::slice::` / `core::iter::` callees; `Unknown` (3 × `UnauditedCallee`) with no trusted prefixes |
| `count_with_tally` | `RequiresDoGate` (1 × `WRITES_CALLER_MEMORY`); `UnderDoGrant` with a covering grant |
| `count_div` | `RequiresDoGate` (1 × `MAY_PANIC`); `UnderDoGrant` with a grant naming the panic |
| `count_generic`, `count_dyn` | `Possible` (`trait:Clause`, `dynamic`), also when a grant names the row |
| `mask_below` | `Possible` (`WRITES_THROUGH_LOCAL_REF`) |
| `Below::test` | `Unknown` under the path join (`UNJOINED`); `Pure` under the span join |
| a method named `count_below`, no attachment | `Unknown` (`NotAttached`) |

All eight methods pass `admit`; only two execute as pure.

The rows in `src/fixture.rs` are reconstructed from the facts in the D-SAG-6
board entries. Row lists are not committed anywhere; the entries record
counts and per-function findings. The provenance is cited per row group in
that file. They were not measured by a Charon run.

## Disable table

Method: commit, apply one patch (the anchor must occur exactly once, and the
script checks that the replacement applied), run the integration tests,
restore with `git checkout`, and check that the anchor is back. The runs were
against commit `b7cfe93`.

| Guard | Patch | Red tests |
|---|---|---|
| G1 `DEFINES` required | drop the `NoDefinition` push | `absence_of_rows_is_not_purity` |
| G2 UNKNOWN kept | UNKNOWN row → ignored | `below_test_is_unknown_under_the_path_join_…`, `unknown_dominates_possible_dominates_effect` |
| G3 POSSIBLE kept apart | POSSIBLE row → effect | `generic_and_dyn_…`, `a_grant_naming_the_dispatch_row_…`, `a_local_buffer_write_…`, `unknown_dominates_…` |
| G4 pure request refuses effects | `Request::Pure` → `Admitted::Pure` | `witnessed_effects_are_refused_…`, `every_fixture_method_admits_but_only_two_…` |
| G5 grant names this method | method check → `false` | `an_empty_grant_or_one_for_another_method_…` |
| G6 grant covers every row | coverage → `true` | `an_empty_grant_…`, `count_div_executes_only_under_a_grant_…` |
| G7 attachment keyed by `SigRev` | drop the `sig_rev` match | `evidence_for_another_signature_revision_…` |
| G8 local callees followed | no recursion | `an_effect_in_a_local_callee_is_found` |
| G9 nesting needs `::` | prefix without `::` | `a_nested_subject_is_in_scope_but_a_name_sharing_a_prefix_is_not` |
| G10 untrusted callee is UNKNOWN | trust every callee | `count_below_without_trust_is_unknown_not_pure` |
| G11 unnamed impl callee is UNKNOWN | `known impl` check → `false` | `an_unnamed_trait_impl_call_is_unknown` |
| G12 mislabelled hand-built row | ignore it | `a_hand_built_mislabelled_row_is_never_pure` |
| G13 parser checks relation/evidence | check → `false` | `a_possible_relation_labelled_witnessed_is_rejected` |
| G14 nested subjects in scope | exact subject only | 9 tests, among them `count_with_tally_executes_only_…`, `witnessed_effects_are_refused_…`, `generic_and_dyn_…` |

Not disabled: the `admit` pass-through. `decide` needs the binding `admit`
returns, so there is no smaller patch than removing the call. Its refusals are
pinned by `an_admit_refusal_passes_through_even_with_perfect_evidence`.
