# X2-DT64-ADMISSION: a typed execution-admission policy after `BindStore::admit`

**Date 2026-10-31 · Plan `d-dynbag-holographic-bags-v1` §18.4 (X2, as briefed) · Status LAB · Crate `crates/dt64-admission-probe` (excluded, own workspace) · Prior [D-SAG-6-2-LAB](2026-10-31-d-sag-6-2-lab-projection.md), [D-SAG-6-2-JOIN-IDENT](2026-10-31-d-sag-6-2-span-join-identity.md)**

**Decision (operator):** `BindStore::admit` stays as it is. It authenticates method identity, signature revision and guard witness. Admission for execution is a separate typed policy, layered after it.

## Shape

`ExecutionPolicy::decide(store, ref, evidence, request)` first runs `admit`; a refusal there passes through as `Binding(..)`. It then classifies the method's D-SAG-6 rows:

| # | Evidence | Result |
|---|---|---|
| 1 | proven pure | `Admitted::Pure { assumed }` |
| 2 | WITNESSED effect, every effect row covered by a `DoGrant` for this exact `MethodRef` | `Admitted::UnderDoGrant` |
| 3 | any POSSIBLE row | `ExecRefusal::Possible` |
| 4 | any UNKNOWN fact | `ExecRefusal::Unknown { causes }` |
| 5 | WITNESSED effect, no covering grant | `RequiresDoGate` / `GrantDoesNotCover` / `GrantForOtherMethod` |

Precedence: UNKNOWN, then POSSIBLE, then a witnessed effect. A grant lifts only witnessed effects.

What the types add:
- `Attachment` keys evidence to `(class, MethodId, SigRev)`, never to `MethodBinding::name`.
- `DoGrant` names one `MethodRef` and the `(relation, subject)` effect rows it accepts.
- `UnknownCause` has six arms: `NotAttached`, `NoDefinition`, `Unjoined`, `UnnamedCallee`, `UnauditedCallee`, `Mislabelled`.

Evidence types:
- `Relation` and `Evidence` mirror `project_rows.py`'s strings, crate-local only.
- No global evidence enum was added and `set_rules::Truth` is not reused.
- No DT64 type was needed: no change to `lance-graph-dt64/src`, no new classid. The lab classids of DT64's own tests are reused.

## Measured (fixture `sag_fixture`)

`cargo test`: **25 pass, 0 fail.**

| Method | Result |
|---|---|
| `count_below` | `Pure`, `assumed` = 3 callees |
| `count_with_tally` | `RequiresDoGate`: 1 × `WRITES_CALLER_MEMORY` |
| `count_div` | `RequiresDoGate`: 1 × `MAY_PANIC` |
| `count_generic` / `count_dyn` | `Possible`, also when a grant names the row |
| `mask_below` | `Possible` |
| `Below::test` | `Unknown` (`Unjoined`) under the path join, `Pure` under the span join |
| a method bound under the name `count_below` with no attachment | `Unknown` (`NotAttached`) |

All 8 bound methods pass `admit`; 2 execute as pure. Binding is not authorization.

**Disable runs: 14 guards, 14 red.** Each run was committed first, then the patch was asserted to have applied, then restored with `git checkout`. The table is in the crate README. The `admit` pass-through was not disabled (no smaller patch than deleting it); it is pinned by a test.

## Open

1. **The rows are reconstructed, not measured.** No `project_rows.py` output is committed. `src/fixture.rs` rebuilds the rows bearing on these functions from the two entries' stated facts, citing each group. The `CALLS` objects are illustrative `core::` spellings. The 8 `MAY_PANIC` rows (profile-dependent) are not all listed. Next step: run Charon → `llbc_calls.py` → `span_join.py` → `project_rows.py` on the fixture and commit the TSV as the fixture.
2. **The trust list is policy, not evidence.** A pure verdict on `count_below` relies on `core::slice::` / `core::iter::` being pure; the verdict returns them in `assumed`. Without the list the answer is UNKNOWN. Who owns the list is undecided.
3. **Scope is by subject prefix.** Closures are found through `subject::`. A closure passed into a *non-local* callee is not in scope unless it is nested under the subject; the span join's closure spelling makes it nested on this fixture. Not tested on closures defined elsewhere and passed in.
4. **`DoGrant` has no issuer.** Who may mint one (RBAC, an operator approval) is not modelled. The policy only checks a grant.
5. **§18.4 was not found in the plan.** The plan file on `origin/main` (d3c2a29) ends at §16. The citation is as given in the brief.
