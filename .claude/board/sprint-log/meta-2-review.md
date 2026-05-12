# Meta-2 Review — medcare-realtime (Round 2, Stage 2)

**Reviewer:** Meta agent 2 of 3 (Round 2 review pass)
**Scope:** medcare-rs/crates/medcare-realtime + workspace registration
**Method:** read W5-W8 commits + log entries; cross-check against
smb-realtime shape parity, upstream dep availability, topology I-1/I-2,
and Round 3 readiness.

> **Tone:** brutally honest. Reviewing my own colleague's work as if
> shipping to production tomorrow. Findings escalate by severity.

---

## Verdict

**Ship Round 2 with one CRITICAL fix applied as W7-revision-2 before
opening Round 3.** The fail-loud choice on `StepDomain::MedCare` may
block compilation if the variant doesn't exist upstream. One MEDIUM
finding flagged for follow-up. Otherwise Round 2 is clean.

| # | Severity | Finding | Action |
|---|---|---|---|
| 1 | **CRITICAL** | W7 hard-depends on `StepDomain::MedCare` variant; sprint cannot verify upstream existence | W7-revision-2: construct DomainProfile inline with documented expected values; switch to `StepDomain::MedCare.profile()` when upstream variant ships |
| 2 | MEDIUM | `MedCareStack` empty struct in v1 — is this a facade or a marker? | Honest doc note in module head; defer field growth to follow-up |
| 3 | MEDIUM | No `with_default_policies()` builder — smb-realtime has 3 default policies registered; medcare-rs has zero | Backlog: add when canonical entity list is firm |
| 4 | LOW | No test asserts cross-crate workspace dep resolves (medcare-realtime → medcare-rbac) | CI catches this on first build; no Sprint action |
| 5 | LOW | Cargo.toml has 0 `[dev-dependencies]` while smb-realtime has tokio-test + pretty_assertions | Defer; v1 tests are simple `assert!` on sync surface |

---

## CRITICAL #1 — `StepDomain::MedCare` may not exist upstream

**Finding.** W7's `domain_profile()` calls
`lance_graph_contract::orchestration::StepDomain::MedCare.profile()`.
W7's self-review explicitly chose "fail loud" — if the variant is
absent, the file won't compile.

**Why this is critical.** The sprint goal is "produce a buildable
3-stage scaffolding ready for Round 3 + future merge". A non-compiling
medcare-realtime blocks Round 3 (W9 imports from medcare-realtime),
blocks workspace `cargo build`, and blocks any CI run on this branch.

**Super-helpful solution.** Fetch `lance-graph-contract/src/orchestration.rs`
to verify variant existence and profile field shape, then commit
W7-revision-2. If variant exists, no revision needed (though doc
strengthening per #2 still recommended). If absent, hand-construct
DomainProfile inline with documented expected values.

---

## MEDIUM #2 — MedCareStack empty struct: facade or marker?

Empty-struct-with-methods is fine for a marker type that locks in the
API surface for future field growth. But the doc comments call it a
"facade" — language that implies composition.

**Solution.** Tighten the doc comment to acknowledge the v1 marker
status explicitly. v1 trades emptiness for symbol stability — worth
saying so explicitly. No code change; doc-only update lands as part
of any future field-growth commit.

---

## MEDIUM #3 — Missing default policies builder

smb-realtime ships `with_default_policies()` registering Customer /
Invoice / TaxDeclaration. medcare-realtime has no equivalent. Once
`MedCareStack` grows an `rls_registry` field, it'll need a parallel
`with_default_medcare_policies()` registering Patient, Diagnosis,
LabResult, Prescription, Anamnese, Ueberweisung.

**Action.** Backlog. v1 doesn't have rls_registry yet.

---

## LOW #4-#5 — Defer

- #4: cross-crate dep resolution test → CI catches naturally
- #5: `[dev-dependencies]` → trivial; v1 has no async tests yet

---

## Round 3 implications

W9 (gate.rs) imports + W10 (lib.rs re-exports) follow standard pattern.
W11 (integration tests) wraps `MedCareStack::new()` and verifies
`MedCareMembraneGate` composes with it.

W12 (§73 SGB V test) verifies:
1. Doctor without Ueberweisung row CANNOT read another Doctor's
   Patient at Detail (row-level check happens above the gate)
2. Doctor with active Ueberweisung CAN read referred Patient at Detail
3. **BtM-flagged Prescription.issue → Escalate** (per Meta-1 #3
   carry-forward; gate.rs in W9 must implement this wrapping)

The Meta-1 HIGH #3 + #4 carry-forward (BtM Escalate, anonymize/merge
Escalate) lands in W9 logic. W12 tests verify.

---

## Feedback loop — apply NOW (W7-revision-2)

Fetch `lance-graph-contract/src/orchestration.rs` to confirm DomainProfile
field names + EscalationStrategy + VerbTaxonomyId variants, THEN commit
W7-revision-2 with the correct inline construction. Five minutes of
verification, much better than guessing.
