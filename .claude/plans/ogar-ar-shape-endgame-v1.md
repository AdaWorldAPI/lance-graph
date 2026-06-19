# ogar-ar-shape-endgame-v1 — increments + gates for the OGAR ontology compiler

> **Companion to:** `docs/OGAR_AR_SHAPE_ENDGAME.md` (the doctrine).
> **Anchor epiphany:** `E-OGAR-AR-SHAPE-ENDGAME` (top of EPIPHANIES.md).
> **Council to run pre-merge:** 5+3 (panel + critique). Without it, the
> CONJECTURE→FINDING gates below are rubber stamps.
> **Status when filed:** PLAN (pre-council). Becomes PLAN-RATIFIED after the
> 5+3 verdict; each Inc becomes its own PR.
>
> **The spine:**
> _Curators teach. OGAR compiles. LanceGraph thinks. SurrealAST + Kanban
> orchestrate. Adapters obey._
>
> **The litmus failure:**
> _Same `OgarAst::Do(PostInvoice, …)` MUST execute semantically identically
> across NativeLance / SurrealAST / Odoo adapter / Rails adapter. If a backend
> leaks its syntax into the semantic result, that curator has started wearing
> the crown — that is the bug._

---

## 0. Why a plan, not just a doctrine

The doctrine in `docs/OGAR_AR_SHAPE_ENDGAME.md` names FIVE pieces that are
CONJECTURE today (§3 partial / §4 partial / §5 / §6 / §10). Without a
falsifying probe, "conjecture" is just synthesis — exactly the
measurement-before-synthesis trap `truth-architect` polices. This plan
sequences the probes that promote each conjecture to FINDING (or sharpen /
retract it). One Inc = one PR = one probe.

**What's already FINDING and stays out of this plan:** §1 (flat-triples
rejected), §2 (curators-not-foundations + ≥2-curator promotion rule), §7
(ractor compile-time + LanceGraph thinking + SurrealAST/Kanban
orchestration), §9 (callcenter as outer membrane). These are operator-
ratified and codify existing architecture.

**What this plan does NOT do:**
- Replace `cypher-kanban-ast-unification-v1`, `lite-unified-surrealql-lance-v1`,
  or `probe-excel-compute-dag-v1`. They each instantiate one slice of the
  endgame; this plan complements them.
- Touch any existing carrier (no `MailboxSoA` layout change, no
  `ENVELOPE_LAYOUT_VERSION` bump, no `NodeGuid` reshape).
- Cross-repo work in Inc 1-3 (in-workspace only; nexgen / woa-rs / surrealdb-
  fork are bounded to the curator-side at Inc 4-5 and stay read-only).

## 1. Increment ladder

The five Incs map 1-to-1 onto the five CONJECTURE rows in
`docs/OGAR_AR_SHAPE_ENDGAME.md` § 12. Each ships as its own PR with the
brutally-honest-tester gate.

### Inc 1 — `ClassView::policies` typed slot (§3 + §4 CONJECTURE → FINDING)

**Owns:** the THINK slot. Add `ClassView::policies: &'static [ThinkSpec]`
beside the existing `ClassView::compute_dag` and `ClassActions`. One
`ThinkSpec` is `{ predicate: &str, kind: ThinkKind, exec: ThinkFn }` where
`ThinkKind ∈ { GoBdImmutability, ActorAuthorized, StateGuardPolicy,
RubiconImpactCheck, Custom }` and `exec: fn(&ActorContext, &ProposedDo,
&ThinkInput) -> Verdict`.

**`Verdict` enum (the cornerstone):**

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    ProceedAs(PlanRef),
    RouteToHuman(&'static str),    // reason for the Kanban card
    Reject(&'static str),           // structural reason
    Defer(DeferCondition),          // wait on (e.g. another action)
}
```

**Wiring:** the existing `ActionDef::commit(def, actor, impact, guard, now)`
remains the gate. Inc 1 inserts a new STEP between `def-match` and `RBAC`:
`for policy in classview.policies { let v = (policy.exec)(actor, ...);
match v { ProceedAs(_) => continue, _ => return ActionState::ThinkRejected(v) } }`.

**Falsifying probe (gate F1):** one `ThinkSpec` impl
(`GoBdImmutability`) registered against a curated `account.move`-like
ClassView. Test: `commit(post_invoice, actor, impact, guard='Draft', now)`
on a row whose `audit_chain_prev_hash` is inconsistent → returns
`ActionState::ThinkRejected(Verdict::Reject("GoBD prev-hash mismatch"))`.
Same call on a consistent row → `ActionState::Committed`.

**Scope:** the THINK slot must NOT carry mutable state and must NOT call
into `MailboxSoA` writes. THINK reads only. Iron rule `THING is read. DO
writes (gated). THINK never writes.` enforced by the slot type signature
(`exec: fn(...) -> Verdict`, no `&mut`).

**Files touched:** `crates/lance-graph-contract/src/class_view.rs` (or
sibling) + `crates/lance-graph-contract/src/action.rs` (commit gate
extension) + 1 test. **~4 files mandatory.**

**Risks:** (a) the `policies` slot lives in lance-graph-contract → must stay
std-only (`Verdict` zero-dep). (b) the commit-gate insertion is a semver-
breaking change to `ActionState`'s outcome shape — managed by adding the
new variant `ThinkRejected(Verdict)` alongside existing variants (additive,
not replacing).

---

### Inc 2 — `OgarAst` enum + `TripletProjection` round-trip (§5 CONJECTURE → FINDING)

**Owns:** the portable operation tree. `OgarAst::{Thing(ClassRef),
Slot(ClassRef, FieldRef), Relation(ClassRef, RelRef, ClassRef),
Do(ActionRef, BindingSet), Transition(ClassRef, StateRef, StateRef),
Think(PolicyRef, BindingSet), Verdict(VerdictRef),
Constraint(ConstraintRef), AdapterCall(AdapterTarget, BindingSet)}`.
Lives in `lance-graph-contract::ogar_ast` (zero-dep, std-only — matches
the `codegen_spine::Triple` precedent).

**Round-trip via `TripletProjection`:** the round-trip gate already in
`codegen_spine` is reused. `OgarAst` ↔ `Vec<Triple>` is a lossless
projection: every variant has a defined triple shape (e.g. `Do(PostInvoice,
{this: I42})` ↔ `(ast:Do:0, ogar:action, PostInvoice) + (ast:Do:0,
ogar:binding:this, odoo:account_move/I42)`). Failure of the
`roundtrip_eq<OgarAstProjection>` test on any non-empty AST = a layout bug.

**Falsifying probe (gate F2):** unit test:
```rust
let asts = vec![
    OgarAst::Thing(class!("Invoice")),
    OgarAst::Do(action!("PostInvoice"), bindings!{this: I42}),
    OgarAst::Think(policy!("GoBdImmutability"), bindings!{this: I42}),
    OgarAst::Transition(class!("Invoice"), state!("Draft"), state!("Posted")),
];
for ast in &asts {
    let triples = ast.to_triples();
    let recovered = OgarAst::from_triples(&triples).unwrap();
    assert_eq!(*ast, recovered);
}
roundtrip_eq::<OgarAstProjection>(&all_triples).unwrap();
```

**Scope guard:** Inc 2 does NOT execute any `OgarAst::Do` — it only proves
the AST is well-formed and round-trippable. Execution is Inc 5.

**Files touched:** `crates/lance-graph-contract/src/ogar_ast.rs` (new
~300 LOC) + `crates/lance-graph-contract/src/lib.rs` (`pub mod ogar_ast`) +
~6 tests. **~3 files mandatory.**

**Risks:** (a) `BindingSet` shape — a `BTreeMap<&'static str, NodeGuid>`
keeps it zero-dep; (b) `ClassRef`/`ActionRef`/`PolicyRef` should be
`&'static str` (semantic names) plus an optional `classid` resolved lazily
— do NOT pre-bake classids into the AST.

---

### Inc 3 — `ArmDecision` + minimal `Executor::{NativeLance, SurrealAst}` (§6 CONJECTURE → FINDING)

**Owns:** the routing layer. `ArmDecision { op: OgarAst, executor: Executor,
fallback: Option<Executor> }`. `Executor::{NativeLance, SurrealAst,
HumanKanban, ExternalHttp(Url), Dll(CapabilityId)}` — Odoo / Rails adapter
variants reserved as enum slots, not implemented in Inc 3.

**Wiring with existing `OrchestrationBridge`:** `ArmDecision` is the typed
input to a new method `OrchestrationBridge::route(&self, op: &OgarAst,
actor: &ActorContext) -> ArmDecision`. The existing `StepDomain` taxonomy
is the codomain — `ArmDecision::executor` maps onto `BridgeSlot::domain`.

**Falsifying probe (gate F3):** two stub executors (NativeLance + SurrealAst)
that each accept one `OgarAst::Do(PostInvoice, {this: I42})` and return a
typed `Outcome { id: NodeGuid, state_after: StateRef }`. Test: same Op
routed via NativeLance vs SurrealAst → same `Outcome.state_after`. Byte-
identical NOT required (the executors render differently); semantic
identity IS.

**Scope:** Inc 3 ships two STUB executors. Real Lance write (`write_row`)
and real SurrealQL emission ride later increments (and the parallel plans
`cypher-kanban-ast-unification-v1` Inc 1 + `lite-unified-surrealql-lance-v1`).

**Files touched:** `crates/lance-graph-contract/src/arm.rs` (new ~200 LOC) +
`crates/lance-graph-contract/src/orchestration.rs` (extension) +
`crates/lance-graph-planner/src/...` (Executor stub impls + tests) +
~3 tests. **~5 files mandatory.**

**Risks:** parallel to `OrchestrationBridge` is the highest drift risk —
must extend, not duplicate. Council MUST verify the integration shape
before code lands.

---

### Inc 4 — Curator promotion probe (§2 promotion rule → automated)

**Owns:** the ≥2-curator promotion rule, mechanized. A test that scans
`crates/lance-graph-ontology/src/odoo_blueprint/*.rs` (curated Odoo Core)
+ `/tmp/sources/<openproject-nexgen-rs>/crates/op-surreal-ast/from_triples.rs`
(the Rails predicate vocabulary) + (later) WoA model surfaces, and produces
a table: `<primitive> | <curator-1> | <curator-2> | <syntactic forms seen>`.

**Falsifying probe (gate F4):** at least 4 primitives surfaced under ≥2
curators with different syntactic forms — e.g.:
- `state machine` (Odoo `fields.Selection('state')` + Rails `acts_as_state_machine` + WoA `WoStatusAction` enum)
- `audit chain / immutability` (Odoo `restrictive_audit_trail` + WoA `audit_chain.rs::chain_hash`)
- `tenant boundary` (Odoo company_id + WoA tenant_id + Rails tenant_scope)
- `number sequence` (Odoo `ir.sequence` + WoA `number_sequence` + Rails `acts_as_sequenced`)

If <4 promotion-qualified primitives surface, the ≥2-curator rule is
unfalsifiable on today's corpus → either the rule needs sharpening OR we
need more curator data before Inc 5.

**Scope:** Inc 4 is a TEST + a generated report file under
`docs/curator-promotion-table-v1.md`. NOT a Core change. NOT a new typed
slot. Pure measurement.

**Files touched:** `crates/lance-graph-ontology/tests/curator_promotion.rs`
(new ~250 LOC) + `docs/curator-promotion-table-v1.md` (generated) +
~3 helper modules. **~5 files mandatory.**

**Risks:** the openproject-nexgen-rs zipball needs to be re-fetched at
test time (or vendored, which adds a 4-MB blob to the workspace). Decision
during council: vendor or fetch.

---

### Inc 5 — Litmus probe: same `OgarAst::Do(PostInvoice, …)` ≡ across executors (§10 worked example → FINDING)

**Owns:** the named falsifying test for the whole doctrine. Lock the
crown-on-the-curator failure-mode.

**Setup:** a minimal `Invoice` ClassView with:
- `policies = &[GoBdImmutability, ActorAuthorized]`
- `actions = &[PostInvoice]`
- `state_machine = Draft → Posted → Reversed`

**The probe (gate F5):**
```rust
let op = OgarAst::Do(action!("PostInvoice"), bindings!{this: I42});
let actor = ActorContext::buchhaltung_user();

let outcome_native    = NativeLanceExecutor::run(&op, &actor)?;
let outcome_surrealql = SurrealAstExecutor::run(&op, &actor)?;
let outcome_odoo_stub = OdooAdapterStub::run(&op, &actor)?;
let outcome_rails_stub= RailsAdapterStub::run(&op, &actor)?;

assert_eq!(outcome_native.state_after,    State::Posted);
assert_eq!(outcome_surrealql.state_after, State::Posted);
assert_eq!(outcome_odoo_stub.state_after, State::Posted);
assert_eq!(outcome_rails_stub.state_after, State::Posted);

// THINK consistency: same actor + same row + same op → same Verdict
let verdict_n = NativeLanceExecutor::dry_run_think(&op, &actor)?;
let verdict_s = SurrealAstExecutor::dry_run_think(&op, &actor)?;
let verdict_o = OdooAdapterStub::dry_run_think(&op, &actor)?;
let verdict_r = RailsAdapterStub::dry_run_think(&op, &actor)?;
assert_eq!(verdict_n, verdict_s);
assert_eq!(verdict_s, verdict_o);
assert_eq!(verdict_o, verdict_r);
```

**Crown-detection sub-probe:** force a divergence by injecting a curator-
specific quirk into ONE executor (e.g. `OdooAdapterStub` rejects the post
because it secretly normalizes `posting_date` to local-tz). The litmus
test MUST fail in a way that prints the curator name. The error message
template:
```
"the curator wearing the crown: <curator-name> diverged on <field>
 (expected <verdict>, got <verdict>) — see docs/OGAR_AR_SHAPE_ENDGAME.md
 §10"
```

**Scope:** Inc 5 ships the test framework + the four stub executors. The
Odoo + Rails stubs are NOT real adapter calls — they're hand-coded outcomes
that demonstrate semantic equivalence (and the divergence case
demonstrates the failure-mode-name's catchability).

**Files touched:** `crates/lance-graph-ontology/tests/litmus_post_invoice.rs`
(new ~500 LOC) + stub-executor support modules + ~6 tests.
**~6 files mandatory.**

**Risks:** the temptation to make the stubs "real" enough to spread into
Inc 6+ ambitions (full SurrealQL emission, real Odoo XML-RPC) — Council
MUST scope-cap at "demonstrate semantic identity on hand-coded outcomes."

---

## 2. Gate ledger

| Gate | Owns | What "green" means | Demoter |
|---|---|---|---|
| F0 | Doctrine + spine + litmus committed | DONE (this branch, commits `e6a1539..3b898e8`) | n/a |
| F1 | `ClassView::policies` THINK slot | One `ThinkSpec` returns `Verdict::Reject` on inconsistent row, `ProceedAs` on consistent. Existing tests stay green. THINK exec signature is `fn(&...) -> Verdict` (no `&mut`). | THINK exec captures `&mut` anywhere → REJECT |
| F2 | `OgarAst` enum + `TripletProjection` round-trip | `roundtrip_eq<OgarAstProjection>` green for a 4-variant input set. No allocator dep added to contract. | round-trip fails on ANY variant → REJECT (not "fix and retry") |
| F3 | `ArmDecision` + Executor stubs | Same `OgarAst::Do` routed via two executors → same `state_after`. ArmDecision integrates with `OrchestrationBridge`, doesn't parallel it. | parallel ArmDispatch + OrchestrationBridge surfaces → REJECT |
| F4 | Curator promotion table | ≥4 primitives surfaced under ≥2 curators with different syntactic forms; report generated to disk | <4 primitives → demote ≥2-curator rule (needs more curators before it's testable) |
| F5 | Litmus probe | 4-way executor semantic identity green; crown-detection sub-probe fails with the named error message | semantic divergence not detected → demote doctrine §5/§6 to CONJECTURE (and write down what leaked) |

**Promotion sequencing:** F1 → F2 → F3 → F4 → F5 strictly ordered. F3
depends on F2 (ArmDecision routes an `OgarAst`); F5 depends on F1+F2+F3
(litmus uses policies+AST+ArmDecision). F4 is independent — could land in
parallel.

## 3. Council pre-merge

The 5+3 council pattern (`docs/OGAR_AR_SHAPE_ENDGAME.md` is operator-
ratified; this PLAN is NOT yet). **Council members and their angles for
THIS plan specifically:**

**Panel:**
1. `convergence-architect` — does this plan have 0-friction alignment with
   `cypher-kanban-ast-unification-v1` Inc 1 (Cypher→SurrealQL lowering),
   `lite-unified-surrealql-lance-v1` (the SurrealQL adapter target), and
   `probe-excel-compute-dag-v1` (the recompute substrate)? Where do they
   compose vs. where do they overlap?
2. `prior-art-savant` — is the THINK slot, OgarAst, or ArmDecision already
   prefigured by an existing type (`ThinkingStyle`, `OrchestrationBridge`,
   `StepDomain`, `ExpertCapability`)? Sweep contract + planner.
3. `dto-soa-savant` — does `ClassView::policies` fit the four-column SoA
   discipline, or is it adding a new layer? Is `OgarAst` a runtime carrier
   or codegen-time output? (Should be codegen-time, like `Triple`.)
4. `cascade-impact-savant` — per-Inc file count. F0=done. F1 ~4, F2 ~3,
   F3 ~5, F4 ~5, F5 ~6 — does it hold under audit?
5. `core-first-architect` — does Inc 1 extend the Core deliberately (the
   doctrine-correct move per `E-ODOO-CORE-FIRST-STRUCTURAL`), or is it
   bolting on? Is the THINK slot RESIDUE-CORE-shaped?

**Critique:**
1. `truth-architect` — are the gates F1-F5 truly falsifying, or rubber
   stamps? Specifically: F4 (≥4 primitives in ≥2 curators) — has truth-
   architect confirmed that ≥2 curators are actually surveyable today?
   F5 crown-detection — does the test design force a divergence the
   doctrine can name, or does it only check absence-of-divergence?
2. `baton-handoff-auditor` — `OgarAst` in lance-graph-contract creates a
   new cross-crate boundary (contract → planner → lance-graph). Does the
   handoff survive sprint-handover? `ActionState::ThinkRejected(Verdict)`
   is a new enum variant — will downstream `match` arms break?
3. `iron-rule-savant` — does any Inc violate I-LEGACY-API-FEATURE-GATED
   (same name, different semantics by feature)? I-VSA-IDENTITIES (operate
   on identity, not content — the AST is identity-shaped, verify)?
   I-SUBSTRATE-MARKOV / I-NOISE-FLOOR-JIRAK (irrelevant for codegen, but
   sweep)?

## 4. Cross-plan integration

| This plan | Companion plan | How they compose |
|---|---|---|
| Inc 3 (Executor::SurrealAst stub) | `cypher-kanban-ast-unification-v1` Inc 1 (Cypher→SurrealQL lowering) | When the lowering ships real, this plan's stub gets replaced; the litmus test rides the real emission. |
| Inc 3 (Executor::SurrealAst stub) | `lite-unified-surrealql-lance-v1` (kv-lance + SurrealQL primary surface) | When SurrealQL becomes the primary query surface, the SurrealAst executor's "real" path lands. |
| Inc 1 (`ClassView::policies`) | `probe-excel-compute-dag-v1` (ClassView::compute_dag) | Both extend `ClassView` additively; council MUST verify the two extensions don't collide in `ClassView` layout. |
| Inc 4 (curator promotion) | `E-AR-PROJECTION-CORRECTION-1` Phase 1 Option A (Odoo predicate arms in nexgen) | The curator promotion table grows when nexgen's Odoo arms ship — this plan + that plan compose at the Inc 4 ledger. |

## 5. Out of scope

- **Real SurrealQL emission** (rides `lite-unified-surrealql-lance-v1`).
- **Real Odoo adapter call / Rails adapter call** (Inc 5 uses stubs to
  demonstrate semantic identity; making them real is Phase 2+).
- **`MailboxSoA` writes for THINK results** (THINK never writes; F1 gate).
- **New runtime types beyond Verdict + OgarAst + ArmDecision** — the
  endgame doctrine names exactly these three; this plan does NOT smuggle
  in more.
- **Cross-repo PRs** to nexgen / surrealdb-fork / woa-rs — those ride
  `E-AR-PROJECTION-CORRECTION-1`'s sequenced path.
- **Curator data for SAP** — SAP is named as a future curator in the
  doctrine; Inc 4 sweeps Odoo + Rails + WoA only.

## 6. Failure → demotion paths

| Inc fails | Demotion |
|---|---|
| F1 | `ClassView::policies` is rejected as a slot; THINK stays as ASSERT-clause-promotion only (the prior approach). Doctrine §4 demotes the new-slot CONJECTURE. |
| F2 | `OgarAst` is rejected as a typed enum; the polyglot harvest stays at `codegen_spine::Triple` level only. Doctrine §5 demotes; future work re-attempts with a different shape. |
| F3 | `ArmDecision` is rejected; `OrchestrationBridge` stays the routing surface. Doctrine §6 demotes; ARM stays a documentation concept. |
| F4 | ≥2-curator rule is unfalsifiable today; either gather more curators OR weaken the rule. Doctrine §2 sub-claim is bounded. |
| F5 | The litmus is the doctrine's load-bearing probe. F5 failure means a curator IS wearing the crown today. Write down which curator, and what leaked. Doctrine §10 + the spine survive; the executors get re-engineered. |

## 7. CONJECTURE / FINDING per Inc

| Inc | Pre-council | Post-council (target) | Post-merge (Inc green) |
|---|---|---|---|
| 0 (doctrine) | FINDING | FINDING | FINDING |
| 1 (THINK slot) | CONJECTURE | PLAN-RATIFIED | FINDING (slot exists + 1 policy works) |
| 2 (OgarAst) | CONJECTURE | PLAN-RATIFIED | FINDING (enum exists + round-trip green) |
| 3 (ArmDecision) | CONJECTURE | PLAN-RATIFIED | FINDING (routing decides + executors semantically equal) |
| 4 (curator promotion) | CONJECTURE | PLAN-RATIFIED | FINDING (≥4 primitives surface) or DEMOTION (<4) |
| 5 (litmus) | CONJECTURE | PLAN-RATIFIED | FINDING (semantic identity holds + crown-detection works) or DEMOTION (a curator leaked) |

## 8. Branch + PR strategy

- This plan lands on `claude/hydrate-dolce-dul-owl-Ce9Oa` (the doctrine
  branch) as one PR — "the plan + the council verdict."
- Each Inc then ships as its own PR on its own branch
  (`claude/ogar-endgame-inc1-think-slot`, etc.).
- Council runs ONCE for this PLAN PR. Per-Inc PRs ride the standard
  `brutally-honest-tester` gate, not a fresh 5+3 (the architecture is
  ratified; per-Inc PRs are implementation).
- Exception: F5 (litmus) gets a council re-run because it's the doctrine's
  load-bearing probe — divergence here might warrant a wider re-design.

## 9. Open questions for the council

1. **THINK slot ownership** — does `policies` live on `ClassView` (per Inc 1
   spec) or on a new sibling `ClassPolicies` table (matching the
   `ClassActions` + `ClassMethods` separation in
   `lance-graph-contract::codegen_manifest`)? Argument for separate:
   symmetry with DO. Argument against: THINK is inherent to a class's
   identity in a way DO isn't (you can have a class with no actions; a
   class with no policies is just unsafe).
2. **OgarAst `BindingSet`** — `BTreeMap<&'static str, NodeGuid>` keeps it
   zero-dep but disallows runtime bindings. Is that a feature or a bug?
3. **Crown-detection error message format** — locked or per-call-site
   variable? The doctrine wants the phrase "curator wearing the crown"
   to be quotable in PR reviews. Spec'd as locked here (`§5 sub-probe`).
4. **F4 corpus surveyability** — has anyone verified the Odoo + Rails
   curator surfaces are scrape-able today without vendoring an openproject-
   nexgen-rs tarball? Acceptable answers: (a) yes, via the existing
   `/tmp/sources/` zipball pattern; (b) no, vendor a snapshot under
   `vendor/`; (c) defer F4 to a follow-up plan.
5. **Cross-plan layout collision** — does `ClassView::policies` (Inc 1)
   conflict with `ClassView::compute_dag` (`probe-excel-compute-dag-v1`)?
   Both extensions are additive in principle; need to verify in code.

## 10. Acceptance criteria for this PLAN PR

- [ ] 5+3 council runs and the verdict is documented (`PLAN-RATIFIED` or
      `REJECT-WITH-REASONS` per Inc).
- [ ] Open questions §9 are answered (each one moved to either "decided"
      or "deferred-to-Inc-N").
- [ ] `INTEGRATION_PLANS.md` entry updated post-council with the verdict
      summary.
- [ ] No code changes in this PR — plan + council only.

When PLAN-RATIFIED: each Inc opens its own PR off the doctrine branch.
