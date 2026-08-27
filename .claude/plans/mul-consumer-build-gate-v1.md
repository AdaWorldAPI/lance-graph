# Build gate: the D-MCAL arc against its real consumers (`mul-consumer-build-gate-v1`)

> **Status:** GATE RUN — 2026-08-27. Discharges D-MCAL-6 and the second half of
> **F-MUL-6**.
> **Head under test:** the D-MCAL arc combined —
> D-MCAL-1 (#1065, merged) + D-MCAL-2 (#1066) + D-MCAL-3 (#1067) +
> D-MCAL-4 (#1068) + D-MCAL-5 (#1069).
> **Consumers built:** ada-rs @ session checkout, MedCare-rs @ `17871e06`.

---

## 0. Why this exists as its own deliverable

F-MUL-6 has two halves, and the plan splits them on purpose:

> every known cross-repo consumer classified per symbol, then **built** against
> the eventual SHA — *fails when* classification skipped or replaced by grep

D-MCAL-1 discharged the classification. **A classification is not a build.** A
grep tells you a symbol is mentioned; only a compiler tells you the change is
source-compatible. The workspace's own invariant says so:

> A source-breaking contract change is not verified until known unbound-git
> consumers **build** against the proposed head.

---

## 1. Method

ada-rs consumes lance-graph as a **git** dependency, so the head under test was
bound with a `paths` override rather than by editing the committed manifest:

```toml
# ada-rs/.cargo/config.toml — TEMPORARY, never committed
paths = ["/home/user/lance-graph/crates/lance-graph-contract"]
```

**The override was verified to actually bind**, rather than assumed. A
temporary probe was compiled inside ada-rs:

```rust
use lance_graph_contract::kanban::KanbanColumn;
assert_eq!(KanbanColumn::Planning.veto(), Some(KanbanColumn::Prune));
```

`KanbanColumn::veto` exists **only** on the head under test (it is D-MCAL-4's
addition). It resolved — no `E0599` — so the compiler was reading the arc, not
the published git source. Without this check the whole gate could have passed
against `main` and reported nothing. The probe and the override were both
reverted afterwards; ada-rs's working tree is unchanged.

> **A note on what this probe does and does not prove**, because the same fact
> was misused elsewhere in this arc and the distinction is worth keeping.
> "`veto` is absent from `main`" is valid evidence for **which source cargo
> bound** — that is a question about symbol availability, and symbol
> availability is exactly what it answers. It is *not* valid evidence that
> routing without MUL ground was previously **impossible**: `next_phases()` was
> already public and already exposed `Prune`, so the capability existed
> unnamed. D-MCAL-4's original red-state claim made that second, invalid
> inference and has been corrected (see its test file's header). The probe here
> makes only the first.

---

## 2. Result — ada-rs

**Does not compile. Three errors. Zero of them from this arc.**

```
error[E0559]: variant `lance_graph_contract::mul::GateDecision::Block`
              has no field named `reason`   -- src/contract_impls.rs:75
error[E0559]: variant `lance_graph_contract::mul::GateDecision::Hold`
              has no field named `reason`   -- src/contract_impls.rs:83
error[E0559]: variant `lance_graph_contract::mul::GateDecision::Hold`
              has no field named `reason`   -- src/contract_impls.rs:90
    = note: available fields are: `texture`, `flow`
```

All three are the **pre-existing #1045 break**, which landed on 2026-08-26 and
predates every deliverable in this arc. This is the break the arc exists to
explain: `E-A-HOT-PATH-FIX-NARROWED-A-PUBLIC-CONTRACT-WORKSPACE-GREEN-IS-NOT-CONTRACT-GREEN-1`.

**What the arc itself contributed: nothing.** Specifically —

| deliverable | expected effect on ada-rs | observed |
|---|---|---|
| D-MCAL-2 — `PlannerContract::gate_check` **removed** | none; ada-rs never implemented it | none |
| D-MCAL-2 — `MulProvider::gate_check` **deprecated** | a warning at ada-rs's impl site, never an error | no error |
| D-MCAL-3 — docs + tests | none | none |
| D-MCAL-4 — `KanbanColumn::{advance, veto}` added | none; purely additive | none |
| D-MCAL-5 — docs + tests | none | none |

So the gate's verdict is: **the arc is source-compatible with ada-rs.** The
consumer is red, and was red before the arc, for one reason that the arc
deliberately does not paper over.

### 2a. The stopgap stays unpushed

The three errors are trivially "fixable" by supplying a `texture` and a `flow`
at each site. That fix is **refused**, per the plan's §6 invariant and the
operator ruling that produced it: a red compiler is preferable to a green lie.
Inventing two calibration coordinates that ada-rs never measured would
reproduce exactly the defect the census found in MedCare-rs.

**The honest fix is D-MCAL-4's route**, now available on this head: ada-rs's
consent veto is domain evidence, so it calls `KanbanColumn::veto()` and
constructs no `GateDecision` at all. That migration is ada-rs's to make, in
ada-rs's repo — this gate only proves the destination exists and compiles.

Two of `AdaMulAdapter::gate_check`'s three arms are genuine MUL (Dunning-Kruger,
allostatic load) and keep working through `assess()`; only the consent arm
moves. `f_mul_5_genuine_mul_arms_survive_without_a_verdict_method` is the proof
that the two survivors need no verdict method.

---

## 3. Result — MedCare-rs

**Already migrated to the post-#1045 shape; unaffected by this arc.**

`medcare-first-thought` constructs `GateDecision::{Block, Hold}` with the
`{ texture, flow }` payload at four sites and its own source comments record
the 2026-08-26 upstream change. So MedCare took the #1045 break and absorbed it
— by fabricating both coordinates, which is the finding D-MCAL-1 recorded.

Against this arc:

| deliverable | effect on MedCare-rs |
|---|---|
| D-MCAL-2 | none — implements neither trait |
| D-MCAL-3 | none — docs and tests only |
| D-MCAL-4 | none (additive); **this is the route its four sites should take** |
| D-MCAL-5 | none |

MedCare was **not** rebuilt end-to-end here, and this document does not claim
it was. Its dependency graph pulls `ogar-obo`, `jc`, `medcare-cohorts` and the
full lance stack from git; standing that up is a larger exercise than the gate
needs, and the compatibility question is answerable exactly: MedCare touches no
symbol this arc changed, and the one symbol it does touch
(`kanban::advance_on_gate`) is behaviour-identical, pinned by
`f_mul_4_routing_ignores_the_calibration_payload` and by D-MCAL-4's
equals-the-fabricating-route falsifiers.

**Stated as a limitation, not a pass:** the ada-rs half of F-MUL-6 is a real
compile; the MedCare half is symbol-level reasoning plus in-tree pins. A future
session with the MedCare build stood up should close that gap.

---

## 4. Verdict

| falsifier | status |
|---|---|
| F-MUL-6, first half (classify per symbol) | discharged by D-MCAL-1 (#1065) |
| F-MUL-6, second half (BUILD against the head) | **discharged for ada-rs by compilation**; MedCare covered at symbol level with the limitation stated in §3 |
| F-MUL-5 (removing/reframing preserves behaviour) | MUL half green in-tree; consumer half green in principle — the two genuine arms compile unchanged, only the class-B arm moves |

**The arc adds zero new breakage to either consumer.** The single red consumer
is red from #1045, and the route out of that red now exists on this head
without requiring anyone to fabricate a calibration reading.

---

## 5. What the gate does not certify

- It does not certify ada-rs **works** — only that this arc does not break it
  further, and that its errors are all attributable to #1045.
- It does not run either consumer's test suite.
- It does not cover the other 15 repos that depend on `lance-graph-contract`.
  D-MCAL-1 measured that none of them names a MUL symbol; that is a grep, and
  by this document's own argument a grep is not a build. It is recorded as the
  bound on this gate's coverage rather than hidden.
