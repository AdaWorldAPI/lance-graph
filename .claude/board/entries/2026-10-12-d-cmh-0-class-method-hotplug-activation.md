## 2026-10-12 — D-CMH-0 — Class × Schema × Methods activation connects HotPlug, WideFieldMask, Quack and R2IL without a second registry

**Status:** [H] RESEARCH PROPOSAL, documentation only. No runtime modification, parity fixture or benchmark executed by this PR.

**Finding:** The hot-plug class mechanism is already split correctly: consumer declares `HotPlug`; OGAR resolves authority/capabilities and storage reading into `Activation`; `SlabDeclaration` pins already-written physical bytes; OGIT resolves canonical names; `ClassView::fields` and `WideFieldMask` provide a field basis. Quack already binds names to numeric `Col/Mask/validity` and lowers into mask-risc; `lance-graph-r2il::procs` has real named fold sequences. The missing entity is **a cold, verified, versioned method-to-field binder** shared by both frontends, not another execution engine or class registry.

**Correction to old gaps:** OGAR `ogar-loco::Inventory`, `VecInventory::push_keyed` and `Interpreter::with_inventory` are **already implemented**. The gap is linking `MethodDecl` + semantic method signature to the existing Inventory key and a verified method body under one ClassView/Schema. `codegen_manifest::MethodSig` is a separate C++ signature surface, not a loco method body.

**Critical fences:**
- No new zero-dep contract dependency, global count fuse, classid mint, opcode, CE64 reinterpretation, per-consumer plug crate or hidden V1 fallback.
- `WideFieldMask` indexes **fields**, Quack row masks index **rows**; presence is not authorization. Require `required ⊆ available ∩ allowed` and current RBAC/source version.
- The 23-field SIMAF "CATS" is not proof of SAP CATSDB/BAPI parity; Odoo `account.move` wide fixture is 216 fields, not an SAP-equivalent model; D-XGP conversion grades and unit mappings must be proved. Wide ComputeEdge has a 64-field gate.
- Quack and R2IL may emit different legal programs over the **same backend**; match typed results under an independent oracle, not arbitrary bytecode identity.
- `do()` remains OGAR `ActionInvocation` + hard gate; reading-only plugs without capabilities still activate.

**First kill test CMH-P13:** bind one SIMAF/CATS pure billable-hours method against one canonical ordered field basis, run Quack and R2IL on the same borrowed dataset generation, compare exact scalar reference (including two batch-local scales and nullable cases). Deliberately swap a numeric lane in one frontend: the test MUST fail. Next prove field positions >63 in the 216-field Odoo fixture and absence of silent authority escalation.

**Home:** [D-CMH-0..12 activation RFC](../../plans/class-method-schema-hotplug-v1.md), [24 falsifiers CMH-P01..P24](../../plans/class-method-hotplug-falsifiers-v1.md), [source inventory](../../knowledge/class-method-hotplug-existing-contracts.md).

**Cross-repo:** [OGAR #357](https://github.com/AdaWorldAPI/OGAR/pull/357) is the semantic companion; lance-graph #1496/#1497/#1498 are NAL/PLN/Hyperon research, not a license to reinterpret bytes or mint a new evaluator. Existing D-XGP-0..8 and D-HPR rulings constrain the first method binder.
