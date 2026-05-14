## W4 run — 2026-05-14

Status: COMPLETE
Spec: .claude/specs/pr-ce64-mb-3-bindspace-efgh.md (~14 KB)

Key findings:
- Column H EntityTypeId already wired (PR #272): entity_type Box<[u16]> + push_typed + driver.rs:311. This PR adds TypeColumn wrapper + BindSpaceView accessor + FIX-5 test.
- MergeMode::Superposition=2 already shipped in collapse_gate.rs. This PR documents semantics + Column E tagging.
- Corrected footprint: 71,777+292=72,069 B/row (plan §1/§2 numbers predate cycle plane migration to Vsa16kF32).
- AccumulatedOntology: global on ShaderDriver (NOT per-row), correctly per plan §11.
- AwareOp ndarray impls D-F4/D-F5 out-of-scope; shipped as no-op stub [128u8;256].

OQs: OQ-W4-1 BindSpaceView placement (par-tile vs driver, BLOCKER MB-5); OQ-W4-2 EdgeColumn 8-slot scope; OQ-W4-3 AwareOp stub.
