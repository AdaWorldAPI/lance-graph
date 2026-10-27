# D-CFA-1 — S1 typed ON × WHEN × DO incidence: source to candidate bindings

**Status: docs-only design / preregistered falsifiers; not run.** Parent: [D-CFA-0](README.md). Dependency: review CFA-0. There is no new compiler, cube ABI, physical tensor, runtime, or authorization to edit source.

## Question
Can source-attested method records from OGAR (Trigger/Applicability/Capability), Ruff SPO (READS/DERIVES/EFFECTS/REQUIRES/AUTHORIZES/TRIGGERS), Odoo, or rust-analyzer be bound to *logical* ON/WHEN/DO registrations without silently manufacturing parameter types, guards, capabilities, or a unique method from a cell?

S1 is a proposed method-analysis view, **not equivalent** to the existing DT64-0 logical 64-predicate catalogue. Do not claim that the ON×WHEN×DO axes are an admitted DR³ interpretation while NFR-OP-2 is open. Identity, scope and proof remain in their owners. #1799 DT64 Q0 eight-predicate lab is a narrower existing proposal, not replaced here.

## Logical shape, no Rust ABI commitment
- `ON`: event edge / trigger kind, and observed-vs-synthetic flag.
- `WHEN`: reference to a typed guard expression with 3-valued result and explicit completeness/authorization. A WHEN axis **does not mean** one Boolean atom; AND/OR/nonmonotone guard clauses cannot be folded as simple OR of satisfied atomic bits.
- `DO`: effect *descriptor* / capability shape (read/write/raise/external call), not authorization to execute it.
- `REGISTRATION`: stable semantic MethodId, signature (receiver and argument types), source identity+revision+schema, method kind/phase, declared precedence/supersession (if any), optional compiled recipe reference, cell coordinate(s), visibility/owner permissions. One cell maps to **zero, one, or several** method identities.
- `WITNESS`: separately attested global↔local focus, pinned source snapshot, scope, codebook generation and knownness; local 6-bit ordinal is not a global identifier.
- `THOUGHT`: orthogonal resident hypothesis slot 0..63, never a fourth axis inside S1 or a surrogate source revision.

Cell presence means a typed registration candidate exists. It says nothing about which method ran, whether a guard was true, whether the effect succeeded, or whether two registrations are mutually exclusive. Use the existing source-owner → DT64 Q0 → borrowed Quack path, and keep the eventual method-binding mapping as a **cold proposed view** until its owner agrees.

## Corpus options (not chosen in this round)
1. Odoo `account_move`, multi-emitter parity from AdaWorldAPI/odoo-rs #44.
2. OGAR `OgitTables` `ogit:allowed` source/verb/target columns, with schema facets only. OGIT lacks concrete deployed guard values/parameter signatures; represent UNKNOWN rather than invent.
3. D-SAG-6 Rust codegraph from rust-analyzer HIR + optional pinned rustc MIR. Ruff is structural crosswalk, not a Rust MIR oracle.

## Kill tests to implement in a separately approved probe
| ID | Negative or discriminating fixture | Expected |
|---|---|---|
| I01 | two methods share one ON/WHEN/DO coordinate and revision | both identities retained; no last-wins |
| I02 | same verb/field name but different source owner/version | distinct source identities; rebinding requires witness |
| I03 | `WHEN=A AND B`; A true, B false/unknown | must not fire |
| I04 | absent guard value, type or authorization in OGIT schema | UNKNOWN or REFUSED, never fabricated |
| I05 | specialized receiver and parent type both match | source-defined precedence or explicit ambiguous/multicast classification, never implicit file order |
| I06 | `@api.constrains` reports postcondition violation | cannot reinterpret as a prerequisite guard without evidence |
| I07 | schema-only source says may-write X | possible effect only; not observed effect |
| I08 | stale source/codebook/scope with same local 6-bit coordinates | refuses; no cache/Keep escape |

## What must be measured later
Coverage by attestation grade, methods per cell, number of occupied cells, axes per class frame, missing guard/parameter fraction, alias/dispatch ambiguity rate, binding cost. **No performance numbers claimed now.**

## Next
[CFA-2](cfa-2-ablation-replay.md) assumes only *explicitly chosen* executable fixtures, and must not turn descriptive source-only facts into simulated outcomes without declaring the model.

Sources: [D-SAG-6](../../../.claude/plans/d-sag-6-rust-codegraph-math-logic-tracert-v1.md), [DT64 Q0](../../../.claude/plans/d-dt64-q0-borrowed-quack-binding-seam-v1.md), [odoo-rs #44](https://github.com/AdaWorldAPI/odoo-rs/pull/44), [Ruff role vocabulary](https://github.com/AdaWorldAPI/ruff/blob/main/crates/ruff_spo_triplet/src/role.rs).
