# tesseract-rs — AST-DLL C++→Rust Codegen Harness v1

> **Type:** plan (sub-plan). Deliverables D-OCR-40/41/42. The transcode *mechanism*.
> **Status:** PLANTED 2026-06-15 — design only.
> **Front:** post-#496. Uses `AdaWorldAPI/ruff` AST/codegen crates as the Rust-emission engine.
> **Canon anchors:** master §4. Deterministic + diff-gated (bit-reproducibility doctrine).
> **Skip-by-rule:** only leaf/mechanical modules are codegen targets; ownership-heavy code is hand-ported or replaced.

---

## 0. Intent

Transcode the *mechanical* C++ leaf modules (container parse, unicharset, recoder,
dawg node-arrays, weight-matrix struct walks) into Rust by a **deterministic,
reviewable codegen harness** rather than by hand — so the faithful tier is
auditable and re-runnable. The harness pairs a **clang C++ AST frontend** with a
**Rust emission backend built on the `ruff` AST/codegen crates**.

## 1. Why ruff (honest scoping)

`ruff` is a *Python* toolchain — `ruff_python_parser` / `ruff_python_ast` parse
**Python**, not C++. So ruff is **not** the C++ frontend. Its value here is the
mature, battle-tested **Rust-side AST → source emission discipline**:
`ruff_python_codegen` (AST → formatted source), `ruff_formatter` (the formatting
IR), `ruff_source_file`, and the `ruff_python_dto_check` pattern (structural
invariant checks on a typed AST). We reuse those *patterns and crates* as the
emission/formatting backend for a `RustAst → rust source` pipeline. The C++ side is
clang.

```
C++ source ──(libclang)──► Clang AST ──► [AST DLL: stable IR dump] ──► RustAst builder
   ──► (ruff codegen/formatter discipline) ──► formatted .rs ──► diff-gate vs FFI oracle
```

## 2. The "AST DLL" — D-OCR-40

The C++ AST is extracted once into a **stable, serializable IR** (the "AST DLL"):
a libclang traversal that dumps the subset we transcode (struct/enum decls, plain
methods, table initializers, fixed-size array walks) as a typed IR — independent of
clang version drift, so the emission step is reproducible. Functions touching
pointers-into-mutable-graphs, virtual dispatch, or template metaprogramming are
**flagged NOT-CODEGENABLE** and routed to hand-port/replace (they are layout code —
already skipped, per master §3).

## 3. Rust emission via ruff crates — D-OCR-41

A `RustAst` builder consumes the IR and emits idiomatic Rust:
- field-by-field struct/enum transcription (canon: byte layout preserved);
- table/array initializers → `const`/`static` Rust tables;
- the emission goes through ruff's formatter IR so output is deterministic and
  diff-stable (re-running codegen produces byte-identical source).
- a `dto_check`-style pass asserts the **LE byte contract** is preserved per struct
  (no silent re-ordering / re-widening — the same invariant the SoA envelope audit
  enforces).

## 4. Diff-gate — D-OCR-42

Every codegen'd module is validated against the FFI oracle:
- behavioral: emitted Rust function vs `libtesseract` function on the same inputs
  (e.g. unicharset id↔utf8, recoder encode/decode, dawg word-membership) → byte-equal;
- structural: `dto_check` confirms each emitted struct's byte image matches the C++
  `sizeof`/offset dump.
Codegen output is committed (not generated at build) so reviewers see real Rust;
the harness is re-runnable to prove the commit equals the generator output.

## 5. Module assignment (codegen vs hand vs replace)

| C++ area | Route |
|---|---|
| `tessdatamanager`, `unicharset`, `unicharcompress` (recoder), `dawg`/`trie` node arrays, `weightmatrix` struct/quant walks | **CODEGEN (D-OCR-41)** |
| `recodebeam` (beam + dawg interaction), int8 GEMV rounding | **HAND-PORT** (numeric/behavioral subtlety) |
| `textord`/`ccstruct` layout, Leptonica | **REPLACE** (ocrs / minimal imageproc) — never enters the harness |

## 6. Deliverables

- **D-OCR-40:** libclang → stable IR dump for the codegen-target module set; NOT-CODEGENABLE flagging works.
- **D-OCR-41:** IR → committed Rust via ruff emission; re-run is byte-identical.
- **D-OCR-42:** behavioral + structural diff-gate green for the target modules vs the FFI oracle.

## 7. Open decisions

- **OD-3 (from master):** libclang in-process vs clang `-ast-dump=json` consumed by
  a Rust IR. JSON is simpler/decoupled; libclang is richer/faster. Default: clang
  JSON dump for v1 (decoupled, reproducible), libclang later if needed.
- **OD-40a:** is the AST-DLL harness OCR-specific, or a reusable
  `AdaWorldAPI/<cpp-transcode>` tool? (It would also serve other C++→Rust ports.)
