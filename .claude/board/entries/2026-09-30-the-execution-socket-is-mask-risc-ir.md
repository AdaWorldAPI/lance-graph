# 2026-09-30 — The execution socket under the lexical substrate already exists: it is `lance-graph-mask-risc`'s IR

**Status:** VERIFIED-IN-CODE · OPEN (segment view, multi-hop) — plan `.claude/plans/deepnsm-v2-cam96-pairwise-v5.md` §11 (D-CPW-15)

## Read (2026-09-30)
- `lance-graph-quack` is the DuckDB-shaped SURFACE: it builds `Program`s and never evaluates one (D-QCK-0..10). The socket is the IR it lowers to: `Program`, `Planes`/`LaneRef::{I32,U32,U64,Strided}`/`Foreign`, `Value`/`Out`/`ExecError`.
- mask-risc `[dependencies]` = `ndarray` only. `src/` reaches it only as `ndarray::simd`. `src/` has zero hits for NARS / Pearl / COCA / Fisher / lemma / `ReferenceSet` / EWA / Jirak / HHTL / shader / thinking / `CausalEdge`, and 26 hits for V3 storage geometry (`facet`, `classid`, `ClassView`, `NodeRow`), almost all in doc comments; one op name, `Pred::MatchFacetStrided`.
- mask-risc `src/` contains no `f32`/`f64`: the IR is integer-only, so executor/oracle `==` is an honest claim.
- `lgj-abi` (Java/Panama) already depends on mask-risc; quack is its dev-dependency only (D-QCK-10).
- `CausalEdge64` is `#[repr(transparent)]` over `u64`; the contract's `MailboxSoaView::edges_raw() -> &[u64]` is the zero-copy view. Its layout is feature-gated (`causal-edge-v2-layout`).
- The six proposed verbs reduce to three existing families: MASK, TRANSPORT (`Gather`, `ScatterOrU32`, `EqU32Via`, `GroupKey::Via`), REDUCE (one terminal). SELECT = mask + `Keep`; an index-compacting select is the forbidden `SelectionVector`.
- **Gap:** `tests/differential.rs` exercises `Pred::MatchU64` only with `care: 0xF000` (bits 12–15); a 32-bit truncation of a `u64` lane would pass.

## Consequence
Nothing new is minted. The ruling names the socket, its firewall and its falsifiers (G-SOCK-1..5); the next PR adds two test files to mask-risc.

## Open
- Arrow `ListArray` has no zero-copy route (no segment view): `ISS-MASK-RISC-HAS-NO-SEGMENT-VIEW`.
- Multi-hop traversal is not in the IR; `ISS-NO-MASK-HOP-OP` predates `Gather`/`ScatterOrU32` and was not re-read against them here.
- Whether "Quack" should name the socket (today it names only the surface above it).
