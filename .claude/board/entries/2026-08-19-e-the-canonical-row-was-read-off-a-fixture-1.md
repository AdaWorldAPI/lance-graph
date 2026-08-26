## 2026-08-19 — E-THE-CANONICAL-ROW-WAS-READ-OFF-A-FIXTURE-1

**Status:** CORRECTION (operator ruling + sweep evidence). Affects
`.claude/plans/hhtl-thinking-tables-le-contract-v1.md` F1, corrected in
place.

**What happened:** ARC-B's F1 froze *"ONE canonical substrate: the 32×(4+12)
facet register"* and cited `lance-graph-java/native/lgj-abi/src/abi.rs:173-175`
as its shipped shape. That citation is a **fixture, and it says so itself** —
`rowstore.rs:5-8,33-39`: *"The Java side may lay its view out differently…
The 64-byte-aligned guarantee arrives with the real `NodeRow`
(`#[repr(C, align(64))]`) wiring, not here."* Grep confirms **zero import of
`NodeRow`/`EdgeBlock`/`NodeGuid`** anywhere in that code path; the only
`canonical_node` import is `EdgeCodecFlavor`, for a trait-default test.
Meanwhile `docs/abi.md:433-438` §10 already names the real target —
*"`NodeRow`'s `16|16|480` … is already a legal lane description"* — but §11
(2026-08-17) and §12 (2026-08-18) shipped the homogeneous 32-lane fixture
and §10 was never revised, so the doc contradicts itself in reading order.

**The lesson, generalizable:** a *conformance target* was mistaken for a
*source of truth* because it was the shape that had shipped most recently
and was easiest to grep. Canon lives in the contract crate; a consumer's
current geometry is evidence about the consumer, never about canon. The same
inversion is what put integration ownership inside the DisMech oracle
(see the PR #7 ownership correction).

**Also corrected the same day:** an earlier session claim that `ogar-elk` is
*"structurally prevented from producing an entailment"* is **false**. It
computes real EL-subsumption closure in-repo, in pure Rust
(`OGAR/crates/ogar-elk/src/lib.rs:163-166,239-367` — `entails`,
`equivalence_cycles`, `merge`; R1/R2/R3 only, existentials and role
composition deliberately excluded at `:70-81`). What it forbids is
*serializing* the verdict — *"An observer that could serialize its verdict
would invite someone to ship the verdict as if it were substrate"*
(`:42-45`). Entailment production is local and real; persistence as
substrate is what is banned.

