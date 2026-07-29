# copy-thinking — `derive(*, Copy)` verdicts for the 6 cognition/compute crates

**Agent:** zero-copy warden (thinking / learning / jc / helix / perturbation-sim / codec-research)
**Branch:** `claude/x265-x266-plans-review-h9osnl`
**Mode:** EDIT ONLY. No `cargo build`/`check`/`test`/`clippy` run; no worktree created.
**Read first:** `AGENT_LOG.md`, `.claude/knowledge/zero-copy-lens-law.md`,
`.claude/rules/data-flow.md` §2, `.claude/rules/borrow-strategy.md`.

> **Mandatory-read path correction.** `data-flow.md` and `borrow-strategy.md` do
> NOT exist under `lance-graph/.claude/rules/` — that directory is absent. The
> canonical copies are `/home/user/ndarray/.claude/rules/data-flow.md` and
> `/home/user/q2/.claude/rules/borrow-strategy.md`, both auto-loaded into session
> context via the workspace CLAUDE.md chain. I read them there. The census file
> and the `copy-tierA` tag both cite the non-existent lance-graph path; worth
> fixing at the source so the next worker does not go looking.

---

## Headline — 0 violations in 123 sites. Nothing removed, nothing refused.

**No type in any of the six crates holds a borrow, and no type in any of the six
crates reads SoA lane bytes at all.** The second fact is what makes the first
structural rather than lucky: you cannot store a second projection of lane bytes
if you never touch a lane. Mechanical confirmation —

```
grep -rl 'NodeRow|SoaEnvelope|from_register_ref|CausalWitnessFacet|
          value_offset|NODE_ROW_STRIDE|ValueTenant'  <the 6 crates>
→ crates/perturbation-sim/src/columns.rs   (one DOC-COMMENT occurrence, line 107,
                                            the prose word "ValueTenant"; no code)
```

So the violation shape the law names — *a struct that holds what a lane holds,
stored beside the lane* — has no instance here. These crates compute over owned
values and hand owned values back.

## The census undercounts my scope by 43 %

`copy-derive-blast-radius.txt` lists **~70** sites for these six crates. The real
count is **123**. The census misses every `#[derive(Debug, Clone, Copy, …)]`
where `Debug` comes first — e.g. all 25 `perturbation-sim/src` sites, all 8
`helix/src` sites, `learning/src/scm.rs:31`, `learning/src/cognitive_styles.rs`
×3, `thinking-engine/src/{reranker_lens,inference_backend,silu_correction}.rs`.
The census header claims *"This census matches both orderings"* — it does not.
**The 369 global total is therefore a floor, not a count.** None of the 53
missing sites changed a verdict (all are value types), but the next sweep should
re-run with `derive\([^)]*\bCopy\b` rather than a two-ordering literal match.

Per-crate: thinking-engine 34 · learning 33 · jc 14 · helix 14 ·
perturbation-sim 27 · lance-graph-codec-research 1 = **123**.

## Method

Two passes, because neither alone is sufficient:

1. **Mechanical** — parsed all 123 declaration bodies (brace-balanced, comments
   stripped) and searched field positions for a lifetime parameter or any `&`.
   Result: **2 candidates, both `&'static str`, 0 lifetime parameters.**
2. **By hand** — read the declaration + surrounding impl for ~85 sites
   individually, including every site you named to watch, plus the producer
   function wherever a type looked like it might be a materialization
   (`BufferResidue`, `FastBusDto`, `CascadeChannels8`, `PathResult`).

The sharper heuristic `copy-tierA` proposed (flag on a **declared lifetime
parameter only**) reproduces my result exactly on this corpus: 0 hits, 0
violations, 0 misses.

---

## The two `&` sites — LEGITIMATE, and they are the census's own false-positive mechanism

| path:line | type | VERDICT | reason |
|---|---|---|---|
| `crates/perturbation-sim/src/columns.rs:58` | `SoaMemberSpec` | LEGITIMATE | `name: &'static str` + 2×`u32` + enum + 2×`bool`. Backs the `const CONTINGENCY_FACTORS: [SoaMemberSpec; 5]` and `const INERTIA` spec tables. |
| `crates/perturbation-sim/src/columns.rs:136` | `InertiaPromotion` | LEGITIMATE | `member` + `signoff` are `&'static str`; backs `const INERTIA_PROMOTION`. |

`&'static str` is a pointer into `.rodata` that outlives every mailbox, so there
is no compartment for it to escape *from* — the same ruling `copy-tierA` reached
independently on 8 sites. These are const **descriptor rows** (widths, encodings,
sign-off provenance), not data. Note the irony worth recording: `SoaMemberSpec`
is the type that *describes* SoA value tenants, and it is the only thing in these
six crates the borrow-heuristic could latch onto — it names lanes, it never
reads them.

## LEGITIMATE families — stated once each, per your instruction

| family | n | why, once |
|---|---|---|
| **`learning/src/cam_ops.rs` op enums** (`OpCategory`, `LanceOp`, `SqlOp`, `CypherOp`, `HammingOp`, `NarsOp`, `FilesystemOp`, `CrystalOp`, `NsmOp`, `ActrOp`, `RlOp`, `CausalOp`, `QualiaOp`, `RungOp`, `MetaOp`, `VerbOp`, `MemoryOp`, `UserOp`, `LearnOp`) | 19 | **Read, not assumed** — I brace-scanned all 19 full bodies (18–131 lines each) for variants carrying a payload. **Zero payload variants across all 19**: every one is C-like with explicit `= 0xNNN` discriminants. An op code is an address, and a discriminant is a value. |
| **Fieldless dispatch/state enums** — `GateState`, `Viscosity`, `Archetype`, `GestaltRole`, `GestaltState`, `GhostType`, `PersonaMode`, `SourceType`, `ThinkingScale`, `QuorumLevel`, `ModelId`, `TableType`, `ThinkingPreset`, `CognitiveOpKind`, `CognitiveAuthResult`, `BackendGrade`, `GatePolicy`, `NarsCopula`, `ActrBuffer`, `CausalRelation`, `QualiaChannel`, `Rung`, `NarsInferenceType`, `CausalEdgeType`, `Operator`, `Atom`, `StyleOrigin`, `FloorBand`, `BusKind`, `DataLevel`, `Regime`, `Encoding`, `GuardrailVerdict`, `IccForm`, `AudioQualia`, `Sign`, `Verdict`, `Mode`, `Kind`, `PBasis`, `PQuant`, `PMode`, `PRank` | ~43 | Nullary variants. Nothing to borrow; `Copy` is the identity function on a discriminant. |
| **Small scalar records** — `StyleParams`, `SelfState`, `UserState`, `FieldState`, `HdrResonance`, `Temperature`, `SpiralSegment`, `SplatField`, `CrossModelResult`, `TruthValue`(×2), `Cx`, `AcBus`, `AcLine`, `CascadeConfig`, `Resilience`, `Splat`, `Edge`, `Yield`, `MetaHop`, `RollingFloor`, `ContingencyFeatures`, `Stats`, `Slab`, `Region`, `Sprite`(×2), `SpriteParams`, `HemispherePoint`, `Similarity`, `CurveRuler`, `ResidueEdge`, `Signed360`, `CascadeKey`, `CascadeKeyV3`, `IsaPath`, `HhtlKey`, `InertiaProvenance`, `Mat2`(×3), `Sym2`, `NounF`, `Pron`, `Lane` | ~53 | `f32`/`f64`/`u8`/`u16`/`u64`/small fixed arrays. **This is `data-flow.md` §2 verbatim** — reasoning = owned `Copy` microcopies, stack-allocated, no heap, no lifetime tracking. Removing `Copy` here breaks the rule in the *other* direction. |

### `learning/src/cognitive_frameworks.rs:18 TruthValue` — MUST STAY `Copy`, explicitly

You asked me to say this rather than skip it. `TruthValue { f: f32, c: f32 }` — 8
bytes, owned, no borrow. It is **named in `data-flow.md` §2 as the canonical
reasoning microcopy**, and `borrow-strategy.md` uses it as the worked example of
the required pattern (`let mut local_truth = hit.record.truth;` → revise on the
owned copy → gated write-back). Stripping `Copy` would force reasoning paths onto
`&mut` during computation, which is the P0 that rule exists to prevent. **Not
touched.** (`learning/src/scm.rs:31` declares a second, independent `TruthValue`
with fields `frequency`/`confidence` — same verdict, but flagging the duplicate
since `docs/TYPE_DUPLICATION_MAP.md` does not list it.)

---

## The four you named to watch

**`thinking-engine/src/contract_bridge.rs:155 FastBusDto` — LEGITIMATE. A value.**
Your question was the right one to ask. `#[repr(C)]`, a `SIZE` const, a `≤24 B`
test — the author is clearly thinking in bytes. But it is not a record *of bytes
that already have a lens*: `from_thought(…)` takes computed `f32`s from the
cascade and **quantizes them** (`dissonance * 255.0 as u8`, `top3` from a slice,
`gate` from a match). The inputs are transient computation outputs, not lane
bytes, and the DTO is the only stored form — it lives as `A2APayload::Thought(FastBusDto)`
on an `A2AMessage`. It crosses an **agent** boundary, not a tenant boundary, and
your own rule (*"borrows are only for the same mailbox"*) is exactly why an owned
value is the correct shape there: removing `Copy` pushes toward a reference, the
forbidden direction. **Standing watch, not a finding:** if a SoA lane ever holds
these same 12 fields, `FastBusDto` becomes the second projection that day.

**`thinking-engine/src/layered.rs:45 CascadeChannels8` — LEGITIMATE.**
`pub struct CascadeChannels8(pub u64)` — a newtype over one `u64`, 8 signed byte
channels read by shift+mask. `u64` is in `data-flow.md` §2's list by name. It is
not a second projection: it is the **accumulator during L1→L2→L3 propagation**,
and it is *transcoded* into `causal_edge::CausalEdge64` at the L3 commit boundary
(the impl block says so), never stored beside it. Converted, not duplicated. The
one field holding it (`domino.rs:90 pub edge: CascadeChannels8`) owns it outright
— there is no lane backing those bytes.

**`jc/src/ewa_sandwich.rs:104 Spd2` — LEGITIMATE, and `Copy` is load-bearing.**
3×`f64` = 24 B. The propagation loop is `sigma = sandwich(&m, &sigma)` — a
**value assignment**, which is precisely `data-flow.md`'s *"engines return
results; they do not mutate themselves while computing"*. Removing `Copy` forces
in-place `&mut` mutation of `sigma`, i.e. the P0 in the other direction. Same
verdict for the identical `Spd2` in `koestenberger.rs:99`, `Spd3` in
`ewa_sandwich_3d.rs:99`, `Sym2` in `sigma_codebook_probe.rs:77`, and the three
`Mat2` copies in jc examples.

**`jc/src/ewa_sandwich.rs:232 PathResult` — LEGITIMATE, with an elevation note.**
`{ final_sigma: Spd2, log_norm_sq: f64, psd_hops: usize }`. `final_sigma` is a
member (same rung as its inputs); `log_norm_sq` and `psd_hops` are **facts about
the set of hops, not members of it** — the Gadamer-refinement shape that *would*
be elevation-eligible. But the rung test is not triggered, because **nothing here
is stored**: `propagate_path` returns it to a local. Same for
`ewa_sandwich_3d.rs:384`.

## ELEVATED — I am claiming this verdict for nothing, and here is why

The rubric licenses a store. Nothing in these six crates writes a tenant, so no
site can earn it. Two sites are **elevation-*shaped*** and I name them so the
question is on the record before someone stores them:

- `jc/src/ewa_sandwich.rs:232` / `_3d.rs:384` `PathResult.{log_norm_sq, psd_hops}`
  — set-facts over a hop sequence.
- `perturbation-sim/src/place_buffer.rs:81 BufferResidue { lanes: [u16; 8] }` —
  the strongest candidate. `buffer_residue()` computes effective resistance from
  the Laplacian pseudo-inverse to **every** other bus, sorts, takes the 8 nearest,
  quantizes to BF16. That is a computation across many reads yielding a value of a
  different KIND (a node's neighbourhood permeability, which lives in no single
  coupling) — not reproducible by a cast. Today it is returned, not stored. The
  module doc already describes it as *"the BF16 residue … a helix-residue value
  slot on the HHTL-OGAR key"*, and `columns.rs:136 INERTIA_PROMOTION` records a
  `RatifiedReuse` verdict promoting it into a `ResidueEdge` slot. **So the store
  is planned.** When it lands it needs the rung named explicitly, per the law's
  one-exception test — it should not inherit "it was already a `Copy` struct" as
  its licence.

---

## Changes: none. Cascades refused: none.

No derive removed, no file edited in the six crates. The only artifact of this run
is this tag file. Nothing to compile beyond what you were already going to build.

## Honest limits

- Verdicts are from reading declarations, impls, and (for the watched types)
  producers and call sites — **not from a compiler**. Since I changed nothing,
  there is no build risk either way.
- The mechanical borrow scan reads field *positions* after stripping `//`
  comments. A borrow hidden behind a type alias (`type Foo = &'a [u8];`) would
  evade it. I grepped for alias declarations in the six crates and found none in
  the 123 bodies, but I did not resolve aliases transitively.
- I judged 123 sites in six crates. The census's own count for these crates was
  wrong by 53; **the other crates' Tier-B counts are likely wrong the same way**
  and should be re-derived before anyone reports "369 audited".
