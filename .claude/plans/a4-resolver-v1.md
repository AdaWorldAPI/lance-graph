# A4 — the resolver: texture → thinking style (~4-cycle integer CAM) (plan v1, 2026-06-01)

> Slice **A4** of the north-star run, on merged `main` (A3 carrier shipped, PR #451).
> Branch `claude/jolly-cori-clnf9`. Doctrine: **plan → 5 savant-dev review → split →
> per-file sprint (commented-out) → 3 brutal-tester review → fix → uncomment → PR.**
> Context: `.claude/knowledge/ephemeral-warm-cold-lifecycle.md`, spec §9–10, the A3
> SHIPPED carrier (`atoms::I4x32`/`I4x64`). **This is the DRAFT for the 5 savant
> developers to red-pen before any code.**

## The one-line job

**Resolve a texture to a thinking style as the fastest integer route — `texture → StyleId`
in ~4 CPU cycles, deterministic CAM addressing, NO f32, NO vector search.** The only fuzzy
step in the whole stack is a coarse upstream *"this smells like odoo → financial OGIT"*
route; everything here is integer below the membrane.

## Scope — A4a (offline, contract-only) vs A4b (cross-crate, deferred)

**A4a (this slice — `lance-graph-contract`, zero-dep, offline, regression-safe):**
1. **`AtomLane(u8)`** — a carrier-lane in `0..64` (NOT a 33-catalogue ordinal — B3 BLOCKER-1).
   `atom(lane: AtomLane) -> Option<&Atom>` becomes a *partial* lookup (Some for 0..32, None
   for the 31 spare). `repr(transparent)`, const-constructible, `PartialEq`.
2. **`LaneMask(u64)`** — a 64-lane attention/active-set mask (NOT a bare `u64` — B3 SERIOUS-4;
   must not collide with `FieldMask`/`thinking_mask`). `has(AtomLane)`/`with(AtomLane)`/
   `count()`/`&`/`|`, mirroring `FieldMask`.
3. **`AtomGroup::is_signed()`** — Pearl/Rung/Sigma → false; Operation/Presence/Meta → true.
   Documents the caller's pre-scale convention (the carrier stays sign-agnostic).
4. **The resolver skeleton** — `StyleResolver` (trait or fn): `resolve(&self, texture: &I4x32)
   -> StyleId` (+ optional `LaneMask` attention return). A **deterministic integer reference
   impl** (the ~4-cycle CAM mechanism — see Open Questions), with the actual OGIT/recipe
   codebook stubbed/minimal so A4a stays zero-dep + offline.
5. Tests (round-trip/guard/isolation, mirror A3's hardened set) + board hygiene.

**A4b (deferred — cross-crate, gated):** the real OGIT-class CAM index, `StyleId`/`StyleMask`/
`FieldId` newtypes touching `cognitive-shader-driver` (≥12 sites) + `lance-graph-ontology`,
the `FieldMask`-attention seam, and the coarse smell-classifier. NOT in A4a.

## Deliverables (A4a)

- **D-A4-1** — `AtomLane(u8)` (0..64) + gate `atom()`; const-constructible, `PartialEq`.
- **D-A4-2** — `LaneMask(u64)` (`has`/`with`/`count`/set-ops), the firewall wall vs `FieldMask`.
- **D-A4-3** — `AtomGroup::is_signed()`.
- **D-A4-4** — `StyleResolver` signature + a deterministic integer reference impl (the ~4-cycle
  texture→`StyleId` CAM lookup; no f32, no search).
- **D-A4-5** — tests (the existing 562 stay green) + board (`PR_ARC` #451 merged + this plan).

## Firewall / determinism (the invariants A4a must hold)

- **No f32, no vector search** anywhere on the texture→style path. The i4-distance "PROPOSE"
  from the spec is *demoted* to the coarse smell-route (A4b, upstream) — A4a's resolver is
  deterministic CAM addressing.
- **~4 CPU cycles**: branchless integer (a few shifts/masks/popcount + a load). Texture is the
  address; style is the addressed value.
- **The carrier QUERIES, never CONTAINS** the address side: `resolve(texture) -> StyleId`,
  never an i4 lane holding a `StyleId`/`u16`. (WATCH-2.)
- **Newtype the width-coincidences**: `AtomLane`/`LaneMask` so the 64-wide spaces (atom-lanes,
  `thinking_mask`, `FieldMask`) cannot cross-wire (compile error).

## Open questions for the 5 savant developers (red-pen these)

1. **The concrete ~4-cycle CAM mechanism.** How does `texture(I4x32) → StyleId` hit ~4 cycles
   *deterministically, integer-only*? Direct index off a few active lanes? A popcount/perfect-
   hash of the `LaneMask`? A small `[StyleId; N]` table keyed by a packed lane signature? Pin
   the exact lookup — this is the heart of A4.
2. **The texture→style codebook source in A4a.** Offline stub (a minimal fixed table) vs the
   OGIT/recipe codebook (A4b-gated). What's the smallest honest reference impl that's testable
   offline without painting A4b into a corner?
3. **`AtomLane` shape** — 0..64 carrier-lane with partial `atom()`; confirm it doesn't break
   the locked-33 catalogue test, and the const-literal ripple is contained (B2 lesson).
4. **`LaneMask` now or A4b?** B3 wanted it born at the carrier root (A4a) so A4b's attention
   return isn't a bare `u64`. Confirm it belongs in A4a + its API.
5. **Where does the resolver live** — `atoms.rs`, a new `resolver.rs`, or `recipe.rs` (the
   orphan)? Zero-dep + offline is the constraint.
6. **`StyleId`** — A4a needs *a* style id type for the resolver return. Is it `ThinkingStyle`
   (the existing 36-enum) directly, or a new `StyleId(u8)`? (R3 deferred `StyleId` to A4 — is
   A4a or A4b its home?)

## Out of scope (do NOT touch in A4a)

`recipe.rs` wiring (orphan, A3.5), `jit::StyleRegistry`, `cognitive-shader-driver`, `vart`/cold,
the warm Louvain, `ractor`, `counterfactual.rs`. The 36-style `ThinkingStyle` enum stays as-is.

## Offline + baseline

`cargo test -p lance-graph-contract --offline` (562 green on merged main). A4a = integer
nibble/mask arithmetic, zero new deps. Gate: 562 + the new A4a tests, all green.
