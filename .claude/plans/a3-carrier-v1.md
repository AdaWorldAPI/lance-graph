# A3 — the `I4x32D` carrier + `pack`/`unpack` (plan v1, 2026-06-01)

> Slice **A3** of the north-star run — the dependency root. Synthesizes the 5-agent
> research council (R1–R5, 2026-06-01). Context:
> `.claude/knowledge/ephemeral-warm-cold-lifecycle.md`, spec §8–11. Branch
> `claude/jolly-cori-clnf9`. **Next:** 3× brutally-honest red-team → fix → execute.

## Scope (one line)

Implement `I4x32::pack`/`unpack` (the two `todo!()`s) **+** add `I4x32D` (dual carrier)
**+** `AtomGroup::is_signed()` **+** the `AtomLane(u8)` newtype **+** delete the 4 stale
`BLOCKED:` notes **+** tests — all in `lance-graph-contract::atoms` (plus one field flip in
`counterfactual.rs`). **Zero new deps, offline, regression-safe** (553-test baseline).

## Research synthesis (R1–R5)

- **Carrier (R1/R5):** `I4x32D = { instance: I4x32, reference: I4x32 }` — dual, 64 i4 lanes,
  `repr(C, align(16))`, **32 bytes**. `instance` = per-step weighting; `reference` = the
  OGIT-class-inherited template; *adjacency between planes = A4's resolution* (so A3 must
  ship the dual, or A4 has nothing to be adjacent to — R5 FLAG-1). `pack/unpack` operate
  **per plane**.
- **Algorithm (R2):** the shipped `QualiaI4_16D` (`qualia.rs:173-263`) is the canonical
  sibling. **Two's-complement nibble** (NOT offset-binary — 3 confirmations: qualia get/set,
  `edge.rs` mantissa, `nars.rs` from_mantissa). **Even lane → low nibble, odd → high**
  (byte `k` holds lanes `2k`,`2k+1`); sign-extend `(x<<4)>>4`; saturate `clamp(-8,7)`.
  `pack/unpack` are **sign-agnostic** (take a pre-scaled `&[i8;32]`, saturate identically) →
  **decoupled from the sign/scale table.** Signatures **frozen**: `pack(&[i8;32]) -> Self`,
  `unpack(&self) -> [i8;32]` (`recipe.rs:55`'s `I4x32Stub=[i8;32]` is the waiting consumer).
- **Sign/scale (R1):** add `AtomGroup::is_signed()` (precedent `high_heel.rs:1083`):
  **Pearl/Rung/Sigma → false** (unsigned ordinal `[0,7]`); **Operation/Presence/Meta → true**
  (signed `[−8,7]`; the 3 ± pairs deduce↔induce d18/19, authentic↔performance d25/26,
  exploration d31). The carrier saturates `[−8,7]` regardless; the caller pre-scales.
- **32-vs-33 (R1):** ride two halves — verbosity (d32) → plane-1 lane-0; all 33
  `CANONICAL_ATOMS` dims stay locked (`catalogue_is_locked_33_in_order` enforces `dim==index`);
  spare = lanes 33–63. Presence+Meta (4+4) canonical; "8 spare" is stale.
- **Range (R1):** `[−8,7]`, **saturating** (not `[−7,7]`, not wrapping) — `-8` is live.
- **Newtypes (R3):** **`AtomLane(u8)` in A3** (`repr(transparent)`; gate `atom(lane: AtomLane)`;
  retype `Atom.dim: AtomLane`; flip `counterfactual::SplitPoles.axis: u8 → AtomLane`) — **zero
  external callers** (only in-file site `atoms.rs:180`; `recipe.rs` refs are dead/orphan).
  Optionally stake **`TermId(u16)`** (greenfield, hot-tier identity). **DEFER to A4:**
  `StyleId`/`StyleMask`/`FieldId` (touch `cognitive-shader-driver` + ontology — ≥12 sites) and
  the `ClassId` promotion. `RungLevel`/`ThinkingStyle` are already enums — don't newtype.
- **Scope/offline (R4):** footprint = exactly the 2 `todo!()`s; `recipe.rs` is an orphan
  (no `pub mod recipe;`) — **A3 does NOT unblock it**; both `todo!()`s are **unreachable**
  (553 lib tests pass, none reach them); `cargo check/test -p lance-graph-contract --offline`
  → **exit 0** today; A3 adds no dep.
- **Firewall (R5):** carrier is integer bytes (no float); `pack/unpack` lossless+deterministic;
  the only float-adjacent quantity is the i4-distance, which only PROPOSES and lives in
  `cognitive-shader-driver`, not `atoms.rs`. Carrier *queries* the address side, never
  *contains* it (an i4 lane can't hold a u16).

## Deliverables

- **D-A3-1 — `I4x32::pack`/`unpack`** (`atoms.rs:83,88`): two's-complement nibble round-trip per
  R2 (even→low, odd→high, `(x<<4)>>4`, `clamp(-8,7)`). Optional private `const fn sext4(u8)->i8`.
  Signatures unchanged.
- **D-A3-2 — `I4x32D { instance: I4x32, reference: I4x32 }`**: `repr(C, align(16))`, 32 bytes;
  per-plane `pack`/`unpack` wrappers; doc the instance/reference semantics + "adjacency = A4".
- **D-A3-3 — `AtomGroup::is_signed()`**: Pearl/Rung/Sigma→false, Operation/Presence/Meta→true.
  Documents the caller's pre-scale convention (carrier stays sign-agnostic).
- **D-A3-4 — `AtomLane(u8)` newtype** + gate `atom()` + retype `Atom.dim` + flip
  `counterfactual::SplitPoles.axis`; (optional) `TermId(u16)`. Update `atoms.rs:196` test
  (`a.dim as usize` → `a.dim.get()`).
- **D-A3-5 — delete/replace the 4 stale `BLOCKED:` notes** (`atoms.rs:39-52`) citing spec §8–11.
- **D-A3-6 — tests** (mirror `qualia.rs` + `v2_layout_tests`): pack/unpack round-trip all lanes,
  full `[−8,7]` sweep, saturation clamp, **sign-correctness** (catches offset-binary),
  **lane-order** (even-low/odd-high), **lane-isolation matrix** (I-LEGACY-API bit-boundary),
  zero, extremes, exhaustive nibble; `I4x32D` `size_of==32`/`align==16`; `is_signed()` per group;
  `AtomLane` gating. Keep `carrier_layout_is_16_bytes`. **553 baseline must stay green.**

## Out of scope (R4 — do NOT touch)

`recipe.rs` (orphan, double-blocked on D-ATOM-1 + `StyleRegistry`), `jit::StyleRegistry`, the A4
resolver/`FieldMask` attention, `vart`/cold-radix, warm Louvain/`DemotionSink`, `ractor`/`bon`,
and the width-coincident neighbors `ClassId`/`thinking_mask`/`FieldMask` (newtype *against*, don't
redefine — `StyleId`/`StyleMask`/`FieldId` are A4). The 30+ other crate `todo!()`s.

## Offline + baseline

`cargo check -p lance-graph-contract --offline` + `cargo test -p lance-graph-contract --offline`
(green today: exit 0, **553 passed**). A3 = pure nibble arithmetic on `[u8;16]`↔`[i8;32]`, zero
new deps. Regression gate: 553 + the new A3 tests, all green.

## Open jan flags (non-blocking)

- **Rung encoding:** 9 rung lanes each `[0,7]` vs an off-by-one if one lane held rung-index 1–9 —
  a caller/A4 quantization convention, orthogonal to `pack/unpack`. One-line confirm later.
- **`TermId(u16)` now or at the hot-store slice?** Greenfield either way; staking it in A3 costs
  nothing. Recommend stake.

## Firewall / determinism

`pack/unpack` are lossless+deterministic (`unpack(pack(v)) == v.clamp(-8,7)`), pure integer — below
the membrane. `is_signed()` documents the caller's pre-scale (the propose side). `AtomLane` makes the
width-coincidence cross-wire a compile error. Nothing in A3 leaks float/language onto the hot path or
touches the address side. Clean.

---

## SHIPPED (2026-06-01) — revised per jan's clarification

**The dual is dropped.** jan: "I4-32D = 32 signed dimensions = 64 poles; focus/fan-out are opposite poles; I4-64D is also fine (256 bit)." `D` = signed **Dimensions**, not Dual; "64" was 64 **poles**, not lanes. So B3's BLOCKER-1/2 (dual semantics, per-plane vs whole-carrier, `I4x64` naming) **dissolve** — the carrier is ONE signed-dim vector.

**No vector search.** jan: the carrier is a deterministic **32×CAM address** + sparse-intensity "smell"; the only fuzzy step is a coarse "this smells like odoo → financial OGIT" route. A4 = CAM addressing, NOT i4-distance nearest-template search. No float anywhere.

**What shipped (`atoms.rs`, contract lib 562 green, offline):** `I4x32::pack`/`unpack` (two's-complement signed-i4 nibble, even→low/odd→high, `sext4`, saturate `[−8,7]`, sign-agnostic) · `I4x64` (256-bit / 64 signed dims, same codec; 33 atoms → dims 0..32, 31 spare) · `sext4` (private) · 3 BLOCKED notes → resolution-pointers · 9 hardened tests incl. the absolute-bit offset-binary catch (B1 WATCH-1).

**Range:** two's-complement `[−8,7]` (byte-compatible with `QualiaI4_16D` / the `CausalEdge64` mantissa). jan's `−7(introspection)..+8(exploration)` asymmetric mapping rides the **caller's pre-scale** (A4) — codec is sign-agnostic.

**Deferred to A4** (B3 + jan): the CAM-address resolver (no vector search), `AtomGroup::is_signed()`, the `AtomLane`/`LaneMask` newtypes (firewall guard — must NOT be bare `u64`), the bipolar catalogue reframe. NOT touched: `counterfactual.rs` (B3 SERIOUS-3), `recipe.rs` (orphan).

### Correction (jan, 2026-06-01) — no f32 round-trip; texture → style in ~4 CPU cycles

The carrier path is **integer end to end** — **no f32 round-trip** (not even a caller-side f32→i4 pre-scale on the hot path). The i4 texture arrives as signed bytes (the "smell") and stays integer; **texture → thinking style is the fastest route, ~4 CPU cycles** — a branchless integer transform (CAM address → style), never a float compute, never a vector search. The asymmetric bipolar pole (`−introspection..+exploration`) lives in the **i4 encoding itself** (sign + magnitude), not an f32 mapping. **This supersedes the earlier "caller pre-scales f32→i4" phrasing.** A4's resolver inherits the 4-cycle integer budget (CAM address → style), no f32.
