# Ephemeral → Warm → Cold — the lifecycle / codebook / hydration doctrine (v1, 2026-06-01)

> **READ BY:** anyone wiring the `I4x32D` carrier (A3/A4), the connectome store
> (WD-3/C7), the codebook/palette, the ractor hydration seam, or the GoBD audit.
> This pins the format + reasoning model the 5-dev council + 3× brutal red-team
> converged on, plus jan's design-dialogue resolution (2026-06-01). It is the
> authoritative context for the A3 research council.
>
> **Companion docs:** `.claude/specs/atoms-styles-nal-planner-dto-unification-v1.md`
> (§0–14, the design), `.claude/plans/north-star-integration-v1.md` (the WD
> resolution + the 3× brutal red-team), `.claude/north-star/README.md` (the two
> ViewAngle diagrams).

---

## 0. The premise correction (what the brutal red-team caught, and how jan resolved it)

The 3× brutally-honest council (B1/B3) flagged the ratified **G-CODEBOOK** invariant
as a fiction: the DeepNSM proposer lives in **4096-space** (COCA vocab ranks), the
`causal-edge` resolver in **256-space** (8-bit palette), no projection connects them,
and a 4096→256 fold is **16:1** — manufacturing false syllogistic middle terms
(`o1 == s2` on quantized indices ≈ a fallacy of four terms) that the "small margin →
escalate" guard cannot catch (the margin is *large* precisely when the counterfeit is
confident).

**That whole objection rested on a wrong premise: that the 256-palette fold sits on the
reasoning path. It does not.** jan's resolution, verbatim in spirit:

> *"We don't need 256 for temporal formats. As long as data is ephemeral we can afford
> O(1) or 4096… grammar resolution is ephemeral… we don't need to introduce aliasing in
> the sentence chaining AT ALL."*

**The 256 palette is a cold-storage codec. Reasoning never folds.** This document is the
corrected model.

---

## 1. The three-tier lifecycle (the spine of everything)

| Tier | Form | Width | Reasoning op | Lives where |
|---|---|---|---|---|
| **HOT / ephemeral** | exact term IDs | **16-bit / 64k, O(1)** | **composition** (`a∘b`), exact | grammar resolution / sentence chaining; sparse (~4096 basins / 32k sentences) |
| **WARM / implicit** | SoA bitmap columns (Quartett masks) | **256-bit member masks** | edges **hydrate** via deterministic multiplex-Louvain over sem/syn/prag links | BindSpace SoA; never stored as explicit edges |
| **COLD / explicit** | vart radix **context loci** (Minecraft palette) | **16 families × 256**, sparse cross | **set-op selection** (Quartett); only collapse-gate-committed facts calcify | OGIT vart radix; O(1), versioned, immutable |

**Vertical gradient — the one-line architecture:**

> **hot** (plastic, 16-bit exact composition) → **warm** (SoA bitmap, Louvain-hydrated
> edges) → **cold** (frozen, O(1) radix context loci).

Things reason fluidly at the top and **sediment** truth to the bottom (`DemotionSink`,
plasticity `ALL_HOT → frozen`). Same firewall top to bottom; one substrate (vart) holds
the cold loci.

---

## 2. HOT — sentence chaining, exact, zero aliasing

- Term identity = **16-bit (64k), O(1) direct index** — generous headroom over the 4096
  COCA vocab (room for compounds, OGIT entities, instances). Exact integer equality:
  `o1 == s2` compares full term IDs → **zero aliasing, by construction.**
- **Composition (`a∘b`) happens HERE** (ephemeral, exact) — this is where the syllogism's
  term-sharing + the NARS truth math run on *full-resolution* identities. The DeepNSM
  `SpoTriple` (12-bit / 4096 vocab) feeds this directly; **no fold.**
- This is the **syntax + semantics** layer (the *semantic ViewAngle*: parse & roles → core
  NSM meaning → concept lattice). Syntax and semantics fuse here per **The Click**
  (parsing = meaning = one VSA op).
- **Width-coincidence hazard (newtype it in A3):** the 16-bit term space, `ClassId` (u16),
  `MetaFilter.thinking_mask` (64-style bitset), `class_view::FieldMask` (64-field
  presence), and the `I4x32D` atom-lanes (64) all coincide in width but **NOT** in meaning.
  Give them nominal newtypes so cross-wiring is a *compile error* (the
  `I-LEGACY-API-FEATURE-GATED` 5×-bug pattern lives exactly here).

## 3. WARM — edges hydrate from the SoA (AGI-as-SoA, made literal)

- **Edges are not stored — they are the runtime behavior of the SoA under dispatch**
  (CLAUDE.md P0, AGI-as-glove). An "edge" = AND two property-masks; a community falls out.
  Nothing persists until it earns it.
- Basins emerge as **communities** via **deterministic multiplex-Louvain** over three
  link-layers = the **3-split**: **semantic** (meaning similarity), **syntactic** (SPO
  structure), **pragmatic** (OGIT-stake). One community structure resolved across all three
  = the `head2head` dual-view, realized as graph community detection.
- **Quartett bitmap index** (the cold/warm representation jan pinned):
  - per family, **one 256-bit mask per property**: `mask[property_x]` = which of the 256
    members have property X. "Which dogs have X" = a 32-byte read.
  - **continuous attributes are bucketed** (km/h → ordinal buckets, one mask each) — same
    binning discipline as the Pearl bands / σ-ladders / i4 rungs.
  - relations = **set-ops**: overlap/term-sharing = `a & b` + `popcount`; distance =
    Hamming/Jaccard. `FieldMask` (class *has* property) is the **transpose** of the
    member-mask (members *have* property).
- **Determinism pin (GoBD):** classic Louvain is order/resolution-dependent. Hydration MUST
  be deterministic — **fixed visit order** (the stable SoA index), **fixed resolution γ**,
  and **local/incremental** (active neighborhood, never a global re-cluster).

## 4. COLD — sedimentation into O(1) context loci

- **Fixed** = collapse-gate-committed (F below floor / named / cited) **+** plasticity-frozen
  (`is_frozen()`, the `ALL_HOT → frozen` descent). The `DemotionSink` pushes it down → it
  **calcifies into the cold OGIT vart radix.**
- The radix path **is** its address: `FixedSizeKey` lookup = O(key-length) = **O(1)** for
  fixed-width keys; vart MVCC stamps it at a version (immutable, append-only).
- **vart adaptive radix = the Minecraft palette:** `Node4 → 16 → 48 → 256` sizes to
  occupancy; sparse families ride small nodes, dense families ride `Node256`. **Per-OGIT
  partition** — the key is **OGIT-class-first, then term** (= the EW64 family-first keying;
  this *resolves* the brutal council's S-first-vs-family-first conflict).
- Structure: **16 families × 256 archetypes** (= 4096, the basin space); **dense within-family**
  (the shipped `network.rs` 256×256 *only if you keep dense*; jan's smart move replaces it
  with the **256-bit member bitmask + set-ops**, ~70× cheaper); **sparse cross-family** —
  each OGIT class points to **3–16** cross-family basins, hardcoded from the class + the
  **OGIT mask** (the `FieldMask` / Picture-1 "Maskierung" doing double duty).
- Each settled **context locus is four things at once:**
  1. **The Click's `global_context`** — the prior reshaping the next cycle's F-landscape;
     warm hydration resolves *toward* the loci.
  2. **The firewall's CAM address** — propose/hydrate above, address a fixed locus below.
  3. **The GoBD anchor** — immutable + versioned = *Unveränderbarkeit*; replay = re-derive
     hot/warm + read the loci (you never recompute what calcified).
  4. **The rung-addressable awareness chain (A2)** — `get_awareness(r=n)` jumps to the locus
     at altitude `r`; the radix is keyed by term *and* indexable by rung.

---

## 5. The firewall, top to bottom (the one invariant)

**similarity PROPOSES** (DeepNSM float / the warm Louvain hydration / the semantic-differential
splat) → **CAM ADDRESSES** (the 16-bit exact term, the 256-bit member-mask set-ops, the cold
radix locus). Integer LE everywhere below the membrane; float/language only *above* it
(upstream, ephemeral). **Determinism = replay = GoBD**, achieved by: exact hot composition +
deterministic warm hydration + immutable versioned cold loci.

## 6. The 3-split ↔ the dual grammar (why two ViewAngles, not three)

The classic semiotic trichotomy maps onto the architecture:
- **Syntax** (Grammatik: structure/well-formedness) = term identity / the SPO parse skeleton.
- **Semantics** (Bedeutung: NSM/DeepNSM) = meaning bound into roles.
- **Pragmatics** (Business Grammatik: roles/context/**goals**) = OGIT O/G/I/T stakes.

The dual grammar collapses syntax+semantics into the **semantic ViewAngle** (per The Click,
parse = meaning = one op) and renames pragmatics the **business ViewAngle**. `head2head` =
aligning meaning ↔ stakes = the *(syntax+semantics) ↔ pragmatics* resolution.

## 7. G-CODEBOOK, re-scoped (SUPERSEDES the 2026-06-01 ratification)

NOT "one 256 codebook." Term-identity has **lifecycle formats**: hot **16-bit exact** / warm
**256-bit member-masks** / cold **16×256-per-family radix**. The "4096→256" is the
**commit/eviction codec** (off the reasoning path). OGIT classes (pragmatics) are a *separate*
addressing layer. The SPO term-palette is shared per-family, so cold archetype-level `o1==s2`
= "same archetype" (the intended aggregate granularity); **live chaining never folds.**

## 8. Cost

- Warm: 256-bit mask = 32 bytes/basin; ~4096 basins ≈ **128 KB** (vs ~9 MB dense 256×256).
  ~70× cheaper, adaptive (sparse basins ≈ free), Minecraft-cheap.
- Cold: bounded property count × 32 bytes/family — KBs.

## 9. What this means for the slices

- **A3 (the carrier) is unaffected by all of §1–8** — it is the `I4x32D` carrier + `pack/unpack`,
  the dependency root, fully offline. Its real footprint is the **2 `todo!()`s in `atoms.rs`**
  (the brutal council confirmed `recipe.rs` is an uncompiled orphan, NOT unblocked by A3). A3
  ratifies the open carrier sub-decisions: **sign/scale per AtomGroup**, the **32-vs-33 lane**
  fold, and **newtyping the width-coincidences** (§2).
- **A4** (resolver → `FieldMask` attention) is the propose(i4-distance)→address(`ClassView`) seam.
- The **warm Louvain hydration** + the **cold vart-radix Quartett loci** + the **`DemotionSink`
  sedimentation** are downstream slices (post-A4), but A3/A4 must not contradict this model.
- **Remaining concrete must-fixes** the brutal council surfaced (independent of the codebook):
  `recipe.rs` orphan (don't claim A3 unblocks it); `ractor`'s `bon` dep missing from the offline
  cache (C6 is "offline after one fetch" or `[patch]` to `/home/user/ractor`);
  `PlanStep`→`CollapseGateEmission` (the real baton type); the replay **timestamp** leak (the
  audit merkle root hashes `SystemTime::now()` — make it a captured input).

---

## Clarification (jan, 2026-06-01) — the carrier is a CAM address, not a similarity vector

**No vector search.** `I4-32D`/`I4-64D` is a deterministic **N×CAM address** (128/256-bit) whose sparse non-zero **signed** dims are the intensity "smell". `D` = signed **Dimensions** (32 → 64 poles; 64 → 128 poles); each dim is a bipolar axis (sign = pole, e.g. −introspection..+exploration). There is **no `{instance, reference}` dual** ("64" was 64 poles, not lanes). The ONLY fuzzy step in the whole stack is a coarse "this smells like odoo → financial OGIT" route; everything else — including the A4 resolver — is **CAM addressing**, not i4-distance nearest-template search. The hot path is integer CAM addressing end to end; float/similarity is the coarse upstream smell only.

**Range:** the carrier stores signed i4 `[−8,7]` (two's-complement, byte-compatible with the i4 substrate). Any asymmetric bipolar mapping (`−7..+8` style) is the **caller's pre-scale**, never the carrier's storage.

A3 shipped this carrier (`I4x32`/`I4x64`); see `.claude/plans/a3-carrier-v1.md` § SHIPPED.

## Correction (jan, 2026-06-01) — no f32 round-trip; texture → style in ~4 CPU cycles

The carrier→style path is **integer end to end**: **no f32 round-trip** anywhere (no caller-side f32→i4 pre-scale on the hot path). The i4 texture arrives as signed bytes and stays integer; **texture → thinking style is the fastest route, ~4 CPU cycles** (a branchless integer transform / CAM address → style), never a float compute or vector search. Any asymmetric bipolar pole mapping lives in the i4 encoding (sign + magnitude), not f32. **This supersedes the earlier "caller pre-scales f32→i4" phrasing** (§2/§9): f32 is not on the texture→style route at all.
