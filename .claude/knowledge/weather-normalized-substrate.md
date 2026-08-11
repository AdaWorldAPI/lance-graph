# Weather AI on the Normalized Substrate — palette256 × helix360

> **READ BY:** family-codec-smith, certification-officer, truth-architect,
> integration-lead, container-architect, v3-envelope-auditor, and ANY session
> executing or amending `.claude/plans/weather-substrate-poc-v2.md`.
>
> **Written:** 2026-08-11, after the operator-corrected helix arc (see §11 —
> the correction ledger is part of the document, not an apology appendix).
> **Companion plan:** `.claude/plans/weather-substrate-poc-v2.md` (PR #915).
>
> ## ⊘ READ §12 FIRST (2026-08-11, same day as authorship)
>
> Probe **P1 ran** and an **envelope audit ran**, and between them they
> **falsified four claims in this document** — including one where a §11
> "correction" was itself the error. Affected: **§1.2** (Fisher-Z doctrine),
> **§2.3 + §5 + §6.6** (helix360 as a wind-bearing carrier), **§11-C2** (the
> `Pair48` rejection), **§6.1/§6.5** (measurement provenance). Every one of
> those sections is superseded by **§12**. The sections are left standing,
> not deleted, so the correction is legible.
>
> **Then §12.2 itself was regraded by §12.12** (operator-directed, same day),
> once the canonical doctrine doc — `.claude/knowledge/helix-cartesian-vs-fisher2z.md`,
> which this arc never opened — was finally read. Two things a reader must not
> act on without §12.12: **"helix360" is this session's coinage for `Signed360`**
> (no such symbol exists in any repo), and the **`Pair48` mint is WITHDRAWN, not
> deferred** — one 6-byte `Signed360` is already a complete full-sphere
> direction. Read **§12.12 before §12.2**.
>
> **Grading (mandatory on every claim):**
> - `[G]` — verified in committed code this session, with `file:line`.
> - `[G-absence]` — verified NOT to exist (grep/read, stated scope).
> - `[H]` — measured this session on real data but the probe is NOT yet a
>   committed runnable example, OR a design mapping onto shipped primitives.
> - `[S]` — proposal. Do not build on an `[S]` without promoting it first.

---

## 0. Executive summary (product view)

Weather fields become **one comparable substrate** by paying the normalization
cost exactly once, at ingest:

- Every bounded scalar quantity (temperature anomaly, humidity, pressure
  anomaly, wind speed) is Fisher-Z normalized and quantized to **one u8** in a
  256-palette (`helix::RollingFloor`). Every scalar-scalar comparison
  thereafter is a **256×256 u16 table lookup — 128 KB, cache-resident, O(1),
  metric-safe** (`helix::DistanceLut`) `[G]` §3.
- Wind **orientation** is the same mechanism one rank up: **helix360**, a
  golden-spiral double-hemisphere projection (Poincaré-depth rim
  densification), stored as the **6-byte `HelixResidue` ABI lane = 2 × 24-bit
  hemispheres = inbound AND outbound bearing** `[G]` §2.
- The z-form is kept **normalized, not materialized**: variance-stabilized
  values from unlike sources land on one scale, so cross-variable correlation
  (T×q, p×wind…) becomes a legitimate LUT operation instead of a unit error.
  Geometry (hyperbolic depth, sphere coordinates, full arcs) is **hydratable
  on demand** from the stored codes — deterministic template, endpoints only
  `[G]` §1, §2.4.
- Measured this session on real ERA5 `[H]` §6: source blosc/lz4 gives 1.27×;
  Lance on 512 B rows gives 1.00× (verbatim, by design) — **the win must come
  from representation, not the storage engine**. Normalize-then-quantize is
  that win: raw-Kelvin BF16 fails (≈1.069 K error) where anomaly BF16
  succeeds (≈0.0110 K, 97×); a u8 bearing is ~8× under observational error.
- The one genuine open decision is **floor scoping** (§4): cross-variable
  comparability of palette codes is a calibration *policy* the ingest must
  set explicitly — the crate provides the primitives and the version stamps,
  and deliberately does not decide for you `[G]`.

---

## 1. The representation doctrine

### 1.1 One `arctanh` core, two readings `[G]`

`helix::fisher_z::Similarity` (`crates/helix/src/fisher_z.rs`) exposes both:

| method | formula | meaning | file:line |
|---|---|---|---|
| `fisher_z()` | `z = ½(ln(1+s) − ln(1−s)) = arctanh(s)` | **variance-stabilizing z-score** | `fisher_z.rs:55-58` |
| `hyperbolic_depth()` | `ρ = 2·arctanh(r)` | **Poincaré-disk geodesic arc length** from centre to Euclidean radius `r` | `fisher_z.rs:76-78` |

The factor 2 is the arc-length integral `∫₀ʳ 2/(1−t²) dt`; *"geometry keeps
the 2 as arc length; statistics drops it for variance stabilisation"*
(`fisher_z.rs:2-4, 60-65`). Header: *"The Poincaré rim-densified depth thus
arrives as a by-product of the Fisher-Z alignment, with NO separate hyperbolic
geometry in the hot path"* (`fisher_z.rs:1-8`). Inputs clamp to
`[−1+ε, 1−ε]`, `ε = 1e-9` (`CLAMP_EPS`, `fisher_z.rs:36`) — every f64 input
yields a finite result. The ln-form (not `f64::atanh`) mirrors the
`simd_ln_f32` hot path (`fisher_z.rs:38-43`); the f32 batch kernel is
`helix::simd::batch_fisher_z` (`simd.rs:24`), parity-tested against the
scalar reference (`simd.rs:81-95`).

### 1.2 Normalized > materialized (operator-ruled this arc)

The hot path stores the **z-form**, never the arc length. `hyperbolic_depth`
is public + tested and called from **no encode path** `[G-absence]` (grep over
`crates/helix/src`, 2026-08-11: call sites are its own tests and docs only).
This is deliberate, not an unwired seam (a mis-read corrected in §11-C3):

1. **Uniform information per bucket.** Raw bounded quantities pile density at
   their bounds; a linear quantizer wastes buckets. After `arctanh`, variance
   no longer depends on the value — *"equal steps mean equal amounts —
   stretching rim-near differences before quantisation"* (`fisher_z.rs:7-8`).
   This is what makes **8 bits sufficient**.
2. **Unlike sources land on one scale.** Raw correlation-like values cannot
   be pooled or cross-compared (value-dependent variance — a category error);
   z-values can. This is the property with no substitute: it is what makes a
   *shared* palette a common substrate rather than a compression trick.
3. **Geometry stays hydratable.** When spatial reading is wanted, `2z` and
   the sphere coordinates regenerate deterministically from the code (§2.4).
   Storage carries the low-entropy normalized form; materialization is a
   read-time projection.

⚠ **Statistical iron rule:** any *significance* claim on pooled/compared
z-values in this substrate uses **Jirak 2016 weak-dependence rates**, not the
classical `var(z)=1/(n−3)` — weather fields are spatially/temporally
autocorrelated, so effective sample size ≪ nominal
(`CLAUDE.md § I-NOISE-FLOOR-JIRAK`). The normalization itself holds
regardless; the *error bar* does not.

### 1.3 Two ranks of one mechanism

| | input | transform | stored carrier | hydrates into |
|---|---|---|---|---|
| **palette256** | bounded scalar in `[−1,1]` after mapping | fisher z | 1 × u8 (+ shared floor version) | value / similarity via 256×256 LUT — the cosine replacement |
| **helix360** | orientation | fisher-2z family (equal-area lift + rim depth) | 6 B = 2 × 24-bit signed hemispheres | spatial in/out orientation on the full sphere |

---

## 2. helix360 — the projection, the codec, the ABI lane

Crate: `crates/helix/` (standalone `[workspace]`; **mandatory** `ndarray`
dep via git per codex P2 #460 — `Cargo.toml`). Doctrine, opening line of
`lib.rs:3-6` `[G]`:

> **HHTL is the deterministic PLACE** (the trie address — *where*); **helix is
> the RESIDUE** (the orthogonal edge at that place — the hemispheric angle the
> place itself does not capture).

### 2.1 Four-stage pipeline `[G]`

1. **Placement** — "tomato-rose" equal-area hemisphere: midpoint rule
   `u = (n+0.5)/N`, `r = √u`, `y = √(1−u)`, azimuth `n·φ`
   (`placement.rs:112-118`). Equal-**area**, explicitly *"NOT hyperbolic —
   the hyperbolic-depth feeling is supplied separately by the Fisher-Z step"*
   (`placement.rs:3-5`). Rim = late/fine detail (`placement.rs:147-151`).
2. **Place coupling** — φ-spiral curve-ruler from the HHTL address:
   `CurveRuler::from_hhtl(path, depth)` → `start = (path+depth) mod 17`;
   arc index `(start + 4k) mod 17`, `gcd(4,17)=1` ⇒ full permutation
   (`curve_ruler.rs:31-56`).
3. **Fisher-Z alignment** — `z = arctanh(r)` (§1.1). Note the shipped
   encoder calls `fisher_z()`, not `hyperbolic_depth()`
   (`residue.rs:150,159`) — §1.2.
4. **Euler hand-off + quantize** — `aligned = z·STRIDE + γ·(rank/N − ln 17)`
   (order FIXED per helix `KNOWLEDGE.md` Open Item #2; `residue.rs:148-153`),
   then `RollingFloor::quantize` into the 256-palette. Encoder self-seeds its
   floor so the bulk lands in-range and the top ~1 % rim saturates —
   controlled-saturation tail (`residue.rs:130-141`).

### 2.2 Wire shapes `[G]`

**`ResidueEdge` — 3 B unsigned hemisphere** (`residue.rs:19-61`):
`[start_idx, end_idx, floor_version]`. `start_idx` = quantized PLACE anchor
(same place ⇒ same start, test `residue.rs:250-256`); `end_idx` = the
residue; `floor_version` = quantizer-calibration stamp (§3.2).
`distance_adaptive` (LUT L1) is metric-safe; `distance_heuristic`
(byte-Hamming) is a HEEL-stage pre-filter only, **never** for CAKES bounds
(`residue.rs:44-61`).

**`Signed360` — 6 B = 48 bit, the full-sphere form** (`residue.rs:63-116`),
LE wire `[rim.start, rim.end, rim.floor_version, polar, azimuth_lo,
azimuth_hi]`:

| field | bytes | content |
|---|---|---|
| `rim: ResidueEdge` | 3 | unsigned hemisphere edge (sign-independent — test `residue.rs:317-324`) |
| `polar` | 1 | signed equal-area lift: `\|y\|` in 7 bits, **hemisphere sign in the partition** — `[128,255]` upper / `[0,127]` lower |
| `azimuth` | 2 | `n·φ mod 2π` mapped onto `[0, 65536)` — full 360° |

⚠ The sign-partition is load-bearing, not stylistic: near the rim
`|y| → 0`, and a naive `128 + y·127` rounds a tiny negative lift up to 128,
which reads as positive — **the hemisphere sign is lost exactly where the
finest detail lives** (codex P2 #498; fix + regression
`residue.rs:186-195, 364-383`). Any re-implementation that "simplifies" this
encoding reintroduces the bug.

### 2.3 The ABI lane — 2 × 24-bit, in AND out `[G]`

`lance-graph-contract/src/canonical_node.rs`:

- `ValueTenant::HelixResidue = 4` (`:837-840`): *"signed full-sphere
  `Signed360`, 48-bit = 6 B (**2× the 24-bit equal-area hemisphere**; produced
  by the `helix` crate's `Signed360`, written here zero-copy)"*. Echoed at
  `placement.rs:170-171`: *"The two 24-bit hemispheres = the 48-bit
  Signed360."*
- Column descriptor (`:960-967`): `kind: ColumnKind::U8`,
  `elems_per_row: 6`, `row_offset: 112` — inside the operator-locked
  512-byte node row `key(16) | edges(16) | value(480)`. (History note kept in
  source: the lane *was* 48 B until a bits→bytes slip was right-sized
  2026-06-15.)
- Schema membership: `ValueSchema::{Compressed, Full}` carry the lane;
  `Cognitive` does not (`:1105-1141`).
- The lane is a **content-blind byte register**; `Signed360::{to,from}_bytes`
  is *one sanctioned reading* of it, "2 × 24-bit hemisphere (in/out)" is
  another — same doctrine as the V3 12-byte facet and the `EdgeBlock`
  flavor rule (never assume THE reading; resolve per class). Sibling reading
  precedent: `facet_schema.rs:13,33-34,89-93` (`FacetSchema::Pair48` names
  `helix Signed360` / `cam_pq [u8;6]` as the two shipped 6-byte codes —
  re-tiling, never re-encoding).

**Product meaning for wind:** "the wind coming from AND going to" is the
lane's native shape — inbound bearing in one hemisphere, outbound in the
other, 6 bytes, zero-copy at offset 112. Not a new codec; a projection of an
existing tenant.

`[G-absence]` **No shipped pair-writer yet:** `ResidueEncoder::encode_signed`
emits ONE signed point (`residue.rs:182-204`); no constructor in
`crates/helix/src` or the contract composes the in/out *pair* into the lane
in one call (grep 2026-08-11). The producer for the pair — two sign-opposite
encodes, or a dedicated reading — is open work (§10-P4). Do not invent it
inline; it belongs next to the tenant's other readings.

### 2.4 Hydration `[G]`

Two levels, both deterministic:

- **Curve level** — the Curve-Ruler Principle (`curve_ruler.rs:4-15`): the
  φ-spiral is the template; endpoints are stored; the interior regenerates
  via `(offset + 4k) mod 17`. Headline: *"8K resolution at Super-8 cost —
  the resolution lives in the deterministic template (free, regenerable);
  the cost is only the endpoint pair"* (`lib.rs:9-13`).
- **Geometry level** — the stored z-code materializes into Poincaré depth
  (`×2`) or sphere coordinates (`HemispherePoint::cartesian`,
  `placement.rs:141-145`) whenever a spatial reading is needed. Storage never
  pays for the geometry; reads project it.

---

## 3. palette256 — quantizer + distance surface

### 3.1 `RollingFloor` mechanics + honest loss accounting `[G]`

`crates/helix/src/quantize.rs`:

- `quantize(value)` is **linear** over a live `[lo, hi]` window:
  `idx = floor((v−lo)/(hi−lo)·256)`, clamped; ≤lo saturates to 0, ≥hi to 255
  (`quantize.rs:99-108`). **All non-linearity comes from the Fisher-Z step
  upstream** — the pipeline is `normalize → linear 256-bucket quantize`.
- **NOT lossless**, and the module says so: ±½ bucket = ±0.195 % of span in
  the informative range; out-of-window values saturate into the two rim
  buckets — controlled saturation, calibrated so tail occupancy stays small
  (`quantize.rs:11-18`).
- The 256 buckets double as the monitoring instrument: `occupancy` IS the
  empirical distribution; no separate histogram (`quantize.rs:5-9`).
- Compute/calibration split honors the workspace `data-flow.md` rule:
  `quantize`/`drift_score` are `&self`; `observe`/`roll` are `&mut self`
  (`quantize.rs:28-38`).

### 3.2 Calibration loop + the version contract `[G]`

- `observe(v)` accumulates occupancy (`quantize.rs:113-117`).
- `drift_score()` = max per-bucket deviation from uniform, in
  multinomial-SD units (`quantize.rs:119-146`). ⚠ This is a **hand-tuned
  trigger** (`drift_sigma` default 3.0, `quantize.rs:72-74`) using classical
  multinomial SD — acceptable per I-NOISE-FLOOR-JIRAK *because it is
  declared as hand-tuned*, but note for weather: spatially autocorrelated
  fields make observations strongly dependent, so the nominal σ is
  optimistic; treat the threshold as a knob to calibrate on real occupancy,
  not a significance test (§10-P5).
- `roll()` fires only above threshold: glides `[lo,hi]` toward the empirical
  0.4 %–99.6 % quantile window with inertia α (default 0.1), guards against
  degenerate bounds, resets occupancy, **bumps `version` (wrapping u8)**
  (`quantize.rs:148-216`).
- **The contract** (`quantize.rs:20-26`): *"Same value → same u8 holds ONLY
  within a stable floor version… callers must embed the version stamp
  alongside the quantised byte and invalidate cached LUTs on version
  change."* `ResidueEdge.floor_version` is exactly that stamp.

### 3.3 `DistanceLut` `[G]`

`crates/helix/src/distance.rs`: 256×256 × u16 = **128 KB** (L1/L2-cache
resident, `U8x64`-friendly; `distance.rs:12`). Two constructors:

- `linear()` — `d(a,b) = |a−b|` on the index order. L1 on a linear order IS
  a metric (triangle inequality by construction) ⇒ safe for CAKES/CLAM
  pruning bounds (`distance.rs:22-33`; regression tests sweep the index cube
  and assert zero violations, `distance.rs:87-105`).
- `from_floor(&floor)` — L1 over the floor's real `bucket_center` values,
  span-normalized back to `[0,255]` (`distance.rs:39-50`). Reflects
  (possibly post-roll non-uniform) bucket spacing; still a metric
  (`distance.rs:118-135`).

The layer discipline is inherited from bgz17: Scent-style bit-lattice
Hamming is NOT a metric and never feeds pruning bounds; palette L1 is
(`distance.rs:4-10`).

---

## 4. Floor scoping — the cross-variable comparability decision

**Resolved this session by reading the code: it is a policy the ingest must
set, not a property the crate grants or withholds.** The facts `[G]`:

1. Floors are **per-instance**; nothing shares them automatically
   (`ResidueEncoder::new` seeds its own; `residue.rs:130-141`).
2. Index meaning depends on the floor: `bucket_center(b) = lo +
   ((b+0.5)/256)(hi−lo)` (`quantize.rs:248-250`). **Two different floors ⇒
   the same u8 denotes different normalized values.** A cross-variable LUT
   lookup between codes from different floors is a unit error wearing a
   table lookup.
3. Nothing *prevents* sharing: `RollingFloor` is `Clone`, any floor feeds
   `DistanceLut::from_floor`, and Fisher-Z stabilization is precisely what
   makes heterogeneous variables coexist in one window.
4. Version divergence is the same hazard in time: independent `roll()`s
   desynchronize `floor_version` stamps per variable; the documented remedy
   is stamp-and-invalidate (§3.2).

**The three admissible policies** (decision owner: weather ingest, D-WXA
arc):

| policy | cross-variable LUT semantics | per-variable resolution | when |
|---|---|---|---|
| **(a) one canonical z-floor** for all scalar lanes | ✅ one 128 KB LUT serves every pair | shared window ⇒ some per-variable resolution ceded | the correlation substrate — the product point |
| **(b) per-variable floors** | ❌ codes not comparable across variables; cross-variable work returns to float z-domain | maximal | archival fidelity lanes |
| **(c) hybrid** — canonical correlation lane **plus** optional per-variable precision lanes | ✅ on the canonical lane | ✅ on the extra lanes | recommended `[S→H]` — matches the tenant model (a reading per purpose, empty lanes cheap) |

**Recommended operating rule `[S]` (needs operator sign-off + probe §10-P2):**
calibrate the canonical z-floor once per dataset epoch on an observation
batch, then **freeze it** (never `roll()` mid-epoch); align any re-roll with
a Lance dataset **version boundary** and record `(lo, hi, version)` in the
dataset metadata — so "same code ⇒ same value" holds exactly within a Lance
version, and time-travel reads (`QueryReference::at`) rehydrate with the
floor that produced them. This makes the floor stamp and the Lance version
the same kind of object: a calibration epoch.

---

## 5. Weather variable mapping (design `[H]` — primitives shipped, wiring not)

`[G-absence]` No weather-specific code exists in the tree yet; the POC plan
is the vehicle. The mapping below composes ONLY shipped primitives.

| variable | decomposition | transform | carrier | note |
|---|---|---|---|---|
| temperature (2 m, per-level) | climatology (PLACE) ⊕ anomaly (RESIDUE) | z-normalize anomaly | 1 × u8 palette lane | the 97× BF16 measurement (§6.3) is this principle one rung down |
| humidity (specific/relative) | climatology ⊕ anomaly | z-normalize | 1 × u8 | bounded variables are the native Fisher-Z case |
| pressure / geopotential | climatology ⊕ anomaly | z-normalize | 1 × u8 | |
| **wind direction, from + to** | — | helix360 | **`HelixResidue` 6 B lane @ offset 112** | 2 × 24-bit signed hemispheres (§2.3) |
| **wind speed** | own quantity — **NOT derivable** from stored (u,v) (§6.5) | z-normalize | 1 × u8 | carries gustiness (Jensen gap); dropping it destroys information |
| u, v components (if vector math needed) | anomalies | z-normalize | 2 × u8 | optional precision lanes under policy (c) |

The decomposition line **is** the crate's PLACE/RESIDUE doctrine applied to
climate: climatology is the deterministic, regenerable part (the template);
the anomaly is the orthogonal residue that quantizes well. HHTL supplies the
spatial address (lat/lon/level as trie path); `CurveRuler::from_hhtl` couples
the residue to it (`curve_ruler.rs:41-43`).

---

## 6. Measured-evidence ledger (session 2026-08-10/11, real ERA5)

> ⚠ **Provenance:** all rows measured in-session on real data; **none is yet
> a committed runnable example.** Per the falsifiability rule, treat each as
> *measured, not yet re-runnable* until §10-P1 lands. Conditions stated are
> the ones recorded; re-derive exact metric definitions when committing the
> probes.

| # | quantity | value | conditions |
|---|---|---|---|
| 6.1 | source-side compression (blosc/lz4 lvl 5, Zarr chunks) | **1.27× mean** | real ERA5 Zarr v2, dtype `<f4` |
| 6.2 | Lance compression on 512 B node rows | **1.00× (verbatim)** | by design: value > `MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE = 256` ⇒ full-zip path; opaque 512 B binary doesn't compress. Consequence: **right-size before storage** |
| 6.3 | BF16 on raw Kelvin vs on anomaly | **1.069 K vs 0.0110 K — 97×** | real ERA5 temperature; anomaly = field − climatology; error metric as recorded (mean-abs) — pin on probe commit |
| 6.4 | wind from ≠ to | **mean turn 15.50°; 90.2 % of gridpoints > 1.7°** | 240×121 grid, bilinear displacement→bearing. ⚠ apparatus lesson: the first run (64×32, `rint`-snapped) returned median 0.00° — displacement 0.17 cells landed both feet in one cell; a 53× swing that would have inverted the conclusion |
| 6.5 | wind_speed non-derivability | stored speed ≥ `hypot(ū,v̄)` at **100 %** of samples; mean ratio **1.115**; max gap **14.37 m/s** | stored speed = mean \|v\| over averaging window; hypot of means = \|mean v\| ⇒ Jensen gap = **gustiness**, real signal, not redundancy |
| 6.6 | u8 bearing sufficiency | error **≈8× under observational bearing error at 1 B/bearing** (2 B for the from/to pair) | 256 angular levels ≈ 1.41° step; exact obs-error reference to pin on probe commit. Canonical carrier remains the 6 B lane (adds polar + rim + version) |

Product readings of the ledger: (6.2) the storage engine will not save a
wrong representation; (6.3) normalization is worth two orders of magnitude
before any codec choice; (6.4)+(6.6) the in/out double hemisphere is both
*necessary* (from ≠ to almost everywhere) and *sufficient at u8*; (6.5) the
variable list cannot be pruned by "derivable" intuition — measure first.

---

## 7. Economics — the inbound tax, paid once `[G structure / H numbers]`

- **Ingest (once per datum):** `arctanh` (batch: `helix::simd::batch_fisher_z`,
  SIMD via the mandatory ndarray dep — `simd.rs:4,24`), placement, quantize.
- **Every read thereafter:** u8 index pair → 128 KB LUT → u16. No float, no
  re-normalization, no per-query derivation. At reanalysis scale (plan §0:
  ~570 k hourly states × 1.04 M gridpoints) comparisons outnumber data by
  orders of magnitude — per-comparison normalization is the difference
  between feasible and not.
- Same bake-once/read-forever doctrine as bgz-tensor's
  attention-as-table-lookup and bgz17's palette semiring; helix `lib.rs:45-56`
  records the honest overlap (deliberate clean-room re-derivation; the new
  pieces are the equal-area `√u` placement and the PLACE/RESIDUE doctrine).
- ABI fit: scalars land as u8 lanes, orientation in the existing
  `HelixResidue` tenant, all inside the canonical 512 B row — Lance's
  columnar I/O writes the LE bytes zero-copy (per the SoA three-tier model);
  empty lanes cost reservation, not encoding.

---

## 8. Ingest specification (recurring job)

- **Source of record (Phase A):** ARCO-ERA5 on public GCS, bucket
  `gcp-public-data-arco-era5`, `ar/` prefix, object
  **`1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr`** — session-verified
  `[H]`. ⚠ The plan's §0-C1 names `era5/1959-2023_01_10-full_37-1h-1440x721.zarr`,
  which does **not** exist — corrected via the plan's ⊘ C3 block (same
  commit as this doc). Grid facts unchanged: 0.25° ⇒ 1440×721 = 1,038,240
  points, hourly, 37 levels. Apparatus lesson recorded: the wrong path was
  "confirmed" from a GitHub issue *titled* "…can't be opened" — a title
  match is not an existence check; list the bucket.
- **Wire format in:** Zarr v2, `<f4` (LE float32), blosc/lz4 level 5 `[H]`.
- **Quick vs permanent** (operator framing; plan D-WXA-1): Stage-A ingest is
  a *disposable* Python `Zarr → xarray → numpy → f32 slab` — no eccodes, no
  C deps. The permanent Rust-side reader decision is deferred until the
  representation is validated; do not gold-plate the throwaway.
- **Pipeline per epoch:** fetch slab → compute/lookup climatology → anomaly
  → map to `[−1,1]` → `batch_fisher_z` → calibrate-or-load canonical floor
  (§4 policy) → quantize scalars to u8 lanes; bearings through helix360 into
  the `HelixResidue` lane → commit as **one Lance dataset version** with
  `(lo, hi, floor_version)` in metadata. Per the S3 doctrine (#901): object
  store hydrates, local mmap-capable dir stores.
- **Pins:** rust 1.97.1 (`rust-toolchain.toml` authoritative), lance =9.0.0
  family / lancedb =0.33.0, arrow 58, datafusion 54 **and** 53 both required
  (deltalake upstream) — see `CLAUDE.md § Key Dependencies`; do not
  re-litigate here.

---

## 9. Verification battery

Two gates, in order (operator-set: representation first, prediction second):

1. **Representation validity** — does the substrate preserve structure?
   - Rank fidelity: Spearman ρ between float-truth distances and LUT
     distances over sampled pairs; Pearson r on rehydrated vs truth values.
   - Reliability: ICC + Cronbach α across re-encodes / epochs.
   - Instruments: `jc` crate `reliability.rs` (pearson / spearman /
     cronbach_alpha / icc → `Option<f64>`) + `stats.rs`; hardware mirror
     `ndarray::hpc::reliability`. ⚠ Known divergence `[H]`: the ndarray
     mirror returns `f64` with `0.0` sentinels where jc returns
     `Option<f64>` — a `0.0` from the mirror is ambiguous
     (undefined-vs-zero); prefer jc for verdicts, ndarray for throughput.
   - Every "N σ above noise floor" claim: Jirak 2016 rates (§1.2 ⚠).
2. **Prediction correctness** — only after gate 1: WeatherBench2 `metrics.py`
   definitions (MSE/RMSE, ACC, SEEPS, CRPS) computed on rehydrated
   substrate vs f32 truth, so any skill delta is attributable to the
   representation, not the metric implementation.

Intrinsic monitors, free with the substrate `[G]`: `RollingFloor.occupancy`
(distribution health per lane), `drift_score` (regime-change telltale —
scientifically interesting for climate data in its own right), triangle-
inequality regression on any new LUT (pattern: `distance.rs:87-105`).

---

## 10. Probe queue (next deliverable is the probe, not more synthesis)

| id | probe | pass criterion | status |
|---|---|---|---|
| P1 | commit §6's session probes as runnable examples (ERA5 slab in CI-fetchable form or documented local fixture) | every §6 row re-derivable by command | NOT RUN |
| P2 | **shared-floor cross-variable probe**: T + q anomalies through ONE canonical z-floor; Spearman ρ of LUT cross-distances vs float z-domain truth | ρ ≥ 0.99 on sampled pairs (threshold hand-tuned, says so) | NOT RUN — gates policy (a)/(c) |
| P3 | per-variable resolution cost of the shared window (max quantization error per variable vs per-variable floors) | documented, bounded; no variable saturates > ~1 % beyond the designed rim tail | NOT RUN |
| P4 | in/out pair producer for the `HelixResidue` lane (two sign-opposite encodes vs dedicated reading), + round-trip test incl. rim-sign regression re-run | byte round-trip; sign exact at rim (extend `residue.rs:364-383` pattern) | NOT RUN — §2.3 absence |
| P5 | `drift_sigma` under spatial autocorrelation: occupancy from real fields, measure false-trigger/miss rates vs the 3.0 default | calibrated threshold with recorded rationale | NOT RUN |
| P6 | Lance-version ↔ floor-version alignment (§4 rule): freeze, ingest 2 epochs, time-travel read rehydrates with the matching floor | exact rehydration across `at(v)` | NOT RUN — promotes §4 `[S]` |

---

## 11. Correction ledger (this arc — kept so the next session doesn't repeat it)

Five corrections, all operator-caught, all of the same species: **asserting
from API surface or first principles what the code already answered.**

- **C1 — invented bit budget.** A "48-bit helix360 = from 15 / to 15 /
  magnitude 17 / sign 1" allocation was designed from scratch while
  `Signed360` (6 B, different and better-reasoned layout, incl. the #498
  sign-partition) already existed. Cause: a recon agent's output was never
  received and the gap was filled by invention. Rule: **the crate is read
  before any budget is proposed.**
- **C2 — doubling a doubled width.** Proposed `Pair48` (12 B) for from/to
  when the 6 B lane is *already* 2 × 24-bit in/out by construction
  (`canonical_node.rs:963`). Cause: read `Signed360`'s field list as THE
  definition of the lane instead of one reading of a content-blind register.
- **C3 — "unwired 2z seam".** Flagged `hyperbolic_depth`-never-called as a
  gap; it is the design — normalized-low-entropy is the carrier, geometry is
  hydratable (§1.2). First mis-described the same fact as "a monotone rescale,
  not a defect" — smoothing instead of naming; both readings were wrong.
- **C4 — "scalar similarity".** Described the palette input as a similarity
  score; it is a *normalized* quantity whose value is variance stabilization
  and cross-source comparability (§1.2-2) — the "correlate what normally
  can't be correlated" property.
- **C5 — measurement apparatus (earlier in session).** The `rint`-snapping
  bearing artifact (§6.4) and the "wind_speed is derivable" claim (§6.5) —
  both inverted by proper measurement.

Meta-rule extracted: in this workspace **"consult, don't guess" applies to
one's own prior messages too** — a confident earlier statement in-session has
the same evidentiary weight as a stale doc: none, until checked against the
tree.

---

## Appendix A — API quick reference `[G]`

```text
helix (crates/helix, standalone; mandatory ndarray dep, git per codex #460)
├─ constants: GOLDEN_RATIO, GOLDEN_ANGLE, EULER_GAMMA, LN_17,
│             MODULUS=17, STRIDE=4, PALETTE_SIZE=256, TRANSIENT_SKIP=17
├─ placement: Sign::{Pos,Neg,of,as_f64}
│             HemispherePoint::{lift, signed_lift, cartesian, rim}   // r²+y²=1
├─ curve_ruler: CurveRuler::{from_place, from_hhtl, start_offset, index, arc}
├─ fisher_z:  Similarity(f64)::{fisher_z, hyperbolic_depth}, CLAMP_EPS=1e-9
├─ quantize:  RollingFloor::{uniform, with_params, quantize(&self),
│             observe(&mut), drift_score(&self), roll(&mut)->bool,
│             version, occupancy, samples, bounds, bucket_center}
├─ distance:  DistanceLut::{linear, from_floor, distance(a,b)->u16}  // 128 KB
├─ residue:   ResidueEdge{start_idx,end_idx,floor_version} (3 B)
│             Signed360{rim,polar,azimuth} (6 B) ::{to_bytes,from_bytes,sign}
│             ResidueEncoder::{new, encode, encode_signed, observe, roll,
│             floor, total}
├─ simd:      batch_fisher_z(&[f32], &mut [f32])
└─ prove:     prove() -> ProofResult          // example prove_residue = the probe

contract (lance-graph-contract/src/canonical_node.rs)
└─ ValueTenant::HelixResidue = 4  → ColumnKind::U8 × 6 @ row_offset 112
   in ValueSchema::{Compressed, Full}; node row = key(16)|edges(16)|value(480)
```

---

## 12. ⊘ CORRECTIONS — what P1 and the envelope audit falsified (2026-08-11)

Written the same day as the rest of this document, by the two gates the
document itself queued. Both fired. Sections above are **left standing and
regraded here**, never silently edited.

### 12.1 — §1.2 FALSIFIED for weather scalars: Fisher-Z *degrades* the palette

> **⊘⊘ PARTIALLY REVERSED by §12.10** — this section was scored with a
> **round-trip** metric, which is the wrong criterion for a one-way address over
> a retained original. Re-measured against noise-floor thresholds the two paths
> are indistinguishable at a 0.5–1 K floor. What survives is *bucket economy*,
> not a validity failure. Read §12.10 before citing anything below.

Measured, real ERA5 `2m_temperature`, 1,038,240 gridpoints, both paths into the
same 256 buckets over the same 0.4–99.6 percentile window
(`probes/weather-p1/`, re-runnable):

| path | MAE (K) | p99 (K) | empty buckets | effective buckets | drift score |
|---|---|---|---|---|---|
| **linear** (no transform) | **0.0684** | 0.0939 | **0** | **115.7 / 256** | 607 |
| Fisher-Z then linear | 0.2168 | 0.4253 | **76** | **28.1 / 256** | 2240 |

Fisher-Z is **3.2× worse** and burns **228 of 256 buckets**.

**Mechanism, measured not assumed:** `arctanh` is ≈identity near 0 and explodes
near ±1, so it moves resolution *toward the bounds*. ERA5 temperature anomaly
over a robust scale has mean |s| = 0.172, **77 % of mass inside |s| < 0.25**,
1.2 % beyond 0.9, kurtosis +3.3 — mass in the **middle**. A correlation-like
control (`tanh(N(0,1.5))`) has 13.5 % inside 0.25 and **32.7 % beyond 0.9** —
mass at the **bounds**. Fisher-Z is right for the second shape and wrong for the
first.

**So §1.2's "this is what makes 8 bits sufficient" is false for weather
scalars.** The correct general statement: **the transform must match the input
distribution's shape.** Fisher-Z earns its keep on correlation-like inputs and
on helix's own rim radius `r = √u`, which equal-area placement concentrates
toward `r → 1` **by construction** (`placement.rs:147-151`) — i.e. helix uses it
on exactly the shape it suits. Nothing about the helix crate is impugned; what
is falsified is *generalizing its transform to bell-shaped geophysical fields*.

**Product consequence:** "Fisher-Z everything into one comparable substrate"
does not survive. **P2 is re-scoped**, not merely un-run: it must now test
cross-variable comparability under a *per-shape* transform choice, and prove
the shared **palette + LUT** still hold once the shared **transform** is gone.

### 12.2 — §2.3 / §5 / §6.6 FALSIFIED: helix360 is not a wind-bearing carrier

> **⊘⊘ GROUND NARROWED by §12.10** — the structural facts below stand, but the
> *"no inverse / no decode exists"* argument was never a defect under a one-way
> address frame. The `Pair48` successor mint returns to the table strengthened.

The `v3-envelope-auditor` verdict on reading the 6-byte `HelixResidue` lane as
`2 × ResidueEdge` (in/out): **LAYOUT-GATED, and the gate does not exist** — with
the premise itself falsified:

1. **`Signed360` is ONE signed orientation**, `rim(3) + polar(1) + azimuth(2)`
   (`residue.rs:76-116`) — not two hemisphere codes. The "2× the 24-bit
   hemisphere" line (`canonical_node.rs:838-839`, `placement.rs:170-171`) is a
   **width identity (48 = 2×24) plus a coverage claim**, never a structural
   in/out claim. §2.3's `[G]` on that reading was **unearned**; regrade `[S]`.
2. **`ResidueEdge` cannot carry a hemisphere sign at all.** `sprite_replay.rs:56-63`:
   sprites seeded `Sign::Neg` reconstruct through the `ResidueEdge` path as
   `Sign::Pos` — *"a real, measurable structural error, not a rounding
   artifact."* An in/out pair of `ResidueEdge` would silently force both to the
   upper hemisphere.
3. **`ResidueEdge` has no azimuth.** The 16-bit `azimuth` is the **only** angular
   field in the entire lane; the proposed reading deletes it, and spends 2 of 6
   bytes on duplicate `floor_version` stamps.
4. **Neither reading is a bearing codec.** The crate says so twice, unprompted:
   *"there is no free 2-DOF direction codec in this crate"* (`sprite_replay.rs:47`)
   and *"NOT a generic scalar quantizer … doing so would be exactly the
   'invented round-trip API'"* (`continuous_field.rs:31-43`). My proposal was
   that invented round-trip.
5. **Metric hazard if forced:** `end_idx` is monotone, i.e. **non-circular**, so
   a wrapped bearing puts 359° and 1° at maximum L1 distance. `DistanceLut`'s
   triangle-inequality guarantee is about a **linear** index order and does not
   survive repurposing as an angular one (`residue.rs:53-55`, `distance.rs:22-33`).
6. **No selector exists.** Exhaustive grep (`HelixResidue`, all `**/*.rs`,
   `head_limit: 0`): 15 hits, **zero writers, zero decoders**. `ReadMode` has
   exactly three axes (tail / value_schema / edge_codec) — none selects a
   value-lane *reading*; `ValueSchema` selects presence only; `EdgeCodecFlavor`
   covers `EdgeBlock`, a different region. There is **no per-value-lane reading
   selector in the contract at all** — that absence is the finding.

**Candidate shape `[S]` if a from/to pair is genuinely needed** (a CONJECTURE, not
a promoted design — it follows from the envelope audit in this section and is
gated on the operator mint decision tracked in §12.6): a **new 16-byte facet
lane** read as `FacetSchema::Pair48` = `[Signed360; 2]` — `classid(4) + 6 B in +
6 B out` — precedent set twice by `Tekamolo` and `CausalWitness` (both 16-byte
lanes appended at the end, no version bump). Codec owner `crates/helix` (next to
`encode_signed`, so the #498 sign-partition is never re-implemented); lane owner
`lance-graph-contract` (zero-dep, reserves bytes only). **⚠ discriminant `15` is
reserved for `BoardAggregates`** (`canonical_node.rs:1027-1034`) — a new tenant
takes `16`, never silently consumes 15.

### 12.3 — §11-C2 was itself a mis-correction (the sharpest lesson here)

C2 recorded rejecting a 12-byte `Pair48` for from/to because "the 6 B lane is
already 2×24 in/out by construction", citing `canonical_node.rs:963`. That line
is a **comment**, and it states a width identity, not a structure. **The 12-byte
`Pair48` I rejected was the structurally correct answer** (§12.2). C2 replaced a
right answer with a wrong one *and then banked the wrong one as a lesson* — the
failure mode compounding one level up, inside the ledger built to catch it.

**Rule extracted:** a correction is a claim and carries a claim's burden of
proof. "I was wrong before" is not evidence that the new version is right, and a
correction written in the same breath as the error it replaces has had no
independent gate. Route corrections through the same audit as originals.

### 12.4 — Measurement provenance (§6.1, §6.3, §6.5)

- **§6.5 wind-speed non-derivability — provenance broken.** ARCO-ERA5 has **no
  `wind_speed` variable** (exhaustive: all 52 arrays enumerated; wind-related are
  exactly the 4 `u`/`v` component arrays). The measurement cannot have come from
  the Phase-A source the plan names. The Jensen-gap *physics* stands; the
  *measurement* must be re-attributed (WeatherBench2) or dropped. §5's "wind
  speed — NOT derivable" row is wrong for this source.
- **§6.1 blosc 1.27×** is not representative: measured **1.794×**
  (`2m_temperature`), **1.771×** (`2m_dewpoint_temperature`), **1.248×**
  (`10m_u_component_of_wind`) at one timestep. Ratio is strongly
  variable-dependent; a single scalar is the wrong summary.
- **§6.3 BF16 anomaly gain** reproduced in **direction only**: 0.456 K → 0.00609 K
  = **74.9×** here (doc: 1.069 K → 0.0110 K = 97×), under a zonal-mean
  climatology proxy. Different climatology ⇒ different measurement, not a
  contradiction — but the doc's numbers stay **unreproduced** and `[H]`.
- **New, unhidden:** 0.82 % of points saturate into the two rim buckets, giving a
  palette **max** error of 17.5 K against a 0.068 K mean. That is
  `quantize.rs:11-18`'s documented controlled-saturation tail. Any product claim
  quotes the tail, not only the mean.

### 12.5 — Pre-existing defect found in passing (not caused by this work)

`Signed360::sign()` is `if polar >= 128 { Pos } else { Neg }`
(`residue.rs:109-115`), so an **all-zero, never-written lane decodes as a
definite lower-hemisphere orientation** rather than as "unaddressed". Both
sibling facet lanes get this right — `Tekamolo`: *"an all-zero facet reads as
unaddressed … never a wrong circumstance"* (`canonical_node.rs:886-887`);
`CausalWitness`: *"reads as unbound … never a wrong binding"* (`:900-902`).
Under the CANON zero-fallback ladder ("zero = fall through to the broader
default") this is a wrong-value-from-dormant-bytes defect. Filed, not fixed here.

### 12.6 — Probe queue, re-graded

| id | status |
|---|---|
| **P1** | **RUN** — `probes/weather-p1/`, results committed. Falsified §1.2 and §6.5; re-measured §6.1/§6.3. |
| **P1-reframed** | **RUN** (§12.10) — CI-vs-noise-floor, no round-trip. Reverses §12.1's magnitude; surfaces the saturation-tail-hits-the-extremes finding. |
| **P2** | **RUN — PASSES** (§12.8). Re-scoped per §12.1 and measured: shared canonical floor ρ ≥ 0.9996 cross-variable vs 0.857–0.875 per-variable; zero within-variable cost. Shared palette + LUT survives; the shared *transform* does not. |
| **P3** | largely ANSWERED by §12.8 (no within-variable penalty from the shared floor); stays open for the saturation-tail question only. |
| **P4** | **CANCELLED as specified.** The premise is falsified (§12.2). Successor: the 16-byte `Pair48` facet lane — a mint decision, operator-gated, not a worker task. |
| **P5** | unchanged, NOT RUN. |
| **P6** | unchanged, NOT RUN. |

### 12.7 — What survives, stated plainly

The **economics** (§7) survive: pay the transform once at ingest, compare via an
O(1) 128 KB cache-resident LUT forever. The **place ⊕ residue decomposition**
survives and strengthened — anomaly-vs-raw is a 74.9× BF16 gain measured on real
data. The **8-bits-per-scalar** claim survives, with a corrected reason: a robust
percentile window over a **shape-appropriate** transform, which for weather
anomalies is the identity. **helix360 as the wind carrier does not survive**, and
neither does the single-shared-transform framing of the substrate.

### 12.8 — P2 RUN (re-scoped): the shared palette + LUT **survives**; only the shared *transform* died

The §12.1 falsification killed the shared **Fisher-Z**. It did not, by itself,
tell us whether the shared **palette + LUT** — the actual product claim — went
with it. Measured (`probes/weather-p1/p2_probe.py`, real ERA5, 3 variables,
200 000 random cross-variable pairs each; truth = float z-domain
`|z_a − z_b|`, candidate = `|palette_idx_a − palette_idx_b|`, Spearman ρ):

| pair | units | ρ shared floor | ρ per-variable floors |
|---|---|---|---|
| `2m_temperature` × `2m_dewpoint_temperature` | K × K | **0.999556** ⚠ *below the 0.9996 bar* | 0.999426 |
| `2m_temperature` × `10m_u_component_of_wind` | K × m/s | **0.999736** PASS | **0.874801** |
| `2m_dewpoint_temperature` × `10m_u_component_of_wind` | K × m/s | **0.999722** PASS | **0.856984** |

**Within-variable control (the anti-vacuity half — a shared window could have
cost same-variable resolution, and did not):**

| variable | ρ shared | ρ per-variable | effective buckets (shared) | empty |
|---|---|---|---|---|
| `2m_temperature` | 0.9996 | 0.9996 | 118.6 / 256 | 0 |
| `2m_dewpoint_temperature` | 0.9995 | 0.9996 | 120.7 / 256 | 0 |
| `10m_u_component_of_wind` | 0.9998 | 0.9996 | 151.0 / 256 | 0 |

> **⊘ CORRECTED (codex/CodeRabbit review on #920).** This table originally
> reported the first pair as `0.9996` — a **rounding of 0.999556, which is
> BELOW the 0.9996 bar**, and the probe's `SHARED` label only ever tested
> "shared beats per-variable", never the bar itself. `p2_probe.py` now carries
> an explicit pre-registered `TARGET_RHO` and prints PASS / BELOW TARGET on
> unrounded values. **2 of 3 pairs pass; the K×K pair does not.**

**Reading.** The shared canonical floor is not free-riding: it beats
per-variable floors by **0.9997 vs 0.857–0.875** exactly where the units differ
(K vs m/s) — **both cross-unit pairs clear the bar** — and the two paths are
close where the units match (K vs K, 0.999556 vs 0.999426, *the shared floor
still winning but missing the 0.9996 bar*). That is the discrimination pattern
theory predicts, and it appears without having been tuned for. The shared window
shows **no material measured cost** within-variable at this sample size
(within-variable ρ differs by ≤ 0.0001, e.g. dewpoint 0.9995 shared vs 0.9996
per-variable) and leaves **zero empty buckets**.

**So the mechanism was never Fisher-Z — it was standardization.** Putting each
variable on its own z-scale is what makes unlike quantities commensurable;
Fisher-Z was an *additional* transform layered on top, and §12.1 measured it as
actively harmful for these shapes. The product claim — *"correlate what normally
cannot be correlated, via one cache-resident 256×256 LUT"* — **stands, with a
corrected mechanism.**

**Policy consequence:** §4's option **(a) one canonical z-floor** is now the
measured recommendation, not merely a candidate; option (c) hybrid remains
available for archival lanes that want per-variable resolution, but this run
shows it buys nothing for correlation, since the shared floor already matches
per-variable ρ within-variable.

**Honest boundaries.** The K×K pair is **below the pre-registered bar** — the
"passes" verdict holds for the cross-unit case this probe was designed to test,
not universally. One timestep (2021-06-15 12:00 UTC); zonal-mean
climatology proxy; three variables of which only one carries a different unit;
ρ is rank-preservation of the *distance*, which is the right test for
LUT-backed ranking/search but is not a claim about absolute error. The shapes
do genuinely differ (kurtosis +3.30 / +3.39 / **+0.41**), so the result is not
an artifact of three identically-shaped fields. Grade `[H]` until re-run across
seasons and a fourth variable in a different unit again.

**Probe queue delta:** **P2 → RUN, PASSES.** P3 (per-variable resolution cost)
is now largely answered in the negative — the shared floor shows no
within-variable penalty — but stays open for the tail/saturation question.
Step 3 (Phase A end-to-end) is unblocked.

### 12.9 — Gate 1 RUN with `jc` (not scipy): codec self-consistency — NOT product-frame evidence

> **⊘ SCOPE (CodeRabbit on #920, correct).** Everything in this section is a
> **round-trip** measurement (`truth` vs `reconstructed`), i.e. an **internal
> codec / consistency diagnostic**. §12.10–§12.11 rule round-trip OUT as the
> *product* evaluation frame, because the original is retained. So these numbers
> say the quantizer is self-consistent and rank-preserving; they are **not**
> evidence for the noise-floor conclusion and must never be cited as such. The
> product-frame verdict lives in §12.11 (CI vs floor) and §12.8 (code-distance
> rank), neither of which decodes.

§9's gate 1 answered with the workspace's own instruments —
`crates/jc/examples/weather_substrate_reliability.rs`, run against the P1/P2
palette256 round-trip on real ERA5 (3 variables × 50 000 sampled points, shared
canonical floor). `jc` rather than `ndarray::hpc::reliability` deliberately: the
mirror returns `0.0` sentinels where `jc` returns `Option<f64>`, so a `0.0` from
the mirror cannot be distinguished from *undefined* — §9's stated rule.

| variable | Pearson r | Spearman ρ | Cronbach α | ICC(2,1) abs | ICC(3,1) cons |
|---|---|---|---|---|---|
| `2m_temperature` | 0.998558 | 0.999866 | 0.999216 | 0.998432 | 0.998432 |
| `2m_dewpoint_temperature` | 0.998506 | 0.999898 | 0.999179 | 0.998359 | 0.998360 |
| `10m_u_component_of_wind` | 0.999785 | 0.999964 | 0.999891 | 0.999781 | 0.999781 |
| **POOLED (shared floor)** | **0.998939** | **0.999926** | **0.999435** | **0.998869** | **0.998870** |

**Negative control — the can-it-fire half.** `--shuffle` feeds deliberately
mismatched pairs; every statistic collapses to noise: r ∈ [−0.0062, −0.0007],
ρ ∈ [−0.0057, −0.0001], α ∈ [−0.0126, −0.0014], both ICCs likewise. So these are
measurements, not assertions implied by their own input.

**The load-bearing detail: ICC(2,1) ≈ ICC(3,1) to six decimals.** They are
reported separately on purpose — ICC(3,1) is *consistency*, ICC(2,1) is
*absolute agreement*, and a quantizer that preserved shape while shifting scale
would show high consistency and degraded agreement. They agree, so **there is no
scale shift hiding behind a consistency number**. A single ICC figure would not
have shown this.

⚠ **Significance, not point estimates.** Per `I-NOISE-FLOOR-JIRAK`, any claim
that these values sit N σ above a noise floor takes **Jirak 2016 weak-dependence
rates** — weather fields are spatially autocorrelated, so effective sample size
is far below the nominal 50 000, and classical IID intervals would be wrong.
Nothing above is a significance claim; they are point estimates with an explicit
negative control.

### 12.10 — ⊘⊘ THE EVALUATION FRAME WAS WRONG: thresholds vs noise floor, never round-trip

**Operator correction, 2026-08-11.** Everything in §12.1 was scored with a
**round-trip** metric — encode, decode, measure reconstruction error against the
original. **That is the wrong criterion for this substrate, and it inverted a
verdict.**

**Why it is wrong.** The substrate never decodes. The original is *retained* —
public knowledge, the hydratable value column, the Lance-versioned f32. The code
is a **one-way address / discriminator layered over a kept original**, so the
question was never *"can I reconstruct the original from the code"* but
**"does the code's confidence interval (its quantization threshold) sit below
the original's own noise floor?"** Deviation only matters where it **exceeds
what the original itself can resolve**. This session's own 2026-08-08 reply had
it right — *"the code is not the bottleneck, the atmosphere is; the instruments
can't see it"* — and §12.1 then scored reconstruction MAE anyway.

> **⊘⊘⊘ THE TABLE BELOW IS STILL WRONG — see §12.11 (codex P1 on #920).**
> `p1_noise_floor.py` computes `dev = |bucket_center − original|`, which is a
> **decoded reconstruction error**: the very round-trip this section declares
> invalid. The CI values it computed were only *printed*, never used in the
> exceedance fractions. §12.11 carries the genuinely CI-based measurement. The
> qualitative conclusion survives; every number below does not.

**Re-measured in the correct frame** (`probes/weather-p1/p1_noise_floor.py`,
same real ERA5 T2m field, 1,038,240 points; fraction of gridpoints whose
code-induced deviation **exceeds** a candidate floor):

| path | CI (mid-range) | >0.25 K | >0.5 K | >1.0 K | >2.0 K | max |
|---|---|---|---|---|---|---|
| linear | ±0.094 K | 0.76 % | **0.70 %** | **0.60 %** | 0.40 % | 17.55 K |
| Fisher-Z | position-dependent | 36.3 % | **0.69 %** | **0.58 %** | 0.40 % | 17.54 K |

**§12.1 is PARTIALLY REVERSED.** At a 0.5–1 K floor the two paths are
**indistinguishable** (0.70 % vs 0.69 %; 0.60 % vs 0.58 %), and the exceedances
are the **shared saturation tail**, not the transform. "Fisher-Z is actively
harmful for weather" was an **artifact of the round-trip metric**. Fisher-Z only
fails at a 0.25 K floor — stricter than the instruments — because its mid-range
buckets are ±0.26 K wide (CI by index: 0.087 / **0.263** / 0.005 / 0.0001 /
0.000 K at idx 0/64/128/192/255; the position-dependence *is* the shape finding,
now correctly expressed as a CI profile rather than an error score).

**What survives of §12.1: bucket economy, not validity.** Fisher-Z spends 228 of
256 addresses where this distribution is not (28.1 effective buckets vs 115.7).
That degrades **discrimination granularity** — more code ties, coarser analogue
retrieval — which is a real cost for the retrieval lane and was measured
frame-correctly in **§12.8** (rank ρ of *code distances*, no round-trip). So the
falsified claim shrinks from *"Fisher-Z is wrong for weather"* to **"Fisher-Z
buys nothing here and costs address space."** §1.2's mechanism (transform should
match distribution shape) stands; its *magnitude* did not survive re-framing.

**§12.2 loses its main ground too.** The structural facts stand (`ResidueEdge`
carries no hemisphere sign; azimuth is `n·φ`-derived, not an external bearing).
But a substantial part of that cancellation rested on *"the crate ships no
inverse — no `decode(edge) -> n` exists."* **Under a one-way address frame that
was never a defect.** What a wind lane actually needs is a **defined
deterministic bearing→address convention** — a design act, which is exactly what
the mint decision is — plus CI-vs-floor validation: a ~1.4–1.7° step against the
**10° METAR/SYNOP reporting increment** is the ~6× margin recorded on
2026-08-08. **The `Pair48 = [Signed360; 2]` mint returns to the table
strengthened, not weakened.** P4 stays cancelled *as specified* (2×`ResidueEdge`
in 6 bytes remains structurally wrong); its successor is live again.

**A finding neither frame produced alone.** The **shared saturation tail** —
0.4–0.7 % of points, max ≈ 17.5 K — is the *only* genuine noise-floor violation
for either path, and those points are the **extremes**: heat waves, cold
outbreaks. Meteorologically the most important events sit exactly where this
representation is worst. The 0.4–99.6 % window is a **knob inherited from the
cognitive substrate's defaults**; weather likely wants it widened, or an
explicit overflow lane for extremes. That is now the highest-value open item on
the representation, ahead of any transform question.

**⚠ The floor itself is NOT yet pinned.** 0.25 / 0.5 / 1.0 / 2.0 K are a
**reference class**, not a citation. No published ERA5 T2m uncertainty figure was
verified in this run, so every verdict above is *"indistinguishable at a
0.5–1 K-class floor"*, conditional on that class being right. **Pinning a citable
per-variable noise floor is a prerequisite** before any of this is graded `[G]`
— and per `I-NOISE-FLOOR-JIRAK`, any σ-distance claim against it takes Jirak
weak-dependence rates, never classical intervals.

**Meta — §12.3's rule, applied to §12.1.** §12.3 stated that *a correction is a
claim and carries a claim's burden of proof*. §12.1 **was** such a correction,
and it over-claimed because it inherited an evaluation frame without examining
it. The rule caught its own author one section later. **The frame a measurement
is scored in is itself a claim requiring an audit** — that is the generalization,
and it is worth more than either verdict it revised.

### 12.11 — the CI metric, done properly (codex P1 on #920): the frame was right, my implementation of it was not

§12.10 announced the correct criterion and then **measured the old one anyway**.
`p1_noise_floor.py` computes `dev = |bucket_center − original|` — a decoded
reconstruction error — and counts `dev > floor`. The confidence intervals it
computed were printed and discarded. Caught by codex, correctly:

> *"The advertised noise-floor reframing still computes a round-trip
> reconstruction metric… the committed exceedance fractions and the conclusions
> in §12.10 are produced by the exact evaluation frame the commit says is
> invalid."*

**The actual metric** (`probes/weather-p1/p1_ci_vs_floor.py`): for each point,
take the **confidence interval of the bucket it encodes to**, expressed in
original units — `CI(b) = ½ × (width of bucket b in K)` — and ask whether that
CI exceeds the floor. Saturated edge buckets (0, 255) have **unbounded** CI and
are reported separately, never folded in as if they had a finite width.

| floor | linear: interior CI > floor | Fisher-Z: interior CI > floor | saturated (unbounded) |
|---|---|---|---|
| 0.25 K | **0.0000 %** | **95.65 %** | 0.848 % / 0.820 % |
| 0.5 K | 0.0000 % | 0.0000 % | 0.848 % / 0.820 % |
| 1.0 K | 0.0000 % | 0.0000 % | 0.848 % / 0.820 % |
| 2.0 K | 0.0000 % | 0.0000 % | 0.848 % / 0.820 % |

Interior CI: **linear is flat at 0.09412 K everywhere**; Fisher-Z spans
0.00000–0.42899 K, median 0.00561.

**What changes versus §12.10's bogus table:**

1. **Linear is cleaner than reported** — 0.0000 % interior exceedance at every
   floor, not 0.60–0.76 %. Those fractions were reconstruction noise, not CI.
2. **Fisher-Z at a tight floor is far worse than reported** — **95.65 %**, not
   36.3 %. And the mechanism is now explicit: the median CI **across buckets**
   is 0.0056 K, but most **points** live in the few wide mid-range buckets
   (up to 0.429 K). Most buckets are narrow tail buckets holding almost no data.
   **This is the 28.1-effective-buckets finding restated in CI units** — the two
   measurements now agree, which is the cross-check §12.10 lacked.
3. **The headline conclusion SURVIVES, and is cleaner.** At a 0.5–1 K floor the
   two paths are identical and **only saturation matters** — 0.848 % vs 0.820 %,
   i.e. the shared tail is the *entire* story above 0.25 K.
4. **The saturation finding is confirmed and sharpened — with an explicit
   floor condition.** At floors **≥ 0.5 K** it is the **only** violation for
   either path: **0.848 %** (linear) and **0.820 %** (Fisher-Z) of the
   1,038,240 gridpoints, i.e. the points landing in buckets 0/255 whose CI is
   unbounded. **At a 0.25 K floor that is false for Fisher-Z**, which adds
   **95.65 %** interior-CI exceedance on top. Every floor-dependent statement
   here stays conditional until a per-variable noise floor carries a citation.

**So §12.1 stays partially reversed, on better evidence** — Fisher-Z is not a
validity failure at the plausible floor; it is an **address-economy** failure,
and at a 0.25 K floor that economy failure becomes a validity failure too.

**Third-order lesson.** §12.3: *a correction carries a claim's burden*. §12.10
was a correction of §12.1 that **inherited the very defect it was correcting** —
it named the right frame in prose and shipped the wrong one in code. Naming a
frame is not adopting it; **the implementation is the claim**, and prose
asserting otherwise is exactly the "doc-comment claim is not a behaviour" rule
(`CLAUDE.md` falsifiability §) one level up. An external reviewer caught what
two internal passes did not.

**Also fixed (codex P2):** `fetch.py` raised an uncaught `HTTPError` on the
first legitimately-missing chunk, so the documented `python3 fetch.py`
reproduction died before writing its manifest — while the README *correctly*
documented 404 as valid fill semantics. The code now returns the fill array and
continues, matching the README it contradicted.

### 12.12 — ⊘⊘⊘ §12.2's REASONING is regraded; the `Pair48` mint is RETIRED; the doctrine doc existed and was never read

**Trigger.** Operator: *"probably signed360, a naming mis remembered"*, with two
links — `crates/helix/src/residue.rs#L74` and
`.claude/knowledge/helix-cartesian-vs-fisher2z.md#L18`. The second is a
knowledge doc **in this repo**, whose `READ BY:` header names *"ANY session that
encodes/decodes/renders an orientation, normal, or direction"*. This session is
exactly that session, and **never opened it** — not before §2.3, not during
§12.2's envelope audit, not through four subsequent corrections.

#### The name, settled

`helix360` **does not exist and never did** `[G-absence]`. Pickaxe over full
history (`git log -S`, all refs, plus unreachable objects) returns **12 blobs,
all authored by this session, zero deletions, zero blast radius**; ndarray and
the `lance-graph2` backup carry none. The real symbol is **`Signed360`**
(`residue.rs:76-116`). Every use of "helix360" in §0–§11 above is this session's
coinage for `Signed360`; read it that way.

#### Three structural corrections to §12.2 — the facts stood, the inferences did not

| §12.2 said | Regrade | Authority |
|---|---|---|
| 1. "`Signed360` is ONE signed orientation, not two hemisphere codes" — used to argue the lane cannot carry a from/to pair | **Fact stands; the inference inverts.** The sign is precisely what makes it whole: *"the normal-only 6-byte `Signed360` is a **complete** full-sphere direction — you do NOT need a second 'pos' helix to complete it."* | `helix-cartesian-vs-fisher2z.md:77-82` |
| 2-3. "`ResidueEdge` cannot carry a hemisphere sign / has no azimuth" — used to argue *the lane* has no bearing capacity | **Fact stands; it was the wrong object.** `ResidueEdge`/`rim` is the **METRIC carrier, not a render input** — *"never run the rim's atanh/tanh to recover a direction."* Direction lives in `(polar, azimuth)`. The operator's *"ResidueEdge is turbovec, nobody was asking you to use turbovec"* named this before the doc confirmed it. | `helix-cartesian-vs-fisher2z.md:83-86`; `residue.rs:21-27` |
| 4. crate says *"no free 2-DOF direction codec in this crate"* ⇒ read as "the representation cannot carry a bearing" | **Misread: no *helper*, not no *capacity*.** `azimuth` is *"`n·φ mod 2π` mapped to `[0, 65536)` over the full **360°**"* — a full-circle 16-bit angular field. What the crate lacks is the **encode helper**: *"Encoding a 3D normal → `(n, sign)` is a nearest spherical-Fibonacci search (the crate has no `from_normal` helper)"*, with a worked pair at q2 `scratch-fma/helixbake`. | `residue.rs:85`; `helix-cartesian-vs-fisher2z.md:90-93` |

#### What in §12.2 still stands, unchanged

- **Point 5 (metric hazard) — stands, and is *already crate-documented*, not a
  discovery.** `distance_heuristic` names its own failure mode as *"the
  raw-azimuth 2π wrap"* and forbids it for CAKES bounds (`residue.rs:52-59`).
  The correct statement is the canonical split, not a defect: **rim = L1-metric
  carrier (`DistanceLut`, triangle inequality); azimuth = render/direction
  carrier (circular, deliberately not L1-metric).** Two fields, two jobs. My
  error was proposing to route a bearing through the *metric* field and then
  reporting the resulting hazard as a property of the codec.
- **Point 6 (no per-value-lane reading selector exists) — stands `[G-absence]`.**
  15 `HelixResidue` hits, zero writers, zero decoders; `ReadMode`'s three axes
  select tail / value_schema / edge_codec, none a value-lane *reading*.
- The **dormant-lane defect** is unaffected and still open: `Signed360::sign()`
  reads an all-zero lane (`polar = 0`) as a definite `Sign::Neg`
  (`residue.rs:109-115`), so "unwritten" and "lower hemisphere" are
  indistinguishable — a real hazard the moment a writer exists.

#### Decision RETIRED: the `Pair48` mint

§12.2's `[S]` candidate (a new 16-byte `FacetSchema::Pair48 = [Signed360; 2]`)
and §12.3's lesson-about-the-lesson both rested on needing a second helix to
complete a direction. **That premise is false**, so the open operator decision is
**withdrawn, not deferred** — do not mint it. A wind bearing is a 2-DOF direction
on the horizontal; one `Signed360` already addresses the full sphere, a fortiori
the circle. §12.3's *extracted rule* survives untouched and is in fact the rule
that just fired again: **a correction is a claim and carries a claim's burden.**
Three links deep now (§11-C2 → §12.2/§12.3 → §12.12).

#### The real gap, stated so it is actionable

Not a missing *carrier* — a missing *encode*. `(bearing, elevation) → Signed360`
is a **nearest spherical-Fibonacci search** over `HemispherePoint::lift`
(pole axis = chosen world axis, `sign = sign(n·pole)`), which the crate
deliberately does not ship as `from_normal`, with a worked reference pair
(q2 `scratch-fma/helixbake` + `cockpit/src/BodyHelix.tsx`). Decode for
render/compare is **pre-materialize one `(polar × azimuth) → direction` LUT,
then gather** — polar 7-bit ≈ 0.45°, azimuth 10-bit LUT column ≈ 0.35°. That is
the mechanical form of the operator's *"you only pay the inbound tax once"*, and
`[S]` until measured on real wind data.

#### Process lesson — the one worth more than the codec

`CLAUDE.md` § *Consult, don't guess* orders it explicitly: specialist card →
**knowledge doc** → board → *only then* grep source. This arc did source-first
and stayed wrong through four corrections, because **reading the primary source
is not the same as reading the doctrine that interprets it.** `residue.rs`
states the layout truthfully; it does not say which field is metric and which is
render, nor that the sign already completes the sphere — that is exactly what
the knowledge doc exists to carry, and it is what a `READ BY:` header is *for*.
Cost: one invented 48-bit budget, one straw-man in/out reading, one retracted
mint decision, and a hunt for a deleted symbol that never existed. **A
`READ BY:` header that matches the session's own subject is a mandatory read,
not a suggestion.**

Cross-ref: `.claude/knowledge/helix-cartesian-vs-fisher2z.md` (the authority for
this whole section), `crates/helix/KNOWLEDGE.md` (place/residue spec).

#### 12.12a — TESTED (added after the fact, on the operator's question "did you test signed360?")

**No — §12.12 above was landed on doc comments and the knowledge doc, with
nothing executed.** That is the same failure it diagnoses, one level down:
having just written *"reading the primary source is not the same as reading the
doctrine"*, I read both and **ran neither**. Corrected here.

The crate's own suite is **77 + 4 + 7 doctests, all green**, but a green suite is
not evidence for a specific claim. Audited claim-by-claim, then closed the gaps:

| claim in §12.12 | before | now |
|---|---|---|
| sign exact at `\|y\| ≈ 0` (partition, not a centred round) | `[G]` — `signed360_neg_sign_survives_near_rim_at_high_total`, a real regression with a documented prior failure (codex P2 #498) | unchanged |
| `azimuth` is a full-360° field | **doc comment only.** The existing `signed360_azimuth_varies_with_n` asserts `a != b` for ONE consecutive pair — golden-angle stepping makes that true of any sane implementation, so it cannot detect a truncated field | `[G]` **measured** over the whole domain: `min 0`, `max 65535`, **256/256** coarse arcs occupied, 54 319 distinct values |
| hemisphere → sphere via the sign | mechanism tested; **nothing decodes to a direction** (there is no decode — see the `from_normal` gap) | unchanged; the *coverage* half stays `[H]` until an encode/decode pair exists |
| dormant lane reports a definite sign | **asserted twice, never constructed** | `[G]` **demonstrated**: `Signed360::from_bytes([0u8; 6]).sign()` → `Sign::Neg` |

Also measured in passing: the `polar` partitions fill their halves **exactly** —
`Pos ∈ [128, 255]`, `Neg ∈ [0, 127]`, no overlap, no gap at either end.

**Landed as `crates/helix/tests/signed360_claims.rs`** (3 tests), each
**disable-verified red-then-green** against an injected defect:

| test | injected defect | observed failure |
|---|---|---|
| `azimuth_spans_the_full_circle_not_merely_varies` | 10-bit truncated field | `left: 1023, right: 65535` |
| `polar_partitions_are_exactly_the_two_halves` | the #498 centred-at-128 round | `left: (0, 128), right: (0, 127)` |
| `dormant_all_zero_lane_decodes_as_a_definite_sign_known_defect` | a "fixed" sentinel returning `Pos` | `left: Pos, right: Neg` |

The third **pins a defect, not a virtue** — when the dormant-lane hazard is
fixed it MUST fail and be re-pinned deliberately, never silently edited.

**Scope caveat, stated rather than implied:** `helix` is **excluded from the root
workspace** (root `Cargo.toml` `exclude`) and named in **no CI workflow**
`[G-absence]` — so these 3 tests, and the crate's existing 77, run **only when
invoked by hand** in that crate. Adding them raises the floor for the next
session that looks; it does not put them on a gate.

#### 12.13 — the WIND reuse is the process, not an invention — and it surfaces a fit problem the normal-encode case never had

**Operator correction (2026-08-11):** *"the reuse of the above for the wind was
always part of the process (what you call invention)."* Accepted, and the line
§12.12 drew was wrong in **both** directions:

| | |
|---|---|
| **Invention** (what the arc rightly punished) | Asserting structure the code already answers — the 48-bit budget, the 2×24 in/out reading, `Pair48`, a round-trip API `continuous_field.rs` explicitly disclaims. |
| **Reuse** (the process) | Applying the **shipped, proven** codec to a new domain. This is what a normalized substrate is *for* — *"you only pay the inbound tax once."* |

§12.12 filed `from_normal` under *invention* and declined to build it. That was
over-correction: generalizing "don't invent structure the code already
determines" into "don't implement the intended reuse." **A missing entry point
for a designed reuse is a plumbing gap, not a design refusal** — the doctrine
doc even names the algorithm and points at a worked reference pair (q2
`scratch-fma/helixbake` + `cockpit/src/BodyHelix.tsx`).

**What the reuse then surfaced (the actually-new finding) `[G]`.**
`encode_signed` derives **all three** direction-bearing fields from `n` alone
(`residue.rs:182-204`): `rim` from `(place, n)`, `polar` from
`signed_lift(n, …)`, `azimuth` from `n·φ`. Two candidate bearing encodes exist,
and **the doctrine's prescribed one does not fit weather** — measured, N=65536
(`crates/helix/tests/bearing_encode_paths.rs`):

| horizontal bearing | Path A — nearest `(n, sign)` (the doc's prescription) | Path B — direct `(polar, azimuth)` write |
|---|---|---|
| 0° | 1.933° | **0.000°** |
| 90° | 2.706° | **0.000°** |
| 270° | 1.897° | **0.000°** |
| **mean, 24 (bearing × elevation) cases** | **0.972°** | **0.097°** (**10×**) |

**Mechanism:** the golden spiral couples latitude and azimuth through ONE index.
Reaching `y ≈ 0` (horizontal) needs `n ≈ N−1`, and those few `n` have their
azimuth *already fixed* at `n·φ` — **you cannot independently choose a bearing at
the horizon.** Compounding it, the lattice's latitude density is ∝ `sin(2·lat)`
(equal-area on the **disk**, not the sphere), i.e. **sparsest exactly at the
equator.** Surface normals — the case the doc was written for — spread over the
whole sphere and never hit this. Wind bearings cluster at the horizon and always
do.

Path B is near-exact there (`y = 0 → polar = 128` exactly; 16-bit azimuth =
0.0055° over 360°) and is licensed by the doctrine's **own** split: the `rim`
keeps carrying `(place, n)` as the metric, `(polar, azimuth)` carry the bearing,
and *"direction is place-INDEPENDENT."* The cost, stated: `(polar, azimuth)` are
no longer functions of the same `n` as `rim` — which under that split is the
intent, not a violation.

**Consequence:** a wind bearing encode should be the **direct field write**, not
the doc's `from_normal`. The doc's prescription is correct for its own case
(normals) and should be labelled as such rather than read as universal. **`[S]`
until an operator decides the API shape** — this section records the measurement
and the tradeoff, and deliberately does not mint a public `from_bearing`.


#### 12.14 — ⊘⊘ §12.13's VERDICT IS INVERTED: a normalized FIELD has different ergonomics than a single value

**Operator (2026-08-11):** *"you didn't factor in that due to normalized values
the field has different ergonomics than the single value — meaning AMX matmul,
tile ops etc."*

**Correct, and it reverses §12.13's recommendation.** §12.13 scored the two
bearing encodes by **angular reconstruction error** — *the operation §12.10
rules out and the whole substrate exists to avoid.* Third instance of the same
error in one document, and this one came three sections after writing the rule.

| | **Path A — nearest `n`** | **Path B — direct `(polar, azimuth)`** |
|---|---|---|
| direction collapses to | **ONE index**, 256-palette domain | 3 lanes, one 16-bit **circular** |
| field comparison | `rim.distance_adaptive` = 2 × `DistanceLut` u8 lookups — **L1 metric**, triangle inequality holds, CAKES/CLAM-safe | azimuth is **NOT a metric** — `distance.rs:8-10`: *"the 2π wrap … must never feed CAKES bounds"* |
| LUT | 256×256×u16 = **128 KB, L1/L2-resident, `U8x64`-friendly** (`distance.rs:12`) | 65536² is not a table |
| tile / AMX shape | a `&[u8]` plane → `ndarray::hpc::int8_tile_gemm::int8_gemm_amx_tiled(a_u8, b_i8, …) → [i32]` **directly** | no single-index plane to tile |
| must decode to compare | **no** | **yes** — the forbidden move |
| per-point angular error | 0.972° | **0.097°** (§12.13) |

**The resolution is a split by OPERATION, not a winner:**

- **Compare / search / correlate a field** — Path A. One index, L1-metric LUT,
  `U8x64`, AMX-tileable, never decodes. This is what *"pay the inbound tax
  once"* actually buys, and it is why palette256 is the pattern one rank down.
- **Materialize one bearing** — Path B, 10× finer. But *"never reconstruct per
  element when the representation is normalized"* makes this the rare path, not
  the design centre.

**Rule extracted (the generalizable one):** **judge a normalized
representation by what its FIELD does, not by what one element decodes to.** A
per-element accuracy number is the round-trip metric wearing a different hat —
and a representation that wins it can simultaneously destroy the index-domain
comparison, the metric guarantee, and the tile shape that made the substrate
worth building. §12.13's measurement stands; its *verdict* does not.
