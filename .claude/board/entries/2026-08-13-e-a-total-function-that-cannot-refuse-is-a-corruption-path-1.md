## 2026-08-13 — E-A-TOTAL-FUNCTION-THAT-CANNOT-REFUSE-IS-A-CORRUPTION-PATH-1

**Status:** FINDING `[G]` — two measured instances, one crate, one hour.
Found by codex review on PR #948; the second by following the first to its
class. **Confidence:** High. Mechanism measured, not inferred.

**The shape.** `CalibratedFloor::quantize(f64) -> u8` is **total**: every input
returns a valid-looking bucket. It has no way to say *"that was not a
measurement."* Measured behaviour on non-finite input:

| input | bucket | why |
|---|---|---|
| `NaN` | **0** | `f64::clamp` **propagates** `NaN` rather than clamping it, then Rust's float→int cast saturates and sends `NaN` to zero |
| `-inf` | **0** | clamps to the low rim |
| `+inf` | **255** | clamps to the high rim |

Every one of those is a **legitimate** bucket. `0` and `255` are the ordinary
saturation values; `bucket_center(0)` is a real number near `lo`. Nothing
downstream can distinguish the result from a genuine reading.

**Why this was live and not theoretical.** ARCO-ERA5 is sparse *by design*:
`probes/weather-p1/README.md` §1 records `fill_value: NaN`, that several
variables 404 at the arc's **own fixture timestep**, and that in Zarr v2 a
missing chunk means all-`fill_value` — so **a 404 is valid store semantics, not
a fetch failure**, and *"any ingest must treat 404 as data."* The 404-ing list
at that timestep includes five variables the W1 field set actually packs.

**Instance 1 — the store path** (`lane.rs::pack_facet`, codex P1). An absent
field would have been written as plausible low-bucket measurements and read
back through `bucket_center` as ordinary numbers. This is the reserved-slot
rule — *"a reserved slot must not read back as a plausible number"* — defeated
one level deeper, where the existing guard could not see it.

**Instance 2 — the INSTRUMENT path, and it is worse**
(`floor.rs::saturation_of`). That function scores *"an ARBITRARY external
population"* by counting rim buckets — and non-finite input lands **on the
rim**. An all-`NaN` population would have scored `1.0`: **"completely
saturated" when the truth is "no data at all."** Those are opposite findings
and the bare fraction could not tell them apart. It is bar B2's instrument, so
the corruption would have propagated into a *measurement* rather than a stored
value.

> **The sharpening worth keeping: a corrupted stored value is bad; a corrupted
> INSTRUMENT is worse.** A bad value is one wrong row. A bad instrument is
> every conclusion drawn with it, each of which looks sound and carries no
> trace of the defect. When a finding lands on a total function, check its
> *measurement* call sites before its storage call sites.

**The other half of the same mistake — do not silently drop.** Skipping
non-finite values without reporting them is equally wrong: the caller never
learns the population was partly or wholly absent. The fix therefore
**reports**: `saturation_of` returns `SaturationScore { fraction, finite,
non_finite }`, matching this crate's standing shape (`calibrate` and `decode`
return `None` on a degenerate case rather than inventing a number) and the
`D-WXS-12` rule that *the degenerate case must be reported, never folded as
`0.0`*.

**Where the guard belongs.** At the boundary where an external value enters
the register — not inside the hot primitive. `quantize` keeps its signature
(changing it ripples through every call site); `pack_facet` refuses, and
`saturation_of` excludes-and-counts. `calibrate` was checked and is **clean**
— it already filters `is_finite`, so the hole never reached calibration. Every
`quantize` call site in the crate is now either guarded or provably finite.

**Generalizable check, cheap to run:** for every total function that maps a
wider domain onto a narrower one — quantisers, clamps, `as` casts,
`unwrap_or`, saturating arithmetic — ask *what does an invalid input return,
and is that return distinguishable from a valid one?* If the answer is "a
valid-looking value", the function cannot refuse, and every call site is a
corruption path until one of them does.

**Cross-ref:** `E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1`;
`E-A-DISABLE-PROBE-CAN-ITSELF-BE-VACUOUS-1` (same session, the verification
layer); `.claude/plans/weather-soa-bake-v1.md` §4 bar B2 (the instrument);
`probes/weather-p1/README.md` §1 (the store semantics); PR #948.

---

