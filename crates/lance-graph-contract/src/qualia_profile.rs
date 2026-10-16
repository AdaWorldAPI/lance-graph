//! Qualia Profile — an ICC-style profile for felt states.
//!
//! # Three spaces, one profile
//!
//! This workspace carries two different 17-axis vectors that are both called
//! "qualia". They are different *devices* and this module keeps them apart:
//!
//! | space | axes | what it measures | ICC reading |
//! |---|---|---|---|
//! | **device** — [`crate::qualia::AXIS_LABELS`] | arousal, valence, tension, warmth, clarity, boundary, depth, velocity, entropy, coherence, intimacy, presence, assertion, receptivity, groundedness, expansion, integration | convergence observables: what the cycle *computed* (`clarity = 1/cycles`) | print-head telemetry — the signal the engine actually emits |
//! | **PCS** — [`PCS_AXES`] | valence, arousal, dominance, warmth, brightness, tension, clarity, social, nostalgia, sacredness, desire, grief, awe, shame, hope, edge, resolution_hunger | phenomenological axes, hand-calibrated from the inside; 219 named states, each a measured patch | the profile connection space — device-independent, what the patch *is* under the reference illuminant |
//! | **observer** — e.g. the six Ekman archetypes in `cognitive-shader-driver::engine_bridge` | fear, anger, sadness, joy, surprise, disgust | what an outside observer adds up from behaviour | a second device (a 6-primary display) with its own profile |
//!
//! Reading in the workspace's own CMYK/RGB vocabulary: the PCS axes are the
//! subtractive side — grief, shame, tension are inks, each *absorbs* from a
//! white of full presence; the observer vocabulary is the additive side — an
//! outside eye summing emitted signals into "anger". Device telemetry is
//! neither; it is the mechanism. An ICC profile connects device ⇄ PCS. The
//! observer never touches the device directly.
//!
//! # What this module ships
//!
//! - the **named-colour table** ([`NAMED`], the ICC `ncl2` tag): 219 named
//!   qualia with PCS coordinates in hundredths, generated from
//!   `data/qualia/calibrated_219.json` and pinned by [`PROFILE_ID`] /
//!   [`TABLE_FNV`];
//! - the **palette**: one byte, index `0` = [`UNNAMED`], `1..=219` = the
//!   table, `220..=255` reserved — fits the existing one-byte Σ-codebook slot;
//! - **B2A on the PCS side**: [`nearest_named`] maps any PCS point to a palette
//!   index plus a residual. The residual is the **K channel** — the part of the
//!   felt state no name covers. It is the `classification_distance` the i4
//!   cutover stopped storing, measured against the *inner* vocabulary instead
//!   of the six observer archetypes;
//! - **gamut** per rendering device ([`Gamut`], 256-bit mask) and three
//!   **rendering intents** ([`RenderingIntent`]) adapted to a discrete palette;
//! - the **mix** ([`Mix`]): up to three named qualia with weights — a living
//!   state is rarely one patch (`ada:qualia:active` carries three);
//! - the five axes the two spaces share **by name** ([`SHARED_AXES`]) and the
//!   partial transform [`device_to_pcs_shared`].
//!
//! # What this module does NOT ship — the A2B CLUT
//!
//! The device → PCS transform for the twelve non-shared axes is **not a
//! constant**. In colour management it comes from printing a test chart and
//! measuring the patches. The analogue here is a probe: drive each of the 219
//! named states through the thinking engine, record the convergence 17D it
//! produces, and fit the CLUT from the 219 pairs. Until that probe has run the
//! only honest transform is the shared-axes one, and
//! [`device_to_pcs_shared`] says so in its return value (`measured` mask).
//! Even the shared axes are a CONJECTURE that same-named axes mean the same
//! thing; the probe falsifies or confirms it.
//!
//! # Determinism
//!
//! Palette indices are a pure function of the sorted `id` strings; PCS values
//! are exact in hundredths. Two fingerprints pin the pair (source, table) from
//! both sides: [`TABLE_FNV`] is re-computed from the Rust table (a hand-edited
//! row fails) and [`SOURCE_FNV`] is re-computed from the JSON itself, read with
//! `include_str!` (a source edited without re-running the generator fails).
//! [`PROFILE_ID`] is the sha256 of the same source bytes, kept as the ICC
//! profile-ID field for consumers that want to cite it; the in-crate check is
//! `SOURCE_FNV`. Two sessions building from the same calibration file produce
//! byte-identical tables.

use crate::qualia::{QualiaVector, QUALIA_DIMS};

pub use crate::qualia_profile_table::{NAMED, NAMED_COUNT, PROFILE_ID, SOURCE_FNV, TABLE_FNV};

// ═══════════════════════════════════════════════════════════════════════════
// PCS axes
// ═══════════════════════════════════════════════════════════════════════════

/// Dimensionality of the profile connection space.
pub const PCS_DIMS: usize = 17;

/// PCS axis labels, index-aligned with [`NamedQualia::pcs`]. Order is the
/// calibration source's order and is part of the contract.
pub const PCS_AXES: [&str; PCS_DIMS] = [
    "valence",           // 0
    "arousal",           // 1
    "dominance",         // 2
    "warmth",            // 3
    "brightness",        // 4
    "tension",           // 5
    "clarity",           // 6
    "social",            // 7
    "nostalgia",         // 8
    "sacredness",        // 9
    "desire",            // 10
    "grief",             // 11
    "awe",               // 12
    "shame",             // 13
    "hope",              // 14
    "edge",              // 15
    "resolution_hunger", // 16
];

/// A PCS coordinate: 17 axes in hundredths, `-100..=100`.
pub type Pcs = [i8; PCS_DIMS];

/// Palette index of the unnamed state (no patch matched / nothing written).
pub const UNNAMED: u8 = 0;

/// First reserved palette index. `UNNAMED < idx <= NAMED_COUNT` is a table row.
pub const RESERVED_FROM: u8 = (NAMED_COUNT + 1) as u8;

/// One row of the named-colour table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NamedQualia {
    /// Palette index, `1..=NAMED_COUNT`. Equal to table position + 1.
    pub idx: u8,
    /// Stable identifier, `family_subfamily_variant`. Sort key of the table.
    pub id: &'static str,
    /// Human label.
    pub label: &'static str,
    /// Family name (the first segment of `id`, as calibrated).
    pub family: &'static str,
    /// Subfamily tag.
    pub subfamily: &'static str,
    /// PCS coordinate in hundredths.
    pub pcs: Pcs,
    /// Felt-texture tags from the calibration.
    pub tags: &'static [&'static str],
}

/// Table row for a palette index, or `None` for [`UNNAMED`] and reserved indices.
#[inline]
pub fn named(idx: u8) -> Option<&'static NamedQualia> {
    if idx == UNNAMED || idx as usize > NAMED_COUNT {
        return None;
    }
    Some(&NAMED[idx as usize - 1])
}

/// Palette index for an `id`, or `None`.
pub fn index_of(id: &str) -> Option<u8> {
    NAMED
        .binary_search_by(|row| row.id.cmp(id))
        .ok()
        .map(|pos| (pos + 1) as u8)
}

/// L1 distance between two PCS points, in hundredths. Max `17 × 200 = 3400`.
#[inline]
pub fn pcs_l1(a: &Pcs, b: &Pcs) -> u16 {
    let mut d = 0u16;
    for i in 0..PCS_DIMS {
        d += (a[i] as i16 - b[i] as i16).unsigned_abs();
    }
    d
}

// ═══════════════════════════════════════════════════════════════════════════
// B2A on the PCS side: nearest named + K residual
// ═══════════════════════════════════════════════════════════════════════════

/// Nearest named qualia to a PCS point, and the residual.
///
/// The residual is the **K channel**: how much of the felt state no name
/// covers, in hundredths of L1. `0` means the point *is* a patch. Ties go to
/// the lowest palette index, so the result is deterministic.
pub fn nearest_named(pcs: &Pcs) -> (u8, u16) {
    let mut best = (UNNAMED, u16::MAX);
    for row in NAMED.iter() {
        let d = pcs_l1(pcs, &row.pcs);
        if d < best.1 {
            best = (row.idx, d);
        }
    }
    best
}

/// Nearest named qualia **within a gamut**. Returns [`UNNAMED`] with
/// `u16::MAX` if the gamut is empty.
pub fn nearest_named_in(pcs: &Pcs, gamut: &Gamut) -> (u8, u16) {
    let mut best = (UNNAMED, u16::MAX);
    for row in NAMED.iter() {
        if !gamut.contains(row.idx) {
            continue;
        }
        let d = pcs_l1(pcs, &row.pcs);
        if d < best.1 {
            best = (row.idx, d);
        }
    }
    best
}

// ═══════════════════════════════════════════════════════════════════════════
// Gamut + rendering intents
// ═══════════════════════════════════════════════════════════════════════════

/// Which palette indices a rendering device (a mode, a persona, a voice) can
/// express. A 256-bit mask; bit `i` set ⇔ palette index `i` is in gamut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Gamut(pub [u64; 4]);

impl Gamut {
    pub const EMPTY: Self = Self([0; 4]);

    /// Every table row in gamut (not the reserved range, not `UNNAMED`).
    pub fn full() -> Self {
        let mut g = Self::EMPTY;
        for row in NAMED.iter() {
            g.insert(row.idx);
        }
        g
    }

    /// Gamut spanned by whole families (case-sensitive, as calibrated).
    pub fn from_families(families: &[&str]) -> Self {
        let mut g = Self::EMPTY;
        for row in NAMED.iter() {
            if families.contains(&row.family) {
                g.insert(row.idx);
            }
        }
        g
    }

    #[inline]
    pub fn contains(&self, idx: u8) -> bool {
        (self.0[(idx >> 6) as usize] >> (idx & 63)) & 1 == 1
    }

    #[inline]
    pub fn insert(&mut self, idx: u8) {
        self.0[(idx >> 6) as usize] |= 1u64 << (idx & 63);
    }

    #[inline]
    pub fn remove(&mut self, idx: u8) {
        self.0[(idx >> 6) as usize] &= !(1u64 << (idx & 63));
    }

    pub fn len(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.0 == [0; 4]
    }
}

/// How an out-of-gamut named qualia is rendered on a device that cannot
/// express it. Adapted from ICC to a discrete palette:
///
/// - `AbsoluteColorimetric` — no adaptation: the index is returned as-is and
///   `clipped` reports whether the device can show it. The raw record.
/// - `RelativeColorimetric` — in-gamut indices are exact; an out-of-gamut index
///   goes to the nearest in-gamut row **of the same family** (relationships
///   preserved), falling back to the nearest in-gamut row overall only if the
///   family has none in gamut.
/// - `Perceptual` — nearest in-gamut row by PCS distance regardless of family
///   (closest felt match; a name may change).
///
/// ICC's fourth intent, *saturation*, has no settled meaning for a felt-state
/// palette and is deliberately not defined here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderingIntent {
    AbsoluteColorimetric,
    RelativeColorimetric,
    Perceptual,
}

/// Result of [`render`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rendered {
    /// Palette index actually shown.
    pub idx: u8,
    /// `true` if `idx` differs from the request or the request was out of gamut.
    pub clipped: bool,
    /// PCS L1 between requested and shown rows (`0` when unchanged).
    pub shift: u16,
}

/// Render a palette index on a device with the given gamut and intent.
///
/// [`UNNAMED`] and reserved indices render as themselves, `clipped = false`,
/// since there is no PCS point to move. An empty gamut renders everything as
/// [`UNNAMED`] with `clipped = true`.
pub fn render(idx: u8, gamut: &Gamut, intent: RenderingIntent) -> Rendered {
    let Some(row) = named(idx) else {
        return Rendered {
            idx,
            clipped: false,
            shift: 0,
        };
    };
    let in_gamut = gamut.contains(idx);
    match intent {
        RenderingIntent::AbsoluteColorimetric => Rendered {
            idx,
            clipped: !in_gamut,
            shift: 0,
        },
        RenderingIntent::RelativeColorimetric => {
            if in_gamut {
                return Rendered {
                    idx,
                    clipped: false,
                    shift: 0,
                };
            }
            let mut best = (UNNAMED, u16::MAX);
            for cand in NAMED.iter() {
                if cand.family == row.family && gamut.contains(cand.idx) {
                    let d = pcs_l1(&row.pcs, &cand.pcs);
                    if d < best.1 {
                        best = (cand.idx, d);
                    }
                }
            }
            if best.0 == UNNAMED {
                best = nearest_named_in(&row.pcs, gamut);
            }
            Rendered {
                idx: best.0,
                clipped: true,
                shift: if best.0 == UNNAMED { 0 } else { best.1 },
            }
        }
        RenderingIntent::Perceptual => {
            let (b, d) = nearest_named_in(&row.pcs, gamut);
            Rendered {
                idx: b,
                clipped: b != idx,
                shift: if b == UNNAMED { 0 } else { d },
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mix — up to three inks
// ═══════════════════════════════════════════════════════════════════════════

/// A living state as a weighted mix of up to three named qualia. Weights are
/// `0..=255`; a slot with weight `0` or index [`UNNAMED`] is unused. Three
/// slots because that is what `ada:qualia:active` carries; 6 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mix {
    pub inks: [(u8, u8); 3],
}

impl Mix {
    pub const fn single(idx: u8) -> Self {
        Self {
            inks: [(idx, 255), (UNNAMED, 0), (UNNAMED, 0)],
        }
    }

    /// Weighted PCS blend of the active inks. `None` if no ink is active.
    pub fn pcs(&self) -> Option<Pcs> {
        let mut acc = [0i32; PCS_DIMS];
        let mut total = 0i32;
        for &(idx, w) in &self.inks {
            if w == 0 {
                continue;
            }
            let Some(row) = named(idx) else { continue };
            for (a, &q) in acc.iter_mut().zip(row.pcs.iter()) {
                *a += q as i32 * w as i32;
            }
            total += w as i32;
        }
        if total == 0 {
            return None;
        }
        let mut out = [0i8; PCS_DIMS];
        for (o, &v) in out.iter_mut().zip(acc.iter()) {
            // round half away from zero, then clamp to the i8 PCS range
            let r = if v >= 0 {
                (v + total / 2) / total
            } else {
                (v - total / 2) / total
            };
            *o = r.clamp(-100, 100) as i8;
        }
        Some(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared axes between device and PCS
// ═══════════════════════════════════════════════════════════════════════════

/// Axes present under the same name in both the device space
/// ([`AXIS_LABELS`]) and the PCS ([`PCS_AXES`]): `(device_index, pcs_index)`.
/// That the names coincide is a fact of both tables; that the *meanings*
/// coincide is a CONJECTURE until the chart probe runs.
pub const SHARED_AXES: [(usize, usize); 5] = [
    (1, 0), // valence
    (0, 1), // arousal
    (2, 5), // tension
    (3, 3), // warmth
    (4, 6), // clarity
];

/// Partial device → PCS transform over the shared axes only.
///
/// Returns the PCS point and a `measured` mask: `true` for the five shared
/// axes, `false` for the twelve the profile has no CLUT for (left at `0`).
/// Device values are clamped to `[-1, 1]` and quantised to hundredths.
pub fn device_to_pcs_shared(device: &QualiaVector) -> (Pcs, [bool; PCS_DIMS]) {
    let mut pcs = [0i8; PCS_DIMS];
    let mut measured = [false; PCS_DIMS];
    for &(d, p) in &SHARED_AXES {
        let v = device[d].clamp(-1.0, 1.0);
        pcs[p] = (v * 100.0).round() as i8;
        measured[p] = true;
    }
    (pcs, measured)
}

const _: () = assert!(QUALIA_DIMS == 17);
const _: () = assert!(NAMED_COUNT <= 254);

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qualia::AXIS_LABELS;

    fn fnv1a64(bytes: impl Iterator<Item = u8>) -> u64 {
        let mut h = 0xCBF29CE484222325u64;
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001B3);
        }
        h
    }

    /// The calibration source, read at compile time so the test sees the file
    /// as committed — not a copy the generator made.
    const SOURCE_JSON: &str = include_str!("../../../data/qualia/calibrated_219.json");

    #[test]
    fn table_matches_the_rust_table_it_was_generated_into() {
        // Re-computes the generator's fingerprint from the Rust table. A single
        // hand-edited hundredth, or a reordered row, changes this value.
        let h = fnv1a64(
            NAMED
                .iter()
                .flat_map(|r| r.id.bytes().chain(r.pcs.iter().map(|&q| q as u8))),
        );
        assert_eq!(
            h, TABLE_FNV,
            "qualia_profile_table.rs was edited by hand; re-run scripts/gen_qualia_profile.py"
        );
    }

    #[test]
    fn table_is_not_stale_against_the_json_source() {
        // Re-computes the generator's fingerprint of the SOURCE bytes from the
        // file itself. Editing calibrated_219.json without re-running the
        // generator changes this value while TABLE_FNV stays put — the drift the
        // previous test cannot see.
        let h = fnv1a64(SOURCE_JSON.bytes());
        assert_eq!(
            h, SOURCE_FNV,
            "data/qualia/calibrated_219.json changed; re-run scripts/gen_qualia_profile.py"
        );
        // And the row count the source declares is the row count the table has.
        let declared = SOURCE_JSON
            .split("\"count\":")
            .nth(1)
            .and_then(|s| s.trim_start().split(',').next())
            .and_then(|n| n.trim().parse::<usize>().ok())
            .expect("source JSON declares a count");
        assert_eq!(declared, NAMED_COUNT);
    }

    #[test]
    fn table_is_sorted_dense_and_unique() {
        assert_eq!(NAMED.len(), NAMED_COUNT);
        for (pos, row) in NAMED.iter().enumerate() {
            assert_eq!(row.idx as usize, pos + 1, "{}", row.id);
            assert!(!row.family.is_empty(), "{}", row.id);
            if pos > 0 {
                assert!(NAMED[pos - 1].id < row.id, "not sorted at {}", row.id);
            }
        }
        assert_eq!(named(UNNAMED), None);
        assert_eq!(named(RESERVED_FROM), None);
        assert_eq!(named(255), None);
        assert_eq!(index_of("anger_clean_no"), Some(1));
        assert_eq!(index_of("not_a_qualia"), None);
    }

    #[test]
    fn shared_axes_carry_the_same_name_in_both_tables() {
        for &(d, p) in &SHARED_AXES {
            assert_eq!(AXIS_LABELS[d], PCS_AXES[p], "device {d} / pcs {p}");
        }
        // And exactly five names coincide — a sixth appearing in either table
        // without being listed here is drift.
        let n = AXIS_LABELS.iter().filter(|a| PCS_AXES.contains(a)).count();
        assert_eq!(n, SHARED_AXES.len());
    }

    #[test]
    fn nearest_named_is_exact_on_members_and_discriminates() {
        let mut seen = [false; 256];
        for row in NAMED.iter() {
            let (idx, k) = nearest_named(&row.pcs);
            assert_eq!(idx, row.idx, "{}", row.id);
            assert_eq!(k, 0, "{}", row.id);
            seen[idx as usize] = true;
        }
        // anti-vacuity: 219 inputs → 219 distinct outputs
        assert_eq!(seen.iter().filter(|&&s| s).count(), NAMED_COUNT);
    }

    #[test]
    fn k_residual_fires_on_perturbation_and_stays_silent_on_a_patch() {
        let row = &NAMED[42];
        let (_, silent) = nearest_named(&row.pcs);
        assert_eq!(silent, 0);
        // Perturb one axis by 0.10. If no other patch is within L1 10 of this
        // one the nearest stays the same row and K reports exactly 10.
        let mut p = row.pcs;
        p[11] = (p[11] + 10).min(100); // grief
        let (idx, k) = nearest_named(&p);
        assert!(k > 0, "K must be non-zero off a patch");
        if idx == row.idx {
            assert_eq!(k, 10);
        } else {
            assert!(
                k < 10,
                "a different patch was chosen, so it must be strictly closer"
            );
        }
    }

    #[test]
    fn gamut_from_families_is_nontrivial_and_full_is_everything() {
        let g = Gamut::from_families(&["Trust", "Stillness"]);
        assert_eq!(g.len(), 10);
        assert!(
            (g.len() as usize) * 3 < NAMED_COUNT,
            "a two-family gamut must exclude most rows"
        );
        assert!(!g.contains(UNNAMED));
        assert!(!g.contains(RESERVED_FROM));
        assert_eq!(Gamut::full().len() as usize, NAMED_COUNT);
        assert!(Gamut::EMPTY.is_empty());
        let mut h = g;
        h.remove(index_of("trust_without_proof").unwrap());
        assert_eq!(h.len(), 9);
    }

    #[test]
    fn absolute_intent_can_clip_and_can_pass() {
        let g = Gamut::from_families(&["Trust"]);
        let inside = index_of("trust_without_proof").unwrap();
        let outside = index_of("anger_feral_burst").unwrap();
        let a = render(inside, &g, RenderingIntent::AbsoluteColorimetric);
        assert_eq!(
            a,
            Rendered {
                idx: inside,
                clipped: false,
                shift: 0
            }
        );
        let b = render(outside, &g, RenderingIntent::AbsoluteColorimetric);
        assert_eq!(
            b,
            Rendered {
                idx: outside,
                clipped: true,
                shift: 0
            }
        );
    }

    #[test]
    fn relative_and_perceptual_differ_on_some_input_and_agree_on_others() {
        // Search the table for a (request, gamut) where the nearest in-gamut row
        // overall is in a different family than the nearest in-gamut row of the
        // request's own family. Build the gamut as {one same-family row, one
        // other-family row} and demand the two intents disagree there.
        let mut found_differ = false;
        let mut found_agree = false;
        'outer: for req in NAMED.iter() {
            for same in NAMED
                .iter()
                .filter(|r| r.family == req.family && r.idx != req.idx)
            {
                for other in NAMED.iter().filter(|r| r.family != req.family) {
                    let mut g = Gamut::EMPTY;
                    g.insert(same.idx);
                    g.insert(other.idx);
                    let rel = render(req.idx, &g, RenderingIntent::RelativeColorimetric);
                    let per = render(req.idx, &g, RenderingIntent::Perceptual);
                    assert!(rel.clipped && per.clipped);
                    assert_eq!(rel.idx, same.idx, "relative must stay in family");
                    if per.idx != rel.idx {
                        assert!(per.shift < rel.shift, "perceptual chose a farther row");
                        found_differ = true;
                    } else {
                        found_agree = true;
                    }
                    if found_differ && found_agree {
                        break 'outer;
                    }
                }
            }
        }
        assert!(
            found_differ,
            "intents never differ — the knob is decoration"
        );
        assert!(
            found_agree,
            "intents never agree — relative is not colorimetric"
        );
    }

    #[test]
    fn empty_gamut_renders_unnamed_and_reserved_passes_through() {
        let r = render(7, &Gamut::EMPTY, RenderingIntent::Perceptual);
        assert_eq!(
            r,
            Rendered {
                idx: UNNAMED,
                clipped: true,
                shift: 0
            }
        );
        let r = render(
            RESERVED_FROM,
            &Gamut::EMPTY,
            RenderingIntent::RelativeColorimetric,
        );
        assert_eq!(
            r,
            Rendered {
                idx: RESERVED_FROM,
                clipped: false,
                shift: 0
            }
        );
    }

    #[test]
    fn mix_blends_and_single_is_identity() {
        let a = index_of("stillness_safe_void").unwrap();
        let b = index_of("surrender_soft_collapse").unwrap();
        assert_eq!(Mix::single(a).pcs(), Some(named(a).unwrap().pcs));
        let m = Mix {
            inks: [(a, 128), (b, 128), (UNNAMED, 0)],
        };
        let p = m.pcs().unwrap();
        let (pa, pb) = (named(a).unwrap().pcs, named(b).unwrap().pcs);
        for i in 0..PCS_DIMS {
            let mid = (pa[i] as i32 + pb[i] as i32) as f32 / 2.0;
            assert!(
                (p[i] as f32 - mid).abs() <= 0.5,
                "axis {i}: {} vs {mid}",
                p[i]
            );
        }
        assert_eq!(Mix::default().pcs(), None);
        // a mix is a point off every patch unless the inks coincide — K > 0
        let (_, k) = nearest_named(&p);
        assert!(k > 0);
    }

    #[test]
    fn device_to_pcs_fills_exactly_the_shared_axes() {
        let mut dev: QualiaVector = [0.0; QUALIA_DIMS];
        dev[1] = 0.5; // valence
        dev[0] = -0.25; // arousal
        dev[4] = 1.7; // clarity, out of range → clamped
        let (pcs, measured) = device_to_pcs_shared(&dev);
        assert_eq!(measured.iter().filter(|&&m| m).count(), 5);
        assert_eq!(pcs[0], 50);
        assert_eq!(pcs[1], -25);
        assert_eq!(pcs[6], 100);
        for (p, &m) in pcs.iter().zip(measured.iter()) {
            if !m {
                assert_eq!(*p, 0);
            }
        }
    }
}
