//! The ClassView-side field manifest: `(facet, pair, byte) <-> (ERA5
//! variable, level, unit, floor id)`.
//!
//! # Why this exists (le-contract.md §2, slot purity)
//!
//! **"Labels and positions come from the ClassView, NEVER from a slot in the
//! payload."** The `NodeRow`'s value slab is dumb bytes; a byte carries no
//! variable name, no level number, no unit, no display label. The mapping
//! *"F1 pair 0 hi-byte = `v_component_of_wind` at 850 hPa, unit m/s, floor id
//! `v850`"* cannot live in the row — it lives here, as a committed data
//! artifact the bake reads. This module is ClassView-side metadata; nothing
//! it describes is ever written into a `NodeRow`.
//!
//! # The lane (plan `.claude/plans/weather-soa-bake-v1.md` §2.2)
//!
//! L4, `6 x (8:8)` palette256² (`.claude/v3/soa_layout/le-contract.md` §3):
//! each 16-byte facet is a 4-byte classid prefix followed by six `(lo, hi)`
//! byte pairs. `u8:u8` means two SEPARATE bytes — never widened to u16. Each
//! byte carries one ERA5 field quantised to 256 levels; a pair holds a
//! physically paired couple (e.g. wind components), so the pair's own
//! similarity is one 256×256 table read.
//!
//! # The committed W1 field set (plan §2.5)
//!
//! Three facets, F0 (surface), F1 (850 hPa), F2 (500 hPa); five occupied
//! pairs in F0 and three each in F1/F2. Every other pair is reserved-zero —
//! **absent from this manifest entirely**, per the "reserved slots are NOT
//! rows" rule: a slot with no entry is dormant by construction, expandable
//! later with zero layout change (RESERVE, DON'T RECLAIM).
//!
//! # `lo`/`hi` convention
//!
//! Within a pair written `A : B` in the plan's table, `A` is the LOW byte
//! and `B` is the HIGH byte — the pair's own left-to-right order, kept
//! consistent across the whole committed manifest (see the comment header
//! of `data/field_manifest_v1.tsv`).
//!
//! # Bar B0 — what is implemented here, and what is deferred
//!
//! Plan §4 W0's bar B0 is *"mutating one manifest entry must change the
//! bytes the bake writes for at least one cell."* **The bake does not exist
//! yet** (`D-WXS-4`, a separate, later deliverable), so that end-to-end half
//! cannot be exercised from this module. What IS implemented and tested
//! here is the manifest-level half of B0: resolution is real (a lookup
//! finds the right slot and nothing else), and a manifest whose two entries
//! collide on one `(facet, pair, byte)` slot is rejected outright. The
//! stay-silent twin — a manifest that differs byte-for-byte (reordered
//! rows) but is semantically identical still resolves every lookup
//! identically — is also tested. Wiring bar B0's end-to-end half is `D-WXS-4`'s
//! job, not this module's.

use std::fmt;

/// The exact header line every field-manifest TSV must start with (after any
/// leading `#` comment / blank lines, which are skipped).
const HEADER: &str = "facet\tpair\tbyte\tvariable\tlevel_hpa\tunit\tfloor_id";

/// The committed W1 field manifest, embedded at compile time so this crate
/// stays dependency-free (no TSV/CSV crate, no serde — plan §6.3).
const COMMITTED_MANIFEST_TSV: &str = include_str!("../data/field_manifest_v1.tsv");

/// Which byte of an `(8:8)` pair a manifest entry occupies.
///
/// `u8:u8` means two separate bytes, never a widened `u16` — le-contract.md
/// §1/§3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairByte {
    /// The low byte of the pair (the first-listed variable in the plan's
    /// `A : B` table entries).
    Lo,
    /// The high byte of the pair (the second-listed variable).
    Hi,
}

impl fmt::Display for PairByte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PairByte::Lo => f.write_str("lo"),
            PairByte::Hi => f.write_str("hi"),
        }
    }
}

/// One row of the field manifest: the (facet, pair, byte) slot a single
/// ERA5 field occupies, plus its physical unit and calibration floor id.
///
/// Nothing on this type is ever written into a `NodeRow` — it is
/// ClassView-side metadata the bake and any reader consults, per the
/// module-level slot-purity note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestEntry {
    /// Facet index: `0` = F0 (surface), `1` = F1 (850 hPa), `2` = F2
    /// (500 hPa). Always `0..=2` for a validated manifest.
    pub facet: u8,
    /// Pair index within the facet, `0..=5`.
    pub pair: u8,
    /// Which byte of the pair this entry occupies.
    pub byte: PairByte,
    /// The ERA5 variable name, exactly as it appears in the Zarr store
    /// (e.g. `u_component_of_wind`, `mean_sea_level_pressure`).
    pub variable: String,
    /// The pressure level in hPa, or `None` for a surface field (the TSV's
    /// literal `surface` token).
    pub level_hpa: Option<u16>,
    /// The physical unit (e.g. `Pa`, `K`, `m s-1`, `kg kg-1`, `m2 s-2`, `1`
    /// for a dimensionless fraction).
    pub unit: String,
    /// A stable identifier for the calibration floor this field uses.
    /// Two entries sharing a `floor_id` share the same `[lo, hi]`
    /// calibration (plan §2.3/D-WXS-3); none do in the committed W1
    /// manifest today.
    pub floor_id: String,
}

/// Everything that can go wrong parsing or validating field-manifest text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// The first non-comment, non-blank line did not match [`HEADER`]
    /// exactly, or no such line existed at all (`line == 0`).
    BadHeader {
        /// 1-based source line number (`0` if no header line was found at
        /// all).
        line: usize,
        /// The line actually found, if any.
        found: String,
    },
    /// A data row did not split into exactly 7 tab-separated columns.
    WrongColumnCount {
        /// 1-based source line number.
        line: usize,
        /// How many columns were actually found.
        found: usize,
    },
    /// The `facet` column was not `0`, `1`, or `2`.
    FacetOutOfRange {
        /// 1-based source line number.
        line: usize,
        /// The raw column text.
        value: String,
    },
    /// The `pair` column was not in `0..=5`.
    PairOutOfRange {
        /// 1-based source line number.
        line: usize,
        /// The raw column text.
        value: String,
    },
    /// The `byte` column was neither `lo` nor `hi`.
    BadByteToken {
        /// 1-based source line number.
        line: usize,
        /// The raw column text.
        value: String,
    },
    /// The `level_hpa` column was neither the literal `surface` nor a
    /// parseable non-negative integer.
    BadLevel {
        /// 1-based source line number.
        line: usize,
        /// The raw column text.
        value: String,
    },
    /// Two entries claim the same `(facet, pair, byte)` slot — the manifest
    /// is not load-bearing if this is possible (bar B0).
    SlotCollision {
        /// The colliding facet index.
        facet: u8,
        /// The colliding pair index.
        pair: u8,
        /// The colliding byte.
        byte: PairByte,
        /// The variable name of the first entry claiming the slot.
        first_variable: String,
        /// The variable name of the second entry claiming the slot.
        second_variable: String,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::BadHeader { line, found } => write!(
                f,
                "manifest line {line}: expected header {HEADER:?}, found {found:?}"
            ),
            ManifestError::WrongColumnCount { line, found } => write!(
                f,
                "manifest line {line}: expected 7 tab-separated columns, found {found}"
            ),
            ManifestError::FacetOutOfRange { line, value } => {
                write!(f, "manifest line {line}: facet {value:?} is not 0, 1, or 2")
            }
            ManifestError::PairOutOfRange { line, value } => {
                write!(f, "manifest line {line}: pair {value:?} is not in 0..=5")
            }
            ManifestError::BadByteToken { line, value } => write!(
                f,
                "manifest line {line}: byte token {value:?} is neither \"lo\" nor \"hi\""
            ),
            ManifestError::BadLevel { line, value } => write!(
                f,
                "manifest line {line}: level_hpa {value:?} is neither \"surface\" nor an integer"
            ),
            ManifestError::SlotCollision {
                facet,
                pair,
                byte,
                first_variable,
                second_variable,
            } => write!(
                f,
                "manifest slot (facet={facet}, pair={pair}, byte={byte}) is claimed by both {first_variable:?} and {second_variable:?}"
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

/// A parsed, validated field manifest: an ordered list of [`ManifestEntry`]
/// rows plus the two resolution directions over them.
#[derive(Debug, Clone)]
pub struct FieldManifest {
    entries: Vec<ManifestEntry>,
}

impl FieldManifest {
    /// Parses and validates field-manifest text.
    ///
    /// Lines that are blank or start with `#` are skipped wherever they
    /// occur. The first remaining line must equal [`HEADER`] exactly (a
    /// permissive "starts with" check would be a silent-misread waiting
    /// for a schema change). Every remaining line is a data row of exactly
    /// 7 tab-separated columns. After every row parses, the full set is
    /// validated for `(facet, pair, byte)` collisions.
    ///
    /// # Errors
    ///
    /// Returns the first [`ManifestError`] encountered — a malformed
    /// header, a malformed row, or (only after every row has parsed
    /// successfully) a slot collision between two otherwise-valid rows.
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        let mut header_seen = false;
        let mut entries = Vec::new();

        for (line_no, raw_line) in text.lines().enumerate().map(|(i, l)| (i + 1, l)) {
            let line = raw_line.trim_end();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if !header_seen {
                if line != HEADER {
                    return Err(ManifestError::BadHeader {
                        line: line_no,
                        found: line.to_string(),
                    });
                }
                header_seen = true;
                continue;
            }

            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() != 7 {
                return Err(ManifestError::WrongColumnCount {
                    line: line_no,
                    found: cols.len(),
                });
            }

            let facet: u8 = cols[0]
                .parse::<u8>()
                .ok()
                .filter(|v| *v <= 2)
                .ok_or_else(|| ManifestError::FacetOutOfRange {
                    line: line_no,
                    value: cols[0].to_string(),
                })?;

            let pair: u8 = cols[1]
                .parse::<u8>()
                .ok()
                .filter(|v| *v <= 5)
                .ok_or_else(|| ManifestError::PairOutOfRange {
                    line: line_no,
                    value: cols[1].to_string(),
                })?;

            let byte = match cols[2] {
                "lo" => PairByte::Lo,
                "hi" => PairByte::Hi,
                other => {
                    return Err(ManifestError::BadByteToken {
                        line: line_no,
                        value: other.to_string(),
                    })
                }
            };

            let variable = cols[3].to_string();

            let level_hpa = if cols[4] == "surface" {
                None
            } else {
                Some(
                    cols[4]
                        .parse::<u16>()
                        .map_err(|_| ManifestError::BadLevel {
                            line: line_no,
                            value: cols[4].to_string(),
                        })?,
                )
            };

            let unit = cols[5].to_string();
            let floor_id = cols[6].to_string();

            entries.push(ManifestEntry {
                facet,
                pair,
                byte,
                variable,
                level_hpa,
                unit,
                floor_id,
            });
        }

        if !header_seen {
            return Err(ManifestError::BadHeader {
                line: 0,
                found: String::new(),
            });
        }

        validate_no_slot_collisions(&entries)?;

        Ok(FieldManifest { entries })
    }

    /// The committed W1 field manifest (`data/field_manifest_v1.tsv`).
    ///
    /// # Panics
    ///
    /// Panics if the committed artifact fails to parse or validate. That is
    /// a build-time invariant, not a runtime one — the committed TSV is
    /// meant to always be well-formed, and a session breaking it should see
    /// the failure immediately rather than downstream. Use [`Self::parse`]
    /// directly for a fallible read of arbitrary manifest text.
    pub fn committed() -> Self {
        Self::parse(COMMITTED_MANIFEST_TSV)
            .expect("committed data/field_manifest_v1.tsv must parse and validate")
    }

    /// Every entry in the manifest, in file order.
    pub fn entries(&self) -> &[ManifestEntry] {
        &self.entries
    }

    /// Resolves `(variable, level_hpa)` to the `(facet, pair, byte)` slot it
    /// occupies, or `None` if this manifest has no entry for it (e.g. a
    /// pressure-level variable at a level outside the W1 set).
    pub fn resolve(&self, variable: &str, level_hpa: Option<u16>) -> Option<(u8, u8, PairByte)> {
        self.entries
            .iter()
            .find(|e| e.variable == variable && e.level_hpa == level_hpa)
            .map(|e| (e.facet, e.pair, e.byte))
    }

    /// The inverse of [`Self::resolve`]: which entry (if any) occupies a
    /// given `(facet, pair, byte)` slot.
    pub fn resolve_slot(&self, facet: u8, pair: u8, byte: PairByte) -> Option<&ManifestEntry> {
        self.entries
            .iter()
            .find(|e| e.facet == facet && e.pair == pair && e.byte == byte)
    }
}

/// Rejects a manifest where two entries claim the same `(facet, pair,
/// byte)` slot. `O(n^2)` over the entry count, which is fine at the W1
/// scale (22 entries) and simple enough to audit by inspection.
fn validate_no_slot_collisions(entries: &[ManifestEntry]) -> Result<(), ManifestError> {
    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            let a = &entries[i];
            let b = &entries[j];
            if a.facet == b.facet && a.pair == b.pair && a.byte == b.byte {
                return Err(ManifestError::SlotCollision {
                    facet: a.facet,
                    pair: a.pair,
                    byte: a.byte,
                    first_variable: a.variable.clone(),
                    second_variable: b.variable.clone(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses and validates: the committed TSV loads, every entry is in
    /// range, and the per-facet slot count matches the plan §2.5 table
    /// exactly (`==`, not `>=` — a permissive arity guard is a silent
    /// misread waiting for a schema change).
    #[test]
    fn committed_manifest_parses_and_every_entry_is_in_range() {
        let manifest = FieldManifest::committed();

        assert_eq!(manifest.entries().len(), 22);

        for entry in manifest.entries() {
            assert!(entry.facet <= 2, "facet {} out of range", entry.facet);
            assert!(entry.pair <= 5, "pair {} out of range", entry.pair);
            assert!(!entry.variable.is_empty());
            assert!(!entry.unit.is_empty());
            assert!(!entry.floor_id.is_empty());
        }

        let count = |facet: u8| {
            manifest
                .entries()
                .iter()
                .filter(|e| e.facet == facet)
                .count()
        };
        assert_eq!(count(0), 10, "F0 (surface): 5 occupied pairs x 2 bytes");
        assert_eq!(count(1), 6, "F1 (850 hPa): 3 occupied pairs x 2 bytes");
        assert_eq!(count(2), 6, "F2 (500 hPa): 3 occupied pairs x 2 bytes");
    }

    /// Can-it-fire: a manifest with two entries claiming the same
    /// `(facet, pair, byte)` slot must be rejected.
    #[test]
    fn colliding_entries_are_rejected() {
        let text = format!(
            "{HEADER}\n0\t0\tlo\tmean_sea_level_pressure\tsurface\tPa\tmsl\n0\t0\tlo\tsurface_pressure\tsurface\tPa\tsp\n"
        );

        let err = FieldManifest::parse(&text).expect_err("colliding slots must be rejected");
        match err {
            ManifestError::SlotCollision {
                facet, pair, byte, ..
            } => {
                assert_eq!(facet, 0);
                assert_eq!(pair, 0);
                assert_eq!(byte, PairByte::Lo);
            }
            other => panic!("expected SlotCollision, got {other:?}"),
        }
    }

    /// Stay-silent twin of the collision guard: two entries on DIFFERENT
    /// bytes of the same pair must be accepted, not flagged as a collision.
    /// A validator that rejects everything carries exactly as much
    /// information as one that accepts everything.
    #[test]
    fn non_colliding_entries_on_the_same_pair_are_accepted() {
        let text = format!(
            "{HEADER}\n0\t0\tlo\tmean_sea_level_pressure\tsurface\tPa\tmsl\n0\t0\thi\tsurface_pressure\tsurface\tPa\tsp\n"
        );

        let manifest = FieldManifest::parse(&text).expect("non-colliding rows must parse");
        assert_eq!(manifest.entries().len(), 2);
    }

    /// Stay-silent twin (non-trivial, over the real committed data): a
    /// manifest that is byte-different — the data rows in reverse order —
    /// but semantically identical to the committed one must be accepted
    /// AND must resolve every lookup identically. Mutating a reserved-zero
    /// slot's description or reordering rows must change not one resolved
    /// answer.
    #[test]
    fn reordering_the_committed_rows_changes_no_resolution() {
        let lines: Vec<&str> = COMMITTED_MANIFEST_TSV.lines().collect();
        let header_idx = lines
            .iter()
            .position(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .expect("committed manifest must have a header line");
        let (head, data) = lines.split_at(header_idx + 1);
        let mut reordered_data: Vec<&str> = data.to_vec();
        reordered_data.reverse();

        let mut reordered_text = head.join("\n");
        reordered_text.push('\n');
        reordered_text.push_str(&reordered_data.join("\n"));
        reordered_text.push('\n');

        // Sanity: the text really is byte-different from the original.
        assert_ne!(reordered_text, COMMITTED_MANIFEST_TSV);

        let original = FieldManifest::committed();
        let reordered =
            FieldManifest::parse(&reordered_text).expect("reordered manifest must still validate");

        assert_eq!(original.entries().len(), reordered.entries().len());
        for entry in original.entries() {
            assert_eq!(
                original.resolve(&entry.variable, entry.level_hpa),
                reordered.resolve(&entry.variable, entry.level_hpa),
                "resolution for {:?} at {:?} diverged after reordering",
                entry.variable,
                entry.level_hpa
            );
        }
    }

    /// Resolution is real: a known `(variable, level)` resolves to its
    /// documented slot, and the inverse lookup round-trips back to the same
    /// entry.
    #[test]
    fn resolves_a_known_slot_and_its_inverse_round_trips() {
        let manifest = FieldManifest::committed();

        let (facet, pair, byte) = manifest
            .resolve("u_component_of_wind", Some(850))
            .expect("u_component_of_wind@850 must resolve");
        assert_eq!((facet, pair), (1, 0));
        assert_eq!(byte, PairByte::Lo);

        let entry = manifest
            .resolve_slot(facet, pair, byte)
            .expect("the inverse lookup must find the entry resolve() just returned");
        assert_eq!(entry.variable, "u_component_of_wind");
        assert_eq!(entry.level_hpa, Some(850));
        assert_eq!(entry.floor_id, "u850");
    }

    /// The other half of resolution being real: a `(variable, level)` pair
    /// outside the W1 set resolves to `None`, not to a wrong slot.
    #[test]
    fn a_variable_level_outside_the_w1_set_does_not_resolve() {
        let manifest = FieldManifest::committed();
        // temperature exists at 850/500 in W1, but not at 250.
        assert_eq!(manifest.resolve("temperature", Some(250)), None);
        // geopotential exists as a pressure-level field, never as a surface one.
        assert_eq!(manifest.resolve("geopotential", None), None);
    }

    /// Out-of-range rejection: `facet = 3` is rejected.
    #[test]
    fn facet_out_of_range_is_rejected() {
        let text = format!("{HEADER}\n3\t0\tlo\tsome_var\tsurface\tPa\tid\n");
        assert!(matches!(
            FieldManifest::parse(&text),
            Err(ManifestError::FacetOutOfRange { .. })
        ));
    }

    /// Out-of-range rejection: `pair = 6` is rejected.
    #[test]
    fn pair_out_of_range_is_rejected() {
        let text = format!("{HEADER}\n0\t6\tlo\tsome_var\tsurface\tPa\tid\n");
        assert!(matches!(
            FieldManifest::parse(&text),
            Err(ManifestError::PairOutOfRange { .. })
        ));
    }

    /// Out-of-range rejection: a bogus byte token (`mid`) is rejected.
    #[test]
    fn bogus_byte_token_is_rejected() {
        let text = format!("{HEADER}\n0\t0\tmid\tsome_var\tsurface\tPa\tid\n");
        assert!(matches!(
            FieldManifest::parse(&text),
            Err(ManifestError::BadByteToken { .. })
        ));
    }

    /// A malformed header is rejected rather than silently accepted with
    /// misaligned columns.
    #[test]
    fn malformed_header_is_rejected() {
        let text = "not\tthe\theader\n0\t0\tlo\tsome_var\tsurface\tPa\tid\n";
        assert!(matches!(
            FieldManifest::parse(text),
            Err(ManifestError::BadHeader { .. })
        ));
    }

    /// A row with the wrong number of columns is rejected (`==`, not a
    /// permissive minimum).
    #[test]
    fn wrong_column_count_is_rejected() {
        let text = format!("{HEADER}\n0\t0\tlo\tsome_var\tsurface\tPa\n"); // 6 columns, not 7
        assert!(matches!(
            FieldManifest::parse(&text),
            Err(ManifestError::WrongColumnCount { line: 2, found: 6 })
        ));
    }

    /// A `level_hpa` that is neither `surface` nor an integer is rejected.
    #[test]
    fn bad_level_token_is_rejected() {
        let text = format!("{HEADER}\n0\t0\tlo\tsome_var\tnot_a_level\tPa\tid\n");
        assert!(matches!(
            FieldManifest::parse(&text),
            Err(ManifestError::BadLevel { .. })
        ));
    }

    /// `#`-prefixed comment lines and blank lines are skipped wherever they
    /// occur, not just before the header.
    #[test]
    fn comments_and_blank_lines_are_skipped_anywhere() {
        let text = format!(
            "# leading comment\n\n{HEADER}\n# a mid-file comment\n0\t0\tlo\tsome_var\tsurface\tPa\tid\n\n# trailing comment\n"
        );
        let manifest = FieldManifest::parse(&text).expect("comments must not break parsing");
        assert_eq!(manifest.entries().len(), 1);
    }
}
