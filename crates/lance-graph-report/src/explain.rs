//! The human-readable physical plan — diagnostics, and the surface that
//! catches accidental hidden materialization before it runs.
//!
//! ```text
//! Source: S1@g3 (1000000 rows)
//! Selection: tile-local mask ops (no population mask)
//! Coordinates:
//!   F2  ordinal lane · domain 32 · fold key
//!   F7  derived bucket · domain 12 · partition
//! Folds: COUNT, SUM(F4)
//! Physical: 12 pass(es) · dense accumulator 384 cells
//! Materialization: terminal only
//! ```

use core::fmt;

use crate::exec::{Accumulator, PhysicalPlan, Provider, SelectionCarrier};
use crate::plan::{CoordSpec, FoldState};

impl fmt::Display for PhysicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Source: {}@g{} ({} rows)",
            self.source.id, self.source.generation, self.n_rows
        )?;
        let sel = match self.selection {
            SelectionCarrier::ValidityPlane => "validity plane (zero ops)",
            SelectionCarrier::TileLocal => "tile-local mask ops (no population mask)",
            SelectionCarrier::ReusedMask => {
                "ONE reused population mask (materialized once, counted)"
            }
        };
        write!(f, "Selection: {sel}")?;
        if let Some(e) = self.extent {
            write!(
                f,
                " · execution extent [{}, {}) (no mask words)",
                e.lo, e.hi
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Coordinates:")?;
        for (i, d) in self.dims.iter().enumerate() {
            let prov = match (&d.provider, &d.coord) {
                (Provider::OrdinalLane { .. }, _) => "ordinal lane".to_string(),
                (Provider::DerivedBucket { .. }, CoordSpec::Bucket { origin, width, .. }) => {
                    format!("derived bucket (origin {origin}, width {width}; not materialized)")
                }
                (Provider::DerivedBucket { .. }, _) => "derived".to_string(),
                (Provider::MaskPlanes { planes }, _) => {
                    format!(
                        "mask set ({} resident masks; a row may sit in several)",
                        planes.len()
                    )
                }
            };
            let role = if self.fold_key == Some(i) {
                "fold key"
            } else {
                "partition"
            };
            writeln!(
                f,
                "  {}  {prov} · domain {} · {role}",
                coord_name(&d.coord),
                d.domain
            )?;
        }
        let folds: Vec<String> = self
            .states
            .iter()
            .map(|s| match s {
                FoldState::Count => "COUNT".to_string(),
                FoldState::Sum(x) => format!("SUM({x})"),
                FoldState::Min(x) => format!("MIN({x})"),
                FoldState::Max(x) => format!("MAX({x})"),
            })
            .collect();
        writeln!(f, "Folds: {}", folds.join(", "))?;
        let acc = match self.accumulator {
            Accumulator::Scalar => "scalar".to_string(),
            Accumulator::Dense { cells } => format!("dense accumulator {cells} cells"),
            Accumulator::Sparse { product } => {
                format!(
                    "sparse accumulator (observed cells only; product {product} never allocated)"
                )
            }
        };
        writeln!(f, "Physical: {} pass(es) · {acc}", self.passes)?;
        write!(f, "Materialization: terminal only")
    }
}

fn coord_name(c: &CoordSpec) -> String {
    match c {
        CoordSpec::MaskSet { base, count } => {
            format!("{base}..M{}", u64::from(base.0) + u64::from(*count))
        }
        _ => c.field().map_or_else(String::new, |f| f.to_string()),
    }
}
