//! The escape hatch: hand config over detected shapes.
//!
//! A value given in [`ColumnConfig`] replaces the detected one; a column with
//! no entry keeps what was detected. Every column gets a [`Verdict`] saying
//! whether the config confirmed the detection, differs from it, or was not
//! given, so a hand value is never silently trusted nor silently ignored.
//! The merged column must still be admitted by `PgColumn::admit`.
//!
//! A column whose config gives all three values is declared: its type is
//! known before any cell is read, so it is not detected at all. Its cells
//! are still checked one by one when they are placed.

use std::collections::BTreeMap;

use lance_graph_contract::pg_row::{AbsentAs, LaneEncoding, PgColumn, PgType, PgTypeRefusal};

use crate::detect::{detect_column, detect_key, Detected};
use crate::IntakeRefusal;

/// Hand values for one column; each `Some` wins over detection.
#[derive(Debug, Default, Clone)]
pub struct ColumnConfig {
    pub ty: Option<PgType>,
    pub encoding: Option<LaneEncoding>,
    pub absent: Option<AbsentAs>,
}

/// Hand values for a table: per column by name, and the key column's name.
#[derive(Debug, Default, Clone)]
pub struct ShapeConfig {
    pub columns: BTreeMap<String, ColumnConfig>,
    pub key: Option<String>,
}

/// How a column's merged shape relates to its detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The config was given and equals the detection.
    Agrees,
    /// The config was given and the merged column is not the detected one.
    Differs { detected: PgColumn },
    /// No config was given for the column.
    DetectedOnly,
    /// The config gave every value; the column was not detected.
    Declared,
}

/// One column as the table will read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnShape {
    pub name: String,
    pub col: PgColumn,
    pub verdict: Verdict,
    /// `None` for a declared column, which is never detected.
    pub detected: Option<Detected>,
}

impl ColumnConfig {
    /// The column this config declares, when it gives every value.
    pub fn declared(&self) -> Option<PgColumn> {
        Some(PgColumn {
            ty: self.ty?,
            encoding: self.encoding?,
            absent: self.absent?,
        })
    }
}

/// The lane encoding of a declared type: the detected one when the contract
/// admits it for that type (`PgColumn::admit`), otherwise the one encoding the
/// type has. A declared type therefore never pairs with the encoding of a
/// shape the data merely suggested.
fn encoding_for(ty: PgType, detected: LaneEncoding) -> LaneEncoding {
    let admits = |encoding| {
        PgColumn {
            ty,
            encoding,
            absent: AbsentAs::Null,
        }
        .admit()
        .is_ok()
    };
    if admits(detected) {
        return detected;
    }
    match ty {
        PgType::Bool => LaneEncoding::Bool,
        PgType::Text | PgType::Varchar(_) => LaneEncoding::TextCode,
        PgType::Timestamp(_) => LaneEncoding::EpochSeconds,
        PgType::Date => LaneEncoding::EpochDays,
        _ => LaneEncoding::Signed,
    }
}

/// Merge hand config over a detected column.
pub fn merge(
    detected: &Detected,
    config: Option<&ColumnConfig>,
) -> Result<(PgColumn, Verdict), PgTypeRefusal> {
    let mut col = detected.col;
    let Some(config) = config else {
        col.admit()?;
        return Ok((col, Verdict::DetectedOnly));
    };
    if let Some(ty) = config.ty {
        col.ty = ty;
        col.encoding = encoding_for(ty, detected.col.encoding);
    }
    if let Some(encoding) = config.encoding {
        col.encoding = encoding;
    }
    if let Some(absent) = config.absent {
        col.absent = absent;
    }
    col.admit()?;
    let verdict = if col == detected.col {
        Verdict::Agrees
    } else {
        Verdict::Differs {
            detected: detected.col,
        }
    };
    Ok((col, verdict))
}

/// The shape of every column of a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub columns: Vec<ColumnShape>,
}

impl Shape {
    /// Detect every column of `rows` (row-major, one cell per name), merge
    /// `config` over it and admit the result.
    pub fn build(
        names: &[&str],
        rows: &[Vec<Option<&str>>],
        config: &ShapeConfig,
    ) -> Result<Shape, IntakeRefusal> {
        let named = |name: &str| names.contains(&name);
        for name in config.columns.keys().chain(config.key.iter()) {
            if !named(name) {
                return Err(IntakeRefusal::UnknownColumn(name.clone()));
            }
        }
        if let Some((row, cells)) = rows
            .iter()
            .enumerate()
            .map(|(row, cells)| (row, cells.len()))
            .find(|&(_, cells)| cells != names.len())
        {
            return Err(IntakeRefusal::Ragged { row, cells });
        }

        let mut columns = Vec::with_capacity(names.len());
        for (k, name) in names.iter().enumerate() {
            let refused = |refusal| IntakeRefusal::Column {
                name: (*name).to_owned(),
                refusal,
            };
            let given = config.columns.get(*name);
            if let Some(col) = given.and_then(ColumnConfig::declared) {
                col.admit().map_err(refused)?;
                columns.push(ColumnShape {
                    name: (*name).to_owned(),
                    col,
                    verdict: Verdict::Declared,
                    detected: None,
                });
                continue;
            }
            let cells: Vec<Option<&str>> = rows.iter().map(|row| row[k]).collect();
            let detected = detect_column(&cells);
            let (col, verdict) = merge(&detected, given).map_err(refused)?;
            columns.push(ColumnShape {
                name: (*name).to_owned(),
                col,
                verdict,
                detected: Some(detected),
            });
        }

        let count =
            |pick: fn(&Verdict) -> bool| columns.iter().filter(|c| pick(&c.verdict)).count();
        let agrees = count(|v| matches!(v, Verdict::Agrees));
        let differs = count(|v| matches!(v, Verdict::Differs { .. }));
        let detected_only = count(|v| matches!(v, Verdict::DetectedOnly));
        let declared = count(|v| matches!(v, Verdict::Declared));
        assert_eq!(
            columns.len(),
            agrees + differs + detected_only + declared,
            "every column has exactly one verdict"
        );
        Ok(Shape { columns })
    }

    /// The key column's position: the configured one, else the detected one.
    ///
    /// The detection runs over the merged columns, so a column the config
    /// turned into text is never chosen.
    pub fn key_column(
        &self,
        names: &[&str],
        rows: &[Vec<Option<&str>>],
        config: &ShapeConfig,
    ) -> Result<usize, IntakeRefusal> {
        let merged: Vec<Detected> = self
            .columns
            .iter()
            .map(|c| Detected {
                col: c.col,
                support: 0,
                total: 0,
                witness: None,
            })
            .collect();
        let key = match &config.key {
            Some(name) => self.columns.iter().position(|c| &c.name == name),
            None => detect_key(names, &merged, rows),
        };
        let k = key.ok_or(IntakeRefusal::NoKey)?;
        match self.columns[k].col.encoding {
            LaneEncoding::Signed | LaneEncoding::Unsigned => Ok(k),
            _ => Err(IntakeRefusal::NoKey),
        }
    }
}
