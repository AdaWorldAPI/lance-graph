//! `Gather(table, ids)` on axis 0 of an external-data f32 initializer,
//! resolved to byte ranges from the source placement.
//!
//! ONNX `Gather` with numeric indices is a row copy, exact by contract. Here it
//! never loads the table: each index becomes one `(offset, length)` range into
//! the initializer's external file, computed from the source `dims` and dtype.
//! Indices follow the ONNX rule: `-dim <= i < dim`, negative counting from the
//! end; anything else is refused before a byte is read.

use crate::registry::Refusal;
use crate::source::{Placement, SourceInventory};

/// One gathered row: the source index, the normalized row, and its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRange {
    pub index: i64,
    pub row: u64,
    pub offset: u64,
    pub length: u64,
}

pub fn row_ranges(
    src: &SourceInventory,
    table: &str,
    ids: &[i64],
) -> Result<Vec<RowRange>, Refusal> {
    let refuse = |what: String| Refusal::Runtime {
        node: table.into(),
        what,
    };
    let init = src
        .initializer(table)
        .ok_or_else(|| refuse("not an initializer".into()))?;
    if init.data_type != 1 || init.dims.len() != 2 {
        return Err(refuse(format!(
            "dtype {} rank {}",
            init.data_type,
            init.dims.len()
        )));
    }
    let (rows, cols) = (init.dims[0], init.dims[1]);
    let Placement::External { offset, length, .. } = &init.placement else {
        return Err(refuse("inline, expected external data".into()));
    };
    let row_bytes = cols as u64 * 4;
    if *length != rows as u64 * row_bytes {
        return Err(refuse(format!("length {length} != rows * cols * 4")));
    }
    ids.iter()
        .map(|&i| {
            let row = if (0..rows).contains(&i) {
                i
            } else if (-rows..0).contains(&i) {
                i + rows
            } else {
                return Err(refuse(format!("Gather index {i} outside [-dim, dim)")));
            };
            Ok(RowRange {
                index: i,
                row: row as u64,
                offset: offset + row as u64 * row_bytes,
                length: row_bytes,
            })
        })
        .collect()
}
