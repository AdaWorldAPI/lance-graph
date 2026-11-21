//! The streaming cell writer (D-INTAKE-2G): cells written into one reused
//! buffer through a sink, with no `String` per cell.
//!
//! [`Reader::rows_at`], [`Reader::rows_in`] and [`read_bag`] return
//! [`PgRows`](super::PgRows), one heap `String` per cell. The functions here
//! apply the same position, presence and live rules, but spell each cell into
//! a single buffer that lives for the call and hand it to a [`CellSink`], which
//! copies what it wants and keeps nothing it was lent.
//!
//! Refusals are the same as the materializing reads. They are not raised
//! before anything is written: a refusal may arrive after earlier rows (and
//! the first cells of the refused row) reached the sink, and `end_row` is not
//! called for a row that refused. A sink that needs all-or-nothing discards
//! its own output on `Err`.

use lance_graph_contract::pg_row::{write_cell, Written};
use lance_graph_mask_risc::{words_for, LaneRef};

use super::{bit, BagSide, ColumnSpec, PgRowRefusal, Reader};

/// Where spelled cells go. A cell's text is valid only during the call.
pub trait CellSink {
    fn begin_row(&mut self) {}
    /// One cell; `None` is SQL NULL.
    fn cell(&mut self, text: Option<&str>);
    fn end_row(&mut self) {}
}

/// Initial capacity of the spelling buffer: covers every non-text cell.
const BUF: usize = 64;

/// Spell one cell into `buf` and give it to `sink`.
fn emit<S: CellSink>(
    spec: &ColumnSpec<'_>,
    present: bool,
    value: u32,
    at: u32,
    buf: &mut String,
    sink: &mut S,
) -> Result<(), PgRowRefusal> {
    buf.clear();
    match write_cell(&spec.col, present, value, spec.codes, buf) {
        Ok(Written::Null) => sink.cell(None),
        Ok(Written::Text) => sink.cell(Some(buf.as_str())),
        Err(refusal) => {
            return Err(PgRowRefusal::Cell {
                name: spec.name.to_owned(),
                at,
                refusal,
            })
        }
    }
    Ok(())
}

impl<'s, 'c> Reader<'s, 'c> {
    /// The cells of row `p` (already checked), in column order.
    fn write_cells<S: CellSink>(
        &self,
        p: usize,
        buf: &mut String,
        sink: &mut S,
    ) -> Result<(), PgRowRefusal> {
        for (spec, (lane, plane)) in self.cols.iter().zip(&self.lanes) {
            let value = match lane {
                LaneRef::U32(v) => v[p],
                LaneRef::I32(v) => v[p] as u32,
                _ => {
                    return Err(PgRowRefusal::LaneKind {
                        name: spec.name.to_owned(),
                    })
                }
            };
            let present = cfg!(feature = "disable-pg-row-presence") || bit(plane, p);
            emit(spec, present, value, p as u32, buf, sink)?;
        }
        Ok(())
    }

    /// −0 on every column: the row of a missing side.
    fn write_absent_cells<S: CellSink>(
        &self,
        buf: &mut String,
        sink: &mut S,
    ) -> Result<(), PgRowRefusal> {
        for spec in self.cols {
            emit(spec, false, 0, u32::MAX, buf, sink)?;
        }
        Ok(())
    }

    /// The rows at `positions` (a bag, in list order, duplicates kept) into
    /// `sink`. Returns the row count.
    ///
    /// Refuses as [`Reader::rows_at`] does, but not before writing: see the
    /// module documentation.
    pub fn write_rows(
        &self,
        positions: &[u32],
        sink: &mut impl CellSink,
    ) -> Result<usize, PgRowRefusal> {
        let mut buf = String::with_capacity(BUF);
        for &at in positions {
            let p = self.check_position(at)?;
            sink.begin_row();
            self.write_cells(p, &mut buf, sink)?;
            sink.end_row();
        }
        Ok(positions.len())
    }

    /// The rows whose bit is set in `mask`, ascending, into `sink`. Returns
    /// the row count.
    ///
    /// Refuses as [`Reader::rows_in`] does, but not before writing: see the
    /// module documentation.
    pub fn write_rows_in(
        &self,
        mask: &[u64],
        sink: &mut impl CellSink,
    ) -> Result<usize, PgRowRefusal> {
        let words = words_for(self.rows);
        if mask.len() < words {
            return Err(PgRowRefusal::Span {
                name: "mask".to_owned(),
            });
        }
        let mut buf = String::with_capacity(BUF);
        let mut count = 0usize;
        for (w, &word) in mask.iter().enumerate().take(words) {
            let mut bits = word;
            while bits != 0 {
                let p = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let p = self.check_position(p as u32)?;
                sink.begin_row();
                self.write_cells(p, &mut buf, sink)?;
                sink.end_row();
                count += 1;
            }
        }
        Ok(count)
    }
}

/// [`read_bag`](super::read_bag) into a sink: bag index `i` reads every side
/// at its `positions[i]`, or −0 on every column of a side whose present bit
/// is 0. Returns the row count.
///
/// Refuses as `read_bag` does, but not before writing: see the module
/// documentation.
pub fn write_bag(
    sides: &[BagSide<'_, '_, '_>],
    sink: &mut impl CellSink,
) -> Result<usize, PgRowRefusal> {
    let Some(first) = sides.first() else {
        return Err(PgRowRefusal::Bag("no sides"));
    };
    let len = first.positions.len();
    for side in sides {
        if side.positions.len() != len {
            return Err(PgRowRefusal::Bag("sides differ in length"));
        }
        if side.present.is_some_and(|p| p.len() < words_for(len)) {
            return Err(PgRowRefusal::Bag("present plane shorter than the bag"));
        }
    }
    let mut buf = String::with_capacity(BUF);
    for i in 0..len {
        sink.begin_row();
        for side in sides {
            let here = cfg!(feature = "disable-pg-row-bag-present")
                || side.present.is_none_or(|p| bit(p, i));
            if here {
                let p = side.reader.check_position(side.positions[i])?;
                side.reader.write_cells(p, &mut buf, sink)?;
            } else {
                side.reader.write_absent_cells(&mut buf, sink)?;
            }
        }
        sink.end_row();
    }
    Ok(len)
}

/// Rows shaped as [`PgRows::rows`](super::PgRows): one `String` per cell. The
/// compatibility sink and the test oracle; it allocates per cell by design.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CollectRows {
    pub rows: Vec<Vec<Option<String>>>,
    current: Vec<Option<String>>,
}

impl CollectRows {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CellSink for CollectRows {
    fn begin_row(&mut self) {
        self.current = Vec::new();
    }

    fn cell(&mut self, text: Option<&str>) {
        self.current.push(text.map(str::to_owned));
    }

    fn end_row(&mut self) {
        self.rows.push(core::mem::take(&mut self.current));
    }
}

/// PostgreSQL's `COPY ... TO STDOUT` text format, appended to `out`.
///
/// The rules, as PostgreSQL's COPY text format defines them (default
/// delimiter and NULL string):
/// - fields in a row are separated by a tab (`\t`), rows end with a newline
///   (`\n`);
/// - NULL is the two characters `\N`;
/// - in a value, a backslash becomes `\\`, a tab `\t`, a newline `\n` and a
///   carriage return `\r` (each as backslash and a letter);
/// - every other character, control characters included, is written as is,
///   which `COPY FROM` reads back unchanged.
///
/// An empty string is an empty field, distinct from NULL. `out` may be cleared
/// between uses; its capacity is kept.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CopyText {
    pub out: Vec<u8>,
    /// Cells written in the current row.
    fields: usize,
}

impl CopyText {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CellSink for CopyText {
    fn begin_row(&mut self) {
        self.fields = 0;
    }

    fn cell(&mut self, text: Option<&str>) {
        if self.fields > 0 {
            self.out.push(b'\t');
        }
        self.fields += 1;
        let Some(text) = text else {
            self.out.extend_from_slice(b"\\N");
            return;
        };
        let bytes = text.as_bytes();
        let mut start = 0;
        for (i, &b) in bytes.iter().enumerate() {
            let escaped = match b {
                b'\\' => b'\\',
                b'\t' => b't',
                b'\n' => b'n',
                b'\r' => b'r',
                0x08 => b'b',
                0x0B => b'v',
                0x0C => b'f',
                _ => continue,
            };
            self.out.extend_from_slice(&bytes[start..i]);
            self.out.push(b'\\');
            self.out.push(escaped);
            start = i + 1;
        }
        self.out.extend_from_slice(&bytes[start..]);
    }

    fn end_row(&mut self) {
        self.out.push(b'\n');
        self.fields = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{CellSink, CopyText};

    /// FAILS IF: a byte PostgreSQL escapes in `COPY ... TO STDOUT` text format
    /// is passed raw, or a byte it passes raw is escaped. The expected bytes
    /// are PostgreSQL 16's output for
    /// `COPY (SELECT 'a'||chr(11)||'b'||chr(8)||chr(12)||chr(1)||chr(127), NULL, '') TO STDOUT`
    /// plus the tab, newline, carriage return and backslash cases.
    #[test]
    fn copy_text_escapes_what_postgresql_escapes() {
        let mut sink = CopyText::new();
        sink.begin_row();
        sink.cell(Some("a\u{b}b\u{8}\u{c}\u{1}\u{7f}"));
        sink.cell(None);
        sink.cell(Some(""));
        sink.cell(Some("t\tn\nr\rs\\"));
        sink.end_row();
        assert_eq!(
            sink.out,
            b"a\\vb\\b\\f\x01\x7f\t\\N\t\tt\\tn\\nr\\rs\\\\\n".to_vec()
        );
    }
}
