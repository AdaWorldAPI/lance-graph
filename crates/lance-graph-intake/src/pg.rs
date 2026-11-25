//! The PostgreSQL egress adapter: a kept selection streamed out as wire
//! cells.
//!
//! Egress is not execution. A selection is computed in the native vocabulary
//! (a bound query or a numeric filter → a mask → DR-kept positions, see
//! [`Table::select_bound`] and [`Table::select_ref`]); this module only
//! spells those positions as PostgreSQL text cells, through the streaming
//! [`CellSink`] writer of `lance-graph-pg-row`. Nothing here filters, joins
//! or resolves.

pub use lance_graph_pg_row::sink::{CellSink, CollectRows, CopyText};
use lance_graph_pg_row::Reader;

use crate::{IntakeRefusal, Lend, SelectionRef, Table, KEPT_SEL};

/// Stream the rows of a kept selection into `sink`, in its order and with
/// its repeats, with one reused cell buffer and no row materialized.
/// Returns the rows written.
///
/// The cells equal [`Table::rows_of`]'s. The selection must have been kept
/// by `table` under its current data version: a stale selection, or one
/// kept by another table, is refused as [`IntakeRefusal::Stale`] before
/// anything is written. The kept positions are lent for this call only; they
/// are released on every exit, a panicking sink included.
pub fn write_selection(
    table: &Table,
    r: &SelectionRef,
    sink: &mut impl CellSink,
) -> Result<usize, IntakeRefusal> {
    table.kept_of(r)?;
    let _lend = Lend::new(&table.owner.lend, r.id);
    let texts = table.texts();
    let specs = table.specs(&texts);
    let (reader, positions) = Reader::open_selected(
        &table.binding,
        &table.profile.store,
        table.owner.table.frames(),
        &table.owner,
        KEPT_SEL,
        &specs,
    )?;
    Ok(reader.write_rows(positions, sink)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShapeConfig;

    fn table() -> Table {
        let rows: Vec<Vec<Option<&str>>> = (1..=6)
            .map(|i| match i % 3 {
                0 => vec![Some(["1", "2", "3", "4", "5", "6"][i - 1]), None],
                1 => vec![Some(["1", "2", "3", "4", "5", "6"][i - 1]), Some("a")],
                _ => vec![Some(["1", "2", "3", "4", "5", "6"][i - 1]), Some("")],
            })
            .collect();
        Table::from_rows(
            "t",
            0x2B09_0001,
            &["id", "note"],
            &rows,
            &ShapeConfig::default(),
        )
        .expect("table")
    }

    struct Panics(usize);
    impl CellSink for Panics {
        fn begin_row(&mut self) {}
        fn cell(&mut self, _text: Option<&str>) {
            self.0 += 1;
            assert!(self.0 < 3, "sink fails after the first row");
        }
        fn end_row(&mut self) {}
    }

    /// FAILS IF: a write that the reader refuses after the lend is taken
    /// (here a kept position whose row was deleted) leaves the selection lent,
    /// or a later write through the same table fails or differs.
    #[test]
    fn a_refused_write_leaves_nothing_lent() {
        let mut t = table();
        t.delete_key(2).expect("delete");
        let dead = t.positions_for_keys(&[1]).expect("live")[0] + 1;
        let bad = t
            .keep_positions(vec![0, dead])
            .expect("keep under current frame");
        let mut sink = CollectRows::new();
        let refused = write_selection(&t, &bad, &mut sink);
        assert_eq!(
            refused,
            Err(IntakeRefusal::PgRow(
                lance_graph_pg_row::PgRowRefusal::DeadPosition { at: dead }
            ))
        );
        assert_eq!(
            sink.rows,
            vec![vec![Some("1".to_owned()), Some("a".to_owned())]]
        );
        assert_eq!(t.owner.lend.get(), None);

        let good = t.select_keys_ref(&[1, 3]).expect("keep");
        let mut sink = CollectRows::new();
        assert_eq!(write_selection(&t, &good, &mut sink).expect("write"), 2);
        assert_eq!(sink.rows, t.rows_of(&good).expect("rows").rows);
        assert_eq!(t.owner.lend.get(), None);
    }

    /// FAILS IF: a sink that panics mid-write leaves the kept selection lent,
    /// or a later write through the same table fails or differs.
    #[test]
    fn a_failing_sink_leaves_nothing_lent() {
        let mut t = table();
        let kept = t.select_keys_ref(&[1, 2, 3, 2]).expect("keep");
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            write_selection(&t, &kept, &mut Panics(0))
        }));
        assert!(unwound.is_err(), "the sink panicked");
        assert_eq!(t.owner.lend.get(), None);

        let mut sink = CollectRows::new();
        assert_eq!(write_selection(&t, &kept, &mut sink).expect("write"), 4);
        assert_eq!(sink.rows, t.rows_of(&kept).expect("rows").rows);
        assert_eq!(t.owner.lend.get(), None);
    }
}
