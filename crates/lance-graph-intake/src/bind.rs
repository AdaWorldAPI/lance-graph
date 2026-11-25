//! The table as a [`Binder`]: names and textual literals are resolved once,
//! through Quack's binding membrane, into a numeric `Query`. Execution then
//! reads codes, lanes and planes only.
//!
//! ```text
//! t.bind(&table("ticket").where_eq("title", "vpn"))   cold: names, DT code lookup
//!     → BoundQuery                                     numeric, under this table's frame
//! t.select_bound(&q)                                  warm: frame check, fold, keep
//!     → SelectionRef                                   DR-owned positions
//! ```
//!
//! Signed-integer lanes (`FieldKind::I32`), text-code lanes
//! (`FieldKind::Code`) and boolean lanes (`FieldKind::Bool`) bind. Quack's
//! membrane has no unsigned or date literal kind, so a field of those
//! encodings is not offered and binds as `BindError::UnknownField`.
//!
//! # Declared empty-as-absent text columns
//!
//! Detection never reads `''` as absent: an observed `''` is a present value
//! with its own code, kept apart from NULL. A source may DECLARE that its
//! empty text means absent (`ColumnConfig::absent = AbsentAs::EmptyText`);
//! admission then stores `''` and NULL alike as absent (−0, code 0), and
//! egress spells them `''`. Under that declared policy `''` and NULL are one
//! value, so such a column is offered without a validity plane, `''` binds to
//! code 0, and a `LIKE` pattern that matches `''` covers code 0. Those
//! answers follow the declared policy; they are not PostgreSQL parity for a
//! nullable text column, which keeps `''` and NULL apart.

use lance_graph_contract::pg_row::{AbsentAs, LaneEncoding};
use lance_graph_dt64::dr::Frame;
use lance_graph_pg_row::{bound_validity, BOUND_LIVE};
use lance_graph_quack::bind::{
    BindError, Binder, BoundField, CodeWitness, Draft, FieldKind, Pattern, TableId, TypedQuery,
};
use lance_graph_quack::{Col, Mask};

use crate::{positions_of, IntakeRefusal, SelectionRef, Table};

/// A query bound against one table under one frame. It runs only on that
/// table and only while the frame is current: the frame names the table's
/// store and its data version, so a query is refused on another table and
/// after any append or delete, whether or not it compares a text literal.
#[derive(Debug, Clone)]
pub struct BoundQuery {
    typed: TypedQuery<Frame>,
    frame: Frame,
}

impl Table {
    fn column_of(&self, t: TableId, col: Col) -> Option<usize> {
        let k = usize::from(col.0);
        (t == self.table_id() && k < self.shape.len()).then_some(k)
    }

    fn table_id(&self) -> TableId {
        TableId(self.owner.class)
    }

    fn empty_absent(&self, k: usize) -> bool {
        self.shape[k].col.absent == AbsentAs::EmptyText
    }

    /// Resolve `draft`'s names and literals against this table once.
    pub fn bind(&self, draft: &Draft) -> Result<BoundQuery, IntakeRefusal> {
        Ok(BoundQuery {
            typed: draft.bind_typed(self).map_err(IntakeRefusal::Bind)?,
            frame: self.owner.frame.clone(),
        })
    }

    /// Run a bound query once, keep the positions it holds for under the
    /// current data version, and return the handle, as [`Table::select_ref`]
    /// does for a numeric filter.
    ///
    /// A query bound by another table, or before an append or a delete, is
    /// refused as [`IntakeRefusal::Stale`] and must be bound again.
    pub fn select_bound(&mut self, q: &BoundQuery) -> Result<SelectionRef, IntakeRefusal> {
        self.owner
            .table
            .frames()
            .check(&q.frame)
            .map_err(|_| IntakeRefusal::Stale)?;
        let query = q.typed.admit(self).map_err(IntakeRefusal::Bind)?.clone();
        let positions = {
            let texts = self.texts();
            let specs = self.specs(&texts);
            let reader = self.reader(&specs)?;
            self.predicate_runs.set(self.predicate_runs.get() + 1);
            let kept = self.with_validity(&reader, |v| reader.keep_bound(&query, v))?;
            positions_of(&kept)
        };
        self.keep_positions(positions)
    }
}

impl Binder for Table {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == self.name).then(|| self.table_id())
    }

    fn live(&self, _table: TableId) -> Mask {
        BOUND_LIVE
    }

    fn field(&self, table: TableId, name: &str) -> Option<BoundField> {
        if table != self.table_id() {
            return None;
        }
        let k = self.shape.iter().position(|c| c.name == name)?;
        let kind = match self.shape[k].col.encoding {
            LaneEncoding::Signed => FieldKind::I32,
            LaneEncoding::TextCode => FieldKind::Code,
            LaneEncoding::Bool => FieldKind::Bool,
            _ => return None,
        };
        let col = u16::try_from(k).ok()?;
        Some(BoundField {
            col: Col(col),
            kind,
            validity: (!self.empty_absent(k)).then(|| bound_validity(col)),
        })
    }

    fn code(&self, table: TableId, field: Col, literal: &str) -> Option<u32> {
        let k = self.column_of(table, field)?;
        self.literal_code(&self.owner.frame, k, literal)
    }

    fn codes_matching(&self, table: TableId, field: Col, pattern: &Pattern) -> Option<Vec<u32>> {
        let k = self.column_of(table, field)?;
        let space = self.spaces.get(k)?.as_ref()?;
        let empty = (self.empty_absent(k) && pattern.matches("")).then_some(0);
        Some(
            empty
                .into_iter()
                .chain((1..=self.code_counts[k]).filter(|&code| {
                    self.codes
                        .key(&self.owner.frame, space, code)
                        .is_ok_and(|key| pattern.matches(&key.0))
                }))
                .collect(),
        )
    }
}

/// The witness of a code space is the frame it was read under. Codes are
/// stable across appends, but the frame moves with every data version, so
/// this is conservative, never wrong. [`BoundQuery`] checks the same frame
/// for every query, coded or not.
impl CodeWitness for Table {
    type Witness = Frame;

    fn code_witness(&self, table: TableId, col: Col) -> Option<Frame> {
        let k = self.column_of(table, col)?;
        self.spaces
            .get(k)?
            .as_ref()
            .map(|_| self.owner.frame.clone())
    }

    fn admit_code(&self, table: TableId, col: Col, held: &Frame) -> Result<(), BindError> {
        self.column_of(table, col)
            .filter(|_| self.owner.table.frames().check(held).is_ok())
            .map(|_| ())
            .ok_or(BindError::StaleCodeSpace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShapeConfig;
    use lance_graph_quack::bind::table;

    fn small(name: &str) -> Table {
        small_with(name, &ShapeConfig::default())
    }

    fn small_with(name: &str, config: &ShapeConfig) -> Table {
        let rows: Vec<Vec<Option<&str>>> = vec![
            vec![Some("1"), Some("a"), Some("-1"), Some("t")],
            vec![Some("2"), Some(""), Some("2"), Some("f")],
            vec![Some("3"), Some("b"), None, Some("t")],
        ];
        Table::from_rows(name, 0x2B09_0002, &["id", "note", "n", "ok"], &rows, config)
            .expect("table")
    }

    /// FAILS IF: an observed `''` makes a column empty-absent without a
    /// declaration, or `''` then has no code of its own.
    #[test]
    fn an_undeclared_empty_string_is_a_present_value() {
        let t = small("t");
        let id = t.table("t").expect("table");
        assert_eq!(t.shape[1].col.absent, AbsentAs::Null);
        let note = t.field(id, "note").expect("note");
        assert_eq!(note.validity, Some(bound_validity(1)));
        assert!(t.code(id, Col(1), "").is_some_and(|c| c > 0));
    }

    /// FAILS IF: a field kind the membrane cannot compare is offered, a
    /// validity plane index does not follow `bound_validity`, or a declared
    /// empty-absent text column is gated by validity.
    #[test]
    fn fields_follow_the_plane_layout() {
        let mut config = ShapeConfig::default();
        config.columns.insert(
            "note".to_owned(),
            crate::ColumnConfig {
                absent: Some(AbsentAs::EmptyText),
                ..Default::default()
            },
        );
        let t = small_with("t", &config);
        let id = t.table("t").expect("table");
        let note = t.field(id, "note").expect("note");
        assert_eq!(t.shape[1].col.absent, AbsentAs::EmptyText);
        assert_eq!(
            (note.col, note.kind, note.validity),
            (Col(1), FieldKind::Code, None)
        );
        let n = t.field(id, "n").expect("n");
        assert_eq!(
            (n.kind, n.validity),
            (FieldKind::I32, Some(bound_validity(2)))
        );
        let ok = t.field(id, "ok").expect("ok");
        assert_eq!(
            (ok.kind, ok.validity),
            (FieldKind::Bool, Some(bound_validity(3)))
        );
        assert_eq!(t.code(id, Col(1), ""), Some(0));
        assert!(t.code(id, Col(1), "zzz").is_none());
    }

    /// FAILS IF: a numeric-only query runs on a table it was not bound by,
    /// or after the data version moves.
    #[test]
    fn a_numeric_query_is_tied_to_its_table_and_version() {
        let mut t = small("t");
        let mut other = small("t");
        let q = t.bind(&table("t").where_eq("n", 2i32)).expect("bind");
        assert!(matches!(other.select_bound(&q), Err(IntakeRefusal::Stale)));
        assert_eq!(t.select_bound(&q).expect("current").len(), 1);
        t.delete_key(1).expect("delete");
        assert!(matches!(t.select_bound(&q), Err(IntakeRefusal::Stale)));
    }
}
