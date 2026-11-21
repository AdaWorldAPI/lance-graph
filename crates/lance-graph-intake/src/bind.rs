//! The table as a [`Binder`]: names and textual literals are resolved once,
//! through Quack's binding membrane, into a numeric `Query`. Execution then
//! reads codes, lanes and planes only.
//!
//! ```text
//! table("ticket").where_eq("title", "vpn").bind_typed(&t)   cold: names, DT code lookup
//!     → TypedQuery<Frame>                                     numeric, witnessed
//! t.select_bound(&q)                                         warm: frame check, fold, keep
//!     → SelectionRef                                          DR-owned positions
//! ```
//!
//! Only signed-integer lanes (`FieldKind::I32`) and text-code lanes
//! (`FieldKind::Code`) bind: Quack's membrane has no unsigned, boolean or
//! date literal kind, so a field of those encodings is not offered and binds
//! as [`BindError::UnknownField`](lance_graph_quack::bind::BindError).
//!
//! A column whose absent spelling is `''` (`AbsentAs::EmptyText`) admitted
//! every `''` as absent, so `''` has no code there and binding `= ''` on it is
//! refused as `UnknownValue`. It is never answered as an empty set:
//! PostgreSQL would keep those rows, and this membrane has no absent literal.

use lance_graph_contract::pg_row::LaneEncoding;
use lance_graph_dt64::dr::{Frame, ValueKey};
use lance_graph_pg_row::{bound_validity, BOUND_LIVE};
use lance_graph_quack::bind::{
    BindError, Binder, BoundField, CodeWitness, FieldKind, Pattern, TableId, TypedQuery,
};
use lance_graph_quack::{Col, Mask};

use crate::{positions_of, IntakeRefusal, SelectionRef, Table};

impl Table {
    fn column_of(&self, t: TableId, col: Col) -> Option<usize> {
        let k = usize::from(col.0);
        (t == self.table_id() && k < self.shape.len()).then_some(k)
    }

    fn table_id(&self) -> TableId {
        TableId(self.owner.class)
    }

    /// Run a query bound against this table once, keep the positions it
    /// holds for under the current data version, and return the handle, as
    /// [`Table::select_ref`] does for a numeric filter.
    ///
    /// The query is admitted first: a code space it was bound under must
    /// still be witnessed by the current frame, so a query bound before an
    /// append or a delete is refused as `Bind(StaleCodeSpace)` and must be
    /// bound again.
    pub fn select_bound(&mut self, q: &TypedQuery<Frame>) -> Result<SelectionRef, IntakeRefusal> {
        let query = q
            .admit(self)
            .map_err(|e| IntakeRefusal::Bind(format!("{e:?}")))?
            .clone();
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
            _ => return None,
        };
        let col = u16::try_from(k).ok()?;
        Some(BoundField {
            col: Col(col),
            kind,
            validity: Some(bound_validity(col)),
        })
    }

    fn code(&self, table: TableId, field: Col, literal: &str) -> Option<u32> {
        let k = self.column_of(table, field)?;
        let space = self.spaces.get(k)?.as_ref()?;
        self.codes
            .code(&self.owner.frame, space, &ValueKey(literal.to_owned()))
            .ok()
    }

    fn codes_matching(&self, table: TableId, field: Col, pattern: &Pattern) -> Option<Vec<u32>> {
        let k = self.column_of(table, field)?;
        let space = self.spaces.get(k)?.as_ref()?;
        Some(
            (1..=self.code_counts[k])
                .filter(|&code| {
                    self.codes
                        .key(&self.owner.frame, space, code)
                        .is_ok_and(|key| pattern.matches(&key.0))
                })
                .collect(),
        )
    }
}

/// The witness of a code space is the frame it was read under: codes are
/// stable across appends, but the frame moves with every data version, so a
/// bound query is refused after any append or delete. That is conservative,
/// never wrong.
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
