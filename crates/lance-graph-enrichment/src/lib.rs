//! Vector's enrichment tables answered by lance-graph, in the same process.
//!
//! VRL reads an enrichment table through `get_enrichment_table_record` and
//! `find_enrichment_table_records`, which call Vector's `enrichment::Table`.
//! [`QuackTable`] implements that trait over an intake [`Table`]:
//!
//! - the conditions are bound once per lookup through Quack's binder
//!   (`draft(..).where_eq(field, value)`, names and literals resolved
//!   against the table's frame), and the positions they hold are read with
//!   [`Table::positions_bound`];
//! - the rows are read from the lanes at those positions by projection
//!   recipes (`lance-graph-vrl`, compiled once per table, at most
//!   `MAX_READS` columns each), and each value is handed to Vector as its
//!   own `vrl::value::Value`, never through text;
//! - an equality on an unsigned column, which Quack's binder does not offer,
//!   is answered from that column's lane at the positions the other
//!   conditions hold.
//!
//! Equality only. A date range, a case-insensitive lookup, a wildcard, a
//! null or a float condition is refused with the enrichment error that names
//! it; nothing is approximated.

use std::sync::{Arc, Mutex};

use enrichment::{Case, Condition, Error, IndexHandle};
use lance_graph_contract::pg_row::LaneEncoding;
use lance_graph_intake::{IntakeRefusal, Table};
use lance_graph_quack::bind::{table as draft, BindError, Draft};
use lance_graph_vrl::{Datum, Recipe, Witness, MAX_READS};
use vrl::value::{ObjectMap, Value};

/// One exposed column: its label (the output name) and its lane encoding.
#[derive(Clone)]
struct Column {
    label: String,
    encoding: LaneEncoding,
}

struct Inner {
    table: Table,
    /// The projection, in groups of at most [`MAX_READS`] columns: one recipe
    /// reads at most that many lanes.
    recipes: Vec<Recipe>,
}

/// An enrichment table backed by an intake table.
#[derive(Clone)]
pub struct QuackTable {
    inner: Arc<Mutex<Inner>>,
    columns: Vec<Column>,
    indexes: Vec<(Case, Vec<String>)>,
}

fn encode(details: impl Into<String>) -> Error {
    Error::FailedToEncodeValue {
        details: details.into(),
    }
}

fn identifier(label: &str) -> bool {
    !label.is_empty()
        && label != "properties"
        && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl QuackTable {
    /// Serve `table` to VRL. Every column whose label is a VRL identifier is
    /// exposed under that name; the others (the route's `lg.*` lanes) are
    /// not.
    pub fn new(table: Table) -> Result<QuackTable, Error> {
        let columns: Vec<Column> = table
            .shape()
            .iter()
            .filter(|c| identifier(&c.name))
            .map(|c| Column {
                label: c.name.clone(),
                encoding: c.col.encoding,
            })
            .collect();
        let recipes = {
            let w = table.witness();
            let class = lance_graph_dt64::view_class(w.binding.class());
            columns
                .chunks(MAX_READS)
                .map(|group| {
                    let program: String = group
                        .iter()
                        .map(|c| match c.label.as_str() {
                            "key" => ".key = .key\n".to_owned(),
                            l => format!(".{l} = .properties.{l}.value\n"),
                        })
                        .collect();
                    Recipe::compile(&program, w.view, class)
                        .map_err(|r| encode(format!("projection: {r:?}")))
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(QuackTable {
            inner: Arc::new(Mutex::new(Inner { table, recipes })),
            columns,
            indexes: Vec::new(),
        })
    }

    fn column(&self, field: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.label == field)
    }

    /// The draft for `conditions`, refusing every condition but equality,
    /// and the equalities on unsigned columns, which Quack's binder does not
    /// offer: they are answered from the lanes (`None` for a value no `u32`
    /// lane can hold, which no row matches).
    #[allow(clippy::type_complexity)]
    fn draft(
        &self,
        name: &str,
        case: Case,
        conditions: &[Condition<'_>],
    ) -> Result<(Draft, Vec<(String, Option<u32>)>), Error> {
        if case == Case::Insensitive {
            return Err(encode("a case-insensitive lookup is not supported"));
        }
        let mut d = draft(name);
        let mut missing = Vec::new();
        let mut unsigned = Vec::new();
        for c in conditions {
            let Condition::Equals { field, value } = c else {
                return Err(Error::OnlyEqualityConditionAllowed);
            };
            let Some(col) = self.column(field) else {
                missing.push((*field).to_owned());
                continue;
            };
            if col.encoding == LaneEncoding::Unsigned {
                let Value::Integer(i) = value else {
                    return Err(encode(format!("condition on {field}: {value:?}")));
                };
                unsigned.push(((*field).to_owned(), u32::try_from(*i).ok()));
                continue;
            }
            d = match value {
                Value::Integer(i) => d.where_eq(field, *i),
                Value::Boolean(b) => d.where_eq(field, *b),
                Value::Bytes(b) => {
                    let t = std::str::from_utf8(b)
                        .map_err(|source| Error::InvalidUtfInColumn { source })?;
                    d.where_eq(field, t)
                }
                Value::String(s) => d.where_eq(field, s.as_ref()),
                other => return Err(encode(format!("condition on {field}: {other:?}"))),
            };
        }
        if missing.is_empty() {
            Ok((d, unsigned))
        } else {
            Err(Error::MissingDatasetFields { fields: missing })
        }
    }

    fn value(&self, t: &Table, col: &Column, v: Datum) -> Value {
        match (col.encoding, v) {
            (LaneEncoding::TextCode, Datum::U32(code)) => t
                .code_key(&col.label, code)
                .map_or(Value::Null, Value::from),
            (LaneEncoding::Bool, Datum::U32(b)) => Value::Boolean(b == 1),
            (LaneEncoding::Bool, Datum::I32(b)) => Value::Boolean(b == 1),
            (LaneEncoding::EpochSeconds, d) => epoch(raw(d)),
            (LaneEncoding::EpochDays, d) => epoch(raw(d) * 86_400),
            (LaneEncoding::Unsigned, d) => Value::Integer(raw(d)),
            (_, Datum::I32(v)) => Value::Integer(i64::from(v)),
            (_, Datum::U32(v)) => Value::Integer(i64::from(v)),
        }
    }

    fn rows(
        &self,
        case: Case,
        conditions: &[Condition<'_>],
        select: Option<&[String]>,
        wildcard: Option<&Value>,
    ) -> Result<Vec<ObjectMap>, Error> {
        if wildcard.is_some() {
            return Err(encode("a wildcard is not supported"));
        }
        if let Some(sel) = select {
            let missing: Vec<String> = sel
                .iter()
                .filter(|f| self.column(f).is_none())
                .cloned()
                .collect();
            if !missing.is_empty() {
                return Err(Error::MissingDatasetFields { fields: missing });
            }
        }
        let inner = self
            .inner
            .lock()
            .map_err(|_| encode("table lock poisoned"))?;
        let t = &inner.table;
        let (d, unsigned) = self.draft(t.name(), case, conditions)?;
        if unsigned.iter().any(|(_, v)| v.is_none()) {
            return Ok(Vec::new());
        }
        let bound = match t.bind(&d) {
            Ok(b) => b,
            // A text value the column's codebook does not hold: no row can
            // match, so the answer is empty, not an error.
            Err(IntakeRefusal::Bind(BindError::UnknownValue { .. })) => return Ok(Vec::new()),
            Err(r) => return Err(encode(format!("bind: {r:?}"))),
        };
        let positions = t
            .positions_bound(&bound)
            .map_err(|r| encode(format!("select: {r:?}")))?;
        let w = t.witness();
        let witness = Witness {
            binding: w.binding,
            store: w.store,
            view: w.view,
            frames: w.frames,
            source: w.source,
        };
        let frame = t.frame();
        let mut rows: Vec<ObjectMap> = vec![ObjectMap::new(); positions.len()];
        for recipe in &inner.recipes {
            let read = recipe
                .rows_at(&witness, &positions, &frame)
                .map_err(|r| encode(format!("read: {r:?}")))?;
            for (c, n) in read.names().iter().enumerate() {
                let col = self.column(n).expect("an exposed column");
                for (i, row) in rows.iter_mut().enumerate() {
                    let v = if read.present(i, c) {
                        self.value(t, col, read.get(i, c))
                    } else {
                        Value::Null
                    };
                    row.insert(n.as_str().into(), v);
                }
            }
        }
        rows.retain(|row| {
            unsigned.iter().all(|(field, v)| {
                row.get(field.as_str()) == v.map(|v| Value::Integer(i64::from(v))).as_ref()
            })
        });
        if let Some(sel) = select {
            for row in &mut rows {
                row.retain(|k, _| sel.iter().any(|s| s == k.as_str()));
            }
        }
        Ok(rows)
    }
}

fn raw(d: Datum) -> i64 {
    match d {
        Datum::I32(v) => i64::from(v),
        Datum::U32(v) => i64::from(v),
    }
}

fn epoch(seconds: i64) -> Value {
    chrono::DateTime::from_timestamp(seconds, 0).map_or(Value::Null, Value::Timestamp)
}

impl enrichment::Table for QuackTable {
    fn find_table_row<'a>(
        &self,
        case: Case,
        condition: &'a [Condition<'a>],
        select: Option<&[String]>,
        wildcard: Option<&Value>,
        _index: Option<IndexHandle>,
    ) -> Result<ObjectMap, Error> {
        let mut rows = self.rows(case, condition, select, wildcard)?;
        match rows.len() {
            0 => Err(Error::NoRowsFound),
            1 => Ok(rows.remove(0)),
            _ => Err(Error::MoreThanOneRowFound),
        }
    }

    fn find_table_rows<'a>(
        &self,
        case: Case,
        condition: &'a [Condition<'a>],
        select: Option<&[String]>,
        wildcard: Option<&Value>,
        _index: Option<IndexHandle>,
    ) -> Result<Vec<ObjectMap>, Error> {
        self.rows(case, condition, select, wildcard)
    }

    /// Lookups are bound through Quack every time; an index is a declared
    /// field list, checked against the table, not a structure built here.
    fn add_index(&mut self, case: Case, fields: &[&str]) -> Result<IndexHandle, Error> {
        let missing: Vec<String> = fields
            .iter()
            .filter(|f| self.column(f).is_none())
            .map(|f| (*f).to_owned())
            .collect();
        if !missing.is_empty() {
            return Err(Error::MissingDatasetFields { fields: missing });
        }
        self.indexes
            .push((case, fields.iter().map(|f| (*f).to_owned()).collect()));
        Ok(IndexHandle(self.indexes.len() - 1))
    }

    fn index_fields(&self) -> Vec<(Case, Vec<String>)> {
        self.indexes.clone()
    }

    fn needs_reload(&self) -> bool {
        false
    }
}
