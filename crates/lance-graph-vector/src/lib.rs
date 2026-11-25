//! Vector as an upstream of lance-graph, with `lance-graph-vrl` as the glue.
//!
//! One VRL program serves both sides. Vector runs it as its `remap`
//! transform; `lance-graph-vrl` compiles the same source against the source
//! profile and, from it, [`output_profile`] declares the intake profile the
//! program's output lands in: one column per output name, carrying the IRI
//! and the declared shape of the source field it reads. A rename in VRL keeps
//! the field's identity, so Vector's output and the in-process polyfill (the
//! same recipe run by `lance-graph-vrl` over intake lanes) meet on one IRI.
//!
//! [`VectorRoute`] is the receiver Vector's `http` sink posts to (`encoding:
//! json`, newline-delimited framing, `application/x-ndjson`). Each line is
//! decoded once, at this boundary, into the row shape intake takes; nothing
//! after it reads JSON. Per batch it
//!
//! - checks `.lg.schema_rev` against the route's revision, refusing the whole
//!   batch on a mismatch;
//! - drops events whose `.lg.event_id` the table already holds or the batch
//!   repeats, through the table's own code space for that column, so a
//!   replayed or retried batch changes nothing;
//! - appends the rest to the tenant's table, all or nothing.
//!
//! The tenant is the table the caller routes to, never a payload field. The
//! answer maps to the status Vector's retry logic reads: [`Ack::status`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lance_graph_contract::pg_row::{AbsentAs, LaneEncoding, PgType};
use lance_graph_intake::{ColumnConfig, IntakeRefusal, ShapeConfig, SourceProfile, Table};
use lance_graph_vrl::Recipe;
use serde_json::value::RawValue;

/// The label and IRI of the event id column.
pub const EVENT_ID: (&str, &str) = ("lg.event_id", "lg:event_id");
/// The label and IRI of the row key, assigned by the route in arrival order.
pub const SEQ: (&str, &str) = ("lg.seq", "lg:seq");

/// Why a program cannot declare an output profile.
#[derive(Debug, Clone, PartialEq)]
pub enum GlueRefusal {
    /// An output reads a source field the source profile does not fully
    /// declare (type, encoding and absent spelling): the receiver checks a
    /// declared shape, it never detects one.
    Undeclared { output: String, field: String },
    /// An output name is reserved for the route (`lg.*`).
    Reserved(String),
    /// The output profile could not be registered.
    Intake(IntakeRefusal),
}

/// The intake profile a program's output lands in: one column per output
/// name, in the recipe's output order, each with the IRI and the declared
/// shape of the source field it reads; then the event id (text) and the row
/// key the route assigns.
pub fn output_profile(
    recipe: &Recipe,
    source: &SourceProfile,
    class: u32,
) -> Result<Arc<SourceProfile>, GlueRefusal> {
    let fields: Vec<(&str, &str)> = source.fields().collect();
    let mut config = ShapeConfig {
        key: Some(SEQ.0.to_owned()),
        ..ShapeConfig::default()
    };
    let mut columns: Vec<(String, String)> = Vec::new();
    for name in recipe.outputs() {
        if name.starts_with("lg.") || name == "lg" {
            return Err(GlueRefusal::Reserved(name.to_owned()));
        }
        let (label, iri) = recipe
            .output_field(name)
            .and_then(|f| fields.get(usize::from(f)))
            .copied()
            .ok_or_else(|| GlueRefusal::Undeclared {
                output: name.to_owned(),
                field: String::new(),
            })?;
        let declared = source
            .config()
            .columns
            .get(label)
            .filter(|c| c.ty.is_some() && c.encoding.is_some() && c.absent.is_some())
            .cloned()
            .ok_or_else(|| GlueRefusal::Undeclared {
                output: name.to_owned(),
                field: label.to_owned(),
            })?;
        config.columns.insert(name.to_owned(), declared);
        columns.push((name.to_owned(), iri.to_owned()));
    }
    config.columns.insert(
        EVENT_ID.0.to_owned(),
        ColumnConfig {
            ty: Some(PgType::Text),
            encoding: Some(LaneEncoding::TextCode),
            absent: Some(AbsentAs::Null),
        },
    );
    config.columns.insert(
        SEQ.0.to_owned(),
        ColumnConfig {
            ty: Some(PgType::Int4),
            encoding: Some(LaneEncoding::Signed),
            absent: Some(AbsentAs::Null),
        },
    );
    columns.push((EVENT_ID.0.to_owned(), EVENT_ID.1.to_owned()));
    columns.push((SEQ.0.to_owned(), SEQ.1.to_owned()));
    let pairs: Vec<(&str, &str)> = columns
        .iter()
        .map(|(l, i)| (l.as_str(), i.as_str()))
        .collect();
    SourceProfile::new(class, &pairs, config).map_err(GlueRefusal::Intake)
}

/// The program as Vector's `remap` runs it: `program`, then a projection
/// onto the recipe's outputs and `.lg`.
///
/// In-fold, the recipe yields only its outputs. A `remap` instead keeps every
/// field it did not delete, so a source object it read from (for example
/// `.properties`) would still be on the event and the route would refuse it
/// as [`Refusal::NotScalar`]. The projection makes the two engines emit the
/// same shape. `.lg` carries the event id and schema revision the remap sets.
pub fn vector_remap(program: &str, recipe: &Recipe) -> String {
    let mut fields: Vec<String> = vec![r#""lg": .lg"#.to_owned()];
    fields.extend(recipe.outputs().into_iter().map(|o| format!("{o:?}: .{o}")));
    format!("{}\n. = {{{}}}\n", program.trim_end(), fields.join(", "))
}

/// Why a batch was refused. Nothing of it was appended.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// Line `line` (from 1) is not a JSON object.
    Malformed { line: usize },
    /// Line `line` has no `.lg.event_id` string.
    MissingEventId { line: usize },
    /// Line `line` was written under another schema revision.
    SchemaRev { line: usize, got: Option<String> },
    /// Line `line` carries a field the output profile does not have.
    UnknownField { line: usize, field: String },
    /// Line `line` carries a nested value for a column.
    NotScalar { line: usize, field: String },
    /// Intake refused the batch (a cell that does not fit its column).
    Intake(IntakeRefusal),
}

/// What the route answers for one batch.
#[derive(Debug, Clone, PartialEq)]
pub enum Ack {
    /// The batch is in: `appended` new events, `duplicates` dropped by id.
    Committed { appended: usize, duplicates: usize },
    /// The batch is refused and nothing was appended.
    Refused(Refusal),
}

impl Ack {
    /// The HTTP status for Vector's `http` sink. 400 is outside Vector's
    /// retry set, so a refused batch is not retried; a committed one is 200,
    /// and a retried duplicate batch is 200 with every event a duplicate.
    pub fn status(&self) -> u16 {
        match self {
            Ack::Committed { .. } => 200,
            Ack::Refused(_) => 400,
        }
    }
}

/// The receiver of one tenant's events under one output profile.
pub struct VectorRoute {
    table: Table,
    labels: Vec<String>,
    schema_rev: String,
    next_seq: i64,
}

/// One typed cell of an event, as Vector holds it after VRL: what a sink
/// inside Vector hands over directly, with no JSON in between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell<'a> {
    /// The field is absent (VRL `null`, or not set).
    Absent,
    Int(i64),
    Bool(bool),
    Text(&'a str),
    /// A number's exact text (a decimal, never read through `f64`).
    Number(&'a str),
}

impl Cell<'_> {
    fn text(self) -> Option<String> {
        match self {
            Cell::Absent => None,
            Cell::Int(v) => Some(v.to_string()),
            Cell::Bool(b) => Some(if b { "true" } else { "false" }.to_owned()),
            Cell::Text(t) | Cell::Number(t) => Some(t.to_owned()),
        }
    }
}

/// One event after VRL: its id, the schema revision it was written under,
/// and its fields by output name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event<'a> {
    pub event_id: &'a str,
    pub schema_rev: Option<&'a str>,
    pub fields: Vec<(&'a str, Cell<'a>)>,
}

/// One event, its fields rendered for intake, numbered from 1 within the batch.
struct Item {
    line: usize,
    id: String,
    rev: Option<String>,
    fields: Vec<(String, Option<String>)>,
}

impl VectorRoute {
    /// An empty route for `profile` (from [`output_profile`]) under schema
    /// revision `schema_rev`.
    pub fn new(
        name: &str,
        profile: &Arc<SourceProfile>,
        schema_rev: &str,
    ) -> Result<VectorRoute, IntakeRefusal> {
        let labels: Vec<String> = profile.fields().map(|(l, _)| l.to_owned()).collect();
        let cols: Vec<&str> = labels.iter().map(String::as_str).collect();
        let table = Table::from_rows_in(name, profile, &cols, &[])?;
        Ok(VectorRoute {
            table,
            labels,
            schema_rev: schema_rev.to_owned(),
            next_seq: 1,
        })
    }

    /// The tenant's table: what the polyfill, Quack and the reports read.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// The landed table, mutably: Quack's `select_bound` keeps a selection on
    /// it. Rows still enter only through [`VectorRoute::receive`] and
    /// [`VectorRoute::receive_events`].
    pub fn table_mut(&mut self) -> &mut Table {
        &mut self.table
    }

    /// Take a batch of typed events, as a sink inside Vector hands them over.
    pub fn receive_events(&mut self, events: &[Event<'_>]) -> Ack {
        let items = events
            .iter()
            .enumerate()
            .map(|(n, e)| Item {
                line: n + 1,
                id: e.event_id.to_owned(),
                rev: e.schema_rev.map(str::to_owned),
                fields: e
                    .fields
                    .iter()
                    .map(|(k, c)| ((*k).to_owned(), c.text()))
                    .collect(),
            })
            .collect();
        self.admit(items)
    }

    /// Take one NDJSON body, as Vector's `http` sink posts it.
    pub fn receive(&mut self, body: &str) -> Ack {
        match ndjson_items(body) {
            Ok(items) => self.admit(items),
            Err(r) => Ack::Refused(r),
        }
    }

    /// Check, deduplicate and append, all or nothing.
    fn admit(&mut self, items: Vec<Item>) -> Ack {
        let rows = match self.rows(items) {
            Ok(rows) => rows,
            Err(r) => return Ack::Refused(r),
        };
        let (rows, duplicates) = rows;
        let cells: Vec<Vec<Option<&str>>> = rows
            .iter()
            .map(|r| r.iter().map(|c| c.as_deref()).collect())
            .collect();
        if let Err(e) = self.table.append(&cells) {
            return Ack::Refused(Refusal::Intake(e));
        }
        self.next_seq += rows.len() as i64;
        Ack::Committed {
            appended: rows.len(),
            duplicates,
        }
    }

    /// The new rows in column order, and how many events were duplicates.
    /// Reads nothing from the table but its event-id codes.
    #[allow(clippy::type_complexity)]
    fn rows(&self, items: Vec<Item>) -> Result<(Vec<Vec<Option<String>>>, usize), Refusal> {
        let frame = self.table.frame();
        let id_col = self.labels.len() - 2;
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut rows = Vec::new();
        let mut duplicates = 0;
        for item in items {
            let line = item.line;
            if item.rev.as_deref() != Some(self.schema_rev.as_str()) {
                return Err(Refusal::SchemaRev {
                    line,
                    got: item.rev,
                });
            }
            let mut row: Vec<Option<String>> = vec![None; self.labels.len()];
            for (field, value) in item.fields {
                let k = self.labels[..id_col]
                    .iter()
                    .position(|l| *l == field)
                    .ok_or(Refusal::UnknownField { line, field })?;
                row[k] = value;
            }
            let known = self.table.code_for(&frame, EVENT_ID.1, &item.id).is_some();
            if known || !seen.insert(item.id.clone()) {
                duplicates += 1;
                continue;
            }
            row[id_col] = Some(item.id);
            row[id_col + 1] = Some((self.next_seq + rows.len() as i64).to_string());
            rows.push(row);
        }
        Ok((rows, duplicates))
    }
}

/// Decode an NDJSON body into events, each line once.
fn ndjson_items(body: &str) -> Result<Vec<Item>, Refusal> {
    let mut items = Vec::new();
    for (n, raw) in body
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let line = n + 1;
        let mut obj: BTreeMap<String, Box<RawValue>> =
            serde_json::from_str(raw).map_err(|_| Refusal::Malformed { line })?;
        let lg: BTreeMap<String, Box<RawValue>> = obj
            .remove("lg")
            .and_then(|v| serde_json::from_str(v.get()).ok())
            .ok_or(Refusal::MissingEventId { line })?;
        let text = |k: &str| {
            lg.get(k)
                .and_then(|v| serde_json::from_str::<String>(v.get()).ok())
        };
        let id = text("event_id").ok_or(Refusal::MissingEventId { line })?;
        let mut fields = Vec::with_capacity(obj.len());
        for (field, value) in obj {
            let v = value.get();
            let cell = match v.as_bytes().first() {
                Some(b'n') => None,
                Some(b'"') => Some(
                    serde_json::from_str::<String>(v).map_err(|_| Refusal::Malformed { line })?,
                ),
                Some(b'{' | b'[') => return Err(Refusal::NotScalar { line, field }),
                // A number or a boolean: its own text, never through f64.
                _ => Some(v.to_owned()),
            };
            fields.push((field, cell));
        }
        items.push(Item {
            line,
            id,
            rev: text("schema_rev"),
            fields,
        });
    }
    Ok(items)
}
