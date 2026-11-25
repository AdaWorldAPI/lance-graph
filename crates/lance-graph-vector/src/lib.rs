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
//! Vector and lance-graph run in one process: nothing between them is
//! serialized. [`VectorRoute`] takes Vector's own `vrl::value::Value`s
//! ([`VectorRoute::receive_values`]) or typed events
//! ([`VectorRoute::receive_events`]), and places each value into its lane with
//! intake's [`Table::append_values`]: no text in between. Per batch it
//!
//! - checks `.lg.schema_rev` against the route's revision, refusing the whole
//!   batch on a mismatch;
//! - drops events whose `.lg.event_id` the route has already taken or the
//!   batch repeats, by a 128-bit hash held in four lanes, so a replayed or
//!   retried batch changes nothing;
//! - appends the rest to the tenant's table, all or nothing.
//!
//! The tenant is the table the caller routes to, never a payload field.

use std::collections::HashSet;
use std::sync::Arc;

use lance_graph_contract::pg_row::{AbsentAs, LaneEncoding, PgType};
use lance_graph_intake::{ColumnConfig, Datum, IntakeRefusal, ShapeConfig, SourceProfile, Table};
use lance_graph_vrl::Recipe;

/// The four lanes of the event id's 128-bit hash, lowest word first.
pub const EVENT_ID: [(&str, &str); 4] = [
    ("lg.id0", "lg:event_id/0"),
    ("lg.id1", "lg:event_id/1"),
    ("lg.id2", "lg:event_id/2"),
    ("lg.id3", "lg:event_id/3"),
];
/// The label and IRI of the row key, assigned by the route in arrival order.
pub const SEQ: (&str, &str) = ("lg.seq", "lg:seq");

/// FNV-1a over the event id's bytes, 128-bit. The id is Vector's (a sha256
/// the remap computes); the hash only has to keep distinct ids apart, and at
/// 128 bits a collision is out of reach for any event count a table holds.
fn id_hash(id: &str) -> u128 {
    const OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013B;
    id.bytes()
        .fold(OFFSET, |h, b| (h ^ u128::from(b)).wrapping_mul(PRIME))
}

fn id_words(h: u128) -> [i64; 4] {
    std::array::from_fn(|i| i64::from((h >> (32 * i)) as u32))
}

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
/// shape of the source field it reads; then the event id's four hash lanes
/// and the row key the route assigns.
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
    for (label, _) in EVENT_ID {
        config.columns.insert(
            label.to_owned(),
            ColumnConfig {
                ty: Some(PgType::Int8),
                encoding: Some(LaneEncoding::Unsigned),
                absent: Some(AbsentAs::Null),
            },
        );
    }
    config.columns.insert(
        SEQ.0.to_owned(),
        ColumnConfig {
            ty: Some(PgType::Int4),
            encoding: Some(LaneEncoding::Signed),
            absent: Some(AbsentAs::Null),
        },
    );
    for (label, iri) in EVENT_ID {
        columns.push((label.to_owned(), iri.to_owned()));
    }
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
    /// Event `event` (from 1) is not an object.
    NotAnObject { event: usize },
    /// Event `event` has no `.lg.event_id` string.
    MissingEventId { event: usize },
    /// Event `event` was written under another schema revision.
    SchemaRev { event: usize, got: Option<String> },
    /// Event `event` carries a field the output profile does not have.
    UnknownField { event: usize, field: String },
    /// Event `event` carries a nested value, a float or a non-UTF-8 string for
    /// a column. A float is an `f64` inside Vector already: a value that must
    /// stay exact is kept as a string by the remap.
    NotScalar { event: usize, field: String },
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

/// One typed cell of an event, as Vector holds it after VRL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell<'a> {
    /// The field is absent (VRL `null`, or not set).
    Absent,
    Int(i64),
    Bool(bool),
    Text(&'a str),
}

impl<'a> Cell<'a> {
    fn datum(self) -> Option<Datum<'a>> {
        match self {
            Cell::Absent => None,
            Cell::Int(v) => Some(Datum::Int(v)),
            Cell::Bool(b) => Some(Datum::Bool(b)),
            Cell::Text(t) => Some(Datum::Text(t)),
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

/// The receiver of one tenant's events under one output profile.
pub struct VectorRoute {
    table: Table,
    labels: Vec<String>,
    schema_rev: String,
    next_seq: i64,
    taken: HashSet<u128>,
}

type Row<'a> = Vec<Option<Datum<'a>>>;

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
            taken: HashSet::new(),
        })
    }

    /// The tenant's table: what the polyfill, Quack and the reports read.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// The landed table, mutably: Quack's `select_bound` keeps a selection on
    /// it. Rows still enter only through the route's `receive_*` methods.
    pub fn table_mut(&mut self) -> &mut Table {
        &mut self.table
    }

    /// Take a batch of typed events, as a sink inside Vector hands them over.
    pub fn receive_events(&mut self, events: &[Event<'_>]) -> Ack {
        let mut batch = Batch::new(self);
        for (n, e) in events.iter().enumerate() {
            let event = n + 1;
            if let Err(r) = batch.event(event, e.event_id, e.schema_rev, |push| {
                for (field, cell) in &e.fields {
                    push(field, cell.datum())?;
                }
                Ok(())
            }) {
                return Ack::Refused(r);
            }
        }
        batch.commit()
    }
}

/// One batch being admitted: the new rows, the duplicates' rows (checked for
/// fit, never appended), and the ids it takes.
struct Batch<'r, 'a> {
    route: &'r mut VectorRoute,
    rows: Vec<Row<'a>>,
    dups: Vec<Row<'a>>,
    fresh: Vec<u128>,
}

impl<'r, 'a> Batch<'r, 'a> {
    fn new(route: &'r mut VectorRoute) -> Self {
        Batch {
            route,
            rows: Vec::new(),
            dups: Vec::new(),
            fresh: Vec::new(),
        }
    }

    /// Admit one event; `fields` pushes each `(name, value)` it carries.
    fn event(
        &mut self,
        event: usize,
        id: &str,
        rev: Option<&str>,
        fields: impl FnOnce(
            &mut dyn FnMut(&str, Option<Datum<'a>>) -> Result<(), Refusal>,
        ) -> Result<(), Refusal>,
    ) -> Result<(), Refusal> {
        let r = &*self.route;
        if rev != Some(r.schema_rev.as_str()) {
            return Err(Refusal::SchemaRev {
                event,
                got: rev.map(str::to_owned),
            });
        }
        let outputs = r.labels.len() - EVENT_ID.len() - 1;
        let mut row: Row<'a> = vec![None; r.labels.len()];
        fields(&mut |field, value| {
            let k = r.labels[..outputs]
                .iter()
                .position(|l| l == field)
                .ok_or_else(|| Refusal::UnknownField {
                    event,
                    field: field.to_owned(),
                })?;
            row[k] = value;
            Ok(())
        })?;
        let h = id_hash(id);
        for (i, w) in id_words(h).into_iter().enumerate() {
            row[outputs + i] = Some(Datum::Int(w));
        }
        let duplicate = r.taken.contains(&h) || self.fresh.contains(&h);
        let seq = r.next_seq + self.rows.len() as i64;
        row[outputs + EVENT_ID.len()] = Some(Datum::Int(seq));
        if duplicate {
            self.dups.push(row);
        } else {
            self.fresh.push(h);
            self.rows.push(row);
        }
        Ok(())
    }

    fn commit(self) -> Ack {
        let Batch {
            route,
            rows,
            dups,
            fresh,
        } = self;
        // A skipped duplicate must still fit: a replay that carries a value
        // its column cannot hold is refused, not acknowledged.
        if let Err(e) = route.table.check_values(&dups) {
            return Ack::Refused(Refusal::Intake(e));
        }
        if let Err(e) = route.table.append_values(&rows) {
            return Ack::Refused(Refusal::Intake(e));
        }
        route.next_seq += rows.len() as i64;
        route.taken.extend(fresh);
        Ack::Committed {
            appended: rows.len(),
            duplicates: dups.len(),
        }
    }
}

mod values {
    use vrl::value::Value;

    use super::{Ack, Batch, Datum, Refusal, VectorRoute};

    fn text(v: &Value) -> Option<&str> {
        match v {
            Value::Bytes(b) => std::str::from_utf8(b).ok(),
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    impl VectorRoute {
        /// Take a batch of Vector's own events (`LogEvent::value()`), read in
        /// place: each value goes into its lane from where Vector holds it.
        pub fn receive_values(&mut self, events: &[&Value]) -> Ack {
            let mut batch = Batch::new(self);
            for (n, v) in events.iter().enumerate() {
                let event = n + 1;
                let Value::Object(map) = v else {
                    return Ack::Refused(Refusal::NotAnObject { event });
                };
                let lg = match map.get("lg") {
                    Some(Value::Object(lg)) => lg,
                    _ => return Ack::Refused(Refusal::MissingEventId { event }),
                };
                let Some(id) = lg.get("event_id").and_then(text) else {
                    return Ack::Refused(Refusal::MissingEventId { event });
                };
                let rev = lg.get("schema_rev").and_then(text);
                let admitted = batch.event(event, id, rev, |push| {
                    for (k, v) in map.iter() {
                        let field = k.as_str();
                        if field == "lg" {
                            continue;
                        }
                        let datum = match v {
                            Value::Null => None,
                            Value::Integer(i) => Some(Datum::Int(*i)),
                            Value::Boolean(b) => Some(Datum::Bool(*b)),
                            other => match text(other) {
                                Some(t) => Some(Datum::Text(t)),
                                None => {
                                    return Err(Refusal::NotScalar {
                                        event,
                                        field: field.to_owned(),
                                    })
                                }
                            },
                        };
                        push(field, datum)?;
                    }
                    Ok(())
                });
                if let Err(r) = admitted {
                    return Ack::Refused(r);
                }
            }
            batch.commit()
        }
    }
}
