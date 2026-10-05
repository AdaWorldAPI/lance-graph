//! The developer world: names and textual literals, bound ONCE into a
//! numeric [`Query`].
//!
//! ```text
//!   table("users").where_eq("smtp", "alice@x.de").count()     ← may hold text
//!          │  bind(&binder)            names → TableId / Col
//!          │                            literals → the field's own numeric id
//!          ▼
//!   Query { filter: and([Plane(live), Cmp(Col(k), EqU32(id))]), agg: Count }
//!          │  lower                     ← no text exists past this line
//!          ▼
//!   Program → mask-risc → ndarray
//! ```
//!
//! **Describe once, resolve once, canonicalize once, execute numeric.** [`Query`] is already
//! the resolved form: every type in it is fixed-width. So there is no
//! separate "resolved query"; binding produces a `Query`, and a `Query` can
//! be lowered and executed any number of times without touching a name, a
//! label or a codebook again. This module is the only part of the crate
//! that holds text (`tests/string_fence.rs`).
//!
//! **Quack is ID-agnostic after binding.** A [`Binder`] is the consumer's
//! resolution membrane: its catalog names tables and fields, and its own
//! codebook turns a textual literal into whatever numeric identity that
//! field uses — a report CAM ordinal (a category whose label can be
//! renamed), an IAM `ValueId` (the exact text) or `KeyId` (its comparison
//! form), a batch-local SAP code. All of them arrive here as a `u32` in an
//! `EqU32`; their meaning, and what survives a rename or a re-observation,
//! stays with the binder that issued them.
//!
//! **Canonicalize once.** The membrane also fixes the physical
//! representation. A `Cmp::EqU32(v)` is a semantic number, not bytes, and
//! carries no byte order; `U32`/`I32`/`U64` lanes hand already-bound numbers
//! to the executor. Byte order exists only where a fixed-width integer is
//! read from or written to raw bytes, and there it is little-endian:
//!
//! - `u8`/`i8` and byte arrays: order-neutral, the sequence is the value
//!   (`Dn128 = [u8; 16]`, `MatchFacet16Strided` pattern/care bytes);
//! - `u16`/`u32`/`u64`/`u128` and signed twins: little-endian;
//! - a byte array's sub-field read as an integer: little-endian;
//! - `Register128`: four little-endian `u32` words.
//!
//! `0x1234_5678` therefore lives on the substrate as `[0x78, 0x56, 0x34,
//! 0x12]` (`tests/canonical_le.rs`, through `EqU32Strided`). This is
//! substrate policy, not application vocabulary: no query, table
//! declaration or `Cmp` names an endianness. Population execution sees no
//! names, no strings, no catalog or label lookup and no ambiguous byte
//! order.
//!
//! The same lifecycle as `lance_graph_contract::hotplug`'s
//! `SlabDeclaration → resolve_for_context → ResolvedReading`: an external
//! description is resolved at one boundary into a stable numeric contract,
//! which is then executed over a population as often as needed.
//!
//! Schema declarations follow the same shape ([`create_table`] →
//! [`ResolvedTable`]); no crate owns a table catalog or storage allocation
//! yet, so [`Registrar`] has no production implementation.

use crate::{Agg, Cmp, Col, Filter, Mask, Query};

/// A table, after its name has been resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableId(pub u32);

/// What literal a bound field compares against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    /// A signed integer lane (`I32`).
    I32,
    /// A `U32` lane of numeric identities issued by the binder's codebook.
    /// Textual literals for it are resolved by [`Binder::code`].
    Code,
}

/// A field, after its name has been resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundField {
    /// The lane the executor reads.
    pub col: Col,
    /// What it compares against.
    pub kind: FieldKind,
}

/// A consumer's resolution membrane: catalog plus codebook. Called only by
/// [`Draft::bind`]; nothing in execution holds or calls it.
pub trait Binder {
    /// A table by name.
    fn table(&self, name: &str) -> Option<TableId>;
    /// The plane every row of the table must pass (its validity plane).
    fn live(&self, table: TableId) -> Mask;
    /// A field of the table by name.
    fn field(&self, table: TableId, name: &str) -> Option<BoundField>;
    /// The numeric identity of a textual literal in a field's domain.
    /// Never mints: a literal the domain has never seen matches nothing and
    /// is reported as [`BindError::UnknownValue`].
    fn code(&self, table: TableId, field: Col, literal: &str) -> Option<u32>;
}

/// A literal as a developer writes it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    /// Text, resolved by the field's codebook.
    Text(String),
    /// An integer.
    Int(i64),
}

impl From<&str> for Literal {
    fn from(s: &str) -> Self {
        Literal::Text(s.to_string())
    }
}
impl From<String> for Literal {
    fn from(s: String) -> Self {
        Literal::Text(s)
    }
}
impl From<i64> for Literal {
    fn from(v: i64) -> Self {
        Literal::Int(v)
    }
}
impl From<i32> for Literal {
    fn from(v: i32) -> Self {
        Literal::Int(i64::from(v))
    }
}

/// A comparison operator a frontend can express on any field kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// `=`
    Eq,
    /// `<>`
    Ne,
}

/// One unbound predicate: `field <op> literal`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Predicate {
    table: Option<String>,
    field: String,
    op: Op,
    value: Literal,
}

/// A typed field descriptor, for a schema written down in code
/// (`const SMTP: FieldRef = FieldRef::new("users", "smtp");`). It holds
/// names only; binding resolves them like any other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldRef {
    table: &'static str,
    field: &'static str,
}

impl FieldRef {
    /// A field of a table.
    pub const fn new(table: &'static str, field: &'static str) -> Self {
        Self { table, field }
    }
    /// `field = value`.
    pub fn eq(self, value: impl Into<Literal>) -> Predicate {
        self.pred(Op::Eq, value)
    }
    /// `field <> value`.
    pub fn ne(self, value: impl Into<Literal>) -> Predicate {
        self.pred(Op::Ne, value)
    }
    fn pred(self, op: Op, value: impl Into<Literal>) -> Predicate {
        Predicate {
            table: Some(self.table.to_string()),
            field: self.field.to_string(),
            op,
            value: value.into(),
        }
    }
}

/// What the query returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Want {
    Count,
    Rows,
}

/// An unbound query over one table. Holds text; [`Draft::bind`] is the only
/// way out of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Draft {
    table: String,
    preds: Vec<Predicate>,
    want: Want,
}

/// Start a query over a table.
pub fn table(name: impl Into<String>) -> Draft {
    Draft {
        table: name.into(),
        preds: Vec::new(),
        want: Want::Rows,
    }
}

/// Why a draft did not bind. Each names the developer-world term that
/// failed, so the error is reported in the developer's vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindError {
    /// No such table.
    UnknownTable(String),
    /// No such field in the table.
    UnknownField {
        /// Table.
        table: String,
        /// Field.
        field: String,
    },
    /// The field's domain has no such value (nothing is minted).
    UnknownValue {
        /// Field.
        field: String,
        /// The literal.
        value: String,
    },
    /// A text literal for an integer field, or an integer for a coded one.
    KindMismatch {
        /// Field.
        field: String,
    },
    /// An integer literal outside the field's lane.
    OutOfRange {
        /// Field.
        field: String,
    },
    /// A typed descriptor of another table used on this one.
    WrongTable {
        /// The query's table.
        query: String,
        /// The descriptor's table.
        field: String,
    },
}

impl Draft {
    /// `AND field = value`.
    pub fn where_eq(self, field: &str, value: impl Into<Literal>) -> Self {
        self.where_(Predicate {
            table: None,
            field: field.to_string(),
            op: Op::Eq,
            value: value.into(),
        })
    }
    /// `AND predicate` (e.g. from a [`FieldRef`]).
    pub fn where_(mut self, p: Predicate) -> Self {
        self.preds.push(p);
        self
    }
    /// Return the number of matching rows.
    pub fn count(mut self) -> Self {
        self.want = Want::Count;
        self
    }
    /// Return the matching rows (the default).
    pub fn rows(mut self) -> Self {
        self.want = Want::Rows;
        self
    }

    /// Resolve every name and literal once and produce the numeric
    /// [`Query`]. Calls the binder once per table, field and textual
    /// literal; the result holds none of them.
    ///
    /// # Errors
    ///
    /// The first [`BindError`]; nothing is minted or registered on failure.
    pub fn bind(&self, b: &dyn Binder) -> Result<Query, BindError> {
        let t = b
            .table(&self.table)
            .ok_or_else(|| BindError::UnknownTable(self.table.clone()))?;
        let mut parts = vec![Filter::plane(b.live(t))];
        for p in &self.preds {
            if let Some(owner) = &p.table {
                if owner != &self.table {
                    return Err(BindError::WrongTable {
                        query: self.table.clone(),
                        field: owner.clone(),
                    });
                }
            }
            parts.push(self.bind_pred(b, t, p)?);
        }
        let agg = match self.want {
            Want::Count => Agg::Count,
            Want::Rows => Agg::Rows,
        };
        Ok(Query {
            filter: Filter::and(parts),
            agg,
        })
    }

    fn bind_pred(&self, b: &dyn Binder, t: TableId, p: &Predicate) -> Result<Filter, BindError> {
        let f = b
            .field(t, &p.field)
            .ok_or_else(|| BindError::UnknownField {
                table: self.table.clone(),
                field: p.field.clone(),
            })?;
        let mismatch = || BindError::KindMismatch {
            field: p.field.clone(),
        };
        let cmp = match (f.kind, &p.value) {
            (FieldKind::I32, Literal::Int(v)) => {
                let v = i32::try_from(*v).map_err(|_| BindError::OutOfRange {
                    field: p.field.clone(),
                })?;
                match p.op {
                    Op::Eq => Cmp::EqI32(v),
                    Op::Ne => Cmp::NeI32(v),
                }
            }
            (FieldKind::Code, Literal::Text(s)) => {
                let id = b.code(t, f.col, s).ok_or_else(|| BindError::UnknownValue {
                    field: p.field.clone(),
                    value: s.clone(),
                })?;
                match p.op {
                    Op::Eq => Cmp::EqU32(id),
                    Op::Ne => Cmp::NeU32(id),
                }
            }
            _ => return Err(mismatch()),
        };
        Ok(Filter::cmp(f.col, cmp))
    }
}

// ---- schema: the same lifecycle for declarations ---------------------------

/// An unbound table declaration: `CREATE TABLE <name> WIDTH <bytes>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableDeclaration {
    name: String,
    width: u32,
}

/// `CREATE TABLE name`; set the row width with [`TableDeclaration::width`].
pub fn create_table(name: impl Into<String>) -> TableDeclaration {
    TableDeclaration {
        name: name.into(),
        width: 0,
    }
}

/// A table after registration: the name is gone, only the id and the
/// physical declaration remain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResolvedTable {
    /// The table's id; every later query and write addresses this.
    pub id: TableId,
    /// Declared row width in bytes.
    pub width: u32,
}

/// Why a declaration did not register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterError {
    /// No width, or zero.
    NoWidth(String),
    /// The name is taken.
    Exists(String),
    /// The owner refused the physical declaration (e.g. an unsupported width).
    Refused {
        /// Table.
        table: String,
        /// Width asked for.
        width: u32,
    },
}

/// The owner of a table catalog and its physical allocation. No crate owns
/// that today; this trait names the seam, and only tests implement it.
pub trait Registrar {
    /// Register a new table, minting its id, or refuse.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Exists`] or [`RegisterError::Refused`].
    fn register(&mut self, name: &str, width: u32) -> Result<TableId, RegisterError>;
}

impl TableDeclaration {
    /// `WIDTH bytes`.
    pub fn width(mut self, bytes: u32) -> Self {
        self.width = bytes;
        self
    }
    /// Register once; the result carries no name.
    ///
    /// # Errors
    ///
    /// [`RegisterError::NoWidth`], or whatever the registrar refuses.
    pub fn register(&self, r: &mut dyn Registrar) -> Result<ResolvedTable, RegisterError> {
        if self.width == 0 {
            return Err(RegisterError::NoWidth(self.name.clone()));
        }
        Ok(ResolvedTable {
            id: r.register(&self.name, self.width)?,
            width: self.width,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower;
    use lance_graph_mask_risc::{
        execute_into, materialize_rows, words_for, Foreign, LaneRef, Out, Planes, Scratch, Value,
    };
    use std::cell::Cell;

    /// A users table: plane 0 = live rows, lane 0 = smtp key (coded),
    /// lane 1 = age (i32). Every binder call is counted.
    struct Users {
        smtp: Vec<&'static str>,
        calls: Cell<u32>,
    }
    impl Users {
        fn new() -> Self {
            Self {
                smtp: vec!["alice@x.de", "bob@x.de", "carol@x.de"],
                calls: Cell::new(0),
            }
        }
        fn tick(&self) {
            self.calls.set(self.calls.get() + 1);
        }
    }
    impl Binder for Users {
        fn table(&self, name: &str) -> Option<TableId> {
            self.tick();
            (name == "users").then_some(TableId(7))
        }
        fn live(&self, _: TableId) -> Mask {
            self.tick();
            Mask(0)
        }
        fn field(&self, _: TableId, name: &str) -> Option<BoundField> {
            self.tick();
            match name {
                "smtp" => Some(BoundField {
                    col: Col(0),
                    kind: FieldKind::Code,
                }),
                "age" => Some(BoundField {
                    col: Col(1),
                    kind: FieldKind::I32,
                }),
                _ => None,
            }
        }
        fn code(&self, _: TableId, _: Col, literal: &str) -> Option<u32> {
            self.tick();
            self.smtp
                .iter()
                .position(|s| *s == literal.to_ascii_lowercase())
                .map(|i| i as u32)
        }
    }

    /// 200 rows: smtp code = i % 3, age = 20 + i % 50, row 13 not live.
    fn run(q: &Query) -> (Value, Vec<usize>) {
        let n = 200;
        let codes: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
        let ages: Vec<i32> = (0..n).map(|i| 20 + (i % 50) as i32).collect();
        let mut live = vec![u64::MAX; words_for(n)];
        live[3] &= (1u64 << (n % 64)) - 1;
        live[0] &= !(1 << 13);
        let lanes = [LaneRef::U32(&codes), LaneRef::I32(&ages)];
        let masks: [&[u64]; 1] = [&live];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        let p = lower(q).unwrap();
        let mut scratch = Scratch::for_program(&p, n).unwrap();
        let mut bits = vec![0u64; words_for(n)];
        let out = if q.agg == Agg::Rows {
            Out::Mask(&mut bits)
        } else {
            Out::None
        };
        let v = execute_into(&p, &planes, &Foreign::NONE, &mut scratch, out).unwrap();
        (v, materialize_rows(&bits, n))
    }

    #[test]
    fn a_text_literal_binds_once_to_the_query_an_expert_writes() {
        let users = Users::new();
        let draft = table("users").where_eq("smtp", "Alice@X.de").count();
        let bound = draft.bind(&users).unwrap();
        // table + live + field + code: one call each, no more.
        assert_eq!(users.calls.get(), 4);
        let expert = Query {
            filter: Filter::and([Filter::plane(Mask(0)), Filter::cmp(Col(0), Cmp::EqU32(0))]),
            agg: Agg::Count,
        };
        assert_eq!(bound, expert);
        assert_eq!(lower(&bound).unwrap(), lower(&expert).unwrap());
        // Execute twice: the binder is not reachable from here, and the
        // counter proves it was not called.
        let first = run(&bound).0;
        let second = run(&bound).0;
        assert_eq!(first, second);
        // Rows with code 0 are i % 3 == 0: 67 of 200, minus none (row 13 is
        // code 1, the dead row is not one of them).
        assert_eq!(first, Value::Count(67));
        assert_eq!(users.calls.get(), 4);
    }

    #[test]
    fn a_typed_descriptor_binds_to_the_same_query() {
        const SMTP: FieldRef = FieldRef::new("users", "smtp");
        const AGE: FieldRef = FieldRef::new("users", "age");
        let users = Users::new();
        let by_name = table("users")
            .where_eq("smtp", "carol@x.de")
            .where_eq("age", 21)
            .rows()
            .bind(&users)
            .unwrap();
        let typed = table("users")
            .where_(SMTP.eq("carol@x.de"))
            .where_(AGE.eq(21))
            .bind(&users)
            .unwrap();
        assert_eq!(by_name, typed);
        // smtp code 2 ⇔ i % 3 == 2; age 21 ⇔ i % 50 == 1. i ≡ 101 (mod 150).
        assert_eq!(run(&typed).1, vec![101]);
        // The dead row is excluded by the bound live plane.
        let row13 = table("users")
            .where_eq("age", 33)
            .rows()
            .bind(&users)
            .unwrap();
        assert_eq!(run(&row13).1, vec![63, 113, 163]);
    }

    #[test]
    fn binding_fails_in_the_developers_vocabulary_and_never_mints() {
        let users = Users::new();
        let err = |d: Draft| d.bind(&users).unwrap_err();
        assert_eq!(
            err(table("people")),
            BindError::UnknownTable("people".into())
        );
        assert_eq!(
            err(table("users").where_eq("mail", "a")),
            BindError::UnknownField {
                table: "users".into(),
                field: "mail".into()
            }
        );
        assert_eq!(
            err(table("users").where_eq("smtp", "dave@x.de")),
            BindError::UnknownValue {
                field: "smtp".into(),
                value: "dave@x.de".into()
            }
        );
        assert_eq!(
            err(table("users").where_eq("age", "old")),
            BindError::KindMismatch {
                field: "age".into()
            }
        );
        assert_eq!(
            err(table("users").where_eq("age", i64::MAX)),
            BindError::OutOfRange {
                field: "age".into()
            }
        );
        const OTHER: FieldRef = FieldRef::new("groups", "smtp");
        assert_eq!(
            err(table("users").where_(OTHER.eq("alice@x.de"))),
            BindError::WrongTable {
                query: "users".into(),
                field: "groups".into()
            }
        );
        assert_eq!(users.smtp.len(), 3, "an unknown value minted nothing");
    }

    #[test]
    fn create_table_registers_once_and_keeps_no_name() {
        #[derive(Default)]
        struct Mem(Vec<(String, u32)>);
        impl Registrar for Mem {
            fn register(&mut self, name: &str, width: u32) -> Result<TableId, RegisterError> {
                if self.0.iter().any(|(n, _)| n == name) {
                    return Err(RegisterError::Exists(name.into()));
                }
                if width != 512 {
                    return Err(RegisterError::Refused {
                        table: name.into(),
                        width,
                    });
                }
                self.0.push((name.into(), width));
                Ok(TableId(self.0.len() as u32 - 1))
            }
        }
        let mut mem = Mem::default();
        let users = create_table("users").width(512).register(&mut mem).unwrap();
        assert_eq!(
            users,
            ResolvedTable {
                id: TableId(0),
                width: 512
            }
        );
        assert_eq!(
            create_table("users").width(512).register(&mut mem),
            Err(RegisterError::Exists("users".into()))
        );
        assert_eq!(
            create_table("groups").register(&mut mem),
            Err(RegisterError::NoWidth("groups".into()))
        );
        assert!(matches!(
            create_table("groups").width(100).register(&mut mem),
            Err(RegisterError::Refused { .. })
        ));
    }
}
