//! D-QSP-H: the shared PostgreSQL parity harness.
//!
//! A lab instrument. It drives `psql` against a real PostgreSQL server and
//! hands back what PostgreSQL printed: the rows, with SQL NULL kept distinct
//! from the empty string, the result column types as `\gdesc` names them, and
//! the SQLSTATE of a rejected statement. It then compares that with an
//! implementation's answer: as a multiset when the query has no ORDER BY, in
//! order (a tie may reorder) when it has one, plus column types and error
//! codes. Without a server `Pg::connect` returns a `Skip` and callers print
//! `SKIP: ...` and return.
//!
//! The harness holds **final result sets only**: PostgreSQL's printed answer
//! and the implementation's printed answer, as text cells. That is an
//! external reference's output and the compared result, bounded by the result
//! size. It is not an execution path and answers no count, exists or sum
//! itself. It reports **no timings**.
//!
//! # Wire format
//!
//! `psql` runs with `-X -q -At -F <FIELD_SEP> -0 -P null=<NULL_MARK>`:
//! fields are separated by U+001F, records are NUL-terminated, and a NULL
//! prints as the marker `\u{1}NULL\u{1}`.
//!
//! Fixture constraint: text values must not contain U+001F and must not equal
//! the NULL marker. NUL cannot occur in PostgreSQL text. A U+001F inside a
//! value shows up as a row whose cell count differs from the `\gdesc` column
//! count and is reported as `PgError::Harness`.
//!
//! # Values
//!
//! Values are compared as PostgreSQL text. The caller renders implementation
//! values in PostgreSQL's output form (`t` / `f` for booleans, decimal
//! integers, ...). No float helper is provided: a caller that cannot render a
//! type returns `Answer::Unsupported`.
//!
//! # Connection
//!
//! The connection comes from the usual `PG*` variables (`PGHOST`, `PGPORT`,
//! `PGUSER`, `PGDATABASE`, ...). `PG_PARITY_PSQL` names a different `psql`
//! program. Each `Pg` works in its own scratch schema, dropped on drop.

mod compare;
mod pg;
mod report;

pub use compare::{
    compare, multiset_diff, ordered_diff, types_diff, Answer, Diff, Order, Outcome, Refusal,
};
pub use pg::{parse_rows, parse_sqlstate, Pg, PgError, Skip};
pub use report::{check, check_mutation, mutation_outcomes, Case, Report};

/// One result cell as PostgreSQL prints it. `None` is SQL NULL; `Some("")` is
/// the empty string. The two are never merged.
pub type Cell = Option<String>;
pub type Row = Vec<Cell>;

/// A result: column types as `\gdesc` names them (`integer`, `bigint`,
/// `text`, `boolean`, `double precision`, `numeric`, ...) and the rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rows {
    pub types: Vec<String>,
    pub rows: Vec<Row>,
}

/// `Some(v.to_string())`.
pub fn text(v: impl ToString) -> Cell {
    Some(v.to_string())
}

/// PostgreSQL's boolean output: `Some("t")` / `Some("f")`.
pub fn boolean(b: bool) -> Cell {
    Some(if b { "t" } else { "f" }.to_string())
}
