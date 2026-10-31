//! D-SAG-6 evidence rows, typed.
//!
//! The row format and its vocabulary are the ones
//! `crates/sag-ra-probe/tools/project_rows.py` prints:
//!
//! ```text
//! relation  subject  object  witness  evidence
//! ```
//!
//! with evidence one of `WITNESSED`, `POSSIBLE`, `UNKNOWN` (that tool spells
//! the UNKNOWN class through its `UNJOINED` relation). The enums below are a
//! typed mirror of those strings for this lab crate only; they are not a
//! global evidence ontology and nothing outside this crate uses them.
//!
//! Parsing is strict: an unknown relation or evidence string, a wrong column
//! count, or a relation carried at an evidence class `project_rows.py` never
//! assigns it (a `MAY_DISPATCH` row labelled `WITNESSED`, say) is a
//! [`RowError`], never a silently accepted row.

/// The `relation` column of a D-SAG-6 row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Relation {
    Defines,
    HasImpls,
    Calls,
    MayDispatch,
    MayPanic,
    WritesCallerMemory,
    WritesThroughLocalRef,
    Unjoined,
}

impl Relation {
    pub fn parse(s: &str) -> Option<Relation> {
        Some(match s {
            "DEFINES" => Relation::Defines,
            "HAS_IMPLS" => Relation::HasImpls,
            "CALLS" => Relation::Calls,
            "MAY_DISPATCH" => Relation::MayDispatch,
            "MAY_PANIC" => Relation::MayPanic,
            "WRITES_CALLER_MEMORY" => Relation::WritesCallerMemory,
            "WRITES_THROUGH_LOCAL_REF" => Relation::WritesThroughLocalRef,
            "UNJOINED" => Relation::Unjoined,
            _ => return None,
        })
    }

    /// The only evidence class `project_rows.py` emits this relation at.
    pub fn evidence(self) -> Evidence {
        match self {
            Relation::Defines
            | Relation::HasImpls
            | Relation::Calls
            | Relation::MayPanic
            | Relation::WritesCallerMemory => Evidence::Witnessed,
            Relation::MayDispatch | Relation::WritesThroughLocalRef => Evidence::Possible,
            Relation::Unjoined => Evidence::Unknown,
        }
    }
}

/// The `evidence` column of a D-SAG-6 row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Evidence {
    /// A compiler frontend reports it.
    Witnessed,
    /// Dispatch or a write whose target the frontend does not fix.
    Possible,
    /// A fact one frontend has and the other cannot be joined to.
    Unknown,
}

impl Evidence {
    pub fn parse(s: &str) -> Option<Evidence> {
        Some(match s {
            "WITNESSED" => Evidence::Witnessed,
            "POSSIBLE" => Evidence::Possible,
            "UNKNOWN" => Evidence::Unknown,
            _ => return None,
        })
    }
}

/// One row.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Row {
    pub relation: Relation,
    pub subject: String,
    pub object: String,
    pub witness: String,
    pub evidence: Evidence,
}

/// Why a row was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowError {
    /// The header line is not `relation subject object witness evidence`.
    BadHeader(String),
    /// A data line without exactly five tab-separated columns.
    Columns {
        line: usize,
        got: usize,
    },
    UnknownRelation {
        line: usize,
        relation: String,
    },
    UnknownEvidence {
        line: usize,
        evidence: String,
    },
    /// The relation at an evidence class it is never emitted at.
    EvidenceMismatch {
        line: usize,
        relation: Relation,
        evidence: Evidence,
    },
}

pub const HEADER: &str = "relation\tsubject\tobject\twitness\tevidence";

/// Parse `project_rows.py` output.
pub fn parse_rows(tsv: &str) -> Result<Vec<Row>, RowError> {
    let mut lines = tsv
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty());
    match lines.next() {
        Some((_, h)) if h == HEADER => {}
        Some((_, h)) => return Err(RowError::BadHeader(h.to_string())),
        None => return Err(RowError::BadHeader(String::new())),
    }
    let mut rows = Vec::new();
    for (i, l) in lines {
        let line = i + 1;
        let cols: Vec<&str> = l.split('\t').collect();
        if cols.len() != 5 {
            return Err(RowError::Columns {
                line,
                got: cols.len(),
            });
        }
        let relation = Relation::parse(cols[0]).ok_or_else(|| RowError::UnknownRelation {
            line,
            relation: cols[0].to_string(),
        })?;
        let evidence = Evidence::parse(cols[4]).ok_or_else(|| RowError::UnknownEvidence {
            line,
            evidence: cols[4].to_string(),
        })?;
        if relation.evidence() != evidence {
            return Err(RowError::EvidenceMismatch {
                line,
                relation,
                evidence,
            });
        }
        rows.push(Row {
            relation,
            subject: cols[1].to_string(),
            object: cols[2].to_string(),
            witness: cols[3].to_string(),
            evidence,
        });
    }
    Ok(rows)
}
