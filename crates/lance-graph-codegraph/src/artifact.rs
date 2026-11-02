//! The reader for `*.rows.tsv` artifacts (version 2).
//!
//! Lines stream straight into column lanes: one push per field per line, no
//! per-row struct. Columns are located by name from the `#schema` line, so a
//! file with permuted columns reads identically.

use std::fs;
use std::path::Path;

use crate::graph::IntakeRefusal;

/// The only artifact version this reader accepts.
pub const ROWS_VERSION: u32 = 2;

/// Columns every artifact must carry.
const REQUIRED: [&str; 7] = [
    "predicate",
    "subject",
    "object",
    "grade",
    "file",
    "start",
    "end",
];

/// Marks an absent (`-`) span bound in [`Lanes::start`] / [`Lanes::end`].
pub const NO_SPAN: u32 = u32::MAX;

/// All rows of a directory as parallel column lanes; index `i` of every lane
/// is row `i`. Invariant: all lanes have equal length.
#[derive(Debug, Default)]
pub struct Lanes {
    /// Relation name of each row.
    pub predicate: Vec<String>,
    /// Subject symbol of each row.
    pub subject: Vec<String>,
    /// Object text of each row.
    pub object: Vec<String>,
    /// Grade (provenance) of each row.
    pub grade: Vec<String>,
    /// Source file of each row, `-` when none.
    pub file: Vec<String>,
    /// Span start, [`NO_SPAN`] when `-`.
    pub start: Vec<u32>,
    /// Span end, [`NO_SPAN`] when `-`.
    pub end: Vec<u32>,
}

/// Read every `*.rows.tsv` in `dir`, sorted by file name so the row order is
/// deterministic.
pub fn read_dir(dir: &Path) -> Result<Lanes, IntakeRefusal> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).map_err(IntakeRefusal::Io)? {
        let path = entry.map_err(IntakeRefusal::Io)?.path();
        let is_rows = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".rows.tsv"));
        if is_rows {
            paths.push(path);
        }
    }
    paths.sort();
    let mut lanes = Lanes::default();
    for p in &paths {
        let text = fs::read_to_string(p).map_err(IntakeRefusal::Io)?;
        read_text(&text, &mut lanes)?;
    }
    Ok(lanes)
}

fn number(field: &str) -> Option<u32> {
    if field == "-" {
        Some(NO_SPAN)
    } else {
        field.parse().ok()
    }
}

/// Append one file's rows to `lanes`.
pub fn read_text(text: &str, lanes: &mut Lanes) -> Result<(), IntakeRefusal> {
    let mut lines = text.lines();
    let found = lines
        .next()
        .and_then(|l| l.strip_prefix("#version "))
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(0);
    if found != ROWS_VERSION {
        return Err(IntakeRefusal::Version { found });
    }
    let schema = lines
        .next()
        .and_then(|l| l.strip_prefix("#schema "))
        .ok_or_else(|| IntakeRefusal::Schema {
            missing: "#schema".to_string(),
        })?;
    let names: Vec<&str> = schema.split('\t').collect();
    let mut at = [0usize; 7];
    for (slot, want) in at.iter_mut().zip(REQUIRED) {
        *slot = names
            .iter()
            .position(|n| *n == want)
            .ok_or_else(|| IntakeRefusal::Schema {
                missing: want.to_string(),
            })?;
    }
    for (i, line) in lines.enumerate() {
        // Lines 1 and 2 were the header; data starts at line 3.
        let lineno = i + 3;
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != names.len() {
            return Err(IntakeRefusal::Arity { line: lineno });
        }
        let start = number(f[at[5]]).ok_or(IntakeRefusal::Arity { line: lineno })?;
        let end = number(f[at[6]]).ok_or(IntakeRefusal::Arity { line: lineno })?;
        lanes.predicate.push(f[at[0]].to_string());
        lanes.subject.push(f[at[1]].to_string());
        lanes.object.push(f[at[2]].to_string());
        lanes.grade.push(f[at[3]].to_string());
        lanes.file.push(f[at[4]].to_string());
        lanes.start.push(start);
        lanes.end.push(end);
    }
    Ok(())
}
