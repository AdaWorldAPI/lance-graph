//! The `psql` runner and the two pure parsers.

use crate::{Row, Rows};
use std::ffi::OsString;
use std::io::Write;
use std::process::{Command, Stdio};

const FIELD_SEP: char = '\u{1f}'; // psql -F
const NULL_MARK: &str = "\u{1}NULL\u{1}"; // psql -P null=
                                          // records are NUL-terminated: psql -0

/// No usable server. Callers print `SKIP: {0}` and return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skip(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PgError {
    /// The server rejected a statement. `sqlstate` is the five-character code.
    Sql { sqlstate: String, stderr: String },
    /// psql failed without a SQLSTATE, or its output could not be parsed.
    /// Never a skip: the server was reachable at `connect`.
    Harness(String),
}

impl std::fmt::Display for PgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PgError::Sql { sqlstate, stderr } => {
                write!(f, "SQLSTATE {sqlstate}: {}", stderr.trim())
            }
            PgError::Harness(m) => write!(f, "harness: {m}"),
        }
    }
}

pub struct Pg {
    program: OsString,
    schema: String,
    version: String,
}

impl Pg {
    /// Checks that psql runs and a server answers, then creates a scratch
    /// schema `pgh_<pid>_<nanos>`. Any failure here is a `Skip` with the
    /// reason (psql missing, connection refused, no CREATE privilege).
    pub fn connect() -> Result<Pg, Skip> {
        let program = std::env::var_os("PG_PARITY_PSQL").unwrap_or_else(|| OsString::from("psql"));
        let out = Command::new(&program)
            .args(["-X", "-q", "-At", "-c", "SHOW server_version"])
            .output()
            .map_err(|e| Skip(format!("could not run psql ({e})")))?;
        if !out.status.success() {
            return Err(Skip(format!(
                "psql failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pg = Pg {
            program,
            schema: format!("pgh_{}_{}", std::process::id(), nanos),
            version,
        };
        pg.run(format!("CREATE SCHEMA {};\n", pg.schema))
            .map_err(|e| Skip(e.to_string()))?;
        Ok(pg)
    }

    /// `SHOW server_version`, read once at connect.
    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Runs a whole script in ONE psql session (setup, mutation, or a batch
    /// of queries; `CREATE TEMP TABLE` lives for this call only). Returns the
    /// rows of every statement, concatenated in order.
    pub fn script(&self, sql: &str) -> Result<Vec<Row>, PgError> {
        let out = self.run(sql.to_string())?;
        parse_rows(&out).map_err(PgError::Harness)
    }

    /// Runs one query. Column types come from `\gdesc` (first session), rows
    /// from executing it (second session). A trailing `;` and whitespace on
    /// `sql` are trimmed. Tables it reads must be non-temporary (created via
    /// `script`; they land in the scratch schema).
    pub fn query(&self, sql: &str) -> Result<Rows, PgError> {
        let q = sql.trim_end();
        let q = q.strip_suffix(';').unwrap_or(q).trim_end();
        let described =
            parse_rows(&self.run(format!("{q}\n\\gdesc\n"))?).map_err(PgError::Harness)?;
        let mut types = Vec::with_capacity(described.len());
        for row in described {
            match <[Option<String>; 2]>::try_from(row) {
                Ok([_name, ty]) => types.push(ty.unwrap_or_default()),
                Err(row) => {
                    return Err(PgError::Harness(format!(
                        "\\gdesc row has {} cells, expected 2",
                        row.len()
                    )))
                }
            }
        }
        let rows = parse_rows(&self.run(format!("{q};\n"))?).map_err(PgError::Harness)?;
        for row in &rows {
            if row.len() != types.len() {
                return Err(PgError::Harness(format!(
                    "row has {} cells, \\gdesc reported {} columns",
                    row.len(),
                    types.len()
                )));
            }
        }
        Ok(Rows { types, rows })
    }

    /// The one place psql runs; shared by `script`, `query` and `Drop`.
    fn run(&self, script: String) -> Result<Vec<u8>, PgError> {
        let mut options = std::env::var("PGOPTIONS").unwrap_or_default();
        options.push_str(&format!(
            " -c search_path={} -c client_min_messages=warning",
            self.schema
        ));
        let mut child = Command::new(&self.program)
            .args(["-X", "-q", "-At", "-F"])
            .arg(FIELD_SEP.to_string())
            .args(["-0", "-P"])
            .arg(format!("null={NULL_MARK}"))
            .args([
                "-v",
                "ON_ERROR_STOP=1",
                "-v",
                "VERBOSITY=sqlstate",
                "-f",
                "-",
            ])
            .env("LC_MESSAGES", "C")
            .env("PGCLIENTENCODING", "UTF8")
            .env("PGOPTIONS", options)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| PgError::Harness(format!("could not run psql ({e})")))?;
        let mut stdin = child.stdin.take().expect("piped stdin");
        // psql answers while it reads. Writing the whole script before reading
        // the output deadlocks once the stdout pipe fills, so the script goes
        // in on its own thread while this one drains the output.
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(script.as_bytes());
        });
        let out = child.wait_with_output();
        let _ = writer.join();
        let out = out.map_err(|e| PgError::Harness(format!("could not run psql ({e})")))?;
        if out.status.success() {
            return Ok(out.stdout);
        }
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        match parse_sqlstate(&stderr) {
            Some(sqlstate) => Err(PgError::Sql { sqlstate, stderr }),
            None => Err(PgError::Harness(format!("psql failed: {}", stderr.trim()))),
        }
    }
}

impl Drop for Pg {
    fn drop(&mut self) {
        let _ = self.run(format!("DROP SCHEMA IF EXISTS {} CASCADE;\n", self.schema));
    }
}

/// Parses psql output produced with the flags `run` uses. Pure.
pub fn parse_rows(stdout: &[u8]) -> Result<Vec<Row>, String> {
    if stdout.is_empty() {
        return Ok(vec![]);
    }
    if stdout[stdout.len() - 1] != 0 {
        return Err("unterminated record".to_string());
    }
    let mut records: Vec<&[u8]> = stdout.split(|b| *b == 0).collect();
    records.pop(); // the empty piece after the final terminator
    let mut rows = Vec::with_capacity(records.len());
    for rec in records {
        let rec = std::str::from_utf8(rec).map_err(|e| format!("record is not UTF-8 ({e})"))?;
        rows.push(
            rec.split(FIELD_SEP)
                .map(|f| {
                    if f == NULL_MARK {
                        None
                    } else {
                        Some(f.to_string())
                    }
                })
                .collect(),
        );
    }
    Ok(rows)
}

/// First SQLSTATE in psql's stderr under `VERBOSITY=sqlstate`. Pure.
pub fn parse_sqlstate(stderr: &str) -> Option<String> {
    for line in stderr.lines() {
        let found = ["ERROR:", "FATAL:", "PANIC:"]
            .iter()
            .filter_map(|tag| line.find(tag).map(|p| p + tag.len()))
            .min();
        let Some(start) = found else { continue };
        let rest = line[start..].trim_start_matches(' ');
        let run = rest
            .bytes()
            .take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
            .count();
        let after = rest[run..].chars().next();
        if run == 5 && matches!(after, None | Some(':')) {
            return Some(rest[..run].to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(cells: &[Option<&str>]) -> Row {
        cells.iter().map(|c| c.map(str::to_string)).collect()
    }

    #[test]
    fn parse_rows_keeps_null_distinct() {
        let input = "1\u{1f}\u{1}NULL\u{1}\u{0}2\u{1f}\u{0}3\u{1f}a|b\nc\0".as_bytes();
        assert_eq!(
            parse_rows(input).unwrap(),
            vec![
                r(&[Some("1"), None]),
                r(&[Some("2"), Some("")]),
                r(&[Some("3"), Some("a|b\nc")]),
            ]
        );
        assert_eq!(parse_rows(b"").unwrap(), Vec::<Row>::new());
        assert_eq!(parse_rows(b"\0").unwrap(), vec![r(&[Some("")])]);
    }

    #[test]
    fn parse_rows_rejects_an_unterminated_record() {
        assert!(parse_rows("1\u{1f}2".as_bytes()).is_err());
        assert_eq!(
            parse_rows("1\u{1f}2\0".as_bytes()).unwrap(),
            vec![r(&[Some("1"), Some("2")])]
        );
    }

    #[test]
    fn parse_sqlstate_reads_the_code() {
        assert_eq!(
            parse_sqlstate("psql:<stdin>:1: ERROR:  22012\n"),
            Some("22012".to_string())
        );
        assert_eq!(
            parse_sqlstate("NOTICE:  x\npsql:<stdin>:3: ERROR:  42P01\n"),
            Some("42P01".to_string())
        );
        assert_eq!(
            parse_sqlstate(
                "psql: error: connection to server on socket \"/x/.s.PGSQL.5\" failed: No such file or directory\n"
            ),
            None
        );
        assert_eq!(parse_sqlstate("ERROR:  division by zero\n"), None);
        assert_eq!(parse_sqlstate("FATAL:  57P01\n"), Some("57P01".to_string()));
        assert_eq!(parse_sqlstate("ERROR:  XX0001\n"), None);
    }
}
