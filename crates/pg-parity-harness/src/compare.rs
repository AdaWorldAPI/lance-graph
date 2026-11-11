//! Comparators over PostgreSQL's answer (`expected`) and an implementation's.

use crate::pg::PgError;
use crate::{Row, Rows};
use std::collections::BTreeMap;

/// A typed refusal: the implementation does not support a feature. `feature`
/// is a stable identifier (e.g. "window-function"), `detail` free text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub feature: &'static str,
    pub detail: String,
}

/// The implementation's answer to one query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    Rows(Rows),
    Error { sqlstate: String },
    Unsupported(Refusal),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order<'a> {
    /// No ORDER BY: results are multisets.
    Unordered,
    /// ORDER BY over these OUTPUT column indices, in key order.
    By(&'a [usize]),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Diff {
    Types {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    /// Rows whose multiplicity differs, with (expected, actual) counts.
    /// At most the first 5 in row order are kept; `total` is the full number.
    Multiset {
        rows: Vec<(Row, usize, usize)>,
        total: usize,
    },
    /// First position where the sequences differ beyond an allowed tie
    /// reordering (`None` = that side ended).
    Ordered {
        index: usize,
        expected: Option<Row>,
        actual: Option<Row>,
    },
    Sqlstate {
        expected: String,
        actual: String,
    },
    /// One side returned rows and the other an error. Values: "rows" / "error".
    Kind {
        expected: &'static str,
        actual: &'static str,
    },
    /// A mutation test whose mutation did not change PostgreSQL's answer.
    VacuousMutation,
    Harness(String),
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Diff::Types { expected, actual } => {
                write!(
                    f,
                    "column types: PostgreSQL {expected:?}, actual {actual:?}"
                )
            }
            Diff::Multiset { rows, total } => {
                write!(f, "{total} row(s) differ in multiplicity; first:")?;
                for (row, e, a) in rows {
                    write!(f, " {row:?} (PostgreSQL {e}, actual {a});")?;
                }
                Ok(())
            }
            Diff::Ordered {
                index,
                expected,
                actual,
            } => write!(
                f,
                "order differs at position {index}: PostgreSQL {expected:?}, actual {actual:?}"
            ),
            Diff::Sqlstate { expected, actual } => {
                write!(f, "SQLSTATE: PostgreSQL {expected}, actual {actual}")
            }
            Diff::Kind { expected, actual } => {
                write!(
                    f,
                    "PostgreSQL returned {expected}, actual returned {actual}"
                )
            }
            Diff::VacuousMutation => {
                write!(f, "the mutation did not change PostgreSQL's answer")
            }
            Diff::Harness(m) => write!(f, "harness: {m}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Pass,
    Fail(Diff),
    Unsupported(Refusal),
    Skip(String),
}

/// `None` when both sides hold the same rows with the same multiplicities,
/// in any order.
pub fn multiset_diff(expected: &[Row], actual: &[Row]) -> Option<Diff> {
    let mut counts: BTreeMap<&Row, (usize, usize)> = BTreeMap::new();
    for row in expected {
        let e = counts.entry(row).or_insert((0, 0));
        e.0 += 1;
    }
    for row in actual {
        let e = counts.entry(row).or_insert((0, 0));
        e.1 += 1;
    }
    let differing: Vec<(Row, usize, usize)> = counts
        .into_iter()
        .filter(|(_, c)| c.0 != c.1)
        .map(|(row, c)| (row.clone(), c.0, c.1))
        .collect();
    if differing.is_empty() {
        return None;
    }
    let total = differing.len();
    Some(Diff::Multiset {
        rows: differing.into_iter().take(5).collect(),
        total,
    })
}

/// Order check for an ORDER BY over output columns `keys`: rows whose key
/// cells are equal form a tie, and a tie may appear in any internal order.
///
/// Keys compare as text, so two key values PostgreSQL treats as equal but
/// prints differently are treated as distinct; that can only make the check
/// stricter, never looser. An ORDER BY expression that is not an output
/// column must be projected by the caller. A key index past a row's end is a
/// caller bug and panics.
pub fn ordered_diff(expected: &[Row], actual: &[Row], keys: &[usize]) -> Option<Diff> {
    let same_key = |a: &Row, b: &Row| keys.iter().all(|&k| a[k] == b[k]);
    let mut i = 0;
    while i < expected.len() {
        let mut j = i + 1;
        while j < expected.len() && same_key(&expected[i], &expected[j]) {
            j += 1;
        }
        if actual.len() < j {
            let index = actual.len().max(i);
            return Some(Diff::Ordered {
                index,
                expected: expected.get(index).cloned(),
                actual: None,
            });
        }
        let mut e = expected[i..j].to_vec();
        let mut a = actual[i..j].to_vec();
        e.sort();
        a.sort();
        if e != a {
            let k = (i..j).find(|&k| actual[k] != expected[k]).unwrap_or(i);
            return Some(Diff::Ordered {
                index: k,
                expected: Some(expected[k].clone()),
                actual: Some(actual[k].clone()),
            });
        }
        i = j;
    }
    if actual.len() > expected.len() {
        return Some(Diff::Ordered {
            index: expected.len(),
            expected: None,
            actual: Some(actual[expected.len()].clone()),
        });
    }
    None
}

/// `None` when equal, else `Diff::Types`.
pub fn types_diff(expected: &[String], actual: &[String]) -> Option<Diff> {
    if expected == actual {
        None
    } else {
        Some(Diff::Types {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        })
    }
}

/// Compares PostgreSQL's result with the implementation's answer.
pub fn compare(pg: &Result<Rows, PgError>, ours: &Answer, order: Order<'_>) -> Outcome {
    if let Answer::Unsupported(r) = ours {
        return Outcome::Unsupported(r.clone());
    }
    match (pg, ours) {
        (Err(PgError::Harness(m)), _) => Outcome::Fail(Diff::Harness(m.clone())),
        (Err(PgError::Sql { sqlstate: e, .. }), Answer::Error { sqlstate: a }) => {
            if e == a {
                Outcome::Pass
            } else {
                Outcome::Fail(Diff::Sqlstate {
                    expected: e.clone(),
                    actual: a.clone(),
                })
            }
        }
        (Err(PgError::Sql { .. }), Answer::Rows(_)) => Outcome::Fail(Diff::Kind {
            expected: "error",
            actual: "rows",
        }),
        (Ok(_), Answer::Error { .. }) => Outcome::Fail(Diff::Kind {
            expected: "rows",
            actual: "error",
        }),
        (Ok(e), Answer::Rows(a)) => {
            if let Some(d) = types_diff(&e.types, &a.types) {
                return Outcome::Fail(d);
            }
            let rows = match order {
                Order::Unordered => multiset_diff(&e.rows, &a.rows),
                Order::By(keys) => ordered_diff(&e.rows, &a.rows, keys),
            };
            match rows {
                Some(d) => Outcome::Fail(d),
                None => Outcome::Pass,
            }
        }
        (_, Answer::Unsupported(r)) => Outcome::Unsupported(r.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(cells: &[Option<&str>]) -> Row {
        cells.iter().map(|c| c.map(str::to_string)).collect()
    }

    fn row2(a: &str, b: &str) -> Row {
        r(&[Some(a), Some(b)])
    }

    #[test]
    fn multiset_counts_duplicates() {
        let a = row2("1", "a");
        let b = row2("2", "b");
        let expected = [a.clone(), a.clone(), b.clone()];
        let actual = [a.clone(), b.clone(), b.clone()];
        match multiset_diff(&expected, &actual) {
            Some(Diff::Multiset { rows, total }) => {
                assert_eq!(total, 2);
                assert_eq!(rows, vec![(a, 2, 1), (b, 1, 2)]);
            }
            other => panic!("expected a multiset diff, got {other:?}"),
        }
    }

    #[test]
    fn multiset_ignores_order() {
        let a = row2("1", "a");
        let b = row2("2", "b");
        let c = row2("3", "c");
        assert_eq!(
            multiset_diff(
                &[a.clone(), a.clone(), b.clone(), c.clone()],
                &[c.clone(), a.clone(), b.clone(), a.clone()]
            ),
            None
        );
        assert_eq!(
            multiset_diff(
                &[a.clone(), a.clone(), b.clone()],
                &[a.clone(), a.clone(), b.clone()]
            ),
            None
        );
    }

    #[test]
    fn ordered_allows_reorder_inside_a_tie() {
        let expected = [row2("1", "x"), row2("1", "y"), row2("2", "z")];
        let actual = [row2("1", "y"), row2("1", "x"), row2("2", "z")];
        assert_eq!(ordered_diff(&expected, &actual, &[0]), None);
    }

    #[test]
    fn ordered_rejects_reorder_across_keys() {
        let expected = [row2("1", "x"), row2("2", "y"), row2("2", "z")];
        let swapped = [row2("2", "y"), row2("1", "x"), row2("2", "z")];
        assert!(matches!(
            ordered_diff(&expected, &swapped, &[0]),
            Some(Diff::Ordered { index: 0, .. })
        ));
        let short = [row2("1", "x"), row2("2", "y")];
        assert!(matches!(
            ordered_diff(&expected, &short, &[0]),
            Some(Diff::Ordered { actual: None, .. })
        ));
        let long = [
            row2("1", "x"),
            row2("2", "y"),
            row2("2", "z"),
            row2("3", "w"),
        ];
        assert!(matches!(
            ordered_diff(&expected, &long, &[0]),
            Some(Diff::Ordered {
                index: 3,
                expected: None,
                actual: Some(_)
            })
        ));
    }

    #[test]
    fn null_is_not_the_empty_string() {
        let null = [r(&[None])];
        let empty = [r(&[Some("")])];
        assert!(multiset_diff(&null, &empty).is_some());
        assert!(ordered_diff(&null, &empty, &[0]).is_some());
    }

    fn rows(types: &[&str], rows: Vec<Row>) -> Rows {
        Rows {
            types: types.iter().map(|t| t.to_string()).collect(),
            rows,
        }
    }

    #[test]
    fn compare_orders_its_checks() {
        let data = vec![row2("1", "a"), row2("2", "b")];
        let pg_ok = Ok(rows(&["integer", "text"], data.clone()));

        // (a) types differ, rows equal
        let ours = Answer::Rows(rows(&["bigint", "text"], data.clone()));
        assert!(matches!(
            compare(&pg_ok, &ours, Order::Unordered),
            Outcome::Fail(Diff::Types { .. })
        ));

        // (b) error codes
        let pg_err = Err(PgError::Sql {
            sqlstate: "22012".into(),
            stderr: String::new(),
        });
        let same = Answer::Error {
            sqlstate: "22012".into(),
        };
        let other = Answer::Error {
            sqlstate: "22003".into(),
        };
        assert_eq!(compare(&pg_err, &same, Order::Unordered), Outcome::Pass);
        assert!(matches!(
            compare(&pg_err, &other, Order::Unordered),
            Outcome::Fail(Diff::Sqlstate { .. })
        ));

        // (c) kind mismatch both ways
        let ours_rows = Answer::Rows(rows(&["integer", "text"], data.clone()));
        assert_eq!(
            compare(&pg_err, &ours_rows, Order::Unordered),
            Outcome::Fail(Diff::Kind {
                expected: "error",
                actual: "rows"
            })
        );
        assert_eq!(
            compare(&pg_ok, &same, Order::Unordered),
            Outcome::Fail(Diff::Kind {
                expected: "rows",
                actual: "error"
            })
        );

        // (d) a refusal is recorded, even when PostgreSQL also failed
        let refusal = Refusal {
            feature: "window-function",
            detail: "not yet".into(),
        };
        let unsupported = Answer::Unsupported(refusal);
        for pg in [&pg_ok, &pg_err] {
            match compare(pg, &unsupported, Order::Unordered) {
                Outcome::Unsupported(r) => assert_eq!(r.feature, "window-function"),
                other => panic!("expected Unsupported, got {other:?}"),
            }
        }

        // (e) a harness failure
        let pg_harness = Err(PgError::Harness("boom".into()));
        assert_eq!(
            compare(&pg_harness, &ours_rows, Order::Unordered),
            Outcome::Fail(Diff::Harness("boom".into()))
        );

        // (f) equal rows
        assert_eq!(compare(&pg_ok, &ours_rows, Order::Unordered), Outcome::Pass);
    }
}
