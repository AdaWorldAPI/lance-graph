//! Recording outcomes, the `check` driver and the mutation test.

use crate::compare::{compare, Answer, Diff, Order, Outcome, Refusal};
use crate::pg::{Pg, PgError};
use crate::Rows;

pub struct Case<'a> {
    pub name: &'a str,
    pub sql: &'a str,
    pub order: Order<'a>,
}

#[derive(Debug, Default)]
pub struct Report {
    entries: Vec<(String, Outcome)>,
}

impl Report {
    pub fn new() -> Report {
        Report::default()
    }

    pub fn record(&mut self, name: &str, outcome: Outcome) {
        self.entries.push((name.to_string(), outcome));
    }

    pub fn outcomes(&self) -> &[(String, Outcome)] {
        &self.entries
    }

    /// (pass, fail, unsupported, skip)
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let mut c = (0, 0, 0, 0);
        for (_, o) in &self.entries {
            match o {
                Outcome::Pass => c.0 += 1,
                Outcome::Fail(_) => c.1 += 1,
                Outcome::Unsupported(_) => c.2 += 1,
                Outcome::Skip(_) => c.3 += 1,
            }
        }
        c
    }

    /// Every recorded refusal, in order: (case name, refusal).
    pub fn unsupported(&self) -> Vec<(&str, &Refusal)> {
        self.entries
            .iter()
            .filter_map(|(n, o)| match o {
                Outcome::Unsupported(r) => Some((n.as_str(), r)),
                _ => None,
            })
            .collect()
    }

    /// One line: `pg-parity: N cases, P pass, F fail, U unsupported, S skip`.
    pub fn summary(&self) -> String {
        let (p, f, u, s) = self.counts();
        format!(
            "pg-parity: {} cases, {p} pass, {f} fail, {u} unsupported, {s} skip",
            self.entries.len()
        )
    }

    /// `summary()` first, then one line per Fail (`FAIL <name>: <diff>`) and
    /// one per Unsupported (`UNSUPPORTED <name>: <feature>: <detail>`).
    pub fn render(&self) -> String {
        let mut out = self.summary();
        for (name, o) in &self.entries {
            match o {
                Outcome::Fail(d) => out.push_str(&format!("\nFAIL {name}: {d}")),
                Outcome::Unsupported(r) => out.push_str(&format!(
                    "\nUNSUPPORTED {name}: {}: {}",
                    r.feature, r.detail
                )),
                _ => {}
            }
        }
        out
    }
}

/// Runs `case.sql` on PostgreSQL, compares with `ours()`, records and returns
/// the outcome.
pub fn check(
    pg: &Pg,
    report: &mut Report,
    case: &Case<'_>,
    ours: impl FnOnce() -> Answer,
) -> Outcome {
    let expected = pg.query(case.sql);
    let outcome = compare(&expected, &ours(), case.order);
    report.record(case.name, outcome.clone());
    outcome
}

/// Pure core of the mutation test. Returns (before, after).
pub fn mutation_outcomes(
    pg_before: &Result<Rows, PgError>,
    ours_before: &Answer,
    pg_after: &Result<Rows, PgError>,
    ours_after: &Answer,
    order: Order<'_>,
) -> (Outcome, Outcome) {
    let before = compare(pg_before, ours_before, order);
    // "Unchanged" uses the same semantics as the parity comparison: a row
    // reorder that the query leaves unspecified is not a change.
    let unchanged = match (pg_before, pg_after) {
        (Ok(b), Ok(a)) => compare(&Ok(b.clone()), &Answer::Rows(a.clone()), order) == Outcome::Pass,
        (b, a) => b == a,
    };
    let after = if unchanged {
        Outcome::Fail(Diff::VacuousMutation)
    } else {
        compare(pg_after, ours_after, order)
    };
    (before, after)
}

/// Mutation test: answer, mutate both sides, answer again.
/// `answer` reads the implementation state; `mutate` changes it. Records
/// `<name> [before]` and `<name> [after]`; returns (before, after).
///
/// A value change must show the new values through an unchanged mapping, and
/// a key change must be visible after the caller's `mutate` invalidates or
/// recomputes its preparation. A stale preparation shows up as `after = Fail`.
pub fn check_mutation<S>(
    pg: &Pg,
    report: &mut Report,
    case: &Case<'_>,
    pg_mutation: &str,
    state: &mut S,
    answer: impl Fn(&S) -> Answer,
    mutate: impl FnOnce(&mut S),
) -> (Outcome, Outcome) {
    let pg_before = pg.query(case.sql);
    let ours_before = answer(state);
    let mutation = pg.script(pg_mutation);
    mutate(state);
    let (before, after) = match mutation {
        Err(e) => (
            compare(&pg_before, &ours_before, case.order),
            Outcome::Fail(Diff::Harness(format!("mutation failed: {e}"))),
        ),
        Ok(_) => {
            let pg_after = pg.query(case.sql);
            let ours_after = answer(state);
            mutation_outcomes(&pg_before, &ours_before, &pg_after, &ours_after, case.order)
        }
    };
    report.record(&format!("{} [before]", case.name), before.clone());
    report.record(&format!("{} [after]", case.name), after.clone());
    (before, after)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Row;

    fn row2(a: &str, b: &str) -> Row {
        vec![Some(a.to_string()), Some(b.to_string())]
    }

    fn rows(data: Vec<Row>) -> Rows {
        Rows {
            types: vec!["integer".into(), "text".into()],
            rows: data,
        }
    }

    #[test]
    fn mutation_guard_fires_on_a_no_op() {
        let data = vec![row2("1", "a"), row2("2", "b")];
        let pg = Ok(rows(data));
        let ours = Answer::Rows(pg.clone().unwrap());
        let (before, after) = mutation_outcomes(&pg, &ours, &pg, &ours, Order::Unordered);
        assert_eq!(before, Outcome::Pass);
        assert_eq!(after, Outcome::Fail(Diff::VacuousMutation));
    }

    #[test]
    fn mutation_guard_fires_when_postgresql_only_reorders_rows() {
        let pg_before = Ok(rows(vec![row2("1", "a"), row2("2", "b")]));
        let pg_after = Ok(rows(vec![row2("2", "b"), row2("1", "a")]));
        let ours = Answer::Rows(pg_before.clone().unwrap());
        let (before, after) =
            mutation_outcomes(&pg_before, &ours, &pg_after, &ours, Order::Unordered);
        assert_eq!(before, Outcome::Pass);
        assert_eq!(after, Outcome::Fail(Diff::VacuousMutation));
    }

    #[test]
    fn mutation_guard_stays_silent_on_a_real_change() {
        let before_rows = rows(vec![row2("1", "a"), row2("2", "b")]);
        let after_rows = rows(vec![row2("1", "a"), row2("2", "c")]);
        let pg_before = Ok(before_rows.clone());
        let pg_after = Ok(after_rows.clone());
        let ours_before = Answer::Rows(before_rows.clone());
        let ours_after = Answer::Rows(after_rows);
        assert_eq!(
            mutation_outcomes(
                &pg_before,
                &ours_before,
                &pg_after,
                &ours_after,
                Order::Unordered
            ),
            (Outcome::Pass, Outcome::Pass)
        );
        // A stale preparation: the implementation still answers the old rows.
        let stale = Answer::Rows(before_rows);
        let (b, a) = mutation_outcomes(
            &pg_before,
            &ours_before,
            &pg_after,
            &stale,
            Order::Unordered,
        );
        assert_eq!(b, Outcome::Pass);
        assert!(matches!(a, Outcome::Fail(Diff::Multiset { .. })));
    }

    #[test]
    fn report_counts_every_outcome() {
        let mut report = Report::new();
        report.record("p1", Outcome::Pass);
        report.record("f1", Outcome::Fail(Diff::VacuousMutation));
        report.record(
            "u1",
            Outcome::Unsupported(Refusal {
                feature: "window-function",
                detail: "not yet".into(),
            }),
        );
        report.record("s1", Outcome::Skip("no server".into()));
        report.record("p2", Outcome::Pass);
        assert_eq!(report.counts(), (2, 1, 1, 1));
        assert_eq!(
            report.summary(),
            "pg-parity: 5 cases, 2 pass, 1 fail, 1 unsupported, 1 skip"
        );
        let un = report.unsupported();
        assert_eq!(un.len(), 1);
        assert_eq!(un[0].0, "u1");
        assert_eq!(un[0].1.feature, "window-function");
        let text = report.render();
        assert_eq!(text.lines().filter(|l| l.starts_with("FAIL ")).count(), 1);
        assert_eq!(
            text.lines()
                .filter(|l| l.starts_with("UNSUPPORTED "))
                .count(),
            1
        );
    }
}
