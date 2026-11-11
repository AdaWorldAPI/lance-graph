//! The one live-server test. Without a server it prints `SKIP` and passes.

use pg_parity_harness::{
    check, check_mutation, multiset_diff, text, Answer, Case, Diff, Order, Outcome, Pg, PgError,
    Refusal, Report, Row, Rows, Skip,
};

fn row(k: Option<&str>, v: Option<&str>) -> Row {
    vec![k.map(str::to_string), v.map(str::to_string)]
}

fn int_text(rows: Vec<Row>) -> Answer {
    Answer::Rows(Rows {
        types: vec!["integer".into(), "text".into()],
        rows,
    })
}

fn sqlstate_of(pg: &Pg, sql: &str) -> String {
    match pg.query(sql) {
        Err(PgError::Sql { sqlstate, .. }) => sqlstate,
        other => panic!("{sql}: expected an SQL error, got {other:?}"),
    }
}

#[test]
fn live_server_roundtrip() {
    let pg = match Pg::connect() {
        Ok(pg) => pg,
        Err(Skip(why)) => {
            println!("SKIP: {why}");
            return;
        }
    };
    let version = pg.version().to_string();
    let mut report = Report::new();

    // 1. setup prints nothing
    let setup = pg
        .script(
            "CREATE TABLE t (k int, v text); \
             INSERT INTO t VALUES (1,'a'),(1,'a'),(2,NULL),(2,''),(3,E'x|y\\nz');",
        )
        .expect("setup");
    assert_eq!(setup.len(), 0);

    // 2. NULL, empty string and delimiter characters survive the wire
    let all = pg.query("SELECT k, v FROM t").expect("select");
    assert_eq!(all.types, ["integer", "text"]);
    assert_eq!(all.rows.len(), 5);
    let expected = vec![
        row(Some("1"), Some("a")),
        row(Some("1"), Some("a")),
        row(Some("2"), None),
        row(Some("2"), Some("")),
        row(Some("3"), Some("x|y\nz")),
    ];
    assert_eq!(multiset_diff(&expected, &all.rows), None);
    assert!(all.rows.contains(&row(Some("2"), None)));
    assert!(all.rows.contains(&row(Some("2"), Some(""))));
    assert!(all.rows.contains(&row(Some("3"), Some("x|y\nz"))));

    // 3. an empty result keeps its column types; trailing `;` is trimmed
    let none = pg.query("SELECT k, v FROM t WHERE k > 99;").expect("empty");
    assert_eq!(none.rows.len(), 0);
    assert_eq!(none.types.len(), 2);

    // 4. aggregates over no rows: count 0, sum NULL
    let agg = pg
        .query("SELECT count(*), sum(k) FROM t WHERE false")
        .expect("agg");
    assert_eq!(agg.types, ["bigint", "bigint"]);
    assert_eq!(agg.rows, vec![vec![text(0), None]]);

    // 5. SQLSTATE
    assert_eq!(sqlstate_of(&pg, "SELECT 1/0"), "22012");
    assert_eq!(sqlstate_of(&pg, "SELEC 1"), "42601");
    assert_eq!(sqlstate_of(&pg, "SELECT * FROM no_such_table"), "42P01");

    // 6. ordered comparison: a tie may reorder, a key may not
    let ordered_sql = "SELECT k, v FROM t ORDER BY k";
    let pg_rows = pg.query(ordered_sql).expect("ordered").rows;
    let case = Case {
        name: "ordered tie",
        sql: ordered_sql,
        order: Order::By(&[0]),
    };
    let mut tie_flipped = pg_rows.clone();
    tie_flipped.swap(0, 1);
    tie_flipped.swap(2, 3);
    assert_ne!(tie_flipped, pg_rows, "the NULL/empty tie must be reordered");
    let out = check(&pg, &mut report, &case, || int_text(tie_flipped));
    assert_eq!(out, Outcome::Pass);
    let mut key_flipped = pg_rows.clone();
    key_flipped.rotate_right(1); // the k = 3 row first
    let case = Case {
        name: "ordered key",
        sql: ordered_sql,
        order: Order::By(&[0]),
    };
    let out = check(&pg, &mut report, &case, || int_text(key_flipped));
    assert!(
        matches!(out, Outcome::Fail(Diff::Ordered { .. })),
        "{out:?}"
    );

    // 7. a refusal is recorded, not failed
    let case = Case {
        name: "refused",
        sql: "SELECT k, v FROM t",
        order: Order::Unordered,
    };
    let out = check(&pg, &mut report, &case, || {
        Answer::Unsupported(Refusal {
            feature: "demo",
            detail: "not implemented".into(),
        })
    });
    assert!(matches!(out, Outcome::Unsupported(_)));
    assert_eq!(report.unsupported().len(), 1);

    // 8. mutation test against a toy implementation holding the table
    let mut state: Vec<Row> = pg.query("SELECT k, v FROM t").expect("toy").rows;
    let answer = |s: &Vec<Row>| int_text(s.clone());
    let case = Case {
        name: "mutation",
        sql: "SELECT k, v FROM t",
        order: Order::Unordered,
    };
    // (a) the same change on both sides
    let outs = check_mutation(
        &pg,
        &mut report,
        &case,
        "UPDATE t SET v = 'b' WHERE k = 3",
        &mut state,
        answer,
        |s| {
            for r in s.iter_mut().filter(|r| r[0].as_deref() == Some("3")) {
                r[1] = text("b");
            }
        },
    );
    assert_eq!(outs, (Outcome::Pass, Outcome::Pass));
    // (b) a key change the toy side never sees: a stale preparation
    let outs = check_mutation(
        &pg,
        &mut report,
        &case,
        "UPDATE t SET k = 7 WHERE k = 3",
        &mut state,
        answer,
        |_| {},
    );
    assert_eq!(outs.0, Outcome::Pass);
    assert!(
        matches!(outs.1, Outcome::Fail(Diff::Multiset { .. })),
        "{:?}",
        outs.1
    );
    for r in state.iter_mut().filter(|r| r[0].as_deref() == Some("3")) {
        r[0] = text(7); // re-sync the toy state
    }
    // (c) a mutation PostgreSQL cannot see
    let outs = check_mutation(
        &pg,
        &mut report,
        &case,
        "UPDATE t SET v = v WHERE false",
        &mut state,
        answer,
        |_| {},
    );
    assert_eq!(outs.0, Outcome::Pass);
    assert_eq!(outs.1, Outcome::Fail(Diff::VacuousMutation));

    // 9. the scratch schema is dropped with the session object
    let schema = pg.schema().to_string();
    drop(pg);
    let again = Pg::connect().expect("second connect");
    let left = again
        .query(&format!(
            "SELECT count(*) FROM pg_namespace WHERE nspname = '{schema}'"
        ))
        .expect("namespace count");
    assert_eq!(left.rows, vec![vec![text(0)]]);

    // 10.
    println!("{}", report.render());
    println!("PostgreSQL {version}: live_server_roundtrip ok");
}
