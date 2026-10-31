//! The committed rows parse, and the parser rejects rows outside the
//! D-SAG-6 vocabulary.

use dt64_admission_probe::evidence::{parse_rows, Evidence, Relation, RowError, HEADER};
use dt64_admission_probe::fixture::{PATH_JOINED_ROWS, SPAN_JOINED_ROWS};

fn count(rows: &[dt64_admission_probe::evidence::Row], rel: Relation) -> usize {
    rows.iter().filter(|r| r.relation == rel).count()
}

#[test]
fn committed_rows_parse_with_exact_counts() {
    let rows = parse_rows(SPAN_JOINED_ROWS).unwrap();
    assert_eq!(rows.len(), 29);
    assert_eq!(count(&rows, Relation::Defines), 7);
    assert_eq!(count(&rows, Relation::Calls), 17);
    assert_eq!(count(&rows, Relation::MayDispatch), 2);
    assert_eq!(count(&rows, Relation::MayPanic), 1);
    assert_eq!(count(&rows, Relation::WritesCallerMemory), 1);
    assert_eq!(count(&rows, Relation::WritesThroughLocalRef), 1);
    assert_eq!(count(&rows, Relation::Unjoined), 0);
    let path = parse_rows(PATH_JOINED_ROWS).unwrap();
    assert_eq!(path.len(), 2);
    assert_eq!(count(&path, Relation::Unjoined), 1);
    assert_eq!(path[1].evidence, Evidence::Unknown);
}

fn one(line: &str) -> Result<Vec<dt64_admission_probe::evidence::Row>, RowError> {
    parse_rows(&format!("{HEADER}\n{line}\n"))
}

#[test]
fn a_possible_relation_labelled_witnessed_is_rejected() {
    assert!(matches!(
        one("MAY_DISPATCH\tx::f\tdynamic\tcharon\tWITNESSED"),
        Err(RowError::EvidenceMismatch {
            relation: Relation::MayDispatch,
            evidence: Evidence::Witnessed,
            ..
        })
    ));
    assert!(matches!(
        one("UNJOINED\tx::f\twhy\tjoin\tWITNESSED"),
        Err(RowError::EvidenceMismatch { .. })
    ));
    // Silence: the same relations at their own class parse.
    assert!(one("MAY_DISPATCH\tx::f\tdynamic\tcharon\tPOSSIBLE").is_ok());
    assert!(one("UNJOINED\tx::f\twhy\tjoin\tUNKNOWN").is_ok());
}

#[test]
fn unknown_vocabulary_and_bad_shape_are_rejected() {
    assert!(matches!(
        one("MAY_EXPLODE\tx::f\t-\tcharon\tWITNESSED"),
        Err(RowError::UnknownRelation { .. })
    ));
    assert!(matches!(
        one("CALLS\tx::f\tx::g\tcharon\tPROBABLY"),
        Err(RowError::UnknownEvidence { .. })
    ));
    assert!(matches!(
        one("CALLS\tx::f\tx::g\tcharon"),
        Err(RowError::Columns { got: 4, .. })
    ));
    assert!(matches!(
        parse_rows("subject\trelation\nCALLS\ta\tb\tc\tWITNESSED"),
        Err(RowError::BadHeader(_))
    ));
}
