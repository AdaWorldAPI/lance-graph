mod common;
use common::*;
use lance_graph_mask_risc::LaneRef;
use lance_graph_sap::{bind::*, schema::CatsSchema};

#[test]
fn columns_borrow_without_copy_and_preserve_null_zero_empty_and_time() {
    let mut input = fixture(65);
    input[6][0] = None;
    input[6][1] = Some("");
    input[21][0] = Some("00000000");
    input[9][0] = Some("2026-09-01T23:59:59Z");
    input[10][1] = Some("0.125");
    let batch = bind(&input);
    assert_eq!(batch.scale(), 3);
    let first = batch.lanes();
    let second = batch.lanes();
    let (LaneRef::I32(a), LaneRef::I32(b)) = (first[HOURS], second[HOURS]) else {
        panic!()
    };
    assert_eq!(a.as_ptr(), b.as_ptr());
    assert_eq!(a[0], 8500);
    assert_eq!(a[1], 125);
    assert_eq!(batch.edge_value(6, 0).unwrap(), None);
    assert_eq!(batch.edge_value(6, 1).unwrap().as_deref(), Some(""));
    assert_eq!(
        batch.edge_value(21, 0).unwrap().as_deref(),
        Some("00000000")
    );
    assert_eq!(batch.edge_value(21, 1).unwrap(), None);
    assert_eq!(
        batch.edge_value(9, 0).unwrap().as_deref(),
        Some("2026-09-01T23:59:59Z")
    );
    assert_eq!(batch.edge_value(10, 1).unwrap().as_deref(), Some("0.125"));
}

#[test]
fn abap_and_csharp_boolean_edges_bind_to_identical_lanes() {
    let mut cs = fixture(2);
    cs[17][1] = Some("false");
    let mut abap = cs.clone();
    abap[17] = vec![Some("X"), Some(" ")];
    let (cs, abap) = (bind(&cs), bind(&abap));
    for field in 0..23 {
        for row in 0..2 {
            assert_eq!(cs.edge_value(field, row), abap.edge_value(field, row));
        }
    }
}

#[test]
fn invalid_inputs_do_not_enter_the_abi() {
    for (field, value) in [
        (5, "42"),
        (6, "12345678901"),
        (10, "0.012345678901"),
        (9, "2025-02-29T00:00:00Z"),
        (11, ""),
    ] {
        let mut input = fixture(1);
        input[field][0] = Some(value);
        assert!(
            CatsBatch::bind(
                CatsSchema::new(42, 0),
                std::array::from_fn(|i| input[i].as_slice())
            )
            .is_err(),
            "{field} {value}"
        );
    }
    let mut input = fixture(1);
    input[3].clear();
    assert!(CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice())
    )
    .is_err());
    assert!(bind(&fixture(0)).is_empty());
}

#[test]
fn unique_text_codes_preserve_first_occurrence_and_edge_values() {
    let labels: Vec<_> = (0..4097).map(|i| format!("note-{i}")).collect();
    let mut input = common::fixture(labels.len() + 2);
    input[20] = labels.iter().map(|s| Some(s.as_str())).collect();
    input[20].extend([Some(labels[0].as_str()), None]);
    let batch = common::bind(&input);
    for (i, label) in labels.iter().enumerate() {
        assert_eq!(
            batch.edge_value(20, i).unwrap().as_deref(),
            Some(label.as_str())
        );
    }
    assert_eq!(
        batch.edge_value(20, labels.len()).unwrap().as_deref(),
        Some("note-0")
    );
    assert_eq!(batch.edge_value(20, labels.len() + 1).unwrap(), None);
}
