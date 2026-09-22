mod common;
use common::*;
use lance_graph_sap::schema::FIELDS;

#[test]
fn repeated_domains_share_one_adapter_and_char_values_remain_lossless() {
    // W5: only demonstrated reuse is generalized: the two PERNR fields,
    // four UTC fields, and width-constrained dictionary fields. No DDIC engine.
    assert_eq!(FIELDS[5].native_type, FIELDS[21].native_type);
    assert_eq!(FIELDS[5].carrier, FIELDS[21].carrier);
    assert_eq!(FIELDS[5].width, FIELDS[21].width);
    let mut input = fixture(3);
    input[5] = vec![Some("00000000"), Some("00000042"), Some("99999999")];
    input[21] = input[5].clone();
    input[6] = vec![Some("0000000123"), Some("123"), Some("123 ")];
    input[7][0] = Some("123456789012345678901234");
    input[20][0] = Some("naïve — Grüße"); // lossless text storage, independent of hash support
    let batch = bind(&input);
    for (row, customer) in input[6].iter().enumerate() {
        assert_eq!(batch.edge_value(5, row), batch.edge_value(21, row));
        assert_eq!(batch.edge_value(6, row).unwrap().as_deref(), *customer);
    }
    assert_eq!(batch.edge_value(7, 0).unwrap().as_deref(), input[7][0]);
    assert_eq!(batch.edge_value(20, 0).unwrap().as_deref(), input[20][0]);
    // No invented ALPHA conversion: distinct source values stay distinct.
    assert_ne!(batch.edge_value(6, 0), batch.edge_value(6, 1));
    assert_ne!(batch.edge_value(6, 1), batch.edge_value(6, 2));
}
