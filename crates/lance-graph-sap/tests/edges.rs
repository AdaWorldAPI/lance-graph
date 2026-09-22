mod common;
use common::*;
use lance_graph_sap::{edge::*, query::CatsQuery};

#[test]
fn same_carrier_emits_totals_and_only_selected_original_bapi_assignments() {
    let mut input = fixture(3);
    input[20] = vec![Some("123456789012345678901234567890123456789012345678901234567890"); 3];
    input[5][1] = Some("00000007");
    input[7][2] = Some("WBS-2");
    let batch = bind(&input);
    let mut query = CatsQuery::prepare(&batch, "00000042", "2026-09-01", "2026-09-30").unwrap();
    let mut sums = vec![0; query.groups()];
    let kept = query.execute_into(&mut sums).unwrap();
    let totals = activity_totals(&batch, &sums).unwrap();
    assert_eq!(
        totals,
        vec![ActivityTotal {
            activity_type: "DEV".into(),
            hours: "17.0".into()
        }]
    );
    let posted = bapi_sink(&batch, kept).unwrap();
    assert_eq!(posted.len(), 2);
    assert_eq!(posted[0].hours, "8.5");
    assert_eq!(posted[0].workdate, "20260901");
    assert_eq!(posted[0].employeenumber, "00000042");
    assert_eq!(posted[0].wbs_element, "WBS-1");
    assert_eq!(posted[1].wbs_element, "WBS-2");
    assert_eq!(posted[0].shorttext.len(), 50);
    assert_eq!(
        csharp_fields(&batch, 0).unwrap()[5].as_deref(),
        Some("00000042")
    );
    assert!(bapi_sink(&batch, &[u64::MAX]).is_err());
}

#[test]
fn current_hash_contracts_are_explicitly_incompatible() {
    let batch = bind(&fixture(1));
    let abap = ordered_hash_projection(
        &batch,
        0,
        HashProfile::SimafPortAbap {
            decimal_separator: '.',
        },
    )
    .unwrap();
    let smb = ordered_hash_projection(&batch, 0, HashProfile::SmbMiddleware).unwrap();
    assert!(abap.starts_with("EntryID=entry-1|SourceSystem=SAP|EntryType=BillableHours|"));
    assert!(smb.starts_with("ENTRY-1|SAP|BILLABLEHOURS|100|"));
    assert!(abap.contains("|HoursLogged=8.50|"));
    assert!(smb.contains("|8.50|DEV|"));
    // Executed against the original pinned TimeTrackingHasher.cs under .NET 8;
    // independently checked with Python hmac/SHA512 by verify_oracles.py.
    assert_eq!(hash_projection(&smb, b"fixture-key"),
        "261a980287713a250f5f2dcd6f7c304f255b1f5673b68604b660bb95f17c3d6570fd4f1e4072357b2e7c714a593478eea6dcc1adad5bc873ffd956923f9162a3");
    assert_ne!(
        hash_projection(&abap, b"fixture-key"),
        hash_projection(&smb, b"fixture-key")
    );
    let comma = ordered_hash_projection(
        &batch,
        0,
        HashProfile::SimafPortAbap {
            decimal_separator: ',',
        },
    )
    .unwrap();
    assert!(comma.contains("HoursLogged=8,50"));
}

#[test]
fn unsupported_oracle_cases_fail_instead_of_claiming_equivalence() {
    for (field, value) in [(10, "0.125"), (3, "tenant-A"), (3, "0100"), (20, "ä")] {
        let mut input = fixture(1);
        input[field][0] = Some(value);
        let batch = bind(&input);
        assert!(ordered_hash_projection(&batch, 0, HashProfile::SmbMiddleware).is_err());
    }
    let batch = bind(&fixture(1));
    assert!(bapi_sink(&batch, batch.alpha()).is_err());
}
