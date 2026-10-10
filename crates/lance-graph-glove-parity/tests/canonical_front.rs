//! D-XGP-1 probe Q1/Q2 (`.claude/plans/cross-glove-business-parity-v1.md` §C.4).
//!
//! The canonical attribute URIs below are FIXTURES (`ogit.GloveFixture:*`).
//! Who mints the real ones for `BILLABLE_WORK_ENTRY` is an open operator
//! ruling (plan §C.5); nothing here claims them.
use lance_graph_contract::property::{Marking, SemanticType};
use lance_graph_glove_parity::{native_field, CanonicalBinder};
use lance_graph_mask_risc::{execute_into, Foreign, Out, Planes, Scratch};
use lance_graph_ontology::{MappingProposal, MappingProposalKind, OgitUri, OntologyRegistry};
use lance_graph_quack::bind::{table, BindError, Binder, TableId};
use lance_graph_quack::{lower, Agg, Cmp, Filter, Query};
use lance_graph_sap::bind::CatsBatch;
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::query::CatsQuery;
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

const CONCEPT: &str = "billable_work_entry";
const WORKER: &str = "ogit.GloveFixture:worker";
const WORK_DAY: &str = "ogit.GloveFixture:workDay";
const QUANTITY: &str = "ogit.GloveFixture:quantity";
const ACTIVITY: &str = "ogit.GloveFixture:activity";
const COST: &str = "ogit.GloveFixture:costAmount";

fn attr(bridge: &str, native: &str, uri: &str, confidence: f32) -> MappingProposal {
    let ogit_uri = OgitUri::parse(uri).unwrap();
    MappingProposal {
        public_name: native.to_string(),
        bridge_id: bridge.to_string(),
        namespace: ogit_uri.namespace().unwrap().to_string(),
        ogit_uri,
        kind: MappingProposalKind::Attribute {
            predicate: uri.to_string(),
            semantic_type: SemanticType::PlainText,
        },
        marking: Marking::Internal,
        confidence,
        source_uri: format!("test://{bridge}/{native}"),
        // The checksum is the caller's content hash: it must cover the URI, or a
        // re-mapping of the same native field is (correctly) treated as idempotent.
        checksum: format!("checksum-{bridge}-{native}-{uri}"),
        created_by: "glove-parity-fixture".to_string(),
    }
}

/// SAP rows are the CATS DTO's own names; Odoo rows are the base
/// `account.analytic.line` names (`odoo_blueprint/extracted/analytic.rs:482`).
/// Odoo `user_id` → worker and `unit_amount` → quantity are HYPOTHESIZED
/// (confidence < 1.0): a login user is not an HR employee, and `unit_amount`
/// is hours only under a time UoM gate that does not exist yet (plan §C.6.2).
fn registry(sap_quantity_native: &str) -> OntologyRegistry {
    let reg = OntologyRegistry::new_in_memory();
    for p in [
        attr("sap", "employee_number", WORKER, 1.0),
        attr("sap", "work_day", WORK_DAY, 1.0),
        attr("sap", sap_quantity_native, QUANTITY, 1.0),
        attr("sap", "activity_type", ACTIVITY, 1.0),
        attr("odoo", "user_id", WORKER, 0.5),
        attr("odoo", "date", WORK_DAY, 1.0),
        attr("odoo", "unit_amount", QUANTITY, 0.5),
        attr("odoo", "amount", COST, 1.0),
    ] {
        reg.append_mapping(p).unwrap();
    }
    reg
}

#[allow(clippy::needless_range_loop)] // Same test-only multi-column fixture as sap's tests/fold.rs.
fn batch(n: usize) -> CatsBatch {
    let base: Vec<Option<&str>> = include_str!("../../lance-graph-sap/fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let mut input: [Vec<Option<&str>>; FIELD_COUNT] = std::array::from_fn(|i| vec![base[i]; n]);
    for i in 0..n {
        input[5][i] = Some(if i % 3 == 0 { "00000007" } else { "00000042" });
        input[9][i] = Some(if i % 5 == 0 {
            "2026-08-31T23:59:59Z"
        } else {
            "2026-09-01T12:34:56Z"
        });
        input[11][i] = Some(if i % 2 == 0 { "DEV" } else { "OPS" });
        input[10][i] = Some(if i % 2 == 0 { "8.50" } else { "0.125" });
    }
    CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap()
}

/// The fold the shipped `CatsQuery` runs, written once against whatever
/// binder resolves the names.
fn fold(
    b: &dyn Binder,
    t: &str,
    worker: &str,
    day: &str,
    qty: &str,
    act: &str,
) -> Result<Query, BindError> {
    let selected = table(t).where_eq(worker, "00000042").rows().bind(b)?;
    let tid = b.table(t).unwrap();
    let col = |name: &str| {
        b.field(tid, name)
            .map(|f| f.col)
            .ok_or_else(|| BindError::UnknownField {
                table: t.into(),
                field: name.into(),
            })
    };
    let d = col(day)?;
    Ok(Query {
        filter: Filter::and([
            selected.filter,
            Filter::cmp(d, Cmp::GeI32(20260901)),
            Filter::cmp(d, Cmp::LeI32(20260930)),
        ]),
        agg: Agg::GroupSumI32 {
            key: col(act)?,
            val: col(qty)?,
        },
    })
}

fn run(q: &Query, batch: &CatsBatch) -> Option<Vec<i64>> {
    let program = lower(q).ok()?;
    let n = batch.len();
    let live = live_words(n);
    let lanes = batch.lanes();
    let planes = Planes {
        n_rows: n,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    let mut scratch = Scratch::for_program(&program, n).ok()?;
    let mut sums = vec![0; batch.activity_groups() as usize];
    execute_into(
        &program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::I64(&mut sums),
    )
    .ok()?;
    Some(sums)
}

#[test]
fn q1_canonical_names_reach_the_same_query_program_and_sums_as_cats() {
    let reg = registry("hours_logged");
    for n in [2, 65, 4097] {
        let batch = batch(n);
        let cats = CatsBinder::new(&batch);
        let canonical = CanonicalBinder::new(&reg, "sap", CONCEPT, TABLE, &cats);

        let by_native = fold(
            &cats,
            TABLE,
            "employee_number",
            "work_day",
            "hours_logged",
            "activity_type",
        )
        .unwrap();
        let by_canonical = fold(&canonical, CONCEPT, WORKER, WORK_DAY, QUANTITY, ACTIVITY).unwrap();
        assert_eq!(by_canonical, by_native, "same bound Query, n={n}");
        assert_eq!(lower(&by_canonical).unwrap(), lower(&by_native).unwrap());

        let mut shipped =
            CatsQuery::prepare(&batch, "00000042", "2026-09-01", "2026-09-30").unwrap();
        let mut expected = vec![0; shipped.groups()];
        shipped.execute_into(&mut expected).unwrap();
        assert_eq!(run(&by_canonical, &batch).unwrap(), expected, "n={n}");
        assert!(expected.iter().any(|s| *s != 0), "anti-vacuity");
    }
}

#[test]
fn q2_one_uri_is_one_identity_across_gloves_and_one_sided_fields_stay_one_sided() {
    let reg = registry("hours_logged");
    let id = |uri: &str| reg.resolve_uri(uri).unwrap().entity_type_id();
    for uri in [WORKER, WORK_DAY, QUANTITY] {
        let mut bridges: Vec<_> = reg
            .rows_with_entity_type(id(uri))
            .into_iter()
            .map(|r| r.bridge_id)
            .collect();
        bridges.sort();
        assert_eq!(bridges, ["odoo", "sap"], "{uri}");
    }
    assert_ne!(
        id(WORK_DAY),
        id(QUANTITY),
        "distinct fields keep distinct ids"
    );
    assert_eq!(
        native_field(&reg, "sap", WORK_DAY).as_deref(),
        Some("work_day")
    );
    assert_eq!(
        native_field(&reg, "odoo", WORK_DAY).as_deref(),
        Some("date")
    );
    // Present on one side only: resolves there, None on the other, never defaulted.
    assert_eq!(
        native_field(&reg, "sap", ACTIVITY).as_deref(),
        Some("activity_type")
    );
    assert_eq!(native_field(&reg, "odoo", ACTIVITY), None);
    assert_eq!(native_field(&reg, "odoo", COST).as_deref(), Some("amount"));
    assert_eq!(native_field(&reg, "sap", COST), None);
}

#[test]
fn hypothesized_rows_exist_but_never_bind() {
    let reg = registry("hours_logged");
    // The rows are in the registry ...
    assert!(reg
        .rows_with_entity_type(reg.resolve_uri(WORKER).unwrap().entity_type_id())
        .iter()
        .any(|r| r.bridge_id == "odoo" && r.confidence < 1.0));
    // ... and the front refuses them.
    assert_eq!(native_field(&reg, "odoo", WORKER), None);
    assert_eq!(native_field(&reg, "odoo", QUANTITY), None);
    assert_eq!(
        native_field(&reg, "sap", WORKER).as_deref(),
        Some("employee_number")
    );
}

#[test]
fn n_a_a_wrong_mapping_changes_the_program_and_the_answer() {
    let reg = registry("employee_number"); // quantity → PERNR: wrong
    let batch = batch(65);
    let cats = CatsBinder::new(&batch);
    let canonical = CanonicalBinder::new(&reg, "sap", CONCEPT, TABLE, &cats);
    let right = fold(
        &cats,
        TABLE,
        "employee_number",
        "work_day",
        "hours_logged",
        "activity_type",
    )
    .unwrap();
    let wrong = fold(&canonical, CONCEPT, WORKER, WORK_DAY, QUANTITY, ACTIVITY).unwrap();
    assert_ne!(wrong, right);
    assert_ne!(run(&wrong, &batch), run(&right, &batch));
}

#[test]
fn n_c_canonical_names_without_a_row_are_unknown_fields_and_tables() {
    let reg = registry("hours_logged");
    let batch = batch(3);
    let cats = CatsBinder::new(&batch);
    let canonical = CanonicalBinder::new(&reg, "sap", CONCEPT, TABLE, &cats);
    let t = canonical.table(CONCEPT).unwrap();
    assert_eq!(t, TableId(0));
    assert_eq!(
        canonical.table(TABLE),
        None,
        "the native table name is not a canonical one"
    );
    assert!(
        canonical.field(t, COST).is_none(),
        "no sap row for costAmount"
    );
    assert!(
        canonical.field(t, "employee_number").is_none(),
        "a native name is not a canonical one"
    );
    assert!(matches!(
        table(CONCEPT).where_eq(COST, "1").bind(&canonical),
        Err(BindError::UnknownField { .. })
    ));
    // Literals still resolve through the glove's own domain.
    assert!(matches!(
        table(CONCEPT).where_eq(ACTIVITY, "QA").bind(&canonical),
        Err(BindError::UnknownValue { .. })
    ));
}

#[test]
fn only_attribute_rows_name_a_field() {
    // An entity row under the same bridge and URI names a class, not a field.
    let reg = OntologyRegistry::new_in_memory();
    let mut p = attr("sap", "activity_entity", ACTIVITY, 1.0);
    p.kind = MappingProposalKind::Entity {
        schema: lance_graph_contract::property::Schema::builder("Activity").build(),
    };
    reg.append_mapping(p).unwrap();
    assert_eq!(native_field(&reg, "sap", ACTIVITY), None);
    reg.append_mapping(attr("sap", "activity_type", ACTIVITY, 1.0))
        .unwrap();
    assert_eq!(
        native_field(&reg, "sap", ACTIVITY).as_deref(),
        Some("activity_type")
    );
}
