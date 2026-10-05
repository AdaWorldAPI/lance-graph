//! The Quack frontend over report's own boundary: `Catalog` names fields,
//! `CamLabels` turns a categorical label into its ordinal. The bound query
//! keeps the ORDINAL, so a presentation rename leaves it valid — report's
//! identity contract survives the generic frontend.

use lance_graph_contract::content_store::ContentId;
use lance_graph_quack::bind::{table, BindError, Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{Col, Mask, Query};
use lance_graph_report::boundary::{BoundaryCounters, CamLabels, Catalog, MemKv};
use lance_graph_report::FieldId;

/// Report's catalog + CAM as a binder. The `Col` is the `FieldId`; a
/// production adapter takes the lane from the batch instead.
struct ReportBinder<'a> {
    cat: &'a Catalog,
    cam: &'a CamLabels,
    categorical: &'a [FieldId],
}

impl Binder for ReportBinder<'_> {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == "sales").then_some(TableId(0))
    }
    fn live(&self, _: TableId) -> Mask {
        Mask(0)
    }
    fn field(&self, _: TableId, name: &str) -> Option<BoundField> {
        let id = self.cat.field(name)?;
        let kind = if self.categorical.contains(&id) {
            FieldKind::Code
        } else {
            FieldKind::I32
        };
        Some(BoundField {
            col: Col(id.0 as u16),
            kind,
        })
    }
    fn code(&self, _: TableId, col: Col, literal: &str) -> Option<u32> {
        self.cam.ordinal(FieldId(u32::from(col.0)), literal)
    }
}

#[test]
fn a_presentation_rename_keeps_the_bound_ordinal() {
    let region = FieldId(1);
    let cat = Catalog::default()
        .with("region", region)
        .with("revenue", FieldId(2));
    let (mut cam, mut kv) = (CamLabels::default(), MemKv::default());
    cam.canonicalize(region, ["North", "South", "Coast"], &mut kv);
    let lookups = |c: &CamLabels| BoundaryCounters::get(&c.counters.cam_lookups);

    let before = lookups(&cam);
    let bound: Query = {
        let b = ReportBinder {
            cat: &cat,
            cam: &cam,
            categorical: &[region],
        };
        table("sales")
            .where_eq("region", "North")
            .where_eq("revenue", 100)
            .count()
            .bind(&b)
            .unwrap()
    };
    assert_eq!(lookups(&cam), before + 1, "one CAM lookup, at bind");

    // Rename the category's label: the ordinal is the identity.
    assert!(cam.rename(region, 0, "Nordland", &mut kv));
    let b = ReportBinder {
        cat: &cat,
        cam: &cam,
        categorical: &[region],
    };
    let renamed = table("sales")
        .where_eq("region", "Nordland")
        .where_eq("revenue", 100)
        .count()
        .bind(&b)
        .unwrap();
    assert_eq!(renamed, bound, "same category, same numeric query");
    assert_eq!(
        table("sales").where_eq("region", "North").bind(&b),
        Err(BindError::UnknownValue {
            field: "region".into(),
            value: "North".into()
        }),
        "the old label no longer names it"
    );
    // The label lives in KV under its content address, never in the query.
    assert_eq!(kv.text(ContentId::of_str("Nordland")), Some("Nordland"));
}
