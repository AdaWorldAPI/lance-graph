//! Ontology-bound DataFusion `TableProvider`.
//!
//! [`OntologyTableProvider`] presents one entity_type as a queryable
//! table. The Arrow schema is derived from the [`Ontology`] via
//! [`OuterSchema`] + [`arrow_schema`]; today the read path is backed by
//! a [`MemTable`] (round-1 stub). Filter pushdown to the inner
//! ontology's SPO store is delegated to [`SpoFilterTranslator`] via
//! [`OntologyTableProvider::translate_filters`].
//!
//! Domain-agnostic. Pass any `(Ontology, entity_type)` and you get a
//! provider — Foundry-style.

use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::{MemTable, TableProvider, TableType};
use datafusion::error::Result as DfResult;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;

use lance_graph_contract::ontology::Ontology;

use super::spo_filter::{FilterTerm, Literal, Op, SpoFilterTranslator, SpoLookup};
use super::zerocopy::{arrow_schema, OuterSchema};

/// DataFusion table provider for one ontology-bound entity_type.
#[derive(Debug)]
pub struct OntologyTableProvider {
    ontology: Arc<Ontology>,
    soa: OuterSchema,
    inner: MemTable,
}

impl OntologyTableProvider {
    /// Build a provider for `entity_type`. Returns `None` if the entity
    /// type isn't declared in the ontology.
    pub fn new(ontology: Arc<Ontology>, entity_type: &str) -> Option<Self> {
        let soa = OuterSchema::from_ontology(&ontology, entity_type)?;
        let arrow_schema = arrow_schema(&soa);
        let inner = MemTable::try_new(arrow_schema, vec![vec![]]).ok()?;
        Some(Self {
            ontology,
            soa,
            inner,
        })
    }

    /// Build a provider with pre-populated batches.
    pub fn with_batches(
        ontology: Arc<Ontology>,
        entity_type: &str,
        batches: Vec<RecordBatch>,
    ) -> Option<Self> {
        let soa = OuterSchema::from_ontology(&ontology, entity_type)?;
        let arrow_schema = arrow_schema(&soa);
        let inner = MemTable::try_new(arrow_schema, vec![batches]).ok()?;
        Some(Self {
            ontology,
            soa,
            inner,
        })
    }

    pub fn entity_type(&self) -> &str {
        self.soa.entity_type
    }

    pub fn ontology(&self) -> &Ontology {
        &self.ontology
    }

    pub fn outer_schema(&self) -> &OuterSchema {
        &self.soa
    }

    /// Translate filter terms to an `SpoLookup`. Useful for direct
    /// callers and tests; the DataFusion path eventually goes through
    /// `scan`'s filter slice.
    pub fn translate_filters(&self, terms: &[FilterTerm]) -> SpoLookup {
        SpoFilterTranslator::new(&self.ontology).translate(terms)
    }
}

#[async_trait]
impl TableProvider for OntologyTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        // Round-1: delegate to MemTable. SpoStore-backed scan is the next
        // round (mirrors Phase 2 of the SQL↔SPO ontology bridge plan).
        self.inner.scan(state, projection, filters, limit).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        // Phase-2-A pushdown classification.
        //
        // For each filter expression we attempt to translate it into a
        // [`FilterTerm`] and pass it through [`SpoFilterTranslator`]. If
        // the translator extracts a recognised lookup field
        // (`entity_type_id`, `predicate_fp`, `entity_id`, `min_frequency`,
        // `min_confidence`), report `Inexact` — the table provider can
        // narrow the read but DataFusion still applies the filter as a
        // residual. Anything we don't recognise stays `Unsupported` so
        // the filter is evaluated by the planner unchanged.
        //
        // Phase-2-B (next PR) replaces the MemTable-backed `scan` with a
        // real `SpoStore` walk that consumes the same `SpoLookup`. Once
        // that's in, the classification here will become `Exact` for the
        // recognised filters.
        let translator = SpoFilterTranslator::new(&self.ontology);
        Ok(filters
            .iter()
            .map(|expr| {
                if let Some(term) = expr_to_filter_term(expr) {
                    let look = translator.translate(std::slice::from_ref(&term));
                    if lookup_is_recognised(&look) {
                        return TableProviderFilterPushDown::Inexact;
                    }
                }
                TableProviderFilterPushDown::Unsupported
            })
            .collect())
    }
}

/// Convert one DataFusion `Expr` into an [`SpoFilterTranslator`]-shaped
/// [`FilterTerm`], if the expression matches the simple `column op
/// literal` pattern. Anything more complex (function calls, arithmetic,
/// nested boolean trees) returns `None` and stays as a DataFusion
/// residual.
///
/// Pattern matched: `BinaryExpr { left: Column(name), op, right: Literal(_) }`
/// (and the symmetric `Literal op Column` shape).
fn expr_to_filter_term(expr: &Expr) -> Option<FilterTerm> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            // Try `column op literal` and `literal op column` (symmetric).
            if let (Expr::Column(c), Expr::Literal(s, _)) = (left.as_ref(), right.as_ref()) {
                let op = operator_to_op(*op)?;
                let lit = scalar_to_literal(s)?;
                return Some(FilterTerm {
                    column: c.name.clone(),
                    op,
                    literal: lit,
                });
            }
            if let (Expr::Literal(s, _), Expr::Column(c)) = (left.as_ref(), right.as_ref()) {
                // Flip the op: `5 < x` ≡ `x > 5`.
                let op = flipped_op(operator_to_op(*op)?);
                let lit = scalar_to_literal(s)?;
                return Some(FilterTerm {
                    column: c.name.clone(),
                    op,
                    literal: lit,
                });
            }
            None
        }
        _ => None,
    }
}

fn operator_to_op(op: Operator) -> Option<Op> {
    Some(match op {
        Operator::Eq => Op::Eq,
        Operator::NotEq => Op::NotEq,
        Operator::Gt => Op::Gt,
        Operator::GtEq => Op::Gte,
        Operator::Lt => Op::Lt,
        Operator::LtEq => Op::Lte,
        _ => return None,
    })
}

fn flipped_op(op: Op) -> Op {
    match op {
        Op::Eq => Op::Eq,
        Op::NotEq => Op::NotEq,
        Op::Gt => Op::Lt,
        Op::Gte => Op::Lte,
        Op::Lt => Op::Gt,
        Op::Lte => Op::Gte,
    }
}

fn scalar_to_literal(s: &ScalarValue) -> Option<Literal> {
    match s {
        ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => {
            Some(Literal::Utf8(v.clone()))
        }
        ScalarValue::UInt64(Some(v)) => Some(Literal::UInt64(*v)),
        ScalarValue::UInt32(Some(v)) => Some(Literal::UInt64(u64::from(*v))),
        ScalarValue::Int64(Some(v)) if *v >= 0 => Some(Literal::UInt64(*v as u64)),
        ScalarValue::Int32(Some(v)) if *v >= 0 => Some(Literal::UInt64(u64::from(*v as u32))),
        ScalarValue::Float32(Some(v)) => Some(Literal::Float32(*v)),
        ScalarValue::Float64(Some(v)) => Some(Literal::Float32(*v as f32)),
        _ => None,
    }
}

/// Return `true` if the lookup carries any recognised field — i.e. the
/// translator extracted at least one term that an SPO scan could narrow
/// on. Used by [`OntologyTableProvider::supports_filters_pushdown`] to
/// classify filters as Inexact (recognised) vs Unsupported.
fn lookup_is_recognised(look: &SpoLookup) -> bool {
    look.entity_type_id.is_some()
        || look.predicate_fp.is_some()
        || look.predicate_fp_excluded.is_some()
        || look.min_frequency.is_some()
        || look.min_confidence.is_some()
        || look.entity_id.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::property::Schema;

    fn medcare_ontology() -> Arc<Ontology> {
        Arc::new(
            Ontology::builder("Medcare")
                .schema(
                    Schema::builder("Patient")
                        .required("patient_id")
                        .required("name")
                        .required("geburtsdatum")
                        .optional("krankenkasse")
                        .build(),
                )
                .build(),
        )
    }

    #[test]
    fn provider_returns_arrow_schema_with_id_first() {
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        let s = p.schema();
        assert_eq!(s.field(0).name(), "id");
        assert_eq!(s.field(1).name(), "entity_type");
        assert!(s.fields().iter().any(|f| f.name() == "patient_id"));
        assert!(s.fields().iter().any(|f| f.name() == "geburtsdatum"));
    }

    #[test]
    fn provider_returns_none_for_unknown_entity_type() {
        let ont = medcare_ontology();
        assert!(OntologyTableProvider::new(ont, "Spaceship").is_none());
    }

    #[test]
    fn provider_table_type_is_base() {
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        assert_eq!(p.table_type(), TableType::Base);
    }

    #[test]
    fn provider_translate_filters_routes_known_columns() {
        use super::super::spo_filter::{Literal, Op};
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        let look = p.translate_filters(&[FilterTerm {
            column: "entity_type".into(),
            op: Op::Eq,
            literal: Literal::Utf8("Patient".into()),
        }]);
        assert_eq!(look.entity_type_id, Some(1));
    }

    // ── Phase-2-A pushdown classification ────────────────────────────────────

    fn col_eq_str(col: &str, val: &str) -> Expr {
        use datafusion::logical_expr::{lit, Expr};
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column(col.into())),
            op: Operator::Eq,
            right: Box::new(lit(val)),
        })
    }

    fn col_gt_f32(col: &str, val: f32) -> Expr {
        use datafusion::logical_expr::{lit, Expr};
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column(col.into())),
            op: Operator::Gt,
            right: Box::new(lit(val)),
        })
    }

    #[test]
    fn pushdown_inexact_for_entity_type_eq() {
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        let f = col_eq_str("entity_type", "Patient");
        let res = p.supports_filters_pushdown(&[&f]).unwrap();
        assert_eq!(res, vec![TableProviderFilterPushDown::Inexact]);
    }

    #[test]
    fn pushdown_inexact_for_predicate_eq() {
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        let f = col_eq_str("predicate", "name");
        let res = p.supports_filters_pushdown(&[&f]).unwrap();
        assert_eq!(res, vec![TableProviderFilterPushDown::Inexact]);
    }

    #[test]
    fn pushdown_inexact_for_nars_frequency_gt() {
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        let f = col_gt_f32("nars_frequency", 0.7);
        let res = p.supports_filters_pushdown(&[&f]).unwrap();
        assert_eq!(res, vec![TableProviderFilterPushDown::Inexact]);
    }

    #[test]
    fn pushdown_unsupported_for_unknown_column() {
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        let f = col_eq_str("weird_field", "whatever");
        let res = p.supports_filters_pushdown(&[&f]).unwrap();
        assert_eq!(res, vec![TableProviderFilterPushDown::Unsupported]);
    }

    #[test]
    fn pushdown_unsupported_for_unknown_entity_type_string() {
        // Even if the column is `entity_type`, if the literal doesn't
        // resolve to a declared schema the lookup is empty — classify
        // as Unsupported so DataFusion handles it as residual.
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        let f = col_eq_str("entity_type", "Spaceship");
        let res = p.supports_filters_pushdown(&[&f]).unwrap();
        assert_eq!(res, vec![TableProviderFilterPushDown::Unsupported]);
    }

    #[test]
    fn pushdown_classifies_each_filter_independently() {
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        let recognized = col_eq_str("predicate", "name");
        let unrecognized = col_eq_str("weird", "x");
        let res = p
            .supports_filters_pushdown(&[&recognized, &unrecognized])
            .unwrap();
        assert_eq!(
            res,
            vec![
                TableProviderFilterPushDown::Inexact,
                TableProviderFilterPushDown::Unsupported,
            ]
        );
    }

    #[test]
    fn pushdown_handles_flipped_literal_op_column() {
        // `'Patient' = entity_type` should classify the same as
        // `entity_type = 'Patient'`. Tests the symmetric pattern in
        // `expr_to_filter_term`.
        use datafusion::logical_expr::lit;
        let ont = medcare_ontology();
        let p = OntologyTableProvider::new(ont, "Patient").unwrap();
        let f = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(lit("Patient")),
            op: Operator::Eq,
            right: Box::new(Expr::Column("entity_type".into())),
        });
        let res = p.supports_filters_pushdown(&[&f]).unwrap();
        assert_eq!(res, vec![TableProviderFilterPushDown::Inexact]);
    }
}
