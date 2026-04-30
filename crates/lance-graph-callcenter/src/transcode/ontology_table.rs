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
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;

use lance_graph_contract::ontology::Ontology;

use super::spo_filter::{FilterTerm, SpoFilterTranslator, SpoLookup};
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
        // Honest round-1 answer: nothing pushed down yet.
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Unsupported)
            .collect())
    }
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
}
