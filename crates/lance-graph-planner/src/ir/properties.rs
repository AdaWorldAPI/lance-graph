//! Plan properties: UCCs, functional dependencies, ordering (from Hyrise).

/// Properties tracked for optimizer rules.
#[derive(Debug, Clone, Default)]
pub struct PlanProperties {
    /// Unique column combinations (keys).
    pub uccs: Vec<Vec<String>>,
    /// Functional dependencies: determinant → dependent.
    pub fds: Vec<(Vec<String>, String)>,
    /// Output ordering (if guaranteed).
    pub ordering: Option<Vec<OrderingColumn>>,
    /// Estimated cardinality.
    pub cardinality: f64,
}

#[derive(Debug, Clone)]
pub struct OrderingColumn {
    pub column: String,
    pub ascending: bool,
}

impl PlanProperties {
    pub fn with_cardinality(cardinality: f64) -> Self {
        Self {
            cardinality,
            ..Default::default()
        }
    }

    /// Check if a set of columns is a unique key.
    pub fn is_unique(&self, columns: &[String]) -> bool {
        self.uccs
            .iter()
            .any(|ucc| ucc.iter().all(|c| columns.contains(c)))
    }

    /// Check if a column is functionally determined by a set of columns.
    pub fn is_determined_by(&self, column: &str, determinant: &[String]) -> bool {
        self.fds
            .iter()
            .any(|(det, dep)| dep == column && det.iter().all(|d| determinant.contains(d)))
    }
}
