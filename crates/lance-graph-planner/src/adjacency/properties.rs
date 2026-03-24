//! Edge properties stored columnar.
//!
//! NARS truth values are columns here: "truth_f", "truth_c", "truth_t".
//! Each property is a separate typed array (Arrow-compatible layout).

use std::collections::HashMap;

/// Edge properties stored in columnar format.
#[derive(Debug, Clone)]
pub struct EdgeProperties {
    /// Float columns (e.g., "truth_f", "truth_c", "weight").
    pub float_columns: HashMap<String, Vec<f32>>,
    /// Integer columns (e.g., "truth_t" temporal stamp).
    pub int_columns: HashMap<String, Vec<u64>>,
    /// String columns (e.g., "label").
    pub string_columns: HashMap<String, Vec<String>>,
    /// Fingerprint columns (e.g., "fingerprint" — Vec<u64> per edge).
    pub fingerprint_columns: HashMap<String, Vec<Vec<u64>>>,
    /// Total number of edges.
    pub len: usize,
}

impl EdgeProperties {
    pub fn new() -> Self {
        Self {
            float_columns: HashMap::new(),
            int_columns: HashMap::new(),
            string_columns: HashMap::new(),
            fingerprint_columns: HashMap::new(),
            len: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            float_columns: HashMap::new(),
            int_columns: HashMap::new(),
            string_columns: HashMap::new(),
            fingerprint_columns: HashMap::new(),
            len: cap,
        }
    }

    /// Add NARS truth value columns.
    pub fn with_nars_truth(mut self, frequencies: Vec<f32>, confidences: Vec<f32>) -> Self {
        assert_eq!(frequencies.len(), confidences.len());
        self.len = frequencies.len();
        self.float_columns.insert("truth_f".into(), frequencies);
        self.float_columns.insert("truth_c".into(), confidences);
        self
    }

    /// Get NARS truth value for a specific edge.
    pub fn truth_value(&self, edge_id: u64) -> Option<(f32, f32)> {
        let f = self.float_columns.get("truth_f")?.get(edge_id as usize)?;
        let c = self.float_columns.get("truth_c")?.get(edge_id as usize)?;
        Some((*f, *c))
    }

    /// Get a float property for a specific edge.
    pub fn get_float(&self, column: &str, edge_id: u64) -> Option<f32> {
        self.float_columns.get(column)?.get(edge_id as usize).copied()
    }

    /// Get fingerprint property for a specific edge.
    pub fn get_fingerprint(&self, column: &str, edge_id: u64) -> Option<&[u64]> {
        self.fingerprint_columns.get(column)?.get(edge_id as usize).map(|v| v.as_slice())
    }
}

impl Default for EdgeProperties {
    fn default() -> Self {
        Self::new()
    }
}
