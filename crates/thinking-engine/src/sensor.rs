//! Sensor module: maps external signals into codebook index activations.
//!
//! Sensors convert embeddings, tokens, and distance-table rows into
//! `(atom_index, weight)` pairs that the ThinkingEngine consumes via `perturb`.
//!
//! No HTTP, no GGUF, no I/O — just activation logic.

use crate::engine::{ThinkingEngine, CODEBOOK_SIZE};
use ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd;

/// A single sensor that activates specific codebook atoms with weights.
#[derive(Clone, Debug)]
pub struct Sensor {
    /// Which codebook atoms this sensor activates.
    atom_indices: Vec<u16>,
    /// Weight per atom (default 1.0). Same length as `atom_indices`.
    weights: Vec<f64>,
    /// Sensor name for debugging.
    name: String,
}

impl Sensor {
    /// Create a sensor with explicit atom indices and weights.
    ///
    /// Panics if `atom_indices` and `weights` have different lengths.
    pub fn new(name: impl Into<String>, atom_indices: Vec<u16>, weights: Vec<f64>) -> Self {
        assert_eq!(
            atom_indices.len(),
            weights.len(),
            "atom_indices and weights must have the same length"
        );
        Self {
            atom_indices,
            weights,
            name: name.into(),
        }
    }

    /// Create a sensor by finding the `top_n` nearest centroids to an embedding.
    ///
    /// Uses SIMD cosine similarity (`f32` input, `f64` precision) to rank all
    /// `k` centroids and picks the `top_n` closest. Cosine similarity becomes
    /// the activation weight (clamped to non-negative).
    ///
    /// # Arguments
    /// - `name`: sensor name for debugging
    /// - `embedding`: the query embedding, length `dim`
    /// - `centroids`: flat array of `k * dim` f32 values (row-major)
    /// - `k`: number of centroids
    /// - `dim`: embedding dimension
    /// - `top_n`: how many centroids to activate
    pub fn from_embedding(
        name: impl Into<String>,
        embedding: &[f32],
        centroids: &[f32],
        k: usize,
        dim: usize,
        top_n: usize,
    ) -> Self {
        assert_eq!(embedding.len(), dim, "embedding length must equal dim");
        assert_eq!(
            centroids.len(),
            k * dim,
            "centroids must be k * dim elements"
        );
        assert!(top_n <= k, "top_n must be <= k");

        // Compute cosine similarity between embedding and each centroid.
        let mut similarities: Vec<(usize, f64)> = (0..k)
            .map(|i| {
                let centroid = &centroids[i * dim..(i + 1) * dim];
                let cos = cosine_f32_to_f64_simd(embedding, centroid);
                (i, cos)
            })
            .collect();

        // Sort descending by similarity.
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_n, clamp weights to non-negative.
        let top = &similarities[..top_n];
        let atom_indices: Vec<u16> = top.iter().map(|&(i, _)| i as u16).collect();
        let weights: Vec<f64> = top.iter().map(|&(_, cos)| cos.max(0.0)).collect();

        Self {
            atom_indices,
            weights,
            name: name.into(),
        }
    }

    /// Create a sensor from a distance table row.
    ///
    /// Activates all atoms whose distance value exceeds `threshold` in the
    /// given row of the distance table. Weight = `(value - threshold)` / 255.0,
    /// so stronger similarities get higher weights.
    ///
    /// # Arguments
    /// - `name`: sensor name for debugging
    /// - `distance_table`: flat `n * n` u8 array (row-major)
    /// - `row`: which row to read (0-indexed)
    /// - `n`: table dimension (number of atoms in the table)
    /// - `threshold`: minimum u8 value to activate (exclusive)
    pub fn from_distance_row(
        name: impl Into<String>,
        distance_table: &[u8],
        row: usize,
        n: usize,
        threshold: u8,
    ) -> Self {
        assert!(row < n, "row must be < n");
        assert!(
            distance_table.len() >= n * n,
            "distance_table too small for n×n"
        );

        let row_start = row * n;
        let mut atom_indices = Vec::new();
        let mut weights = Vec::new();

        for j in 0..n {
            let val = distance_table[row_start + j];
            if val > threshold {
                atom_indices.push(j as u16);
                weights.push((val - threshold) as f64 / 255.0);
            }
        }

        Self {
            atom_indices,
            weights,
            name: name.into(),
        }
    }

    /// Return (atom_index, weight) activation pairs.
    pub fn activate(&self) -> Vec<(u16, f64)> {
        self.atom_indices
            .iter()
            .zip(&self.weights)
            .map(|(&idx, &w)| (idx, w))
            .collect()
    }

    /// Sensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of atoms this sensor activates.
    pub fn len(&self) -> usize {
        self.atom_indices.len()
    }

    /// Whether this sensor activates zero atoms.
    pub fn is_empty(&self) -> bool {
        self.atom_indices.is_empty()
    }
}

/// A bank of sensors that can fire together into a ThinkingEngine.
#[derive(Clone, Debug, Default)]
pub struct SensorBank {
    sensors: Vec<Sensor>,
}

impl SensorBank {
    /// Create an empty sensor bank.
    pub fn new() -> Self {
        Self {
            sensors: Vec::new(),
        }
    }

    /// Register a sensor.
    pub fn add(&mut self, sensor: Sensor) {
        self.sensors.push(sensor);
    }

    /// Number of registered sensors.
    pub fn len(&self) -> usize {
        self.sensors.len()
    }

    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.sensors.is_empty()
    }

    /// Fire all sensors and merge activations.
    ///
    /// Duplicate atom indices have their weights summed.
    /// Returns merged `(atom_index, weight)` pairs sorted by index.
    pub fn fire_all(&self) -> Vec<(u16, f64)> {
        let mut map = std::collections::HashMap::new();
        for sensor in &self.sensors {
            for (idx, w) in sensor.activate() {
                *map.entry(idx).or_insert(0.0f64) += w;
            }
        }
        let mut result: Vec<(u16, f64)> = map.into_iter().collect();
        result.sort_by_key(|&(idx, _)| idx);
        result
    }

    /// Fire all sensors and inject directly into engine's energy array.
    pub fn fire_into(&self, engine: &mut ThinkingEngine) {
        let activations = self.fire_all();
        for &(idx, weight) in &activations {
            let i = idx as usize;
            if i < engine.size {
                engine.energy[i] += weight;
            }
        }
        let total: f64 = engine.energy.iter().sum();
        if total > 1e-15 {
            for e in &mut engine.energy { *e /= total; }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_new_and_activate() {
        let s = Sensor::new("test", vec![10, 20, 30], vec![1.0, 0.5, 0.25]);
        assert_eq!(s.name(), "test");
        assert_eq!(s.len(), 3);

        let acts = s.activate();
        assert_eq!(acts, vec![(10, 1.0), (20, 0.5), (30, 0.25)]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn sensor_new_mismatched_lengths() {
        Sensor::new("bad", vec![1, 2], vec![1.0]);
    }

    #[test]
    fn sensor_empty() {
        let s = Sensor::new("empty", vec![], vec![]);
        assert!(s.is_empty());
        assert_eq!(s.activate(), vec![]);
    }

    #[test]
    fn from_embedding_basic() {
        // 4 centroids in 3D.
        let centroids: Vec<f32> = vec![
            1.0, 0.0, 0.0, // centroid 0: x-axis
            0.0, 1.0, 0.0, // centroid 1: y-axis
            0.0, 0.0, 1.0, // centroid 2: z-axis
            0.7, 0.7, 0.0, // centroid 3: between x and y
        ];
        // Query along x-axis.
        let embedding = vec![1.0f32, 0.0, 0.0];

        let s = Sensor::from_embedding("emb", &embedding, &centroids, 4, 3, 2);
        assert_eq!(s.len(), 2);

        let acts = s.activate();
        // Closest to [1,0,0] should be centroid 0 (cos=1.0), then centroid 3.
        assert_eq!(acts[0].0, 0); // centroid 0
        assert!((acts[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(acts[1].0, 3); // centroid 3
    }

    #[test]
    fn from_embedding_top_n_one() {
        let centroids: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let embedding = vec![0.9f32, 0.1];

        let s = Sensor::from_embedding("single", &embedding, &centroids, 2, 2, 1);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn from_distance_row_basic() {
        // 4x4 table.
        let table: Vec<u8> = vec![
            255, 200, 100, 50, // row 0
            200, 255, 150, 80, // row 1
            100, 150, 255, 120, // row 2
            50, 80, 120, 255, // row 3
        ];

        let s = Sensor::from_distance_row("dist", &table, 0, 4, 128);
        let acts = s.activate();
        // Row 0: [255, 200, 100, 50]. Above 128: index 0 (255) and index 1 (200).
        assert_eq!(acts.len(), 2);
        assert_eq!(acts[0].0, 0);
        assert_eq!(acts[1].0, 1);
        // Weights: (255-128)/255 and (200-128)/255.
        assert!((acts[0].1 - 127.0 / 255.0).abs() < 1e-10);
        assert!((acts[1].1 - 72.0 / 255.0).abs() < 1e-10);
    }

    #[test]
    fn from_distance_row_none_above_threshold() {
        let table: Vec<u8> = vec![50, 60, 70, 80];
        let s = Sensor::from_distance_row("empty", &table, 0, 2, 200);
        assert!(s.is_empty());
    }

    #[test]
    fn sensor_bank_fire_all_merges_duplicates() {
        let s1 = Sensor::new("a", vec![10, 20], vec![1.0, 0.5]);
        let s2 = Sensor::new("b", vec![20, 30], vec![0.3, 0.7]);

        let mut bank = SensorBank::new();
        bank.add(s1);
        bank.add(s2);

        let acts = bank.fire_all();
        // Index 10: 1.0, Index 20: 0.5 + 0.3 = 0.8, Index 30: 0.7.
        let map: std::collections::HashMap<u16, f64> =
            acts.into_iter().collect();
        assert!((map[&10] - 1.0).abs() < 1e-10);
        assert!((map[&20] - 0.8).abs() < 1e-10);
        assert!((map[&30] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn sensor_bank_empty() {
        let bank = SensorBank::new();
        assert!(bank.is_empty());
        assert_eq!(bank.fire_all(), vec![]);
    }

    #[test]
    fn fire_into_engine() {
        // Build a minimal distance table.
        let table = vec![128u8; CODEBOOK_SIZE * CODEBOOK_SIZE];
        let mut engine = ThinkingEngine::new(table);

        let s = Sensor::new("inject", vec![42, 100], vec![2.0, 1.0]);
        let mut bank = SensorBank::new();
        bank.add(s);

        bank.fire_into(&mut engine);

        // Energy should be normalized.
        let total: f64 = engine.energy.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);

        // Atom 42 should have 2/3 of energy, atom 100 should have 1/3.
        assert!((engine.energy[42] - 2.0 / 3.0).abs() < 1e-10);
        assert!((engine.energy[100] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn fire_into_adds_to_existing_energy() {
        let table = vec![128u8; CODEBOOK_SIZE * CODEBOOK_SIZE];
        let mut engine = ThinkingEngine::new(table);

        // First perturbation via engine.perturb.
        engine.perturb(&[42]);
        assert!((engine.energy[42] - 1.0).abs() < 1e-10);

        // Second perturbation via sensor bank.
        let s = Sensor::new("add", vec![100], vec![1.0]);
        let mut bank = SensorBank::new();
        bank.add(s);
        bank.fire_into(&mut engine);

        // Both should have energy now.
        assert!(engine.energy[42] > 0.0);
        assert!(engine.energy[100] > 0.0);
        let total: f64 = engine.energy.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fire_into_then_think() {
        // Structured table: nearby indices are similar.
        let mut table = vec![128u8; CODEBOOK_SIZE * CODEBOOK_SIZE];
        for i in 0..CODEBOOK_SIZE {
            table[i * CODEBOOK_SIZE + i] = 255;
            for j in 0..CODEBOOK_SIZE {
                let dist = (i as i64 - j as i64).unsigned_abs() as usize;
                if dist < 50 {
                    table[i * CODEBOOK_SIZE + j] = (255 - dist * 2).min(255) as u8;
                }
            }
        }

        let mut engine = ThinkingEngine::new(table);

        let s1 = Sensor::new("s1", vec![100, 101, 102], vec![1.0, 1.0, 1.0]);
        let s2 = Sensor::new("s2", vec![110, 111], vec![0.5, 0.5]);
        let mut bank = SensorBank::new();
        bank.add(s1);
        bank.add(s2);

        bank.fire_into(&mut engine);
        let resonance = engine.think(15);

        // Should converge with peaks near the sensor region.
        assert!(resonance.top_k[0].1 > 0.0);
        let top_idx = resonance.top_k[0].0 as usize;
        // Top peak should be in the neighborhood of 100-111.
        assert!(
            top_idx >= 80 && top_idx <= 130,
            "expected peak near 100-111, got {}",
            top_idx
        );
    }
}
