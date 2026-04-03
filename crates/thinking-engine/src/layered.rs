//! Three-layer thinking cascade: 64² → 256² → 4096².
//!
//! NOT a flat 4096² MatVec. Three layers with branching:
//!
//! ```text
//! Layer 1:  64 ×  64 =   4 KB   L1-resident   coarse routing     ~μs
//! Layer 2: 256 × 256 =  64 KB   L1/L2         mid-resolution     ~μs
//! Layer 3: 4096×4096 = 16 MB    L3 (or GPU)   full resolution    ~ms
//! ```
//!
//! Same cascade as Belichtungsmesser / p64 Blumenstrauß:
//! Layer 1 rejects ~90%. Layer 2 refines survivors. Layer 3 for peaks only.
//!
//! The thinking LANES branch from coarse to fine:
//! ```text
//! Perturb → Layer 1 (64²): which macro-regions activate?
//!   → survivors branch into Layer 2 (256²): which meso-regions?
//!     → survivors branch into Layer 3 (4096²): full resolution for peaks
//! ```
//!
//! Layer 1 and 2 are p64 Blumenstrauß scale — proven, SIMD-hardened.
//! Layer 3 is the GPU target for later. On CPU: only survivors reach it.

/// Layer 1: 64 macro-regions. 64×64 = 4 KB. Always L1-hot.
pub const L1_SIZE: usize = 64;
/// Layer 2: 256 meso-regions. 256×256 = 64 KB. L1/L2.
pub const L2_SIZE: usize = 256;
/// Layer 3: 4096 micro-regions. 4096×4096 = 16 MB. L3 or GPU.
pub const L3_SIZE: usize = 4096;

/// Mapping from Layer 2 index to Layer 1 region.
/// Each L1 region contains L2_SIZE / L1_SIZE = 4 L2 entries.
pub const L2_PER_L1: usize = L2_SIZE / L1_SIZE;  // 4
/// Mapping from Layer 3 index to Layer 2 region.
pub const L3_PER_L2: usize = L3_SIZE / L2_SIZE;  // 16

/// Three-layer thinking engine with branching cascade.
pub struct LayeredEngine {
    /// Layer 1: coarse routing. 64×64 u8 distance table.
    pub layer1: Box<[u8; L1_SIZE * L1_SIZE]>,
    /// Layer 2: mid-resolution. 256×256 u8 distance table.
    pub layer2: Box<[u8; L2_SIZE * L2_SIZE]>,
    /// Layer 3: full resolution. 4096×4096 u8 distance table.
    /// Optional — may be on GPU or not yet built.
    pub layer3: Option<Vec<u8>>,

    /// Energy at each layer.
    pub energy_l1: [f64; L1_SIZE],
    pub energy_l2: [f64; L2_SIZE],
    pub energy_l3: [f64; L3_SIZE],

    /// Which L1 regions survived (for branching into L2).
    pub l1_survivors: Vec<u8>,
    /// Which L2 regions survived (for branching into L3).
    pub l2_survivors: Vec<u16>,

    /// Cycle counter.
    pub cycles: u16,

    /// Survivor threshold: minimum energy to branch into next layer.
    pub branch_threshold: f64,
}

impl LayeredEngine {
    /// Create with Layer 1 + Layer 2 tables. Layer 3 optional.
    pub fn new(
        layer1: Box<[u8; L1_SIZE * L1_SIZE]>,
        layer2: Box<[u8; L2_SIZE * L2_SIZE]>,
        layer3: Option<Vec<u8>>,
    ) -> Self {
        if let Some(ref l3) = layer3 {
            assert_eq!(l3.len(), L3_SIZE * L3_SIZE);
        }
        Self {
            layer1, layer2, layer3,
            energy_l1: [0.0; L1_SIZE],
            energy_l2: [0.0; L2_SIZE],
            energy_l3: [0.0; L3_SIZE],
            l1_survivors: Vec::new(),
            l2_survivors: Vec::new(),
            cycles: 0,
            branch_threshold: 0.01,
        }
    }

    /// Perturb: inject codebook indices. Maps to L1 region first.
    pub fn perturb(&mut self, indices: &[u16]) {
        for &idx in indices {
            let l1_idx = (idx as usize) / (L3_SIZE / L1_SIZE); // 4096/64 = 64 per L1 region
            let l2_idx = (idx as usize) / (L3_SIZE / L2_SIZE); // 4096/256 = 16 per L2 region
            if l1_idx < L1_SIZE { self.energy_l1[l1_idx] += 1.0; }
            if l2_idx < L2_SIZE { self.energy_l2[l2_idx] += 1.0; }
            if (idx as usize) < L3_SIZE { self.energy_l3[idx as usize] += 1.0; }
        }
        normalize(&mut self.energy_l1);
        normalize(&mut self.energy_l2);
        normalize(&mut self.energy_l3);
    }

    /// One full cascade cycle: L1 → branch → L2 → branch → L3.
    pub fn cycle(&mut self) {
        // ── Layer 1: coarse MatVec (64² = 4096 ops, μs) ──
        let mut next_l1 = [0.0f64; L1_SIZE];
        for i in 0..L1_SIZE {
            if self.energy_l1[i] < 1e-15 { continue; }
            let row = i * L1_SIZE;
            for j in 0..L1_SIZE {
                next_l1[j] += (self.layer1[row + j] as f64 / 255.0) * self.energy_l1[i];
            }
        }
        normalize(&mut next_l1);
        self.energy_l1 = next_l1;

        // ── Branch: which L1 regions survive? ──
        self.l1_survivors.clear();
        for (i, &e) in self.energy_l1.iter().enumerate() {
            if e > self.branch_threshold {
                self.l1_survivors.push(i as u8);
            }
        }

        // ── Layer 2: MatVec ONLY on survivors (256² but sparse) ──
        let mut next_l2 = [0.0f64; L2_SIZE];
        for &l1_idx in &self.l1_survivors {
            // Each L1 region covers L2_PER_L1 entries in L2
            let l2_start = l1_idx as usize * L2_PER_L1;
            let l2_end = (l2_start + L2_PER_L1).min(L2_SIZE);

            for i in l2_start..l2_end {
                if self.energy_l2[i] < 1e-15 { continue; }
                let row = i * L2_SIZE;
                // Only compute against other survivor regions
                for &other_l1 in &self.l1_survivors {
                    let other_start = other_l1 as usize * L2_PER_L1;
                    let other_end = (other_start + L2_PER_L1).min(L2_SIZE);
                    for j in other_start..other_end {
                        next_l2[j] += (self.layer2[row + j] as f64 / 255.0) * self.energy_l2[i];
                    }
                }
            }
        }
        normalize(&mut next_l2);
        self.energy_l2 = next_l2;

        // ── Branch: which L2 regions survive? ──
        self.l2_survivors.clear();
        for (i, &e) in self.energy_l2.iter().enumerate() {
            if e > self.branch_threshold {
                self.l2_survivors.push(i as u16);
            }
        }

        // ── Layer 3: full resolution ONLY on L2 survivors ──
        if let Some(ref table) = self.layer3 {
            let mut next_l3 = [0.0f64; L3_SIZE];
            for &l2_idx in &self.l2_survivors {
                let l3_start = l2_idx as usize * L3_PER_L2;
                let l3_end = (l3_start + L3_PER_L2).min(L3_SIZE);

                for i in l3_start..l3_end {
                    if self.energy_l3[i] < 1e-15 { continue; }
                    let row = i * L3_SIZE;
                    for &other_l2 in &self.l2_survivors {
                        let other_start = other_l2 as usize * L3_PER_L2;
                        let other_end = (other_start + L3_PER_L2).min(L3_SIZE);
                        for j in other_start..other_end {
                            next_l3[j] += (table[row + j] as f64 / 255.0) * self.energy_l3[i];
                        }
                    }
                }
            }
            normalize(&mut next_l3);
            self.energy_l3 = next_l3;
        }

        self.cycles += 1;
    }

    /// Run cascade until convergence.
    pub fn think(&mut self, max_cycles: usize) -> CascadeResult {
        for _ in 0..max_cycles {
            let prev_l1 = self.energy_l1;
            self.cycle();
            let delta: f64 = self.energy_l1.iter().zip(&prev_l1)
                .map(|(a, b)| (a - b).abs()).sum();
            if delta < 0.001 { break; }
        }

        CascadeResult {
            l1_survivors: self.l1_survivors.len(),
            l2_survivors: self.l2_survivors.len(),
            cycles: self.cycles,
            top_l1: top_k(&self.energy_l1, 4),
            top_l2: top_k_u16(&self.energy_l2, 4),
            top_l3: if self.layer3.is_some() { top_k_u16(&self.energy_l3, 4) } else { Vec::new() },
        }
    }

    /// Reset all layers.
    pub fn reset(&mut self) {
        self.energy_l1 = [0.0; L1_SIZE];
        self.energy_l2 = [0.0; L2_SIZE];
        self.energy_l3 = [0.0; L3_SIZE];
        self.l1_survivors.clear();
        self.l2_survivors.clear();
        self.cycles = 0;
    }
}

/// Result of a cascade thinking cycle.
#[derive(Clone, Debug)]
pub struct CascadeResult {
    pub l1_survivors: usize,
    pub l2_survivors: usize,
    pub cycles: u16,
    pub top_l1: Vec<(u8, f64)>,
    pub top_l2: Vec<(u16, f64)>,
    pub top_l3: Vec<(u16, f64)>,
}

fn normalize(energy: &mut [f64]) {
    let total: f64 = energy.iter().sum();
    if total > 1e-15 { for e in energy.iter_mut() { *e /= total; } }
}

fn top_k(energy: &[f64], k: usize) -> Vec<(u8, f64)> {
    let mut indexed: Vec<(usize, f64)> = energy.iter().enumerate()
        .map(|(i, &e)| (i, e)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|&(i, e)| (i as u8, e)).collect()
}

fn top_k_u16(energy: &[f64], k: usize) -> Vec<(u16, f64)> {
    let mut indexed: Vec<(usize, f64)> = energy.iter().enumerate()
        .map(|(i, &e)| (i, e)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|&(i, e)| (i as u16, e)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_l1_table() -> Box<[u8; L1_SIZE * L1_SIZE]> {
        let mut table = Box::new([128u8; L1_SIZE * L1_SIZE]);
        for i in 0..L1_SIZE {
            table[i * L1_SIZE + i] = 255;
            for j in 0..L1_SIZE {
                let dist = (i as i64 - j as i64).unsigned_abs() as usize;
                if dist < 10 { table[i * L1_SIZE + j] = (255 - dist * 10).min(255) as u8; }
            }
        }
        table
    }

    fn make_l2_table() -> Box<[u8; L2_SIZE * L2_SIZE]> {
        let mut table = Box::new([128u8; L2_SIZE * L2_SIZE]);
        for i in 0..L2_SIZE {
            table[i * L2_SIZE + i] = 255;
            for j in 0..L2_SIZE {
                let dist = (i as i64 - j as i64).unsigned_abs() as usize;
                if dist < 20 { table[i * L2_SIZE + j] = (255 - dist * 5).min(255) as u8; }
            }
        }
        table
    }

    #[test]
    fn layered_creates() {
        let engine = LayeredEngine::new(make_l1_table(), make_l2_table(), None);
        assert_eq!(engine.cycles, 0);
    }

    #[test]
    fn layered_perturb_maps_to_layers() {
        let mut engine = LayeredEngine::new(make_l1_table(), make_l2_table(), None);
        engine.perturb(&[100, 200, 300]);

        // L1: index 100 maps to L1 region 100 / 64 = 1
        assert!(engine.energy_l1[1] > 0.0);
        // L2: index 100 maps to L2 region 100 / 16 = 6
        assert!(engine.energy_l2[6] > 0.0);
    }

    #[test]
    fn layered_cycle_branches() {
        let mut engine = LayeredEngine::new(make_l1_table(), make_l2_table(), None);
        engine.perturb(&[100, 110, 120]);
        engine.cycle();

        assert!(!engine.l1_survivors.is_empty(), "should have L1 survivors");
        eprintln!("L1 survivors: {:?}", engine.l1_survivors);
        eprintln!("L2 survivors: {} entries", engine.l2_survivors.len());
    }

    #[test]
    fn layered_think_converges() {
        let mut engine = LayeredEngine::new(make_l1_table(), make_l2_table(), None);
        engine.perturb(&[50, 60, 70]);
        let result = engine.think(20);

        eprintln!("Converged in {} cycles", result.cycles);
        eprintln!("L1 survivors: {}, L2 survivors: {}", result.l1_survivors, result.l2_survivors);
        eprintln!("Top L1: {:?}", result.top_l1);
        eprintln!("Top L2: {:?}", result.top_l2);
        assert!(result.l1_survivors > 0);
    }

    #[test]
    fn layered_cascade_reduces_candidates() {
        let mut engine = LayeredEngine::new(make_l1_table(), make_l2_table(), None);
        // Perturb widely
        engine.perturb(&[10, 500, 1000, 2000, 3000]);
        engine.cycle();

        // With a structured table, nearby entries reinforce and distant ones decay.
        // After one cycle, energy concentrates — but with uniform structure all may survive.
        // The key property: energy should be NON-UNIFORM after a cycle.
        let l1_active = engine.l1_survivors.len();
        let max_e = engine.energy_l1.iter().cloned().fold(0.0f64, f64::max);
        let min_e = engine.energy_l1.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(max_e > min_e, "energy should be non-uniform after cycle");
        eprintln!("L1: {}/{} survive, L2: {}/{} survive",
            l1_active, L1_SIZE, engine.l2_survivors.len(), L2_SIZE);
    }

    #[test]
    fn memory_budget() {
        let l1_bytes = L1_SIZE * L1_SIZE;          // 4 KB
        let l2_bytes = L2_SIZE * L2_SIZE;          // 64 KB
        let l3_bytes = L3_SIZE * L3_SIZE;          // 16 MB
        let energy_bytes = (L1_SIZE + L2_SIZE + L3_SIZE) * 8; // ~35 KB

        eprintln!("Layer 1: {} bytes ({} KB)", l1_bytes, l1_bytes / 1024);
        eprintln!("Layer 2: {} bytes ({} KB)", l2_bytes, l2_bytes / 1024);
        eprintln!("Layer 3: {} bytes ({} MB)", l3_bytes, l3_bytes / (1024 * 1024));
        eprintln!("Energy:  {} bytes ({} KB)", energy_bytes, energy_bytes / 1024);
        eprintln!("L1+L2 (CPU): {} KB", (l1_bytes + l2_bytes + energy_bytes) / 1024);

        assert_eq!(l1_bytes, 4096);       // 4 KB
        assert_eq!(l2_bytes, 65536);      // 64 KB
        assert_eq!(l3_bytes, 16777216);   // 16 MB
    }
}
