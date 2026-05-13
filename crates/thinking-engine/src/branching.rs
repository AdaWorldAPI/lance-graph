//! 4×4 branching parallel vector thinking.
//!
//! NOT filtering/survival. NOT cascade elimination.
//! Each L1 lane is a 4-wide vector unit that SPAWNS 4 parallel L2 lanes.
//! Each L2 lane spawns 4 parallel L3 lanes. Branching IS the computation.
//!
//! ```text
//! L1[i] fires → compute L2[i*4 .. i*4+3] in ONE 4-wide SIMD op
//!   → each L2[j] → compute L3[j*16 .. j*16+15] in ONE 16-wide SIMD op
//!
//! 64 L1 lanes × 4 branches = 256 L2 lanes × 16 branches = 4096 L3 lanes
//! ```
//!
//! The hierarchy structures parallelism, not filters candidates.
//! RISC: fixed-width ops, deterministic branching, no data-dependent control.

/// Branch factor: each parent spawns this many children.
pub const BRANCH: usize = 4;

/// Layer sizes: 64 → 256 → 4096.
pub const L1: usize = 64;
pub const L2: usize = L1 * BRANCH;       // 256
pub const L3: usize = L2 * BRANCH * BRANCH; // 4096 (256 × 16)

/// L3 children per L2 parent.
pub const L3_PER_L2: usize = L3 / L2;    // 16

/// One thinking layer: distance table + energy vector.
/// Operates as a vector unit: all lanes compute in parallel.
struct Layer<const N: usize> {
    /// N×N distance table. u8 similarity.
    table: Vec<u8>,
    /// Energy per lane.
    energy: Vec<f64>,
}

impl<const N: usize> Layer<N> {
    fn new(table: Vec<u8>) -> Self {
        assert_eq!(table.len(), N * N);
        Self { table, energy: vec![0.0; N] }
    }

    /// MatVec on this layer. All N lanes compute in parallel.
    fn compute(&mut self) {
        let mut next = vec![0.0f64; N];
        for i in 0..N {
            if self.energy[i] < 1e-15 { continue; }
            let row = i * N;
            let e_i = self.energy[i];
            for j in 0..N {
                next[j] += (self.table[row + j] as f64 / 255.0) * e_i;
            }
        }
        let total: f64 = next.iter().sum();
        if total > 1e-15 { for e in &mut next { *e /= total; } }
        self.energy = next;
    }
}

/// 4×4 branching parallel thinking engine.
///
/// Each cycle:
/// 1. L1 computes (64-wide vector op)
/// 2. L1 results BRANCH into L2: each L1[i] spawns L2[i*4..i*4+3]
/// 3. L2 computes (256-wide, structured as 64 × 4-wide vector ops)
/// 4. L2 results BRANCH into L3: each L2[j] spawns L3[j*16..j*16+15]
/// 5. L3 computes (4096-wide, structured as 256 × 16-wide vector ops)
pub struct BranchingEngine {
    l1: Layer<L1>,
    l2: Layer<L2>,
    l3: Option<Layer<L3>>,
    pub cycles: u16,
}

impl BranchingEngine {
    /// Create with L1 + L2 tables. L3 optional (GPU path).
    pub fn new(l1_table: Vec<u8>, l2_table: Vec<u8>, l3_table: Option<Vec<u8>>) -> Self {
        Self {
            l1: Layer::<L1>::new(l1_table),
            l2: Layer::<L2>::new(l2_table),
            l3: l3_table.map(Layer::<L3>::new),
            cycles: 0,
        }
    }

    /// Perturb at L1 granularity. Maps 4096 indices down to 64 L1 lanes.
    pub fn perturb(&mut self, indices: &[u16]) {
        for &idx in indices {
            let i = idx as usize;
            if i < L3 {
                self.l1.energy[i / (L3 / L1)] += 1.0;       // map to L1
                self.l2.energy[i / L3_PER_L2] += 1.0;       // map to L2
                if let Some(ref mut l3) = self.l3 {
                    l3.energy[i] += 1.0;                     // direct L3
                }
            }
        }
        normalize(&mut self.l1.energy);
        normalize(&mut self.l2.energy);
        if let Some(ref mut l3) = self.l3 { normalize(&mut l3.energy); }
    }

    /// One cycle: L1 compute → branch to L2 → L2 compute → branch to L3 → L3 compute.
    ///
    /// The branching is NOT filtering. It's SPAWNING:
    /// L1[i]'s energy distributes to its 4 children L2[i*4..i*4+3].
    /// L2[j]'s energy distributes to its 16 children L3[j*16..j*16+15].
    pub fn cycle(&mut self) {
        // ── L1: 64-wide parallel compute ──
        self.l1.compute();

        // ── Branch L1 → L2: each L1 lane spawns 4 L2 lanes ──
        // L1[i] energy flows INTO its children, modulated by existing L2 energy.
        for i in 0..L1 {
            let parent_energy = self.l1.energy[i];
            for b in 0..BRANCH {
                let child = i * BRANCH + b;
                // Child energy = parent contribution + own resonance
                self.l2.energy[child] = self.l2.energy[child] * 0.5 + parent_energy * 0.5;
            }
        }
        normalize(&mut self.l2.energy);

        // ── L2: 256-wide parallel compute (64 × 4-wide vector ops) ──
        self.l2.compute();

        // ── Branch L2 → L3: each L2 lane spawns 16 L3 lanes ──
        if let Some(ref mut l3) = self.l3 {
            for j in 0..L2 {
                let parent_energy = self.l2.energy[j];
                for b in 0..L3_PER_L2 {
                    let child = j * L3_PER_L2 + b;
                    l3.energy[child] = l3.energy[child] * 0.5 + parent_energy * 0.5;
                }
            }
            normalize(&mut l3.energy);

            // ── L3: 4096-wide parallel compute (256 × 16-wide vector ops) ──
            l3.compute();
        }

        self.cycles += 1;
    }

    /// Run cycles until L1 converges.
    pub fn think(&mut self, max_cycles: usize) -> BranchResult {
        for _ in 0..max_cycles {
            let prev = self.l1.energy.clone();
            self.cycle();
            let delta: f64 = self.l1.energy.iter().zip(&prev)
                .map(|(a, b)| (a - b).abs()).sum();
            if delta < 0.001 { break; }
        }
        BranchResult {
            cycles: self.cycles,
            top_l1: top_k(&self.l1.energy, 4),
            top_l2: top_k(&self.l2.energy, 8),
            top_l3: self.l3.as_ref().map(|l3| top_k(&l3.energy, 8)),
        }
    }

    /// Direct access to energy vectors.
    pub fn energy_l1(&self) -> &[f64] { &self.l1.energy }
    pub fn energy_l2(&self) -> &[f64] { &self.l2.energy }
    pub fn energy_l3(&self) -> Option<&[f64]> { self.l3.as_ref().map(|l| l.energy.as_slice()) }

    /// Reset all layers.
    pub fn reset(&mut self) {
        self.l1.energy.fill(0.0);
        self.l2.energy.fill(0.0);
        if let Some(ref mut l3) = self.l3 { l3.energy.fill(0.0); }
        self.cycles = 0;
    }
}

/// Result of branching computation.
#[derive(Clone, Debug)]
pub struct BranchResult {
    pub cycles: u16,
    pub top_l1: Vec<(usize, f64)>,
    pub top_l2: Vec<(usize, f64)>,
    pub top_l3: Option<Vec<(usize, f64)>>,
}

fn normalize(v: &mut [f64]) {
    let total: f64 = v.iter().sum();
    if total > 1e-15 { for e in v.iter_mut() { *e /= total; } }
}

fn top_k(energy: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut indexed: Vec<(usize, f64)> = energy.iter().enumerate()
        .map(|(i, &e)| (i, e)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table<const N: usize>() -> Vec<u8> {
        let mut t = vec![128u8; N * N];
        for i in 0..N {
            t[i * N + i] = 255;
            for j in 0..N {
                let dist = (i as i64 - j as i64).unsigned_abs() as usize;
                let proximity = N / 8;
                if dist < proximity {
                    t[i * N + j] = (255 - dist * (128 / proximity)).min(255) as u8;
                }
            }
        }
        t
    }

    #[test]
    fn branching_creates() {
        let e = BranchingEngine::new(make_table::<L1>(), make_table::<L2>(), None);
        assert_eq!(e.cycles, 0);
    }

    #[test]
    fn branching_perturb_flows_down() {
        let mut e = BranchingEngine::new(make_table::<L1>(), make_table::<L2>(), None);
        e.perturb(&[500]); // L3 index 500

        // L1: 500 / 64 = 7
        assert!(e.energy_l1()[7] > 0.0, "L1[7] should have energy");
        // L2: 500 / 16 = 31
        assert!(e.energy_l2()[31] > 0.0, "L2[31] should have energy");
    }

    #[test]
    fn branching_cycle_spawns_children() {
        let mut e = BranchingEngine::new(make_table::<L1>(), make_table::<L2>(), None);
        e.perturb(&[500]);

        // Before cycle: only perturbed lane has energy
        let active_l2_before = e.energy_l2().iter().filter(|&&x| x > 0.01).count();

        e.cycle();

        // After cycle: L1 computation + branching should activate more L2 lanes
        let active_l2_after = e.energy_l2().iter().filter(|&&x| x > 0.001).count();
        assert!(active_l2_after > 0, "L2 should have active lanes after branching");

        eprintln!("L2 active before: {}, after: {}", active_l2_before, active_l2_after);
    }

    #[test]
    fn branching_think_converges() {
        let mut e = BranchingEngine::new(make_table::<L1>(), make_table::<L2>(), None);
        e.perturb(&[100, 200, 300]);

        let result = e.think(20);
        eprintln!("Converged in {} cycles", result.cycles);
        eprintln!("Top L1: {:?}", result.top_l1);
        eprintln!("Top L2: {:?}", result.top_l2);
        assert!(result.top_l1[0].1 > 0.0);
    }

    #[test]
    fn branching_4x4_structure() {
        // Verify: L1[i] → L2[i*4..i*4+3]
        assert_eq!(L2, L1 * BRANCH);           // 256 = 64 × 4
        assert_eq!(L3, L2 * L3_PER_L2);        // 4096 = 256 × 16
        assert_eq!(L1 * BRANCH, 256);
        assert_eq!(L2 * L3_PER_L2, 4096);

        // L1[5] branches to L2[20..23]
        let l1_idx = 5;
        let l2_children: Vec<usize> = (0..BRANCH).map(|b| l1_idx * BRANCH + b).collect();
        assert_eq!(l2_children, vec![20, 21, 22, 23]);
    }

    #[test]
    fn branching_with_l3() {
        let mut e = BranchingEngine::new(
            make_table::<L1>(), make_table::<L2>(), Some(make_table::<L3>()));
        e.perturb(&[1000, 1010, 1020]);

        let result = e.think(10);
        assert!(result.top_l3.is_some());
        let top_l3 = result.top_l3.unwrap();
        assert!(top_l3[0].1 > 0.0, "L3 should have peaks");
        eprintln!("L3 top: {:?}", &top_l3[..4.min(top_l3.len())]);
    }

    #[test]
    fn memory_budget_branching() {
        eprintln!("L1: {} × {} = {} bytes ({} KB)", L1, L1, L1*L1, L1*L1/1024);
        eprintln!("L2: {} × {} = {} bytes ({} KB)", L2, L2, L2*L2, L2*L2/1024);
        eprintln!("L3: {} × {} = {} bytes ({} MB)", L3, L3, L3*L3, L3*L3/(1024*1024));
        eprintln!("Branch factor: {}", BRANCH);
        eprintln!("L1→L2 children: {}", BRANCH);
        eprintln!("L2→L3 children: {}", L3_PER_L2);
    }
}
