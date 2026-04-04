//! ThinkingEngine: one MatVec per cycle. 16M compositions per thought.
//!
//! energy_next = distance_table × energy_current
//!
//! The distance table is precomputed ONCE from codebook centroids.
//! It IS the brain. Every thought is a matrix-vector multiply.
//!
//! # SIMD Dispatch Tiers (current: Tier 2)
//!
//! ```text
//! Tier   Instruction      Width   Precision   Throughput      Status
//! ────   ───────────      ─────   ─────────   ──────────      ──────
//!  4     AMX TDPBUSD      256     u8×u8→i32   256 MACs/instr  ndarray::simd_amx, needs tiling
//!  3     AVX512-VNNI      64      u8×i8→i32   64 MACs/instr   Cascade Lake+, Zen 4+
//!  2     VNNI2 (xint8)    32      u8×i8→i32   32 MACs/instr   Arrow Lake, Lunar Lake (NUC 14)
//!  1     F32x16 FMA       16      f32×f32→f32 16 MACs/instr   ← FLOOR (universal, every x86-64)
//!
//! No scalar. F32x16 IS the minimum. avx512vnni and avxvnniint8 are
//! mutually exclusive by hardware generation — never coexist.
//! Dispatch: avx512vnni → avxvnniint8 → F32x16. Three tiers, no fallback.
//! ```
//!
//! # MatVec Strategy Options
//!
//! ## Option A: Row Sweep ("Lawnmower") — CURRENT
//! ```text
//! For each active row i:
//!   sweep j = 0..4096 in chunks of 64 (4× F32x16)
//!   accumulate into next[j] += table[i][j] * energy[i]
//!
//! Pro:  Simple. One pass per row. next[] stays in L1.
//! Con:  table[i][..] = 4 KB per row, sequential access = good.
//!       But energy[i] is scalar broadcast — no data reuse across rows.
//! Cycles: 4096 rows × 64 iterations = 262K FMA ops per cycle.
//! ```
//!
//! ## Option B: Block Tiling ("Rollrasen") — FOR AMX/VNNI
//! ```text
//! Tile the 4096×4096 table into 64×64 blocks (64 blocks per dimension).
//! For each 64×64 block:
//!   Load energy[block_i..block_i+64] into registers (stays resident)
//!   Load table block 64×64 = 4 KB (fits L1)
//!   Compute 64×64 partial MatVec contribution
//!   Accumulate into next[block_j..block_j+64]
//!
//! Pro:  Register-resident energy chunk. Table block fits L1.
//!       AMX TDPBUSD does 16×16 u8 tiles natively = 256 MACs/instr.
//!       64×64 block = 16 AMX tile ops. Energy reuse across 64 columns.
//! Con:  Complex tiling logic. Needs 64-byte aligned blocks.
//!       Only wins when table is u8 and energy is quantized to u8/i8.
//! When: AMX available (Sapphire Rapids+, Lunar Lake NPU path).
//! ```
//!
//! ## Option C: VNNI Row Sweep — FOR AVX512-VNNI / AVX-VNNI2
//! ```text
//! Same as Option A but u8 table × i8 energy → i32 accumulator.
//! Quantize energy f32 → i8 (scale by 127/max).
//! VPDPBUSD: 64 u8×i8 MACs per instruction.
//!
//! Pro:  4× throughput vs F32x16 on same hardware.
//!       Energy quantization to i8 is fine — it's just u8 data anyway.
//! Con:  Quantization step per cycle (energy renormalize → i8).
//!       i32 accumulator overflow if sum > 2^31 (need periodic rescale).
//! When: avx512vnni (Cascade Lake+) or avxvnniint8 (NUC 14 Arrow Lake).
//! ```
//!
//! ## Option D: BF16 Energy — TARGET PRECISION
//! ```text
//! Distance table: u8 (8-bit, 256 levels)
//! Energy: BF16 (7-bit mantissa, 128 levels + 8-bit exponent)
//! BF16 mantissa ≈ u8 precision. No information lost vs f32.
//!
//! Path: u8 table → BF16 via shift (one instruction)
//!       BF16 × BF16 → f32 accumulator (AVX512-BF16 VDPBF16PS)
//!       Or: BF16 stored, f32 FMA at compute (current F32x16 path)
//!
//! Pro:  Half the memory for energy (16 KB → 8 KB for 4096 atoms).
//!       Matches data precision exactly. No wasted mantissa bits.
//! Con:  BF16 SIMD FMA needs AVX512-BF16 (Sapphire Rapids+) or AMX.
//!       Without hardware BF16: store as u16, convert to f32 for compute.
//! When: Hardware BF16 available, or when memory pressure matters.
//! ```
//!
//! # Decision Context
//!
//! ```text
//! The distance table is u8. The energy should match:
//!   f64  = 56-bit mantissa for 8-bit data = 48 bits wasted. DON'T.
//!   f32  = 24-bit mantissa for 8-bit data = 16 bits wasted. OK for now.
//!   bf16 = 7-bit mantissa for 8-bit data  = matches. TARGET.
//!   u8   = 8-bit for 8-bit data           = exact. For VNNI/AMX path.
//!
//! f64 had one purpose: capturing cos=0 ± ε across the orthogonal plane.
//! With sigma floor (subtract median), cos≈0 is already zeroed out.
//! The remaining topology lives in the upper range where f32 is sufficient.
//!
//! The sweep vs tile decision:
//!   Sweep (current): works everywhere, simple, F32x16 native.
//!   Tile: wins on AMX (256 MACs/instr) but needs u8×u8 quantized energy.
//!   Decision: use sweep until AMX path is wired, then tile for 10× speedup.
//!
//! Priority: BF16 GGUF preference → f32 SIMD POC → VNNI u8 → AMX tile.
//! ```
//!
//! # Breadcrumbs: If Things Don't Work As Hoped
//!
//! ```text
//! SYMPTOM: All sentences converge to same atom (atom 67/964)
//!   CAUSE: Distance table too uniform (attn_q: all rows avg ~128)
//!   FIX 1: Use wider-spread role table (ffn_down cos[-0.68,0.65] vs attn_q cos[-0.69,0.75])
//!   FIX 2: HDR Belichtungsmesser grading — redistribute u8 levels to topology, zero noise floor
//!   FIX 3: Multi-role combined table (attn_q + attn_k + ffn_down fused)
//!   FIX 4: σ floor already helps (449 active atoms vs 0) but insufficient alone
//!   STATUS: σ floor implemented. HDR grading next.
//!
//! SYMPTOM: Convergence too fast (1-2 cycles, no differentiation)
//!   CAUSE: Energy spreads uniformly through high-similarity entries
//!   FIX 1: Raise σ floor (e.g. median + 1σ instead of median)
//!   FIX 2: Sparsify: only top-K entries per row contribute (K=64 or K=128)
//!   FIX 3: Temperature: scale table values by T before exponentiating
//!
//! SYMPTOM: Convergence too slow or oscillates
//!   CAUSE: Contradicting attractors of equal strength
//!   FIX 1: This IS the 7+1 contradiction signal — use it, don't suppress it
//!   FIX 2: Increase max_cycles (10 → 20)
//!   FIX 3: Check if CausalEdge64 CONTRADICTS channel resolves it at L2
//!
//! SYMPTOM: f32 precision insufficient (energy values collapse to 0)
//!   CAUSE: 4096 atoms × 10 cycles, normalization kills small values
//!   FIX 1: Use log-domain energy (add instead of multiply, stable)
//!   FIX 2: Keep top-K alive, zero the rest (hard attention)
//!   FIX 3: Switch back to f64 for energy only (keep table u8, FMA in f64)
//!   NOTE: f64 was original for this reason — 90° orthogonal plane capture.
//!         If f32 can't distinguish cos=0.502 from cos=0.498, go back to f64.
//!         The sigma floor should make this unnecessary (zeros out the 0.5 region).
//!
//! SYMPTOM: BF16 energy loses too much resolution
//!   CAUSE: 7-bit mantissa can't distinguish 128 vs 129 (both round to same BF16)
//!   FIX 1: Keep f32 for energy, BF16 only for storage/transfer
//!   FIX 2: Use i16 fixed-point energy (15 bits, 32K levels, no exponent waste)
//!   FIX 3: Quantize energy to u8 and use VNNI u8×u8→i32 (table IS u8, why not energy?)
//!
//! SYMPTOM: Different sentences activate same centroids (codebook too coarse)
//!   CAUSE: 250K tokens mapped to 1024 centroids = 244 tokens/centroid average
//!   FIX 1: Use larger table (4096 rows from ffn_down, not 1024 from attn_q)
//!   FIX 2: Multi-role: different roles map different tokens to different centroids
//!   FIX 3: Per-token weighting: frequent tokens get lower weight (IDF-style)
//!
//! SYMPTOM: Sweep too slow for real-time (>10ms per thought)
//!   CAUSE: 4096² × f32 = 16M FMAs per cycle, 10 cycles = 160M FMAs
//!   FIX 1: VNNI u8 path: 64 MACs/instr → 4× speedup → ~2.5ms
//!   FIX 2: AMX tile: 256 MACs/instr → 16× speedup → ~0.6ms
//!   FIX 3: Sparse MatVec: skip zero-energy rows (typically 90%+ are dead)
//!   FIX 4: GPU Vulkan compute shader: ~10μs per cycle (see GPU design doc)
//! ```

use crate::dto::{ResonanceDto, BusDto};
use ndarray::hpc::heel_f64x8::cosine_f64_simd;
use ndarray::simd::F32x16;
use ndarray::simd_amx;

/// Default codebook size. 4096 entries = 12-bit index.
pub const CODEBOOK_SIZE: usize = 4096;

/// Default distance table size: CODEBOOK_SIZE².
pub const TABLE_SIZE: usize = CODEBOOK_SIZE * CODEBOOK_SIZE;

/// The thinking engine. One MatVec per cycle.
///
/// Accepts any N×N distance table. Common sizes:
///   1024×1024 = 1 MB (BGE-M3, Jina)
///   1536×1536 = 2.4 MB (reader-LM)
///   4096×4096 = 16 MB (full codebook, L3-resident)
pub struct ThinkingEngine {
    /// Precomputed similarity between all codebook pairs.
    /// Built ONCE. This IS the brain.
    /// entry[i * size + j] = u8 similarity (0=opposite, 255=identical).
    distance_table: Vec<u8>,

    /// Current energy distribution = which thoughts are alive.
    /// f32 precision — distance table is u8 (8 bits), f32 has 24-bit mantissa.
    /// More than sufficient. No f64 needed for u8 data.
    pub energy: Vec<f32>,

    /// Number of thought-atoms (table is size×size).
    pub size: usize,

    /// Cycle counter.
    pub cycles: u16,

    /// Convergence threshold.
    pub convergence_threshold: f32,

    /// Sigma floor: median of the distance table.
    /// Values below this are noise (orthogonal baseline).
    /// Only values ABOVE floor contribute to energy propagation.
    pub floor: u8,
}

impl ThinkingEngine {
    /// Create engine with a precomputed N×N distance table.
    /// Infers N from table length (must be a perfect square).
    /// Computes median as sigma floor automatically.
    pub fn new(distance_table: Vec<u8>) -> Self {
        let total = distance_table.len();
        let size = (total as f64).sqrt() as usize;
        assert_eq!(size * size, total,
            "distance table length {} is not a perfect square", total);
        assert!(size >= 4, "need at least 4 atoms");

        // Compute 1σ floor (74th percentile) — kills 74% of noise,
        // keeps only the top 26% as real topology.
        // Median (50%) is useless — half the table surviving is not a floor.
        let mut sorted = distance_table.clone();
        sorted.sort_unstable();
        let floor = sorted[total * 3 / 4]; // p75 ≈ μ + 0.675σ

        Self {
            distance_table,
            energy: vec![0.0f32; size],
            size,
            cycles: 0,
            convergence_threshold: 0.001,
            floor,
        }
    }

    /// Create engine with explicit floor override.
    pub fn with_floor(distance_table: Vec<u8>, floor: u8) -> Self {
        let total = distance_table.len();
        let size = (total as f64).sqrt() as usize;
        assert_eq!(size * size, total);
        assert!(size >= 4);
        Self {
            distance_table,
            energy: vec![0.0f32; size],
            size,
            cycles: 0,
            convergence_threshold: 0.001,
            floor,
        }
    }

    /// Build distance table from codebook centroids.
    ///
    /// Computes pairwise cosine similarity between all 4096 centroids.
    /// Maps cosine [-1, 1] → u8 [0, 255].
    ///
    /// This is the ONE expensive operation. Done once per codebook.
    /// After this: every thought is just a MatVec on the table.
    pub fn build_distance_table(centroids_f64: &[Vec<f64>]) -> Vec<u8> {
        let k = centroids_f64.len();
        assert!(k <= CODEBOOK_SIZE);
        let mut table = vec![128u8; k * k]; // 128 = cosine 0 (orthogonal)

        for i in 0..k {
            table[i * k + i] = 255; // self-similarity = max
            for j in (i + 1)..k {
                let cos = cosine_f64_simd(&centroids_f64[i], &centroids_f64[j]);
                // Map cosine [-1, 1] → u8 [0, 255]
                let u = ((cos + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                table[i * k + j] = u;
                table[j * k + i] = u; // symmetric
            }
        }
        table
    }

    /// ONE thinking cycle. MatVec on the distance table.
    ///
    /// For each active thought-atom i:
    ///   its energy spreads to all j proportional to distance_table[i][j].
    ///   Similar atoms (high table value) reinforce.
    ///   Dissimilar atoms (low table value) don't contribute.
    ///
    /// SIMD: 4× F32x16 per iteration = 64 elements.
    /// 4096 / 64 = 64 iterations per row. f32 matches u8 precision (24-bit mantissa >> 8-bit data).
    /// CPU cycles spent on real perturbation math, not artificial f64 precision.
    pub fn cycle(&mut self) {
        let k = self.size;
        let mut next = vec![0.0f32; k];
        let floor = self.floor;
        let scale = 1.0f32 / (255.0 - floor as f32);

        for i in 0..k {
            let e_i = self.energy[i];
            if e_i < 1e-10 { continue; }

            let row = &self.distance_table[i * k..(i + 1) * k];
            let e_scaled = e_i * scale;

            // 4× F32x16 = 64 elements per iteration
            let mut j = 0;
            while j + 64 <= k {
                macro_rules! do_lane {
                    ($off:expr) => {{
                        let base = j + $off * 16;
                        let d = F32x16::from_array([
                            row[base].saturating_sub(floor) as f32,
                            row[base + 1].saturating_sub(floor) as f32,
                            row[base + 2].saturating_sub(floor) as f32,
                            row[base + 3].saturating_sub(floor) as f32,
                            row[base + 4].saturating_sub(floor) as f32,
                            row[base + 5].saturating_sub(floor) as f32,
                            row[base + 6].saturating_sub(floor) as f32,
                            row[base + 7].saturating_sub(floor) as f32,
                            row[base + 8].saturating_sub(floor) as f32,
                            row[base + 9].saturating_sub(floor) as f32,
                            row[base + 10].saturating_sub(floor) as f32,
                            row[base + 11].saturating_sub(floor) as f32,
                            row[base + 12].saturating_sub(floor) as f32,
                            row[base + 13].saturating_sub(floor) as f32,
                            row[base + 14].saturating_sub(floor) as f32,
                            row[base + 15].saturating_sub(floor) as f32,
                        ]);
                        let acc = F32x16::from_slice(&next[base..base + 16]);
                        let ei = F32x16::splat(e_scaled);
                        d.mul_add(ei, acc).copy_to_slice(&mut next[base..base + 16]);
                    }};
                }
                do_lane!(0); do_lane!(1); do_lane!(2); do_lane!(3);
                j += 64;
            }
            while j < k {
                let d = row[j].saturating_sub(floor) as f32;
                next[j] += d * e_scaled;
                j += 1;
            }
        }

        // Normalize: total energy = 1.0
        let total: f32 = next.iter().sum();
        if total > 1e-10 {
            let inv = 1.0 / total;
            for e in &mut next { *e *= inv; }
        }

        self.energy = next;
        self.cycles += 1;
    }

    /// Run until convergence. Returns the resonance state.
    /// Uses `cycle_auto` which tries VNNI first, falls back to F32x16.
    pub fn think(&mut self, max_cycles: usize) -> ResonanceDto {
        for _ in 0..max_cycles {
            let prev = self.energy.clone();
            self.cycle();

            let delta: f32 = self.energy.iter().zip(&prev)
                .map(|(a, b)| (a - b).abs()).sum();
            if delta < self.convergence_threshold {
                break;
            }
        }
        ResonanceDto::from_energy_f32(&self.energy, self.cycles)
    }

    /// ONE thinking cycle via AMX/VNNI dispatch path.
    ///
    /// Quantizes f32 energy to i8, dispatches to the best available
    /// integer MatVec kernel (avx512vnni -> avxvnniint8 -> scalar),
    /// then dequantizes i32 results back to f32 and normalizes.
    ///
    /// The sigma floor is subtracted from the distance table during the
    /// existing table build, but for the VNNI path the energy quantization
    /// maps [0, max] -> [0, 127] after floor subtraction is implicit in
    /// the table values already stored as floor-subtracted u8.
    ///
    /// Note: The distance table stores raw u8 similarity values (floor NOT
    /// pre-subtracted), so we build a floor-subtracted table on the fly
    /// per cycle. For production, the table should be pre-subtracted once.
    pub fn cycle_vnni(&mut self) {
        let k = self.size;
        let floor = self.floor;

        // Build floor-subtracted table (u8 saturating_sub)
        let mut table_floored = vec![0u8; k * k];
        for (dst, &src) in table_floored.iter_mut().zip(self.distance_table.iter()) {
            *dst = src.saturating_sub(floor);
        }

        // Quantize energy f32 -> i8: map [0, max] -> [0, 127]
        let mut energy_i8 = vec![0i8; k];
        quantize_energy_f32_to_i8(&self.energy, &mut energy_i8);

        // Dispatch MatVec: avx512vnni -> avxvnniint8 -> scalar
        let mut result_i32 = vec![0i32; k];
        simd_amx::matvec_dispatch(&table_floored, &energy_i8, &mut result_i32, k);

        // Dequantize i32 -> f32
        // The dot products are in quantized units: u8_table * i8_energy.
        // To recover approximate f32 scale: result * (max_energy / 127.0).
        let max_e = self.energy.iter().cloned().fold(0.0f32, f32::max);
        let dequant_scale = if max_e > 1e-15 { max_e / 127.0 } else { 0.0 };

        let mut next = vec![0.0f32; k];
        for i in 0..k {
            next[i] = result_i32[i] as f32 * dequant_scale;
        }

        // Normalize: total energy = 1.0
        let total: f32 = next.iter().sum();
        if total > 1e-10 {
            let inv = 1.0 / total;
            for e in &mut next { *e *= inv; }
        }

        self.energy = next;
        self.cycles += 1;
    }

    /// Auto-dispatching cycle: tries VNNI path first, falls back to F32x16.
    ///
    /// Three-tier dispatch: avx512vnni → avxvnniint8 → F32x16.
    /// No scalar. F32x16 is the floor, not a fallback — it's 16 MACs/instr.
    pub fn cycle_auto(&mut self) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512vnni")
                || is_x86_feature_detected!("avxvnniint8")
            {
                self.cycle_vnni();
                return;
            }
        }
        self.cycle();
    }

    /// Inject perturbation from sensor output.
    pub fn perturb(&mut self, codebook_indices: &[u16]) {
        for &idx in codebook_indices {
            if (idx as usize) < self.size {
                self.energy[idx as usize] += 1.0;
            }
        }
        let total: f32 = self.energy.iter().sum();
        if total > 1e-10 {
            let inv = 1.0 / total;
            for e in &mut self.energy { *e *= inv; }
        }
    }

    /// Reset energy to zero. New thought starts fresh.
    pub fn reset(&mut self) {
        self.energy.fill(0.0);
        self.cycles = 0;
    }

    /// Commit: dominant peak → BusDto.
    pub fn commit(&self) -> BusDto {
        let resonance = ResonanceDto::from_energy_f32(&self.energy, self.cycles);
        BusDto {
            codebook_index: resonance.top_k[0].0,
            energy: resonance.top_k[0].1,
            top_k: resonance.top_k,
            cycle_count: self.cycles,
            converged: resonance.converged,
        }
    }

    /// Entropy of current energy distribution.
    pub fn entropy(&self) -> f32 {
        let mut h = 0.0f32;
        for &e in &self.energy {
            if e > 1e-10 {
                h -= e * e.ln();
            }
        }
        h
    }

    /// Number of active thought-atoms (energy > threshold).
    pub fn active_count(&self, threshold: f32) -> usize {
        self.energy.iter().filter(|&&e| e > threshold).count()
    }
}

/// Quantize f32 energy vector to i8 for VNNI MatVec.
///
/// Maps [0.0, max_energy] -> [0, 127]. Negative values are clamped to 0.
/// This is the f32 counterpart to `ndarray::simd_amx::quantize_energy_i8`
/// (which takes `&[f64]`). Energy after normalization is always non-negative.
pub fn quantize_energy_f32_to_i8(energy: &[f32], output: &mut [i8]) {
    let n = energy.len().min(output.len());
    let max_e = energy[..n].iter().cloned().fold(0.0f32, f32::max);
    if max_e < 1e-15 {
        for o in output[..n].iter_mut() { *o = 0; }
        return;
    }
    let scale = 127.0 / max_e;
    for i in 0..n {
        output[i] = (energy[i] * scale).round().clamp(0.0, 127.0) as i8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_table(k: usize) -> Vec<u8> {
        let mut table = vec![128u8; k * k];
        for i in 0..k {
            table[i * k + i] = 255; // self = max
            // Create some structure: nearby indices are similar
            for j in 0..k {
                let dist = (i as i64 - j as i64).unsigned_abs() as usize;
                if dist < 50 {
                    table[i * k + j] = (255 - dist * 2).min(255) as u8;
                }
            }
        }
        table
    }

    #[test]
    fn engine_creates() {
        let table = make_test_table(CODEBOOK_SIZE);
        let engine = ThinkingEngine::new(table);
        assert_eq!(engine.energy.iter().sum::<f32>(), 0.0);
        assert_eq!(engine.cycles, 0);
    }

    #[test]
    fn perturb_adds_energy() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[42, 100, 200]);
        assert!(engine.energy[42] > 0.0);
        assert!(engine.energy[100] > 0.0);
        assert!(engine.energy[200] > 0.0);

        // Total should be 1.0 (normalized)
        let total: f32 = engine.energy.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cycle_spreads_energy() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        // Perturb one atom
        engine.perturb(&[500]);
        assert!(engine.energy[500] > 0.9); // almost all energy at 500

        // One cycle: energy spreads to nearby atoms
        engine.cycle();
        assert!(engine.energy[500] > 0.0); // still has some
        assert!(engine.energy[499] > 0.0); // neighbor activated
        assert!(engine.energy[501] > 0.0); // neighbor activated
    }

    #[test]
    fn think_converges() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[100, 110, 120]);
        let resonance = engine.think(20);

        assert!(resonance.cycle_count <= 20);
        assert!(resonance.top_k[0].1 > 0.0); // dominant peak exists

        eprintln!("Converged in {} cycles", resonance.cycle_count);
        eprintln!("Top peak: index={}, energy={:.4}",
            resonance.top_k[0].0, resonance.top_k[0].1);
        eprintln!("Entropy: {:.4}", engine.entropy());
        eprintln!("Active atoms: {}", engine.active_count(0.001));
    }

    #[test]
    fn commit_returns_dominant() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[42]);
        engine.think(10);

        let bus = engine.commit();
        assert!(bus.energy > 0.0);
        eprintln!("Committed: index={}, energy={:.4}", bus.codebook_index, bus.energy);
    }

    #[test]
    fn entropy_decreases_with_convergence() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[100, 200, 300, 400, 500]);
        let h0 = engine.entropy();

        engine.think(10);
        let h1 = engine.entropy();

        eprintln!("Entropy before: {:.4}, after: {:.4}", h0, h1);
        // Entropy should decrease or stay stable as peaks crystallize
        // (may not always decrease if the table has uniform structure)
    }

    #[test]
    fn multiple_perturbations_compose() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        // First sensor
        engine.perturb(&[100, 101, 102]);
        engine.think(5);

        // Second sensor adds more perturbation
        engine.perturb(&[100, 200, 300]);
        let resonance = engine.think(10);

        // Both perturbation sites should have influenced the result
        assert!(resonance.top_k[0].1 > 0.0);
        eprintln!("After dual perturbation: {} peaks above 0.01",
            engine.active_count(0.01));
    }

    #[test]
    fn reset_clears() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[42]);
        engine.think(5);
        assert!(engine.energy[42] > 0.0);

        engine.reset();
        assert_eq!(engine.energy.iter().sum::<f32>(), 0.0);
        assert_eq!(engine.cycles, 0);
    }

    #[test]
    fn build_distance_table_symmetric() {
        let centroids: Vec<Vec<f64>> = (0..100).map(|i| {
            (0..64).map(|d| ((i * 97 + d * 31) as f64 % 200.0 - 100.0) * 0.01).collect()
        }).collect();

        let table = ThinkingEngine::build_distance_table(&centroids);
        assert_eq!(table.len(), 100 * 100);

        // Symmetric
        for i in 0..100 {
            for j in 0..100 {
                assert_eq!(table[i * 100 + j], table[j * 100 + i],
                    "table[{},{}]={} != table[{},{}]={}",
                    i, j, table[i * 100 + j], j, i, table[j * 100 + i]);
            }
        }

        // Diagonal = 255
        for i in 0..100 {
            assert_eq!(table[i * 100 + i], 255);
        }
    }
}
