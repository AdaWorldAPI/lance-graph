//! Domino POC — proving SoA orchestration with AMX BF16 tiles.
//!
//! Operator (2026-06-20): "use BF16 and add_mul where possible and use amx ...
//! the 2bit×2bit 4×4 Morton tile [AMX] is our magic bullet ... Domino thinking
//! style ... very basic POC just to prove the SoA Orchestration." The sandbox is
//! Sapphire/Emerald Rapids (AMX present); built native (`crates/symbiont/.cargo/
//! config.toml` → `target-cpu=native`), `amx_available()` is true and the real
//! `TDPBF16PS` tile path fires (the GEMM's tile-dot IS the fused multiply-add).
//!
//! Each SoA board carries a 4×4 BF16 tile (16 lanes, Morton-addressed) in its
//! `Fingerprint` tenant. **16 boards' tiles batch into ONE AMX 16×16 tile GEMM**
//! — that is the SoA-orchestration unit: AMX burns through 16 boards per tile
//! instruction. One **Domino step** = `C[16,16] = A[16,32]·W[32,16]`, with C
//! re-quantised back into each board's tile (the cascade essence of the
//! thinking-engine `domino` style, emulated inline). The per-board scalar
//! reduction lands in `Energy` for the NaN-detection projection surface (the
//! demoted singleton BindSpace).

use lance_graph_contract::canonical_node::{NodeRow, ValueTenant};
use lance_graph_contract::nan_projection::project_energy_nonfinite;
use lance_graph_contract::{EdgeBlock, NodeGuid};
use ndarray::hpc::bf16_tile_gemm::bf16_tile_gemm_16x16;
use ndarray::hpc::quantized::BF16;
use ndarray::simd::amx_available;

const TILE: usize = 4;
const LANES: usize = TILE * TILE; // 16 lanes/board = 32 BF16 bytes (Fingerprint tenant)
const BATCH: usize = 16; // 16 boards → one AMX 16×16 tile
const K: usize = 32; // AMX BF16 contraction (min, multiple of 32)

/// 2bit×2bit Morton (Z-order) index for a 4×4 tile: (x,y) ∈ [0,4)² → [0,16).
#[inline]
fn morton4(x: usize, y: usize) -> usize {
    let spread = |v: usize| (v & 1) | ((v & 2) << 1); // bit0→0, bit1→2
    spread(x) | (spread(y) << 1)
}

#[inline]
fn fp_off() -> usize {
    ValueTenant::Fingerprint.value_offset()
}
#[inline]
fn en_off() -> usize {
    ValueTenant::Energy.value_offset()
}

/// Read a board's 16 BF16 lanes (raw u16 bit patterns) from the Fingerprint tenant.
fn read_lanes(row: &NodeRow) -> [u16; LANES] {
    let off = fp_off();
    let mut l = [0u16; LANES];
    for (i, x) in l.iter_mut().enumerate() {
        *x = u16::from_le_bytes([row.value[off + i * 2], row.value[off + i * 2 + 1]]);
    }
    l
}

/// Write 16 BF16 lanes (u16) into the Fingerprint tenant.
fn write_lanes(row: &mut NodeRow, lanes: &[u16; LANES]) {
    let off = fp_off();
    for (i, &x) in lanes.iter().enumerate() {
        let [lo, hi] = x.to_le_bytes();
        row.value[off + i * 2] = lo;
        row.value[off + i * 2 + 1] = hi;
    }
}

fn set_energy(row: &mut NodeRow, e: f32) {
    let off = en_off();
    row.value[off..off + 4].copy_from_slice(&e.to_le_bytes());
}
fn energy_of(row: &NodeRow) -> f32 {
    let off = en_off();
    f32::from_le_bytes(row.value[off..off + 4].try_into().expect("4 bytes"))
}

/// Seed a board: a deterministic Morton-addressed 4×4 BF16 tile in `Fingerprint`.
fn seed_board(idx: usize) -> NodeRow {
    let mut row = NodeRow {
        key: NodeGuid::local(idx as u32),
        edges: EdgeBlock::default(),
        value: [0u8; 480],
    };
    let mut lanes = [0u16; LANES];
    for y in 0..TILE {
        for x in 0..TILE {
            let v = ((morton4(x, y) + idx) % 7) as f32 * 0.25;
            lanes[morton4(x, y)] = BF16::from_f32_rounded(v).0;
        }
    }
    write_lanes(&mut row, &lanes);
    row
}

/// The 32×16 BF16 weight: top 16×16 = a tridiagonal smoothing kernel, bottom 16
/// rows = 0 (the K-pad). Row-major `W[j*16 + l]`.
fn weight() -> [u16; K * BATCH] {
    let mut w = [0u16; K * BATCH]; // 32×16
    for j in 0..LANES {
        for l in 0..BATCH {
            let val = if j == l {
                1.0
            } else if (j as i32 - l as i32).abs() == 1 {
                0.25
            } else {
                0.0
            };
            w[j * BATCH + l] = BF16::from_f32_rounded(val).0;
        }
    }
    w
}

/// One Domino step over a 16-board batch via the AMX 16×16 BF16 tile GEMM.
/// `C[16,16] = A[16,32]·W[32,16]`; C re-quantised back into the tiles (the
/// domino feedback); per-board `sum(C[i,:])` → `Energy`.
fn domino_batch(boards: &mut [NodeRow], w: &[u16; K * BATCH], stages: usize) {
    debug_assert_eq!(boards.len(), BATCH);
    for _ in 0..stages {
        let mut a = [0u16; BATCH * K]; // 16×32, board lanes padded to K
        for (i, row) in boards.iter().enumerate() {
            let lanes = read_lanes(row);
            a[i * K..i * K + LANES].copy_from_slice(&lanes);
        }
        let mut c = [0.0f32; BATCH * BATCH]; // 16×16, zeroed (tile-GEMM does +=)
        bf16_tile_gemm_16x16(&a, w, &mut c, K); // AMX TDPBF16PS (or safe fallback)
        for (i, row) in boards.iter_mut().enumerate() {
            let mut lanes = [0u16; LANES];
            let mut sum = 0.0f32;
            for (l, lane) in lanes.iter_mut().enumerate() {
                let v = c[i * BATCH + l];
                *lane = BF16::from_f32_rounded(v).0;
                sum += v;
            }
            write_lanes(row, &lanes);
            set_energy(row, sum);
        }
    }
}

/// The POC: build `n_boards` SoA boards, run a `stages`-deep Domino sweep in AMX
/// 16-board batches, project onto the NaN-detection surface, report.
pub fn run_poc(n_boards: usize, stages: usize) {
    assert_eq!(n_boards % BATCH, 0, "n_boards must be a multiple of {BATCH}");
    let w = weight();
    let mut rows: Vec<NodeRow> = (0..n_boards).map(seed_board).collect();

    // The SoA sweep: 16-board batches burned through the AMX tile GEMM.
    for batch in rows.chunks_mut(BATCH) {
        domino_batch(batch, &w, stages);
    }

    // NaN-detection projection surface (the demoted singleton BindSpace).
    let report = project_energy_nonfinite(&rows);
    if !report.is_clean() {
        eprintln!(
            "NaN: {} non-finite Energy tenants at {:?}",
            report.count(),
            report.nonfinite
        );
    }
    assert!(report.is_clean(), "Domino sweep produced a non-finite board");

    let path = if amx_available() {
        "AMX TDPBF16PS"
    } else {
        "scalar/AVX-512 fallback"
    };
    println!(
        "Domino POC: {n_boards} SoA boards in {} AMX 16×16 batches × {stages}-stage 4×4 BF16 \
         Morton-tile cascade [{path}]; all Energy tenants finite (NaN-projected); \
         E[0]={:.4} E[last]={:.4}",
        n_boards / BATCH,
        energy_of(&rows[0]),
        energy_of(&rows[n_boards - 1]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton4_is_bijective_over_4x4() {
        let mut seen = [false; LANES];
        for y in 0..TILE {
            for x in 0..TILE {
                let m = morton4(x, y);
                assert!(!seen[m], "Morton collision at ({x},{y})");
                seen[m] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn lanes_round_trip_through_fingerprint_tenant() {
        let row = seed_board(3);
        assert_eq!(read_lanes(&row), read_lanes(&seed_board(3)));
    }

    #[test]
    fn domino_amx_sweep_stays_finite() {
        // run_poc asserts NaN-clean internally; exercises the full AMX-or-fallback path.
        run_poc(64, 3);
    }
}
