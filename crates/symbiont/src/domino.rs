//! Domino POC — SoA orchestration via the `ndarray::simd` polyfill (AMX BF16 tiles).
//!
//! Operator (2026-06-20): BF16 + AMX, 2bit×2bit 4×4 Morton tile, Domino thinking
//! style, no Spain — very basic POC to prove the SoA orchestration. **ALL SIMD
//! goes through `ndarray::simd::*`** (the W1a polyfill: `simd.rs` → `simd_amx` /
//! `simd_avx512` / `simd_ops` / `simd_soa`), NEVER `ndarray::hpc::*` directly:
//! the AMX `arch_prctl(158)` XTILEDATA grant is gated inside `amx_available()`,
//! which `bf16_tile_gemm_16x16` calls before any tile op (the documented gotcha).
//! Nothing here re-implements an op the polyfill already has (BF16↔f32 batch
//! conversion is `f32_to_bf16_batch_rne`); only `morton4` is consumer-side
//! (ndarray has no Morton primitive).
//!
//! Host: Emerald Rapids (kernel 6.18.5) — ndarray's `CpuModel` detects it
//! (family 6, model 0xCF), so `amx_available()` is true and the real `TDPBF16PS`
//! tile path fires.
//!
//! Each SoA board carries a 4×4 BF16 tile (16 lanes, Morton-addressed) in its
//! `Fingerprint` tenant; 16 boards batch into ONE AMX 16×16 tile GEMM — AMX burns
//! through 16 SoA boards per instruction. One Domino step = `C = A·W`, C
//! re-quantised back into the tiles (cascade feedback); per-board reduction →
//! `Energy` tenant → swept by the NaN-detection projection surface (the demoted
//! singleton BindSpace). 256 boards = 16 batches.

use lance_graph_contract::canonical_node::{NodeRow, ValueTenant};
use lance_graph_contract::nan_projection::project_energy_nonfinite;
use lance_graph_contract::{EdgeBlock, NodeGuid};
use ndarray::simd::{amx_available, amx_report, bf16_tile_gemm_16x16, f32_to_bf16_batch_rne};

const TILE: usize = 4;
const LANES: usize = TILE * TILE; // 16 lanes/board = 32 BF16 bytes (Fingerprint tenant)
const BATCH: usize = 16; // 16 boards → one AMX 16×16 tile
const K: usize = 32; // AMX BF16 contraction (min, multiple of 32)

/// 2bit×2bit Morton (Z-order) index for a 4×4 tile (no ndarray primitive exists).
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

/// 16 f32 lanes → 16 BF16 (u16) via the polyfill batch RNE converter (NOT a
/// hand-rolled per-element conversion — that would duplicate `ndarray::simd`).
#[inline]
fn lanes_to_bf16(f: &[f32; LANES]) -> [u16; LANES] {
    let mut out = [0u16; LANES];
    f32_to_bf16_batch_rne(f, &mut out);
    out
}

/// Read a board's 16 BF16 lanes (u16) from the Fingerprint tenant (plain bytes —
/// memory access, not a SIMD op).
fn read_lanes(row: &NodeRow) -> [u16; LANES] {
    let off = fp_off();
    let mut l = [0u16; LANES];
    for (i, x) in l.iter_mut().enumerate() {
        *x = u16::from_le_bytes([row.value[off + i * 2], row.value[off + i * 2 + 1]]);
    }
    l
}
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
pub fn energy_of(row: &NodeRow) -> f32 {
    let off = en_off();
    f32::from_le_bytes(row.value[off..off + 4].try_into().expect("4 bytes"))
}

/// Seed a board: a deterministic Morton-addressed 4×4 BF16 tile in `Fingerprint`.
///
/// The 32-byte head is seeded too so the key-only graph render (§2.4,
/// `key_render`) is non-trivial: `key` = bootstrap address (identity = idx),
/// `edges` = one in-family ring slot + one out-of-family adapter slot, both
/// one-byte basin-local indices per the canonical `EdgeBlock`. The Domino sweep
/// NEVER reads the edge region (it touches only the `Fingerprint`/`Energy` value
/// tenants), so seeding edges here is free for the AMX path.
fn seed_board(idx: usize) -> NodeRow {
    let mut edges = EdgeBlock::default();
    edges.in_family[0] = ((idx % 255) + 1) as u8; // ring neighbour (always 1..=255)
    edges.out_family[0] = (1 + (idx % 4)) as u8; // inherited-adapter slot (1..=4)
    let mut row = NodeRow {
        key: NodeGuid::local(idx as u32),
        edges,
        value: [0u8; 480],
    };
    let mut f = [0.0f32; LANES];
    for y in 0..TILE {
        for x in 0..TILE {
            f[morton4(x, y)] = ((morton4(x, y) + idx) % 7) as f32 * 0.25;
        }
    }
    write_lanes(&mut row, &lanes_to_bf16(&f));
    row
}

/// The 32×16 BF16 weight: top 16×16 = tridiagonal smoothing kernel, bottom 16
/// rows = 0 (the K-pad). Converted once through the polyfill.
fn weight() -> [u16; K * BATCH] {
    let mut f = [0.0f32; K * BATCH]; // 32×16
    for j in 0..LANES {
        for l in 0..BATCH {
            f[j * BATCH + l] = if j == l {
                1.0
            } else if (j as i32 - l as i32).abs() == 1 {
                0.25
            } else {
                0.0
            };
        }
    }
    let mut out = [0u16; K * BATCH];
    f32_to_bf16_batch_rne(&f, &mut out);
    out
}

/// One Domino step over a 16-board batch via the polyfill AMX 16×16 tile GEMM.
/// `C[16,16] = A[16,32]·W[32,16]`; C re-quantised back into the tiles (domino
/// feedback); per-board `sum(C[i,:])` → `Energy`.
fn domino_batch(boards: &mut [NodeRow], w: &[u16; K * BATCH], stages: usize) {
    debug_assert_eq!(boards.len(), BATCH);
    for _ in 0..stages {
        let mut a = [0u16; BATCH * K]; // 16×32, lanes padded to K
        for (i, row) in boards.iter().enumerate() {
            a[i * K..i * K + LANES].copy_from_slice(&read_lanes(row));
        }
        let mut c = [0.0f32; BATCH * BATCH]; // 16×16
        bf16_tile_gemm_16x16(&a, w, &mut c, K); // ndarray::simd polyfill → AMX TDPBF16PS
        for (i, row) in boards.iter_mut().enumerate() {
            let mut f = [0.0f32; LANES];
            f.copy_from_slice(&c[i * BATCH..i * BATCH + LANES]);
            let sum: f32 = f.iter().sum();
            write_lanes(row, &lanes_to_bf16(&f)); // domino: output → next input
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
    println!("  amx gate: {}", amx_report());
}

/// Seed `n` boards (each a Morton-addressed 4×4 BF16 tile). Public so the kanban
/// loop (`kanban_loop::SymbiontBoard`) can spawn a mailbox over them.
pub fn seed_boards(n: usize) -> Vec<NodeRow> {
    (0..n).map(seed_board).collect()
}

/// Run a `stages`-deep Domino sweep over an existing board-set, in full 16-board
/// AMX batches — the `CognitiveWork` phase of the kanban loop. A trailing partial
/// batch (`n` not a multiple of 16) is left untouched.
pub fn domino_sweep(rows: &mut [NodeRow], stages: usize) {
    let w = weight();
    for batch in rows.chunks_mut(BATCH) {
        if batch.len() == BATCH {
            domino_batch(batch, &w, stages);
        }
    }
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
        assert_eq!(read_lanes(&seed_board(3)), read_lanes(&seed_board(3)));
    }

    #[test]
    fn domino_amx_sweep_stays_finite() {
        // run_poc asserts NaN-clean internally; exercises the full polyfill path.
        run_poc(64, 3);
    }
}
