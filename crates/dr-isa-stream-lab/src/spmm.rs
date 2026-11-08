//! Neighbor aggregation over the virtual CSR, as BF16 GEMMs.
//!
//! `Y[p][i] = sum over j in row(i) of lane_p[j]` (plus-times). Per nonempty
//! block (row tile `tr`, column tile `tc`):
//!
//! * A is the 16 demanded lanes over column tile `tc`. In [`Mode::Direct`]
//!   it is read in place from the slab at the slab pitch, exactly as in
//!   [`crate::gemm`]; in [`Mode::Staged`] it is copied into a dense tile.
//! * B is the block's adjacency, `B[c][r] = edge(row r, col c)`, staged from
//!   the bitmask: in VNNI layout 16 rows at a time for the tile GEMM, or as a
//!   row-major 256-column tile for MKL. Row groups with no edges are skipped.
//!
//! This is the plus-times semiring. blasgraph's `CsrStorage` carries `BitVec`
//! values under HDR semirings (XorBundle, HammingMin, ...); those are not
//! realized by a floating-point GEMM, and this module does not claim them.

use lance_graph_dt64::{LaneBinding, LaneRefusal};
use ndarray::simd::{
    bf16_tile_gemm_16x16_packed, bf16_tile_gemm_16x16_packed_strided, bf16_tile_gemm_tier,
    PackedBf16B,
};

use crate::bind::World;
use crate::cube::{ArmRefusal, TILE};
use crate::fixture::{Fixture, BLOCK, ROWS};
use crate::gemm::{geometry, GemmRefusal, Geometry, Mode, ONE};
use crate::source::{SlabLane, Socket};
use crate::vcsr::{Block, ExplicitCsr, VirtualCsr};

/// What one aggregation did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpmmReport {
    pub tier: &'static str,
    /// Nonempty blocks visited.
    pub blocks: usize,
    /// GEMM calls.
    pub gemms: usize,
    /// Bytes of A copied into staging tiles.
    pub a_staged_bytes: usize,
    /// Nonzero B entries written.
    pub b_entries: usize,
}

/// `Y[p][i]` for the listed properties, from the fixture and the explicit
/// CSR alone.
pub fn oracle(fx: &Fixture, csr: &ExplicitCsr, props: &[usize]) -> Vec<Vec<i64>> {
    props
        .iter()
        .map(|&p| {
            (0..fx.n)
                .map(|i| csr.row(i).iter().map(|&j| i64::from(fx.value(j, p))).sum())
                .collect()
        })
        .collect()
}

fn check_graph(binding: &LaneBinding, g: &VirtualCsr) -> Result<(), GemmRefusal> {
    if g.space == binding.selection() {
        Ok(())
    } else {
        Err(GemmRefusal::Arm(ArmRefusal::GraphSpace {
            graph: g.space,
            selection: binding.selection(),
        }))
    }
}

fn borrow<'s>(
    world: &World,
    binding: &LaneBinding,
    socket: &'s Socket<'_>,
    fields: &[u8],
) -> Result<Vec<SlabLane<'s>>, LaneRefusal> {
    fields
        .iter()
        .map(|&f| binding.lane(&world.store, &world.view, socket.lanes.frames(), socket, f))
        .collect()
}

/// K for column tile `tc`: 256, or the padded remainder of the last tile.
fn k_of(n: usize, tc: usize) -> usize {
    (n.div_ceil(BLOCK) * BLOCK - tc * TILE).min(TILE)
}

/// Stage block rows `g*16..g*16+16` into `b` as `B[c][r]`; returns the
/// number of entries written.
fn stage_rows(block: &Block, g: usize, k: usize, b: &mut PackedBf16B, value: u16) -> usize {
    let mut written = 0;
    for r in 0..ROWS {
        for (w, &word) in block.row_words(g * ROWS + r).iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let c = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if c < k {
                    b.data_mut()[PackedBf16B::vnni_index(c, r)] = value;
                    written += 1;
                }
            }
        }
    }
    written
}

/// Aggregate 16 demanded lanes (or fewer, staged) over the virtual CSR with
/// the ndarray BF16 tile GEMM.
pub fn aggregate(
    world: &World,
    binding: &LaneBinding,
    socket: &Socket<'_>,
    g: &VirtualCsr,
    fields: &[u8],
    mode: Mode,
) -> Result<(Vec<Vec<f32>>, SpmmReport), GemmRefusal> {
    check_graph(binding, g)?;
    let lanes = borrow(world, binding, socket, fields)?;
    let geo: Option<Geometry<'_>> = match mode {
        Mode::Direct => Some(geometry(fields, &lanes).map_err(GemmRefusal::Direct)?),
        Mode::Staged => None,
    };
    let mut report = SpmmReport {
        tier: bf16_tile_gemm_tier(),
        ..SpmmReport::default()
    };
    let mut y = vec![vec![0f32; g.n]; fields.len()];
    let mut b_full = PackedBf16B::zeroed(TILE);
    let mut b_tail = PackedBf16B::zeroed(k_of(g.n, g.row_tiles().saturating_sub(1)).max(BLOCK));
    let mut a = vec![0u16; ROWS * TILE];
    let mut c = vec![[0f32; ROWS * ROWS]; TILE / ROWS];
    for tr in 0..g.row_tiles() {
        c.iter_mut().for_each(|t| t.fill(0.0));
        for block in g.row_tile(tr) {
            report.blocks += 1;
            let k = k_of(g.n, block.tc);
            let b = if k == TILE { &mut b_full } else { &mut b_tail };
            if geo.is_none() {
                a[..ROWS * k].fill(0);
                for (r, lane) in lanes.iter().enumerate() {
                    let v = lane.values();
                    let lo = block.tc * TILE;
                    let hi = (lo + k).min(v.len());
                    a[r * k..r * k + (hi - lo)].copy_from_slice(&v[lo..hi]);
                    report.a_staged_bytes += (hi - lo) * 2;
                }
            }
            for (grp, acc) in c.iter_mut().enumerate() {
                if !block.any_in_rows(grp * ROWS..(grp + 1) * ROWS) {
                    continue;
                }
                report.b_entries += stage_rows(block, grp, k, b, ONE);
                match &geo {
                    Some(geo) => {
                        let start = geo.first_row * geo.pitch + block.tc * TILE;
                        bf16_tile_gemm_16x16_packed_strided(&geo.slab[start..], geo.pitch, b, acc);
                    }
                    None => bf16_tile_gemm_16x16_packed(&a[..ROWS * k], b, acc),
                }
                report.gemms += 1;
                stage_rows(block, grp, k, b, 0);
            }
        }
        for (grp, acc) in c.iter().enumerate() {
            for r in 0..ROWS {
                let i = tr * TILE + grp * ROWS + r;
                if i < g.n {
                    for (p, yp) in y.iter_mut().enumerate() {
                        yp[i] = acc[p * ROWS + r];
                    }
                }
            }
        }
    }
    Ok((y, report))
}

/// The same aggregation through MKL `cblas_gemm_bf16bf16f32`: one call per
/// nonempty block (M = 16 lanes, N = 256 rows, K = the block's columns), A a
/// strided `BF16` view of the slab (in place, `lda` = pitch), B the block
/// expanded to row-major bf16. Needs the in-place geometry.
#[cfg(feature = "mkl")]
pub fn aggregate_mkl(
    world: &World,
    binding: &LaneBinding,
    socket: &Socket<'_>,
    g: &VirtualCsr,
    fields: &[u8],
) -> Result<(Vec<Vec<f32>>, SpmmReport), GemmRefusal> {
    use ndarray::hpc::quantized::BF16;
    use ndarray::{Array2, ArrayView2, ShapeBuilder};

    check_graph(binding, g)?;
    let lanes = borrow(world, binding, socket, fields)?;
    let geo = geometry(fields, &lanes).map_err(GemmRefusal::Direct)?;
    let mut report = SpmmReport {
        tier: "MKL cblas_gemm_bf16bf16f32",
        ..SpmmReport::default()
    };
    let slab = BF16::slice_from_bits(geo.slab);
    let mut y = vec![vec![0f32; g.n]; fields.len()];
    let mut b = Array2::from_elem((TILE, TILE), BF16::ZERO);
    let mut c = Array2::<f32>::zeros((ROWS, TILE));
    for tr in 0..g.row_tiles() {
        c.fill(0.0);
        for block in g.row_tile(tr) {
            report.blocks += 1;
            let k = k_of(g.n, block.tc);
            let start = geo.first_row * geo.pitch + block.tc * TILE;
            let a = ArrayView2::from_shape((ROWS, k).strides((geo.pitch, 1)), &slab[start..])
                .expect("the geometry check covers 15 * pitch + k");
            for r in 0..TILE {
                for (w, &word) in block.row_words(r).iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let col = w * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        b[(col, r)] = BF16::ONE;
                        report.b_entries += 1;
                    }
                }
            }
            ndarray::backend::mkl::sgemm_bf16(
                a,
                b.slice(ndarray::s![..k, ..]),
                c.view_mut(),
                1.0,
                1.0,
            )
            .expect("shapes agree by construction");
            report.gemms += 1;
            for r in 0..TILE {
                for (w, &word) in block.row_words(r).iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        b[(w * 64 + bits.trailing_zeros() as usize, r)] = BF16::ZERO;
                        bits &= bits - 1;
                    }
                }
            }
        }
        for r in 0..TILE {
            let i = tr * TILE + r;
            if i < g.n {
                for (p, yp) in y.iter_mut().enumerate() {
                    yp[i] = c[(p, r)];
                }
            }
        }
    }
    Ok((y, report))
}
