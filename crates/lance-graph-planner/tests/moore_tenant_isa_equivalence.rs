//! Test J: ISA observational equivalence through the CANONICAL tenant.
//!
//! #1406 measured this with a probe-local lane type. This test runs the same
//! check through the shipped contract surface: eight CE64 edges are lifted
//! into `ValueTenant::MooreNars16` with `MooreTenantMut::lift_ce64`, read back
//! with `MooreTenantView`, and rebuilt as `payload40 | ce64_upper(witness)`.
//! The shipped ops (`forward` both ways, `learn` both ways, `syllogize`) must
//! give the same result on the rebuilt edge as on the original, on every
//! field except the S/P/O sign triple, which the Moore reading does not carry.

use causal_edge::edge::CausalEdge64;
use lance_graph_contract::canonical_node::VALUE_SLAB_LEN;
use lance_graph_contract::moore_tenant::{MooreSlot, MooreTenantMut, MooreTenantView};

const DIRECTION: u64 = 0b111 << 43;
const W: u64 = 0x3F << 53;
const EPI5: u64 = 0x1F << 59;
const PAYLOAD40: u64 = (1 << 40) - 1;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn canonical(r: &mut Rng, w: u64) -> u64 {
    (r.next() & !W & !EPI5) | (w << 53) | ((r.next() % 24) << 59)
}

fn tables() -> [Box<[u8; 256 * 256]>; 3] {
    let mk = |a: usize, b: usize| {
        let mut t = vec![0u8; 256 * 256].into_boxed_slice();
        for x in 0..256 {
            for y in 0..256 {
                t[x * 256 + y] = (x * a + y * b) as u8;
            }
        }
        t.try_into().expect("256*256")
    };
    [mk(31, 17), mk(7, 13), mk(11, 29)]
}

/// Results of the five shipped ops with `x` in each operand position.
fn ops(x: u64, y: u64, t: &[Box<[u8; 256 * 256]>; 3]) -> [u64; 5] {
    let (x, y) = (CausalEdge64(x), CausalEdge64(y));
    let mut lx = x;
    lx.learn(y, 0);
    let mut ly = y;
    ly.learn(x, 0);
    [
        x.forward(y, &t[0], &t[1], &t[2]).0,
        y.forward(x, &t[0], &t[1], &t[2]).0,
        lx.0,
        ly.0,
        x.syllogize(y).expect("chain figure").conclusion.0,
    ]
}

/// Count divergent (lane, op) results over `rounds` random tenants when the
/// rebuilt edge uses `witness_of(true_witness)`.
fn divergences(rounds: usize, witness_of: fn(u8) -> u8) -> usize {
    let t = tables();
    let mut r = Rng(0x1406);
    let mut diff = 0;
    for _ in 0..rounds {
        let w = r.next() % 64;
        let edges: [u64; 8] = std::array::from_fn(|_| canonical(&mut r, w));
        let pol: [bool; 8] = std::array::from_fn(|_| r.next() & 1 == 1);
        let mut slab = [0u8; VALUE_SLAB_LEN];
        let got = MooreTenantMut::new(&mut slab)
            .lift_ce64(&edges, pol)
            .expect("homogeneous");
        assert_eq!(u64::from(got), w);
        let view = MooreTenantView::new(&slab);
        for slot in MooreSlot::ALL {
            let k = slot.index();
            let lane = view.moore_nars16(slot);
            assert_eq!(lane.polarity(), pol[k]);
            let rebuilt = (edges[k] & PAYLOAD40) | lane.ce64_upper(witness_of(got));
            // A partner whose S equals this lane's O, so syllogize chains.
            let pw = r.next() % 64;
            let p = canonical(&mut r, pw);
            let partner = (p & !0xFF) | (edges[k] >> 16 & 0xFF);
            let a = ops(edges[k], partner, &t);
            let b = ops(rebuilt, partner, &t);
            for i in 0..5 {
                if a[i] & !DIRECTION != b[i] & !DIRECTION {
                    diff += 1;
                }
            }
        }
    }
    diff
}

#[test]
fn j_tenant_lanes_are_isa_observationally_equivalent() {
    assert_eq!(divergences(256, |w| w), 0);
}

/// Anti-vacuity: restoring the wrong witness is visible to the ISA, so the
/// equivalence above is not true of every rebuild.
#[test]
fn j_wrong_witness_is_observable() {
    assert!(divergences(64, |w| (w + 1) & 63) > 0);
}
