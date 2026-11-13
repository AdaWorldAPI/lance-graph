//! D-VAS-C4 / C6: a contract `SetExpr` lowered to a Quack `Filter`.
//!
//! Three independent answers per expression:
//! - a scalar evaluator written here from the definition (two-valued and
//!   Kleene), which knows nothing of `normalize` or of Quack;
//! - Quack: `lower_set` -> `Filter` -> `lower` and `lower_fused` -> mask-risc;
//! - the existing library lowering `set_lower::SetPlan` (D-CMR-2), compiled
//!   into this test binary from its own source file, because the crate that
//!   owns it also pulls the storage stack.

use lance_graph_contract::set_algebra::{Cmp, Logic, Res, SetExpr};
use lance_graph_mask_risc::{execute_into, words_for, Foreign, LaneRef, Out, Planes, Scratch};
use lance_graph_quack::set::{lower_set, SetLowerError, SetLowered};
use lance_graph_quack::{lower, lower_fused, Agg, Filter, Mask, Query};

#[allow(dead_code)]
#[path = "../../lance-graph/src/fold/set_lower.rs"]
mod set_lower;

const N: usize = 197; // not a word multiple
const SYMS: u32 = 3;

struct World {
    i32_lanes: [Vec<i32>; 2],
    u32_lanes: [Vec<u32>; 2],
    syms: Vec<Vec<u64>>,
    /// Validity of lanes 0..4 (`I32` 0, 1 then `U32` 2, 3): bit set = present.
    valid: Vec<Vec<u64>>,
}

struct Gen(u64);
impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn plane(&mut self, one_in: u64, set: bool) -> Vec<u64> {
        let mut w = vec![0u64; words_for(N)];
        for r in 0..N {
            if self.next().is_multiple_of(one_in) == set {
                w[r / 64] |= 1 << (r % 64);
            }
        }
        w
    }
    fn leaf(&mut self) -> SetExpr {
        const OPS: [Cmp; 6] = [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq, Cmp::Ne];
        match self.next() % 5 {
            0 => SetExpr::Sym((self.next() % u64::from(SYMS)) as u32),
            1 | 2 => SetExpr::i32(
                (self.next() % 2) as u16,
                OPS[(self.next() % 6) as usize],
                (self.next() % 13) as i32 - 6,
            ),
            _ => SetExpr::Res(Res::U32 {
                lane: 2 + (self.next() % 2) as u16,
                eq: self.next().is_multiple_of(2),
                v: (self.next() % 4) as u32,
            }),
        }
    }
    fn expr(&mut self, depth: u32) -> SetExpr {
        if depth == 0 || self.next().is_multiple_of(4) {
            return match self.next() % 14 {
                0 => SetExpr::All,
                1 => SetExpr::Empty,
                _ => self.leaf(),
            };
        }
        match self.next() % 3 {
            0 => SetExpr::negated(self.expr(depth - 1)),
            k => {
                let n = 1 + self.next() % 4;
                let cs = (0..n).map(|_| self.expr(depth - 1)).collect();
                if k == 1 {
                    SetExpr::And(cs)
                } else {
                    SetExpr::Or(cs)
                }
            }
        }
    }
}

fn world(seed: u64) -> World {
    let mut g = Gen(seed);
    World {
        i32_lanes: [
            (0..N).map(|_| (g.next() % 11) as i32 - 5).collect(),
            (0..N).map(|_| (g.next() % 7) as i32).collect(),
        ],
        u32_lanes: [
            (0..N).map(|_| (g.next() % 4) as u32).collect(),
            (0..N).map(|_| (g.next() % 3) as u32).collect(),
        ],
        syms: (0..SYMS).map(|_| g.plane(3, true)).collect(),
        // About one row in four is NULL, per lane.
        valid: (0..4).map(|_| g.plane(4, false)).collect(),
    }
}

fn bit(w: &[u64], r: usize) -> bool {
    w[r / 64] >> (r % 64) & 1 == 1
}

fn res_holds(w: &World, r: Res, row: usize) -> bool {
    match r {
        Res::I32 { lane, op, t } => {
            let x = w.i32_lanes[lane as usize][row];
            match op {
                Cmp::Lt => x < t,
                Cmp::Le => x <= t,
                Cmp::Gt => x > t,
                Cmp::Ge => x >= t,
                Cmp::Eq => x == t,
                Cmp::Ne => x != t,
            }
        }
        Res::U32 { lane, eq, v } => (w.u32_lanes[lane as usize - 2][row] == v) == eq,
    }
}

/// Kleene evaluation from the definition; `None` is UNKNOWN. With `nulls`
/// false every lane is present and this is the two-valued evaluator.
fn kleene(w: &World, e: &SetExpr, row: usize, nulls: bool) -> Option<bool> {
    match e {
        SetExpr::All => Some(true),
        SetExpr::Empty => Some(false),
        SetExpr::Sym(i) => Some(bit(&w.syms[*i as usize], row)),
        SetExpr::Res(r) => {
            let lane = match *r {
                Res::I32 { lane, .. } | Res::U32 { lane, .. } => lane as usize,
            };
            (!nulls || bit(&w.valid[lane], row)).then(|| res_holds(w, *r, row))
        }
        SetExpr::Not(c) => kleene(w, c, row, nulls).map(|b| !b),
        SetExpr::And(cs) => {
            let mut unknown = false;
            for c in cs {
                match kleene(w, c, row, nulls) {
                    Some(false) => return Some(false),
                    None => unknown = true,
                    Some(true) => {}
                }
            }
            (!unknown).then_some(true)
        }
        SetExpr::Or(cs) => {
            let mut unknown = false;
            for c in cs {
                match kleene(w, c, row, nulls) {
                    Some(true) => return Some(true),
                    None => unknown = true,
                    Some(false) => {}
                }
            }
            (!unknown).then_some(false)
        }
    }
}

fn oracle(w: &World, e: &SetExpr, nulls: bool) -> Vec<u64> {
    let mut out = vec![0u64; words_for(N)];
    for r in 0..N {
        if kleene(w, e, r, nulls) == Some(true) {
            out[r / 64] |= 1 << (r % 64);
        }
    }
    out
}

/// Lanes in field order 0..4; masks are the syms, then the four validities.
fn run(w: &World, filter: &Filter, fused: bool) -> Vec<u64> {
    let lanes = [
        LaneRef::I32(&w.i32_lanes[0]),
        LaneRef::I32(&w.i32_lanes[1]),
        LaneRef::U32(&w.u32_lanes[0]),
        LaneRef::U32(&w.u32_lanes[1]),
    ];
    let masks: Vec<&[u64]> = w.syms.iter().chain(&w.valid).map(Vec::as_slice).collect();
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &lanes,
    };
    let q = Query {
        filter: filter.clone(),
        agg: Agg::Rows,
    };
    let program = if fused { lower_fused(&q) } else { lower(&q) }.unwrap();
    let mut scratch = Scratch::for_program(&program, N).unwrap();
    let mut out = vec![0u64; words_for(N)];
    execute_into(
        &program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::Mask(&mut out),
    )
    .unwrap();
    out
}

const ALL_LANES: [u16; 4] = [0, 1, 2, 3];

fn constant(all: bool) -> Vec<u64> {
    let mut out = vec![0u64; words_for(N)];
    if all {
        for r in 0..N {
            out[r / 64] |= 1 << (r % 64);
        }
    }
    out
}

/// Positive, two-valued: Quack's answer equals the scalar evaluator's and
/// `set_lower`'s, bit for bit, through both of Quack's lowerings.
#[test]
fn two_valued_sets_agree_with_the_scalar_oracle_and_set_lower() {
    let mut g = Gen(0x5EED);
    let (mut filters, mut constants, mut with_sym) = (0, 0, 0);
    for k in 0..1500u64 {
        let w = world(1 + k % 5);
        let e = g.expr(4);
        let want = oracle(&w, &e, false);

        let got = match lower_set(&e, Logic::TwoValued).unwrap() {
            SetLowered::All => {
                constants += 1;
                constant(true)
            }
            SetLowered::Empty => {
                constants += 1;
                constant(false)
            }
            SetLowered::Filter(f) => {
                filters += 1;
                with_sym += usize::from(!f.planes().is_empty());
                let filter = f.two_valued(&ALL_LANES).unwrap();
                let a = run(&w, &filter, false);
                assert_eq!(a, run(&w, &filter, true), "lowerings disagree: {e:?}");
                a
            }
        };
        assert_eq!(got, want, "quack vs oracle: {e:?}");

        let lanes = [
            LaneRef::I32(&w.i32_lanes[0]),
            LaneRef::I32(&w.i32_lanes[1]),
            LaneRef::U32(&w.u32_lanes[0]),
            LaneRef::U32(&w.u32_lanes[1]),
        ];
        let masks: Vec<&[u64]> = w.syms.iter().map(Vec::as_slice).collect();
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        let mut kept = vec![0u64; words_for(N)];
        set_lower::SetPlan::unsplit(&e, N)
            .keep(&planes, &mut kept)
            .unwrap();
        assert_eq!(kept, want, "set_lower vs oracle: {e:?}");
    }
    // Anti-vacuity: real filters, constants and plane leaves all occurred.
    assert!(filters > 800, "{filters}");
    assert!(constants > 30, "{constants}");
    assert!(with_sym > 200, "{with_sym}");
}

/// Positive, three-valued: the rows Quack keeps are the rows where the
/// expression is TRUE under Kleene logic with NULLs. Twin: the same filter
/// read two-valued differs from that on many inputs, so the NULLs matter.
#[test]
fn kleene_sets_keep_exactly_the_true_rows() {
    let mut g = Gen(0xC6);
    // Validity planes sit after the syms in `run`'s mask list.
    let nullable: Vec<(u16, Mask)> = (0..4u16).map(|l| (l, Mask(SYMS as u16 + l))).collect();
    let (mut filters, mut differs) = (0, 0);
    for k in 0..1500u64 {
        let w = world(11 + k % 5);
        let e = g.expr(4);
        let want = oracle(&w, &e, true);
        let got = match lower_set(&e, Logic::Kleene).unwrap() {
            SetLowered::All => constant(true),
            SetLowered::Empty => constant(false),
            SetLowered::Filter(f) => {
                filters += 1;
                let filter = f.where_true(&ALL_LANES, &nullable).unwrap();
                let a = run(&w, &filter, false);
                assert_eq!(a, run(&w, &filter, true), "lowerings disagree: {e:?}");
                differs += usize::from(oracle(&w, &e, false) != want);
                a
            }
        };
        assert_eq!(got, want, "{e:?}");
    }
    assert!(filters > 800, "{filters}");
    assert!(differs > 300, "NULLs changed the answer on only {differs}");
}

/// Negative: a three-valued set is never handed out as a two-valued filter,
/// and a two-valued one is never rewritten for NULLs after a two-valued
/// normalization.
#[test]
fn the_logic_of_a_lowered_set_cannot_be_swapped() {
    let e = SetExpr::Or(vec![
        SetExpr::i32(0, Cmp::Lt, 2),
        SetExpr::negated(SetExpr::eq_u32(2, 1)),
    ]);
    let SetLowered::Filter(k) = lower_set(&e, Logic::Kleene).unwrap() else {
        panic!("a filter")
    };
    assert_eq!(k.two_valued(&ALL_LANES), Err(SetLowerError::ThreeValued));
    let SetLowered::Filter(t) = lower_set(&e, Logic::TwoValued).unwrap() else {
        panic!("a filter")
    };
    assert_eq!(
        t.where_true(&ALL_LANES, &[(0, Mask(3))]),
        Err(SetLowerError::NotThreeValued)
    );
}

/// Negative: a lane the caller did not supply is refused by name, and the
/// lanes a set names are reported ascending and distinct.
#[test]
fn a_lane_the_caller_did_not_supply_is_refused() {
    let e = SetExpr::And(vec![
        SetExpr::eq_u32(3, 1),
        SetExpr::i32(1, Cmp::Ge, 0),
        SetExpr::ne_u32(3, 2),
        SetExpr::Sym(2),
    ]);
    let SetLowered::Filter(f) = lower_set(&e, Logic::TwoValued).unwrap() else {
        panic!("a filter")
    };
    assert_eq!(f.lanes(), &[1, 3]);
    assert_eq!(f.planes(), &[2]);
    assert_eq!(
        f.two_valued(&[1]),
        Err(SetLowerError::LaneNotSupplied { lane: 3 })
    );
    // Supplied in another order: column i reads `lanes[i]`.
    let w = world(3);
    let compact = f.two_valued(&[3, 1]).unwrap();
    let direct = f.two_valued(&ALL_LANES).unwrap();
    assert_ne!(compact, direct);
    assert_eq!(run(&w, &direct, false), oracle(&w, &e, false));
}

/// Negative: a `Sym` past what a plane index can name is refused.
#[test]
fn a_sym_past_the_plane_index_range_is_refused() {
    assert_eq!(
        lower_set(&SetExpr::Sym(70_000), Logic::TwoValued).err(),
        Some(SetLowerError::SymOutOfRange(70_000))
    );
}
