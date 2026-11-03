//! D-BAG co-addressed fold falsifier. Lab probe outside the repo; real Quack lowering,
//! real mask-risc executor. Keys: u64 = classid(32) | t1(8) | t2(8) | t3(8) | t4(8).
use lance_graph_mask_risc::{execute_into, materialize_rows, words_for, Foreign, LaneRef, Out, Planes, Scratch, Value};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Mask, Query};

const KEY: Col = Col(0);
const R_IN: Mask = Mask(0); // R(parent(x), x) present, stamped at the child row x
const S_IN: Mask = Mask(1);
const T_IN: Mask = Mask(2);

fn key(cls: u32, t: [u8; 4]) -> u64 {
    (u64::from(cls) << 32) | (u64::from(t[0]) << 24) | (u64::from(t[1]) << 16) | (u64::from(t[2]) << 8) | u64::from(t[3])
}

struct Store {
    keys: Vec<u64>,      // row order as STORED (planes are in this order)
    planes: Vec<Vec<u64>>,
}

fn plane(n: usize, rows: &[usize]) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for &r in rows { w[r / 64] |= 1 << (r % 64); }
    w
}

impl Store {
    /// rows sorted by key (the "seal"); `perm` optionally reorders the stored rows afterwards.
    fn new(nodes: &[(u64, bool, bool, bool)], perm: Option<&[usize]>) -> (Store, Vec<u64>) {
        let mut sorted: Vec<_> = nodes.to_vec();
        sorted.sort_by_key(|n| n.0);
        let sealed: Vec<u64> = sorted.iter().map(|n| n.0).collect();
        let stored: Vec<_> = match perm { Some(p) => p.iter().map(|&i| sorted[i]).collect(), None => sorted };
        let n = stored.len();
        let pick = |f: fn(&(u64, bool, bool, bool)) -> bool| plane(n, &(0..n).filter(|&i| f(&stored[i])).collect::<Vec<_>>());
        let planes = vec![pick(|x| x.1), pick(|x| x.2), pick(|x| x.3)];
        (Store { keys: stored.iter().map(|x| x.0).collect(), planes }, sealed)
    }
    fn rows(&self, f: Filter) -> Vec<usize> {
        let q = Query { filter: f, agg: Agg::Rows };
        let prog = lower(&q).expect("lowers");
        let n = self.keys.len();
        let lanes = [LaneRef::U64(&self.keys)];
        let masks: Vec<&[u64]> = self.planes.iter().map(|p| p.as_slice()).collect();
        let pl = Planes { n_rows: n, masks: &masks, lanes: &lanes };
        let mut sc = Scratch::for_program(&prog, n).expect("carves");
        let mut out = vec![0u64; words_for(n)];
        match execute_into(&prog, &pl, &Foreign::NONE, &mut sc, Out::Mask(&mut out)).expect("runs") {
            Value::Mask(_) => {}
            v => panic!("{v:?}"),
        }
        materialize_rows(&out, n) // test-side readout only
    }
}

/// Ordinal of `k` in the SEALED order: representation translation of a query
/// constant (the address), never a value read from a relation row.
fn ord(sealed: &[u64], k: u64) -> u32 { sealed.binary_search(&k).expect("node exists") as u32 }
fn single(sealed: &[u64], k: u64) -> Filter { let o = ord(sealed, k); Filter::Cmp(KEY, Cmp::Range { lo: o, hi: o + 1 }) }

/// METHOD P — the single-range reduction: T-lane rows under prefix(a).
fn method_p(s: &Store, a: u64, bits: u32) -> usize {
    s.rows(Filter::And(vec![Filter::Plane(T_IN), Filter::prefix_u64(KEY, a, bits)])).len()
}
/// METHOD C — one fold over co-addressed presence, fixed endpoints a..d:
/// Or(R_in@b, S_in@c, T_in@d) must select exactly 3 rows. b, c are d's own
/// prefixes (address arithmetic on the query constant), not values from any row.
fn method_c(s: &Store, sealed: &[u64], b: u64, c: u64, d: u64) -> usize {
    s.rows(Filter::Or(vec![
        Filter::And(vec![Filter::Plane(R_IN), single(sealed, b)]),
        Filter::And(vec![Filter::Plane(S_IN), single(sealed, c)]),
        Filter::And(vec![Filter::Plane(T_IN), single(sealed, d)]),
    ])).len()
}

fn main() {
    let k = 0x0905_000A;
    let a = key(k, [1, 0, 0, 0]);
    let b = key(k, [1, 2, 0, 0]);
    let c = key(k, [1, 2, 3, 0]);
    let d = key(k, [1, 2, 3, 4]);
    let x = key(k, [9, 9, 0, 0]); // a distractor sibling subtree
    let prefix_a = 32 + 8; // classid + tier 1

    // (key, R_in, S_in, T_in)
    let valid = [(a, false, false, false), (b, true, false, false), (c, false, true, false), (d, false, false, true), (x, true, false, false)];
    let mut ce = valid; ce[2].2 = false; // S(b,c) DELETED, T(c,d) still present

    let mut rdel = valid; rdel[1].1 = false;
    let mut tdel = valid; tdel[3].3 = false;
    for (name, nodes) in [("4 valid hierarchy", &valid[..]), ("0 COUNTEREXAMPLE S deleted", &ce[..]), ("0b R deleted", &rdel[..]), ("0c T deleted", &tdel[..])] {
        let (s, sealed) = Store::new(nodes, None);
        let oracle = nodes.iter().any(|n| n.0 == b && n.1) && nodes.iter().any(|n| n.0 == c && n.2) && nodes.iter().any(|n| n.0 == d && n.3);
        let p = method_p(&s, a, prefix_a);
        let cnt = method_c(&s, &sealed, b, c, d);
        println!("{name}: oracle={oracle} | P(prefix count)={p} -> says {} | C(presence fold)={cnt}/3 -> says {}", p > 0, cnt == 3);
    }

    // 1. permuted rows vs the claimed seal (ordinals from sealed order, planes stored reversed)
    let perm: Vec<usize> = (0..valid.len()).rev().collect();
    let (s, sealed) = Store::new(&valid, Some(&perm));
    let p = method_p(&s, a, prefix_a);
    let cnt = method_c(&s, &sealed, b, c, d);
    println!("1 permuted planes: oracle=true | P(sweep reads lane)={p} -> {} | C(sealed-order ranges)={cnt}/3 -> {} (no error raised)", p > 0, cnt == 3);

    // 2. duplicate D keys: a T edge table whose D lane holds d twice
    let d2 = key(k, [1, 2, 3, 5]);
    let t_rows = [d, d, d2];
    let s = Store { keys: t_rows.to_vec(), planes: vec![plane(3, &[]), plane(3, &[]), plane(3, &[0, 1, 2])] };
    let rows = method_p(&s, a, prefix_a);
    let mut distinct = t_rows.to_vec(); distinct.dedup();
    println!("2 duplicate D keys: range rows (hi-lo)={rows}, distinct D={}", distinct.len());

    // 3. cross-ClassId: same concept (hi u16 0x0905), other app prefix, identical tiers
    let other = key(0x0905_000B, [1, 2, 3, 4]);
    let mut nodes = valid.to_vec(); nodes.push((other, false, false, true));
    let (s, _) = Store::new(&nodes, None);
    println!("3 cross-ClassId: prefix incl. full classid ({}b)={} | concept-only prefix (16b)={} | tiers-only key compare would equal: {}",
        prefix_a, method_p(&s, a, prefix_a), method_p(&s, a, 16), (other & 0xFFFF_FFFF) == (d & 0xFFFF_FFFF));
}
