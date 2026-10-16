//! D-ASM-11 pre-work — the cyclic fixture, the bag oracle, and the P-7
//! expressibility check (`.claude/board/entries/2026-10-16-coresearch-wcoj-leapfrog.md`).
//!
//! The query is the triangle `Q(a) :- R(a,b), S(b,c), T(a,c)`, existence half:
//! which head rows `a` extend to an answer. R and T are functional over `a`
//! (lanes `rb`, `tc` of the head table), S is functional over `b` (a foreign
//! table: plane `s_valid`, lane `s_dst`).
//!
//! - **A0, the semijoin chain**: `R.valid ∧ T.valid ∧ Semijoin(rb, s_valid)`.
//!   On a cyclic body it over-approximates: it never checks that the `c`
//!   reached through `S` is the `c` that `T` names.
//! - **P-7, the exact mask with today's ops**: the missing check is a
//!   comparison of two lanes (`s_dst[rb[a]] == tc[a]`), and mask-risc only
//!   compares a lane with a constant. It can still be written, as a
//!   disjunction over every value `v` of the `c` domain:
//!   `∨_v (EqU32Via(rb, s_dst, v) ∧ tc == v)`. That is correct but costs one
//!   pass per value, `Θ(rows · |D|)`, which misses the `N^1.5` bound once
//!   `|D|` exceeds `√N`. This is the insufficiency proof the map asked for:
//!   expressible, not at the bound.
//! - **X17, the bag oracle**: datafrog is a SET; answers with multiplicity
//!   need a count carried through the join and summed after it.
//!
//! ```text
//! cargo test --manifest-path crates/lance-graph-metta/Cargo.toml --example wcoj_probe
//! ```

use std::collections::BTreeMap;

use datafrog::Relation;
use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::{words_for, Foreign, ForeignPlane, LaneRef, Out, Planes, Value};
use lance_graph_quack::{
    lower, Agg, Cmp, Col, Filter, ForeignLane, ForeignPlane as QForeign, Mask, Query,
};

const RB: Col = Col(0);
const TC: Col = Col(1);
const R_VALID: Mask = Mask(0);
const T_VALID: Mask = Mask(1);
const S_VALID: QForeign = QForeign(0);
const S_DST: ForeignLane = ForeignLane(0);

/// A functional relation over the address space `0..n`: `dst[a]` where
/// `valid` has bit `a`.
struct Func {
    dst: Vec<u32>,
    valid: Vec<u64>,
}

impl Func {
    fn new(n: usize, facts: &[(u32, u32)]) -> Self {
        let mut dst = vec![0u32; n];
        let mut valid = vec![0u64; words_for(n)];
        for &(s, o) in facts {
            assert!((s as usize) < n && (o as usize) < n, "fact outside 0..n");
            assert_eq!(
                (valid[s as usize / 64] >> (s % 64)) & 1,
                0,
                "not functional"
            );
            valid[s as usize / 64] |= 1 << (s % 64);
            dst[s as usize] = o;
        }
        Func { dst, valid }
    }
}

/// A triangle instance over one address space `0..n`.
struct Triangle {
    n: usize,
    r: Func,
    s: Func,
    t: Func,
    facts: [Vec<(u32, u32)>; 3],
}

impl Triangle {
    fn new(n: usize, r: Vec<(u32, u32)>, s: Vec<(u32, u32)>, t: Vec<(u32, u32)>) -> Self {
        Triangle {
            n,
            r: Func::new(n, &r),
            s: Func::new(n, &s),
            t: Func::new(n, &t),
            facts: [r, s, t],
        }
    }

    /// Runs `filter` over the head table and returns the kept rows.
    fn run(&self, filter: Filter) -> (Vec<u32>, usize) {
        let program = lower(&Query {
            filter,
            agg: Agg::Rows,
        })
        .expect("lowers");
        let ops = program.ops.len();
        let lanes = [LaneRef::U32(&self.r.dst), LaneRef::U32(&self.t.dst)];
        let masks = [self.r.valid.as_slice(), self.t.valid.as_slice()];
        let planes = Planes {
            n_rows: self.n,
            masks: &masks,
            lanes: &lanes,
        };
        let fplanes = [ForeignPlane {
            words: &self.s.valid,
            rows: self.n,
        }];
        let flanes = [LaneRef::U32(&self.s.dst)];
        let foreign = Foreign {
            planes: &fplanes,
            lanes: &flanes,
        };
        let mut scratch = Scratch::for_program(&program, self.n).expect("carves");
        let mut kept = vec![0u64; words_for(self.n)];
        match execute_into(
            &program,
            &planes,
            &foreign,
            &mut scratch,
            Out::Mask(&mut kept),
        )
        .expect("runs")
        {
            Value::Mask(_) => {}
            other => panic!("not a kept mask: {other:?}"),
        }
        let rows = (0..self.n as u32)
            .filter(|&a| (kept[a as usize / 64] >> (a % 64)) & 1 == 1)
            .collect();
        (rows, ops)
    }

    /// A0: the semijoin chain.
    fn semijoin_chain(&self) -> Vec<u32> {
        self.run(Filter::and([
            Filter::plane(R_VALID),
            Filter::plane(T_VALID),
            Filter::semijoin(RB, S_VALID),
        ]))
        .0
    }

    /// P-7: the exact mask as a disjunction over the `c` domain `0..d`.
    /// Returns the rows and the program's op count.
    fn exact_by_disjunction(&self, d: u32) -> (Vec<u32>, usize) {
        self.run(Filter::and([
            Filter::plane(R_VALID),
            Filter::plane(T_VALID),
            Filter::semijoin(RB, S_VALID),
            Filter::or((0..d).map(|v| {
                Filter::and([
                    Filter::eq_u32_via(RB, S_DST, v),
                    Filter::cmp(TC, Cmp::EqU32(v)),
                ])
            })),
        ]))
    }

    /// The datafrog SET oracle: head rows `a` with a triangle.
    fn oracle(&self) -> Vec<u32> {
        let [r, s, t] = &self.facts;
        let r_by_b: Relation<(u32, u32)> = r.iter().map(|&(a, b)| (b, a)).collect();
        let s: Relation<(u32, u32)> = s.iter().copied().collect();
        let ac: Relation<((u32, u32), ())> =
            Relation::from_join(&r_by_b, &s, |_b, &a, &c| ((a, c), ()));
        let t: Relation<((u32, u32), ())> = t.iter().map(|&p| (p, ())).collect();
        let heads: Relation<(u32, ())> = Relation::from_join(&ac, &t, |&(a, _c), _, _| (a, ()));
        heads.elements.iter().map(|&(a, ())| a).collect()
    }
}

/// F4: `k` disjoint copies of the 2+2+2 counterexample. Every semijoin keeps
/// every row; the query is empty.
fn f4(k: u32) -> Triangle {
    let (mut r, mut s, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..k {
        let (x, y) = (2 * i, 2 * i + 1);
        r.extend([(x, x), (y, y)]);
        s.extend([(x, y), (y, x)]);
        t.extend([(x, x), (y, y)]);
    }
    Triangle::new(2 * k as usize, r, s, t)
}

/// A functional instance with real triangles: `T` agrees with `S ∘ R` on
/// every fourth row and disagrees on the rest; R and S have holes.
fn mixed(n: u32) -> Triangle {
    let r: Vec<(u32, u32)> = (0..n)
        .filter(|a| a % 3 != 0)
        .map(|a| (a, (7 * a + 3) % n))
        .collect();
    let s: Vec<(u32, u32)> = (0..n)
        .filter(|b| b % 5 != 0)
        .map(|b| (b, (5 * b + 1) % n))
        .collect();
    let sf: BTreeMap<u32, u32> = s.iter().copied().collect();
    let t: Vec<(u32, u32)> = r
        .iter()
        .map(|&(a, b)| {
            let c = sf.get(&b).copied().unwrap_or(a);
            (a, if a % 4 == 0 { c } else { (c + 1) % n })
        })
        .collect();
    Triangle::new(n as usize, r, s, t)
}

// ───────────── X17: the bag oracle ─────────────

/// Bag answer of `Q(x, z) :- R(x, y), S(y, z)` with duplicate facts, by
/// nested loops: the multiplicity of `(x, z)` is the number of pairs of
/// facts that produce it.
fn bag_brute(r: &[(u32, u32)], s: &[(u32, u32)]) -> BTreeMap<(u32, u32), u64> {
    let mut out = BTreeMap::new();
    for &(x, y) in r {
        for &(y2, z) in s {
            if y == y2 {
                *out.entry((x, z)).or_insert(0) += 1;
            }
        }
    }
    out
}

/// The same bag through datafrog. Datafrog relations are sets, so each fact
/// is collapsed to `(fact, count)` first and the count rides the join; the
/// join tuple keeps `y` so two paths with equal counts are not merged.
fn bag_datafrog(r: &[(u32, u32)], s: &[(u32, u32)], dedup: bool) -> BTreeMap<(u32, u32), u64> {
    fn counted(facts: &[(u32, u32)], dedup: bool) -> BTreeMap<(u32, u32), u64> {
        let mut m = BTreeMap::new();
        for &f in facts {
            let c = m.entry(f).or_insert(0);
            *c = if dedup { 1 } else { *c + 1 };
        }
        m
    }
    let r: Relation<(u32, (u32, u64))> = counted(r, dedup)
        .into_iter()
        .map(|((x, y), m)| (y, (x, m)))
        .collect();
    let s: Relation<(u32, (u32, u64))> = counted(s, dedup)
        .into_iter()
        .map(|((y, z), m)| (y, (z, m)))
        .collect();
    let paths: Relation<((u32, u32, u32), u64)> =
        Relation::from_join(&r, &s, |&y, &(x, m1), &(z, m2)| ((x, z, y), m1 * m2));
    let mut out = BTreeMap::new();
    for &((x, z, _y), m) in paths.elements.iter() {
        *out.entry((x, z)).or_insert(0) += m;
    }
    out
}

/// n:m facts with duplicates (a bag), and an n:1 set with none.
/// `(R, S, has_duplicates)`.
type BagFixture = (Vec<(u32, u32)>, Vec<(u32, u32)>, bool);

fn bag_fixtures() -> [BagFixture; 2] {
    let mut r = Vec::new();
    let mut s = Vec::new();
    for x in 0..40u32 {
        for j in 0..3 {
            r.push((x, (x + j) % 13));
        }
        if x % 4 == 0 {
            r.push((x, x % 13)); // a duplicate fact
        }
    }
    for y in 0..13u32 {
        for j in 0..4 {
            s.push((y, (y * 3 + j) % 17));
        }
        s.push((y, (y * 3) % 17)); // a duplicate fact
    }
    let r1: Vec<(u32, u32)> = (0..40).map(|x| (x, x % 13)).collect();
    let s1: Vec<(u32, u32)> = (0..13).map(|y| (y, (y * 3) % 17)).collect();
    [(r, s, true), (r1, s1, false)]
}

fn main() {
    println!(
        "tier: avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    println!("fixture | rows | A0 kept | exact | oracle | ops(disjunction over |D|)");
    for (name, tri) in [
        ("F4 x8", f4(8)),
        ("mixed 64", mixed(64)),
        ("mixed 256", mixed(256)),
    ] {
        let a0 = tri.semijoin_chain();
        let (exact, ops) = tri.exact_by_disjunction(tri.n as u32);
        let want = tri.oracle();
        println!(
            "{name} | {} | {} | {} | {} | {ops} (|D| = {})",
            tri.n,
            a0.len(),
            exact.len(),
            want.len(),
            tri.n
        );
    }
    for (r, s, dup) in bag_fixtures() {
        let brute = bag_brute(&r, &s);
        println!(
            "bag (duplicates={dup}): {} keys, total multiplicity {}, datafrog+counts agrees: {}",
            brute.len(),
            brute.values().sum::<u64>(),
            bag_datafrog(&r, &s, false) == brute
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// X6 / F4. FAILS IF the semijoin chain stops over-approximating on the
    /// counterexample (then the fixture no longer tests anything), or if the
    /// oracle finds a triangle in it.
    #[test]
    fn f4_semijoin_chain_keeps_every_row_and_the_query_is_empty() {
        for k in [1, 2, 8, 50] {
            let tri = f4(k);
            assert_eq!(tri.semijoin_chain().len(), 2 * k as usize, "k={k}");
            assert!(tri.oracle().is_empty(), "k={k}");
        }
    }

    /// P-7, correctness: the disjunction over the value domain is the exact
    /// mask. Disable run: drop the `tc == v` conjunct and it collapses to A0.
    #[test]
    fn p7_disjunction_over_the_domain_equals_the_set_oracle() {
        for tri in [f4(1), f4(8), mixed(30), mixed(64), mixed(200)] {
            let want = tri.oracle();
            let (got, _) = tri.exact_by_disjunction(tri.n as u32);
            assert_eq!(got, want, "n={}", tri.n);
        }
        // Anti-vacuity: on `mixed` the semijoin chain keeps rows the oracle
        // rejects, and the oracle keeps some rows.
        let tri = mixed(200);
        let (a0, want) = (tri.semijoin_chain(), tri.oracle());
        assert!(
            !want.is_empty() && want.len() * 2 < a0.len(),
            "{} vs {}",
            want.len(),
            a0.len()
        );
    }

    /// P-7, cost: the program grows linearly with the value domain, so the
    /// row work is `Θ(rows · |D|)`. This is why the exact cyclic mask is not
    /// at the `N^1.5` bound with today's ops.
    #[test]
    fn p7_disjunction_costs_one_pass_per_domain_value() {
        let tri = mixed(256);
        let (_, ops64) = tri.exact_by_disjunction(64);
        let (_, ops128) = tri.exact_by_disjunction(128);
        let (_, ops256) = tri.exact_by_disjunction(256);
        assert!(
            ops128 - ops64 >= 64 && ops256 - ops128 >= 128,
            "{ops64} {ops128} {ops256}"
        );
        assert_eq!(ops256 - ops128, 2 * (ops128 - ops64), "not linear in |D|");
    }

    /// X17. FAILS IF datafrog-with-counts disagrees with the nested loop on
    /// the bag. Disable run: collapse counts to 1 (`dedup = true`) and the
    /// n:m fixture with duplicates goes red while the n:1 one stays green.
    #[test]
    fn x17_bag_oracle_matches_the_nested_loop() {
        for (r, s, _) in bag_fixtures() {
            assert_eq!(bag_datafrog(&r, &s, false), bag_brute(&r, &s));
        }
    }

    /// The disable run of X17, kept as a test: deduplicating inside the
    /// oracle must be detectable on the bag and invisible on the set.
    #[test]
    fn x17_dedup_is_caught_on_the_bag_only() {
        for (r, s, has_duplicates) in bag_fixtures() {
            let same = bag_datafrog(&r, &s, true) == bag_brute(&r, &s);
            assert_eq!(same, !has_duplicates);
        }
        // Multiplicities above one exist, so the bag is not a set.
        let (r, s, _) = &bag_fixtures()[0];
        assert!(bag_brute(r, s).values().any(|&m| m > 1));
    }
}
