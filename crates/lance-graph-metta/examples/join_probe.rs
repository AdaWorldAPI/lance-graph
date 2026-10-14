//! D-MTP-1 — the two-hop MeTTa join lowered through Quack into mask-risc.
//!
//! MeTTa: `(match &kb (, (parent $x $y) (parent $y $z)) ($x $z))`.
//!
//! Layout (ingest): a functional binary relation over the subject's address
//! space. Row `a` is the fact whose subject has address `a`; lane `dst` holds
//! the object's address; plane `valid` marks the addresses that have a fact.
//! A second fact for the same subject is refused.
//!
//! Lowering: `Plane(valid) AND Semijoin { fk: dst, foreign: valid }`, then
//! `Count` (the size of the collapsed result) or `Rows` + binding emission
//! `(x, dst[dst[x]])` over the kept bits — the only place pairs exist.
//!
//! Oracle: datafrog (AdaWorldAPI fork), compared as sorted pair vectors.
//! Pre-registration: plan §6a.
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=0 RUSTFLAGS=-Ctarget-cpu=native \
//!   cargo run --release --manifest-path crates/lance-graph-metta/Cargo.toml --example join_probe
//! ```

use std::time::Instant;

use datafrog::Relation;
use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::{words_for, Foreign, ForeignPlane, LaneRef, Out, Planes, Value};
use lance_graph_quack::{lower, Agg, Col, Filter, ForeignPlane as QForeign, Mask, Query};

const DST: Col = Col(0);
const VALID: Mask = Mask(0);
const VALID_FOREIGN: QForeign = QForeign(0);

/// A functional binary relation laid out over its subjects' address space.
struct Relation2 {
    n: usize,
    dst: Vec<u32>,
    valid: Vec<u64>,
}

#[derive(Debug, PartialEq, Eq)]
enum IngestError {
    /// A subject already has a fact: not a functional relation.
    SecondFact { subject: u32 },
    /// An address outside the declared address space.
    OutOfSpace { addr: u32 },
}

impl Relation2 {
    /// Ingest facts `(subject, object)` over addresses `0..n`.
    fn ingest(n: usize, facts: &[(u32, u32)]) -> Result<Self, IngestError> {
        let mut dst = vec![0u32; n];
        let mut valid = vec![0u64; words_for(n)];
        for &(s, o) in facts {
            for a in [s, o] {
                if a as usize >= n {
                    return Err(IngestError::OutOfSpace { addr: a });
                }
            }
            let (w, b) = (s as usize / 64, s as usize % 64);
            if (valid[w] >> b) & 1 == 1 {
                return Err(IngestError::SecondFact { subject: s });
            }
            valid[w] |= 1 << b;
            dst[s as usize] = o;
        }
        Ok(Relation2 { n, dst, valid })
    }

    fn query(&self) -> Filter {
        Filter::and([Filter::plane(VALID), Filter::semijoin(DST, VALID_FOREIGN)])
    }

    fn run(&self, agg: Agg, kept: &mut [u64]) -> Value {
        let keep = matches!(agg, Agg::Rows);
        let program = lower(&Query { filter: self.query(), agg }).expect("lowers");
        let lanes = [LaneRef::U32(&self.dst)];
        let masks = [self.valid.as_slice()];
        let planes = Planes { n_rows: self.n, masks: &masks, lanes: &lanes };
        let fplanes = [ForeignPlane { words: &self.valid, rows: self.n }];
        let foreign = Foreign { planes: &fplanes, lanes: &[] };
        let mut scratch = Scratch::for_program(&program, self.n).expect("carves");
        let out = if keep { Out::Mask(kept) } else { Out::None };
        execute_into(&program, &planes, &foreign, &mut scratch, out).expect("runs")
    }

    /// `size-atom` of the collapsed result.
    fn count(&self) -> usize {
        match self.run(Agg::Count, &mut []) {
            Value::Count(c) => c,
            other => panic!("not a count: {other:?}"),
        }
    }

    /// The bindings `($x, $z)`, emitted only for kept rows.
    fn pairs(&self, kept: &mut [u64], out: &mut Vec<(u32, u32)>) {
        match self.run(Agg::Rows, kept) {
            Value::Mask(_) => {}
            other => panic!("not a kept mask: {other:?}"),
        }
        out.clear();
        for (w, &word) in kept.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let x = (w * 64 + bits.trailing_zeros() as usize) as u32;
                let y = self.dst[x as usize];
                out.push((x, self.dst[y as usize]));
                bits &= bits - 1;
            }
        }
    }
}

/// F1 (`partial = false`): `parent(i, (7i+3) mod n)` for every `i`.
/// F2 (`partial = true`): the same, only for `i mod 3 != 0`.
fn facts(n: u32, partial: bool) -> Vec<(u32, u32)> {
    (0..n)
        .filter(|i| !partial || i % 3 != 0)
        .map(|i| (i, ((7u64 * i as u64 + 3) % n as u64) as u32))
        .collect()
}

/// The datafrog oracle: `(x, z)` with `parent(x, y)` and `parent(y, z)`.
fn oracle(facts: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let parent: Relation<(u32, u32)> = facts.iter().copied().collect();
    let by_dst: Relation<(u32, u32)> = facts.iter().map(|&(x, y)| (y, x)).collect();
    let pairs = Relation::from_join(&by_dst, &parent, |_y, &x, &z| (x, z));
    pairs.elements.clone()
}

fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..reps {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64());
        out = Some(r);
    }
    (out.expect("reps > 0"), best)
}

fn main() {
    println!(
        "tier: avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    println!("fixture n | pairs | ingest ns/fact | count ns/row | rows+emit ns/pair | datafrog ns/pair");
    for partial in [false, true] {
        for n in [100u32, 1_000, 10_000, 100_000, 1_000_000] {
            let f = facts(n, partial);
            let (rel, t_ingest) = best(5, || Relation2::ingest(n as usize, &f).expect("functional"));
            let (count, t_count) = best(5, || rel.count());
            let mut kept = vec![0u64; words_for(n as usize)];
            let mut pairs = Vec::with_capacity(n as usize);
            let (_, t_pairs) = best(5, || rel.pairs(&mut kept, &mut pairs));
            let (want, t_frog) = best(5, || oracle(&f));
            let mut got = pairs.clone();
            got.sort_unstable();
            assert_eq!(got, want, "pairs differ from datafrog (F{}, n={n})", 1 + partial as u8);
            assert_eq!(count, want.len(), "count differs from datafrog");
            let p = want.len().max(1) as f64;
            println!(
                "F{} {n:>9} | {:>7} | {:>6.2} | {:>6.3} | {:>6.2} | {:>6.1}",
                1 + partial as u8,
                want.len(),
                t_ingest * 1e9 / f.len() as f64,
                t_count * 1e9 / n as f64,
                t_pairs * 1e9 / p,
                t_frog * 1e9 / p,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(n: u32, partial: bool) {
        let f = facts(n, partial);
        let rel = Relation2::ingest(n as usize, &f).expect("functional");
        let mut kept = vec![0u64; words_for(n as usize)];
        let mut got = Vec::new();
        rel.pairs(&mut kept, &mut got);
        got.sort_unstable();
        let want = oracle(&f);
        assert_eq!(got, want, "F{} n={n}", 1 + partial as u8);
        assert_eq!(rel.count(), want.len(), "F{} n={n}", 1 + partial as u8);
    }

    #[test]
    fn f1_total_relation_matches_datafrog() {
        for n in [1, 2, 63, 64, 65, 100, 1_000] {
            check(n, false);
        }
    }

    #[test]
    fn f2_partial_relation_matches_datafrog() {
        for n in [3, 63, 64, 65, 100, 1_000] {
            check(n, true);
        }
    }

    /// Anti-vacuity: F2 must reject rows, or it cannot tell a semijoin from
    /// no semijoin.
    #[test]
    fn f2_semijoin_rejects_rows() {
        let f = facts(1_000, true);
        let want = oracle(&f);
        assert!(want.len() < f.len(), "{} of {} facts survive", want.len(), f.len());
        assert!(want.len() * 2 > f.len(), "fixture too sparse to be informative");
    }

    /// The counts Hyperon and PeTTa returned for the same MeTTa programs.
    #[test]
    fn counts_match_hyperon_and_petta() {
        // F1 from the D-MTP-0 runs; F2 measured 2026-10-14: Hyperon 46 / 476,
        // PeTTa 46 / 476 / 4,762.
        for (n, partial, expected) in [
            (100, false, 100),
            (1_000, false, 1_000),
            (100, true, 46),
            (1_000, true, 476),
            (10_000, true, 4_762),
        ] {
            let f = facts(n, partial);
            let rel = Relation2::ingest(n as usize, &f).expect("functional");
            assert_eq!(rel.count(), expected, "n={n}");
        }
    }

    #[test]
    fn ingest_refuses_a_second_fact_for_one_subject() {
        assert_eq!(
            Relation2::ingest(4, &[(1, 2), (1, 3)]).err(),
            Some(IngestError::SecondFact { subject: 1 })
        );
        assert_eq!(
            Relation2::ingest(4, &[(1, 4)]).err(),
            Some(IngestError::OutOfSpace { addr: 4 })
        );
        assert!(Relation2::ingest(4, &[(1, 2), (2, 1)]).is_ok());
    }
}
