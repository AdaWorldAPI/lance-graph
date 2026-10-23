//! L1 (CC2 Law Frontier): a D-DRMC-5 reassociation certificate, issued on one
//! frame, executed on a second, independently issued frame without egglog.
//! Pre-registered in `.claude/board/entries/2026-10-23-drmc-l1-x8-fresh-frame-preregistration.md`.
use std::hint::black_box;
use std::time::Instant;

use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Scope, bind_dr64,
    recipe::{Backend, CompiledReference, LawReference, LawShape, Primitive, RecipeCertificate},
    relcomp::{
        CertifyRefusal, Chain, ProofRefusal, RelExpr, ReuseRefusal, Rows, admit_composition_assoc,
        all_orders, certify_reassociation, egglog_calls, execute_certified, lower, lowering_digest,
    },
};

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };
const ABI: u32 = 1;

fn axis(i: u16) -> Axis {
    let kind = [
        AxisKind::Identity,
        AxisKind::Property,
        AxisKind::Community,
        AxisKind::Capability,
        AxisKind::Identity,
        AxisKind::Property,
    ][i as usize];
    Axis {
        class: 10 + i,
        kind,
    }
}
fn method(i: u16) -> MethodSignature {
    MethodSignature::presence(10 + i, i, axis(i), axis(i + 1))
}
/// A store at the given revisions. Two stores never share an issue, even at
/// equal revisions, so each call is an independently issued frame.
fn store(source: u64) -> FrameStore {
    FrameStore::new(FrameRevisions {
        source,
        relation_map: 1,
        coordinates: 1,
    })
}
fn bind<'a>(st: &FrameStore, m: MethodSignature, rows: &'a Rows) -> BoundDr64<'a> {
    let at = Guards {
        frame: st.seal(),
        focus: FOCUS,
        scope: SCOPE,
    };
    bind_dr64(
        st,
        m,
        m,
        &at,
        (FOCUS, SCOPE),
        Operand {
            rows,
            validity: None,
        },
    )
    .unwrap()
}
fn fill(seed: u64, keep_of_64: u32) -> Rows {
    let mut x = seed | 1;
    let mut rows = [0u64; 64];
    for r in rows.iter_mut() {
        let mut word = 0u64;
        for b in 0..64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if (x % 64) < u64::from(keep_of_64) {
                word |= 1 << b;
            }
        }
        *r = word;
    }
    rows
}
/// Independent oracle, sharing no code with the compose kernel.
fn oracle(a: &Rows, b: &Rows) -> Rows {
    let mut out = [0u64; 64];
    for i in 0..64 {
        for k in 0..64 {
            if (0..64).any(|j| (a[i] >> j) & 1 == 1 && (b[j] >> k) & 1 == 1) {
                out[i] |= 1 << k;
            }
        }
    }
    out
}
fn oracle_eval(ops: &[Rows], order: &RelExpr) -> Rows {
    match order {
        RelExpr::Leaf(i) => ops[*i],
        RelExpr::Comp(a, b) => oracle(&oracle_eval(ops, a), &oracle_eval(ops, b)),
    }
}
fn data(seed: u64, density: u32, n: usize) -> Vec<Rows> {
    (0..n as u64)
        .map(|i| fill(seed * 1009 + i * 31 + 7, density))
        .collect()
}
fn chain<'a>(st: &'a FrameStore, ops: &'a [Rows]) -> Chain<'a> {
    let bound = ops
        .iter()
        .enumerate()
        .map(|(i, r)| bind(st, method(i as u16), r))
        .collect();
    Chain::new(st, bound).unwrap()
}
fn law() -> LawReference {
    LawReference {
        id: 7,
        revision: 1,
        shape: LawShape::for_method(method(0), Primitive::BooleanCompose),
    }
}
fn compiled(to: &RelExpr) -> CompiledReference {
    CompiledReference {
        law: law(),
        compiled_digest: lowering_digest(&lower(to)),
        compiler_revision: 1,
        target_abi: ABI,
        backend: Backend::ResidentDr64Simd,
    }
}
fn l(i: usize) -> RelExpr {
    RelExpr::Leaf(i)
}
fn c(a: RelExpr, b: RelExpr) -> RelExpr {
    RelExpr::comp(a, b)
}
/// Five lanes, five proposed association orders of one 5-chain.
fn proposals() -> Vec<RelExpr> {
    vec![
        c(c(c(c(l(0), l(1)), l(2)), l(3)), l(4)), // left-deep
        c(l(0), c(l(1), c(l(2), c(l(3), l(4))))), // right-deep
        c(c(l(0), l(1)), c(l(2), c(l(3), l(4)))),
        c(c(c(l(0), l(1)), l(2)), c(l(3), l(4))),
        c(l(0), c(c(l(1), l(2)), c(l(3), l(4)))),
    ]
}

/// Frame 1: certify every proposal against the cheapest order by measured
/// work on frame 1's data. Returns (from, certificate) pairs.
fn certify_on_frame_one() -> Vec<(RelExpr, RecipeCertificate)> {
    let st1 = store(1);
    let ops1 = data(1, 8, 5);
    let ch1 = chain(&st1, &ops1);
    let to = all_orders(5)
        .into_iter()
        .min_by_key(|o| ch1.eval(o).unwrap().1)
        .unwrap();
    let admitted = admit_composition_assoc(law().shape).unwrap();
    proposals()
        .into_iter()
        .map(|from| {
            let cert = certify_reassociation(&admitted, law(), &from, &to, compiled(&to)).unwrap();
            (from, cert)
        })
        .collect()
}

/// Positive + silent twin: frame 2 is another store with other data. Every
/// certificate executes there, calls egglog zero times, and equals the
/// independent oracle and the chain's own evaluation of the requested order.
#[test]
fn a_certificate_from_frame_one_executes_on_a_fresh_frame_without_egglog() {
    let certs = certify_on_frame_one();
    for (seed, density) in [(2, 0), (3, 4), (4, 32), (5, 63)] {
        let st2 = store(2);
        let ops2 = data(seed, density, 5);
        let ch2 = chain(&st2, &ops2);
        for (from, cert) in &certs {
            let before = egglog_calls();
            let (rows, _) = execute_certified(cert, &ch2, from, ABI)
                .unwrap_or_else(|e| panic!("silent twin refused: {e:?} (seed {seed})"));
            assert_eq!(egglog_calls(), before, "the hit path ran egglog");
            assert_eq!(rows, oracle_eval(&ops2, from), "oracle, seed {seed}");
            assert_eq!(rows, ch2.eval(from).unwrap().0, "chain eval, seed {seed}");
        }
    }
}

/// Anti-vacuity: certification did call egglog, so a zero delta on the hit
/// path is a measurement, not a counter that never moves.
#[test]
fn certification_itself_calls_egglog() {
    let before = egglog_calls();
    let _ = certify_on_frame_one();
    assert!(egglog_calls() >= before + 5);
}

/// The adversarial lane: commutation is not a reassociation. No certificate is
/// issued, so there is nothing to reuse.
#[test]
fn a_commutation_proposal_gets_no_certificate() {
    let admitted = admit_composition_assoc(law().shape).unwrap();
    let ab = c(l(0), l(1));
    let ba = c(l(1), l(0));
    assert_eq!(
        certify_reassociation(&admitted, law(), &ab, &ba, compiled(&ba)).unwrap_err(),
        CertifyRefusal::Proof(ProofRefusal::DifferentLeaves)
    );
}

#[test]
fn a_different_runtime_abi_is_refused() {
    let certs = certify_on_frame_one();
    let st2 = store(2);
    let ops2 = data(9, 8, 5);
    let ch2 = chain(&st2, &ops2);
    let (from, cert) = &certs[0];
    assert_eq!(
        execute_certified(cert, &ch2, from, ABI + 1).unwrap_err(),
        ReuseRefusal::WrongTarget
    );
}

/// A certificate answers its own query only: another order of the same chain
/// is refused, though associativity would make the answer equal.
#[test]
fn an_order_the_certificate_does_not_answer_is_refused() {
    let certs = certify_on_frame_one();
    let st2 = store(2);
    let ops2 = data(9, 8, 5);
    let ch2 = chain(&st2, &ops2);
    let (from0, cert0) = &certs[0];
    let (from1, _) = &certs[1];
    // Anti-vacuity: the two queries differ.
    assert_ne!(from0, from1);
    assert_eq!(
        execute_certified(cert0, &ch2, from1, ABI).unwrap_err(),
        ReuseRefusal::NotCertifiedQuery
    );
    assert!(execute_certified(cert0, &ch2, from0, ABI).is_ok());
}

/// The certified query names five positions; a four-operand chain on frame 2
/// is not that chain.
#[test]
fn a_chain_of_another_length_is_refused() {
    let certs = certify_on_frame_one();
    let st2 = store(2);
    let ops2 = data(9, 8, 4);
    let ch4 = chain(&st2, &ops2);
    let (from, cert) = &certs[0];
    assert_eq!(
        execute_certified(cert, &ch4, from, ABI).unwrap_err(),
        ReuseRefusal::NotThisChain
    );
}

/// Measured, reported, not gated: cold certification in µs, hot execution in
/// ns per call, certificate size. Run with `--release -- --nocapture`.
#[test]
fn report_costs() {
    let st1 = store(1);
    let ops1 = data(1, 8, 5);
    let ch1 = chain(&st1, &ops1);
    let to = all_orders(5)
        .into_iter()
        .min_by_key(|o| ch1.eval(o).unwrap().1)
        .unwrap();
    let from = proposals().remove(0);
    let admitted = admit_composition_assoc(law().shape).unwrap();
    let t = Instant::now();
    let cert = certify_reassociation(&admitted, law(), &from, &to, compiled(&to)).unwrap();
    let cold_us = t.elapsed().as_secs_f64() * 1e6;

    let st2 = store(2);
    let ops2 = data(3, 8, 5);
    let ch2 = chain(&st2, &ops2);
    let iters = 20_000u32;
    let before = egglog_calls();
    let t = Instant::now();
    for _ in 0..iters {
        black_box(execute_certified(black_box(&cert), &ch2, &from, ABI).unwrap());
    }
    let hot_ns = t.elapsed().as_secs_f64() * 1e9 / f64::from(iters);
    assert_eq!(egglog_calls(), before);
    let t = Instant::now();
    for _ in 0..iters {
        black_box(ch2.eval(black_box(&from)).unwrap());
    }
    let eval_ns = t.elapsed().as_secs_f64() * 1e9 / f64::from(iters);
    // The proof is egglog's arena, whose heap size is not measured here; the
    // rest of the certificate is counted in bytes.
    let bytes = std::mem::size_of::<LawReference>()
        + std::mem::size_of::<CompiledReference>()
        + 32
        + std::mem::size_of_val(cert.program());
    println!(
        "L1 costs (reported, not gates): cold certify {cold_us:.0} us; \
         hot execute_certified {hot_ns:.0} ns/call; plain eval of the requested order {eval_ns:.0} ns/call; \
         certificate without proof {bytes} B, {} steps; proof {} nodes ({} B rendered, display only)",
        cert.program().len(),
        cert.proof().node_count(),
        cert.proof().render().len()
    );
}
