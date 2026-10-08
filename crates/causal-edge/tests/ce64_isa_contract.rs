//! CE64 ISA contract: the decoder, the register methods, and each
//! instruction's declared field contract, measured.
//!
//! Reasoning chooses the operation. `CausalEdge64` defines what that operation
//! means.
#![cfg(feature = "causal-edge-v2-layout")]

use causal_edge::isa::{contracts, truth, Contract, Field, IsaFault, Opcode, Operand};
use causal_edge::{CausalEdge64, Compose};

fn tables() -> [Box<[u8; 65536]>; 3] {
    let mk = |a: usize, b: usize| {
        let mut t = vec![0u8; 65536].into_boxed_slice();
        for x in 0..256 {
            for y in 0..256 {
                t[x * 256 + y] = (x * a + y * b) as u8;
            }
        }
        t.try_into().unwrap()
    };
    [mk(31, 17), mk(7, 13), mk(11, 29)]
}

fn compose(t: &[Box<[u8; 65536]>; 3]) -> Compose<'_> {
    Compose {
        s: &t[0],
        p: &t[1],
        o: &t[2],
    }
}

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

fn set(word: u64, f: Field, v: u64) -> u64 {
    (word & !f.mask()) | ((v << f.span().0) & f.mask())
}

// ─── Decoder ────────────────────────────────────────────────────────────

/// Exactly five of the sixteen codes decode; every other code is refused
/// with that same code.
#[test]
fn the_decoder_accepts_exactly_the_implemented_codes() {
    let mut accepted = Vec::new();
    for m in -8i8..=7 {
        match Opcode::decode(m) {
            Ok(op) => {
                assert_eq!(op.encoding(), m, "decode is the inverse of encoding");
                accepted.push(m);
            }
            Err(IsaFault::Unsupported { mantissa }) => assert_eq!(mantissa, m),
        }
    }
    accepted.sort_unstable();
    assert_eq!(accepted, vec![-1, 1, 2, 4, 5]);
}

/// Counterfactual, intervention and reserved codes fault in `forward`; they
/// are never executed as another instruction and re-stamped.
#[test]
fn counterfactual_intervention_and_reserved_fault_in_forward() {
    let t = tables();
    let running = CausalEdge64(0x6ab8_2fff_ff03_0201);
    for m in [-6i8, 6, 7, -7] {
        let weight = CausalEdge64(0xb54c_13fe_0106_0504).with_inference_mantissa(m);
        assert_eq!(
            running.forward(weight, &t[0], &t[1], &t[2]),
            Err(IsaFault::Unsupported { mantissa: m })
        );
    }
}

/// opcode = X executes exactly instruction X: `forward` on a weight carrying
/// X equals the named register method, bit for bit, and differs from every
/// other instruction on the same operands.
#[test]
fn forward_executes_exactly_the_decoded_instruction() {
    let t = tables();
    let c = compose(&t);
    let mut r = Rng(11);
    let mut discriminated = 0;
    for _ in 0..400 {
        let running = CausalEdge64(r.next());
        let base = CausalEdge64(r.next());
        for op in Opcode::ALL {
            let weight = base.with_inference_mantissa(op.encoding());
            let via_forward = running.forward(weight, c.s, c.p, c.o).unwrap();
            assert_eq!(via_forward, running.execute(op, weight, c));
            let named = match op {
                Opcode::Deduction => running.deduction(weight, c),
                Opcode::Induction => running.induction(weight, c),
                Opcode::Abduction => running.abduction(weight, c),
                Opcode::Revision => running.execute(Opcode::Revision, weight, c),
                Opcode::Synthesis => running.synthesis(weight, c),
            };
            assert_eq!(via_forward, named);
            // The output carries the instruction that ran.
            assert_eq!(via_forward.inference_mantissa(), op.encoding());
            let others_differ = Opcode::ALL.iter().filter(|&&o| o != op).all(|&o| {
                let w = base.with_inference_mantissa(o.encoding());
                (
                    running.execute(o, w, c).frequency_u8(),
                    running.execute(o, w, c).confidence_u8(),
                ) != (via_forward.frequency_u8(), via_forward.confidence_u8())
            });
            discriminated += usize::from(others_differ);
        }
    }
    // Anti-vacuity: on most operands every instruction is distinguishable.
    assert!(
        discriminated > 1500,
        "only {discriminated}/2000 discriminated"
    );
}

/// The register methods route through the one canonical truth function per
/// instruction: their F/C equal `Opcode::truth` packed, over a u8 grid.
#[test]
fn every_path_uses_the_canonical_truth_function() {
    let t = tables();
    let c = compose(&t);
    let pack = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    for fa in (0..=255u8).step_by(17) {
        for ca in (0..=254u8).step_by(23) {
            for fb in (0..=255u8).step_by(19) {
                for cb in (0..=254u8).step_by(29) {
                    let mut a = CausalEdge64(0);
                    a.set_frequency_u8(fa);
                    a.set_confidence_u8(ca);
                    let mut b = CausalEdge64(0);
                    b.set_frequency_u8(fb);
                    b.set_confidence_u8(cb);
                    let (f1, c1, f2, c2) =
                        (a.frequency(), a.confidence(), b.frequency(), b.confidence());
                    for op in Opcode::ALL {
                        let (f, cc) = op.truth(f1, c1, f2, c2);
                        let out = a.execute(op, b, c);
                        assert_eq!(
                            (out.frequency_u8(), out.confidence_u8()),
                            (pack(f), pack(cc))
                        );
                    }
                    // learn's revision is the same function.
                    let mut l = a;
                    l.learn(b, 0);
                    match truth::revision(f1, c1, f2, c2) {
                        Some((f, cc)) => {
                            assert_eq!((l.frequency_u8(), l.confidence_u8()), (pack(f), pack(cc)))
                        }
                        None => assert_eq!((l.frequency_u8(), l.confidence_u8()), (fa, ca)),
                    }
                }
            }
        }
    }
}

/// Partial instructions refuse with their own code; they never compute.
#[test]
fn counterfactual_and_intervention_methods_refuse() {
    let t = tables();
    let c = compose(&t);
    let (a, b) = (
        CausalEdge64(0x6ab8_2fff_ff03_0201),
        CausalEdge64(0xb54c_13fe_0106_0504),
    );
    assert_eq!(
        a.counterfactual(b, c),
        Err(IsaFault::Unsupported { mantissa: -6 })
    );
    assert_eq!(
        a.intervention(b, c),
        Err(IsaFault::Unsupported { mantissa: 6 })
    );
}

/// `Compose` is operand algebra only: two different payload algebras change
/// S/P/O and nothing else. Every ISA-owned field is identical.
#[test]
fn compose_changes_only_the_payload() {
    let t1 = tables();
    let mk = |a: usize, b: usize| -> Box<[u8; 65536]> {
        let mut t = vec![0u8; 65536].into_boxed_slice();
        for x in 0..256 {
            for y in 0..256 {
                t[x * 256 + y] = (x * a ^ y * b) as u8;
            }
        }
        t.try_into().unwrap()
    };
    let t2 = [mk(3, 5), mk(9, 1), mk(13, 7)];
    let (c1, c2) = (compose(&t1), compose(&t2));
    let mut r = Rng(5);
    let mut payload_differs = 0;
    for _ in 0..500 {
        let (a, b) = (CausalEdge64(r.next()), CausalEdge64(r.next()));
        for op in Opcode::ALL {
            let (x, y) = (a.execute(op, b, c1).0, a.execute(op, b, c2).0);
            assert_eq!(x & !Field::Spo.mask(), y & !Field::Spo.mask(), "{op:?}");
            payload_differs += usize::from(x != y);
        }
    }
    assert!(
        payload_differs > 2000,
        "the two algebras must actually differ"
    );
}

// ─── Field contracts, measured ──────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Instr {
    Forward,
    Learn,
    Syllogize,
    Revision,
}

fn exec(i: Instr, a: u64, b: u64, c: Compose<'_>) -> Option<u64> {
    let (a, b) = (CausalEdge64(a), CausalEdge64(b));
    match i {
        Instr::Forward => a.forward(b, c.s, c.p, c.o).ok().map(|e| e.0),
        Instr::Learn => {
            let mut e = a;
            e.learn(b, 0);
            Some(e.0)
        }
        Instr::Syllogize => a.syllogize(b).map(|s| s.conclusion.0),
        Instr::Revision => Some(a.revision(b).0),
    }
}

/// Operands valid for the instruction: forward needs an executable weight
/// code; syllogize needs a figure (chain: a.O == b.S).
fn operands(i: Instr, r: &mut Rng) -> (u64, u64) {
    let (a, mut b) = (r.next(), r.next());
    match i {
        Instr::Forward => {
            let op = Opcode::ALL[(r.next() % 5) as usize];
            b = set(b, Field::Inference, u64::from(op.encoding() as u8 & 0xF));
        }
        Instr::Syllogize => {
            b = (b & !0xFF) | ((a >> 16) & 0xFF);
        }
        Instr::Learn | Instr::Revision => {}
    }
    (a, b)
}

fn varied(i: Instr, word: u64, f: Field, r: &mut Rng) -> u64 {
    let bits = (1u64 << f.span().1) - 1;
    let mut v = (f.get(word) + 1 + r.next() % bits) & bits;
    if matches!(i, Instr::Forward) && f == Field::Inference {
        // Stay inside the decodable set: vary to another executable code.
        let cur = f.get(word);
        let ops: Vec<u64> = Opcode::ALL
            .iter()
            .map(|o| u64::from(o.encoding() as u8 & 0xF))
            .filter(|&e| e != cur)
            .collect();
        v = ops[(r.next() % ops.len() as u64) as usize];
    }
    set(word, f, v)
}

fn check(contract: Contract, i: Instr) {
    let t = tables();
    let c = compose(&t);
    // Every output field is declared exactly once.
    // A Spo output that is computed by composition needs an algebra.
    assert_eq!(
        contract.needs_compose,
        matches!(i, Instr::Forward),
        "{}: needs_compose",
        contract.name
    );
    for f in Field::ALL {
        let n = usize::from(contract.computes.contains(&f))
            + contract.passes.iter().filter(|(g, _)| *g == f).count()
            + contract.constants.iter().filter(|(g, _)| *g == f).count();
        assert_eq!(n, 1, "{}: {f:?} declared {n} times", contract.name);
    }
    let mut r = Rng(0xC0FFEE);
    // passes / constants hold on every sample.
    for _ in 0..2000 {
        let (a, b) = operands(i, &mut r);
        let out = exec(i, a, b, c).expect("valid operands");
        for &(f, side) in contract.passes {
            let src = if side == Operand::A { a } else { b };
            assert_eq!(f.get(out), f.get(src), "{}: {f:?} passes", contract.name);
        }
        for &(f, v) in contract.constants {
            assert_eq!(f.get(out), v, "{}: {f:?} constant", contract.name);
        }
    }
    // reads: exactly the declared operand fields influence an output that is
    // not their own pass-through.
    let mut measured = Vec::new();
    for side in [Operand::A, Operand::B] {
        for f in Field::ALL {
            let mut hit = false;
            for _ in 0..600 {
                let (a, b) = operands(i, &mut r);
                let (a2, b2) = if side == Operand::A {
                    (varied(i, a, f, &mut r), b)
                } else {
                    (a, varied(i, b, f, &mut r))
                };
                // Keep the syllogism's chain figure intact under the variation.
                let b2 = if matches!(i, Instr::Syllogize) {
                    (b2 & !0xFF) | ((a2 >> 16) & 0xFF)
                } else {
                    b2
                };
                let (Some(o1), Some(o2)) = (exec(i, a, b, c), exec(i, a2, b2, c)) else {
                    continue;
                };
                let passthrough = contract.passes.contains(&(f, side));
                let diff = Field::ALL
                    .iter()
                    .any(|&g| !(passthrough && g == f) && g.get(o1) != g.get(o2));
                if diff {
                    hit = true;
                    break;
                }
            }
            if hit {
                measured.push((side, f));
            }
        }
    }
    let mut declared = contract.reads.to_vec();
    let key = |x: &(Operand, Field)| (x.0 as u8, x.1 as u8);
    declared.sort_by_key(key);
    measured.sort_by_key(key);
    assert_eq!(measured, declared, "{}: reads", contract.name);
}

#[test]
fn forward_contract_holds() {
    check(contracts::FORWARD, Instr::Forward);
}

#[test]
fn learn_contract_holds() {
    check(contracts::LEARN, Instr::Learn);
}

#[test]
fn revision_contract_holds() {
    check(contracts::REVISION, Instr::Revision);
}

#[test]
fn syllogize_contract_holds() {
    check(contracts::SYLLOGIZE, Instr::Syllogize);
}

/// No instruction zeroes a field without declaring it a constant: W and
/// Epi5 are either passed through or declared constant 0.
#[test]
fn witness_and_epistemic_are_never_cleared_undeclared() {
    for k in contracts::ALL {
        for f in [
            Field::Witness,
            Field::Epistemic,
            Field::Direction,
            Field::Pearl,
        ] {
            let declared = k.computes.contains(&f)
                || k.passes.iter().any(|(g, _)| *g == f)
                || k.constants.iter().any(|(g, _)| *g == f);
            assert!(declared, "{}: {f:?}", k.name);
        }
    }
}

// ─── Layering ───────────────────────────────────────────────────────────

/// `isa.rs` defines arithmetic only. Reasoning-level policy (Gadamer
/// revision, horizons, grammar, codebooks, lenses, recipes) stays outside.
#[test]
fn isa_rs_names_no_reasoning_policy() {
    let src = include_str!("../src/isa.rs");
    let code: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for banned in [
        "lance_graph_contract",
        "Gadamer",
        "RevisionPolicy",
        "Horizon",
        "Grammar",
        "Codebook",
        "Lens",
        "ThoughtCtx",
        "Recipe",
    ] {
        assert!(!code.contains(banned), "isa.rs names {banned}");
    }
}
