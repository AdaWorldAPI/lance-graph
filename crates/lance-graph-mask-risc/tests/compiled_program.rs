//! `CompiledProgram`: a program owned with its recognised lowering.

use lance_graph_mask_risc::{
    execute_compiled, CompiledProgram, Foreign, Lowering, MaskOp, Operand, Out, Planes, Pred,
    Program, Scratch, Terminal, Value,
};

fn range_count(lo: u32, hi: u32) -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: Pred::Range { lo, hi },
            under: None,
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

/// FAILS IF: the owned lowering differs from the one the program recognises,
/// or the compiled view does not answer like `Program::compile`.
#[test]
fn the_owned_lowering_is_the_recognised_one_and_answers_alike() {
    let p = range_count(3, 70);
    let owned = CompiledProgram::new(p.clone());
    assert_eq!(owned.lowering(), p.lowering());
    assert!(matches!(owned.lowering(), Lowering::Range(_)));
    assert_eq!(owned.program(), &p);
    let planes = Planes {
        n_rows: 100,
        masks: &[],
        lanes: &[],
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &[],
    };
    let mut s = Scratch::for_program(&p, 100).unwrap();
    let a = execute_compiled(
        &owned.compiled(),
        &planes,
        &foreign,
        &mut s,
        Out::None,
        0..100,
    )
    .unwrap();
    let b = execute_compiled(&p.compile(), &planes, &foreign, &mut s, Out::None, 0..100).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, Value::Count(67));
}
