//! The one place a Quack [`Query`] is lowered and run.
//!
//! Every population operation in this crate is a `Filter` + `Agg` over
//! borrowed lanes and resident planes, lowered by `lance-graph-quack` and
//! executed by `lance-graph-mask-risc`. Nothing here iterates rows, builds a
//! hash table, or produces joined tuples. The caller owns every buffer:
//! scratch is one tile per slot (`Scratch::for_program`), a `Keep` lands in a
//! `words_for(n_rows)` bitmap, a `GroupReduce` in a `K`-slot sink.

use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, Out, Planes, Program, Scratch, Terminal, Value,
};
use lance_graph_quack::{lower, Agg, Col, Filter, GroupAddr, GroupAgg, Query};

/// Lower a directory query. Directory queries are fixed shapes built in this
/// crate, so a lowering failure is a programming error, not input.
pub(crate) fn program(filter: Filter, agg: Agg) -> Program {
    lower(&Query { filter, agg }).expect("directory queries lower")
}

fn run(p: &Program, planes: &Planes<'_>, foreign: &Foreign<'_>, out: Out<'_>) -> Value {
    let mut scratch = Scratch::for_program(p, planes.n_rows).expect("scratch carves");
    execute_into(p, planes, foreign, &mut scratch, out).expect("directory program runs")
}

/// Surviving rows as a `words_for(n_rows)` bitmap (`Agg::Rows` → `Keep`).
pub(crate) fn keep(p: &Program, planes: &Planes<'_>, foreign: &Foreign<'_>) -> Vec<u64> {
    debug_assert!(matches!(p.terminal, Terminal::Keep { .. }));
    let mut bits = vec![0u64; words_for(planes.n_rows)];
    if planes.n_rows > 0 {
        run(p, planes, foreign, Out::Mask(&mut bits));
    }
    bits
}

/// `GROUP BY key COUNT(*)` over the rows `filter` keeps, ADDED into `sink`
/// (whose length is the group universe; keys past it — e.g. `NONE` — drop).
pub(crate) fn group_count_into(
    filter: Filter,
    key: Col,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
    sink: &mut [i64],
) {
    if planes.n_rows == 0 || sink.is_empty() {
        return;
    }
    let p = program(
        filter,
        Agg::GroupReduce {
            key: GroupAddr::Local(key),
            agg: GroupAgg::Count,
        },
    );
    let mut part = vec![0i64; sink.len()];
    let v = run(&p, planes, foreign, Out::I64(&mut part));
    debug_assert_eq!(v, Value::GroupReduced);
    for (s, x) in sink.iter_mut().zip(part) {
        *s += x;
    }
}
