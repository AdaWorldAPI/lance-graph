//! The one place a Quack [`Query`] is lowered and run.
//!
//! Every population operation in this crate is a `Filter` + `Agg` over
//! borrowed lanes and resident planes, lowered by `lance-graph-quack` and
//! executed by `lance-graph-mask-risc`. Nothing here iterates rows, builds a
//! hash table, or produces joined tuples. The caller owns every buffer:
//! scratch is one tile per slot (`Scratch::for_program`), a `Keep` lands in a
//! `words_for(n_rows)` bitmap, a `GroupReduce` in a `K`-slot sink.

use lance_graph_mask_risc::{
    execute_into, materialize_rows, words_for, Foreign, Out, Planes, Program, Scratch, Terminal, Value,
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

/// The rows one program kept. Its only exit is [`Kept::rows`], the evidence
/// boundary: it has no `&[u64]` view, so it cannot become another program's
/// `ForeignPlane`. A `Semijoin` gathers only from resident planes (node kinds,
/// the active-user plane), never from a population a program produced.
///
/// ```compile_fail
/// use lance_graph_dir_sim::Kept;
/// use lance_graph_mask_risc::ForeignPlane;
/// fn feed(k: &Kept) -> ForeignPlane<'_> {
///     ForeignPlane { words: k, rows: 0 }
/// }
/// ```
///
/// The same imports and shape compile against a resident plane:
///
/// ```
/// use lance_graph_dir_sim::Kept;
/// use lance_graph_mask_risc::ForeignPlane;
/// fn feed<'a>(_k: &Kept, resident: &'a [u64]) -> ForeignPlane<'a> {
///     ForeignPlane { words: resident, rows: 0 }
/// }
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct Kept {
    bits: Vec<u64>,
    n_rows: usize,
}

impl Kept {
    /// Kept row indices, ascending. Bounded by the number of survivors.
    pub fn rows(&self) -> Vec<usize> {
        materialize_rows(&self.bits, self.n_rows)
    }
}

/// Surviving rows (`Agg::Rows` → `Keep`), sealed in a [`Kept`].
pub(crate) fn keep(p: &Program, planes: &Planes<'_>, foreign: &Foreign<'_>) -> Kept {
    debug_assert!(matches!(p.terminal, Terminal::Keep { .. }));
    let mut bits = vec![0u64; words_for(planes.n_rows)];
    if planes.n_rows > 0 {
        run(p, planes, foreign, Out::Mask(&mut bits));
    }
    Kept {
        bits,
        n_rows: planes.n_rows,
    }
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
