//! Running a [`SetProgram`]: each let once into a scratch plane, then every
//! query over the resident planes and those scratch planes (D-RU-2 / U2).
//!
//! A let's plane is one more resident plane to the queries that read it, so
//! every fold stays a tree and keeps [`run`](crate::run)'s width-free scratch
//! bound. Nothing here loops over rows: each let and each query is one
//! [`run_with_program`] call.
//!
//! [`run_recompute`] is the reference: every query with its lets inlined,
//! run on its own. Both report the same [`ProgramCost`], so a caller can
//! compare what sharing saves (lane reads, ops) with what it costs (one
//! `n_rows / 8`-byte plane per let).

use lance_graph_contract::set_program::{ProgramError as ShapeError, SetProgram};
use lance_graph_mask_risc::{words_for, Planes};

use crate::ir::{Measure, Query, Rel};
use crate::lower::{run_with_program, Answer, RelError};
use crate::normalize::{from_set, normalize};

/// What running a program cost, counted from the folds that ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProgramCost {
    /// mask-risc programs executed.
    pub programs: usize,
    /// mask-risc ops executed.
    pub passes: usize,
    /// Comparisons lowered: each reads one whole value lane.
    pub lane_reads: usize,
    /// Plane loads lowered: each reads one whole bit plane.
    pub plane_reads: usize,
    /// Lets computed into scratch planes.
    pub planes_materialized: usize,
    /// Bytes of those planes.
    pub plane_bytes: usize,
    /// The most scratch slots any one fold used.
    pub max_scratch_slots: u32,
}

/// The answers, in query order, and their cost.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramRun {
    pub answers: Vec<Answer>,
    pub cost: ProgramCost,
}

/// Why a program was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramError {
    /// The program describes another version of the table.
    VersionMismatch { program: u64, table: u64 },
    /// The program's references are malformed.
    Shape(ShapeError),
    /// The program expects `expected` resident planes; the caller gave `got`.
    ResidentPlanes { expected: u32, got: usize },
    /// One measure per query is needed.
    Measures { queries: usize, measures: usize },
    /// Resident planes plus lets exceed the `u16` plane index.
    TooManyPlanes,
    /// A fold was refused.
    Rel(RelError),
}

impl From<RelError> for ProgramError {
    fn from(e: RelError) -> Self {
        ProgramError::Rel(e)
    }
}

fn check(
    p: &SetProgram,
    table_version: u64,
    planes: &Planes<'_>,
    measures: &[Measure],
) -> Result<(), ProgramError> {
    if p.version != table_version {
        return Err(ProgramError::VersionMismatch {
            program: p.version,
            table: table_version,
        });
    }
    // Before `validate`, so an oversized program is refused by the plane
    // limit it cannot meet rather than by its id arithmetic.
    if p.resident as usize + p.lets.len() > usize::from(u16::MAX) + 1 {
        return Err(ProgramError::TooManyPlanes);
    }
    p.validate().map_err(ProgramError::Shape)?;
    if planes.masks.len() != p.resident as usize {
        return Err(ProgramError::ResidentPlanes {
            expected: p.resident,
            got: planes.masks.len(),
        });
    }
    if measures.len() != p.queries.len() {
        return Err(ProgramError::Measures {
            queries: p.queries.len(),
            measures: measures.len(),
        });
    }
    Ok(())
}

fn plane_count(rel: &Rel) -> usize {
    match rel {
        Rel::Plane(_) => 1,
        Rel::And(cs) | Rel::Or(cs) => cs.iter().map(plane_count).sum(),
        Rel::Not(c) => plane_count(c),
        _ => 0,
    }
}

/// Runs one fold and adds what it cost.
fn fold(
    q: &Query,
    planes: &Planes<'_>,
    keep: Option<&mut [u64]>,
    cost: &mut ProgramCost,
) -> Result<Answer, ProgramError> {
    let rel = normalize(&q.rel);
    let (run, program) = run_with_program(q, planes, keep)?;
    cost.programs += run.programs;
    cost.passes += run.passes;
    if run.programs > 0 {
        cost.lane_reads += rel.atoms();
        cost.plane_reads += plane_count(&rel);
    }
    if let Some(p) = program {
        cost.max_scratch_slots = cost.max_scratch_slots.max(p.scratch_slots);
    }
    Ok(run.answer)
}

/// Runs the queries with `measures[i]` for `queries[i]`. A `Measure::Keep`
/// writes into the next sink of `keeps`, in query order.
///
/// Refused when the program was built for another version, its references
/// are malformed, `planes` does not hold exactly its resident planes, or the
/// measures do not match the queries.
pub fn run_program(
    p: &SetProgram,
    table_version: u64,
    planes: &Planes<'_>,
    measures: &[Measure],
    keeps: &mut [&mut [u64]],
) -> Result<ProgramRun, ProgramError> {
    check(p, table_version, planes, measures)?;
    let words = words_for(planes.n_rows);
    let mut cost = ProgramCost::default();
    let mut scratch: Vec<Vec<u64>> = vec![vec![0; words]; p.lets.len()];
    for k in 0..p.lets.len() {
        let (done, rest) = scratch.split_at_mut(k);
        let masks: Vec<&[u64]> = planes
            .masks
            .iter()
            .copied()
            .chain(done.iter().map(Vec::as_slice))
            .collect();
        let view = Planes {
            n_rows: planes.n_rows,
            masks: &masks,
            lanes: planes.lanes,
        };
        let q = Query {
            rel: from_set(&p.lets[k]),
            measure: Measure::Keep,
        };
        fold(&q, &view, Some(&mut rest[0]), &mut cost)?;
        cost.planes_materialized += 1;
        cost.plane_bytes += words * 8;
    }
    let masks: Vec<&[u64]> = planes
        .masks
        .iter()
        .copied()
        .chain(scratch.iter().map(Vec::as_slice))
        .collect();
    let view = Planes {
        n_rows: planes.n_rows,
        masks: &masks,
        lanes: planes.lanes,
    };
    let rels: Vec<Rel> = p.queries.iter().map(from_set).collect();
    answer_all(&rels, measures, &view, keeps, &mut cost).map(|answers| ProgramRun { answers, cost })
}

/// The reference for [`run_program`]: each query with its lets inlined, run
/// on its own over the resident planes. Same checks, same answers.
pub fn run_recompute(
    p: &SetProgram,
    table_version: u64,
    planes: &Planes<'_>,
    measures: &[Measure],
    keeps: &mut [&mut [u64]],
) -> Result<ProgramRun, ProgramError> {
    check(p, table_version, planes, measures)?;
    let mut cost = ProgramCost::default();
    let rels: Vec<Rel> = p.queries.iter().map(|q| from_set(&p.inline(q))).collect();
    answer_all(&rels, measures, planes, keeps, &mut cost)
        .map(|answers| ProgramRun { answers, cost })
}

fn answer_all(
    rels: &[Rel],
    measures: &[Measure],
    planes: &Planes<'_>,
    keeps: &mut [&mut [u64]],
    cost: &mut ProgramCost,
) -> Result<Vec<Answer>, ProgramError> {
    let mut sinks = keeps.iter_mut();
    rels.iter()
        .zip(measures)
        .map(|(rel, &measure)| {
            let sink = match measure {
                Measure::Keep => Some(&mut **sinks.next().ok_or(RelError::NoKeepSink)?),
                _ => None,
            };
            let q = Query {
                rel: rel.clone(),
                measure,
            };
            fold(&q, planes, sink, cost)
        })
        .collect()
}
