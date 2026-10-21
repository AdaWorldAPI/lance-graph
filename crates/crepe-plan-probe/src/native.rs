//! Native lowering of an admitted `query_multi` star onto the D-MPS-8
//! versioned relationship table, through relalg into one mask-risc program.
//!
//! STATUS: PROBE. The table (`mork_pathmap_probe::join`) is measured but not
//! admitted (D-RMAP-1's binding membrane waits on EXO-46). Nothing here is a
//! production path.
//!
//! # The only shape lowered
//!
//! A star on one shared variable `c`: every pattern names `c`, and every other
//! variable occurs in exactly one pattern. Each pattern becomes one *side*, laid
//! out by the slot of `c` over the shared domain. A constant on the side's other
//! column restricts the rows the side is prepared from (cold, at preparation,
//! once per prepared table). Then
//!
//! ```text
//! joint = present_0 ∧ present_1 ∧ … ∧ present_k        (relalg: And of Plane(i))
//! ```
//!
//! is the set of `c` values every side has. That mask is all relalg computes.
//! Existence and distinct-`c` counts read only it; tuple emission and bag
//! counts are a separate terminal that reads each side at the slot the mask
//! names, never at a position taken from another side's values.
//!
//! Anything else (a chain, a pattern of arity > 2, a side without `c`) is
//! refused by name.

use lance_graph_mask_risc::{MaskOp, Planes, Pred, Program, Terminal, words_for};
use lance_graph_relalg::{Answer, Measure, Query, Rel, RelError, Run, run_with_program};
use mork_pathmap_probe::RelationView;
use mork_pathmap_probe::join::{JoinDomain, JoinRefusal, JoinTable, Side, for_each_set};

use crate::logic::{RelId, VarId};
use crate::mork_api::{Expr, QueryMulti};

/// Which rows of which relation a side holds, and how they are keyed. Two
/// sides are interchangeable only when their keys are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SideKey {
    pub rel: RelId,
    /// The column holding the shared variable (`0` for a unary relation).
    pub key_col: u8,
    /// `Some(k)`: only rows whose other column is `k`.
    pub restrict: Option<u64>,
    pub unary: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidePlan {
    pub key: SideKey,
    /// The variable this side's other column binds, if the template uses it.
    pub emit: Option<VarId>,
}

/// Where an output column comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadTerm {
    Const(u64),
    /// The shared variable.
    Center,
    /// The value column of side `i`.
    Side(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StarPlan {
    pub center: VarId,
    pub sides: Vec<SidePlan>,
    pub head: Vec<HeadTerm>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeRefusal {
    /// No variable occurs in every pattern.
    NoSharedVariable,
    /// A variable other than the center occurs in two patterns (a chain or a
    /// second join variable).
    NotAStar {
        var: VarId,
    },
    /// The D-MPS-8 sides are binary or unary.
    Arity {
        pattern: usize,
        arity: usize,
    },
    /// A template variable no pattern binds.
    UnboundTemplateVar(VarId),
    /// The prepared table was built for other sides than this plan reads.
    NoCoordinateWitness,
    /// A side was laid out over another domain than the table's.
    DomainMismatch {
        side: usize,
    },
    /// The caller passed a generation vector of another length than the
    /// table has sides; `JoinTable::check` would skip the missing ones.
    Generations {
        sides: usize,
        given: usize,
    },
    /// From the table: a stale generation, or rows outside the domain.
    Table(JoinRefusal),
    Relalg(RelError),
}

/// Reads the star out of an admitted pattern.
pub fn star(q: &QueryMulti) -> Result<StarPlan, NativeRefusal> {
    let args: Vec<&[Expr]> = q.patterns.iter().map(|p| &p[1..]).collect();
    let rel = |i: usize| match q.patterns[i][0] {
        Expr::Sym(r) => r,
        _ => unreachable!("admitted patterns start with a symbol"),
    };
    let vars_of = |a: &[Expr]| -> Vec<VarId> {
        a.iter()
            .filter_map(|e| match e {
                Expr::Var(v) => Some(*v),
                _ => None,
            })
            .collect()
    };
    let first = vars_of(args.first().copied().unwrap_or(&[]));
    let center = first
        .into_iter()
        .find(|v| args.iter().all(|a| vars_of(a).contains(v)))
        .ok_or(NativeRefusal::NoSharedVariable)?;
    let mut sides = Vec::new();
    for (i, a) in args.iter().enumerate() {
        let key_col = a
            .iter()
            .position(|e| *e == Expr::Var(center))
            .expect("center") as u8;
        let (restrict, emit, unary) = match a.len() {
            1 => (None, None, true),
            2 => match &a[1 - key_col as usize] {
                Expr::Int(k) => (Some(*k), None, false),
                Expr::Var(v) => (None, Some(*v), false),
                _ => unreachable!("admitted patterns are flat"),
            },
            n => {
                return Err(NativeRefusal::Arity {
                    pattern: i,
                    arity: n,
                });
            }
        };
        sides.push(SidePlan {
            key: SideKey {
                rel: rel(i),
                key_col,
                restrict,
                unary,
            },
            emit,
        });
    }
    for (i, s) in sides.iter().enumerate() {
        if let Some(v) = s.emit
            && sides
                .iter()
                .enumerate()
                .any(|(j, t)| j != i && t.emit == Some(v))
        {
            return Err(NativeRefusal::NotAStar { var: v });
        }
    }
    let head = q.template[1..]
        .iter()
        .map(|e| match *e {
            Expr::Int(c) => Ok(HeadTerm::Const(c)),
            Expr::Var(v) if v == center => Ok(HeadTerm::Center),
            Expr::Var(v) => sides
                .iter()
                .position(|s| s.emit == Some(v))
                .map(HeadTerm::Side)
                .ok_or(NativeRefusal::UnboundTemplateVar(v)),
            _ => unreachable!("templates are flat"),
        })
        .collect::<Result<_, _>>()?;
    Ok(StarPlan {
        center,
        sides,
        head,
    })
}

/// The relations a plan reads, by id, with their generations.
pub struct Store<'a> {
    /// `(relation, rows, generation)`. A unary relation's rows are `[v, v]`.
    pub relations: Vec<(RelId, &'a [[u32; 2]], u64)>,
}

impl<'a> Store<'a> {
    pub fn view(&self, rel: RelId) -> (&'a [[u32; 2]], u64) {
        let (_, rows, g) = self
            .relations
            .iter()
            .find(|(r, _, _)| *r == rel)
            .expect("relation in store");
        (rows, *g)
    }
}

/// A relationship table prepared for one plan's sides, with the witness that
/// says which sides those are and which domain they share.
pub struct PreparedStar {
    pub table: JoinTable,
    pub keys: Vec<SideKey>,
    /// Fingerprint of the domain every side is laid out over.
    pub domain_fp: u64,
    /// Rows laid out, over all sides (after restriction).
    pub rows_prepared: usize,
}

/// Lays the plan's sides out over one shared domain. A side's restriction is
/// applied to the rows it is prepared from.
pub fn prepare(plan: &StarPlan, store: &Store<'_>) -> PreparedStar {
    prepare_with(plan, store, false)
}

/// [`prepare`] with semijoin reduction: when at least one side is restricted
/// by a constant, every side keeps only the rows whose center the restricted
/// sides all have. A center outside that set cannot reach the joint mask, so
/// the answer and its multiplicities are unchanged; the domain shrinks to the
/// reduced centers. Without a restricted side nothing changes.
pub fn prepare_semijoin(plan: &StarPlan, store: &Store<'_>) -> PreparedStar {
    prepare_with(plan, store, true)
}

fn prepare_with(plan: &StarPlan, store: &Store<'_>, semijoin: bool) -> PreparedStar {
    let mut owned: Vec<(Vec<[u32; 2]>, u64)> = plan
        .sides
        .iter()
        .map(|s| {
            let (rows, g) = store.view(s.key.rel);
            let other = 1 - s.key.key_col as usize;
            let kept = rows
                .iter()
                .filter(|r| s.key.unary || s.key.restrict.is_none_or(|k| u64::from(r[other]) == k))
                .copied()
                .collect();
            (kept, g)
        })
        .collect();
    if semijoin {
        // Centers every restricted side has: sorted, deduplicated, intersected.
        let mut centers: Option<Vec<u32>> = None;
        for (s, (rows, _)) in plan.sides.iter().zip(&owned) {
            if s.key.unary || s.key.restrict.is_none() {
                continue;
            }
            let mut c: Vec<u32> = rows.iter().map(|r| r[s.key.key_col as usize]).collect();
            c.sort_unstable();
            c.dedup();
            centers = Some(match centers {
                None => c,
                Some(prev) => prev
                    .into_iter()
                    .filter(|v| c.binary_search(v).is_ok())
                    .collect(),
            });
        }
        if let Some(centers) = centers {
            for (s, (rows, _)) in plan.sides.iter().zip(owned.iter_mut()) {
                let col = s.key.key_col as usize;
                rows.retain(|r| centers.binary_search(&r[col]).is_ok());
            }
        }
    }
    prepare_from_rows(plan, owned)
}

/// Lays out rows already selected per side (`(rows, generation)`, in plan
/// side order) over one shared domain.
pub fn prepare_from_rows(plan: &StarPlan, owned: Vec<(Vec<[u32; 2]>, u64)>) -> PreparedStar {
    let inputs: Vec<(RelationView<'_>, usize)> = owned
        .iter()
        .zip(&plan.sides)
        .map(|((rows, g), s)| {
            (
                RelationView {
                    rows,
                    generation: *g,
                },
                s.key.key_col as usize,
            )
        })
        .collect();
    let table = JoinTable::prepare_star(&inputs);
    PreparedStar {
        domain_fp: fingerprint(&table),
        keys: plan.sides.iter().map(|s| s.key).collect(),
        rows_prepared: owned.iter().map(|(r, _)| r.len()).sum(),
        table,
    }
}

/// FNV-1a over the domain's values: equal domains, equal fingerprints.
pub fn fingerprint(table: &JoinTable) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for slot in 0..table.domain.len() {
        for b in table.domain.id(slot).to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    h ^ table.domain.len() as u64
}

impl PreparedStar {
    /// Replaces side `i` with one prepared elsewhere. Refused unless that side
    /// was laid out over this table's domain: equal slot numbers over two
    /// different domains name different values.
    pub fn graft(
        &mut self,
        i: usize,
        side: Side,
        key: SideKey,
        side_domain_fp: u64,
    ) -> Result<(), NativeRefusal> {
        if side_domain_fp != self.domain_fp {
            return Err(NativeRefusal::DomainMismatch { side: i });
        }
        self.table.sides[i] = side;
        self.keys[i] = key;
        Ok(())
    }

    /// The coordinate witness: this table holds exactly the plan's sides.
    pub fn witness(&self, plan: &StarPlan) -> Result<(), NativeRefusal> {
        let wanted: Vec<SideKey> = plan.sides.iter().map(|s| s.key).collect();
        if wanted != self.keys {
            return Err(NativeRefusal::NoCoordinateWitness);
        }
        Ok(())
    }
}

/// What the native path answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeAnswer {
    Exists(bool),
    Count(u64),
    /// Output tuples with their multiplicities (all 1 under set semantics).
    Bindings(Vec<(Vec<u64>, u64)>),
}

/// The joint mask and its relalg run, after the witness and generation checks.
pub fn joint(
    prep: &PreparedStar,
    plan: &StarPlan,
    current: &[u64],
    measure: Measure,
    keep: Option<&mut [u64]>,
) -> Result<(Run, Option<Program>), NativeRefusal> {
    prep.witness(plan)?;
    if current.len() != prep.table.sides.len() {
        return Err(NativeRefusal::Generations {
            sides: prep.table.sides.len(),
            given: current.len(),
        });
    }
    prep.table.check(current).map_err(NativeRefusal::Table)?;
    let masks: Vec<&[u64]> = prep.table.sides.iter().map(|s| s.present()).collect();
    let planes = Planes {
        n_rows: prep.table.domain.len(),
        masks: &masks,
        lanes: &[],
    };
    let rel = Rel::And((0..masks.len() as u16).map(Rel::Plane).collect());
    run_with_program(&Query { rel, measure }, &planes, keep).map_err(NativeRefusal::Relalg)
}

/// Number of distinct center values every side has: relalg's popcount of the
/// joint mask, no payload read.
pub fn distinct_centers(
    prep: &PreparedStar,
    plan: &StarPlan,
    current: &[u64],
) -> Result<u64, NativeRefusal> {
    match joint(prep, plan, current, Measure::Count, None)?.0.answer {
        Answer::Count(n) => Ok(n),
        other => unreachable!("Count measure answered {other:?}"),
    }
}

/// Tuples over the joint mask. Reads each side at the slot the mask names.
/// A side whose value the template does not use still multiplies the count of
/// every tuple under bag semantics: each of its rows at that slot is one more
/// derivation.
pub fn bindings(
    prep: &PreparedStar,
    plan: &StarPlan,
    current: &[u64],
    bag: bool,
) -> Result<Vec<(Vec<u64>, u64)>, NativeRefusal> {
    bindings_run(prep, plan, current, bag).map(|(rows, _, _)| rows)
}

/// Rows with multiplicities, the relalg run, and the program behind the mask.
pub type BindingsRun = (Vec<(Vec<u64>, u64)>, Run, Option<Program>);

/// [`bindings`], plus the relalg run and the program that produced the mask.
pub fn bindings_run(
    prep: &PreparedStar,
    plan: &StarPlan,
    current: &[u64],
    bag: bool,
) -> Result<BindingsRun, NativeRefusal> {
    let n = prep.table.domain.len();
    let mut mask = vec![0u64; words_for(n)];
    let (run, program) = joint(prep, plan, current, Measure::Keep, Some(&mut mask))?;
    let sides: Vec<&Side> = prep.table.sides.iter().collect();
    let rows = tuples_over_mask(&prep.table.domain, &sides, plan, &mask, bag);
    Ok((rows, run, program))
}

/// The output tuples of `plan` at the slots `mask` names, with
/// multiplicities. `sides[i]` is the laid-out side for `plan.sides[i]`, over
/// `domain`. Each side is read at the slot the mask names, never at a
/// position taken from another side's values. Without `bag`, every
/// multiplicity is 1.
pub fn tuples_over_mask(
    domain: &JoinDomain,
    sides: &[&Side],
    plan: &StarPlan,
    mask: &[u64],
    bag: bool,
) -> Vec<(Vec<u64>, u64)> {
    let n = domain.len();
    let mut out: Vec<(Vec<u64>, u64)> = Vec::new();
    for_each_set(mask, n, |slot| {
        let center = u64::from(domain.id(slot));
        let mut tuples: Vec<(Vec<Option<u64>>, u64)> = vec![(vec![None; plan.sides.len()], 1)];
        for (i, s) in plan.sides.iter().enumerate() {
            let vals = sides[i].at(slot);
            if s.emit.is_some() {
                tuples = tuples
                    .into_iter()
                    .flat_map(|(t, m)| {
                        vals.iter().map(move |&v| {
                            let mut t = t.clone();
                            t[i] = Some(u64::from(v));
                            (t, m)
                        })
                    })
                    .collect();
            } else {
                let k = vals.len() as u64;
                for t in &mut tuples {
                    t.1 *= k;
                }
            }
        }
        for (t, m) in tuples {
            let row: Vec<u64> = plan
                .head
                .iter()
                .map(|h| match *h {
                    HeadTerm::Const(c) => c,
                    HeadTerm::Center => center,
                    HeadTerm::Side(i) => t[i].expect("emitting side"),
                })
                .collect();
            out.push((row, m));
        }
    });
    out.sort();
    let mut merged: Vec<(Vec<u64>, u64)> = Vec::new();
    for (row, m) in out {
        match merged.last_mut() {
            Some((r, c)) if *r == row => *c += m,
            _ => merged.push((row, m)),
        }
    }
    if !bag {
        for e in &mut merged {
            e.1 = 1;
        }
    }
    merged
}

/// True when the program contains an op that reads at a position taken from
/// another lane's values, or writes by one (the 2026-10-16 ruling): a
/// `Gather`, an `EqU32Via` predicate, a scatter terminal or `GroupSumViaI32`.
pub fn has_foreign_hop(p: &Program) -> bool {
    let op_hops = p.ops.iter().any(|op| match op {
        MaskOp::Gather { .. } => true,
        MaskOp::Pred { pred, .. } => matches!(pred, Pred::EqU32Via { .. }),
        _ => false,
    });
    let terminal_hops = matches!(
        p.terminal,
        Terminal::ScatterOrU32 { .. }
            | Terminal::ScatterCountU32 { .. }
            | Terminal::GroupSumViaI32 { .. }
    );
    op_hops || terminal_hops
}
