//! D-COGNITIVE-POLYFILL P0 — four frontends, one native algebra (LAB ONLY).
//!
//! STATUS: PROBE. Nothing here is wired into Quack, R2IL, RelAlg or any
//! production path, and nothing here parses a language. Each frontend's query
//! is hand-lowered into one [`Pattern`], the way D-MTP-1
//! (`lance-graph-metta/examples/join_probe.rs`) hand-lowers its MeTTa query.
//!
//! # The common subproblem
//!
//! Ground facts of one binary relation `R`, and the bounded existential
//! "is there a `y` in the selection `W` with `R(a, y)` and `R(y, c)`":
//!
//! | frontend | spelling of the same subproblem |
//! |---|---|
//! | MeTTa (Hyperon) | `(match &self (, (R A $y) (R $y C)) $y)` |
//! | MORK | `(exec P (, (R A $y) (R $y C)) (, (hit $y)))` |
//! | Atomese (typed projection) | `(EvaluationLink (PredicateNode "R") (ListLink A $y))` ∧ `(… (ListLink $y C))` |
//! | Narsese | NO direct spelling: a relation `R(A, B)` needs the product term `<(*, A, B) --> R>`, a compound term the NAL crate refuses (ledger R10); `<A --> ?x>` is one statement, and `<A-->B>, <B-->C> ⊢ <A-->C>` is an INFERENCE with a truth value, not an existence test. |
//!
//! These are equivalent only on this subset. The probe does not claim language
//! equivalence; it measures where each frontend first needs something the
//! substrate does not have.
//!
//! # Native arm
//!
//! - Identity: one DR tile ([`lance_graph_dt64::dr::LocalMap`], `Width::W64`)
//!   under a [`MapTable`] frame. Every term is a `u64` global id; its column in
//!   the planes is its local position in that tile.
//! - Relation: per source a presence word over targets (`out`), and per target
//!   a presence word over sources (`inn`), both laid out by the same tile's
//!   local positions. `inn` is a derived index built once at ingest (the
//!   co-addressed second side, the same shape as `mork-pathmap-probe::join::Side`);
//!   it is not a second authority and is rebuilt with `out`.
//! - Query: `Any` / `Count` over `out[a] ∧ inn[c] ∧ W` — ONE mask-risc program,
//!   prepared once, run over borrowed slices of the relation's own storage. No
//!   triple, no pair, no `(x, y, z)` row is formed.
//! - Selection routing: a selection from another tile is routed into the
//!   relation tile ONCE per frame pair ([`Selection::route`]); joining on raw
//!   local positions is a different (wrong) answer.
//!
//! # What does not lower
//!
//! [`lower`] refuses every pattern outside the subset with a reason: repeated
//! variables, free endpoints (binding enumeration), nested terms, a second
//! relation name. Bag multiplicity is answered by a lab fold over the kept
//! mask ([`Relation::bag_derivations`]); the existing native route for it is
//! `lance_graph_xlog::bag_count` (EXO-108 / MISSING-3), see that method.

#![forbid(unsafe_code)]

use lance_graph_dt64::dr::{Frame, LocalMap, MapRead, MapRefusal, MapTable, TileId, Width};
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, MaskOp, Operand, Out, Planes, Program, Scratch, Terminal,
    Value,
};

/// The one tile every probe relation lives in.
pub const TILE: TileId = 0;
/// Tile width: 64 positions, so one presence row is one `u64` word.
pub const WIDTH: Width = Width::W64;

// ---------------------------------------------------------------------------
// Frontend descriptor
// ---------------------------------------------------------------------------

/// One argument of an atom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    /// A ground identity (already resolved to its global id).
    Ground(u64),
    /// A variable, numbered by the frontend.
    Var(u8),
    /// A nested term `f(args…)` — structural, never a relation lookup.
    Nested { functor: u64, args: Vec<Term> },
}

/// `rel(args…)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Atom {
    pub rel: u64,
    pub args: Vec<Term>,
}

/// A conjunctive query: every atom must hold; `answer` names the variable
/// reported (it is only ever existentially counted here).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pattern {
    pub atoms: Vec<Atom>,
}

/// The one shape this probe lowers: `R(a, $y) ∧ R($y, c)`, `$y ∈ W`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hop2 {
    pub rel: u64,
    pub src: u64,
    pub dst: u64,
}

/// Why a pattern was not lowered. Every refusal is a semantic boundary the
/// report names; none is approximated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Not exactly two binary atoms.
    Arity,
    /// The two atoms name different relations (a cross-relation join is
    /// lowerable in principle, but not by this probe).
    TwoRelations,
    /// A nested term: structural unification, not relational lookup (P2).
    NestedTerm,
    /// The same variable twice in one atom (`R($x, $x)`): diagonal equality.
    RepeatedVariable,
    /// An endpoint is a variable: the answer is a binding enumeration, not an
    /// existence fold.
    FreeEndpoint,
    /// The atoms do not share their middle variable in the `R(a,$y) R($y,c)`
    /// order.
    NotAChain,
}

/// Lower a pattern into the one shape the native arm runs, or refuse.
pub fn lower(p: &Pattern) -> Result<Hop2, Refusal> {
    for atom in &p.atoms {
        for t in &atom.args {
            if matches!(t, Term::Nested { .. }) {
                return Err(Refusal::NestedTerm);
            }
        }
        if let [Term::Var(a), Term::Var(b)] = atom.args.as_slice() {
            if a == b {
                return Err(Refusal::RepeatedVariable);
            }
        }
    }
    let [first, second] = p.atoms.as_slice() else {
        return Err(Refusal::Arity);
    };
    if first.args.len() != 2 || second.args.len() != 2 {
        return Err(Refusal::Arity);
    }
    if first.rel != second.rel {
        return Err(Refusal::TwoRelations);
    }
    match (&first.args[..], &second.args[..]) {
        ([Term::Ground(src), Term::Var(y1)], [Term::Var(y2), Term::Ground(dst)]) if y1 == y2 => {
            Ok(Hop2 {
                rel: first.rel,
                src: *src,
                dst: *dst,
            })
        }
        ([Term::Var(_), _], _) | (_, [_, Term::Var(_)]) => Err(Refusal::FreeEndpoint),
        _ => Err(Refusal::NotAChain),
    }
}

// ---------------------------------------------------------------------------
// Relation over one DR tile
// ---------------------------------------------------------------------------

/// Why the native arm refused to answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeRefusal {
    /// A term has no local position in the relation tile.
    Map(MapRefusal),
    /// The tile could not be read under the frame (foreign frame, no tile).
    Read(MapRead),
    /// The query names another relation than the one bound.
    WrongRelation { bound: u64, asked: u64 },
    /// The selection frame does not belong to the selection table.
    SelectionRead(MapRead),
}

/// One binary relation, laid out over the local positions of one DR tile.
pub struct Relation {
    rel: u64,
    /// `out[local(s)]`: bit `local(t)` set when `R(s, t)`.
    out: Vec<u64>,
    /// `inn[local(t)]`: bit `local(s)` set when `R(s, t)` (derived index).
    inn: Vec<u64>,
    /// Derivation count of each fact, `mult[local(s) * 64 + local(t)]`.
    /// Only the Bag fold reads it.
    mult: Vec<u32>,
}

impl Relation {
    /// Ingest ground facts under `frame` of `table`. Duplicate facts collapse
    /// in `out`/`inn` (Set) and are counted in `mult` (Bag).
    pub fn ingest(
        rel: u64,
        table: &MapTable,
        frame: &Frame,
        facts: &[(u64, u64)],
    ) -> Result<Self, NativeRefusal> {
        let map = table.map(frame, TILE).map_err(NativeRefusal::Read)?;
        let n = WIDTH.positions();
        let mut out = vec![0u64; n];
        let mut inn = vec![0u64; n];
        let mut mult = vec![0u32; n * n];
        for &(s, t) in facts {
            let ls = usize::from(map.local(s).map_err(NativeRefusal::Map)?);
            let lt = usize::from(map.local(t).map_err(NativeRefusal::Map)?);
            out[ls] |= 1 << lt;
            inn[lt] |= 1 << ls;
            mult[ls * n + lt] += 1;
        }
        Ok(Self {
            rel,
            out,
            inn,
            mult,
        })
    }

    /// The relation's own `out` row of `local` — borrowed, never copied.
    pub fn out_row(&self, local: u8) -> &[u64] {
        let i = usize::from(local);
        &self.out[i..i + 1]
    }

    /// The relation's own `inn` row of `local` — borrowed, never copied.
    pub fn inn_row(&self, local: u8) -> &[u64] {
        let i = usize::from(local);
        &self.inn[i..i + 1]
    }

    /// Bag answer: Σ over kept witnesses `y` of `mult(a,y) · mult(y,c)`.
    ///
    /// LAB FOLD, not a mask-risc terminal: mask-risc folds one lane under a
    /// mask (`MaskedSumI32`) but has no product of two lanes. The existing
    /// native route for the same Bag answer is `lance_graph_xlog::bag_count`
    /// (two `GroupReduce` degree sinks, then the TRACKED
    /// `checked_binary_i64(Mul)`, `d-dynbag-holographic-bags-v1.md` §17.5
    /// MISSING-3); EXO-108 owns closing it. This fold only checks that the Set
    /// fold cannot answer the Bag question.
    pub fn bag_derivations(&self, src: u8, dst: u8, kept: u64) -> u64 {
        let n = WIDTH.positions();
        let (a, c) = (usize::from(src), usize::from(dst));
        let mut total = 0u64;
        let mut bits = kept;
        while bits != 0 {
            let y = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            total += u64::from(self.mult[a * n + y]) * u64::from(self.mult[y * n + c]);
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Selection (virtual WHERE) and its DR routing
// ---------------------------------------------------------------------------

/// A WHERE selection held in ITS OWN tile's local positions (another frame,
/// another order).
pub struct Selection {
    pub mask: u64,
}

impl Selection {
    /// Route this selection into the relation tile: once per frame pair, by
    /// global identity. A global the relation tile does not hold is dropped
    /// (it cannot be a witness there) and reported.
    pub fn route(
        &self,
        sel_table: &MapTable,
        sel_frame: &Frame,
        rel_table: &MapTable,
        rel_frame: &Frame,
    ) -> Result<Routed, NativeRefusal> {
        let from = sel_table
            .map(sel_frame, TILE)
            .map_err(NativeRefusal::SelectionRead)?;
        let to = rel_table
            .map(rel_frame, TILE)
            .map_err(NativeRefusal::Read)?;
        let mut mask = 0u64;
        let mut dropped = 0u32;
        let mut bits = self.mask;
        while bits != 0 {
            let local = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let Some(global) = from.global(local) else {
                continue;
            };
            match to.local(global) {
                Ok(l) => mask |= 1 << l,
                Err(_) => dropped += 1,
            }
        }
        Ok(Routed { mask, dropped })
    }
}

/// A selection in the relation tile's coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Routed {
    pub mask: u64,
    /// Selected identities the relation tile does not hold.
    pub dropped: u32,
}

// ---------------------------------------------------------------------------
// The prepared native query
// ---------------------------------------------------------------------------

/// Which fold the program ends in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    /// EXISTS.
    Any,
    /// Distinct witnesses (Set).
    Count,
    /// The witness mask itself, for the Bag fold.
    Keep,
}

/// `out[a] ∧ inn[c] ∧ W` — one program, compiled once.
pub struct Prepared {
    pub hop: Hop2,
    pub src: u8,
    pub dst: u8,
    program: Program,
    scratch: Scratch<'static>,
    kept: Vec<u64>,
    /// Lab switch for the disable run: `false` drops the `inn[c]` operand,
    /// i.e. removes the join. Never `false` outside a test.
    pub join: bool,
}

impl Prepared {
    /// Resolve the query's identities under the relation frame and build the
    /// program. Cold: once per (query, frame).
    pub fn prepare(
        hop: Hop2,
        relation: &Relation,
        table: &MapTable,
        frame: &Frame,
        fold: Fold,
    ) -> Result<Self, NativeRefusal> {
        if hop.rel != relation.rel {
            return Err(NativeRefusal::WrongRelation {
                bound: relation.rel,
                asked: hop.rel,
            });
        }
        let map = table.map(frame, TILE).map_err(NativeRefusal::Read)?;
        let src = map.local(hop.src).map_err(NativeRefusal::Map)?;
        let dst = map.local(hop.dst).map_err(NativeRefusal::Map)?;
        Ok(Self {
            hop,
            src,
            dst,
            program: Self::program(fold, true),
            scratch: Scratch::new(words_for(WIDTH.positions()), 2),
            kept: vec![0; words_for(WIDTH.positions())],
            join: true,
        })
    }

    fn program(fold: Fold, join: bool) -> Program {
        // Plane 0 = out[a], plane 1 = inn[c], plane 2 = W.
        let mut ops = Vec::new();
        if join {
            ops.push(MaskOp::And {
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                dst: 0,
            });
            ops.push(MaskOp::And {
                a: Operand::Scratch(0),
                b: Operand::Plane(2),
                dst: 0,
            });
        } else {
            ops.push(MaskOp::And {
                a: Operand::Plane(0),
                b: Operand::Plane(2),
                dst: 0,
            });
        }
        let mask = Operand::Scratch(0);
        let terminal = match fold {
            Fold::Any => Terminal::Any { mask },
            Fold::Count => Terminal::Count { mask },
            Fold::Keep => Terminal::Keep { mask },
        };
        Program::new(ops, terminal)
    }

    /// Disable the join (lab falsifier only): rebuild the program without
    /// `inn[c]`.
    pub fn without_join(mut self, fold: Fold) -> Self {
        self.join = false;
        self.program = Self::program(fold, false);
        self
    }

    /// Hot: run the kept program over borrowed rows of `relation` and the
    /// routed selection. Allocates nothing.
    pub fn run(&mut self, relation: &Relation, selection: &Routed) -> Value {
        let w = [selection.mask];
        let masks: [&[u64]; 3] = [relation.out_row(self.src), relation.inn_row(self.dst), &w];
        let planes = Planes {
            n_rows: WIDTH.positions(),
            masks: &masks,
            lanes: &[],
        };
        let out = Out::Mask(&mut self.kept);
        execute_into(
            &self.program,
            &planes,
            &Foreign::NONE,
            &mut self.scratch,
            out,
        )
        .expect("probe program is valid")
    }

    /// The witness mask a [`Fold::Keep`] run wrote (relation-tile positions).
    pub fn kept(&self) -> u64 {
        self.kept[0]
    }
}

/// Build a DR tile `[globals…]` in a fresh table and seal its frame.
pub fn tile(source: u64, globals: &[u64]) -> (MapTable, Frame) {
    let mut t = MapTable::new(source, 0);
    let map = LocalMap::prepare(WIDTH, globals).expect("probe tile");
    t.advance(source, 0, vec![(TILE, Some(map))]);
    let f = t.seal();
    (t, f)
}
