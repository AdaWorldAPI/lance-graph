//! D-GRAMMAR-WHEN-1 — binding a grammatical WHEN through DT64 (LAB ONLY).
//!
//! Two class-bound read-only methods, `CandidateRecipient` and
//! `CandidateExperiencer`, share `deprel = obl:arg ∧ Case = Dat` and differ in
//! the head verb's frame. This module tests how DT64 can carry that
//! difference, and what runs it, in two physical forms:
//!
//! - [`Physical::Via`]: the head's lemma is read through the token's HEAD
//!   pointer (`Filter::EqU32Via` → `Pred::EqU32Via`), as in #1947. The guard
//!   names the frame only as an opaque `SetExpr::Sym(digest)`. The read is
//!   at a position taken from another lane's value: ADMISSION B
//!   (`entries/2026-10-20-six-decisions.md` §1) says such a position "is not
//!   admissible as a read address", and `EqU32Via` "need[s] call-site
//!   classification" (EXO-140, OPEN). LAB.
//! - [`Physical::CoAddressed`]: "the join is the layout" (D-HOLO H14c/H14d).
//!   The head's frame code is a lane at the DEPENDENT's own coordinate
//!   (`head_frame[i]`), written once by the producer ([`FramedPopulation`]).
//!   The guard is then row-local and fully stated in today's `SetExpr`; the
//!   query reads every lane at address `i`. What stays OPEN is the producer
//!   step itself: it derives `head_frame[i]` from `lemma[head[i]]`, which is
//!   a cross-row read at ingestion (EXO-46 / D-RMAP-1 membrane), counted
//!   here.
//!
//! Neither form is an admission. The composer ([`compose`]) is the
//! approach-C model: DT64 admits the reference, the composer checks the
//! description against the bound guard witness, and lowers the description
//! plus the global WHERE plane into ONE mask-risc program through
//! `lance_graph_quack::lower`. Nothing is written; the result is a candidate
//! mask.

use std::cell::Cell;

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population as Multiplicity, Truth};
use lance_graph_dt64::dr::{Frame, MapTable};
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, Guard, MethodBinding, MethodId, MethodRef, Refusal, SchemaRev,
    SigRev, ValueDemand,
};
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, LaneRef, Out, Planes, Program, Scratch,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, ForeignLane, LowerError, Mask, Query};

use crate::{Population, Refusal as PopRefusal, Tok, Vocab, CASE, DEPREL, HEAD, LEMMA, TILE, WIDTH};

/// Field position (and lane) of the head's frame code, co-addressed.
pub const HEAD_FRAME: u16 = 4;
/// Fields of the lab token class: lemma, deprel, case, head, head_frame.
pub const FIELDS: usize = 5;

/// Lab stand-in classids. Unminted; they say nothing about OGAR allocation.
pub const CLASS_VIA: u32 = 0x7F21_0000;
pub const CLASS_COADDR: u32 = 0x7F21_0001;
pub const CLASS_FOREIGN: u32 = 0x7F22_0000;

pub const RECIPIENT: MethodId = MethodId(1);
pub const EXPERIENCER: MethodId = MethodId(2);

/// Frame codes of the lab lexicon. 0 = no entry (never a negative role).
pub const FRAME_NONE: u32 = 0;
pub const FRAME_GIVE: u32 = 1;
pub const FRAME_EXPERIENCE: u32 = 2;

/// A versioned, language-scoped verb-frame lexicon (lab data, German).
#[derive(Clone, Debug)]
pub struct Lexicon {
    pub rev: u64,
    pub language: &'static str,
    pub entries: Vec<(&'static str, u32)>,
}

impl Lexicon {
    /// `de-dative-frames-fixture-v0` from #1947.
    pub fn v0() -> Self {
        Lexicon {
            rev: 1,
            language: "de",
            entries: vec![
                ("geben", FRAME_GIVE),
                ("schenken", FRAME_GIVE),
                ("gefallen", FRAME_EXPERIENCE),
            ],
        }
    }

    /// The same lemmas with GIVE and EXPERIENCE exchanged (control 2).
    pub fn swapped() -> Self {
        let mut l = Self::v0();
        l.rev = 2;
        for e in &mut l.entries {
            e.1 = match e.1 {
                FRAME_GIVE => FRAME_EXPERIENCE,
                FRAME_EXPERIENCE => FRAME_GIVE,
                x => x,
            };
        }
        l
    }

    pub fn frame(&self, lemma: &str) -> u32 {
        self.entries
            .iter()
            .find(|(l, _)| *l == lemma)
            .map_or(FRAME_NONE, |e| e.1)
    }

    /// Lemmas carrying `frame`, sorted.
    pub fn lemmas(&self, frame: u32) -> Vec<&'static str> {
        let mut v: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.1 == frame)
            .map(|e| e.0)
            .collect();
        v.sort_unstable();
        v
    }
}

/// The base population plus the co-addressed `head_frame` lane, produced once
/// under one lexicon revision.
pub struct FramedPopulation {
    pub base: Population,
    pub head_frame: Vec<u32>,
    pub lexicon_rev: u64,
    /// Cross-row reads the producer performed (`lemma[head[i]]`): the
    /// EXO-46 / D-RMAP-1 membrane cost, counted, never in the query.
    pub producer_lookups: usize,
}

impl FramedPopulation {
    /// Produce `head_frame[i] = frame(lemma(head(i)))`. Cold.
    pub fn produce(
        toks: &[Tok],
        vocab: &mut Vocab,
        table: &MapTable,
        frame: &Frame,
        lex: &Lexicon,
    ) -> Result<Self, PopRefusal> {
        let base = Population::bind(toks, vocab, table, frame)?;
        let map = table.map(frame, TILE).map_err(PopRefusal::Read)?;
        let lookups = Cell::new(0usize);
        let mut head_frame = vec![FRAME_NONE; base.n];
        for t in toks {
            if t.head == 0 {
                continue;
            }
            let i = usize::from(map.local(t.gid).map_err(PopRefusal::Map)?);
            lookups.set(lookups.get() + 1);
            let head = toks.iter().find(|h| h.gid == t.head);
            head_frame[i] = head.map_or(FRAME_NONE, |h| lex.frame(h.lemma));
        }
        Ok(FramedPopulation {
            base,
            head_frame,
            lexicon_rev: lex.rev,
            producer_lookups: lookups.get(),
        })
    }
}

/// Which physical form the description lowers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Physical {
    Via,
    CoAddressed,
}

/// A candidate-role rule as a relational description (data, not code).
#[derive(Clone, Debug)]
pub struct RuleDesc {
    pub method: MethodId,
    pub name: &'static str,
    pub deprel: Option<&'static str>,
    pub case: Option<&'static str>,
    /// `None` = the head-frame condition removed (control 1).
    pub frame: Option<u32>,
}

/// Resolved ids a description needs (cold).
#[derive(Clone, Copy, Debug)]
struct Bound {
    deprel: u32,
    case: u32,
}

impl RuleDesc {
    pub fn recipient() -> Self {
        RuleDesc {
            method: RECIPIENT,
            name: "CandidateRecipient",
            deprel: Some("obl:arg"),
            case: Some("Dat"),
            frame: Some(FRAME_GIVE),
        }
    }

    pub fn experiencer() -> Self {
        RuleDesc {
            method: EXPERIENCER,
            name: "CandidateExperiencer",
            deprel: Some("obl:arg"),
            case: Some("Dat"),
            frame: Some(FRAME_EXPERIENCE),
        }
    }

    fn bound(&self, v: &Vocab) -> Bound {
        Bound {
            deprel: self.deprel.map_or(0, |d| v.get(d)),
            case: self.case.map_or(0, |c| v.get(c)),
        }
    }

    /// FNV-1a over the frame's lemma set and the lexicon revision: the only
    /// identity a `Sym` leaf can carry for the Via form.
    pub fn frame_digest(&self, lex: &Lexicon) -> u32 {
        let mut h: u32 = 0x811c_9dc5;
        let mut eat = |b: &[u8]| {
            for &x in b {
                h ^= u32::from(x);
                h = h.wrapping_mul(0x0100_0193);
            }
        };
        eat(&lex.rev.to_le_bytes());
        eat(lex.language.as_bytes());
        if let Some(f) = self.frame {
            for l in lex.lemmas(f) {
                eat(l.as_bytes());
                eat(&[0]);
            }
        }
        h
    }

    /// The WHEN this description binds as.
    pub fn guard(&self, phys: Physical, v: &Vocab, lex: &Lexicon) -> Guard {
        let b = self.bound(v);
        let mut parts = Vec::new();
        if self.deprel.is_some() {
            parts.push(SetExpr::eq_u32(DEPREL, b.deprel));
        }
        if self.case.is_some() {
            parts.push(SetExpr::eq_u32(CASE, b.case));
        }
        if let Some(f) = self.frame {
            parts.push(match phys {
                // Only an opaque symbol can stand for a hop.
                Physical::Via => SetExpr::Sym(self.frame_digest(lex)),
                // Row-local: stated in full.
                Physical::CoAddressed => SetExpr::eq_u32(HEAD_FRAME, f),
            });
        }
        Guard::new(SetExpr::And(parts), Logic::TwoValued)
    }

    /// The relational description in Quack's vocabulary, scoped by the
    /// global WHERE plane (`Mask(0)`).
    pub fn filter(&self, phys: Physical, v: &Vocab, lex: &Lexicon) -> Filter {
        let b = self.bound(v);
        let mut parts = vec![Filter::plane(Mask(0))];
        if self.deprel.is_some() {
            parts.push(Filter::cmp(Col(DEPREL), Cmp::EqU32(b.deprel)));
        }
        if self.case.is_some() {
            parts.push(Filter::cmp(Col(CASE), Cmp::EqU32(b.case)));
        }
        if let Some(f) = self.frame {
            parts.push(match phys {
                Physical::Via => Filter::or(
                    lex.lemmas(f)
                        .into_iter()
                        .map(|l| Filter::eq_u32_via(Col(HEAD), ForeignLane(0), v.get(l))),
                ),
                Physical::CoAddressed => Filter::cmp(Col(HEAD_FRAME), Cmp::EqU32(f)),
            });
        }
        Filter::and(parts)
    }

    /// The DT64 binding: read-only (no effects), reading every field.
    pub fn binding(&self, phys: Physical, v: &Vocab, lex: &Lexicon) -> MethodBinding {
        MethodBinding {
            id: self.method,
            sig_rev: SigRev(1),
            name: self.name.to_string(),
            on: WideFieldMask::from_positions(&[1, 2, 3]),
            when: Some(self.guard(phys, v, lex)),
            reads: WideFieldMask::from_positions(&[0, 1, 2, 3, 4]),
            effects: Vec::new(),
            population: Multiplicity::Distinct,
            truth: Truth::TwoValued,
            demand: ValueDemand::Values,
        }
    }
}

pub fn schema() -> ClassSchema {
    ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: FIELDS,
        extent: Extent::DR256_SQ_X64,
    }
}

/// Register both candidate methods for `class`.
pub fn register(
    store: &mut BindStore,
    class: u32,
    phys: Physical,
    rules: &[RuleDesc],
    v: &Vocab,
    lex: &Lexicon,
) -> Result<(), Refusal> {
    store.register(
        class,
        schema(),
        rules.iter().map(|r| r.binding(phys, v, lex)).collect(),
    )
}

/// Why the composer refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposeRefusal {
    Binding(Refusal),
    /// The method writes: not a read-only candidate.
    NotReadOnly,
    NoGuard,
    /// The description given does not produce the bound guard's witness.
    DescriptionMismatch,
    Lower(LowerError),
}

/// Why a composed run refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunRefusal {
    Frame(PopRefusal),
    /// The population was produced under another lexicon revision.
    LexiconRevision { bound: u64, population: u64 },
    /// The WHERE plane is not over this population.
    WhereShape,
}

/// One admitted method, its description inlined with the WHERE plane into
/// one program. Cold to build; hot to run.
pub struct Composed {
    pub method: MethodRef,
    pub phys: Physical,
    pub lexicon_rev: u64,
    program: Program,
    scratch: Scratch<'static>,
    out: Vec<u64>,
}

/// Approach C (lab): admit, check the description against the bound witness,
/// lower once.
pub fn compose(
    store: &BindStore,
    mref: &MethodRef,
    desc: &RuleDesc,
    phys: Physical,
    v: &Vocab,
    lex: &Lexicon,
) -> Result<Composed, ComposeRefusal> {
    let b = store.admit(mref).map_err(ComposeRefusal::Binding)?;
    if !b.effects.is_empty() {
        return Err(ComposeRefusal::NotReadOnly);
    }
    let bound = b.when.as_ref().ok_or(ComposeRefusal::NoGuard)?;
    if desc.guard(phys, v, lex).witness() != bound.witness() {
        return Err(ComposeRefusal::DescriptionMismatch);
    }
    let program = lower(&Query {
        filter: desc.filter(phys, v, lex),
        agg: Agg::Rows,
    })
    .map_err(ComposeRefusal::Lower)?;
    let words = words_for(WIDTH.positions());
    Ok(Composed {
        method: mref.clone(),
        phys,
        lexicon_rev: lex.rev,
        scratch: Scratch::new(words, program.scratch_slots as usize),
        program,
        out: vec![0; words],
    })
}

impl Composed {
    /// Program size (mask ops).
    pub fn ops(&self) -> usize {
        self.program.ops.len()
    }

    /// Hot: one program over borrowed lanes and the shared WHERE plane.
    pub fn run(
        &mut self,
        pop: &FramedPopulation,
        table: &MapTable,
        frame: &Frame,
        where_plane: &[u64],
    ) -> Result<[u64; 4], RunRefusal> {
        let map = table
            .map(frame, TILE)
            .map_err(|e| RunRefusal::Frame(PopRefusal::Read(e)))?;
        if map.len() != pop.base.n {
            return Err(RunRefusal::Frame(PopRefusal::Map(
                lance_graph_dt64::dr::MapRefusal::OutOfTile,
            )));
        }
        if pop.lexicon_rev != self.lexicon_rev {
            return Err(RunRefusal::LexiconRevision {
                bound: self.lexicon_rev,
                population: pop.lexicon_rev,
            });
        }
        let words = words_for(pop.base.n);
        if where_plane.len() != words {
            return Err(RunRefusal::WhereShape);
        }
        let p = &pop.base;
        let lanes = [
            LaneRef::U32(&p.lemma),
            LaneRef::U32(&p.deprel),
            LaneRef::U32(&p.case),
            LaneRef::U32(&p.head),
            LaneRef::U32(&pop.head_frame),
        ];
        let flanes = [LaneRef::U32(&p.lemma)];
        let masks = [where_plane];
        let planes = Planes {
            n_rows: p.n,
            masks: &masks,
            lanes: &lanes,
        };
        let foreign = Foreign {
            planes: &[],
            lanes: &flanes,
        };
        execute_into(
            &self.program,
            &planes,
            &foreign,
            &mut self.scratch,
            Out::Mask(&mut self.out[..words]),
        )
        .expect("composed program is valid");
        let mut r = [0u64; 4];
        r[..words].copy_from_slice(&self.out[..words]);
        Ok(r)
    }
}

// Lane-index sanity: the lab class fields are the lanes, in order.
const _: () = assert!(LEMMA == 0 && DEPREL == 1 && CASE == 2 && HEAD == 3 && HEAD_FRAME == 4);
