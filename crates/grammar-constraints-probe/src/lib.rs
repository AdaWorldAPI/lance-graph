//! D-GRAMMAR-CONSTRAINTS-0 — grammatical constraints as typed relations over
//! UD-annotated tokens (LAB ONLY).
//!
//! STATUS: PROBE. No parser, no grammar compiler, no production path. The
//! token annotations are hand-written in UD German-GSD convention (see
//! `fixtures`), and the frame lexicon is a declared, versioned fixture.
//!
//! # What is tested
//!
//! Whether a German dative argument's candidate semantic function (one of
//! Haspelmath's dative-map functions, the same eight `causal_edge::Orientation`
//! names) can be selected by ONE mask-risc program per rule, over resident
//! lanes of the token population, with the head hop read in place through
//! `Pred::EqU32Via` (fk = the token's head pointer). In UD German-GSD the
//! recipient of *geben* and the experiencer of *gefallen* carry the SAME
//! `deprel = obl:arg` and `Case=Dat`; only the head verb's frame separates
//! them, so a rule that reads only the dependent's own lanes must fail.
//!
//! # Constraint classes (kept apart)
//!
//! - STRUCTURAL: `deprel`, `head`, `Case` of the token (UD layer, exact).
//! - LANGUAGE-SPECIFIC: the frame lexicon [`FRAMES_VERSION`] (German verbs →
//!   role of their `obl:arg` dative), and `zu` + Dat `obl` → {Direction,
//!   Purpose}.
//! - HEURISTIC: none is applied: Direction vs Purpose is NOT chosen; both are
//!   kept (no selection policy exists).
//! - EVIDENTIAL / EFFECTFUL: none. Every output is a CANDIDATE mask; nothing is
//!   written back, nothing is promoted to a fact.
//!
//! # UNKNOWN
//!
//! A dative `obl:arg` whose head has no frame entry is reported as
//! [`Readout::abstain_frame`]; an `obl:arg` with a frame but no `Case`
//! feature is [`Readout::unknown_case`]. Neither is FALSE.
//!
//! # EqU32Via admission
//!
//! `Pred::EqU32Via` executes in mask-risc today, but its Quack call site is
//! EXO-140-unclassified. This probe uses it as a lab measurement of what the
//! algebra can express, not as an admission.

#![forbid(unsafe_code)]

pub mod fixtures;
pub mod when;

use std::collections::HashMap;

use lance_graph_dt64::dr::{Frame, LocalMap, MapRead, MapRefusal, MapTable, TileId, Width};
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program,
    Scratch, Terminal,
};

pub const TILE: TileId = 0;
pub const WIDTH: Width = Width::W256;
/// Head pointer of a root token: names no row, so `EqU32Via` does not match
/// and `ScatterOrU32` drops it.
pub const ROOT: u32 = u32::MAX;

/// Version of the declared language-specific frame lexicon.
pub const FRAMES_VERSION: &str = "de-dative-frames-fixture-v0";

// ---------------------------------------------------------------------------
// Cold vocabulary binding: labels → u32 ids (0 = absent / UNKNOWN)
// ---------------------------------------------------------------------------

/// Label → id, built once (cold). Id 0 is reserved for "no value".
pub struct Vocab {
    ids: HashMap<String, u32>,
}

impl Vocab {
    pub fn new() -> Self {
        Self {
            ids: HashMap::new(),
        }
    }

    /// Intern a label (cold). The empty label is 0.
    pub fn id(&mut self, label: &str) -> u32 {
        if label.is_empty() {
            return 0;
        }
        let next = self.ids.len() as u32 + 1;
        *self.ids.entry(label.to_string()).or_insert(next)
    }

    /// Look up without interning; 0 if unknown.
    pub fn get(&self, label: &str) -> u32 {
        self.ids.get(label).copied().unwrap_or(0)
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Token population
// ---------------------------------------------------------------------------

/// One UD syntactic word, as annotated (string form; bound cold).
#[derive(Clone, Copy, Debug)]
pub struct Tok {
    /// Global identity.
    pub gid: u64,
    pub form: &'static str,
    pub lemma: &'static str,
    pub deprel: &'static str,
    /// `Case=` value, "" when the feature is absent.
    pub case: &'static str,
    /// Global id of the head, 0 for the root.
    pub head: u64,
}

/// Lane indices of a [`Population`].
pub const LEMMA: u16 = 0;
pub const DEPREL: u16 = 1;
pub const CASE: u16 = 2;
pub const HEAD: u16 = 3;
/// Foreign lane indices: the same population read through `head`.
pub const F_LEMMA: u16 = 0;
pub const F_DEPREL: u16 = 1;
pub const F_CASE: u16 = 2;

/// A token population over one DR tile: four `u32` lanes in local order.
pub struct Population {
    pub n: usize,
    lemma: Vec<u32>,
    deprel: Vec<u32>,
    case: Vec<u32>,
    head: Vec<u32>,
    /// The frame the lanes were bound under; every run must present it.
    frame: Frame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Map(MapRefusal),
    Read(MapRead),
    /// The frame is not the one the population was bound under (the table
    /// may have advanced to another layout of the same size).
    PopulationFrame,
}

impl Population {
    /// Bind `toks` (cold) under `frame`: labels through `vocab`, heads through
    /// the tile's global → local map.
    pub fn bind(
        toks: &[Tok],
        vocab: &mut Vocab,
        table: &MapTable,
        frame: &Frame,
    ) -> Result<Self, Refusal> {
        let map = table.map(frame, TILE).map_err(Refusal::Read)?;
        let n = map.len();
        let mut p = Population {
            n,
            lemma: vec![0; n],
            deprel: vec![0; n],
            case: vec![0; n],
            head: vec![ROOT; n],
            frame: frame.clone(),
        };
        for t in toks {
            let l = usize::from(map.local(t.gid).map_err(Refusal::Map)?);
            p.lemma[l] = vocab.id(t.lemma);
            p.deprel[l] = vocab.id(t.deprel);
            p.case[l] = vocab.id(t.case);
            if t.head != 0 {
                p.head[l] = u32::from(map.local(t.head).map_err(Refusal::Map)?);
            }
        }
        Ok(p)
    }

    fn lanes(&self) -> [LaneRef<'_>; 4] {
        [
            LaneRef::U32(&self.lemma),
            LaneRef::U32(&self.deprel),
            LaneRef::U32(&self.case),
            LaneRef::U32(&self.head),
        ]
    }

    fn foreign_lanes(&self) -> [LaneRef<'_>; 3] {
        [
            LaneRef::U32(&self.lemma),
            LaneRef::U32(&self.deprel),
            LaneRef::U32(&self.case),
        ]
    }
}

/// Build a DR tile of `gids` in a fresh table and seal its frame.
pub fn tile(source: u64, gids: &[u64]) -> (MapTable, Frame) {
    let mut t = MapTable::new(source, 0);
    let map = LocalMap::prepare(WIDTH, gids).expect("probe tile");
    t.advance(source, 0, vec![(TILE, Some(map))]);
    let f = t.seal();
    (t, f)
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

/// The ids a rule set needs, resolved once from the vocabulary.
#[derive(Clone, Copy, Debug)]
pub struct Ids {
    pub obl_arg: u32,
    pub obl: u32,
    pub case_rel: u32,
    pub dat: u32,
    pub zu: u32,
    /// GIVE frame: `obl:arg` dative = Recipient.
    pub give: [u32; 2],
    /// EXPERIENCE frame: `obl:arg` dative = Experiencer.
    pub experience: [u32; 1],
}

impl Ids {
    /// The declared frame lexicon ([`FRAMES_VERSION`]).
    pub fn bind(v: &mut Vocab) -> Self {
        Ids {
            obl_arg: v.id("obl:arg"),
            obl: v.id("obl"),
            case_rel: v.id("case"),
            dat: v.id("Dat"),
            zu: v.id("zu"),
            give: [v.id("geben"), v.id("schenken")],
            experience: [v.id("gefallen")],
        }
    }
}

/// Lab switches for the disable runs. All `true` outside a falsifier.
#[derive(Clone, Copy, Debug)]
pub struct Operands {
    pub deprel: bool,
    pub case: bool,
    pub frame: bool,
}

impl Operands {
    pub const ALL: Operands = Operands {
        deprel: true,
        case: true,
        frame: true,
    };
}

fn pred(p: Pred, dst: u16) -> MaskOp {
    MaskOp::Pred {
        pred: p,
        under: None,
        dst,
    }
}

/// `[deprel == obl:arg] ∧ [case == Dat] ∧ head.lemma ∈ verbs`, Keep.
/// A disabled operand is replaced by `deprel != 0` (any token), so the
/// program keeps its shape.
fn dative_arg(ids: &Ids, verbs: &[u32], ops: Operands) -> Program {
    let mut v = Vec::new();
    v.push(if ops.deprel {
        pred(
            Pred::EqU32 {
                lane: DEPREL,
                v: ids.obl_arg,
            },
            0,
        )
    } else {
        pred(Pred::NeU32 { lane: DEPREL, v: 0 }, 0)
    });
    v.push(if ops.case {
        pred(
            Pred::EqU32 {
                lane: CASE,
                v: ids.dat,
            },
            1,
        )
    } else {
        pred(Pred::NeU32 { lane: DEPREL, v: 0 }, 1)
    });
    v.push(MaskOp::And {
        a: Operand::Scratch(0),
        b: Operand::Scratch(1),
        dst: 0,
    });
    if ops.frame {
        for (i, &verb) in verbs.iter().enumerate() {
            let dst = if i == 0 { 1 } else { 2 };
            v.push(pred(
                Pred::EqU32Via {
                    fk: HEAD,
                    key: F_LEMMA,
                    v: verb,
                },
                dst,
            ));
            if i > 0 {
                v.push(MaskOp::Or {
                    a: Operand::Scratch(1),
                    b: Operand::Scratch(2),
                    dst: 1,
                });
            }
        }
        v.push(MaskOp::And {
            a: Operand::Scratch(0),
            b: Operand::Scratch(1),
            dst: 0,
        });
    }
    Program::new(
        v,
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    )
}

/// Evaluated at the case marker `t`: `t.deprel == case ∧ t.lemma == zu ∧
/// head(t).deprel == obl ∧ head(t).case == Dat`; the demanded result is the
/// set of heads (`ScatterOrU32` over `head`): nouns with candidates
/// {Direction, Purpose}.
fn zu_goal(ids: &Ids) -> Program {
    let v = vec![
        pred(
            Pred::EqU32 {
                lane: DEPREL,
                v: ids.case_rel,
            },
            0,
        ),
        pred(
            Pred::EqU32 {
                lane: LEMMA,
                v: ids.zu,
            },
            1,
        ),
        MaskOp::And {
            a: Operand::Scratch(0),
            b: Operand::Scratch(1),
            dst: 0,
        },
        pred(
            Pred::EqU32Via {
                fk: HEAD,
                key: F_DEPREL,
                v: ids.obl,
            },
            1,
        ),
        MaskOp::And {
            a: Operand::Scratch(0),
            b: Operand::Scratch(1),
            dst: 0,
        },
        pred(
            Pred::EqU32Via {
                fk: HEAD,
                key: F_CASE,
                v: ids.dat,
            },
            1,
        ),
        MaskOp::And {
            a: Operand::Scratch(0),
            b: Operand::Scratch(1),
            dst: 0,
        },
    ];
    Program::new(
        v,
        Terminal::ScatterOrU32 {
            mask: Operand::Scratch(0),
            lane: HEAD,
            out_rows: WIDTH.positions() as u32,
        },
    )
}

/// `[deprel == obl:arg] ∧ ¬[head.lemma ∈ every framed verb]`, with the case
/// split: `case == Dat` → frame abstention; `case == 0` (absent) ∧ framed →
/// unknown case. Two programs.
fn abstain_frame(ids: &Ids) -> Program {
    let mut v = vec![
        pred(
            Pred::EqU32 {
                lane: DEPREL,
                v: ids.obl_arg,
            },
            0,
        ),
        pred(
            Pred::EqU32 {
                lane: CASE,
                v: ids.dat,
            },
            1,
        ),
        MaskOp::And {
            a: Operand::Scratch(0),
            b: Operand::Scratch(1),
            dst: 0,
        },
    ];
    push_any_frame(&mut v, ids, 1);
    v.push(MaskOp::AndNot {
        a: Operand::Scratch(0),
        b: Operand::Scratch(1),
        dst: 0,
    });
    Program::new(
        v,
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    )
}

fn unknown_case(ids: &Ids) -> Program {
    let mut v = vec![
        pred(
            Pred::EqU32 {
                lane: DEPREL,
                v: ids.obl_arg,
            },
            0,
        ),
        pred(Pred::EqU32 { lane: CASE, v: 0 }, 1),
        MaskOp::And {
            a: Operand::Scratch(0),
            b: Operand::Scratch(1),
            dst: 0,
        },
    ];
    push_any_frame(&mut v, ids, 1);
    v.push(MaskOp::And {
        a: Operand::Scratch(0),
        b: Operand::Scratch(1),
        dst: 0,
    });
    Program::new(
        v,
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    )
}

/// `dst = head.lemma ∈ give ∪ experience`, using `dst` and `dst + 1`.
fn push_any_frame(v: &mut Vec<MaskOp>, ids: &Ids, dst: u16) {
    let verbs: Vec<u32> = ids
        .give
        .iter()
        .chain(ids.experience.iter())
        .copied()
        .collect();
    for (i, &verb) in verbs.iter().enumerate() {
        let d = if i == 0 { dst } else { dst + 1 };
        v.push(pred(
            Pred::EqU32Via {
                fk: HEAD,
                key: F_LEMMA,
                v: verb,
            },
            d,
        ));
        if i > 0 {
            v.push(MaskOp::Or {
                a: Operand::Scratch(dst),
                b: Operand::Scratch(dst + 1),
                dst,
            });
        }
    }
}

/// The five prepared programs, compiled once.
pub struct Rules {
    recipient: Program,
    experiencer: Program,
    goal: Program,
    abstain: Program,
    unknown: Program,
    scratch: Scratch<'static>,
}

/// Candidate masks over the population's local positions (one bit per
/// token). A token in both `direction_or_purpose` and nothing else carries
/// two candidates: the ambiguity is kept.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Readout {
    pub recipient: [u64; 4],
    pub experiencer: [u64; 4],
    pub direction_or_purpose: [u64; 4],
    pub abstain_frame: [u64; 4],
    pub unknown_case: [u64; 4],
}

impl Rules {
    pub fn prepare(ids: &Ids, ops: Operands) -> Self {
        Rules {
            recipient: dative_arg(ids, &ids.give, ops),
            experiencer: dative_arg(ids, &ids.experience, ops),
            goal: zu_goal(ids),
            abstain: abstain_frame(ids),
            unknown: unknown_case(ids),
            scratch: Scratch::new(words_for(WIDTH.positions()), 3),
        }
    }

    /// The number of mask ops across the five programs.
    pub fn op_count(&self) -> usize {
        [
            &self.recipient,
            &self.experiencer,
            &self.goal,
            &self.abstain,
            &self.unknown,
        ]
        .iter()
        .map(|p| p.ops.len())
        .sum()
    }

    /// Hot: run the five programs over borrowed lanes. Re-checks the frame.
    /// Allocates nothing.
    pub fn run(
        &mut self,
        pop: &Population,
        table: &MapTable,
        frame: &Frame,
    ) -> Result<Readout, Refusal> {
        let map = table.map(frame, TILE).map_err(Refusal::Read)?;
        if map.len() != pop.n {
            return Err(Refusal::Map(MapRefusal::OutOfTile));
        }
        if !frame.same_issue(&pop.frame) {
            return Err(Refusal::PopulationFrame);
        }
        let lanes = pop.lanes();
        let flanes = pop.foreign_lanes();
        let planes = Planes {
            n_rows: pop.n,
            masks: &[],
            lanes: &lanes,
        };
        let foreign = Foreign {
            planes: &[],
            lanes: &flanes,
        };
        let mut r = Readout::default();
        let words = words_for(pop.n);
        for (prog, out) in [
            (&self.recipient, &mut r.recipient),
            (&self.experiencer, &mut r.experiencer),
            (&self.goal, &mut r.direction_or_purpose),
            (&self.abstain, &mut r.abstain_frame),
            (&self.unknown, &mut r.unknown_case),
        ] {
            let full = matches!(prog.terminal, Terminal::ScatterOrU32 { .. });
            let dst: &mut [u64] = if full {
                &mut out[..]
            } else {
                &mut out[..words]
            };
            execute_into(prog, &planes, &foreign, &mut self.scratch, Out::Mask(dst))
                .expect("probe program is valid");
        }
        Ok(r)
    }
}

/// Bit test on a 256-bit mask.
pub fn has(mask: &[u64; 4], local: u8) -> bool {
    mask[usize::from(local) / 64] >> (local % 64) & 1 == 1
}
