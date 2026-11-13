//! D-TWINS-0 — one guarded-candidate rule shape in two domains (LAB ONLY).
//!
//! STATUS: PROBE. No production path, no new evaluator, no minted identity.
//!
//! # What is tested
//!
//! Whether ONE rule shape serves two unrelated domains without a
//! domain-specific guard type:
//!
//! - a [`Family`] of at most eight candidate [`Rule`]s;
//! - a candidate set per row as ONE BYTE, bit `i` = rule `i` is proposed
//!   (the selector; it proposes, it decides nothing);
//! - a WHEN per rule: a conjunction of `lane ∈ {values}` tests over the
//!   row's OWN lanes, stored as a DT64 [`Guard`] and lowered through Quack
//!   to one mask-risc program;
//! - a producer identity ([`Family::producer`]) carried in every guard as
//!   `SetExpr::Sym`, so a method bound against one producer of the lanes
//!   does not compose against another.
//!
//! The crate has no chess and no Rust-syntax knowledge. The two arms live in
//! `tests/`, each with its own authority.
//!
//! # What a rule here cannot say
//!
//! A guard reads the lanes of its own row, before anything happens. It
//! cannot read another row, an earlier time, or the state a move would
//! produce. Whatever of those a rule needs must already be a lane, written
//! by a producer. The tests measure where that is enough and where it is
//! not.

#![forbid(unsafe_code)]

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population as Multiplicity, Truth};
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, Guard, MethodBinding, MethodId, Refusal, SchemaRev, SigRev,
    ValueDemand,
};
use lance_graph_mask_risc::{execute_into, words_for, Foreign, LaneRef, Out, Planes, Scratch};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, LowerError, Query};

/// Most rules a family holds: one bit each in a candidate byte.
pub const MAX_RULES: usize = 8;

/// One candidate rule: `AND over (lane ∈ values)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    pub name: &'static str,
    /// `(lane, admitted values)`. An empty list is an unguarded rule.
    pub when: Vec<(u16, Vec<u32>)>,
}

/// A rule family over one row class.
#[derive(Clone, Debug)]
pub struct Family {
    /// Lab stand-in classid. Unminted.
    pub class: u32,
    /// Number of lanes (fields) of the row class.
    pub lanes: usize,
    pub rules: Vec<Rule>,
    /// Identity of whatever produced the lanes (a digest the arm chooses).
    pub producer: u32,
}

/// Why a family could not be bound or run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TwinRefusal {
    TooManyRules(usize),
    Register(Refusal),
    Admit(Refusal),
    NoGuard,
    /// The description does not produce the bound guard's witness: another
    /// rule, a dropped condition, or another producer.
    DescriptionMismatch,
    Lower(LowerError),
    /// The lanes or the candidate bytes are not over one row universe.
    Shape,
}

impl Family {
    fn method(i: usize) -> MethodId {
        MethodId(i as u32 + 1)
    }

    /// The WHEN rule `i` binds as, under `producer`.
    pub fn guard(&self, i: usize, producer: u32) -> Guard {
        let mut parts: Vec<SetExpr> = self.rules[i]
            .when
            .iter()
            .map(|(lane, values)| {
                SetExpr::Or(values.iter().map(|&v| SetExpr::eq_u32(*lane, v)).collect())
            })
            .collect();
        parts.push(SetExpr::Sym(producer));
        Guard::new(SetExpr::And(parts), Logic::TwoValued)
    }

    fn filter(&self, i: usize) -> Filter {
        Filter::and(self.rules[i].when.iter().map(|(lane, values)| {
            Filter::or(
                values
                    .iter()
                    .map(|&v| Filter::cmp(Col(*lane), Cmp::EqU32(v))),
            )
        }))
    }

    fn binding(&self, i: usize) -> MethodBinding {
        let all: Vec<u8> = (0..self.lanes as u8).collect();
        MethodBinding {
            id: Self::method(i),
            sig_rev: SigRev(1),
            name: self.rules[i].name.to_string(),
            on: WideFieldMask::from_positions(&all),
            when: Some(self.guard(i, self.producer)),
            reads: WideFieldMask::from_positions(&all),
            effects: Vec::new(),
            population: Multiplicity::Distinct,
            truth: Truth::TwoValued,
            demand: ValueDemand::Values,
        }
    }

    /// Register every rule as a guarded, read-only DT64 method.
    pub fn bind(&self) -> Result<Bound, TwinRefusal> {
        if self.rules.len() > MAX_RULES {
            return Err(TwinRefusal::TooManyRules(self.rules.len()));
        }
        let mut store = BindStore::new();
        store
            .register(
                &LabView::new(self.class, self.lanes),
                self.class,
                ClassSchema {
                    schema_rev: SchemaRev(1),
                    field_count: self.lanes,
                    extent: Extent::DR256_SQ_X64,
                },
                (0..self.rules.len()).map(|i| self.binding(i)).collect(),
            )
            .map_err(TwinRefusal::Register)?;
        Ok(Bound {
            store,
            family: self.clone(),
        })
    }
}

/// A family whose rules are registered methods of one store.
pub struct Bound {
    store: BindStore,
    family: Family,
}

/// What one run did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answers {
    /// Per row: bit `i` set when rule `i` was proposed for the row AND its
    /// guard holds there.
    pub holds: Vec<u8>,
    /// Rules composed (proposed for at least one row).
    pub composed: usize,
}

impl Bound {
    /// Decide the candidates. `described` is the caller's description of the
    /// family (it must reproduce every bound guard's witness, producer
    /// included); `candidates[r]` is the selector's byte for row `r`. A rule
    /// is admitted, checked and lowered only if some row proposes it.
    pub fn decide(
        &self,
        described: &Family,
        lanes: &[Vec<u32>],
        candidates: &[u8],
    ) -> Result<Answers, TwinRefusal> {
        let n = candidates.len();
        if lanes.len() != self.family.lanes || lanes.iter().any(|l| l.len() != n) {
            return Err(TwinRefusal::Shape);
        }
        if described.rules.len() != self.family.rules.len() {
            return Err(TwinRefusal::DescriptionMismatch);
        }
        let refs: Vec<LaneRef<'_>> = lanes.iter().map(|l| LaneRef::U32(l)).collect();
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &refs,
        };
        let words = words_for(n);
        let mut holds = vec![0u8; n];
        let mut composed = 0;
        for i in 0..self.family.rules.len() {
            let bit = 1u8 << i;
            if !candidates.iter().any(|c| c & bit != 0) {
                continue;
            }
            composed += 1;
            let mref = self
                .store
                .method_ref(self.family.class, Family::method(i))
                .map_err(TwinRefusal::Admit)?;
            let binding = self.store.admit(&mref).map_err(TwinRefusal::Admit)?;
            let bound = binding.when.as_ref().ok_or(TwinRefusal::NoGuard)?;
            if described.guard(i, described.producer).witness() != bound.witness() {
                return Err(TwinRefusal::DescriptionMismatch);
            }
            let program = lower(&Query {
                filter: described.filter(i),
                agg: Agg::Rows,
            })
            .map_err(TwinRefusal::Lower)?;
            let mut scratch = Scratch::new(words, program.scratch_slots as usize);
            let mut out = vec![0u64; words];
            execute_into(
                &program,
                &planes,
                &Foreign::NONE,
                &mut scratch,
                Out::Mask(&mut out),
            )
            .expect("lowered program is valid");
            for (r, h) in holds.iter_mut().enumerate() {
                if candidates[r] & bit != 0 && out[r / 64] >> (r % 64) & 1 == 1 {
                    *h |= bit;
                }
            }
        }
        Ok(Answers { holds, composed })
    }
}

/// The lab row class's field basis: placeholder fields.
struct LabView {
    class: u32,
    basis: Vec<FieldRef>,
}

impl LabView {
    fn new(class: u32, lanes: usize) -> Self {
        LabView {
            class,
            basis: (0..lanes)
                .map(|i| {
                    FieldRef::new(
                        format!("urn:hidden-state-twins-probe/f{i}"),
                        format!("f{i}"),
                    )
                })
                .collect(),
        }
    }
}

impl ClassView for LabView {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if lance_graph_dt64::view_class(self.class) == class {
            &self.basis
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// FNV-1a, for producer identities.
pub fn digest(parts: &[&str]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for p in parts {
        for &b in p.as_bytes() {
            h ^= u32::from(b);
            h = h.wrapping_mul(0x0100_0193);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}
