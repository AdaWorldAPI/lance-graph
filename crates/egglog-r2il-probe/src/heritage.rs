//! Stage E (lab): mathematics heritage as explicitly labelled source claims,
//! not as automatically certified executable proofs.
//!
//! The first runnable entry is the SYNTHETIC #1690 E1 substitution theorem.
//! Historical examples are catalogue leads only. Leibniz's product rule and
//! Euler's identity require new higher-level term grammars, domain guards and
//! independent theorem checkers before they may become proof capsules.
//!
//! The 27 atoms are a falsifiable vocabulary hypothesis, NOT an admitted
//! universal mathematical basis and NOT 27 R2IL opcodes.

use std::collections::BTreeSet;

use crate::Step;
use crate::artifact::{ReplayCapsule, ReplayRefusal};
use crate::e1_problem;

/// Proposal for a cross-domain low-code vocabulary, NOT the R2IL ISA.
/// Grouped 6 terms + 8 algebra + 7 relation + 6 derivation = 27 candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Atom27 {
    Const,
    Var,
    Lambda,
    Apply,
    Let,
    Tuple,
    Add,
    Neg,
    Multiply,
    Divide,
    Power,
    Equal,
    Order,
    If,
    Select,
    Project,
    Join,
    Union,
    Distinct,
    Group,
    Aggregate,
    Substitute,
    Rewrite,
    Differentiate,
    Integrate,
    Iterate,
    Prove,
}

pub const PROPOSED_ATOMS: [Atom27; 27] = [
    Atom27::Const,
    Atom27::Var,
    Atom27::Lambda,
    Atom27::Apply,
    Atom27::Let,
    Atom27::Tuple,
    Atom27::Add,
    Atom27::Neg,
    Atom27::Multiply,
    Atom27::Divide,
    Atom27::Power,
    Atom27::Equal,
    Atom27::Order,
    Atom27::If,
    Atom27::Select,
    Atom27::Project,
    Atom27::Join,
    Atom27::Union,
    Atom27::Distinct,
    Atom27::Group,
    Atom27::Aggregate,
    Atom27::Substitute,
    Atom27::Rewrite,
    Atom27::Differentiate,
    Atom27::Integrate,
    Atom27::Iterate,
    Atom27::Prove,
];

/// Unverified historical bibliographic lead, not a proof and not the
/// attribution of the modern formal reconstruction to its historical author.
#[derive(Debug, Clone)]
pub struct SourceLead {
    pub author: &'static str,
    pub work: &'static str,
    pub date_note: &'static str,
    pub candidate_catalogue_uri: &'static str,
}

/// Historical source and mathematical implementation certification are
/// orthogonal: proof replay can certify the math without establishing
/// historical authorship, and vice versa.
#[derive(Debug, Clone)]
pub struct HeritageEntry {
    pub id: &'static str,
    pub normalized_statement: &'static str,
    pub source: Option<SourceLead>,
    capsule: Option<ReplayCapsule>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeritageRefusal {
    NoCheckedMathematicalProof,
    Replay(ReplayRefusal),
}

impl HeritageEntry {
    pub fn synthetic_e1() -> Result<Self, ReplayRefusal> {
        let capsule = ReplayCapsule::certify(&e1_problem(), 8)?;
        Ok(Self {
            id: "synthetic-e1-1690",
            normalized_statement: "a=2b, b=2c, z=2a |- z=8c (exact integer premises)",
            source: None,
            capsule: Some(capsule),
        })
    }

    /// Historical lead only; the existing E1 grammar has no differentiation,
    /// function application or quantification over differentiable functions.
    pub fn leibniz_product_rule() -> Self {
        Self {
            id: "leibniz-product-rule-candidate",
            normalized_statement: "D(f*g)=D(f)*g+f*D(g), for differentiable f,g",
            source: Some(SourceLead {
                author: "Gottfried Wilhelm Leibniz",
                work: "Nova Methodus pro Maximis et Minimis",
                date_note: "1684; attribution and textual correspondence require primary-source review",
                candidate_catalogue_uri: "https://digitalcollections.hkust.edu.hk/Detail/objects/b540295",
            }),
            capsule: None,
        }
    }

    /// An additional corpus lead. No proof or original-edition claim.
    pub fn euler_identity() -> Self {
        Self {
            id: "euler-identity-candidate",
            normalized_statement: "exp(i*pi)+1=0 (complex exponential)",
            source: Some(SourceLead {
                author: "Leonhard Euler",
                work: "Historical attribution and notation to be verified",
                date_note: "18th-century work; no primary-source edition checked by this probe",
                candidate_catalogue_uri: "",
            }),
            capsule: None,
        }
    }

    pub fn is_mathematically_certified(&self) -> bool {
        self.capsule.is_some()
    }

    /// Replay is permitted only with an actual checked E1 capsule. A text
    /// claim and bibliographic lead can never implicitly become executable.
    pub fn replay(&self, value: i64) -> Result<i64, HeritageRefusal> {
        self.capsule
            .as_ref()
            .ok_or(HeritageRefusal::NoCheckedMathematicalProof)?
            .replay(value)
            .map_err(HeritageRefusal::Replay)
    }

    fn program(&self) -> Option<&[Step]> {
        self.capsule.as_ref().map(|c| c.program())
    }
}

/// Research fixture, intentionally mixing an executable *synthetic* theorem
/// with historically attributed *uncompiled* mathematical statements.
pub fn sample_corpus() -> Result<Vec<HeritageEntry>, ReplayRefusal> {
    Ok(vec![
        HeritageEntry::synthetic_e1()?,
        HeritageEntry::leibniz_product_rule(),
        HeritageEntry::euler_identity(),
    ])
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaStudy {
    pub total_entries: usize,
    pub historical_leads: usize,
    pub mathematically_certified: usize,
    pub distinct_compiled_programs: usize,
    /// Only atoms witnessed by an actually compiled R2IL program.
    pub observed_atoms: Vec<Atom27>,
}

/// Deterministic descriptive analysis. No equivalence between historical
/// methods, no complexity claim, and no unlicensed mathematical law inference.
pub fn analyse(corpus: &[HeritageEntry]) -> MetaStudy {
    let mut programs = BTreeSet::new();
    let mut atoms = BTreeSet::new();
    for entry in corpus {
        if let Some(steps) = entry.program() {
            programs.insert(steps.iter().map(Step::mnemonic).collect::<Vec<_>>());
            for step in steps {
                match step {
                    Step::PushConst(_) => { atoms.insert(Atom27::Const); }
                    Step::PushVar(_) => { atoms.insert(Atom27::Var); }
                    Step::IntMult => { atoms.insert(Atom27::Multiply); }
                }
            }
        }
    }
    MetaStudy {
        total_entries: corpus.len(),
        historical_leads: corpus.iter().filter(|x| x.source.is_some()).count(),
        mathematically_certified: corpus.iter()
            .filter(|x| x.is_mathematically_certified()).count(),
        distinct_compiled_programs: programs.len(),
        observed_atoms: atoms.into_iter().collect(),
    }
}
