//! Gate B: which component of a method changed, on what evidence, and which
//! obligations that makes stale.
//!
//! Two queries, because the four kinds of change are visible in two
//! different places:
//!
//! - [`diff`] compares two fact snapshots. A changed ON declaration, a
//!   changed WHEN guard, a changed DO registration and a changed field basis
//!   each change a different registered fact.
//! - [`impl_change`] takes a changed source line. A changed DO
//!   IMPLEMENTATION changes no registered fact at all, so it is found from
//!   the dispatch arm the line sits in.

use crate::dispatch::{self, Arm, Dispatch};
use crate::facts::{MethodFacts, Snapshot};
use crate::Evidence;

/// A part of a method's registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Component {
    /// The ordered predicates of the class basis.
    Basis,
    Signature,
    /// ON: the trigger field positions.
    On,
    Reads,
    /// WHEN: the guard fires on other rows (another witness).
    When,
    /// WHEN is written differently and fires on the same rows (same witness).
    WhenSpelling,
    /// DO: the ordered steps.
    Do,
    /// Population, truth or value demand.
    Demand,
}

/// Something that was established against the old state and no longer holds.
/// Each names the DT64 answer that shows it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Obligation {
    /// `resolve` and `bind_lanes` over the changed basis at the bound
    /// revision: `PredicateBasisChanged`.
    WitnessedResolve,
    /// A reference made at the old signature revision:
    /// `SignatureRevisionMismatch`.
    SignatureRevision,
    /// The guard witness a reference carries: `GuardMismatch`, `GuardDropped`
    /// or `GuardAdded`.
    GuardWitness,
    /// The stamp of a reference issued for the old binding:
    /// `BindingMismatch`.
    MethodRefStamp,
    /// The answer of `Resolved::methods_on` for the moved positions.
    TriggerSet,
    /// An executed comparison of this word against an oracle. This crate
    /// holds none, so it is always open.
    ExecutedParity { lens: u32, word: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Unchanged,
    Changed(Vec<Component>),
    Added,
    Removed,
}

/// The first refusal `admit` gives a reference issued from the old state when
/// it is presented to a store holding the new state. `admit` checks in this
/// order; a store other than the issuer is refused last, whatever else holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OldRef {
    Signature,
    Guard,
    Binding,
    IssuerOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodDelta {
    pub class: u32,
    pub method: u32,
    pub status: Status,
    pub binding_changed: bool,
    pub evidence: Vec<Evidence>,
    pub stale: Vec<Obligation>,
}

impl MethodDelta {
    pub fn has(&self, c: Component) -> bool {
        matches!(&self.status, Status::Changed(cs) if cs.contains(&c))
    }

    /// See [`OldRef`]. `None` for an added or removed method.
    pub fn old_ref(&self) -> Option<OldRef> {
        match &self.status {
            Status::Added | Status::Removed => None,
            Status::Unchanged => Some(OldRef::IssuerOnly),
            Status::Changed(cs) => Some(if cs.contains(&Component::Signature) {
                OldRef::Signature
            } else if cs.contains(&Component::When) {
                OldRef::Guard
            } else if self.binding_changed {
                OldRef::Binding
            } else {
                OldRef::IssuerOnly
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffRefusal {
    /// A snapshot has a method whose class it has no basis for.
    ClassMissing { class: u32 },
    /// The facts and the binding digest disagree: the digest moved with no
    /// component that it covers, or a component it covers moved and the
    /// digest did not. Either the facts were altered, or something the facts
    /// do not carry changed (the lens revision, a word's arity).
    Inconsistent {
        class: u32,
        method: u32,
        why: &'static str,
    },
}

fn positions(fields: &[crate::facts::Field]) -> Vec<u8> {
    fields.iter().map(|f| f.position).collect()
}

fn do_key(m: &MethodFacts) -> Vec<(u32, u8, u8, u8, Vec<String>)> {
    m.steps
        .iter()
        .map(|s| {
            (
                s.lens,
                s.word,
                s.arity,
                s.target.position,
                s.args
                    .iter()
                    .map(|a| match a {
                        crate::facts::ArgFact::Field(f) => format!("F{}", f.position),
                        crate::facts::ArgFact::Const(c) => format!("C{c}"),
                    })
                    .collect(),
            )
        })
        .collect()
}

fn one(
    before: &Snapshot,
    after: &Snapshot,
    b: &MethodFacts,
    a: &MethodFacts,
) -> Result<MethodDelta, DiffRefusal> {
    let (class, method) = (b.class, b.method);
    let basis = |s: &Snapshot| {
        s.class(class)
            .map(|c| c.basis_fnv)
            .ok_or(DiffRefusal::ClassMissing { class })
    };
    let (basis_b, basis_a) = (basis(before)?, basis(after)?);
    let mut cs = Vec::new();
    let mut evidence = Vec::new();
    let mut stale = Vec::new();

    if basis_b != basis_a {
        cs.push(Component::Basis);
        evidence.push(Evidence::registration(format!(
            "class {class:#010X} basis: predicates in position order {basis_b:016x} -> {basis_a:016x} (resolve accepted each against its own store)"
        )));
        stale.push(Obligation::WitnessedResolve);
    }
    if b.sig_rev != a.sig_rev {
        cs.push(Component::Signature);
        evidence.push(Evidence::registration(format!(
            "sig_rev {} -> {}",
            b.sig_rev, a.sig_rev
        )));
        stale.push(Obligation::SignatureRevision);
    }
    if positions(&b.on) != positions(&a.on) {
        cs.push(Component::On);
        evidence.push(Evidence::registration(format!(
            "ON positions {:?} -> {:?}",
            positions(&b.on),
            positions(&a.on)
        )));
        stale.push(Obligation::TriggerSet);
    }
    if positions(&b.reads) != positions(&a.reads) {
        cs.push(Component::Reads);
        evidence.push(Evidence::registration(format!(
            "reads positions {:?} -> {:?}",
            positions(&b.reads),
            positions(&a.reads)
        )));
    }
    let witness = |m: &MethodFacts| m.when.as_ref().map(|w| (w.logic, w.witness.clone()));
    let spelling = |m: &MethodFacts| m.when.as_ref().map(|w| w.expr.clone());
    if witness(b) != witness(a) {
        cs.push(Component::When);
        evidence.push(Evidence::registration(format!(
            "WHEN witness {:?} -> {:?}",
            witness(b),
            witness(a)
        )));
        stale.push(Obligation::GuardWitness);
    } else if spelling(b) != spelling(a) {
        cs.push(Component::WhenSpelling);
        evidence.push(Evidence::registration(
            "WHEN expression differs, witness equal: fires on the same rows",
        ));
    }
    if do_key(b) != do_key(a) {
        cs.push(Component::Do);
        evidence.push(Evidence::registration(format!(
            "DO steps (lens, word, arity, target, args) in order {:?} -> {:?}",
            do_key(b),
            do_key(a)
        )));
    }
    if (b.population, b.truth, b.demand) != (a.population, a.truth, a.demand) {
        cs.push(Component::Demand);
        evidence.push(Evidence::registration(
            "population, truth or demand differs",
        ));
    }

    // The binding digest covers the basis, the signature revision, ON, reads,
    // the ordered DO steps and the demand. It does not cover the guard.
    let covered = cs.iter().any(|c| {
        matches!(
            c,
            Component::Basis
                | Component::Signature
                | Component::On
                | Component::Reads
                | Component::Do
                | Component::Demand
        )
    });
    let binding_changed = b.binding != a.binding;
    if binding_changed && !covered {
        return Err(DiffRefusal::Inconsistent {
            class,
            method,
            why: "binding digest moved and no component it covers did",
        });
    }
    if covered && !binding_changed {
        return Err(DiffRefusal::Inconsistent {
            class,
            method,
            why: "a component the binding digest covers moved and the digest did not",
        });
    }
    if binding_changed {
        evidence.push(Evidence::registration(format!(
            "binding digest {:032x} -> {:032x}",
            b.binding, a.binding
        )));
        stale.push(Obligation::MethodRefStamp);
    }
    stale.sort();
    Ok(MethodDelta {
        class,
        method,
        status: if cs.is_empty() {
            Status::Unchanged
        } else {
            Status::Changed(cs)
        },
        binding_changed,
        evidence,
        stale,
    })
}

/// Compare the registered facts of every method in either snapshot. A method
/// that did not change is reported as [`Status::Unchanged`], with no stale
/// obligation; it is never left out.
pub fn diff(before: &Snapshot, after: &Snapshot) -> Result<Vec<MethodDelta>, DiffRefusal> {
    let mut out = Vec::new();
    for b in &before.methods {
        match after.method(b.class, b.method) {
            Some(a) => out.push(one(before, after, b, a)?),
            None => out.push(MethodDelta {
                class: b.class,
                method: b.method,
                status: Status::Removed,
                binding_changed: true,
                evidence: vec![Evidence::registration("method absent from the later facts")],
                stale: vec![Obligation::MethodRefStamp],
            }),
        }
    }
    for a in &after.methods {
        if before.method(a.class, a.method).is_none() {
            out.push(MethodDelta {
                class: a.class,
                method: a.method,
                status: Status::Added,
                binding_changed: true,
                evidence: vec![Evidence::registration(
                    "method absent from the earlier facts",
                )],
                stale: Vec::new(),
            });
        }
    }
    Ok(out)
}

/// One word a changed arm realizes, and the DO steps registered on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordUse {
    pub lens: u32,
    pub word: u8,
    pub opcode: String,
    /// `(class, method, ordinal)`. Empty means no registered user IN THIS
    /// SNAPSHOT; it does not mean the word is unused or the change harmless.
    pub users: Vec<(u32, u32, usize)>,
    pub stale: Vec<Obligation>,
}

/// What a changed line in the dispatch function reaches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplImpact {
    pub arm: Arm,
    pub words: Vec<WordUse>,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImplRefusal {
    /// The dispatch reading was taken from other text than the current
    /// source. No impact row is produced from it.
    StaleWitness { recorded: u64, current: u64 },
    /// The line is in no dispatch arm. There is no behaviour edge from it
    /// here; that is not a statement that the change has no effect.
    NotInDispatch { line: usize },
    /// The arm names no polyfill word (a loader, a terminal, the catch-all).
    ArmNotAWord { names: Vec<String> },
    /// The word has no arm in this reading: what realizes it is UNKNOWN.
    NoDispatchArm { word: u8 },
}

fn word_use(snap: &Snapshot, lens: u32, word: u8, opcode: &str) -> WordUse {
    let users: Vec<(u32, u32, usize)> = snap
        .methods
        .iter()
        .flat_map(|m| {
            m.steps
                .iter()
                .filter(move |s| s.lens == lens && s.word == word)
                .map(move |s| (m.class, m.method, s.ordinal))
        })
        .collect();
    let stale = if users.is_empty() {
        Vec::new()
    } else {
        vec![Obligation::ExecutedParity { lens, word }]
    };
    WordUse {
        lens,
        word,
        opcode: opcode.to_string(),
        users,
        stale,
    }
}

fn check_witness(dispatch: &Dispatch, current_src: &str) -> Result<(), ImplRefusal> {
    let current = dispatch::text_fnv(current_src);
    if current != dispatch.file_fnv {
        return Err(ImplRefusal::StaleWitness {
            recorded: dispatch.file_fnv,
            current,
        });
    }
    Ok(())
}

/// A changed line of the dispatch source: the arm it is in, the words that
/// arm realizes, and the registered DO steps on those words. No registered
/// fact changes, so no reference or binding becomes stale; what becomes open
/// is the executed parity of each word that has a user.
pub fn impl_change(
    snap: &Snapshot,
    dispatch: &Dispatch,
    current_src: &str,
    lens: u32,
    line: usize,
) -> Result<ImplImpact, ImplRefusal> {
    check_witness(dispatch, current_src)?;
    let arm = dispatch
        .arm_at(line)
        .ok_or(ImplRefusal::NotInDispatch { line })?;
    let words: Vec<WordUse> = arm
        .names
        .iter()
        .filter_map(|n| dispatch::word_of(n).map(|w| word_use(snap, lens, w, n)))
        .collect();
    if words.is_empty() {
        return Err(ImplRefusal::ArmNotAWord {
            names: arm.names.clone(),
        });
    }
    let evidence = vec![
        Evidence::source(format!(
            "line {line} is inside the match arm `{}` (lines {}..={}) of FoldDialect::call",
            arm.names.join(" | "),
            arm.start,
            arm.end
        )),
        Evidence::source(
            "opcode name -> polyfill position: the compiled POLYFILL table, by the constant the name resolves to",
        ),
        Evidence::registration(format!(
            "DO steps on those positions under lens {lens:#010X}, from the snapshot at {}",
            snap.source_rev
        )),
    ];
    Ok(ImplImpact {
        arm: arm.clone(),
        words,
        evidence,
    })
}

/// The registered DO steps on one word, if this reading says what realizes
/// it. A word with no arm is refused: its users are not reported as
/// unaffected.
pub fn word_impact(
    snap: &Snapshot,
    dispatch: &Dispatch,
    current_src: &str,
    lens: u32,
    word: u8,
) -> Result<WordUse, ImplRefusal> {
    check_witness(dispatch, current_src)?;
    dispatch
        .arm_of_word(word)
        .ok_or(ImplRefusal::NoDispatchArm { word })?;
    let opcode = dispatch::opcode_name(word).ok_or(ImplRefusal::NoDispatchArm { word })?;
    Ok(word_use(snap, lens, word, opcode))
}
