//! X2: a typed execution-admission policy for DT64 methods (LAB).
//!
//! [`BindStore::admit`] authenticates a router's [`MethodRef`]: the method
//! exists, at the referenced signature revision, with exactly the guard the
//! reference witnesses. It says nothing about what the method's source does
//! when it runs. A binding that admits is bound, not authorized.
//!
//! [`ExecutionPolicy::decide`] runs `admit` unchanged and then reads the
//! method's D-SAG-6 evidence rows ([`evidence`]) to decide whether it may
//! execute. The answer is one of:
//!
//! | Evidence of the method's source | Answer |
//! |---|---|
//! | 1. proven pure | [`Admitted::Pure`] |
//! | 2. a WITNESSED effect, under a DO grant that covers every effect row | [`Admitted::UnderDoGrant`] |
//! | 3. any POSSIBLE row | [`ExecRefusal::Possible`] |
//! | 4. any UNKNOWN fact | [`ExecRefusal::Unknown`] |
//! | 5. a WITNESSED effect and no covering DO grant | [`ExecRefusal::RequiresDoGate`] (or [`ExecRefusal::GrantDoesNotCover`] / [`ExecRefusal::GrantForOtherMethod`]) |
//!
//! UNKNOWN dominates POSSIBLE, which dominates a witnessed effect. A DO
//! grant authorizes witnessed effects only: it never lifts a POSSIBLE row or
//! an UNKNOWN fact. Every refusal is a named variant, never an empty answer.
//!
//! # What "proven pure" means here
//!
//! The method is attached ([`Attachment`]) to a subject the frontend defined
//! (a `DEFINES` row), and the subject, every nested subject (closures, by the
//! `subject::` prefix) and every local callee (a `CALLS` target with its own
//! `DEFINES` row) carry no effect, POSSIBLE or UNKNOWN row. A call to a
//! non-local callee is UNKNOWN unless the policy names it as trusted
//! ([`ExecutionPolicy::trusted_external`]); the trusted callees a pure
//! verdict relies on are returned with it ([`Admitted::Pure::assumed`]), so
//! the assumption is visible rather than silent.
//!
//! Absence of rows is not purity. A method with no attachment, or attached to
//! a subject with no `DEFINES` row, is UNKNOWN.
//!
//! # Identity
//!
//! Evidence attaches to `(class, MethodId, SigRev)`, never to
//! [`MethodBinding::name`], which DT64 carries and never uses as identity.
//! Evidence recorded for another signature revision does not attach.
//!
//! [`MethodBinding::name`]: lance_graph_dt64::MethodBinding::name

#![forbid(unsafe_code)]

pub mod evidence;
pub mod fixture;

use std::collections::BTreeSet;

use evidence::{Evidence, Relation, Row};
use lance_graph_dt64::{BindStore, MethodBinding, MethodId, MethodRef, Refusal, SigRev};

/// Which evidence subject a method's source is, recorded per
/// `(class, method, signature revision)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attachment {
    pub class: u32,
    pub method: MethodId,
    pub sig_rev: SigRev,
    /// A subject spelling in the rows (a rust-analyzer moniker after the
    /// span join, a Charon path under the path join).
    pub subject: String,
}

/// One evidence set: the rows of one projection run and the attachments of
/// methods to its subjects.
#[derive(Debug, Clone, Default)]
pub struct EvidenceSet {
    pub rows: Vec<Row>,
    pub attachments: Vec<Attachment>,
}

/// An explicit DO authorization for one method reference: which witnessed
/// effect rows its holder accepts. Issued outside this crate; the policy
/// only checks it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoGrant {
    pub method: MethodRef,
    /// `(relation, subject)` of every effect row the grant accepts.
    pub effects: Vec<(Relation, String)>,
}

/// What the caller asks for.
#[derive(Debug, Clone, Copy)]
pub enum Request<'g> {
    /// Execute (or fuse) as pure.
    Pure,
    /// Execute under a DO grant.
    Do(&'g DoGrant),
}

/// An execution the policy admits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admitted<'s> {
    /// Class 1: no effect, POSSIBLE or UNKNOWN row in the method's scope.
    Pure {
        binding: &'s MethodBinding,
        /// The trusted non-local callees this verdict relies on.
        assumed: Vec<String>,
    },
    /// Class 2: witnessed effects, every one covered by the grant.
    UnderDoGrant {
        binding: &'s MethodBinding,
        effects: Vec<Row>,
    },
}

/// Why a fact is UNKNOWN.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownCause {
    /// No attachment for this `(class, method, sig_rev)`.
    NotAttached {
        class: u32,
        method: MethodId,
        sig_rev: SigRev,
    },
    /// The attached subject has no `DEFINES` row: the frontend never saw it.
    NoDefinition { subject: String },
    /// An `UNJOINED` row in scope.
    Unjoined(Row),
    /// A resolved call whose callee the row does not name (`known impl`).
    UnnamedCallee(Row),
    /// A non-local callee the policy does not trust.
    UnauditedCallee(Row),
    /// A row at an evidence class its relation is never emitted at.
    Mislabelled(Row),
}

/// Why the policy refused. `admit`'s own refusal passes through unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecRefusal {
    /// `BindStore::admit` refused: not even bound.
    Binding(Refusal),
    /// Class 3: POSSIBLE rows. No grant lifts these.
    Possible { rows: Vec<Row> },
    /// Class 4: UNKNOWN facts. No grant lifts these.
    Unknown { causes: Vec<UnknownCause> },
    /// Class 5: witnessed effects and pure execution was asked.
    RequiresDoGate { effects: Vec<Row> },
    /// Class 5: the grant does not cover these effect rows.
    GrantDoesNotCover { uncovered: Vec<Row> },
    /// Class 5: the grant names another method reference.
    GrantForOtherMethod {
        granted: MethodRef,
        asked: MethodRef,
    },
}

/// The evidence class of one method's source, before any DO gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceClass {
    ProvenPure { assumed: Vec<String> },
    WitnessedEffect { effects: Vec<Row> },
    Possible { rows: Vec<Row> },
    Unknown { causes: Vec<UnknownCause> },
}

/// The execution-admission policy.
#[derive(Debug, Clone, Default)]
pub struct ExecutionPolicy {
    /// Path prefixes of non-local callees treated as pure (for example
    /// `core::iter::`). Empty means no non-local call is trusted.
    pub trusted_external: Vec<String>,
}

/// The CALLS object `project_rows.py` writes for a resolved trait impl it
/// does not name.
pub const UNNAMED_IMPL: &str = "known impl";

impl ExecutionPolicy {
    /// Authenticate with `admit`, then decide from `evidence`.
    pub fn decide<'s>(
        &self,
        store: &'s BindStore,
        r: &MethodRef,
        evidence: &EvidenceSet,
        request: Request<'_>,
    ) -> Result<Admitted<'s>, ExecRefusal> {
        let binding = store.admit(r).map_err(ExecRefusal::Binding)?;
        match self.classify(r, evidence) {
            EvidenceClass::Unknown { causes } => Err(ExecRefusal::Unknown { causes }),
            EvidenceClass::Possible { rows } => Err(ExecRefusal::Possible { rows }),
            EvidenceClass::ProvenPure { assumed } => Ok(Admitted::Pure { binding, assumed }),
            EvidenceClass::WitnessedEffect { effects } => match request {
                Request::Pure => Err(ExecRefusal::RequiresDoGate { effects }),
                Request::Do(grant) => {
                    if grant.method != *r {
                        return Err(ExecRefusal::GrantForOtherMethod {
                            granted: grant.method.clone(),
                            asked: r.clone(),
                        });
                    }
                    let uncovered: Vec<Row> = effects
                        .iter()
                        .filter(|e| {
                            !grant
                                .effects
                                .iter()
                                .any(|(rel, subj)| *rel == e.relation && *subj == e.subject)
                        })
                        .cloned()
                        .collect();
                    if uncovered.is_empty() {
                        Ok(Admitted::UnderDoGrant { binding, effects })
                    } else {
                        Err(ExecRefusal::GrantDoesNotCover { uncovered })
                    }
                }
            },
        }
    }

    /// The evidence class of the method `r` names. Does not authenticate.
    pub fn classify(&self, r: &MethodRef, evidence: &EvidenceSet) -> EvidenceClass {
        let Some(att) = evidence
            .attachments
            .iter()
            .find(|a| a.class == r.class && a.method == r.method && a.sig_rev == r.sig_rev)
        else {
            return EvidenceClass::Unknown {
                causes: vec![UnknownCause::NotAttached {
                    class: r.class,
                    method: r.method,
                    sig_rev: r.sig_rev,
                }],
            };
        };

        let defined = |s: &str| {
            evidence
                .rows
                .iter()
                .any(|row| row.relation == Relation::Defines && row.subject == s)
        };

        let mut unknown = Vec::new();
        let mut possible = Vec::new();
        let mut effects = Vec::new();
        let mut assumed = BTreeSet::new();

        let mut visited = BTreeSet::new();
        let mut todo = vec![att.subject.clone()];
        while let Some(subject) = todo.pop() {
            if !visited.insert(subject.clone()) {
                continue;
            }
            // Undefined is UNKNOWN; its rows are still read, so an UNJOINED
            // row under the same subject is reported too.
            if !defined(&subject) {
                unknown.push(UnknownCause::NoDefinition {
                    subject: subject.clone(),
                });
            }
            let nested = format!("{subject}::");
            for row in evidence
                .rows
                .iter()
                .filter(|row| row.subject == subject || row.subject.starts_with(&nested))
            {
                match (row.relation, row.evidence) {
                    (_, Evidence::Unknown) => unknown.push(UnknownCause::Unjoined(row.clone())),
                    (_, Evidence::Possible) => possible.push(row.clone()),
                    (Relation::MayPanic | Relation::WritesCallerMemory, Evidence::Witnessed) => {
                        effects.push(row.clone())
                    }
                    (Relation::Calls, Evidence::Witnessed) => {
                        // A callee nested under a subject already in scope
                        // (a closure body the span join named after its
                        // enclosing function, with no `DEFINES` row of its
                        // own) has its rows read through that subject's
                        // prefix. It is in scope, not an external callee.
                        let in_scope = visited
                            .iter()
                            .any(|v| row.object.starts_with(&format!("{v}::")));
                        if in_scope {
                        } else if row.object == UNNAMED_IMPL {
                            unknown.push(UnknownCause::UnnamedCallee(row.clone()));
                        } else if defined(&row.object) {
                            todo.push(row.object.clone());
                        } else if self
                            .trusted_external
                            .iter()
                            .any(|p| row.object.starts_with(p.as_str()))
                        {
                            assumed.insert(row.object.clone());
                        } else {
                            unknown.push(UnknownCause::UnauditedCallee(row.clone()));
                        }
                    }
                    (Relation::Defines | Relation::HasImpls, Evidence::Witnessed) => {}
                    // A relation at an evidence class `parse_rows` rejects.
                    // Rows built by hand may still carry one: never pure.
                    (_, Evidence::Witnessed) => {
                        unknown.push(UnknownCause::Mislabelled(row.clone()));
                    }
                }
            }
        }

        if !unknown.is_empty() {
            EvidenceClass::Unknown { causes: unknown }
        } else if !possible.is_empty() {
            EvidenceClass::Possible { rows: possible }
        } else if !effects.is_empty() {
            EvidenceClass::WitnessedEffect { effects }
        } else {
            EvidenceClass::ProvenPure {
                assumed: assumed.into_iter().collect(),
            }
        }
    }
}
