//! P3: a cold projection of an OGAR `ActionDef` onto the fields of the DT64
//! `MethodBinding` contract (`lance-graph-dt64/src/lib.rs`, `MethodBinding`).
//!
//! Nothing here builds a `MethodBinding`. Each binding field gets a class
//! saying how much of it the OGAR source actually states:
//!
//! - **Direct**: the source states it in the shape the field needs.
//! - **SoundlyDerivable**: a rule maps it without loss.
//! - **OverApproximated**: a superset is derivable; using it would fire or
//!   read more than the method does.
//! - **Unknown**: the source has something, but cannot say whether it is
//!   complete.
//! - **NotRepresentable**: the source has nothing the field could be built
//!   from, or the field needs data no source carries.
//!
//! The rule is: nothing is invented. A missing operand, value, order, guard
//! or algebra word is reported, never filled in.
//!
//! The field basis is supplied by the caller (`Basis`). In the tests it is
//! the lifted `Class`'s attribute and association names, in order. The real
//! basis is the `ClassView` of the method's classid; the harvested Odoo
//! models mint classid 0 (bootstrap), so there is no `ClassView` to read.
//! That is recorded as a gap, not hidden.

use lance_graph_contract::class_view::WideFieldMask;
use ogar_vocab::{ActionDef, KausalSpec};

/// How much of a binding field the source states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Grade {
    Direct,
    SoundlyDerivable,
    OverApproximated,
    Unknown,
    NotRepresentable,
}

/// The `MethodBinding` fields, in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Field {
    Id,
    SigRev,
    Name,
    On,
    When,
    Reads,
    EffectTargets,
    EffectAlgebra,
    EffectArgs,
    EffectOrder,
    Population,
    Truth,
    Demand,
}

pub const FIELDS: [Field; 13] = [
    Field::Id,
    Field::SigRev,
    Field::Name,
    Field::On,
    Field::When,
    Field::Reads,
    Field::EffectTargets,
    Field::EffectAlgebra,
    Field::EffectArgs,
    Field::EffectOrder,
    Field::Population,
    Field::Truth,
    Field::Demand,
];

/// One field's grade and the reason, in words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graded {
    pub field: Field,
    pub grade: Grade,
    pub why: String,
}

/// A name → field-position lookup over one class.
pub struct Basis {
    names: Vec<String>,
}

impl Basis {
    pub fn new(names: Vec<String>) -> Basis {
        Basis { names }
    }

    pub fn position(&self, name: &str) -> Option<u8> {
        self.names
            .iter()
            .position(|n| n == name)
            .and_then(|p| u8::try_from(p).ok())
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// The projection of one `ActionDef`.
#[derive(Debug, Clone)]
pub struct Projection {
    pub identity: String,
    pub grades: Vec<Graded>,
    /// The ON mask, when the grade allows one to be built.
    pub on: Option<WideFieldMask>,
    /// The resolved read positions (never complete: reads are inferred).
    pub reads: WideFieldMask,
    /// The resolved write targets (complete as names, values absent).
    pub write_targets: WideFieldMask,
}

impl Projection {
    pub fn grade(&self, f: Field) -> Grade {
        self.grades
            .iter()
            .find(|g| g.field == f)
            .map(|g| g.grade)
            .expect("every field is graded")
    }

    /// The fields that prevent a sound `MethodBinding`: anything not Direct
    /// or SoundlyDerivable.
    pub fn blockers(&self) -> Vec<&Graded> {
        self.grades
            .iter()
            .filter(|g| g.grade > Grade::SoundlyDerivable)
            .collect()
    }
}

fn g(field: Field, grade: Grade, why: impl Into<String>) -> Graded {
    Graded {
        field,
        grade,
        why: why.into(),
    }
}

/// Resolve every name; `None` names the first one that is not a single
/// field of this class (a dotted path, or a name outside the basis).
fn resolve_all(basis: &Basis, names: &[String]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        if n.contains('.') {
            return Err(format!("`{n}` is a dotted path into another class"));
        }
        match basis.position(n) {
            Some(p) => out.push(p),
            None => return Err(format!("`{n}` is not a field of this class")),
        }
    }
    Ok(out)
}

fn grade_on(a: &ActionDef, basis: &Basis) -> (Graded, Option<WideFieldMask>) {
    let paths = match &a.kausal {
        Some(KausalSpec::Depends { paths })
        | Some(KausalSpec::Constrains { paths })
        | Some(KausalSpec::Onchange { paths }) => paths,
        Some(KausalSpec::StateGuard { .. }) => {
            return (
                g(Field::On, Grade::Unknown, "a state guard says WHEN, not what triggers the method"),
                None,
            )
        }
        Some(KausalSpec::LifecycleTrigger { event }) => {
            return (
                g(
                    Field::On,
                    Grade::OverApproximated,
                    format!("lifecycle event `{event}`: any field change of the record, a superset of the method's own trigger"),
                ),
                None,
            )
        }
        Some(KausalSpec::ContextDepends { keys }) => {
            return (
                g(
                    Field::On,
                    Grade::NotRepresentable,
                    format!("{} environment key(s); ON is a mask over class fields", keys.len()),
                ),
                None,
            )
        }
        Some(KausalSpec::External) => {
            return (
                g(Field::On, Grade::NotRepresentable, "an external cause (RPC/HTTP) has no field trigger"),
                None,
            )
        }
        Some(_) => {
            return (
                g(Field::On, Grade::Unknown, "a KausalSpec variant this probe does not know"),
                None,
            )
        }
        None => return (g(Field::On, Grade::Unknown, "no kausal fact"), None),
    };
    match resolve_all(basis, paths) {
        Ok(ps) => (
            g(
                Field::On,
                Grade::Direct,
                format!("{} single-field path(s)", ps.len()),
            ),
            Some(WideFieldMask::from_positions(&ps)),
        ),
        Err(why) => (
            g(
                Field::On,
                Grade::NotRepresentable,
                format!("{why}; a change in a related record has no position in this class's mask"),
            ),
            None,
        ),
    }
}

fn grade_when(a: &ActionDef, basis: &Basis) -> Graded {
    match &a.kausal {
        Some(KausalSpec::StateGuard {
            guard_field,
            guard_values,
        }) => {
            let field = if basis.position(guard_field).is_some() {
                "resolves"
            } else {
                "does not resolve"
            };
            g(
                Field::When,
                Grade::NotRepresentable,
                format!(
                    "guard field `{guard_field}` {field}, but its {} value(s) are strings and a DT64 Guard compares lane values; no value codebook maps them",
                    guard_values.len()
                ),
            )
        }
        _ if !a.raises.is_empty() => g(
            Field::When,
            Grade::Unknown,
            format!(
                "no declared guard, but the body raises {:?}; a guard inside the body is not represented",
                a.raises
            ),
        ),
        _ => g(
            Field::When,
            Grade::Unknown,
            "no declared guard; whether the body guards is not represented",
        ),
    }
}

/// Grade one `ActionDef` against a class basis.
pub fn project(a: &ActionDef, basis: &Basis) -> Projection {
    let mut grades = Vec::with_capacity(FIELDS.len());
    grades.push(g(
        Field::Id,
        Grade::NotRepresentable,
        "ActionDef has a string identity; a MethodId is chosen by its registrant (owner decision)",
    ));
    grades.push(g(
        Field::SigRev,
        Grade::NotRepresentable,
        "no revision exists on ActionDef",
    ));
    grades.push(g(Field::Name, Grade::Direct, a.predicate.clone()));

    let (on, on_mask) = grade_on(a, basis);
    grades.push(on);
    grades.push(grade_when(a, basis));

    let resolved_reads: Vec<u8> = a.reads.iter().filter_map(|n| basis.position(n)).collect();
    grades.push(if a.reads.is_empty() {
        g(
            Field::Reads,
            Grade::Unknown,
            "no read facts; reads are inferred, so empty is not proof of none",
        )
    } else {
        g(
            Field::Reads,
            Grade::Unknown,
            format!(
                "{} of {} inferred read(s) resolve; inferred reads may be incomplete",
                resolved_reads.len(),
                a.reads.len()
            ),
        )
    });

    let mut target_names: Vec<String> = a.writes.clone();
    if let Some(e) = &a.on_enter {
        target_names.push(e.field.clone());
    }
    let resolved_writes: Vec<u8> = target_names
        .iter()
        .filter_map(|n| basis.position(n))
        .collect();
    grades.push(if target_names.is_empty() {
        g(
            Field::EffectTargets,
            Grade::Unknown,
            "no write facts; a write through a call or a helper is not captured",
        )
    } else if resolved_writes.len() == target_names.len() {
        g(
            Field::EffectTargets,
            Grade::Direct,
            format!("{} write target(s), source-attested", target_names.len()),
        )
    } else {
        g(
            Field::EffectTargets,
            Grade::Unknown,
            format!(
                "{} of {} write target(s) resolve in the basis",
                resolved_writes.len(),
                target_names.len()
            ),
        )
    });
    grades.push(g(
        Field::EffectAlgebra,
        Grade::NotRepresentable,
        "no algebra word: the value written is not captured (ActionDef::writes is name-level)",
    ));
    grades.push(match &a.on_enter {
        Some(e) => g(
            Field::EffectArgs,
            Grade::NotRepresentable,
            format!(
                "on_enter sets `{}` to the string `{}`; no typed value or operand",
                e.field, e.to_value
            ),
        ),
        None => g(
            Field::EffectArgs,
            Grade::NotRepresentable,
            "no operands are captured",
        ),
    });
    grades.push(g(
        Field::EffectOrder,
        if target_names.len() <= 1 {
            Grade::SoundlyDerivable
        } else {
            Grade::NotRepresentable
        },
        if target_names.len() <= 1 {
            "at most one write: order is trivial".to_owned()
        } else {
            format!(
                "{} writes as a set; their order is not captured",
                target_names.len()
            )
        },
    ));
    grades.push(g(
        Field::Population,
        Grade::NotRepresentable,
        "no source states it",
    ));
    grades.push(g(
        Field::Truth,
        Grade::NotRepresentable,
        "no source states it",
    ));
    grades.push(g(
        Field::Demand,
        Grade::NotRepresentable,
        "no source states it",
    ));

    Projection {
        identity: a.identity.clone(),
        grades,
        on: on_mask,
        reads: WideFieldMask::from_positions(&resolved_reads),
        write_targets: WideFieldMask::from_positions(&resolved_writes),
    }
}
