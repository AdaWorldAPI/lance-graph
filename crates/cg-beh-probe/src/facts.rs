//! Gate A: the registered facts of one method, ON, WHEN and DO kept apart.

use lance_graph_contract::class_view::{ClassView, WideFieldMask};
use lance_graph_contract::set_algebra::{Logic, Res, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{Arg, BindStore, MethodId, MethodRef, Refusal, SchemaRev, ValueDemand};

use crate::{fnv64, FNV64_OFFSET};

/// A field position of the bound class and the predicate the pinned basis
/// gives it. The label is not carried: it is locale-bound and not identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub position: u8,
    pub predicate: String,
}

/// How a consumer reads the numbers inside a guard. `SetExpr::Sym` is "an
/// opaque two-valued set, identified by number" and a residual names "one
/// lane"; DT64 keeps the guard and checks neither against the basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardConvention {
    /// No reading is declared. Every dependency stays UNKNOWN.
    Opaque,
    /// The consumer states that a number is a field position of the class
    /// (Foundry's `method.rs` does). Accepted per dependency only when the
    /// position is in the method's registered `reads` and resolves through
    /// the basis.
    FieldPosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepKind {
    Sym,
    ResI32,
    ResU32,
}

/// One leaf a guard depends on. To DT64 it is always UNKNOWN; `consumer` is
/// the field it is under the declared convention, when that could be checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dep {
    pub kind: DepKind,
    pub number: u32,
    pub consumer: Option<Field>,
}

/// WHEN. `expr` and `witness` are the `Debug` forms of the registered
/// `SetExpr` and of `Guard::witness()` (its normal form under its logic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhenFacts {
    pub logic: &'static str,
    pub expr: String,
    pub witness: String,
    pub deps: Vec<Dep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgFact {
    Field(Field),
    Const(i64),
}

/// One DO step, at its place in the order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoStep {
    pub ordinal: usize,
    pub lens: u32,
    pub word: u8,
    /// The word's predicate in the lens basis the view gives.
    pub word_predicate: String,
    pub arity: u8,
    pub target: Field,
    pub args: Vec<ArgFact>,
}

/// Everything read about one method. `binding` is `Stamp::binding()` of the
/// reference the store issued and then admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodFacts {
    pub class: u32,
    pub method: u32,
    pub sig_rev: u32,
    pub name: String,
    pub binding: u128,
    pub on: Vec<Field>,
    pub reads: Vec<Field>,
    pub when: Option<WhenFacts>,
    pub steps: Vec<DoStep>,
    pub population: &'static str,
    pub truth: &'static str,
    pub demand: &'static str,
}

/// The pinned basis of one class, as the view gave it and the store accepted
/// it. `basis_fnv` is this crate's own digest of the ordered predicates (the
/// store's `BasisDigest` is private).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassFacts {
    pub class: u32,
    pub schema_rev: u32,
    pub basis_fnv: u64,
    pub basis: Vec<Field>,
}

/// Why facts could not be read. Each is a missing witness; none is an empty
/// answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Missing {
    /// The snapshot was given no source revision.
    SourceRevision,
    /// The store did not issue a reference for the method.
    Issue(Refusal),
    /// The store did not admit the reference (other issuer, other binding,
    /// other guard, other signature revision).
    Admit(Refusal),
    /// The view does not give the basis the class was bound against.
    Basis(Refusal),
    /// A method has DO steps and no lens revision was supplied to read the
    /// lens through.
    LensRevision { lens: u32 },
    /// The view does not give the lens the store bound.
    Lens(Refusal),
    /// A registered position does not resolve through the basis.
    Position { position: u8, role: &'static str },
    /// A word is outside the lens the view gives.
    Word { lens: u32, word: u8 },
}

/// What a read is made at.
#[derive(Debug, Clone, Copy)]
pub struct Read {
    pub schema_rev: SchemaRev,
    /// The revision the polyfill lens was declared at. `BindStore` does not
    /// expose it, so the caller states it and `resolve_algebra` checks it.
    pub lens_rev: Option<SchemaRev>,
    pub convention: GuardConvention,
}

pub(crate) fn positions(mask: &WideFieldMask) -> impl Iterator<Item = u8> + '_ {
    (0..=u8::MAX).filter(move |&p| mask.has(p))
}

fn deps_of(e: &SetExpr, out: &mut Vec<(DepKind, u32)>) {
    match e {
        SetExpr::Sym(n) => out.push((DepKind::Sym, *n)),
        SetExpr::Res(Res::I32 { lane, .. }) => out.push((DepKind::ResI32, u32::from(*lane))),
        SetExpr::Res(Res::U32 { lane, .. }) => out.push((DepKind::ResU32, u32::from(*lane))),
        SetExpr::And(cs) | SetExpr::Or(cs) => cs.iter().for_each(|c| deps_of(c, out)),
        SetExpr::Not(c) => deps_of(c, out),
        SetExpr::All | SetExpr::Empty => {}
    }
}

/// The pinned basis of `class` at `schema_rev`: `resolve` checks the revision,
/// the field count and the predicate witness before it answers.
pub fn class_facts<V: ClassView + ?Sized>(
    store: &BindStore,
    view: &V,
    class: u32,
    schema_rev: SchemaRev,
) -> Result<ClassFacts, Missing> {
    let resolved = store
        .resolve(view, class, schema_rev)
        .map_err(Missing::Basis)?;
    let mut h = fnv64(FNV64_OFFSET, b"cg-beh.basis.v1");
    let mut basis = Vec::with_capacity(resolved.fields.len());
    for (i, f) in resolved.fields.iter().enumerate() {
        let iri = f.predicate_iri.as_bytes();
        h = fnv64(h, &(iri.len() as u64).to_le_bytes());
        h = fnv64(h, iri);
        basis.push(Field {
            position: u8::try_from(i).map_err(|_| Missing::Position {
                position: u8::MAX,
                role: "basis",
            })?,
            predicate: f.predicate_iri.clone(),
        });
    }
    Ok(ClassFacts {
        class,
        schema_rev: schema_rev.0,
        basis_fnv: h,
        basis,
    })
}

/// The facts of the method `r` refers to. `r` must be a reference this store
/// issued for the binding it holds; anything else is [`Missing::Admit`].
pub fn extract<V: ClassView + ?Sized>(
    store: &BindStore,
    view: &V,
    r: &MethodRef,
    read: &Read,
) -> Result<MethodFacts, Missing> {
    let binding = store.admit(r).map_err(Missing::Admit)?;
    let resolved = store
        .resolve(view, r.class, read.schema_rev)
        .map_err(Missing::Basis)?;
    let field = |position: u8, role: &'static str| {
        resolved
            .fields
            .get(usize::from(position))
            .map(|f| Field {
                position,
                predicate: f.predicate_iri.clone(),
            })
            .ok_or(Missing::Position { position, role })
    };
    let on = positions(&binding.on)
        .map(|p| field(p, "on"))
        .collect::<Result<Vec<_>, _>>()?;
    let reads = positions(&binding.reads)
        .map(|p| field(p, "reads"))
        .collect::<Result<Vec<_>, _>>()?;

    let when = binding.when.as_ref().map(|g| {
        let mut leaves = Vec::new();
        deps_of(&g.expr, &mut leaves);
        let deps = leaves
            .into_iter()
            .map(|(kind, number)| {
                let consumer = match read.convention {
                    GuardConvention::Opaque => None,
                    GuardConvention::FieldPosition => u8::try_from(number)
                        .ok()
                        .filter(|p| binding.reads.has(*p))
                        .and_then(|p| field(p, "when").ok()),
                };
                Dep {
                    kind,
                    number,
                    consumer,
                }
            })
            .collect();
        WhenFacts {
            logic: match g.logic {
                Logic::TwoValued => "TwoValued",
                Logic::Kleene => "Kleene",
            },
            expr: format!("{:?}", g.expr),
            witness: format!("{:?}", g.witness()),
            deps,
        }
    });

    let mut steps = Vec::with_capacity(binding.effects.len());
    for (ordinal, e) in binding.effects.iter().enumerate() {
        let (lens, word) = (e.algebra.lens, e.algebra.word);
        let lens_rev = read.lens_rev.ok_or(Missing::LensRevision { lens })?;
        let words = store
            .resolve_algebra(view, lens, lens_rev)
            .map_err(Missing::Lens)?;
        let word_predicate = words
            .fields
            .get(usize::from(word))
            .map(|f| f.predicate_iri.clone())
            .ok_or(Missing::Word { lens, word })?;
        let arity = store.arity(e.algebra).ok_or(Missing::Word { lens, word })?;
        let args = e
            .args
            .iter()
            .map(|a| match *a {
                Arg::Field(p) => field(p, "do.arg").map(ArgFact::Field),
                Arg::Const(c) => Ok(ArgFact::Const(c)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        steps.push(DoStep {
            ordinal,
            lens,
            word,
            word_predicate,
            arity,
            target: field(e.target, "do.target")?,
            args,
        });
    }

    Ok(MethodFacts {
        class: r.class,
        method: r.method.0,
        sig_rev: binding.sig_rev.0,
        name: binding.name.clone(),
        binding: r.stamp.binding(),
        on,
        reads,
        when,
        steps,
        population: match binding.population {
            Population::Distinct => "Distinct",
            Population::Bag => "Bag",
        },
        truth: match binding.truth {
            Truth::TwoValued => "TwoValued",
            Truth::ThreeValued => "ThreeValued",
        },
        demand: match binding.demand {
            ValueDemand::MapOnly => "MapOnly",
            ValueDemand::Values => "Values",
        },
    })
}

/// The facts of several classes at one source revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub source_rev: String,
    pub label: String,
    pub classes: Vec<ClassFacts>,
    pub methods: Vec<MethodFacts>,
}

impl Snapshot {
    /// An empty snapshot. A blank or `UNKNOWN` source revision is refused: a
    /// snapshot that cannot say what it was taken from is not evidence.
    pub fn new(source_rev: &str, label: &str) -> Result<Snapshot, Missing> {
        let rev = source_rev.trim();
        if rev.is_empty() || rev.eq_ignore_ascii_case("unknown") {
            return Err(Missing::SourceRevision);
        }
        Ok(Snapshot {
            source_rev: rev.to_string(),
            label: label.to_string(),
            classes: Vec::new(),
            methods: Vec::new(),
        })
    }

    /// Read `class` and the methods `ids` of it. The store issues each
    /// reference and admits it again before anything is recorded.
    pub fn add_class<V: ClassView + ?Sized>(
        &mut self,
        store: &BindStore,
        view: &V,
        class: u32,
        ids: &[u32],
        read: &Read,
    ) -> Result<(), Missing> {
        let class_facts = class_facts(store, view, class, read.schema_rev)?;
        let mut methods = Vec::with_capacity(ids.len());
        for &id in ids {
            let r = store
                .method_ref(class, MethodId(id))
                .map_err(Missing::Issue)?;
            methods.push(extract(store, view, &r, read)?);
        }
        self.classes.push(class_facts);
        self.methods.extend(methods);
        Ok(())
    }

    pub fn method(&self, class: u32, method: u32) -> Option<&MethodFacts> {
        self.methods
            .iter()
            .find(|m| m.class == class && m.method == method)
    }

    pub fn class(&self, class: u32) -> Option<&ClassFacts> {
        self.classes.iter().find(|c| c.class == class)
    }

    /// The snapshot as `cg-beh.facts.v1` rows (see the crate README). One row
    /// per fact, tab separated, the row kind first.
    pub fn to_tsv(&self) -> String {
        let mut o = String::new();
        let mut row = |cells: &[String]| {
            o.push_str(&cells.join("\t"));
            o.push('\n');
        };
        let s = |v: &str| v.to_string();
        let unknown = || s("UNKNOWN");
        row(&[
            s("H"),
            s("cg-beh.facts.v1"),
            self.source_rev.clone(),
            self.label.clone(),
        ]);
        for c in &self.classes {
            row(&[
                s("CLASS"),
                format!("{:#010X}", c.class),
                c.schema_rev.to_string(),
                c.basis.len().to_string(),
                format!("{:016x}", c.basis_fnv),
            ]);
            for f in &c.basis {
                row(&[
                    s("BASIS"),
                    format!("{:#010X}", c.class),
                    f.position.to_string(),
                    f.predicate.clone(),
                ]);
            }
        }
        for m in &self.methods {
            let key = [format!("{:#010X}", m.class), m.method.to_string()];
            row(&[
                s("METHOD"),
                key[0].clone(),
                key[1].clone(),
                m.sig_rev.to_string(),
                format!("{:032x}", m.binding),
                s("ADMITTED_BY_ISSUER"),
                s(m.population),
                s(m.truth),
                s(m.demand),
                m.name.clone(),
            ]);
            for (kind, fields) in [("ON", &m.on), ("READ", &m.reads)] {
                for f in fields {
                    row(&[
                        s(kind),
                        key[0].clone(),
                        key[1].clone(),
                        f.position.to_string(),
                        f.predicate.clone(),
                    ]);
                }
            }
            match &m.when {
                None => row(&[s("WHEN_NONE"), key[0].clone(), key[1].clone()]),
                Some(w) => {
                    row(&[
                        s("WHEN"),
                        key[0].clone(),
                        key[1].clone(),
                        s(w.logic),
                        w.expr.clone(),
                        w.witness.clone(),
                    ]);
                    for (i, d) in w.deps.iter().enumerate() {
                        let (pos, pred) = match &d.consumer {
                            Some(f) => (f.position.to_string(), f.predicate.clone()),
                            None => (unknown(), unknown()),
                        };
                        row(&[
                            s("DEP"),
                            key[0].clone(),
                            key[1].clone(),
                            i.to_string(),
                            format!("{:?}", d.kind),
                            d.number.to_string(),
                            unknown(),
                            pos,
                            pred,
                        ]);
                    }
                }
            }
            if m.steps.is_empty() {
                row(&[s("DO_NONE"), key[0].clone(), key[1].clone()]);
            }
            for st in &m.steps {
                let args: Vec<String> = st
                    .args
                    .iter()
                    .map(|a| match a {
                        ArgFact::Field(f) => format!("F{}", f.position),
                        ArgFact::Const(c) => format!("C{c}"),
                    })
                    .collect();
                row(&[
                    s("DO"),
                    key[0].clone(),
                    key[1].clone(),
                    st.ordinal.to_string(),
                    format!("{:#010X}", st.lens),
                    st.word.to_string(),
                    st.word_predicate.clone(),
                    st.arity.to_string(),
                    st.target.position.to_string(),
                    st.target.predicate.clone(),
                    args.join(","),
                ]);
            }
        }
        o
    }
}
