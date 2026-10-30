//! DT64: the ClassId-addressed binding store.
//!
//! D-DRDT §1: "DT is the existing DTO binding store, addressed by ClassId:
//! `ClassId → class schema + method schema`". D-DT64-6 §1 (D4, D5, D6) and §3
//! give its rules; D-DRDT §4 lists what the contract did not yet carry, and
//! this crate closes those four gaps:
//!
//! 1. **Method identity independent of its name.** [`MethodId`] plus a
//!    [`SigRev`] signature revision. The source name
//!    (`MethodSig::name`, `ActionDef::predicate`) rides along as
//!    [`MethodBinding::name`] and is never the identity.
//! 2. **K3, conjunctive guards.** [`Guard`] is a
//!    [`set_algebra::SetExpr`](lance_graph_contract::set_algebra::SetExpr)
//!    under a declared [`Logic`]: the same description Quack's WHERE
//!    evaluates (D-DT64-6 §3, "WHEN is a Quack WHERE into a guard mask G").
//!    The store keeps it; it evaluates nothing.
//! 3. **Effect order.** [`MethodBinding::effects`] is an ordered list, so two
//!    methods with the same read and write sets and different order stay
//!    distinct (the CR-H0 negative mold, D-DT64-6 §3).
//! 4. **One lookup.** [`BindStore::resolve`] joins the class schema (read
//!    through a [`ClassView`], never copied) with the method schema for one
//!    classid at one schema revision.
//!
//! # What is stored, and what is not
//!
//! Bindings and their version log only (the D-DT64-6 §4 reading of the Q0
//! contract): per classid, its [`ClassSchema`] and [`MethodBinding`]s; and the
//! append-only route log from a DR position to a classid. No business value,
//! no field label (labels come from the [`ClassView`] at resolution), no
//! evaluator, no persistence path.
//!
//! # Immutability (D5) and replay
//!
//! A classid's behaviour never changes once registered.
//! [`BindStore::register`] is idempotent for an identical binding and refuses
//! any other with [`Refusal::BehaviourImmutable`]. An override is a new
//! derived classid ([`BindStore::register_derived`]) plus a re-route
//! ([`BindStore::route`]). Only routing is versioned: [`BindStore::routed_at`]
//! answers which classid a position routed to at a [`RouteVersion`], and that
//! classid's behaviour is the same today as then.
//!
//! # UNKNOWN is never empty
//!
//! An unregistered classid, an unrouted position, an unknown method, a
//! dropped guard, a mismatched revision: each is a named [`Refusal`], never an
//! empty answer that reads as "no behaviour" (D-SAG-6 F06).
//!
//! # Classids
//!
//! A bound class is a `u32`, the key [`ActionDef::object_class`] and
//! [`ClassMethods::classid`] already use. A classid is a codebook lens: its
//! field basis is the [`ClassView`] of its concept half ([`view_class`], the
//! codebook's own `classid_concept` split), labels are resolved late through
//! that view, and its entries are the [`MethodId`]s inside it. A codebook item
//! never gets a classid of its own.
//!
//! # Agnostic algebra, switched by the classid
//!
//! Addresses and algebra methods are agnostic. The algebra catalogue
//! ([`AlgebraEntry`], [`BindStore::declare_algebra`]) is classid-free global
//! behaviour (D-DT64-6 D6). A classid switches labels (its [`ClassView`]) and
//! polyfill (its bindings' [`Effect`]s, each an [`AlgebraRef`] with ordered
//! operands) over them. Two consumers reuse one entry by referring to the
//! same [`AlgebraId`]; [`BindStore::users_of`] lists them.
//!
//! [`ActionDef::object_class`]: lance_graph_contract::action::ActionDef::object_class
//! [`ClassMethods::classid`]: lance_graph_contract::codegen_manifest::ClassMethods::classid

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ogar_codebook::classid_concept;
use lance_graph_contract::ontology::FieldRef;
use lance_graph_contract::property::{PropertySchema, PropertySpec};
use lance_graph_contract::set_algebra::{normalize, normalize_kleene, Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};

/// A method's identity within its class, independent of its source name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MethodId(pub u32);

/// A method's signature revision. A reference made against one revision is
/// refused under another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SigRev(pub u32);

/// A class's schema revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaRev(pub u32);

/// A position in the append-only route log. `RouteVersion(0)` is the empty
/// log; each route or retract advances it by one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RouteVersion(pub u64);

/// A classid-free algebra method: global behaviour (D-DT64-6 D6, "global
/// behaviour needs no classid and lives in lance-graph"). Addresses and
/// algebra methods are agnostic; a classid's binding is what switches labels
/// and polyfill over them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AlgebraId(pub u32);

/// One entry of the algebra catalogue: the polyfill (D4: over mask-risc /
/// ndarray) named by reference, how many operands it takes, and its signature
/// revision. Immutable once declared; versioned in the store directly (D6).
/// The store never resolves or runs the polyfill.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlgebraEntry {
    pub id: AlgebraId,
    pub sig_rev: SigRev,
    pub polyfill: String,
    pub inputs: u8,
}

/// What a class binds an effect to: an algebra entry at the signature
/// revision it was bound against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlgebraRef {
    pub id: AlgebraId,
    pub sig_rev: SigRev,
}

impl AlgebraRef {
    /// The faithful reference to `entry`.
    pub fn of(entry: &AlgebraEntry) -> AlgebraRef {
        AlgebraRef {
            id: entry.id,
            sig_rev: entry.sig_rev,
        }
    }
}

/// One DO step: which field position of the class it writes, the algebra
/// entry that writes it, and what it reads. A method's effects are an ORDERED
/// list.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Effect {
    /// Field position in the class basis (the `WideFieldMask` bit).
    pub target: u8,
    pub algebra: AlgebraRef,
    /// What the step reads, in order. Order is load-bearing (a `Sub` or a
    /// keyed count), so two steps with the same entry and different arguments
    /// are different steps.
    pub args: Vec<Arg>,
}

/// One input of an effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arg {
    /// A field position of the bound class (checked against its basis).
    Field(u8),
    /// A constant parameter: a key column index, a lane.
    Const(i64),
}

/// WHEN: a guard over the population, stated as the set description Quack's
/// WHERE evaluates, under a declared logic. Conjunction is
/// [`SetExpr::And`]; a [`Logic::Kleene`] guard is three-valued, and a row
/// where it is UNKNOWN does not fire.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Guard {
    pub expr: SetExpr,
    pub logic: Logic,
}

impl Guard {
    pub fn new(expr: SetExpr, logic: Logic) -> Guard {
        Guard { expr, logic }
    }

    /// What a router carries to prove it kept this guard: the expression's
    /// normal form under the guard's own logic. Two guards with equal
    /// witnesses fire on the same rows. Under [`Logic::Kleene`] the witness is
    /// the TRUE-rows form ([`normalize_kleene`]), which is all a guard is read
    /// as.
    pub fn witness(&self) -> GuardWitness {
        let normal = match self.logic {
            Logic::TwoValued => normalize(&self.expr),
            Logic::Kleene => normalize_kleene(&self.expr).into_where_true(),
        };
        GuardWitness {
            normal,
            logic: self.logic,
        }
    }
}

/// See [`Guard::witness`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GuardWitness {
    normal: SetExpr,
    logic: Logic,
}

/// Whether a method needs field values, or only the map (D-DRAV-1 §2: map and
/// value lifetimes are separate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueDemand {
    MapOnly,
    Values,
}

/// One method's binding: identity, ON × WHEN × DO, and what it needs from the
/// population. Several bindings may share a cell; all are kept.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MethodBinding {
    pub id: MethodId,
    pub sig_rev: SigRev,
    /// The source name (`MethodSig::name`, `ActionDef::predicate`). Carried,
    /// never used as identity.
    pub name: String,
    /// ON: the field positions whose change triggers the method.
    pub on: WideFieldMask,
    /// WHEN: `None` for an unguarded method.
    pub when: Option<Guard>,
    pub reads: WideFieldMask,
    /// DO, in order.
    pub effects: Vec<Effect>,
    pub population: Population,
    pub truth: Truth,
    pub demand: ValueDemand,
}

impl MethodBinding {
    /// The field positions the effects write, as an unordered set. Equal
    /// write sets do not make two bindings equal: the order is in
    /// [`MethodBinding::effects`].
    pub fn writes(&self) -> WideFieldMask {
        let targets: Vec<u8> = self.effects.iter().map(|e| e.target).collect();
        WideFieldMask::from_positions(&targets)
    }
}

/// A class's schema as bound: the schema revision and the field count the
/// methods were bound against. The field basis itself is the [`ClassView`]
/// class [`view_class`] of the classid, never declared here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClassSchema {
    pub schema_rev: SchemaRev,
    pub field_count: usize,
}

/// The [`ClassView`] class a classid is read through: its concept half, by
/// the codebook's own split ([`classid_concept`]). The classid is the lens;
/// this crate does no bit arithmetic of its own.
pub fn view_class(class: u32) -> ClassId {
    classid_concept(class)
}

/// The widest basis a [`WideFieldMask`] addresses (positions are `u8`).
pub const MAX_FIELDS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClassEntry {
    schema: ClassSchema,
    methods: Vec<MethodBinding>,
    derived_from: Option<u32>,
}

/// A reference a router (DR #4) carries to a method: the class, the method
/// identity, the signature revision it was made against, and the witness of
/// the guard it kept.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MethodRef {
    pub class: u32,
    pub method: MethodId,
    pub sig_rev: SigRev,
    pub guard: Option<GuardWitness>,
}

impl MethodRef {
    /// The faithful reference to `binding` in `class`.
    pub fn of(class: u32, binding: &MethodBinding) -> MethodRef {
        MethodRef {
            class,
            method: binding.id,
            sig_rev: binding.sig_rev,
            guard: binding.when.as_ref().map(Guard::witness),
        }
    }
}

/// Why the store refused. Every arm is a named UNKNOWN or conflict; none is an
/// empty answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The classid has no binding.
    UnknownClass(u32),
    /// The classid is bound to other behaviour (D5). Override with a derived
    /// classid and a re-route.
    BehaviourImmutable(u32),
    /// Two bindings in one class share a MethodId.
    DuplicateMethodId { class: u32, method: MethodId },
    /// A binding names a field position at or past the class's field count.
    FieldOutOfBasis {
        class: u32,
        method: MethodId,
        position: u8,
    },
    /// The declared field count exceeds [`MAX_FIELDS`].
    BasisTooWide { class: u32, field_count: usize },
    /// Asked at another schema revision than the one bound.
    SchemaRevisionMismatch {
        class: u32,
        bound: SchemaRev,
        asked: SchemaRev,
    },
    /// The [`ClassView`] no longer has the field count the methods were bound
    /// against.
    FieldBasisChanged {
        class: u32,
        bound: usize,
        view: usize,
    },
    /// The class has no method with this identity.
    UnknownMethod { class: u32, method: MethodId },
    SignatureRevisionMismatch {
        class: u32,
        method: MethodId,
        bound: SigRev,
        asked: SigRev,
    },
    /// The method is guarded and the reference carries no guard witness.
    GuardDropped { class: u32, method: MethodId },
    /// The reference carries a guard the method does not have.
    GuardAdded { class: u32, method: MethodId },
    /// The reference carries another guard than the bound one.
    GuardMismatch { class: u32, method: MethodId },
    /// An effect names an algebra entry the catalogue does not hold.
    UnknownAlgebra {
        class: u32,
        method: MethodId,
        algebra: AlgebraId,
    },
    /// An effect was bound against another revision of the entry.
    AlgebraRevisionMismatch {
        class: u32,
        method: MethodId,
        algebra: AlgebraId,
        bound: SigRev,
        asked: SigRev,
    },
    /// An effect passes another number of operands than the entry takes.
    ArityMismatch {
        class: u32,
        method: MethodId,
        algebra: AlgebraId,
        expected: u8,
        got: usize,
    },
    /// The algebra entry is already declared with other content.
    AlgebraImmutable(AlgebraId),
    /// No classid was routed to the position at that version.
    Unrouted {
        position: u16,
        version: RouteVersion,
    },
    /// The version is past the end of the route log.
    FutureVersion {
        asked: RouteVersion,
        current: RouteVersion,
    },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Refusal {}

/// The result of one lookup: the class schema as bound, its fields as the
/// [`ClassView`] holds them now (borrowed), and its methods.
#[derive(Debug, Clone, Copy)]
pub struct Resolved<'s, 'v> {
    pub class: u32,
    pub schema: &'s ClassSchema,
    pub fields: &'v [FieldRef],
    pub methods: &'s [MethodBinding],
    pub derived_from: Option<u32>,
}

/// One row of the fieldview projection (the Redmine ERB fieldview pattern,
/// OGAR `ogar-render-askama::field_view::FieldView` minus the value): the
/// field's mask position is its layout address, and its label and predicate
/// are borrowed late from the [`ClassView`]. The store holds no values; a
/// renderer pairs each row with the consumer's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldRow<'v> {
    pub position: u8,
    pub label: &'v str,
    pub predicate: &'v str,
}

/// A fieldview row with its meaning (see [`Resolved::field_meaning`]).
#[derive(Debug, Clone, Copy)]
pub struct FieldMeaning<'v, 'p> {
    pub row: FieldRow<'v>,
    /// The field's semantics by predicate; `None` is UNKNOWN.
    pub spec: Option<&'p PropertySpec>,
}

impl<'s, 'v> Resolved<'s, 'v> {
    /// Every method whose ON includes `field`, in registration order. Several
    /// are normal and all are returned.
    pub fn methods_on(&self, field: u8) -> impl Iterator<Item = &'s MethodBinding> {
        self.methods.iter().filter(move |m| m.on.has(field))
    }

    /// The fieldview: the fields `mask` selects, ascending by position, with
    /// their labels. A position past the class's basis selects nothing.
    pub fn field_view<'m>(&self, mask: &'m WideFieldMask) -> impl Iterator<Item = FieldRow<'v>> + 'm
    where
        'v: 'm,
    {
        let fields = self.fields;
        mask_positions(mask).filter_map(move |p| {
            fields.get(usize::from(p)).map(|f| FieldRow {
                position: p,
                label: f.label.as_str(),
                predicate: f.predicate_iri.as_str(),
            })
        })
    }

    /// The fieldview with meaning: each selected row joined by predicate to
    /// its [`PropertySpec`] (obligation, semantic type, marking). `spec` is
    /// `None` when the schema has no entry: UNKNOWN meaning, never a default.
    pub fn field_meaning<'m, 'p>(
        &self,
        mask: &'m WideFieldMask,
        schema: &'p PropertySchema,
    ) -> impl Iterator<Item = FieldMeaning<'v, 'p>> + 'm
    where
        'v: 'm,
        'p: 'm,
    {
        self.field_view(mask).map(move |row| FieldMeaning {
            row,
            spec: schema.get(row.predicate),
        })
    }

    /// A mask from meaning: the lens's fields whose spec satisfies `pred`
    /// (a risk mask is `|s| s.marking >= Marking::Pii`, a presence obligation
    /// is `|s| s.kind == PropertyKind::Required`). A field with no spec is
    /// never selected, whatever `pred` would say about a default.
    pub fn mask_where(
        &self,
        schema: &PropertySchema,
        pred: impl Fn(&PropertySpec) -> bool,
    ) -> WideFieldMask {
        self.positions(|f| schema.get(&f.predicate_iri).is_some_and(&pred))
    }

    /// The lens's fields the schema says nothing about: UNKNOWN meaning,
    /// surfaced rather than defaulted.
    pub fn unspecified(&self, schema: &PropertySchema) -> WideFieldMask {
        self.positions(|f| schema.get(&f.predicate_iri).is_none())
    }

    fn positions(&self, keep: impl Fn(&FieldRef) -> bool) -> WideFieldMask {
        let picked: Vec<u8> = self
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| keep(f))
            .filter_map(|(i, _)| u8::try_from(i).ok())
            .collect();
        WideFieldMask::from_positions(&picked)
    }

    /// The action addresses the class exposes: its method ordinals, in
    /// registration order (fieldview charter T2: behaviour by address only).
    pub fn action_ordinals(&self) -> impl Iterator<Item = MethodId> + 's {
        self.methods.iter().map(|m| m.id)
    }
}

/// The binding store. Building it takes `&mut self`; every read is `&self`.
#[derive(Debug, Clone, Default)]
pub struct BindStore {
    classes: BTreeMap<u32, ClassEntry>,
    /// The classid-free algebra catalogue (D6).
    algebra: BTreeMap<AlgebraId, AlgebraEntry>,
    /// Per position, the route events in version order: `(version, class)`,
    /// `None` for a retract.
    routes: BTreeMap<u16, Vec<(RouteVersion, Option<u32>)>>,
    version: RouteVersion,
}

impl BindStore {
    pub fn new() -> BindStore {
        BindStore::default()
    }

    /// Bind `class`. Idempotent for an identical binding; any other binding of
    /// an already bound class is refused (D5).
    pub fn register(
        &mut self,
        class: u32,
        schema: ClassSchema,
        methods: Vec<MethodBinding>,
    ) -> Result<(), Refusal> {
        self.insert(class, schema, methods, None)
    }

    /// Bind `class` as an override of the bound class `base` (D-DT64-6 §3).
    /// `base` keeps its behaviour; route positions to `class` to use it.
    pub fn register_derived(
        &mut self,
        base: u32,
        class: u32,
        schema: ClassSchema,
        methods: Vec<MethodBinding>,
    ) -> Result<(), Refusal> {
        if !self.classes.contains_key(&base) {
            return Err(Refusal::UnknownClass(base));
        }
        if base == class {
            return Err(Refusal::BehaviourImmutable(class));
        }
        self.insert(class, schema, methods, Some(base))
    }

    fn insert(
        &mut self,
        class: u32,
        schema: ClassSchema,
        methods: Vec<MethodBinding>,
        derived_from: Option<u32>,
    ) -> Result<(), Refusal> {
        validate(class, &schema, &methods)?;
        for m in &methods {
            for e in &m.effects {
                let (method, algebra) = (m.id, e.algebra.id);
                let entry = self.algebra.get(&algebra).ok_or(Refusal::UnknownAlgebra {
                    class,
                    method,
                    algebra,
                })?;
                if entry.sig_rev != e.algebra.sig_rev {
                    return Err(Refusal::AlgebraRevisionMismatch {
                        class,
                        method,
                        algebra,
                        bound: entry.sig_rev,
                        asked: e.algebra.sig_rev,
                    });
                }
                if usize::from(entry.inputs) != e.args.len() {
                    return Err(Refusal::ArityMismatch {
                        class,
                        method,
                        algebra,
                        expected: entry.inputs,
                        got: e.args.len(),
                    });
                }
            }
        }
        let entry = ClassEntry {
            schema,
            methods,
            derived_from,
        };
        match self.classes.get(&class) {
            Some(bound) if *bound == entry => Ok(()),
            Some(_) => Err(Refusal::BehaviourImmutable(class)),
            None => {
                self.classes.insert(class, entry);
                Ok(())
            }
        }
    }

    /// Declare a classid-free algebra entry. Idempotent for an identical
    /// entry; any other content under a declared id is refused.
    pub fn declare_algebra(&mut self, entry: AlgebraEntry) -> Result<(), Refusal> {
        match self.algebra.get(&entry.id) {
            Some(bound) if *bound == entry => Ok(()),
            Some(_) => Err(Refusal::AlgebraImmutable(entry.id)),
            None => {
                self.algebra.insert(entry.id, entry);
                Ok(())
            }
        }
    }

    /// The algebra entry `id`, if declared.
    pub fn algebra(&self, id: AlgebraId) -> Option<&AlgebraEntry> {
        self.algebra.get(&id)
    }

    /// Every `(classid, method)` whose DO uses algebra entry `id`: which
    /// classids switch over it.
    pub fn users_of(&self, id: AlgebraId) -> Vec<(u32, MethodId)> {
        let mut out = Vec::new();
        for (&class, entry) in &self.classes {
            for m in &entry.methods {
                if m.effects.iter().any(|e| e.algebra.id == id) {
                    out.push((class, m.id));
                }
            }
        }
        out
    }

    /// Number of bound classes.
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// The current end of the route log.
    pub fn current_version(&self) -> RouteVersion {
        self.version
    }

    /// Route `position` to the bound `class`; returns the new version.
    pub fn route(&mut self, position: u16, class: u32) -> Result<RouteVersion, Refusal> {
        if !self.classes.contains_key(&class) {
            return Err(Refusal::UnknownClass(class));
        }
        Ok(self.push_route(position, Some(class)))
    }

    /// Remove `position`'s route; returns the new version. The classid it
    /// routed to keeps its behaviour.
    pub fn retract(&mut self, position: u16) -> RouteVersion {
        self.push_route(position, None)
    }

    fn push_route(&mut self, position: u16, class: Option<u32>) -> RouteVersion {
        self.version = RouteVersion(self.version.0 + 1);
        self.routes
            .entry(position)
            .or_default()
            .push((self.version, class));
        self.version
    }

    /// The classid `position` routed to at `version`, `None` if it was
    /// unrouted then.
    pub fn routed_at(&self, version: RouteVersion, position: u16) -> Result<Option<u32>, Refusal> {
        if version > self.version {
            return Err(Refusal::FutureVersion {
                asked: version,
                current: self.version,
            });
        }
        let Some(events) = self.routes.get(&position) else {
            return Ok(None);
        };
        // Events are in version order; take the last one at or before `version`.
        let seen = events.partition_point(|(v, _)| *v <= version);
        Ok(seen.checked_sub(1).and_then(|i| events[i].1))
    }

    /// The one lookup: `(classid, schema revision) → { fields, methods }`.
    pub fn resolve<'s, 'v, V: ClassView + ?Sized>(
        &'s self,
        view: &'v V,
        class: u32,
        schema_rev: SchemaRev,
    ) -> Result<Resolved<'s, 'v>, Refusal> {
        let entry = self
            .classes
            .get(&class)
            .ok_or(Refusal::UnknownClass(class))?;
        if entry.schema.schema_rev != schema_rev {
            return Err(Refusal::SchemaRevisionMismatch {
                class,
                bound: entry.schema.schema_rev,
                asked: schema_rev,
            });
        }
        let fields = view.fields(view_class(class));
        if fields.len() != entry.schema.field_count {
            return Err(Refusal::FieldBasisChanged {
                class,
                bound: entry.schema.field_count,
                view: fields.len(),
            });
        }
        Ok(Resolved {
            class,
            schema: &entry.schema,
            fields,
            methods: &entry.methods,
            derived_from: entry.derived_from,
        })
    }

    /// The lookup by DR position: the classid routed there at `version`, then
    /// [`BindStore::resolve`].
    pub fn resolve_at<'s, 'v, V: ClassView + ?Sized>(
        &'s self,
        view: &'v V,
        version: RouteVersion,
        position: u16,
        schema_rev: SchemaRev,
    ) -> Result<Resolved<'s, 'v>, Refusal> {
        let class = self
            .routed_at(version, position)?
            .ok_or(Refusal::Unrouted { position, version })?;
        self.resolve(view, class, schema_rev)
    }

    /// Check a router's method reference against the binding and return the
    /// binding (D-DRDT §5): the method must exist, at the referenced signature
    /// revision, with exactly the guard the reference witnesses.
    pub fn admit(&self, r: &MethodRef) -> Result<&MethodBinding, Refusal> {
        let entry = self
            .classes
            .get(&r.class)
            .ok_or(Refusal::UnknownClass(r.class))?;
        let (class, method) = (r.class, r.method);
        let binding = entry
            .methods
            .iter()
            .find(|m| m.id == method)
            .ok_or(Refusal::UnknownMethod { class, method })?;
        if binding.sig_rev != r.sig_rev {
            return Err(Refusal::SignatureRevisionMismatch {
                class,
                method,
                bound: binding.sig_rev,
                asked: r.sig_rev,
            });
        }
        match (&binding.when, &r.guard) {
            (None, None) => Ok(binding),
            (Some(_), None) => Err(Refusal::GuardDropped { class, method }),
            (None, Some(_)) => Err(Refusal::GuardAdded { class, method }),
            (Some(g), Some(w)) if g.witness() == *w => Ok(binding),
            (Some(_), Some(_)) => Err(Refusal::GuardMismatch { class, method }),
        }
    }
}

/// Registration checks: the basis fits a [`WideFieldMask`], MethodIds are
/// unique in the class, and every field position a binding names is inside
/// the declared basis.
fn validate(class: u32, schema: &ClassSchema, methods: &[MethodBinding]) -> Result<(), Refusal> {
    if schema.field_count > MAX_FIELDS {
        return Err(Refusal::BasisTooWide {
            class,
            field_count: schema.field_count,
        });
    }
    for (i, m) in methods.iter().enumerate() {
        if methods[..i].iter().any(|p| p.id == m.id) {
            return Err(Refusal::DuplicateMethodId {
                class,
                method: m.id,
            });
        }
    }
    let inside = |p: u8| usize::from(p) < schema.field_count;
    for m in methods {
        let positions = mask_positions(&m.on)
            .chain(mask_positions(&m.reads))
            .chain(m.effects.iter().map(|e| e.target))
            .chain(m.effects.iter().flat_map(|e| {
                e.args.iter().filter_map(|a| match a {
                    Arg::Field(p) => Some(*p),
                    Arg::Const(_) => None,
                })
            }));
        for position in positions {
            if !inside(position) {
                return Err(Refusal::FieldOutOfBasis {
                    class,
                    method: m.id,
                    position,
                });
            }
        }
    }
    Ok(())
}

/// The set positions of a mask, ascending.
fn mask_positions(mask: &WideFieldMask) -> impl Iterator<Item = u8> + '_ {
    (0..=u8::MAX).filter(move |&p| mask.has(p))
}
