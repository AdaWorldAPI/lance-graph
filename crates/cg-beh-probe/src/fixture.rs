//! The bindings the gate reads.
//!
//! [`world`] is this crate's own class under a lab-local classid: three
//! methods over the polyfill lens, one with all of ON, WHEN and DO. Each
//! [`Variant`] moves exactly one part of it.
//!
//! [`foundry`] builds Foundry's `CanCook` class through the lab's own
//! `World::with_methods`; each [`FoundryVariant`] is `method::methods()` with
//! one part of `CanCook(0, 1)` moved. Nothing in `foundry-parity-lab` is
//! edited, and no DO step is added to a `CanCook` binding.

use foundry_parity_lab::bind::{self, World, VIEW_CLASS as FOUNDRY_VIEW_CLASS};
use foundry_parity_lab::method;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Cmp, Logic, Res, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::Extent;
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Guard, MethodBinding, MethodId, SchemaRev,
    SigRev, ValueDemand,
};
use lance_graph_r2il::polyfill::{
    polyfill_arities, position_of, AliasLanguage, PolyfillView, POLYFILL_LENS,
};

use crate::facts::{GuardConvention, Read};

/// The lab class (lab-local id; no OGAR concept is minted for it).
pub const CLASS: u32 = 0x7F0C_0001;
pub const VIEW_CLASS: ClassId = 0x7F0C;
pub const REV: SchemaRev = SchemaRev(1);
pub const FIELDS: [&str; 8] = ["a", "b", "k", "m", "s", "t", "u", "v"];

/// ON `a`, WHEN `a < 5 and b >= 0`, DO `m = IntSLess(a, 5)` then
/// `s = SUM(m, b)`.
pub const GUARDED: u32 = 1;
/// Two steps that do not read each other, so either order registers.
pub const INDEPENDENT: u32 = 2;
/// Never moved by any variant.
pub const UNRELATED: u32 = 3;
pub const METHODS: [u32; 3] = [GUARDED, INDEPENDENT, UNRELATED];

/// How the lab class is read.
pub const READ: Read = Read {
    schema_rev: REV,
    lens_rev: Some(REV),
    convention: GuardConvention::FieldPosition,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Base,
    /// `GUARDED` triggers on `b` instead of `a`.
    OnMoved,
    /// `GUARDED`'s guard reads `a < 6`.
    GuardMoved,
    /// `GUARDED`'s guard lists its two conditions in the other order.
    GuardRespelled,
    /// `INDEPENDENT`'s two steps in the other order.
    DoReversed,
    /// The predicates of positions 0 and 1 exchanged. Same field count.
    BasisSwapped,
}

/// The view of the lab class and of the polyfill lens.
pub struct TestView {
    class: Vec<FieldRef>,
    polyfill: PolyfillView,
}

impl TestView {
    pub fn new(swapped: bool) -> TestView {
        let mut class: Vec<FieldRef> = FIELDS
            .iter()
            .map(|l| FieldRef::new(format!("lab:beh/{l}"), (*l).to_string()))
            .collect();
        if swapped {
            class.swap(0, 1);
        }
        TestView {
            class,
            polyfill: PolyfillView::new(AliasLanguage::R2il),
        }
    }
}

impl ClassView for TestView {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == VIEW_CLASS {
            &self.class
        } else {
            &[]
        }
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        self.polyfill.lens_fields(lens)
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

pub struct TestWorld {
    pub store: BindStore,
    pub view: TestView,
}

fn step(target: u8, name: &str, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: AlgebraRef {
            lens: POLYFILL_LENS,
            word: position_of(name).expect("a polyfill alias"),
        },
        args,
    }
}

fn binding(
    id: u32,
    on: &[u8],
    reads: &[u8],
    when: Option<Guard>,
    effects: Vec<Effect>,
) -> MethodBinding {
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("lab{id}"),
        on: WideFieldMask::from_positions(on),
        when,
        reads: WideFieldMask::from_positions(reads),
        effects,
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// The three lab methods under `v`.
pub fn methods(v: Variant) -> Vec<MethodBinding> {
    let below = SetExpr::i32(0, Cmp::Lt, if v == Variant::GuardMoved { 6 } else { 5 });
    let non_negative = SetExpr::i32(1, Cmp::Ge, 0);
    let guard = if v == Variant::GuardRespelled {
        SetExpr::And(vec![non_negative, below])
    } else {
        SetExpr::And(vec![below, non_negative])
    };
    let mut independent = vec![
        step(5, "IntAdd", vec![Arg::Field(0), Arg::Field(1)]),
        step(6, "IntSub", vec![Arg::Field(0), Arg::Field(1)]),
    ];
    if v == Variant::DoReversed {
        independent.reverse();
    }
    vec![
        binding(
            GUARDED,
            if v == Variant::OnMoved { &[1] } else { &[0] },
            &[0, 1],
            Some(Guard::new(guard, Logic::TwoValued)),
            vec![
                step(3, "IntSLess", vec![Arg::Field(0), Arg::Const(5)]),
                step(4, "SUM", vec![Arg::Field(3), Arg::Field(1)]),
            ],
        ),
        binding(INDEPENDENT, &[0, 1], &[0, 1], None, independent),
        binding(
            UNRELATED,
            &[2],
            &[2],
            None,
            vec![step(7, "PopCount", vec![Arg::Field(2)])],
        ),
    ]
}

/// A store holding `methods` for the lab class, over the plain or the
/// swapped basis.
pub fn world_with(methods: Vec<MethodBinding>, swapped: bool) -> TestWorld {
    world_at(methods, swapped, REV)
}

/// [`world_with`] with the lab class bound at `schema_rev`. The polyfill lens
/// stays declared at [`REV`].
pub fn world_at(methods: Vec<MethodBinding>, swapped: bool, schema_rev: SchemaRev) -> TestWorld {
    let view = TestView::new(swapped);
    let mut store = BindStore::new();
    store
        .declare_algebra(POLYFILL_LENS, REV, Extent::DR64_CUBE, polyfill_arities())
        .expect("the polyfill lens declares");
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev,
                field_count: FIELDS.len(),
                extent: Extent::DR64_CUBE,
            },
            methods,
        )
        .expect("the lab class registers");
    TestWorld { store, view }
}

pub fn world(v: Variant) -> TestWorld {
    world_with(methods(v), v == Variant::BasisSwapped)
}

// ---------------------------------------------------------------- Foundry

/// The `CanCook` method the Foundry variants move: item 0, quantity 1.
pub const CAN_COOK: (usize, i32) = (0, 1);

/// How Foundry's class is read. No lens revision: `CanCook` has no DO step,
/// and a method that had one would be refused, not guessed at.
pub const FOUNDRY_READ: Read = Read {
    schema_rev: SchemaRev(1),
    lens_rev: None,
    convention: GuardConvention::FieldPosition,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoundryVariant {
    /// `method::methods()` as the lab registers it.
    Base,
    /// The last residual threshold of the guard raised by one. Same fields.
    GuardThreshold,
    /// ON loses its lowest position. Guard and reads unchanged.
    OnNarrowed,
}

fn raise_last_threshold(e: &SetExpr) -> SetExpr {
    let SetExpr::And(cs) = e else {
        panic!("a CanCook guard is a conjunction")
    };
    let mut cs = cs.clone();
    let last = cs.last_mut().expect("a CanCook guard has conditions");
    let SetExpr::Res(Res::I32 { lane, op, t }) = *last else {
        panic!("a CanCook guard ends in its staff residual")
    };
    *last = SetExpr::i32(lane, op, t + 1);
    SetExpr::And(cs)
}

/// Foundry's method list with `CanCook(0, 1)` moved as `v` says.
pub fn foundry_methods(v: FoundryVariant) -> Vec<MethodBinding> {
    let target = method::method_id(CAN_COOK.0, CAN_COOK.1);
    let mut all = method::methods();
    let m = all
        .iter_mut()
        .find(|m| m.id == target)
        .expect("CanCook(0, 1) is registered");
    match v {
        FoundryVariant::Base => {}
        FoundryVariant::GuardThreshold => {
            let g = m.when.as_ref().expect("CanCook is guarded");
            m.when = Some(Guard::new(raise_last_threshold(&g.expr), g.logic));
        }
        FoundryVariant::OnNarrowed => {
            let kept: Vec<u8> = (0..=u8::MAX).filter(|&p| m.on.has(p)).skip(1).collect();
            m.on = WideFieldMask::from_positions(&kept);
        }
    }
    all
}

pub fn foundry(v: FoundryVariant) -> World {
    World::with_methods(foundry_methods(v))
}

/// Every registered `CanCook` method id.
pub fn foundry_ids() -> Vec<u32> {
    method::methods().iter().map(|m| m.id.0).collect()
}

/// Foundry's basis with the predicates of two positions exchanged: the same
/// field count, another ordered basis.
pub struct SwappedBasis(Vec<FieldRef>);

impl SwappedBasis {
    pub fn of(world: &World, a: usize, b: usize) -> SwappedBasis {
        let mut fields = world.view.fields(FOUNDRY_VIEW_CLASS).to_vec();
        fields.swap(a, b);
        SwappedBasis(fields)
    }
}

impl ClassView for SwappedBasis {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == FOUNDRY_VIEW_CLASS {
            &self.0
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

pub const FOUNDRY_CLASS: u32 = bind::CLASS;
