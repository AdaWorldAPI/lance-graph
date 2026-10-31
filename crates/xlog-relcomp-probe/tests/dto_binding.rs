//! The bag-multiplicity operation bound through the production bind store
//! (`lance-graph-dt64`): which identities, which property mappings, which
//! method by reference, and the expected values.
//!
//! The stages stay separate and in order:
//!
//! 1. **DR routes.** Lab ids are DR positions routed to the XLOG lens classid
//!    in the bind store's route log, under a sealed `dt64::dr::FrameStore`.
//! 2. **DT64 binds meaning.** The lens's fields (r, s and the three counts)
//!    are labelled by its `ClassView` and selected by a `WideFieldMask`. The
//!    counting method is a `MethodBinding` of `Population::Bag`, reached by a
//!    `MethodRef` and checked by `admit`. Its steps name polyfill words by
//!    position (mask-risc's keyed count twice, then a checked i64 multiply);
//!    no word has a classid.
//! 3. **Execution** has no production home. The lab register
//!    (`dr-math-binding-probe::bind_dr64`) refuses a Bag method
//!    (`Refusal::BagNeedsMultiplicity`): that is the failing half.
//!
//! The classids are lab values (`0x7F01_0001` for the XLOG lens,
//! `0x7F0A_0001` for the polyfill lens). Production classids are OGAR mints.

use dr_math_binding_probe as lab;
use lance_graph_contract::canonical_node::{NodeGuid, TailVariant};
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, FrameRefusal, FrameRevisions, FrameStore, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, MethodBinding, MethodId, MethodRef,
    Refusal, RouteVersion, SchemaRev, SigRev, ValueDemand, view_class,
};
use xlog_logic::parse_program;
use xlog_relcomp_probe::{AddressMap, SLOTS, facts, register};

const MULT: &str = "pred r(u32, u32).
pred s(u32, u32).
pred q(u32).
r(1, 2). r(5, 2). r(1, 3). r(9, 7).
s(2, 4). s(2, 6). s(3, 4). s(7, 8).
q(Y) :- r(X, Y), s(Y, Z).
?- q(Y).
";

/// The XLOG lens (lab). Its concept half is the `ClassView` class.
const XLOG: u32 = 0x7F01_0001;
const XLOG_VIEW: ClassId = 0x7F01;
/// The polyfill vocabulary is one lens (lab); its words are its fields.
const POLYFILL: u32 = 0x7F0A_0001;
const POLYFILL_VIEW: ClassId = 0x7F0A;

/// The lens's fields: the relations R and S, then the two keyed counts and
/// their product.
const XLOG_FIELDS: [&str; 5] = ["r", "s", "cnt_r", "cnt_s", "count"];
/// The polyfill words used here, by position. mask-risc folds (the keyed
/// count); `ndarray::simd` computes what mask-risc lacks (the i64 multiply).
const WORDS: [&str; 2] = [
    "lance_graph_mask_risc::Terminal::GroupReduce(GroupKey::Lane, GroupFold::Count)",
    "ndarray::simd::checked_binary_i64(I64Op::Mul)",
];
const ARITY: [u8; 2] = [2, 2];
const GROUP_COUNT: AlgebraRef = AlgebraRef {
    lens: POLYFILL,
    word: 0,
};
const MUL_I64: AlgebraRef = AlgebraRef {
    lens: POLYFILL,
    word: 1,
};
const COUNT: MethodId = MethodId(1);

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };

/// Both lenses' fields, labelled late. Nothing here is stored in the bind
/// store.
struct Views {
    xlog: Vec<FieldRef>,
    polyfill: Vec<FieldRef>,
}

impl Views {
    fn new() -> Self {
        let refs = |labels: &[&str], ns: &str| {
            labels
                .iter()
                .map(|l| FieldRef::new(format!("lab:{ns}/{l}"), (*l).to_string()))
                .collect()
        };
        Self {
            xlog: refs(&XLOG_FIELDS, "xlog"),
            polyfill: refs(&WORDS, "polyfill"),
        }
    }
}

impl ClassView for Views {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            XLOG_VIEW => &self.xlog,
            POLYFILL_VIEW => &self.polyfill,
            _ => &[],
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// `q(Y) :- r(X,Y), s(Y,Z)` counted per Y: R keyed by its y column (1), S by
/// its y column (0), then the product. Bag: one count per derivation.
fn xlog_count() -> MethodBinding {
    MethodBinding {
        id: COUNT,
        sig_rev: SigRev(1),
        name: "count q(Y)".to_string(),
        on: WideFieldMask::from_positions(&[0, 1]),
        when: None,
        reads: WideFieldMask::from_positions(&[0, 1]),
        effects: vec![
            Effect {
                target: 2,
                algebra: GROUP_COUNT,
                args: vec![Arg::Field(0), Arg::Const(1)],
            },
            Effect {
                target: 3,
                algebra: GROUP_COUNT,
                args: vec![Arg::Field(1), Arg::Const(0)],
            },
            Effect {
                target: 4,
                algebra: MUL_I64,
                args: vec![Arg::Field(2), Arg::Field(3)],
            },
        ],
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn bind_store() -> BindStore {
    let mut s = BindStore::new();
    s.declare_algebra(POLYFILL, SchemaRev(1), Extent::DR64_CUBE, ARITY.to_vec())
        .unwrap();
    let schema = ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: XLOG_FIELDS.len(),
        extent: Extent::DR64_CUBE,
    };
    s.register(XLOG, schema, vec![xlog_count()]).unwrap();
    s
}

/// A lab id on the answer axis, as a `NodeGuid` reference (V3 tail).
fn guid(id: u32) -> NodeGuid {
    NodeGuid::mint_for(TailVariant::V3, XLOG, 0, 0, 0, 0, 11, id)
}

type Tuples = Vec<(u32, u32)>;

fn fixture() -> (Tuples, Tuples) {
    let p = parse_program(MULT).unwrap();
    (facts(&p, "r").unwrap(), facts(&p, "s").unwrap())
}

/// The answer axis: every Y the join can reach, in id order.
fn answer_ids(r: &[(u32, u32)], s: &[(u32, u32)]) -> Vec<u16> {
    let mut ys: Vec<u16> = r
        .iter()
        .map(|t| t.1)
        .chain(s.iter().map(|t| t.0))
        .map(|y| u16::try_from(y).unwrap())
        .collect();
    ys.sort_unstable();
    ys.dedup();
    ys
}

/// DR's output: the answer positions routed to the XLOG lens at one route
/// version, under a sealed frame.
struct Routed {
    positions: Vec<u16>,
    version: RouteVersion,
    guards: Guards,
}

fn route(store: &mut BindStore, frames: &FrameStore, positions: Vec<u16>) -> Routed {
    let mut version = store.current_version();
    for &p in &positions {
        version = store.route(p, XLOG).unwrap();
    }
    Routed {
        positions,
        version,
        guards: Guards {
            frame: frames.seal(),
            focus: FOCUS,
            scope: SCOPE,
        },
    }
}

/// What DT64 hands back for the operation: identities by reference, the
/// selected property mappings, the admitted method, the expected values.
struct Binding<'s> {
    answer_identities: Vec<NodeGuid>,
    selected: WideFieldMask,
    method: &'s MethodBinding,
    expected: Vec<(NodeGuid, u64)>,
}

#[derive(Debug, PartialEq, Eq)]
enum BindRefusal {
    /// DT64 never binds against a frame that is no longer current.
    Frame(FrameRefusal),
    /// The bind store refused a lookup.
    Store(Refusal),
}

fn bind<'s>(
    store: &'s BindStore,
    frames: &FrameStore,
    views: &Views,
    routed: &Routed,
) -> Result<Binding<'s>, BindRefusal> {
    frames
        .check(&routed.guards.frame)
        .map_err(BindRefusal::Frame)?;
    for &p in &routed.positions {
        let r = store
            .resolve_at(views, routed.version, p, SchemaRev(1))
            .map_err(BindRefusal::Store)?;
        assert_eq!(r.class, XLOG);
    }
    let method = store
        .admit(&MethodRef::of(XLOG, &xlog_count()))
        .map_err(BindRefusal::Store)?;
    Ok(Binding {
        answer_identities: routed
            .positions
            .iter()
            .map(|&p| guid(u32::from(p)))
            .collect(),
        selected: WideFieldMask::from_positions(&[0, 1]),
        method,
        // Per-Y derivation counts; see tests/litho.rs for the two oracles.
        expected: vec![(guid(2), 4), (guid(3), 1), (guid(7), 1)],
    })
}

fn revisions(relation_map: u64) -> FrameRevisions {
    FrameRevisions {
        source: 1,
        relation_map,
        coordinates: 1,
    }
}

#[test]
fn the_binding_resolves_through_the_bind_store() {
    let (r, s) = fixture();
    let views = Views::new();
    let mut store = bind_store();
    let frames = FrameStore::new(revisions(1));
    let routed = route(&mut store, &frames, answer_ids(&r, &s));
    let b = bind(&store, &frames, &views, &routed).unwrap();

    // Identities: references that decode back to the lab ids, in id order.
    let ids: Vec<(u32, u16)> = b
        .answer_identities
        .iter()
        .map(|g| (g.classid(), g.identity_v2()))
        .collect();
    assert_eq!(ids, vec![(XLOG, 2), (XLOG, 3), (XLOG, 7)]);

    // Property mappings: R and S, labelled by the lens's ClassView.
    let resolved = store.resolve(&views, XLOG, SchemaRev(1)).unwrap();
    let rows: Vec<(u8, &str)> = resolved
        .field_view(&b.selected)
        .map(|r| (r.position, r.label))
        .collect();
    assert_eq!(rows, vec![(0, "r"), (1, "s")]);
    assert_eq!(view_class(XLOG), XLOG_VIEW);

    // Method: admitted by reference; Bag; two words of the polyfill lens.
    assert_eq!(b.method.population, Population::Bag);
    assert_eq!(
        store.words_used(XLOG, POLYFILL),
        Ok(WideFieldMask::from_positions(&[0, 1]))
    );
    assert_eq!(store.users_of(GROUP_COUNT), vec![(XLOG, COUNT)]);
    let words = store
        .resolve_algebra(&views, POLYFILL, SchemaRev(1))
        .unwrap();
    let labels: Vec<&str> = words
        .field_view(&WideFieldMask::from_positions(&[0, 1]))
        .map(|r| r.label)
        .collect();
    assert_eq!(labels, WORDS.to_vec());

    // Expected values, keyed by identity.
    assert_eq!(b.expected.len(), b.answer_identities.len());
    assert!(
        b.expected
            .iter()
            .zip(&b.answer_identities)
            .all(|((g, _), a)| g == a)
    );
}

#[test]
fn dt64_does_not_bind_against_a_stale_frame() {
    let (r, s) = fixture();
    let views = Views::new();
    let mut store = bind_store();
    let mut frames = FrameStore::new(revisions(1));
    let routed = route(&mut store, &frames, answer_ids(&r, &s));
    frames.advance(revisions(2));
    assert_eq!(
        bind(&store, &frames, &views, &routed).err(),
        Some(BindRefusal::Frame(FrameRefusal::MapChanged))
    );
}

/// Routing is versioned; behaviour is not. A retracted answer position is
/// unrouted from the new version on, and the old version still resolves.
#[test]
fn a_retracted_position_is_unrouted_and_the_old_version_replays() {
    let (r, s) = fixture();
    let views = Views::new();
    let mut store = bind_store();
    let frames = FrameStore::new(revisions(1));
    let mut routed = route(&mut store, &frames, answer_ids(&r, &s));
    let old = routed.version;
    let new = store.retract(7);
    routed.version = new;
    assert_eq!(
        bind(&store, &frames, &views, &routed).err(),
        Some(BindRefusal::Store(Refusal::Unrouted {
            position: 7,
            version: new
        }))
    );
    assert_eq!(
        store
            .resolve_at(&views, old, 7, SchemaRev(1))
            .map(|r| r.class),
        Ok(XLOG)
    );
}

/// The failing half: production has no executor, and the lab register
/// refuses the admitted Bag method, so the expected counts cannot be produced
/// from the register yet.
#[test]
fn executing_the_admitted_bag_method_is_refused_in_the_lab_register() {
    let (r, s) = fixture();
    let views = Views::new();
    let mut store = bind_store();
    let frames = FrameStore::new(revisions(1));
    let routed = route(&mut store, &frames, answer_ids(&r, &s));
    let b = bind(&store, &frames, &views, &routed).unwrap();

    // The lab register has its own frame witness and signature type; the
    // admitted binding's population is carried across, never re-chosen.
    let population = match b.method.population {
        Population::Bag => lab::Population::Bag,
        Population::Distinct => lab::Population::Distinct,
    };
    let (x, y, z) = (
        lab::Axis {
            class: 10,
            kind: lab::AxisKind::Identity,
        },
        lab::Axis {
            class: 11,
            kind: lab::AxisKind::Property,
        },
        lab::Axis {
            class: 12,
            kind: lab::AxisKind::Community,
        },
    );
    let lab_frames = lab::FrameStore::new(lab::FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let xs = AddressMap::new(x, r.iter().map(|t| t.0)).unwrap();
    let ys = AddressMap::new(y, r.iter().map(|t| t.1).chain(s.iter().map(|t| t.0))).unwrap();
    let zs = AddressMap::new(z, s.iter().map(|t| t.1)).unwrap();
    let rr = register(&r, &xs, &ys).unwrap();
    let _ss = register(&s, &ys, &zs).unwrap();
    let sig = lab::MethodSignature {
        population,
        ..lab::MethodSignature::presence(XLOG_VIEW, COUNT.0 as u16, y, y)
    };
    let guards = lab::Guards {
        frame: lab_frames.seal(),
        focus: lab::Focus { revision: 1 },
        scope: lab::Scope { id: 2 },
    };
    let e = lab::bind_dr64(
        &lab_frames,
        sig,
        sig,
        &guards,
        (guards.focus, guards.scope),
        lab::Operand {
            rows: &rr,
            validity: None,
        },
    );
    assert_eq!(e.err(), Some(lab::Refusal::BagNeedsMultiplicity));
    assert_eq!(rr.len(), SLOTS);
}
