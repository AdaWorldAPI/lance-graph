//! Plan `dr-dt64-class-governance-v3` §4 W12, against the real OGAR authority
//! and OGAR's own ClassView. The consumer is MedCare (`HealthcarePort`).

use lance_graph_contract::canonical_node::{classid_read_mode, ReadMode};
use lance_graph_contract::class_view::{ClassView, WideFieldMask};
use lance_graph_contract::hotplug::{ActivationDrift, CapabilityAuthority, HotPlug};
use lance_graph_contract::ogar_codebook::render_classid;
use lance_graph_contract::ontology::FieldRef;
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneRequest, LaneSource, MethodBinding,
    MethodId, Refusal, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_ogar::ogar_vocab::ports::{HealthcarePort, PortSpec};
use lance_graph_ogar::{OgarAuthority, OgarClassView};

use dt64_hotplug_probe::{register_activated, Unbound};

/// MedCare's patient and diagnosis concepts, through the port, never as
/// literals.
fn concept(name: &str) -> u16 {
    HealthcarePort::class_id(name).expect("the port names this class")
}

/// The full classid: the concept in the canon half, MedCare's render prefix
/// in the custom half.
fn classid(name: &str) -> u32 {
    render_classid(HealthcarePort::APP_PREFIX, concept(name))
}

/// MedCare's plug for the patient concept only, covering exactly its four
/// capabilities.
const PATIENT_ONLY: HotPlug = HotPlug {
    consumer: "medcare-rs",
    classids: &[0x0901],
    covered: &[
        "register_patient",
        "get_patient_record",
        "list_patients",
        "update_patient_access",
    ],
};

fn schema(fields: usize) -> ClassSchema {
    ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: fields,
        extent: Extent::DR64_CUBE,
    }
}

/// One method that reads field 0 of the class: no guard, no effects, so no
/// algebra has to be declared for it.
fn read_first_field(id: u32) -> MethodBinding {
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("read{id}"),
        on: WideFieldMask::from_positions(&[0]),
        when: None,
        reads: WideFieldMask::from_positions(&[0]),
        effects: Vec::new(),
        population: Population::Distinct,
        truth: Truth::ThreeValued,
        demand: ValueDemand::Values,
    }
}

const SPACE: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
const READER: Scope = Scope { id: 7 };
const POSITION: u16 = 3;

/// A source with one lane for field 0 of the patient class.
struct Source {
    class: u32,
    lane: Vec<i32>,
    frames: FrameStore,
}

impl LaneSource for Source {
    type Lane<'a> = &'a [i32];
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        (class == self.class && rev == SchemaRev(1) && field == 0).then_some(LaneAddr {
            space: SPACE,
            lane: 0,
            scope: READER,
        })
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<&[i32]> {
        (addr.lane == 0).then_some(self.lane.as_slice())
    }
    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

/// The happy path: MedCare activates the patient concept through OGAR, the
/// class is resolved by OGAR's ClassView, registered in DT64, routed, its
/// method admitted, and the field the method reads is bound to a lane.
#[test]
fn an_activated_consumer_class_binds_admits_and_reads_through_dt64() {
    assert_eq!(
        PATIENT_ONLY.classids,
        &[concept("Patient")][..],
        "the plug literal is the port's patient concept"
    );
    let activation = OgarAuthority
        .activate(&PATIENT_ONLY)
        .expect("MedCare's patient plug activates");
    let view = OgarClassView::new();
    let class = classid("Patient");
    let fields = view.fields(concept("Patient")).len();
    assert!(fields > 0, "OGAR's ClassView describes the patient class");

    let mut store = BindStore::new();
    let reading = register_activated(
        &activation,
        &mut store,
        &view,
        class,
        schema(fields),
        vec![read_first_field(1)],
    )
    .expect("an activated class registers");
    assert_eq!(reading, ReadMode::PLUG_AND_PLAY_V3);

    let version = store.route(POSITION, class).expect("routed");

    let method = store.method_ref(class, MethodId(1)).expect("bound method");
    let reads = store
        .admit(&method)
        .expect("DT64 admits its own method")
        .reads
        .clone();
    assert_eq!(reads, WideFieldMask::from_positions(&[0]));
    let source = Source {
        class,
        lane: vec![11, 22, 33],
        frames: FrameStore::new(FrameRevisions {
            source: 1,
            relation_map: 1,
            coordinates: 1,
        }),
    };
    let frame = source.frames.seal();
    let request = LaneRequest {
        route: RouteAt {
            position: POSITION,
            version,
        },
        rev: SchemaRev(1),
        fields: &reads,
        frame: &frame,
        scope: READER,
        selection: SPACE,
    };
    let lanes = store
        .bind_lanes(&view, &source.frames, &source, request)
        .expect("the method's field binds to its lane");
    assert_eq!(lanes.class(), class);
    let lane = lanes
        .lane(&store, &view, &source.frames, &source, 0)
        .expect("the bound lane borrows");
    assert_eq!(lane, &[11, 22, 33][..]);
}

/// A classid the plug did not activate is refused before DT64 is called.
#[test]
fn a_classid_the_plug_did_not_activate_is_refused_before_dt64() {
    let activation = OgarAuthority
        .activate(&PATIENT_ONLY)
        .expect("MedCare's patient plug activates");
    let view = OgarClassView::new();
    let diagnosis = classid("Diagnosis");
    let fields = view.fields(concept("Diagnosis")).len();
    assert!(fields > 0, "OGAR's ClassView describes the diagnosis class");

    let mut store = BindStore::new();
    assert_eq!(
        register_activated(
            &activation,
            &mut store,
            &view,
            diagnosis,
            schema(fields),
            vec![read_first_field(1)],
        ),
        Err(Unbound::NotActivated(ActivationDrift::NoReadingFor(
            concept("Diagnosis")
        )))
    );
    assert_eq!(store.class_count(), 0, "DT64 was not called");

    // Anti-vacuity: DT64 alone accepts the same registration. The refusal
    // above comes from the activation, not from the store.
    assert_eq!(
        store.register(&view, diagnosis, schema(fields), vec![read_first_field(1)]),
        Ok(())
    );
}

/// An activation is not an admission: an activated, registered class has
/// only the methods it was registered with.
#[test]
fn activation_does_not_admit_a_method_binding() {
    let activation = OgarAuthority
        .activate(&PATIENT_ONLY)
        .expect("MedCare's patient plug activates");
    let view = OgarClassView::new();
    let class = classid("Patient");
    let fields = view.fields(concept("Patient")).len();

    let mut store = BindStore::new();
    register_activated(
        &activation,
        &mut store,
        &view,
        class,
        schema(fields),
        vec![read_first_field(1)],
    )
    .expect("an activated class registers");
    assert_eq!(
        store.method_ref(class, MethodId(2)).unwrap_err(),
        Refusal::UnknownMethod {
            class,
            method: MethodId(2)
        }
    );
}

/// No global table learns the consumer class: the contract's builtin reading
/// for it is the default, and only the activation gives it a reading.
#[test]
fn the_global_builtins_do_not_learn_the_consumer_class() {
    let activation = OgarAuthority
        .activate(&PATIENT_ONLY)
        .expect("MedCare's patient plug activates");
    let class = classid("Patient");
    assert_ne!(
        ReadMode::DEFAULT,
        ReadMode::PLUG_AND_PLAY_V3,
        "fixture can tell them apart"
    );

    let view = OgarClassView::new();
    let mut store = BindStore::new();
    register_activated(
        &activation,
        &mut store,
        &view,
        class,
        schema(view.fields(concept("Patient")).len()),
        Vec::new(),
    )
    .expect("an activated class registers");

    assert_eq!(classid_read_mode(class), ReadMode::DEFAULT);
    assert_eq!(
        activation.read_mode_for(concept("Patient")),
        Ok(ReadMode::PLUG_AND_PLAY_V3)
    );
}

/// The activation check reads the class (left) half only, so a classid with
/// another right half over an activated class passes it. Ruled intended
/// (plan v3 §2e, 2026-11-12): DR routes, DT64 gives meaning, so what the right
/// half means is DT64's to resolve, not the activation's.
#[test]
fn the_app_half_of_the_classid_is_not_part_of_the_activation_check() {
    let activation = OgarAuthority
        .activate(&PATIENT_ONLY)
        .expect("MedCare's patient plug activates");
    let view = OgarClassView::new();
    let fields = view.fields(concept("Patient")).len();
    let other_prefix = render_classid(0x0007, concept("Patient"));
    assert_ne!(other_prefix, classid("Patient"));

    let mut store = BindStore::new();
    assert!(register_activated(
        &activation,
        &mut store,
        &view,
        other_prefix,
        schema(fields),
        Vec::new(),
    )
    .is_ok());
}
