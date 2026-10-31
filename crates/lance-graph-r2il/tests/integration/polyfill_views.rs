//! The production alias views of the polyfill lens (D-DT64-7 L4): one
//! `ClassView` per language, derived from `POLYFILL`, and an adapter that
//! puts one over a consumer's own view.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, MethodBinding, MethodId, MethodRef,
    SchemaRev, SigRev, ValueDemand,
};
use lance_graph_r2il::binding::{prepare, ExecRefusal, Prepared, R2ilLens, Resolution};
use lance_graph_r2il::polyfill::{
    polyfill_arities, position_of, AliasLanguage, PolyfillView, WithPolyfill, HARDCODED, POLYFILL,
    POLYFILL_LENS,
};
use ogar_loco::Vocabulary;
use ogar_r2il::R2ILVocabulary;

const LANGS: [AliasLanguage; 3] = [
    AliasLanguage::R2il,
    AliasLanguage::Sql,
    AliasLanguage::Quack,
];

// ----------------------------------------------------- the views themselves

#[test]
fn each_language_view_labels_every_position_with_its_own_name() {
    for lang in LANGS {
        let v = PolyfillView::new(lang);
        let fields = v.fields(HARDCODED);
        assert_eq!(fields.len(), POLYFILL.len(), "{lang:?}: one label per word");
        for (p, w) in POLYFILL.iter().enumerate() {
            let label = fields[p].label.as_str();
            assert_eq!(label, w.name_in(lang), "{lang:?} position {p}");
            assert!(w.is_alias(label), "{lang:?} position {p}: {label:?}");
            assert_eq!(
                position_of(label),
                u8::try_from(p).ok(),
                "{lang:?}: {label:?} names exactly this position"
            );
        }
    }
}

#[test]
fn the_word_identity_is_the_same_in_every_language() {
    let views: Vec<_> = LANGS.into_iter().map(PolyfillView::new).collect();
    for p in 0..POLYFILL.len() {
        let iris: Vec<&str> = views
            .iter()
            .map(|v| v.fields(HARDCODED)[p].predicate_iri.as_str())
            .collect();
        assert!(iris.iter().all(|i| *i == iris[0]), "position {p}: {iris:?}");
    }
    // ...while the labels differ where the languages differ.
    let lt = position_of("IntSLess").unwrap() as usize;
    let labels: Vec<&str> = views
        .iter()
        .map(|v| v.fields(HARDCODED)[lt].label.as_str())
        .collect();
    assert_eq!(labels, ["IntSLess", "<", "Cmp::LtI32"]);
}

#[test]
fn each_language_names_come_from_the_table_not_a_spelling_rule() {
    for w in &POLYFILL {
        assert_eq!(w.name_in(AliasLanguage::R2il), w.aliases[0]);
        assert_eq!(
            Some(w.name_in(AliasLanguage::R2il)),
            R2ILVocabulary.name(w.opcode)
        );
        for lang in [AliasLanguage::Sql, AliasLanguage::Quack] {
            match w.name_of(lang) {
                Some(n) => assert!(w.aliases[1..].contains(&n), "{n:?}"),
                // No name in that language: the view falls back to the
                // R2IL name, so the basis keeps one label per position.
                None => assert_eq!(w.name_in(lang), w.aliases[0]),
            }
        }
    }
    // SUM has no SQL alias: SQL's SUM is NULL on an empty selection,
    // this word is 0 (polyfill module doc).
    let sum = &POLYFILL[position_of("SUM").unwrap() as usize];
    assert_eq!(sum.name_of(AliasLanguage::Sql), None);
    let lt = &POLYFILL[position_of("IntSLess").unwrap() as usize];
    assert_eq!(lt.name_of(AliasLanguage::Sql), Some("<"));
    assert_eq!(lt.name_of(AliasLanguage::Quack), Some("Cmp::LtI32"));
}

#[test]
fn a_polyfill_view_answers_only_the_hardcoded_concept() {
    let v = PolyfillView::new(AliasLanguage::Sql);
    assert!(v.fields(0x7F0B).is_empty());
    assert_eq!(v.field_count(HARDCODED), POLYFILL.len());
}

// --------------------------------------- over a consumer view, through prepare

const CLASS: u32 = 0x7F0B_0001;
const CLASS_VIEW: ClassId = 0x7F0B;
const REV: SchemaRev = SchemaRev(1);
const POSITION: u16 = 3;

/// A consumer's own view: its class, and a stale label set for `0xFFFF`
/// that the adapter must hide.
struct Consumer {
    class: Vec<FieldRef>,
    stale: Vec<FieldRef>,
}

impl Consumer {
    fn new() -> Self {
        Self {
            class: ["a", "b", "k", "m", "s"]
                .iter()
                .map(|l| FieldRef::new(format!("lab:class/{l}"), *l))
                .collect(),
            stale: vec![FieldRef::new("lab:stale", "MIN")],
        }
    }
}

impl ClassView for Consumer {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW => &self.class,
            HARDCODED => &self.stale,
            _ => &[],
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        7
    }
}

fn store() -> (BindStore, lance_graph_dt64::RouteVersion) {
    let mut s = BindStore::new();
    s.declare_algebra(POLYFILL_LENS, REV, Extent::DR64_CUBE, polyfill_arities())
        .unwrap();
    let word = |name: &str| AlgebraRef {
        lens: POLYFILL_LENS,
        word: position_of(name).unwrap(),
    };
    s.register(
        CLASS,
        ClassSchema {
            schema_rev: REV,
            field_count: 5,
            extent: Extent::DR64_CUBE,
        },
        vec![MethodBinding {
            id: MethodId(1),
            sig_rev: SigRev(1),
            name: "sum_below".into(),
            on: WideFieldMask::from_positions(&[0, 1]),
            when: None,
            reads: WideFieldMask::from_positions(&[0, 1]),
            effects: vec![
                Effect {
                    target: 3,
                    algebra: word("IntSLess"),
                    args: vec![Arg::Field(0), Arg::Const(5)],
                },
                Effect {
                    target: 4,
                    algebra: word("SUM"),
                    args: vec![Arg::Field(3), Arg::Field(1)],
                },
            ],
            population: Population::Distinct,
            truth: Truth::TwoValued,
            demand: ValueDemand::Values,
        }],
    )
    .unwrap();
    let v = s.route(POSITION, CLASS).unwrap();
    (s, v)
}

fn prep<V: ClassView + ?Sized>(view: &V) -> Result<Prepared, ExecRefusal> {
    let (s, version) = store();
    let frames = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let now = (Focus { revision: 1 }, Scope { id: 1 });
    let r = MethodRef {
        class: CLASS,
        method: MethodId(1),
        sig_rev: SigRev(1),
        guard: None,
    };
    let g = Guards {
        frame: frames.seal(),
        focus: now.0,
        scope: now.1,
    };
    prepare(
        &s,
        view,
        R2ilLens {
            lens: POLYFILL_LENS,
            schema_rev: REV,
        },
        &Resolution {
            method: &r,
            schema_rev: REV,
            route: (version, POSITION),
            guards: &g,
        },
        &frames,
        now,
    )
}

#[test]
fn the_adapter_binds_the_same_plan_in_every_language() {
    let consumer = Consumer::new();
    let plans: Vec<Prepared> = LANGS
        .into_iter()
        .map(|lang| {
            let pv = PolyfillView::new(lang);
            prep(&WithPolyfill::new(&consumer, &pv)).expect("prepares")
        })
        .collect();
    for p in &plans[1..] {
        assert_eq!(p.steps, plans[0].steps);
        assert_eq!(p.plan_id, plans[0].plan_id);
    }
}

#[test]
fn the_adapter_hides_the_consumers_own_hardcoded_labels() {
    let consumer = Consumer::new();
    // Twin: the consumer's view alone carries a one-label basis for 0xFFFF,
    // so the lens's 17-word basis does not match it.
    assert!(matches!(prep(&consumer), Err(ExecRefusal::Binding(_))));
    let pv = PolyfillView::new(AliasLanguage::Quack);
    let both = WithPolyfill::new(&consumer, &pv);
    assert_eq!(both.fields(HARDCODED).len(), POLYFILL.len());
    // Every other concept is the consumer's, including template and DOLCE.
    assert_eq!(both.fields(CLASS_VIEW), consumer.fields(CLASS_VIEW));
    assert_eq!(both.template(CLASS_VIEW), DisplayTemplate::Detail);
    assert_eq!(both.dolce_category_id(CLASS_VIEW), 7);
}
