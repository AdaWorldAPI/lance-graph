//! The r2il parity check (EXO-290 Ask 1, as decided: a parity test, not the
//! execution path). Each recipe is registered, independently of its own
//! lowering, as a DT64 method of frozen `F000` words under `POLYFILL_LENS`
//! (`IntEqual`, `IntSLess`, `IntNot`, `IntAnd`, `PopCount`, `SUM`) and run
//! through `lance-graph-r2il`'s `prepare`/`run`. Its count and sum must equal
//! the recipe's prepared mask-risc program and the reference interpreter.

mod common;

use common::{all_fields, bind, witness, Source, LABELS, VIEW_CLASS};
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, MethodBinding, MethodId, SchemaRev,
    SigRev, ValueDemand,
};
use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_r2il::binding::{prepare, run, ExecPermit, FieldLanes, R2ilLens, Resolution};
use lance_graph_r2il::polyfill::{
    polyfill_arities, position_of, HARDCODED, POLYFILL, POLYFILL_LENS,
};
use lance_graph_r2il::FoldDialect;
use vrl_recipe_probe::{interpret, Cmp, Recipe};

const CLASS: u32 = 0x7F0C_0001;
const CLASS_VIEW: ClassId = 0x7F0C;
const REV: SchemaRev = SchemaRev(1);
const LENS: R2ilLens = R2ilLens {
    lens: POLYFILL_LENS,
    schema_rev: REV,
};
const NOW: (Focus, Scope) = (Focus { revision: 1 }, Scope { id: 1 });
const POSITION: u16 = 5;
/// Fields 0..=4 are the recipe's class (`LABELS`); fields from `SCRATCH` on
/// hold the method's intermediate values and its answer.
const SCRATCH: u8 = 5;
const N_FIELDS: usize = 40;

struct Views {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

impl Views {
    fn new() -> Views {
        let class = (0..N_FIELDS)
            .map(|i| {
                let l = LABELS.get(i).map_or(format!("t{i}"), |l| l.to_string());
                FieldRef::new(format!("lab:class/{l}"), l)
            })
            .collect();
        let lens = POLYFILL
            .iter()
            .map(|w| FieldRef::new(format!("lab:polyfill/{}", w.aliases[0]), w.aliases[0]))
            .collect();
        Views { class, lens }
    }
}

impl ClassView for Views {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW => &self.class,
            HARDCODED => &self.lens,
            _ => &[],
        }
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        (lens == POLYFILL_LENS).then_some(self.lens.as_slice())
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

fn step(target: u8, name: &str, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: AlgebraRef {
            lens: POLYFILL_LENS,
            word: position_of(name).unwrap_or_else(|| panic!("{name:?} is no polyfill alias")),
        },
        args,
    }
}

/// The recipe's population and one fold as F000 words. Each term `field
/// <cmp> v` drops rows; its kept rows are `NOT b` or `b` for a base compare
/// `b` that the frozen polyfill has (`IntEqual`, `IntSLess`).
fn f000_method(id: u32, recipe: &Recipe, sum_of: Option<u8>) -> (MethodBinding, u8) {
    let mut effects = Vec::new();
    let mut next = SCRATCH;
    let mut fresh = || {
        let f = next;
        next += 1;
        f
    };
    let mut kept: Option<u8> = None;
    for &(field, cmp, v) in recipe.terms() {
        let v = i64::from(v);
        let (base, threshold, negate) = match cmp {
            Cmp::Eq => ("IntEqual", v, true),
            Cmp::Ne => ("IntEqual", v, false),
            Cmp::Lt => ("IntSLess", v, true),
            Cmp::Le => ("IntSLess", v + 1, true),
            Cmp::Gt => ("IntSLess", v + 1, false),
            Cmp::Ge => ("IntSLess", v, false),
        };
        let b = fresh();
        effects.push(step(
            b,
            base,
            vec![Arg::Field(field), Arg::Const(threshold)],
        ));
        let keep = if negate {
            let k = fresh();
            effects.push(step(k, "IntNot", vec![Arg::Field(b)]));
            k
        } else {
            b
        };
        kept = Some(match kept {
            None => keep,
            Some(prev) => {
                let a = fresh();
                effects.push(step(a, "IntAnd", vec![Arg::Field(prev), Arg::Field(keep)]));
                a
            }
        });
    }
    let kept = kept.expect("parity programs filter");
    let answer = fresh();
    effects.push(match sum_of {
        None => step(answer, "PopCount", vec![Arg::Field(kept)]),
        Some(f) => step(answer, "SUM", vec![Arg::Field(kept), Arg::Field(f)]),
    });
    let reads: Vec<u8> = recipe.terms().iter().map(|t| t.0).chain(sum_of).collect();
    let m = MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("vrl{id}"),
        on: WideFieldMask::from_positions(&reads),
        when: None,
        reads: WideFieldMask::from_positions(&reads),
        effects,
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    };
    (m, answer)
}

/// Run `methods` through r2il over `source`'s lanes; the answer each one
/// leaves in its answer field.
fn r2il_answers(source: &Source, methods: Vec<(MethodBinding, u8)>) -> Vec<i64> {
    let views = Views::new();
    let mut store = BindStore::new();
    store
        .declare_algebra(POLYFILL_LENS, REV, Extent::DR64_CUBE, polyfill_arities())
        .unwrap();
    let answers: Vec<(u32, u8)> = methods.iter().map(|(m, a)| (m.id.0, *a)).collect();
    store
        .register(
            &views,
            CLASS,
            ClassSchema {
                schema_rev: REV,
                field_count: N_FIELDS,
                extent: Extent::DR64_CUBE,
            },
            methods.into_iter().map(|(m, _)| m).collect(),
        )
        .unwrap();
    let route = store.route(POSITION, CLASS).unwrap();
    let frames = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let lanes: Vec<LaneRef<'_>> = source.lanes.iter().map(|l| LaneRef::I32(l)).collect();
    let mut field_lanes = [None; N_FIELDS];
    for (f, slot) in field_lanes.iter_mut().enumerate().take(source.lanes.len()) {
        *slot = Some(f as u16);
    }
    let planes = Planes {
        n_rows: source.lanes[0].len(),
        masks: &[],
        lanes: &lanes,
    };
    answers
        .into_iter()
        .map(|(id, answer)| {
            let r = store.method_ref(CLASS, MethodId(id)).unwrap();
            let guards = Guards {
                frame: frames.seal(),
                focus: NOW.0,
                scope: NOW.1,
            };
            let plan = prepare(
                &store,
                &views,
                LENS,
                &Resolution {
                    method: &r,
                    schema_rev: REV,
                    route: (route, POSITION),
                    guards: &guards,
                },
                &frames,
                NOW,
            )
            .unwrap_or_else(|e| panic!("method {id} prepares: {e:?}"));
            let foreign = Foreign::NONE;
            let mut d = FoldDialect::new(&planes, &foreign);
            let receipt = run(
                &plan,
                &store,
                &frames,
                NOW,
                &ExecPermit {
                    methods: vec![plan.method().clone()],
                },
                FieldLanes {
                    lanes: &field_lanes,
                },
                &mut d,
            )
            .unwrap_or_else(|e| panic!("method {id} runs: {e:?}"));
            receipt
                .output
                .values
                .iter()
                .find(|(f, _)| *f == answer)
                .map(|(_, v)| *v)
                .expect("answer field written")
        })
        .collect()
}

/// Recipes covering every comparison, both polarities of the base compares,
/// and conjunctions of two and three terms.
const PROGRAMS: &[&str] = &[
    ".a = .properties.amount.value\nif .a == 7 { abort }",
    ".a = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".a = .properties.amount.value\nif .a < 0 { abort }",
    ".a = .properties.amount.value\nif .a <= 120 { abort }",
    ".a = .properties.amount.value\nif .a > 300 { abort }",
    ".a = .properties.amount.value\nif .a >= -50 { abort }",
    ".a = .properties.amount.value\nif .key < 1010 { abort }\nif .key >= 1050 { abort }",
    ".a = .properties.amount.value\nif .properties.region.value == 0 { abort }\nif .properties.status.value > 2 { abort }\nif .a <= 10 { abort }",
];

/// For every program at widths 1, 63, 64, 65, 300 and 1000, the F000 method
/// run through r2il answers what the prepared mask-risc program and the
/// reference interpreter answer, for the count and the sum of `.a`.
#[test]
fn f000_through_r2il_matches_the_prepared_program_and_the_interpreter() {
    let mut distinct = std::collections::BTreeSet::new();
    for &n in &[1usize, 63, 64, 65, 300, 1000] {
        let source = Source::random(n, 11 + n as u64);
        let world = bind(&source, &all_fields());
        let w = witness(&world, &source);
        let view = common::View::new();
        let lanes: Vec<&[i32]> = source.lanes.iter().map(Vec::as_slice).collect();
        let mut methods = Vec::new();
        let mut want = Vec::new();
        for (i, src) in PROGRAMS.iter().enumerate() {
            let r = Recipe::compile(src, &view, VIEW_CLASS).unwrap();
            let amount = r.output_field("a").unwrap();
            let events = interpret(src, &LABELS[..4], &lanes).unwrap();
            let count = r.count(&w).unwrap() as i64;
            let sum = r.sum(&w, "a").unwrap();
            assert_eq!(
                count,
                events.len() as i64,
                "{src} at {n}: interpreter count"
            );
            assert_eq!(
                sum,
                events.iter().map(|e| e["a"]).sum::<i64>(),
                "{src} at {n}: interpreter sum"
            );
            let id = 2 * i as u32;
            methods.push(f000_method(id + 1, &r, None));
            methods.push(f000_method(id + 2, &r, Some(amount)));
            want.extend([count, sum]);
        }
        let got = r2il_answers(&source, methods);
        assert_eq!(got, want, "r2il vs prepared program at n = {n}");
        distinct.extend(got);
    }
    // Anti-vacuity: the answers do not collapse to a few values.
    assert!(distinct.len() > 40, "answers: {distinct:?}");
}
