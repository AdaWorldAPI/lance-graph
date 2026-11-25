//! The VRL recipe on the existing witnessed path, against the naive VRL
//! interpreter, at every width.

mod common;

use common::{all_fields, bind, witness, Source, CODE, LABELS, VIEW_CLASS};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::LaneRefusal;
use vrl_recipe_probe::{interpret, Event, Recipe, Refusal, Rows};

/// Positional rows as events, for comparison with the interpreter only.
fn events(rows: &Rows<'_>) -> Vec<Event> {
    (0..rows.len())
        .map(|r| {
            rows.names()
                .iter()
                .enumerate()
                .map(|(c, n)| (n.clone(), i64::from(rows.get(r, c))))
                .collect()
        })
        .collect()
}

fn compile(src: &str) -> Recipe {
    Recipe::compile(src, &common::View::new(), VIEW_CLASS)
        .unwrap_or_else(|e| panic!("{src}: {e:?}"))
}

/// Programs inside the subset, each exercising a different part of it.
const PROGRAMS: &[&str] = &[
    ".name = .key\n.amount = .properties.amount.value",
    ".name = .key\n.amount = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".id = .key\nif .properties.region.value == 0 { abort }\nif .properties.status.value == 3 { abort }",
    ".orig = .key\n.key = .properties.region.value\n.reg = .key\nif .key != 1 { abort }",
    ".a = .key; .b = .properties.amount.value; del(.a)",
    ".a = .properties.status.value; del(.key); if .a == 0 { abort }",
    ".s = .properties.status.value\n.amt = .properties.amount.value\nif .s != 1 { abort }",
    ".id = .key\nif .key < 1010 { abort }\nif .key >= 1050 { abort }",
    ".a = .properties.amount.value\nif .a <= 0 { abort }\nif .properties.status.value > 1 { abort }",
    ".r = .properties.region.value\nif .r < 1 { abort }\nif .r >= 2 { abort }",
];

/// The same recipe answers as the interpreter does over materialized events,
/// at widths 0, 1, 63, 64, 65, 256, 512 and 1000: rows in row order, names,
/// the count, and the sum of every output.
#[test]
fn recipe_equals_the_interpreter_at_every_width() {
    let mut kept_any = false;
    for &n in &[0usize, 1, 63, 64, 65, 256, 512, 1000] {
        let source = Source::random(n, 0x7e51_0000 + n as u64);
        let world = bind(&source, &all_fields());
        let w = witness(&world, &source);
        for (i, src) in PROGRAMS.iter().enumerate() {
            let recipe = compile(src);
            let want = interpret(src, &LABELS[..4], &source.i32_lanes()).unwrap();
            let rows = recipe.materialize_rows(&w).unwrap();
            assert_eq!(
                rows.names(),
                recipe.outputs().as_slice(),
                "n {n} p{i} names"
            );
            assert!(
                rows.positions().windows(2).all(|p| p[0] < p[1]),
                "row order"
            );
            assert_eq!(events(&rows), want, "n {n} program {i}");
            assert_eq!(
                recipe.count(&w).unwrap(),
                want.len() as u64,
                "n {n} p{i} count"
            );
            for out in recipe.outputs() {
                let total: i64 = want.iter().map(|e| e[out]).sum();
                assert_eq!(recipe.sum(&w, out).unwrap(), total, "n {n} p{i} sum {out}");
            }
            kept_any |= !want.is_empty() && want.len() < n;
        }
    }
    assert!(
        kept_any,
        "no program both kept and dropped rows: the filters are not exercised"
    );
}

/// The filters are load-bearing: each filtered program keeps strictly fewer
/// rows than the same program without its `if` lines.
#[test]
fn every_filter_drops_rows() {
    let source = Source::random(512, 9);
    let world = bind(&source, &all_fields());
    let w = witness(&world, &source);
    for src in PROGRAMS.iter().filter(|p| p.contains("if ")) {
        let unfiltered: String = src
            .split(['\n', ';'])
            .filter(|l| !l.trim_start().starts_with("if "))
            .collect::<Vec<_>>()
            .join("\n");
        let kept = compile(src).count(&w).unwrap();
        let all = compile(&unfiltered).count(&w).unwrap();
        assert!(kept < all && all == 512, "{src}: kept {kept} of {all}");
    }
}

/// The recipe asks for exactly the fields it reads, outputs and filters.
#[test]
fn the_recipe_binds_exactly_the_fields_it_reads() {
    let r = compile(".n = .key\nif .properties.region.value == 1 { abort }");
    assert_eq!(r.fields().count(), 2);
    assert!(r.fields().has(0) && r.fields().has(2) && !r.fields().has(3));
}

/// Each construct that computes a value is refused, by name, and never
/// approximated; each unknown path is refused as such.
#[test]
fn computing_constructs_are_refused_by_name() {
    let cases: &[(&str, Refusal)] = &[
        (
            ".a = parse_json(.key)",
            Refusal::Computes("function parse_json".into()),
        ),
        (".t = now()", Refusal::Computes("function now".into())),
        (".a = .key + 1", Refusal::Computes("arithmetic".into())),
        (".a = \"x\"", Refusal::Computes("string literal".into())),
        (".a = 5", Refusal::Computes("integer literal".into())),
        (
            "if .key =~ 3 { abort }",
            Refusal::Computes("comparison =~".into()),
        ),
        (
            "if .key > 3 && .key < 9 { abort }",
            Refusal::Computes("condition .key > 3 && .key < 9".into()),
        ),
        (
            "if .key == 3 { .a = .key }",
            Refusal::Computes("if body other than abort".into()),
        ),
        (
            "to_string(.key)",
            Refusal::Computes("function to_string".into()),
        ),
        (
            ".properties.x.value = .key",
            Refusal::Computes("assignment into properties".into()),
        ),
        (".a = .nope", Refusal::UnknownPath(".nope".into())),
        (
            ".a = .properties.nope.value",
            Refusal::UnknownPath(".properties.nope.value".into()),
        ),
        ("del(.nope)", Refusal::UnknownPath(".nope".into())),
    ];
    for (src, want) in cases {
        let got = Recipe::compile(src, &common::View::new(), VIEW_CLASS).unwrap_err();
        assert_eq!(got, *want, "{src}");
    }
}

/// After the source writes (its frame advances), every terminal of a recipe
/// is refused by DT64's frame check before any lane is borrowed. A recipe is
/// read only against the frame it was bound under.
#[test]
fn a_moved_frame_is_refused_at_every_terminal() {
    let mut source = Source::random(64, 3);
    let world = bind(&source, &all_fields());
    let r = compile(".n = .key");
    assert_eq!(r.count(&witness(&world, &source)).unwrap(), 64);
    source.advance();
    let w = witness(&world, &source);
    let stale = |e: Refusal| e == Refusal::Binding(LaneRefusal::Frame(FrameRefusal::SourceChanged));
    assert!(stale(r.count(&w).unwrap_err()), "count");
    assert!(stale(r.sum(&w, "n").unwrap_err()), "sum");
    assert!(stale(r.materialize_rows(&w).unwrap_err()), "materialize");
}

/// A recipe reading a field the binding does not hold is refused, and a
/// field that is not an `i32` lane is refused, both before folding.
#[test]
fn unbound_and_non_i32_fields_are_refused() {
    let source = Source::random(64, 4);
    let narrow = bind(&source, &WideFieldMask::from_positions(&[0]));
    let r = compile(".n = .key\n.a = .properties.amount.value");
    assert_eq!(
        r.count(&witness(&narrow, &source)).unwrap_err(),
        Refusal::NotBound { field: 3 }
    );
    let world = bind(&source, &all_fields());
    let c = compile(".c = .properties.code.value");
    assert_eq!(
        c.count(&witness(&world, &source)).unwrap_err(),
        Refusal::NotI32 { field: CODE }
    );
}

/// A prepared program is run only by the recipe that prepared it and only
/// over a frame of the row count it was prepared for. A different recipe
/// would number its lanes differently, and a different row count would make
/// the every-row range describe other rows. Both are refused before folding.
#[test]
fn a_prepared_program_is_refused_by_another_recipe_or_row_count() {
    use lance_graph_mask_risc::{Out, Value};
    let small = Source::random(64, 5);
    let big = Source::random(128, 5);
    let (ws, wb) = (bind(&small, &all_fields()), bind(&big, &all_fields()));
    let r = compile(".n = .key\n.a = .properties.amount.value");
    let other = compile(".n = .key");
    let p = r.prepare_count(64).unwrap();
    let mut scratch = p.scratch().unwrap();
    assert_eq!(
        r.run_into(&p, &witness(&ws, &small), &mut scratch, Out::None),
        Ok(Value::Count(64))
    );
    assert_eq!(
        other.run_into(&p, &witness(&ws, &small), &mut scratch, Out::None),
        Err(Refusal::NotThisRecipe)
    );
    assert_eq!(
        r.run_into(&p, &witness(&wb, &big), &mut scratch, Out::None),
        Err(Refusal::RowsChanged {
            prepared: 64,
            now: 128
        })
    );
}

/// A recipe reading more fields than the fixed lane array holds is refused
/// at compile time, not truncated; one field fewer compiles. Both sides
/// read through a view with `MAX_READS + 1` fields.
#[test]
fn more_reads_than_the_lane_array_holds_are_refused() {
    use vrl_recipe_probe::MAX_READS;
    let labels: Vec<String> = std::iter::once("key".to_string())
        .chain((1..=MAX_READS).map(|i| format!("f{i}")))
        .collect();
    let view = common::View::with_labels(&labels);
    let reading = |k: usize| -> String {
        std::iter::once(".o0 = .key\n".to_string())
            .chain((1..k).map(|i| format!(".o{i} = .properties.f{i}.value\n")))
            .collect()
    };
    assert!(Recipe::compile(&reading(MAX_READS), &view, VIEW_CLASS).is_ok());
    assert_eq!(
        Recipe::compile(&reading(MAX_READS + 1), &view, VIEW_CLASS).unwrap_err(),
        Refusal::TooManyReads {
            reads: MAX_READS + 1
        }
    );
}

/// Each comparison drops exactly the rows its meaning names, at the
/// boundary: over amounts -2..=2 with the constant 0, `<` and `<=` (and `>`
/// and `>=`) differ by the one row equal to 0, so an off-by-one or a swapped
/// lowering is caught. Signed: the negative amounts are below 0.
#[test]
fn each_comparison_drops_exactly_its_rows_at_the_boundary() {
    let amounts: Vec<i32> = vec![-2, -1, 0, 1, 2];
    let n = amounts.len();
    let source = Source::from_lanes([
        (0..n as i32).collect(),
        vec![0; n],
        vec![0; n],
        amounts.clone(),
    ]);
    let world = bind(&source, &all_fields());
    let w = witness(&world, &source);
    // Rows kept = rows where the comparison does NOT hold.
    for (op, kept) in [
        ("==", 4),
        ("!=", 1),
        ("<", 3),
        ("<=", 2),
        (">", 3),
        (">=", 2),
    ] {
        let src = format!(".a = .properties.amount.value\nif .a {op} 0 {{ abort }}");
        let r = compile(&src);
        assert_eq!(r.count(&w).unwrap(), kept, "{op}");
        let want: i64 = amounts
            .iter()
            .filter(|&&a| match op {
                "==" => a != 0,
                "!=" => a == 0,
                "<" => a >= 0,
                "<=" => a > 0,
                ">" => a <= 0,
                _ => a < 0,
            })
            .map(|&a| i64::from(a))
            .sum();
        assert_eq!(r.sum(&w, "a").unwrap(), want, "{op}");
    }
}

/// Two recipes that read the same fields but drop different rows build
/// different programs; one's prepared program is refused by the other.
#[test]
fn a_prepared_program_is_refused_by_a_recipe_with_the_same_reads_and_other_drops() {
    use lance_graph_mask_risc::Out;
    let src = Source::random(64, 9);
    let w = bind(&src, &all_fields());
    let keep_low = compile(".n = .key\nif .key >= 0 { abort }");
    let keep_high = compile(".n = .key\nif .key < 0 { abort }");
    let p = keep_low.prepare_count(64).unwrap();
    let mut scratch = p.scratch().unwrap();
    assert!(keep_low
        .run_into(&p, &witness(&w, &src), &mut scratch, Out::None)
        .is_ok());
    assert_eq!(
        keep_high.run_into(&p, &witness(&w, &src), &mut scratch, Out::None),
        Err(Refusal::NotThisRecipe)
    );
}

/// A witness that binds another field at the same position is refused: the
/// recipe checks the predicate it compiled against, not only the ordinal.
#[test]
fn a_witness_with_another_field_at_the_same_position_is_refused() {
    let src = Source::random(64, 3);
    let w = bind(&src, &all_fields());
    let mut labels: Vec<String> = LABELS.iter().map(|l| l.to_string()).collect();
    labels[3] = "price".into();
    let r = Recipe::compile(
        ".p = .properties.price.value",
        &common::View::with_labels(&labels),
        VIEW_CLASS,
    )
    .unwrap();
    assert_eq!(
        r.materialize_rows(&witness(&w, &src))
            .map(|rows| rows.len()),
        Err(Refusal::OtherField { field: 3 })
    );
    // The same recipe compiled against the witness's own view reads.
    assert!(compile(".p = .properties.amount.value")
        .materialize_rows(&witness(&w, &src))
        .is_ok());
}

/// A witness bound for another class is refused before any lane is read.
#[test]
fn a_witness_for_another_class_is_refused() {
    use lance_graph_contract::class_view::{ClassId, ClassView};
    use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
    struct Any(Vec<FieldRef>);
    impl ClassView for Any {
        fn fields(&self, _: ClassId) -> &[FieldRef] {
            &self.0
        }
        fn template(&self, _: ClassId) -> DisplayTemplate {
            DisplayTemplate::Detail
        }
        fn dolce_category_id(&self, _: ClassId) -> u8 {
            0
        }
    }
    let any = Any(LABELS
        .iter()
        .map(|l| FieldRef::new(format!("ord:{l}"), *l))
        .collect());
    let r = Recipe::compile(".n = .key", &any, VIEW_CLASS + 1).unwrap();
    let src = Source::random(64, 4);
    let w = bind(&src, &all_fields());
    assert_eq!(
        r.materialize_rows(&witness(&w, &src))
            .map(|rows| rows.len()),
        Err(Refusal::OtherClass)
    );
}

/// Deleting the only output leaves the population: one empty row per row, as
/// the interpreter answers, not a refusal for reading no field.
#[test]
fn deleting_every_output_keeps_the_population() {
    let src = Source::random(64, 6);
    let w = bind(&src, &all_fields());
    let r = compile(".id = .key\ndel(.id)");
    assert!(r.outputs().is_empty());
    let rows = r.materialize_rows(&witness(&w, &src)).unwrap();
    assert_eq!(rows.len(), 64);
    assert!(rows.names().is_empty());
    let lanes = src.i32_lanes();
    assert_eq!(
        interpret(".id = .key\ndel(.id)", &LABELS[..4], &lanes)
            .unwrap()
            .len(),
        64
    );
}

/// Rows carry the seal they were taken under. Held positions re-read
/// through `borrow_all_at` under that seal are admitted while the source is
/// unchanged and refused once a write has moved it.
#[test]
fn held_positions_are_refused_after_the_source_moves() {
    let mut src = Source::random(64, 8);
    let w = bind(&src, &all_fields());
    let r = compile(".n = .key\nif .key < 0 { abort }");
    let (frame, held) = {
        let rows = r.materialize_rows(&witness(&w, &src)).unwrap();
        (rows.frame().clone(), rows.positions().to_vec())
    };
    assert!(!held.is_empty());
    assert!(w
        .binding
        .borrow_all_at(&w.store, &src.frames, &src, &held, &frame)
        .is_ok());
    src.advance();
    assert!(matches!(
        w.binding
            .borrow_all_at(&w.store, &src.frames, &src, &held, &frame),
        Err(lance_graph_dt64::LaneRefusal::Frame(_))
    ));
}

/// Two recipes that read the same fields and drop the same rows but swap
/// their outputs are different recipes: a sum prepared for one is refused
/// by the other. A recipe compiled again from the same source is the same
/// recipe and runs it.
#[test]
fn a_prepared_sum_is_refused_by_a_recipe_with_swapped_outputs() {
    use lance_graph_mask_risc::Out;
    let src = Source::random(64, 10);
    let w = bind(&src, &all_fields());
    let a = compile(".a = .key\n.b = .properties.amount.value");
    let b = compile(".a = .properties.amount.value\n.b = .key");
    let p = a.prepare_sum(64, "a").unwrap();
    let mut scratch = p.scratch().unwrap();
    assert!(a
        .run_into(&p, &witness(&w, &src), &mut scratch, Out::None)
        .is_ok());
    assert!(compile(".a = .key\n.b = .properties.amount.value")
        .run_into(&p, &witness(&w, &src), &mut scratch, Out::None)
        .is_ok());
    assert_eq!(
        b.run_into(&p, &witness(&w, &src), &mut scratch, Out::None),
        Err(Refusal::NotThisRecipe)
    );
}
