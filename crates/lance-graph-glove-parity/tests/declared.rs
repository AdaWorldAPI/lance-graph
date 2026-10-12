//! D-CMH-1, step 2: method declarations loaded at run time.
//!
//! This test binary is compiled once. The declarations it runs are text read
//! from a file, or built from runtime values, after it starts: none of them is
//! known to the compiler. They go through the same path a compiled spec does,
//! [`MethodTable::prepare`] (binding, claim and law checks, Quack lowering),
//! and execute on the same two frontends. There is no second interpreter and
//! no second registry.
//!
//! The method keys, the reading id and the population id are FIXTURES, as in
//! `tests/method.rs`.
use lance_graph_glove_parity::basis::{canonical_fields, FieldMap, Grade, SAP_FIELDS};
use lance_graph_glove_parity::decl::{parse_decls, DeclError};
use lance_graph_glove_parity::method::{
    Answer, Cond, EffectClass, Fold, FoldValue, Frontend, Glove, Law, MethodBody, MethodKey,
    MethodSpec, MethodTable, PrepareError, ReadingId, SchemaSeal, Universe,
};
use lance_graph_mask_risc::{LaneRef, Planes};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_quack::bind::Binder;
use lance_graph_sap::bind::{billable_code, CatsBatch, BILLING, HOURS};
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

const READING: ReadingId = ReadingId(1);
const UNIVERSE: Universe = Universe(42);
const INDICATORS: [&str; 4] = ["Billable", "Non-Billable", "Internal Cost", "Unknown"];
const HOURS_TEXT: [&str; 3] = ["8.50", "0.125", "2"];

fn indicator(i: usize) -> &'static str {
    INDICATORS[(i * 7 + i / 3) % 4]
}

fn batch(n: usize) -> CatsBatch {
    let base: Vec<Option<&str>> = include_str!("../../lance-graph-sap/fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let mut input: [Vec<Option<&str>>; FIELD_COUNT] = std::array::from_fn(|i| vec![base[i]; n]);
    for i in 0..n {
        input[BILLING][i] = Some(indicator(i));
        input[HOURS][i] = Some(HOURS_TEXT[i % 3]);
    }
    CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap()
}

/// FIXTURE: SAP's real claim on `duration` is `Hypothesized`; this table
/// proves it, as `tests/method.rs` does. Loading a declaration never does.
fn with_duration() -> Vec<FieldMap> {
    SAP_FIELDS
        .iter()
        .map(|m| FieldMap {
            grade: if m.canonical == "duration" {
                Grade::Exact
            } else {
                m.grade
            },
            ..*m
        })
        .collect()
}

fn glove<'a>(basis: &'a [String], binder: &'a dyn Binder, claims: &'a [FieldMap]) -> Glove<'a> {
    Glove {
        concept: BILLABLE_WORK_ENTRY,
        basis,
        claims,
        binder,
        table: TABLE,
        reading: READING,
        universe: UNIVERSE,
    }
}

fn seal_hex() -> String {
    format!("{:016x}", SchemaSeal::of(&canonical_fields()).0)
}

fn call(t: &MethodTable, batch: &CatsBatch, label: &str, f: Frontend) -> Answer {
    let n = batch.len();
    let live = live_words(n);
    let lanes = batch.lanes();
    let planes = Planes {
        n_rows: n,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    let key = t.resolve(label).unwrap_or_else(|| panic!("label {label}"));
    t.call(key, &planes, UNIVERSE, f).unwrap()
}

fn scalars(v: &[i64]) -> Answer {
    Answer::Values(v.iter().map(|&n| FoldValue::Scalar(n)).collect())
}

/// The independent oracle: indicator text and the DTO's documented values,
/// summed by a row loop. Never the lanes a method binds.
fn oracle(n: usize, batch: &CatsBatch) -> (Vec<u32>, Vec<i64>) {
    let LaneRef::I32(hours) = batch.lanes()[HOURS] else {
        panic!("hours is an I32 lane")
    };
    (
        (0..n).map(|i| billable_code(indicator(i))).collect(),
        hours.iter().map(|&h| i64::from(h)).collect(),
    )
}

/// The declarations file, read from disk at run time.
fn fixture_text() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/declared.methods"
    );
    let text = std::fs::read_to_string(path).expect("fixture");
    // The file carries its seal literally; if the basis changes the methods
    // are refused (SchemaMismatch), so check it here to fail with a reason.
    assert!(
        text.contains(&seal_hex()),
        "re-seal the fixture: {}",
        seal_hex()
    );
    text
}

/// FAILS IF: a method that exists only as runtime text does not prepare and
/// run on both frontends, or disagrees with the row-loop oracle.
#[test]
fn methods_loaded_at_run_time_run_on_both_frontends() {
    let decls = parse_decls(&fixture_text()).expect("parses");
    let labels: Vec<&str> = decls.iter().map(|d| d.label.as_str()).collect();
    assert_eq!(
        labels,
        ["count_not_billed", "hours_by_billing", "billable_hours"]
    );
    for n in [1, 64, 65, 4097] {
        let batch = batch(n);
        let cats = CatsBinder::new(&batch);
        let basis = canonical_fields();
        let claims = with_duration();
        let t = MethodTable::prepare_with(&glove(&basis, &cats, &claims), &[], &decls);
        let (code, hours) = oracle(n, &batch);
        let not_billed = code.iter().filter(|&&c| c == 0).count() as i64;
        // Rows whose code is outside 0..2 (the undocumented indicator) are
        // dropped by the grouped terminal; the ledger records that gap.
        let mut groups = vec![0i64; 2];
        for (c, h) in code.iter().zip(&hours) {
            if (*c as usize) < 2 {
                groups[*c as usize] += h;
            }
        }
        let billable_count = code.iter().filter(|&&c| c == 1).count() as i64;
        let billable_sum: i64 = code
            .iter()
            .zip(&hours)
            .filter(|(c, _)| **c == 1)
            .map(|(_, h)| h)
            .sum();
        let cases = [
            ("count_not_billed", scalars(&[not_billed])),
            (
                "hours_by_billing",
                Answer::Values(vec![FoldValue::Groups(groups.clone())]),
            ),
            ("billable_hours", scalars(&[billable_count, billable_sum])),
        ];
        for (label, want) in cases {
            for f in [Frontend::Quack, Frontend::R2il] {
                assert_eq!(call(&t, &batch, label, f), want, "{label} {f:?} n={n}");
            }
        }
    }
    // Non-vacuity: both documented codes and the undocumented one occur, so
    // a wrong literal, a swapped group or a summed wrong lane each changes an
    // answer.
    let batch = batch(4097);
    let (code, _) = oracle(4097, &batch);
    for c in [0, 1, u32::MAX] {
        assert!(code.contains(&c), "code {c} occurs");
    }
}

/// FAILS IF: a loaded method runs a program other than the one its compiled
/// twin runs, or answers differently.
#[test]
fn a_loaded_method_is_the_compiled_method() {
    const COMPILED: MethodSpec<'static> = MethodSpec {
        key: MethodKey([0xC1; 16]),
        concept: BILLABLE_WORK_ENTRY,
        schema: SchemaSeal(0), // replaced below: the seal is computed
        reading: READING,
        effect: EffectClass::Pure,
        body: MethodBody::Query {
            filter: &[Cond::Eq {
                field: "billable",
                literal: "true",
            }],
            folds: &[Fold::Count, Fold::Sum("duration")],
            law: Law::Values,
        },
    };
    let compiled = MethodSpec {
        schema: SchemaSeal::of(&canonical_fields()),
        ..COMPILED
    };
    let decls = parse_decls(&fixture_text()).unwrap();
    for n in [1, 65, 4097] {
        let batch = batch(n);
        let cats = CatsBinder::new(&batch);
        let basis = canonical_fields();
        let claims = with_duration();
        let g = glove(&basis, &cats, &claims);
        let t = MethodTable::prepare_with(&g, &[("compiled_hours", compiled)], &decls);
        let loaded = t.get(t.resolve("billable_hours").unwrap()).unwrap();
        let built = t.get(t.resolve("compiled_hours").unwrap()).unwrap();
        let (loaded, built) = (loaded.as_ref().unwrap(), built.as_ref().unwrap());
        assert_eq!(loaded.programs(), built.programs(), "same lowered programs");
        assert_eq!(loaded.required, built.required);
        for f in [Frontend::Quack, Frontend::R2il] {
            assert_eq!(
                call(&t, &batch, "billable_hours", f),
                call(&t, &batch, "compiled_hours", f),
                "{f:?} n={n}"
            );
        }
    }
}

/// FAILS IF: loading a declaration upgrades a hypothesized claim. Against
/// SAP's real claims, `duration` is unproven, so every loaded method reading
/// it is refused; one reading only `billable` still prepares.
#[test]
fn loading_never_upgrades_a_hypothesized_claim() {
    let decls = parse_decls(&fixture_text()).unwrap();
    let batch = batch(65);
    let cats = CatsBinder::new(&batch);
    let basis = canonical_fields();
    let t = MethodTable::prepare_with(&glove(&basis, &cats, SAP_FIELDS), &[], &decls);
    for label in ["hours_by_billing", "billable_hours"] {
        assert_eq!(
            t.get(t.resolve(label).unwrap()),
            Some(&Err(PrepareError::UnmappedRequired("duration".into()))),
            "{label}"
        );
    }
    assert!(t
        .get(t.resolve("count_not_billed").unwrap())
        .unwrap()
        .is_ok());
    // The claim table is untouched.
    let dur = SAP_FIELDS
        .iter()
        .find(|m| m.canonical == "duration")
        .unwrap();
    assert_eq!(dur.grade, Grade::Hypothesized);
}

/// FAILS IF: a prepared method borrows the declaration text or the parsed
/// declarations: both are dropped before the call.
#[test]
fn a_prepared_method_outlives_its_declaration() {
    let batch = batch(65);
    let cats = CatsBinder::new(&batch);
    let basis = canonical_fields();
    let claims = with_duration();
    // A declaration built from runtime values: neither the label nor the
    // literal exists in the source.
    let label: String = ["count", "billed", "at"].join("_");
    let literal = if batch.len() > 1 { "true" } else { "false" };
    let text = format!(
        "billable_work_entry::{label} key={} schema={} reading=1 := COUNT(*) WHERE billable = {literal}\n",
        "7e".repeat(16),
        seal_hex(),
    );
    let t = {
        let decls = parse_decls(&text).unwrap();
        MethodTable::prepare_with(&glove(&basis, &cats, &claims), &[], &decls)
    };
    drop(text);
    let (code, _) = oracle(65, &batch);
    let want = scalars(&[code.iter().filter(|&&c| c == 1).count() as i64]);
    for f in [Frontend::Quack, Frontend::R2il] {
        assert_eq!(call(&t, &batch, &label, f), want);
    }
}

/// FAILS IF: the loader accepts a form outside its grammar instead of naming
/// it, or a malformed line instead of refusing it.
#[test]
fn unsupported_formulas_are_refused_by_name() {
    let head = format!(
        "billable_work_entry::m key={} schema={} reading=1 := ",
        "ab".repeat(16),
        seal_hex()
    );
    let unsupported = [
        ("COUNT(*) WHERE billable = true OR billable = false", "OR"),
        ("COUNT(*) WHERE NOT billable = true", "NOT"),
        ("COUNT(*) WHERE duration > 1", ">"),
        ("COUNT(*) WHERE duration < 1", "<"),
        ("AVG(duration)", "AVG"),
        ("MIN(duration)", "MIN"),
        ("SUM(duration * 2)", "*"),
        ("COUNT(WHERE is_billable())", "method reference"),
        ("SUM(duration WHERE is_billable())", "method reference"),
        ("COUNT(*) WHERE (billable = true)", "("),
        ("COUNT(*) WHERE billable IS NULL", "IS"),
        ("REVISE(statement)", "REVISE"),
    ];
    for (body, what) in unsupported {
        match parse_decls(&format!("{head}{body}")) {
            Err(DeclError::Unsupported { line: 1, what: w }) => {
                assert_eq!(w, what, "{body}");
            }
            other => panic!("{body}: {other:?}"),
        }
    }
    let malformed = [
        head.clone(),
        format!("{head}COUNT(*) WHERE"),
        format!("{head}COUNT(*) WHERE billable ="),
        format!("{head}SUM()"),
        format!("{head}COUNT(*),"),
        format!("{head}GROUP_SUM(duration BY billable IN 0)"),
        "billable_work_entry::m := COUNT(*)".to_owned(),
        format!(
            "billable_work_entry::m key=ab schema={} reading=1 := COUNT(*)",
            seal_hex()
        ),
        format!(
            "billable_work_entry::m key={} schema={} reading=x := COUNT(*)",
            "ab".repeat(16),
            seal_hex()
        ),
        format!(
            "billable_work_entry m key={} schema={} reading=1 := COUNT(*)",
            "ab".repeat(16),
            seal_hex()
        ),
    ];
    for text in &malformed {
        assert!(
            matches!(parse_decls(text), Err(DeclError::Syntax { line: 1, .. })),
            "{text}: {:?}",
            parse_decls(text)
        );
    }
    assert!(matches!(
        parse_decls(&format!(
            "no_such_concept::m key={} schema={} reading=1 := COUNT(*)",
            "ab".repeat(16),
            seal_hex()
        )),
        Err(DeclError::UnknownConcept { line: 1, .. })
    ));
    // Silence twin: comments and blank lines are not declarations, and a
    // later line's error names that line.
    let text = format!("# a comment\n\n{head}COUNT(*)\n{head}AVG(duration)\n");
    assert!(matches!(
        parse_decls(&text),
        Err(DeclError::Unsupported { line: 4, .. })
    ));
    assert_eq!(
        parse_decls(&format!("# c\n{head}COUNT(*)\n"))
            .unwrap()
            .len(),
        1
    );
}

/// FAILS IF: a loaded method can shadow a compiled one. Keys and labels are
/// checked across both, by the same rule as within one list.
#[test]
fn a_loaded_method_cannot_shadow_a_compiled_one() {
    let decls = parse_decls(&fixture_text()).unwrap();
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let basis = canonical_fields();
    let claims = with_duration();
    let key = decls[0].key;
    let clash = MethodSpec {
        key,
        concept: BILLABLE_WORK_ENTRY,
        schema: SchemaSeal::of(&basis),
        reading: READING,
        effect: EffectClass::Pure,
        body: MethodBody::Query {
            filter: &[],
            folds: &[Fold::Count],
            law: Law::Values,
        },
    };
    let t = MethodTable::prepare_with(&glove(&basis, &cats, &claims), &[("other", clash)], &decls);
    assert_eq!(t.get(key), Some(&Err(PrepareError::DuplicateKey(key))));
    let other = MethodSpec {
        key: MethodKey([0x01; 16]),
        ..clash
    };
    let t = MethodTable::prepare_with(
        &glove(&basis, &cats, &claims),
        &[("billable_hours", other)],
        &decls,
    );
    assert_eq!(
        t.resolve("billable_hours"),
        None,
        "an ambiguous label names nothing"
    );
    assert_eq!(
        t.get(MethodKey([0x01; 16])),
        Some(&Err(PrepareError::DuplicateLabel("billable_hours".into())))
    );
}
