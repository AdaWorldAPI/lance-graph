//! `Effects`: the real foundry_parity_lab file and synthetic refusal cases.

use std::ops::Deref;
use std::path::{Path, PathBuf};

use lance_graph_codegraph::effects::{
    EffectAnswer, EffectClass, EffectKind, EffectSite, Effects, EffectsRefusal, ReachGrade,
    ReachLimit,
};
use lance_graph_codegraph::Codegraph;

const HEAD: &str = "#version 1\n#schema subject\teffect\tkind\tfile\tstart\tend\n";

fn real() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/effects")
}

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Tmp {
        let p = std::env::temp_dir().join(format!("cg_effects_{tag}_{}", std::process::id()));
        if p.exists() {
            std::fs::remove_dir_all(&p).unwrap();
        }
        std::fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }
}

impl Deref for Tmp {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write(dir: &Path, krate: &str, body: &str) {
    std::fs::write(dir.join(format!("{krate}.effects.tsv")), body).unwrap();
}

const GUARD: &str = "foundry_parity_lab::method::can_cook_guard";

#[test]
fn the_real_file_reads_every_row() {
    let e = Effects::open(&real()).unwrap();
    assert_eq!(e.rows().len(), 213);
    assert_eq!(e.crates().collect::<Vec<_>>(), ["foundry_parity_lab"]);
}

#[test]
fn division_sites_are_found_by_kind() {
    let e = Effects::open(&real()).unwrap();
    let div = EffectKind::Assert("DivisionByZero".into());
    let got: Vec<(&str, &str, u32, u32)> = e
        .with_kind(&div)
        .map(|r| {
            let s = r.site.as_ref().unwrap();
            (r.subject.as_str(), s.file.as_str(), s.start, s.end)
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (GUARD, "src/method.rs", 2699, 2734),
            (
                "foundry_parity_lab::arm_a::{Objects}::can_cook",
                "src/arm_a.rs",
                2572,
                2577
            ),
            (
                "foundry_parity_lab::arm_b::{ItemQuery<'s>}::cook_ops",
                "src/arm_b.rs",
                7745,
                7780
            ),
            (
                "foundry_parity_lab::arm_b::{ItemQuery<'s>}::cook_ops",
                "src/arm_b.rs",
                7745,
                7780
            ),
            (
                "foundry_parity_lab::oracle::{Oracle}::can_cook",
                "src/oracle.rs",
                2063,
                2098
            ),
        ]
    );
    assert_eq!(
        e.with_kind(&EffectKind::Assert("NoSuchKind".into()))
            .count(),
        0
    );
    assert_eq!(e.with_kind(&EffectKind::DerefWriteArg).count(), 3);
}

#[test]
fn a_function_with_rows_is_witnessed() {
    let e = Effects::open(&real()).unwrap();
    match e.of(GUARD) {
        EffectAnswer::Witnessed(v) => {
            // 11 rows in the fixture carry this subject (counted with awk on column 1).
            assert_eq!(v.len(), 11);
            assert!(v
                .iter()
                .any(|r| r.kind == EffectKind::Assert("DivisionByZero".into())));
        }
        other => panic!("expected Witnessed, got {other:?}"),
    }
}

#[test]
fn no_row_is_not_a_purity_verdict() {
    let e = Effects::open(&real()).unwrap();
    assert_eq!(
        e.of("foundry_parity_lab::no_such_function"),
        EffectAnswer::NoneWitnessed
    );
}

#[test]
fn an_unread_crate_is_not_looked_at() {
    let e = Effects::open(&real()).unwrap();
    assert_eq!(e.of("other_crate::f"), EffectAnswer::CrateNotRead);

    let tmp = Tmp::new("empty");
    let empty = Effects::open(&tmp).unwrap();
    assert_eq!(empty.rows().len(), 0);
    assert_eq!(empty.of(GUARD), EffectAnswer::CrateNotRead);
}

#[test]
fn two_crates_are_read_in_name_order() {
    let tmp = Tmp::new("two");
    write(
        &tmp,
        "b",
        &format!("{HEAD}b::f\tASSERT\tUser\tsrc/lib.rs\t1\t2\n"),
    );
    write(
        &tmp,
        "a",
        &format!("{HEAD}a::g\tDEREF_WRITE\tLocal\t-\t-\t-\n"),
    );
    let e = Effects::open(&tmp).unwrap();
    assert_eq!(e.crates().collect::<Vec<_>>(), ["a", "b"]);
    assert_eq!(e.rows().len(), 2);
    assert_eq!(e.rows()[0].subject, "a::g");
    assert_eq!(e.rows()[0].site, None);
    assert_eq!(e.rows()[0].kind, EffectKind::DerefWriteLocal);
    assert_eq!(e.rows()[1].kind, EffectKind::Assert("User".into()));
    assert_eq!(
        e.rows()[1].site,
        Some(EffectSite {
            file: "src/lib.rs".into(),
            start: 1,
            end: 2
        })
    );
}

fn refusal(tag: &str, body: &str) -> EffectsRefusal {
    let tmp = Tmp::new(tag);
    write(&tmp, "k", body);
    Effects::open(&tmp).unwrap_err()
}

#[test]
fn a_wrong_version_is_refused() {
    assert_eq!(
        refusal(
            "ver2",
            "#version 2\n#schema subject\teffect\tkind\tfile\tstart\tend\n"
        ),
        EffectsRefusal::Version {
            krate: "k".into(),
            found: 2
        }
    );
    assert_eq!(
        refusal("ver0", "#schema subject\teffect\tkind\tfile\tstart\tend\n"),
        EffectsRefusal::Version {
            krate: "k".into(),
            found: 0
        }
    );
}

#[test]
fn a_wrong_schema_is_refused() {
    assert_eq!(
        refusal("schema", "#version 1\n#schema subject\teffect\tkind\n"),
        EffectsRefusal::Schema { krate: "k".into() }
    );
}

#[test]
fn a_short_or_long_row_is_refused() {
    assert_eq!(
        refusal(
            "short",
            &format!("{HEAD}k::f\tASSERT\tUser\tsrc/lib.rs\t1\n")
        ),
        EffectsRefusal::Arity {
            krate: "k".into(),
            line: 3
        }
    );
    assert_eq!(
        refusal(
            "long",
            &format!("{HEAD}k::f\tASSERT\tUser\tsrc/lib.rs\t1\t2\textra\n")
        ),
        EffectsRefusal::Arity {
            krate: "k".into(),
            line: 3
        }
    );
    assert_eq!(
        refusal(
            "later",
            &format!(
                "{HEAD}k::f\tASSERT\tUser\tsrc/lib.rs\t1\t2\nk::f\tASSERT\tUser\tsrc/lib.rs\t1\n"
            )
        ),
        EffectsRefusal::Arity {
            krate: "k".into(),
            line: 4
        }
    );
}

#[test]
fn an_unknown_effect_is_refused() {
    let want = EffectsRefusal::UnknownEffect {
        krate: "k".into(),
        line: 3,
    };
    for (tag, row) in [
        ("e1", "k::f\tPANIC\tUser\t-\t-\t-\n"),
        ("e2", "k::f\tDEREF_WRITE\tGlobal\t-\t-\t-\n"),
        ("e3", "k::f\tASSERT\t\t-\t-\t-\n"),
    ] {
        assert_eq!(refusal(tag, &format!("{HEAD}{row}")), want, "{row:?}");
    }
}

#[test]
fn a_malformed_site_is_refused() {
    let want = EffectsRefusal::Site {
        krate: "k".into(),
        line: 3,
    };
    for (i, site) in [
        "src/lib.rs\t5\t2",
        "src/lib.rs\tx\t2",
        "-\t1\t2",
        "src/lib.rs\t-\t-",
        "-\t-\t3",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            refusal(
                &format!("site{i}"),
                &format!("{HEAD}k::f\tASSERT\tUser\t{site}\n")
            ),
            want,
            "{site:?}"
        );
    }

    let tmp = Tmp::new("site_ok");
    write(
        &tmp,
        "k",
        &format!("{HEAD}k::f\tASSERT\tUser\tsrc/lib.rs\t2\t2\n"),
    );
    let e = Effects::open(&tmp).unwrap();
    assert_eq!(
        e.rows()[0].site,
        Some(EffectSite {
            file: "src/lib.rs".into(),
            start: 2,
            end: 2
        })
    );
}

#[test]
fn duplicate_rows_are_kept() {
    let tmp = Tmp::new("dup");
    let row = "k::f\tASSERT\tUser\tsrc/lib.rs\t1\t2\n";
    write(&tmp, "k", &format!("{HEAD}{row}{row}"));
    let e = Effects::open(&tmp).unwrap();
    assert_eq!(e.rows().len(), 2);
}

// --- effects joined with the call rows ---

const DIV: &str = "DivisionByZero";

fn div(k: &EffectKind) -> bool {
    *k == EffectKind::Assert(DIV.into())
}

fn reach(dir: &Path, symbol: &str) -> Option<Vec<(String, ReachGrade)>> {
    let g = Codegraph::open(dir).unwrap();
    let fx = Effects::open(dir).unwrap();
    let r = g.effects_reached(symbol, &fx, div).unwrap()?;
    Some(
        r.hits
            .iter()
            .map(|h| (h.effect.subject.clone(), h.grade))
            .collect(),
    )
}

#[test]
fn a_caller_reaches_its_callees_division() {
    let cook_ops = "foundry_parity_lab::arm_b::{ItemQuery<'s>}::cook_ops";
    // max_quantities calls cook_ops; the division is in cook_ops (two rows, one site).
    assert_eq!(
        reach(
            &real(),
            "foundry_parity_lab::arm_b::{ItemQuery<'s>}::max_quantities"
        ),
        Some(vec![
            (cook_ops.to_string(), ReachGrade::Confirmed),
            (cook_ops.to_string(), ReachGrade::Confirmed)
        ])
    );
    assert_eq!(
        reach(&real(), cook_ops),
        Some(vec![
            (cook_ops.to_string(), ReachGrade::Own),
            (cook_ops.to_string(), ReachGrade::Own)
        ])
    );
    // A function that reaches no division body.
    assert_eq!(
        reach(&real(), "foundry_parity_lab::fixture::recipe"),
        Some(vec![])
    );
    // A symbol without a position has no answer, which is not an empty one.
    assert_eq!(reach(&real(), "foundry_parity_lab::no_such_function"), None);
}

#[test]
fn classes_select_division_remainder_shift_and_write() {
    let fx = Effects::open(&real()).unwrap();
    let count = |c: EffectClass| {
        fx.rows()
            .iter()
            .filter(|e| e.kind.class() == Some(c))
            .count()
    };
    // Counted from the fixture: DivisionByZero 5 + Overflow:Div 5; RemainderByZero 11 +
    // Overflow:Rem 3; Overflow:Shr 7 + Overflow:Shl 5; DEREF_WRITE Local 11 + Arg 3.
    assert_eq!(count(EffectClass::Division), 10);
    assert_eq!(count(EffectClass::Remainder), 14);
    assert_eq!(count(EffectClass::Shift), 12);
    assert_eq!(count(EffectClass::Write), 14);
    // Bounds checks and add/sub/mul overflow have no class: 213 - 50.
    assert_eq!(
        fx.rows()
            .iter()
            .filter(|e| e.kind.class().is_none())
            .count(),
        163
    );
    // Reach by class: max_quantities reaches cook_ops's division rows of both kinds.
    let g = Codegraph::open(&real()).unwrap();
    let r = g
        .effects_reached(
            "foundry_parity_lab::arm_b::{ItemQuery<'s>}::max_quantities",
            &fx,
            |k| k.class() == Some(EffectClass::Division),
        )
        .unwrap()
        .unwrap();
    let kinds: Vec<&str> = r.hits.iter().map(|h| h.effect.kind.columns().1).collect();
    assert!(
        kinds.contains(&"DivisionByZero") && kinds.contains(&"Overflow:Div"),
        "{kinds:?}"
    );
    assert!(r
        .hits
        .iter()
        .all(|h| h.effect.kind.class() == Some(EffectClass::Division)));
}

#[test]
fn the_effects_file_is_identified_by_its_bytes() {
    let fx = Effects::open(&real()).unwrap();
    let bytes = std::fs::read(real().join("foundry_parity_lab.effects.tsv")).unwrap();
    let id = fx.identity("foundry_parity_lab").unwrap();
    assert_eq!(id.rows, 213);
    assert_eq!(id.fnv64, lance_graph_contract::hash::fnv1a(&bytes));
    assert_eq!(fx.identity("other_crate"), None);
    // The identity is of the bytes, not of the rows.
    let dir = Tmp::new("identity");
    let mut edited = bytes.clone();
    // A trailing blank line: the same rows, other bytes.
    edited.extend_from_slice(b"\n");
    std::fs::write(dir.join("foundry_parity_lab.effects.tsv"), &edited).unwrap();
    let other = Effects::open(&dir).unwrap();
    let other_id = other.identity("foundry_parity_lab").unwrap();
    assert_eq!(other_id.rows, 213);
    assert_ne!(other_id.fnv64, id.fnv64);
}

/// The rev1 call rows, with one division effect placed in `subject` when given.
fn rev1_with_effect(tag: &str, subject: Option<&str>) -> Tmp {
    let dir = Tmp::new(tag);
    let rev1 = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/rev1");
    for e in std::fs::read_dir(&rev1).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), dir.join(e.file_name())).unwrap();
    }
    if let Some(subject) = subject {
        write(
            &dir,
            "cg_lib",
            &format!("{HEAD}{subject}\tASSERT\t{DIV}\tsrc/util.rs\t1\t2\n"),
        );
    }
    dir
}

#[test]
fn the_reach_grade_follows_the_call_rows() {
    let dir = rev1_with_effect("grades", Some("cg_lib::util::double"));
    let one = |symbol: &str| reach(&dir, symbol).unwrap();
    let hit = |g| vec![("cg_lib::util::double".to_string(), g)];
    assert_eq!(one("cg_lib::util::double"), hit(ReachGrade::Own));
    // run -> Square::area -> double, all resolved.
    assert_eq!(one("cg_app::run"), hit(ReachGrade::Confirmed));
    // area_of reaches Square::area only through trait dispatch.
    assert_eq!(one("cg_lib::area_of"), hit(ReachGrade::Possible));
    // area_dyn holds the dynamic call.
    assert_eq!(one("cg_lib::area_dyn"), hit(ReachGrade::Unknown));
    // triple does not reach double at all.
    assert_eq!(one("cg_lib::util::triple"), vec![]);
}

fn limits(dir: &Path, symbol: &str) -> Vec<ReachLimit> {
    let g = Codegraph::open(dir).unwrap();
    let fx = Effects::open(dir).unwrap();
    g.effects_reached(symbol, &fx, div).unwrap().unwrap().limits
}

#[test]
fn a_missing_effects_file_makes_the_answer_partial_never_clean() {
    let not_read = |k: &str| ReachLimit::EffectsNotRead { krate: k.into() };
    // No effects file at all: nothing is reached, and both crates are named as unread.
    let none = rev1_with_effect("no_effects", None);
    let g = Codegraph::open(&none).unwrap();
    let fx = Effects::open(&none).unwrap();
    let r = g.effects_reached("cg_app::run", &fx, div).unwrap().unwrap();
    assert!(r.hits.is_empty());
    assert!(r.partial());
    assert!(r.limits.contains(&not_read("cg_lib")) && r.limits.contains(&not_read("cg_app")));
    // cg_lib's file present: only cg_app is still unread.
    let one = rev1_with_effect("one_effects", Some("cg_lib::util::double"));
    let l = limits(&one, "cg_app::run");
    assert!(
        !l.contains(&not_read("cg_lib")) && l.contains(&not_read("cg_app")),
        "{l:?}"
    );
    // Both present: no crate is unread.
    write(&one, "cg_app", HEAD);
    let l = limits(&one, "cg_app::run");
    assert!(
        !l.iter()
            .any(|x| matches!(x, ReachLimit::EffectsNotRead { .. })),
        "{l:?}"
    );
}

#[test]
fn dynamic_calls_and_an_incomplete_harvest_are_limits() {
    let dir = rev1_with_effect("limits", Some("cg_lib::util::double"));
    write(&dir, "cg_app", HEAD);
    // rev1 holds one dynamic call, in cg_lib::area_dyn, which cg_app::run calls. Both harvests
    // are clean and both effects files are read, so that is the only limit.
    assert_eq!(
        limits(&dir, "cg_app::run"),
        vec![ReachLimit::DynamicCalls {
            store_dynamic_calls: 1
        }]
    );
    // triple reaches no dynamic call and nothing outside the store: no limit is known.
    assert_eq!(limits(&dir, "cg_lib::util::triple"), vec![]);
    // Without its residuals file, what cg_lib's harvest missed is unknown.
    std::fs::remove_file(dir.join("cg_lib.residuals.tsv")).unwrap();
    assert!(
        limits(&dir, "cg_app::run").contains(&ReachLimit::HarvestIncomplete {
            krate: "cg_lib".into()
        })
    );
}

#[test]
fn an_effect_subject_without_a_position_is_a_limit() {
    let dir = rev1_with_effect("unplaced", Some("cg_lib::not_in_the_rows"));
    let g = Codegraph::open(&dir).unwrap();
    let fx = Effects::open(&dir).unwrap();
    let r = g.effects_reached("cg_app::run", &fx, div).unwrap().unwrap();
    assert!(r.hits.is_empty());
    assert!(r.limits.contains(&ReachLimit::Unplaced {
        subject: "cg_lib::not_in_the_rows".into()
    }));
}

#[test]
fn callees_outside_the_store_are_a_limit() {
    let outside = |crates: &[&str]| ReachLimit::CalleesOutsideStore {
        crates: crates.iter().map(|c| c.to_string()).collect(),
    };
    // recipe calls into core and alloc only; their bodies are not in the store.
    assert_eq!(
        limits(&real(), "foundry_parity_lab::fixture::recipe"),
        vec![outside(&["alloc", "core"])]
    );
    // max_quantities also reaches the mask-risc executor, which was not extracted.
    let l = limits(
        &real(),
        "foundry_parity_lab::arm_b::{ItemQuery<'s>}::max_quantities",
    );
    assert!(
        l.contains(&outside(&["alloc", "core", "lance_graph_mask_risc"])),
        "{l:?}"
    );
}
