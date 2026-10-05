//! Source fence: the execution modules contain no text types. Text belongs
//! to ingress and egress — `snapshot.rs` (observation, the label/value
//! table), `observe.rs` (records) and `store.rs` (tags, `intern`/`value`).
//! Same shape as `lance-graph-report/tests/string_fence.rs`.

const EXECUTION: &[&str] = &["exec.rs", "view.rs", "validate.rs", "rule.rs", "lib.rs"];
const FORBIDDEN: &[&str] = &[
    "String",
    "&str",
    "str::",
    "format!",
    "to_string",
    "char",
    "normalize",
];

fn violations(src: &str) -> Vec<String> {
    src.lines()
        .enumerate()
        .filter(|(_, l)| {
            let t = l.trim_start();
            !t.starts_with("//") && !t.starts_with("///") && !t.starts_with("//!")
        })
        .filter(|(_, l)| FORBIDDEN.iter().any(|t| l.contains(t)))
        .map(|(i, l)| format!("{}: {}", i + 1, l.trim()))
        .collect()
}

#[test]
fn execution_modules_are_string_free() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/");
    let mut bad = Vec::new();
    for f in EXECUTION {
        let src = std::fs::read_to_string(format!("{dir}{f}")).unwrap();
        bad.extend(violations(&src).into_iter().map(|v| format!("{f}:{v}")));
    }
    assert!(
        bad.is_empty(),
        "text in the execution modules:\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_fence_can_fire() {
    assert_eq!(violations("    pub to: String,").len(), 1);
    assert_eq!(violations("        let k = normalize(s);").len(), 1);
    assert!(violations("    /// a String is presentation").is_empty());
    assert!(violations("    pub to: ValueId,").is_empty());
}
