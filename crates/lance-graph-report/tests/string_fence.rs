//! S1–S4 / S10 source fence: the execution modules contain no text types.
//!
//! Strings are presentation metadata, not execution coordinates. Text may
//! appear only in `boundary.rs` (ingest / KV / CAM / catalog), `render.rs`
//! (terminal) and `explain.rs` (diagnostics). This walks the execution
//! modules' NON-COMMENT lines and rejects any text-type token.

const EXECUTION: &[&str] = &[
    "ids.rs",
    "batch.rs",
    "selection.rs",
    "plan.rs",
    "exec.rs",
    "result.rs",
];
const FORBIDDEN: &[&str] = &[
    "String",
    "&str",
    "str::",
    "format!",
    "to_string",
    "HashMap<String",
    "char",
];

fn violations(src: &str) -> Vec<String> {
    src.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with("//"))
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
        "text types in the execution core:\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_fence_can_fire() {
    // Disable-verified shape: a line an earlier draft of this crate really
    // had (`labels: Option<Arc<[String]>>` on a dimension) is caught, and a
    // comment mentioning String is not.
    assert_eq!(
        violations("    pub labels: Option<Arc<[String]>>,").len(),
        1
    );
    assert_eq!(violations("    Label(String),").len(), 1);
    assert!(violations("    /// a String is presentation metadata").is_empty());
    assert!(violations("    pub field: FieldId,").is_empty());
}
