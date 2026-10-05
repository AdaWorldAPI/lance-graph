//! The two-world fence: the executable IR holds no text.
//!
//! Text may appear only in `bind.rs`, the developer-world frontend. Every
//! other line of the crate's non-test source — the `Query` / `Filter` /
//! `Cmp` / `Agg` vocabulary and its lowering — must be free of text types,
//! so a bound query cannot carry a name, a label or a literal string into
//! `lower` or the executor.

const FORBIDDEN: &[&str] = &["String", "&str", "str::", "format!", "to_string", "char"];

fn violations(src: &str) -> Vec<String> {
    src.lines()
        .enumerate()
        .take_while(|(_, l)| l.trim() != "#[cfg(test)]")
        .filter(|(_, l)| !l.trim_start().starts_with("//"))
        .filter(|(_, l)| FORBIDDEN.iter().any(|t| l.contains(t)))
        .map(|(i, l)| format!("{}: {}", i + 1, l.trim()))
        .collect()
}

#[test]
fn the_executable_ir_is_string_free() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs")).unwrap();
    let bad = violations(&src);
    assert!(
        bad.is_empty(),
        "text in the executable IR:\n{}",
        bad.join("\n")
    );
    // Anti-vacuity: the scan covered the IR, not an empty prefix.
    let scanned = src
        .lines()
        .take_while(|l| l.trim() != "#[cfg(test)]")
        .count();
    assert!(scanned > 1_000, "only {scanned} lines scanned");
}

#[test]
fn the_only_text_module_is_bind() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    let mut modules: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    modules.sort();
    assert_eq!(
        modules,
        ["bind.rs", "lib.rs"],
        "a new module must be fenced"
    );
}

#[test]
fn the_fence_can_fire() {
    assert_eq!(violations("    Label(String),").len(), 1);
    assert_eq!(violations("    name: &str,").len(), 1);
    assert!(violations("    // a String is presentation").is_empty());
    assert!(violations("#[cfg(test)]\n    Label(String),").is_empty());
}
