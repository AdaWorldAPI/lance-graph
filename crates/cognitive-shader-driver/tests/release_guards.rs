//! A public function must not rely on `debug_assert!` to reject bad input.
//!
//! `debug_assert!` is compiled out of release builds. When it is the only
//! check on a public argument, a release build silently accepts the bad value.
//! `CartesianAddress12::new` did exactly this (#1337, fixed in #1340):
//! `new(8, 0, 0, 0)` packed into the neighbouring field and read back as
//! `[0, 0, 0, 0]`.
//!
//! This test scans the crate's sources and fails on any `debug_assert!` inside
//! a `pub fn` body. An assertion that checks the crate's own invariant rather
//! than caller input belongs in `ALLOWED`, with the reason written next to it.
//! The scan is a plain text pass, not a parser: it finds `pub fn` signatures,
//! follows braces to the end of the body, and stops at `#[cfg(test)]`.

use std::path::Path;

/// `(file, function, why the assertion is not input validation)`.
const ALLOWED: &[(&str, &str, &str)] = &[(
    "wire.rs",
    "decode",
    "asserts the alignment of a buffer this function just allocated; no caller input",
)];

/// One `debug_assert!` found inside a `pub fn` body.
#[derive(Debug, PartialEq, Eq)]
struct Hit {
    function: String,
    line: usize,
}

/// Find every `debug_assert!` inside a `pub fn` body of `src`.
fn scan(src: &str) -> Vec<Hit> {
    let mut hits = Vec::new();
    let lines: Vec<&str> = src.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim_start();
        if line.starts_with("#[cfg(test)]") {
            break;
        }
        if let Some(name) = pub_fn_name(line) {
            // Walk to the end of the body by brace depth, starting at the
            // signature line (the opening brace may be on a later line).
            let mut depth = 0i32;
            let mut opened = false;
            let mut j = i;
            while j < lines.len() {
                let l = lines[j];
                // Inside the body, or after the opening brace on this line.
                let in_body = match (l.find('{'), l.find("debug_assert")) {
                    (_, None) => false,
                    (Some(b), Some(d)) => opened || b < d,
                    (None, Some(_)) => opened,
                };
                if in_body {
                    hits.push(Hit {
                        function: name.clone(),
                        line: j + 1,
                    });
                }
                for c in l.chars() {
                    match c {
                        '{' => {
                            depth += 1;
                            opened = true;
                        }
                        '}' => depth -= 1,
                        _ => {}
                    }
                }
                if opened && depth <= 0 {
                    break;
                }
                // A declaration with no body (trait method) ends at `;`.
                if !opened && l.trim_end().ends_with(';') {
                    break;
                }
                j += 1;
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    hits
}

/// The function name if `line` opens a `pub` (not `pub(crate)`) fn.
fn pub_fn_name(line: &str) -> Option<String> {
    let rest = line.strip_prefix("pub ")?;
    let rest = ["const ", "unsafe ", "async ", "extern \"C\" "]
        .iter()
        .fold(rest, |r, q| r.strip_prefix(q).unwrap_or(r));
    let rest = rest.strip_prefix("fn ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable src dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_public_function_validates_input_with_debug_assert_alone() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    assert!(
        files.len() > 10,
        "scan must actually see the crate's sources"
    );

    let mut violations = Vec::new();
    let mut allowed_seen = Vec::new();
    for path in &files {
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(path).expect("readable source");
        for hit in scan(&text) {
            match ALLOWED
                .iter()
                .find(|(f, func, _)| *f == file && *func == hit.function)
            {
                Some(entry) => allowed_seen.push(*entry),
                None => violations.push(format!("{file}:{} in pub fn {}", hit.line, hit.function)),
            }
        }
    }

    assert!(
        violations.is_empty(),
        "debug_assert! is the only guard in a public function (compiled out in release):\n{}",
        violations.join("\n")
    );
    // An allowlist entry that no longer matches anything is stale.
    for entry in ALLOWED {
        assert!(
            allowed_seen.contains(entry),
            "stale ALLOWED entry {entry:?}: remove it"
        );
    }
}

/// Can-fire: the shape of the #1337 bug is reported.
#[test]
fn the_scan_reports_a_debug_assert_guarded_public_constructor() {
    let src = "impl A {\n    pub const fn new(a: u8) -> Self {\n        debug_assert!(a < 8);\n        Self(a)\n    }\n}\n";
    assert_eq!(
        scan(src),
        vec![Hit {
            function: "new".into(),
            line: 3
        }]
    );
}

/// Can-fire on a one-line body, where the brace and the assertion share a line.
#[test]
fn the_scan_reports_a_one_line_public_function() {
    let src = "pub fn f(a: u8) { debug_assert!(a < 8); }\n";
    assert_eq!(
        scan(src),
        vec![Hit {
            function: "f".into(),
            line: 1
        }]
    );
}

/// Can-stay-silent: private functions, `pub(crate)`, real checks and test
/// modules are not reported.
#[test]
fn the_scan_ignores_private_functions_real_checks_and_tests() {
    let src = "\
fn private(a: u8) { debug_assert!(a < 8); }
pub(crate) fn internal(a: u8) { debug_assert!(a < 8); }
pub fn checked(a: u8) -> Option<u8> {
    if a < 8 { Some(a) } else { None }
}
#[cfg(test)]
mod tests {
    pub fn helper(a: u8) { debug_assert!(a < 8); }
}
";
    assert_eq!(scan(src), vec![]);
}
