//! Static source scanner — analyzes .rs files to build the function registry.
//!
//! Detects:
//! - todo!() / unimplemented!() / unreachable!() → DEAD
//! - Default::default() as sole return → STUB
//! - Empty function bodies → STUB
//! - f32::NAN / division patterns → NaN risk
//! - Normal functions with real logic → STATIC (alive at compile time)

use crate::diagnosis::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Scan configuration.
pub struct ScanConfig {
    /// Repository roots to scan: (name, path)
    pub repos: Vec<(String, PathBuf)>,
    /// Directories to skip (relative to repo root)
    pub skip_dirs: Vec<String>,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            repos: Vec::new(),
            skip_dirs: vec![
                "target".to_string(),
                ".git".to_string(),
                "benches".to_string(),
            ],
        }
    }
}

/// Scan all repos and produce a full stack diagnosis.
pub fn scan_stack(config: &ScanConfig) -> StackDiagnosis {
    let start = std::time::Instant::now();
    let mut repos = Vec::new();
    let mut total_functions = 0;
    let mut total_files = 0;
    let mut total_dead = 0;
    let mut total_stub = 0;
    let mut total_nan_risk = 0;

    for (repo_name, repo_path) in &config.repos {
        let repo_diag = scan_repo(repo_name, repo_path, &config.skip_dirs);
        total_functions += repo_diag.total_functions;
        total_dead += repo_diag.total_dead;
        total_stub += repo_diag.total_stub;
        total_nan_risk += repo_diag.total_nan_risk;
        total_files += repo_diag.modules.iter().map(|m| m.functions.len()).count();
        repos.push(repo_diag);
    }

    let health_pct = if total_functions > 0 {
        ((total_functions - total_dead - total_stub) as f32 / total_functions as f32) * 100.0
    } else {
        0.0
    };

    StackDiagnosis {
        repos,
        total_functions,
        total_files,
        total_dead,
        total_stub,
        total_nan_risk,
        health_pct,
        scan_duration_ms: start.elapsed().as_millis() as u64,
    }
}

fn scan_repo(name: &str, path: &Path, skip_dirs: &[String]) -> RepoDiagnosis {
    let mut modules_map: HashMap<String, Vec<FunctionMeta>> = HashMap::new();
    let mut total_functions = 0;

    for entry in WalkDir::new(path)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !skip_dirs.iter().any(|s| name == *s)
        })
        .filter_map(|e| e.ok())
    {
        let fpath = entry.path();
        if fpath.extension().map_or(true, |ext| ext != "rs") {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(fpath) {
            let rel_path = fpath
                .strip_prefix(path)
                .unwrap_or(fpath)
                .to_string_lossy()
                .to_string();

            // Determine module from directory
            let module = rel_path
                .split('/')
                .take_while(|s| *s != "mod.rs" && !s.ends_with(".rs"))
                .last()
                .unwrap_or("root")
                .to_string();

            let functions = extract_functions(&content, &rel_path, &module, name);
            total_functions += functions.len();

            modules_map.entry(module).or_default().extend(functions);
        }
    }

    let mut modules: Vec<ModuleDiagnosis> = modules_map
        .into_iter()
        .map(|(mod_name, functions)| {
            let total = functions.len();
            let dead = functions
                .iter()
                .filter(|f| f.state == NeuronState::Dead)
                .count();
            let stub = functions
                .iter()
                .filter(|f| f.state == NeuronState::Stub)
                .count();
            let nan_risk = functions.iter().filter(|f| f.has_nan_risk).count();
            let alive_or_static = total - dead - stub;
            let health_pct = if total > 0 {
                (alive_or_static as f32 / total as f32) * 100.0
            } else {
                100.0
            };

            ModuleDiagnosis {
                name: mod_name,
                repo: name.to_string(),
                path: String::new(), // filled below
                functions,
                total,
                alive_or_static,
                dead,
                stub,
                nan_risk,
                health_pct,
            }
        })
        .collect();

    modules.sort_by(|a, b| a.name.cmp(&b.name));

    let total_dead = modules.iter().map(|m| m.dead).sum();
    let total_stub = modules.iter().map(|m| m.stub).sum();
    let total_nan_risk = modules.iter().map(|m| m.nan_risk).sum();
    let health_pct = if total_functions > 0 {
        ((total_functions - total_dead - total_stub) as f32 / total_functions as f32) * 100.0
    } else {
        0.0
    };

    RepoDiagnosis {
        name: name.to_string(),
        modules,
        total_functions,
        total_dead,
        total_stub,
        total_nan_risk,
        health_pct,
    }
}

/// Extract public function signatures and analyze their bodies.
fn extract_functions(
    content: &str,
    file_path: &str,
    module: &str,
    repo: &str,
) -> Vec<FunctionMeta> {
    let mut functions = Vec::new();
    let fn_re = Regex::new(
        r"(?m)^\s*(?:pub(?:\(crate\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^\{]+?))?\s*\{"
    ).unwrap();

    let _lines: Vec<&str> = content.lines().collect();

    for cap in fn_re.captures_iter(content) {
        let fn_name = cap.get(1).unwrap().as_str().to_string();
        let params = cap.get(2).map_or("", |m| m.as_str()).trim().to_string();
        let return_type = cap.get(3).map_or("()", |m| m.as_str()).trim().to_string();

        // Find the line number
        let match_start = cap.get(0).unwrap().start();
        let line_num = content[..match_start].lines().count() + 1;

        // Extract function body (find matching brace)
        let body_start = cap.get(0).unwrap().end();
        let body = extract_body(content, body_start);
        let body_loc = body.lines().count();

        // Analyze the body
        let has_todo = body.contains("todo!()") || body.contains("todo!(\"");
        let has_unimplemented =
            body.contains("unimplemented!()") || body.contains("unimplemented!(\"");
        let has_panic = body.contains("panic!(") && !body.contains("catch_unwind");

        let is_stub = detect_stub(&body, &return_type);
        let has_nan_risk = detect_nan_risk(&body, &return_type);

        let state = if has_todo || has_unimplemented {
            NeuronState::Dead
        } else if has_panic && body_loc <= 5 {
            NeuronState::Dead
        } else if is_stub {
            NeuronState::Stub
        } else {
            NeuronState::Static // compile-time: we know it exists, runtime tells us if it's called
        };

        // Build a simple signature
        let sig = if params.len() > 50 {
            format!("fn {}(...) -> {}", fn_name, return_type)
        } else {
            format!("fn {}({}) -> {}", fn_name, params, return_type)
        };

        let id = format!("{}::{}::{}", repo, module, fn_name);

        functions.push(FunctionMeta {
            id,
            file: file_path.to_string(),
            line: line_num,
            module: module.to_string(),
            repo: repo.to_string(),
            signature: sig,
            has_todo,
            has_unimplemented,
            has_panic,
            is_stub,
            has_nan_risk,
            state,
            body_loc,
        });
    }

    functions
}

/// Extract function body by matching braces from the opening '{'.
fn extract_body(content: &str, start: usize) -> String {
    let bytes = content.as_bytes();
    let mut depth = 1;
    let mut i = start;
    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                // Skip line comment
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'"' => {
                // Skip string literal
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    if start < i && i <= content.len() {
        content[start..i.saturating_sub(1)].to_string()
    } else {
        String::new()
    }
}

/// Detect if a function is a stub (returns default/zero/empty).
fn detect_stub(body: &str, _return_type: &str) -> bool {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return true;
    }
    // Single-expression stubs
    let patterns = [
        "Default::default()",
        "Self::default()",
        "Vec::new()",
        "HashMap::new()",
        "String::new()",
        "Ok(())",
        "0",
        "0.0",
        "false",
        "None",
        "\"\"",
    ];
    // Check if body is just one of these patterns (with optional Ok() wrapper)
    let body_lines: Vec<&str> = trimmed
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with("//"))
        .collect();
    if body_lines.len() == 1 {
        let line = body_lines[0].trim_end_matches(';');
        for pat in &patterns {
            if line == *pat || line == format!("Ok({})", pat) {
                return true;
            }
        }
    }
    false
}

/// Detect if a function has NaN risk (returns float with division).
fn detect_nan_risk(body: &str, return_type: &str) -> bool {
    let returns_float = return_type.contains("f32")
        || return_type.contains("f64")
        || return_type.contains("Float")
        || return_type.contains("Score");

    if !returns_float {
        return false;
    }

    // Check for division without NaN guards
    let has_division = body.contains(" / ") || body.contains(" /=");
    let has_nan_literal = body.contains("f32::NAN")
        || body.contains("f64::NAN")
        || body.contains("NaN")
        || body.contains("INFINITY");
    let has_nan_guard =
        body.contains("is_nan()") || body.contains("is_finite()") || body.contains("is_infinite()");

    (has_division && !has_nan_guard) || has_nan_literal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_todo_as_dead() {
        let content = r#"
pub fn broken() -> i32 {
    todo!()
}
"#;
        let fns = extract_functions(content, "test.rs", "test", "test_repo");
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].state, NeuronState::Dead);
        assert!(fns[0].has_todo);
    }

    #[test]
    fn detects_stub() {
        let content = r#"
pub fn stub_fn() -> Vec<i32> {
    Vec::new()
}
"#;
        let fns = extract_functions(content, "test.rs", "test", "test_repo");
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].state, NeuronState::Stub);
    }

    #[test]
    fn detects_nan_risk() {
        let content = r#"
pub fn risky(a: f32, b: f32) -> f32 {
    a / b
}
"#;
        let fns = extract_functions(content, "test.rs", "test", "test_repo");
        assert_eq!(fns.len(), 1);
        assert!(fns[0].has_nan_risk);
    }

    #[test]
    fn normal_function_is_static() {
        let content = r#"
pub fn healthy(x: i32) -> i32 {
    let y = x * 2;
    y + 1
}
"#;
        let fns = extract_functions(content, "test.rs", "test", "test_repo");
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].state, NeuronState::Static);
    }
}
