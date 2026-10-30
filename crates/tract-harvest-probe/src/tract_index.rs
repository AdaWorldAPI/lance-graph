//! An index of the ONNX ops a frontend's source registers, read from that
//! source as text (Tract's `onnx/src`, where each op is a
//! `reg.insert("<Op>", <parser>)`).
//!
//! For every op the index records where it is registered, its parser (a named
//! function or an inline closure), and the attributes the parser reads,
//! with the default it applies (`get_attr_opt("axis")?.unwrap_or(0)` gives
//! `axis = 0`). A parser that branches on the opset version is marked
//! [`OpEntry::opset_dependent`]: its defaults can differ by opset, so a
//! generator must not pick one.
//!
//! This reads source text; it never runs the frontend. What it finds is a
//! candidate for a registration, checked against the graph's own attributes
//! before any field is built from it.

use std::collections::BTreeMap;

/// One attribute a parser reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrRead {
    pub name: String,
    /// The `unwrap_or` default as written, if any.
    pub default: Option<String>,
    /// `get_attr` (required) rather than `get_attr_opt` / `get_attr_vec`.
    pub required: bool,
}

/// One registered op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpEntry {
    pub file: String,
    pub line: usize,
    /// The parser function's name, or `None` for an inline closure.
    pub parser: Option<String>,
    pub attrs: Vec<AttrRead>,
    pub opset_dependent: bool,
}

/// `reg.insert("Op", rest` → `(Op, rest of the statement up to its `;`)`.
fn inserts(text: &str) -> Vec<(usize, String, String)> {
    let mut out = Vec::new();
    let mut rest = text;
    let mut offset = 0;
    while let Some(i) = rest.find("reg.insert(\"") {
        let start = offset + i;
        let after = &text[start + "reg.insert(\"".len()..];
        let Some(q) = after.find('"') else { break };
        let op = after[..q].to_string();
        let tail = &after[q + 1..];
        // The statement ends at the first `;` outside parentheses.
        let mut depth = 1i32;
        let mut end = tail.len();
        for (k, c) in tail.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                ';' if depth <= 0 => {
                    end = k;
                    break;
                }
                _ => {}
            }
        }
        let line = text[..start].matches('\n').count() + 1;
        out.push((line, op, tail[..end].to_string()));
        offset = start + 1;
        rest = &text[offset..];
    }
    out
}

/// The body of `fn <name>(` in `text`, from the `fn` to its matching `}`.
fn fn_body<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let pat = format!("fn {name}(");
    let s = text.find(&pat)?;
    let open = s + text[s..].find('{')?;
    let mut depth = 0i32;
    for (k, c) in text[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[s..open + k + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every `get_attr*("name")` in `code`, with its `unwrap_or` default.
fn attr_reads(code: &str) -> Vec<AttrRead> {
    let mut out: Vec<AttrRead> = Vec::new();
    let mut rest = code;
    while let Some(i) = rest.find("get_attr") {
        let tail = &rest[i + "get_attr".len()..];
        let required = !tail.starts_with("_opt") && !tail.starts_with("_vec");
        let Some(q) = tail.find("(\"") else {
            rest = tail;
            continue;
        };
        let after = &tail[q + 2..];
        let Some(e) = after.find('"') else { break };
        let name = after[..e].to_string();
        let near = &after[e..after.len().min(e + 80)];
        let default = near.find(".unwrap_or(").and_then(|u| {
            let d = &near[u + ".unwrap_or(".len()..];
            d.find(')').map(|c| d[..c].trim().to_string())
        });
        if !out.iter().any(|a| a.name == name) {
            out.push(AttrRead {
                name,
                default,
                required,
            });
        }
        rest = after;
    }
    out
}

/// Index `files` (`(path, source text)`), resolving a named parser in any of
/// them.
pub fn index(files: &[(String, String)]) -> BTreeMap<String, OpEntry> {
    let mut out = BTreeMap::new();
    for (path, text) in files {
        for (line, op, rest) in inserts(text) {
            let arg = rest.trim_start_matches([',', ' ', '\n']).trim();
            let is_closure = arg.starts_with('|');
            let parser = (!is_closure).then(|| {
                arg.rsplit("::")
                    .next()
                    .unwrap_or(arg)
                    .trim_end_matches(')')
                    .trim()
                    .to_string()
            });
            let body = match &parser {
                None => Some(arg.to_string()),
                Some(name) => files
                    .iter()
                    .find_map(|(_, t)| fn_body(t, name))
                    .map(str::to_string),
            };
            let body = body.unwrap_or_default();
            out.insert(
                op,
                OpEntry {
                    file: path.clone(),
                    line,
                    parser,
                    attrs: attr_reads(&body),
                    opset_dependent: body.contains("onnx_operator_set_version"),
                },
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"
pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Neg", |_, _| Ok((ops::math::neg().into_hir(), vec![])));
    reg.insert("Gather", gather);
    reg.insert("Softmax", layer_soft_max);
    reg.insert("Reshape", |_, node| {
        let allowzero = node.get_attr_opt::<i64>("allowzero")?.unwrap_or(0) != 0;
        Ok((expand(array::Reshape::new(allowzero)), vec![]))
    });
    reg.insert("Cast", cast);
}
pub fn gather(_ctx: &ParsingContext, node: &NodeProto) -> R {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    Ok((expand(array::Gather::new(axis)), vec![]))
}
pub fn layer_soft_max(ctx: &ParsingContext, node: &NodeProto) -> R {
    let axis = node.get_attr_opt("axis")?;
    if ctx.onnx_operator_set_version < 13 {
        Ok((expand(LayerSoftmax::new(axis.unwrap_or(1), true)), vec![]))
    } else {
        Ok((expand(Softmax::new(axis.unwrap_or(-1))), vec![]))
    }
}
fn cast(_ctx: &ParsingContext, node: &NodeProto) -> R {
    let mut to = node.get_attr::<DatumType>("to")?;
    Ok((expand(Cast { to }), vec![]))
}
"#;

    /// Inserts, parsers and attribute defaults come out as written, and an
    /// opset-branching parser is marked rather than given one default.
    #[test]
    fn reads_inserts_parsers_and_defaults() {
        let ix = index(&[("ops.rs".into(), SRC.into())]);
        assert_eq!(ix.len(), 5);
        assert_eq!(ix["Neg"].parser, None);
        assert!(ix["Neg"].attrs.is_empty());
        assert_eq!(ix["Gather"].parser.as_deref(), Some("gather"));
        assert_eq!(
            ix["Gather"].attrs,
            vec![AttrRead {
                name: "axis".into(),
                default: Some("0".into()),
                required: false
            }]
        );
        assert!(!ix["Gather"].opset_dependent);
        assert_eq!(ix["Reshape"].attrs[0].default.as_deref(), Some("0"));
        assert!(ix["Softmax"].opset_dependent);
        assert_eq!(
            ix["Softmax"].attrs[0].default, None,
            "the default sits under the branch"
        );
        assert!(ix["Cast"].attrs[0].required);
        assert_eq!(ix["Gather"].line, 4);
    }
}
