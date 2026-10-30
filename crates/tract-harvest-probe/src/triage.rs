//! Triage of the nodes a registry does not bind, for any ONNX graph.
//!
//! For each unbound node the triage recovers, without a person reading the
//! graph:
//!
//! - **the gap**: a *dtype* gap (a field with the same op type and matching
//!   attributes exists for other dtypes), an *attribute* gap (the op type is
//!   registered at this dtype, under other attribute rules), or a missing *op*;
//! - **the value dtype**, keyed the way the registry keys the op (its fields'
//!   `dtype_arg`), else the first non-constant argument, else the first;
//! - **the role**: the node name with every digit run replaced by `N`, so
//!   per-layer repeats group together;
//! - **the argument profile**: dtype, rank and constness per argument;
//! - **a reference**: whether the frontend that typed the graph (Tract)
//!   registers the op;
//! - **native hints**: the rows of a caller-supplied `op → native word` table
//!   that match the op and value dtype.
//!
//! The triage reads; it decides nothing. Hints and references are candidates
//! for a person or a generator, never registrations.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::registry::{attrs_match, signature_dtype, Dim, Fact, Registry};
use crate::source::SourceInventory;
use crate::value::DType;

/// What separates an unbound node from a registration.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gap {
    /// The op type and attributes match fields registered for these dtypes.
    Dtype { registered: Vec<DType> },
    /// The op type is registered at this dtype under other attribute rules.
    Attr,
    /// No field has this op type.
    Op,
}

impl Gap {
    pub fn kind(&self) -> &'static str {
        match self {
            Gap::Dtype { .. } => "dtype",
            Gap::Attr => "attr",
            Gap::Op => "op",
        }
    }
}

/// One argument as the triage sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arg {
    pub dtype: Option<DType>,
    pub rank: Option<usize>,
    pub constant: bool,
}

/// One row of a native-word table: an op at a dtype, the word that computes
/// it, and how the word maps onto the op (`template`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeHint {
    pub op: String,
    pub dtype: DType,
    pub word: String,
    pub template: String,
    pub numeric: String,
}

/// One unbound node.
#[derive(Debug, Clone)]
pub struct Finding {
    pub node: usize,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub value_dtype: Option<DType>,
    pub gap: Gap,
    pub role: String,
    pub args: Vec<Arg>,
    pub in_frontend: bool,
    pub hints: Vec<usize>,
}

/// The node name with every digit run replaced by `N` and a trailing `_N`
/// dropped; an unnamed node is its op type.
pub fn role(name: &str, op_type: &str) -> String {
    if name.is_empty() {
        return op_type.to_string();
    }
    let mut out = String::with_capacity(name.len());
    let mut in_digits = false;
    for c in name.chars() {
        if c.is_ascii_digit() {
            if !in_digits {
                out.push('N');
            }
            in_digits = true;
        } else {
            out.push(c);
            in_digits = false;
        }
    }
    out.strip_suffix("_N").map(str::to_string).unwrap_or(out)
}

fn rank(f: &Fact) -> Option<usize> {
    f.shape.as_ref().map(Vec::len)
}

/// Values known before execution: initializers with data, `Constant`
/// outputs, and the caller's extra constants (folded values).
pub fn constant_names<'a>(
    src: &'a SourceInventory,
    extra: impl IntoIterator<Item = &'a str>,
) -> HashSet<&'a str> {
    let mut c: HashSet<&str> = src
        .initializers
        .iter()
        .filter(|i| i.value.is_some())
        .map(|i| i.name.as_str())
        .collect();
    for n in src.nodes.iter().filter(|n| n.op_type == "Constant") {
        c.extend(n.outputs.iter().map(String::as_str));
    }
    c.extend(extra);
    c
}

/// Triage every node `registry` does not bind.
pub fn triage(
    src: &SourceInventory,
    facts: &HashMap<String, Fact>,
    constants: &HashSet<&str>,
    registry: &Registry,
    frontend_ops: &BTreeSet<String>,
    hints: &[NativeHint],
) -> Vec<Finding> {
    let mut out = Vec::new();
    for (n, node) in src.nodes.iter().enumerate() {
        let args: Vec<Option<&Fact>> = node.inputs.iter().map(|i| facts.get(i)).collect();
        if registry.lookup(node, &args).is_some() {
            continue;
        }
        let domain = if node.domain.is_empty() {
            "ai.onnx".to_string()
        } else {
            node.domain.clone()
        };
        let same_op: Vec<_> = registry
            .regs
            .iter()
            .enumerate()
            .filter(|&(i, r)| {
                registry.exposes(i) && r.domain == domain && r.op_type == node.op_type
            })
            .map(|(_, r)| r)
            .collect();
        let value_dtype = match same_op.first() {
            Some(r) => signature_dtype(node, &args, r.dtype_arg),
            None => node
                .inputs
                .iter()
                .zip(&args)
                .find(|(i, _)| !constants.contains(i.as_str()))
                .or_else(|| node.inputs.iter().zip(&args).next())
                .and_then(|(_, f)| f.and_then(|f| f.dtype)),
        };
        let gap = if same_op.is_empty() {
            Gap::Op
        } else {
            let mut registered: Vec<DType> = same_op
                .iter()
                .filter(|r| attrs_match(node, r.attrs))
                .map(|r| r.dtype)
                .collect();
            registered.sort_by_key(|d| format!("{d:?}"));
            registered.dedup();
            let here = value_dtype;
            if !registered.is_empty() && here.is_none_or(|d| !registered.contains(&d)) {
                Gap::Dtype { registered }
            } else {
                Gap::Attr
            }
        };
        let arg_profile = node
            .inputs
            .iter()
            .zip(&args)
            .filter(|(i, _)| !i.is_empty())
            .map(|(i, f)| Arg {
                dtype: f.and_then(|f| f.dtype),
                rank: f.and_then(rank),
                constant: constants.contains(i.as_str()),
            })
            .collect();
        let hint_rows = hints
            .iter()
            .enumerate()
            .filter(|(_, h)| h.op == node.op_type && Some(h.dtype) == value_dtype)
            .map(|(i, _)| i)
            .collect();
        out.push(Finding {
            node: n,
            name: node.name.clone(),
            op_type: node.op_type.clone(),
            domain,
            value_dtype,
            gap,
            role: role(&node.name, &node.op_type),
            args: arg_profile,
            in_frontend: frontend_ops.contains(&node.op_type),
            hints: hint_rows,
        });
    }
    out
}

/// Findings grouped by `(op type, value dtype, gap kind)`, each with its node
/// count and roles (role → count), largest group first.
pub fn groups(findings: &[Finding]) -> Vec<Group> {
    let mut by: BTreeMap<(String, String, &'static str), Group> = BTreeMap::new();
    for f in findings {
        let key = (
            f.op_type.clone(),
            f.value_dtype
                .map(|d| format!("{d:?}"))
                .unwrap_or("?".into()),
            f.gap.kind(),
        );
        let g = by.entry(key.clone()).or_insert_with(|| Group {
            op_type: key.0.clone(),
            value_dtype: key.1.clone(),
            gap: f.gap.clone(),
            nodes: 0,
            roles: BTreeMap::new(),
            in_frontend: f.in_frontend,
            hints: f.hints.clone(),
        });
        g.nodes += 1;
        *g.roles.entry(f.role.clone()).or_default() += 1;
    }
    let mut v: Vec<Group> = by.into_values().collect();
    v.sort_by(|a, b| b.nodes.cmp(&a.nodes).then(a.op_type.cmp(&b.op_type)));
    v
}

/// One `(op type, value dtype, gap kind)` group.
#[derive(Debug, Clone)]
pub struct Group {
    pub op_type: String,
    pub value_dtype: String,
    pub gap: Gap,
    pub nodes: usize,
    pub roles: BTreeMap<String, usize>,
    pub in_frontend: bool,
    pub hints: Vec<usize>,
}

/// Parse a native-word table: tab-separated `op dtype word template numeric`,
/// `#` comments and blank lines ignored. A row with an unknown dtype or the
/// wrong number of columns is an error naming its line.
pub fn parse_hints(text: &str) -> Result<Vec<NativeHint>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() != 5 {
            return Err(format!("line {}: {} columns, expected 5", i + 1, c.len()));
        }
        let dtype = match c[1] {
            "f32" => DType::F32,
            "f64" => DType::F64,
            "i32" => DType::I32,
            "i64" => DType::I64,
            "bool" => DType::Bool,
            other => return Err(format!("line {}: unknown dtype {other:?}", i + 1)),
        };
        out.push(NativeHint {
            op: c[0].into(),
            dtype,
            word: c[2].into(),
            template: c[3].into(),
            numeric: c[4].into(),
        });
    }
    Ok(out)
}

/// A `Dim` list as text, for reports.
pub fn dims(shape: &[Dim]) -> String {
    shape
        .iter()
        .map(|d| match d {
            Dim::Known(k) => k.to_string(),
            Dim::Symbol(s) => s.clone(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::registry;
    use crate::source::{Attr, SourceNode};

    fn node(name: &str, op: &str, inputs: &[&str], attrs: Vec<(String, Attr)>) -> SourceNode {
        SourceNode {
            name: name.into(),
            op_type: op.into(),
            domain: String::new(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: vec![format!("{name}_out")],
            attributes: attrs.iter().map(|(n, _)| n.clone()).collect(),
            attrs,
        }
    }

    fn fact(dt: DType, shape: &[usize]) -> Fact {
        Fact {
            dtype: Some(dt),
            shape: Some(shape.iter().map(|&d| Dim::Known(d)).collect()),
        }
    }

    /// The three gap kinds on one graph, plus a bound node that is not
    /// reported: Where with f32 values is a dtype gap (keyed on argument 1,
    /// not the bool condition), Neg is a missing op, Reshape with
    /// `allowzero = 1` is an attribute gap, and Add f32 binds.
    #[test]
    fn classifies_dtype_attr_and_op_gaps() {
        let src = SourceInventory {
            nodes: vec![
                node("/layers.3/attn/Where_7", "Where", &["c", "x", "y"], vec![]),
                node("/layers.3/attn/Neg", "Neg", &["x"], vec![]),
                node(
                    "r",
                    "Reshape",
                    &["x", "shape"],
                    vec![("allowzero".into(), Attr::Int(1))],
                ),
                node("a", "Add", &["x", "x"], vec![]),
            ],
            inputs: vec!["c".into(), "x".into(), "y".into()],
            model_sha256: String::new(),
            inlined: vec![],
            ir_version: 8,
            opsets: vec![],
            initializers: vec![],
            outputs: vec![],
        };
        let facts: HashMap<String, Fact> = [
            ("c", fact(DType::Bool, &[2, 3])),
            ("x", fact(DType::F32, &[2, 3])),
            ("y", fact(DType::F32, &[2, 3])),
            ("shape", fact(DType::I64, &[1])),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let consts: HashSet<&str> = ["shape"].into_iter().collect();
        let tract: BTreeSet<String> = ["Where", "Reshape"].iter().map(|s| s.to_string()).collect();
        let hints = parse_hints("Neg\tf32\tndarray::hpc::vml::vsneg\tunary_view\texact\n").unwrap();
        let reg = registry();
        let f = triage(&src, &facts, &consts, &reg, &tract, &hints);
        let got: Vec<(&str, &str, Option<DType>)> = f
            .iter()
            .map(|f| (f.op_type.as_str(), f.gap.kind(), f.value_dtype))
            .collect();
        assert_eq!(
            got,
            vec![
                ("Where", "dtype", Some(DType::F32)),
                ("Neg", "op", Some(DType::F32)),
                ("Reshape", "attr", Some(DType::F32)),
            ]
        );
        assert_eq!(f[0].role, "/layers.N/attn/Where");
        assert!(f[0].in_frontend && !f[1].in_frontend);
        assert_eq!(f[1].hints, vec![0]);
        assert!(f[0].hints.is_empty(), "a hint is matched by op and dtype");
        assert!(f[2].args[1].constant && !f[2].args[0].constant);
        assert_eq!(f[2].args[0].rank, Some(2));
        let g = groups(&f);
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn role_folds_digit_runs_and_names_unnamed_nodes() {
        assert_eq!(
            role("/layers.12/mlp/act_fn/Sigmoid", "Sigmoid"),
            "/layers.N/mlp/act_fn/Sigmoid"
        );
        assert_eq!(
            role("/layers.0/self_attn/Slice_3", "Slice"),
            "/layers.N/self_attn/Slice"
        );
        assert_eq!(role("", "Squeeze"), "Squeeze");
    }

    #[test]
    fn hint_table_rejects_bad_rows() {
        assert!(parse_hints("# c\n\nNeg\tf32\tw\tt\texact\n").is_ok());
        assert!(parse_hints("Neg\tf16\tw\tt\texact\n")
            .unwrap_err()
            .contains("line 1"));
        assert!(parse_hints("Neg\tf32\tw\n")
            .unwrap_err()
            .contains("3 columns"));
    }

    /// The committed table parses, and names only ops a registration could
    /// key on (no duplicate `(op, dtype, word)` rows).
    #[test]
    fn committed_hint_table_parses() {
        let t = parse_hints(include_str!("../data/native_words.tsv")).unwrap();
        assert!(t.len() >= 10);
        let mut keys: Vec<_> = t
            .iter()
            .map(|h| (h.op.clone(), format!("{:?}", h.dtype), h.word.clone()))
            .collect();
        let n = keys.len();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), n);
    }
}
