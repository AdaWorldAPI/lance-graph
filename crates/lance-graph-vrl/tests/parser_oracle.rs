//! S3: the polyfill parser against upstream VRL's own parser. Every corpus
//! program the polyfill accepts must parse upstream and lower, from
//! upstream's AST, to the same statements. The polyfill stays the front end;
//! upstream's AST is only the oracle.
//!
//! Run with `--features upstream-vector` or `--features upstream-o2`.
#![cfg(any(feature = "upstream-vector", feature = "upstream-o2"))]

#[cfg(all(feature = "upstream-o2", not(feature = "upstream-vector")))]
use vrl_o2 as vrl;
#[cfg(feature = "upstream-vector")]
use vrl_vector as vrl;

use lance_graph_vrl::{parse, Cmp, Path, Stmt};
use vrl::parser::ast::{Assignment, Expr, Literal, Node, Predicate, RootExpr, Unary};

const CORPUS: &str = include_str!("parse_corpus.vrl");

fn programs() -> Vec<&'static str> {
    CORPUS
        .strip_suffix('\n')
        .unwrap_or(CORPUS)
        .split("\n%%%%\n")
        .collect()
}

/// A query or assignment target as written (`.a`, `.properties.f.value`).
fn path(written: &str) -> Option<Path> {
    let rest = written.strip_prefix('.')?;
    let segs: Vec<&str> = rest.split('.').collect();
    match segs.as_slice() {
        [n] if *n != "properties" => Some(Path::Name((*n).to_owned())),
        ["properties", f, "value"] => Some(Path::Prop((*f).to_owned())),
        _ => None,
    }
}

fn query(e: &Node<Expr>) -> Option<Path> {
    match e.inner() {
        Expr::Query(q) => path(&q.inner().to_string()),
        _ => None,
    }
}

fn cmp(op: &str) -> Option<Cmp> {
    Some(match op {
        "==" => Cmp::Eq,
        "!=" => Cmp::Ne,
        "<" => Cmp::Lt,
        "<=" => Cmp::Le,
        ">" => Cmp::Gt,
        ">=" => Cmp::Ge,
        _ => return None,
    })
}

/// Upstream's AST as the polyfill's statements; `None` where it lies
/// outside the subset.
fn lower(e: Node<Expr>) -> Option<Stmt> {
    match e.into_inner() {
        Expr::Assignment(a) => match a.into_inner() {
            Assignment::Single { target, expr, .. } => {
                let Path::Name(name) = path(&target.inner().to_string())? else {
                    return None;
                };
                Some(Stmt::Assign(name, query(&expr)?))
            }
            _ => None,
        },
        Expr::FunctionCall(f) => {
            let f = f.into_inner();
            if f.ident.inner().to_string() != "del" || f.arguments.len() != 1 {
                return None;
            }
            match path(&f.arguments[0].inner().expr.inner().to_string())? {
                Path::Name(n) => Some(Stmt::Del(n)),
                Path::Prop(_) => None,
            }
        }
        Expr::IfStatement(i) => {
            let i = i.into_inner();
            if i.else_node.is_some() {
                return None;
            }
            let block = i.if_node.into_inner().into_inner();
            let [only] = block.as_slice() else {
                return None;
            };
            match only.inner() {
                Expr::Abort(a) if a.inner().message.is_none() => {}
                _ => return None,
            }
            let Predicate::One(cond) = i.predicate.into_inner() else {
                return None;
            };
            let cond = *cond;
            match cond.inner() {
                Expr::Query(_) => Some(Stmt::AbortBool {
                    path: query(&cond)?,
                    value: true,
                }),
                Expr::Unary(u) => {
                    let Unary::Not(n) = u.inner().clone();
                    let (_, inner) = n.into_inner().take();
                    Some(Stmt::AbortBool {
                        path: query(&inner)?,
                        value: false,
                    })
                }
                Expr::Op(op) => {
                    let op = op.inner();
                    let p = query(&op.0)?;
                    let code = op.1.inner().to_string();
                    match op.2.inner() {
                        Expr::Literal(l) => match l.inner() {
                            Literal::Integer(v) => Some(Stmt::Abort {
                                path: p,
                                cmp: cmp(&code)?,
                                value: i32::try_from(*v).ok()?,
                            }),
                            Literal::Boolean(b) => match code.as_str() {
                                "==" => Some(Stmt::AbortBool { path: p, value: *b }),
                                "!=" => Some(Stmt::AbortBool {
                                    path: p,
                                    value: !*b,
                                }),
                                _ => None,
                            },
                            Literal::String(t) => match code.as_str() {
                                "==" | "!=" => Some(Stmt::AbortText {
                                    path: p,
                                    eq: code == "==",
                                    // `""` is a template with no segments.
                                    text: if t.0.is_empty() {
                                        String::new()
                                    } else {
                                        t.as_literal_string()?.to_owned()
                                    },
                                }),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn upstream(src: &str) -> Option<Vec<Stmt>> {
    let program = vrl::parser::parse(src).ok()?;
    program
        .0
        .into_iter()
        .map(|root| match root.into_inner() {
            RootExpr::Expr(e) => lower(e),
            RootExpr::Error(_) => None,
        })
        .collect()
}

/// FAILS IF: a program the polyfill parses is refused by upstream's parser,
/// or lowers from upstream's AST to other statements; or too few corpus
/// programs are compared for the check to mean anything.
#[test]
fn the_polyfill_parses_as_upstream_does() {
    let mut compared = 0;
    for p in programs() {
        let Ok(ours) = parse(p) else { continue };
        assert_eq!(upstream(p).as_ref(), Some(&ours), "{p:?}");
        compared += 1;
    }
    assert!(compared >= 60, "compared {compared}");
}
