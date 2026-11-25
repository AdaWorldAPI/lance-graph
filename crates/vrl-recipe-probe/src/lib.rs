//! The VRL recipe probe (D-VRL-RECIPE-0, lab only): the reference interpreter
//! and the parity tests for [`lance_graph_vrl`], where the recipe core now
//! lives. The interpreter materializes every row as an [`Event`] on purpose; it
//! shares only the parser with [`Recipe`] and reads no DT64 or mask-risc code.

use std::collections::BTreeMap;

pub use lance_graph_vrl::*;

/// One event of the reference interpreter: names repeated per row, which is
/// exactly what the recipe's [`Rows`] avoid.
pub type Event = BTreeMap<String, i64>;

fn bit(plane: &[u64], row: usize) -> bool {
    plane[row / 64] >> (row % 64) & 1 == 1
}

/// The naive reference, every value present: [`interpret_present`] with no
/// presence planes.
pub fn interpret(src: &str, labels: &[&str], lanes: &[&[i32]]) -> Result<Vec<Event>, Refusal> {
    interpret_present(src, labels, lanes, None).map(|i| i.events)
}

/// What the reference interpreter answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interpreted {
    /// The kept events, in row order. An absent output reads 0.
    pub events: Vec<Event>,
    /// For each kept event, the outputs that are absent.
    pub absent: Vec<Vec<String>>,
    /// Rows no abort dropped but some abort could not decide.
    pub unknown: usize,
}

/// The naive reference: materialize every row of `lanes` (labelled by
/// `labels`, `key` among them) as a VRL event, run the program statement by
/// statement, and project each surviving event onto the names the program
/// assigned and did not delete. Shares only the parser with [`Recipe`];
/// evaluation is independent and reads no DT64 or mask-risc code.
///
/// With `presence`, a value is ABSENT when its presence bit is 0 and it is 0
/// (EXO-294). An abort over an absent value is undecided: the row is dropped
/// if a later (or earlier) abort is decided and true, and otherwise counted
/// as unknown instead of kept.
#[expect(
    clippy::needless_range_loop,
    reason = "row r indexes every lane and its presence plane at once"
)]
pub fn interpret_present(
    src: &str,
    labels: &[&str],
    lanes: &[&[i32]],
    presence: Option<&[&[u64]]>,
) -> Result<Interpreted, Refusal> {
    let prog = parse(src)?;
    let n = lanes.first().map_or(0, |l| l.len());
    let mut out = Interpreted {
        events: Vec::new(),
        absent: Vec::new(),
        unknown: 0,
    };
    'rows: for r in 0..n {
        let value = |i: usize| -> Option<i64> {
            let v = lanes[i][r];
            let present = presence.is_none_or(|p| bit(p[i], r)) || v != 0;
            present.then_some(i64::from(v))
        };
        let mut top: BTreeMap<String, Option<i64>> = BTreeMap::new();
        let mut props: BTreeMap<&str, Option<i64>> = BTreeMap::new();
        for (i, l) in labels.iter().enumerate() {
            if *l == "key" {
                top.insert("key".into(), value(i));
            } else {
                props.insert(l, value(i));
            }
        }
        let mut assigned: Vec<String> = Vec::new();
        let read =
            |top: &BTreeMap<String, Option<i64>>, p: &Path| -> Result<Option<i64>, Refusal> {
                match p {
                    Path::Name(n) => top
                        .get(n)
                        .copied()
                        .ok_or_else(|| Refusal::UnknownPath(format!(".{n}"))),
                    Path::Prop(f) => props
                        .get(f.as_str())
                        .copied()
                        .ok_or_else(|| Refusal::UnknownPath(format!(".properties.{f}.value"))),
                }
            };
        let mut undecided = false;
        for stmt in &prog {
            match stmt {
                Stmt::Assign(name, p) => {
                    let v = read(&top, p)?;
                    top.insert(name.clone(), v);
                    if !assigned.contains(name) {
                        assigned.push(name.clone());
                    }
                }
                Stmt::Del(name) => {
                    if top.remove(name).is_none() {
                        return Err(Refusal::UnknownPath(format!(".{name}")));
                    }
                    assigned.retain(|a| a != name);
                }
                Stmt::Abort { path, cmp, value } => match read(&top, path)? {
                    Some(v) if cmp.holds(v, i64::from(*value)) => continue 'rows,
                    Some(_) => {}
                    None => undecided = true,
                },
                // An absent boolean is untouched: it holds neither value.
                Stmt::AbortBool { path, value } => match read(&top, path)? {
                    Some(v) if (v != 0) == *value => continue 'rows,
                    Some(_) => {}
                    None => undecided = true,
                },
                // The reference reads `i32` lanes only; a text literal is
                // checked against an intake table in lance-graph-vrl's tests.
                Stmt::AbortText { .. } => {
                    return Err(Refusal::Computes(
                        "text literal (the reference interpreter reads i32 lanes)".into(),
                    ))
                }
            }
        }
        if undecided {
            out.unknown += 1;
            continue;
        }
        out.events.push(
            assigned
                .iter()
                .map(|a| (a.clone(), top[a].unwrap_or(0)))
                .collect(),
        );
        out.absent.push(
            assigned
                .iter()
                .filter(|a| top[*a].is_none())
                .cloned()
                .collect(),
        );
    }
    Ok(out)
}
