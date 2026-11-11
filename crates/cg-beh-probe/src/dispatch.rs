//! Source structure: which arm of `FoldDialect::call` a line sits in, and
//! which polyfill word that arm's opcode names.
//!
//! This reads text. It says a line is inside the arm for an opcode; it does
//! not say the arm computes what the word means.

use lance_graph_r2il::polyfill::POLYFILL;
use lance_graph_r2il::{
    INT_ADD, INT_AND, INT_EQUAL, INT_MULT, INT_NOT, INT_OR, INT_SUB, INT_S_DIV, INT_S_LESS,
    INT_XOR, POP_COUNT,
};
use ogar_loco::FnIndex;
use ogar_r2il::{ALL, ANY, KEY_RUNS, MAX, MIN, SUM};

use crate::{fnv64, FNV64_OFFSET};

macro_rules! opcodes {
    ($($name:ident),* $(,)?) => { &[$((stringify!($name), $name)),*] };
}

/// The opcode constants the arms are written with, by the name they have in
/// source. The compiler resolves each name to its constant.
pub const OPCODES: &[(&str, FnIndex)] = opcodes!(
    INT_ADD, INT_SUB, INT_MULT, INT_S_DIV, INT_EQUAL, INT_S_LESS, INT_AND, INT_OR, INT_XOR,
    INT_NOT, POP_COUNT, SUM, MIN, MAX, ANY, ALL, KEY_RUNS,
);

/// The polyfill position whose opcode has this source name, read from the
/// compiled [`POLYFILL`] table.
pub fn word_of(opcode_name: &str) -> Option<u8> {
    let (_, opcode) = OPCODES.iter().find(|(n, _)| *n == opcode_name)?;
    POLYFILL
        .iter()
        .position(|w| w.opcode == *opcode)
        .and_then(|p| u8::try_from(p).ok())
}

/// The source name of the opcode at polyfill position `word`.
pub fn opcode_name(word: u8) -> Option<&'static str> {
    let w = POLYFILL.get(usize::from(word))?;
    OPCODES
        .iter()
        .find(|(_, op)| *op == w.opcode)
        .map(|(n, _)| *n)
}

/// One arm of the dispatch `match`: the pattern names and its line span
/// (1-based, inclusive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arm {
    pub names: Vec<String>,
    pub start: usize,
    pub end: usize,
}

/// The arms of one reading of the dispatch function, with the digest of the
/// text they were read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispatch {
    pub file_fnv: u64,
    pub arms: Vec<Arm>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// `pub fn call(&mut self, f: FnIndex` was not found.
    NoCallFn,
    /// The function has no `match f {` at the expected depth.
    NoMatch,
    /// The `match` was found and no arm was read.
    NoArms,
}

/// The digest [`Dispatch::file_fnv`] holds for `src`.
pub fn text_fnv(src: &str) -> u64 {
    fnv64(FNV64_OFFSET, src.as_bytes())
}

const ARM_INDENT: &str = "            ";
const MATCH_INDENT: &str = "        ";

fn arm_names(line: &str) -> Option<Vec<String>> {
    let rest = line.strip_prefix(ARM_INDENT)?;
    if rest.starts_with(' ') {
        return None;
    }
    let head = rest.split(" => ").next()?;
    if head.len() == rest.len() {
        return None;
    }
    let names: Vec<String> = head.split(" | ").map(str::to_string).collect();
    let is_path = |n: &String| {
        !n.is_empty()
            && n.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
    };
    names.iter().all(is_path).then_some(names)
}

/// Read the arms of `FoldDialect::call` from the text of
/// `lance-graph-r2il/src/lib.rs`.
pub fn read(src: &str) -> Result<Dispatch, DispatchError> {
    let lines: Vec<&str> = src.lines().collect();
    let call = lines
        .iter()
        .position(|l| l.contains("pub fn call(&mut self, f: FnIndex"))
        .ok_or(DispatchError::NoCallFn)?;
    let open = (call..lines.len())
        .find(|&i| lines[i] == format!("{MATCH_INDENT}match f {{"))
        .ok_or(DispatchError::NoMatch)?;
    let close = (open + 1..lines.len())
        .find(|&i| lines[i] == format!("{MATCH_INDENT}}}"))
        .ok_or(DispatchError::NoMatch)?;
    let mut arms: Vec<Arm> = Vec::new();
    for (i, line) in lines.iter().enumerate().take(close).skip(open + 1) {
        if let Some(names) = arm_names(line) {
            if let Some(last) = arms.last_mut() {
                last.end = i;
            }
            arms.push(Arm {
                names,
                start: i + 1,
                end: close,
            });
        }
    }
    if arms.is_empty() {
        return Err(DispatchError::NoArms);
    }
    Ok(Dispatch {
        file_fnv: text_fnv(src),
        arms,
    })
}

impl Dispatch {
    /// The arm `line` sits in.
    pub fn arm_at(&self, line: usize) -> Option<&Arm> {
        self.arms.iter().find(|a| a.start <= line && line <= a.end)
    }

    /// The arm that names the opcode of polyfill position `word`.
    pub fn arm_of_word(&self, word: u8) -> Option<&Arm> {
        let name = opcode_name(word)?;
        self.arms.iter().find(|a| a.names.iter().any(|n| n == name))
    }

    /// The polyfill positions with no arm in this reading.
    pub fn words_without_arm(&self) -> Vec<u8> {
        (0..POLYFILL.len())
            .filter_map(|p| u8::try_from(p).ok())
            .filter(|&p| self.arm_of_word(p).is_none())
            .collect()
    }

    /// The reading as `cg-beh.dispatch.v1` rows.
    pub fn to_tsv(&self) -> String {
        let mut o = format!("H\tcg-beh.dispatch.v1\t{:016x}\n", self.file_fnv);
        for a in &self.arms {
            for n in &a.names {
                let word = word_of(n).map_or("NOT_A_WORD".to_string(), |w| w.to_string());
                o.push_str(&format!("ARM\t{n}\t{word}\t{}\t{}\n", a.start, a.end));
            }
        }
        o
    }
}
