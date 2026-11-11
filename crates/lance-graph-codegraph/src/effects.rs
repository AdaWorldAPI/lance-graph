//! A read-only view of `<crate>.effects.tsv`: the effects a function body witnesses.
//!
//! ruff_charon writes one row per occurrence (`ruff_charon::store`, version 1, columns
//! `subject effect kind file start end`): an `ASSERT` that can panic (`DivisionByZero`,
//! `BoundsCheck`, `Overflow:<BinOp>`, `User`, ...) or a `DEREF_WRITE` rooted in an argument or a
//! local. Before this module the Codegraph did not read the file at all.
//!
//! These rows are observations inside one body, kept apart from the call rows on purpose:
//!
//! - They are never folded into [`crate::Codegraph::impact`] or any mask.
//! - A function with no row is **not** proven free of that effect. Charon may fold a check into
//!   a fallible operation, a callee's effects are not included, and an unstructured body records
//!   nothing. [`EffectAnswer::NoneWitnessed`] says exactly that and no more.
//! - A function whose crate has no effects file was not looked at
//!   ([`EffectAnswer::CrateNotRead`]).

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use lance_graph_contract::hash::fnv1a;

use crate::graph::RunStatus;

/// File-name suffix of the effects file of a crate.
pub const EFFECTS_SUFFIX: &str = ".effects.tsv";

/// The kind of a witnessed effect: the `effect` and `kind` columns.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EffectKind {
    /// An `Assert` that can panic, with Charon's tag (`DivisionByZero`, `Overflow:Mul`, `User`).
    /// The tag set is Charon's and is kept as text.
    Assert(String),
    /// A write through a dereference rooted in an argument.
    DerefWriteArg,
    /// A write through a dereference rooted in a local.
    DerefWriteLocal,
}

impl EffectKind {
    /// The `effect` and `kind` columns of this kind.
    pub fn columns(&self) -> (&'static str, &str) {
        match self {
            EffectKind::Assert(k) => ("ASSERT", k),
            EffectKind::DerefWriteArg => ("DEREF_WRITE", "Arg"),
            EffectKind::DerefWriteLocal => ("DEREF_WRITE", "Local"),
        }
    }

    fn parse(effect: &str, kind: &str) -> Option<EffectKind> {
        match (effect, kind) {
            ("ASSERT", k) if !k.is_empty() => Some(EffectKind::Assert(k.to_owned())),
            ("DEREF_WRITE", "Arg") => Some(EffectKind::DerefWriteArg),
            ("DEREF_WRITE", "Local") => Some(EffectKind::DerefWriteLocal),
            _ => None,
        }
    }
}

/// The effect classes a review asks about. Other asserts (bounds, add/sub/mul overflow, a source
/// `assert!`) have no class and are reached through [`EffectKind`] directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EffectClass {
    /// `DivisionByZero` and `Overflow:Div`: a `/` the compiler guards.
    Division,
    /// `RemainderByZero` and `Overflow:Rem`: a `%`.
    Remainder,
    /// `Overflow:Shl` and `Overflow:Shr`.
    Shift,
    /// A write through a dereference, rooted in an argument or a local.
    Write,
}

impl EffectKind {
    /// The class of this kind, if it has one.
    pub fn class(&self) -> Option<EffectClass> {
        match self {
            EffectKind::DerefWriteArg | EffectKind::DerefWriteLocal => Some(EffectClass::Write),
            EffectKind::Assert(k) => match k.as_str() {
                "DivisionByZero" | "Overflow:Div" => Some(EffectClass::Division),
                "RemainderByZero" | "Overflow:Rem" => Some(EffectClass::Remainder),
                "Overflow:Shl" | "Overflow:Shr" => Some(EffectClass::Shift),
                _ => None,
            },
        }
    }
}

/// What identifies one effects file as read: its content digest and row count. The file carries
/// no source revision (ruff_charon effects version 1), so this says which bytes were read, not
/// which commit they were extracted from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EffectsIdentity {
    /// FNV-1a 64 of the file's bytes.
    pub fnv64: u64,
    /// Data rows read from it.
    pub rows: u32,
}

/// Where an effect sits: a file relative to the source root and a byte span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectSite {
    /// The recorded file path.
    pub file: String,
    /// Span start, in bytes.
    pub start: u32,
    /// Span end, in bytes.
    pub end: u32,
}

/// One witnessed effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Effect {
    /// The function whose body holds the effect.
    pub subject: String,
    /// What was witnessed.
    pub kind: EffectKind,
    /// Its site; `None` when the row carries `-`.
    pub site: Option<EffectSite>,
}

/// Why an effects file was refused. Nothing is guessed: a file that does not match the format
/// is refused whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectsRefusal {
    /// The directory or a file could not be read.
    Io(String),
    /// The `#version` line is not 1 (0 when absent or unparsable).
    Version {
        /// The crate.
        krate: String,
        /// The version found.
        found: u32,
    },
    /// The `#schema` line is not `subject effect kind file start end`.
    Schema {
        /// The crate.
        krate: String,
    },
    /// A data line without exactly six fields; 1-based line.
    Arity {
        /// The crate.
        krate: String,
        /// The offending line.
        line: usize,
    },
    /// An `effect`/`kind` pair outside the format; 1-based line.
    UnknownEffect {
        /// The crate.
        krate: String,
        /// The offending line.
        line: usize,
    },
    /// A site that is neither `- - -` nor a file with `start <= end`; 1-based line.
    Site {
        /// The crate.
        krate: String,
        /// The offending line.
        line: usize,
    },
}

/// What the store says about one function's effects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectAnswer<'a> {
    /// Rows the body witnesses, in file order.
    Witnessed(Vec<&'a Effect>),
    /// The crate's effects file was read and holds no matching row. Not a purity verdict.
    NoneWitnessed,
    /// No effects file was read for the function's crate: nothing was looked at.
    CrateNotRead,
}

/// Every effects file of a store directory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Effects {
    rows: Vec<Effect>,
    crates: BTreeSet<String>,
    identities: Vec<(String, EffectsIdentity)>,
}

const SCHEMA: &str = "#schema subject\teffect\tkind\tfile\tstart\tend";

fn parse(krate: &str, text: &str, rows: &mut Vec<Effect>) -> Result<(), EffectsRefusal> {
    let k = || krate.to_owned();
    let mut lines = text.lines();
    let found = lines
        .next()
        .and_then(|l| l.strip_prefix("#version "))
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(0);
    if found != 1 {
        return Err(EffectsRefusal::Version { krate: k(), found });
    }
    if lines.next() != Some(SCHEMA) {
        return Err(EffectsRefusal::Schema { krate: k() });
    }
    for (i, l) in lines.enumerate() {
        let line = i + 3;
        if l.is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split('\t').collect();
        let [subject, effect, kind, file, start, end] = f[..] else {
            return Err(EffectsRefusal::Arity { krate: k(), line });
        };
        let kind = EffectKind::parse(effect, kind)
            .ok_or(EffectsRefusal::UnknownEffect { krate: k(), line })?;
        let site = match (file, start.parse::<u32>(), end.parse::<u32>()) {
            ("-", _, _) if start == "-" && end == "-" => None,
            (file, Ok(start), Ok(end)) if file != "-" && start <= end => Some(EffectSite {
                file: file.to_owned(),
                start,
                end,
            }),
            _ => return Err(EffectsRefusal::Site { krate: k(), line }),
        };
        rows.push(Effect {
            subject: subject.to_owned(),
            kind,
            site,
        });
    }
    Ok(())
}

impl Effects {
    /// Read every `<crate>.effects.tsv` of `dir`, sorted by crate name. A directory without any
    /// is an empty view in which every crate is [`EffectAnswer::CrateNotRead`].
    pub fn open(dir: &Path) -> Result<Effects, EffectsRefusal> {
        let io = |e: std::io::Error| EffectsRefusal::Io(e.to_string());
        let mut crates = BTreeSet::new();
        for entry in fs::read_dir(dir).map_err(io)? {
            let name = entry.map_err(io)?.file_name();
            if let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(EFFECTS_SUFFIX)) {
                crates.insert(stem.to_owned());
            }
        }
        let mut rows = Vec::new();
        let mut identities = Vec::new();
        for krate in &crates {
            let text =
                fs::read_to_string(dir.join(format!("{krate}{EFFECTS_SUFFIX}"))).map_err(io)?;
            let before = rows.len();
            parse(krate, &text, &mut rows)?;
            identities.push((
                krate.clone(),
                EffectsIdentity {
                    fnv64: fnv1a(text.as_bytes()),
                    rows: (rows.len() - before) as u32,
                },
            ));
        }
        Ok(Effects {
            rows,
            crates,
            identities,
        })
    }

    /// The crates whose effects file was read, sorted.
    pub fn crates(&self) -> impl Iterator<Item = &str> {
        self.crates.iter().map(String::as_str)
    }

    /// The identity of `krate`'s effects file; `None` when none was read.
    pub fn identity(&self, krate: &str) -> Option<EffectsIdentity> {
        self.identities
            .iter()
            .find(|(k, _)| k == krate)
            .map(|(_, i)| *i)
    }

    /// Every row, in crate then file order.
    pub fn rows(&self) -> &[Effect] {
        &self.rows
    }

    /// The effects witnessed in the body of `subject`. The subject's crate is its first path
    /// segment.
    pub fn of(&self, subject: &str) -> EffectAnswer<'_> {
        let krate = subject.split("::").next().unwrap_or(subject);
        if !self.crates.contains(krate) {
            return EffectAnswer::CrateNotRead;
        }
        let hits: Vec<&Effect> = self.rows.iter().filter(|e| e.subject == subject).collect();
        if hits.is_empty() {
            EffectAnswer::NoneWitnessed
        } else {
            EffectAnswer::Witnessed(hits)
        }
    }

    /// Every row of `kind`, in crate then file order. An empty answer covers only the crates in
    /// [`Effects::crates`].
    pub fn with_kind<'a>(&'a self, kind: &'a EffectKind) -> impl Iterator<Item = &'a Effect> {
        self.rows.iter().filter(move |e| e.kind == *kind)
    }
}

/// How a function reaches an effect site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReachGrade {
    /// The effect is in the function's own body.
    Own,
    /// Through resolved calls only.
    Confirmed,
    /// Through at least one trait dispatch the compiler did not resolve.
    Possible,
    /// Through a dynamic call or function pointer.
    Unknown,
}

/// One effect a function reaches, with how.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReachedEffect<'a> {
    /// The effect row.
    pub effect: &'a Effect,
    /// How the queried function reaches the body that holds it.
    pub grade: ReachGrade,
}

/// A reason the reach answer may be missing effects. Each comes from a store fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReachLimit {
    /// A crate with call rows has no effects file: its bodies' effects were not read.
    EffectsNotRead {
        /// The crate.
        krate: String,
    },
    /// A crate's harvest did not finish, recorded residuals, or has no residuals file, so a call
    /// or a body may be missing.
    HarvestIncomplete {
        /// The crate.
        krate: String,
    },
    /// The function reaches symbols of crates that are not in the store (`core`, a dependency
    /// that was not extracted). Their bodies, and so their effects, were never read.
    CalleesOutsideStore {
        /// The crates reached, sorted.
        crates: Vec<String>,
    },
    /// The function holds or reaches a dynamic call; an effect behind it has no call row to be
    /// reached by.
    DynamicCalls {
        /// `UNRESOLVED_CALL` rows in the store.
        store_dynamic_calls: u32,
    },
    /// A matching effect's subject has no position in the call rows, so whether the function
    /// reaches it is not known.
    Unplaced {
        /// The effect's subject.
        subject: String,
    },
}

/// The effects a function reaches through the call rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reached<'a> {
    /// Reached rows, in effects-file order.
    pub hits: Vec<ReachedEffect<'a>>,
    /// Why the answer may be missing effects; empty only when nothing known is missing.
    pub limits: Vec<ReachLimit>,
}

impl Reached<'_> {
    /// `true` when a limit is known: the answer is then partial, and an empty `hits` is
    /// "unknown", never "none". `false` is still not a proof of absence (see [`Effects`]).
    pub fn partial(&self) -> bool {
        !self.limits.is_empty()
    }
}

impl crate::Codegraph {
    /// The effect rows `keep` selects whose body `symbol` is or reaches.
    ///
    /// This joins two observation lists and inherits the limits of both. What is known to be
    /// missing is listed in [`Reached::limits`]: a crate without an effects file, an incomplete
    /// harvest, callees in crates outside the store, dynamic calls, an effect that cannot be
    /// placed. `None` when `symbol` has no
    /// position.
    pub fn effects_reached<'a>(
        &self,
        symbol: &str,
        effects: &'a Effects,
        keep: impl Fn(&EffectKind) -> bool,
    ) -> Result<Option<Reached<'a>>, crate::query::QueryRefusal> {
        let Some(me) = self.pos(symbol) else {
            return Ok(None);
        };
        let has = |mask: &[u64]| {
            mask.get(me as usize / 64)
                .is_some_and(|w| w >> (me % 64) & 1 == 1)
        };
        let mut limits = Vec::new();
        for (krate, status) in self.crate_list() {
            if effects.identity(&krate).is_none() {
                limits.push(ReachLimit::EffectsNotRead {
                    krate: krate.clone(),
                });
            }
            let clean_harvest =
                matches!(self.crate_residuals(&krate), Some(Some(r)) if r.total() == 0);
            if status == RunStatus::Partial || !clean_harvest {
                limits.push(ReachLimit::HarvestIncomplete { krate });
            }
        }
        // Symbols outside every stored crate that the function reaches: who reaches each one is
        // its impact, read in the same direction as the effect subjects below.
        let stored_owner = self.stored_owner();
        let mut outside = BTreeSet::new();
        let mut reaches_dynamic = false;
        for p in 0..(self.words() as u32 * 64) {
            let stored = stored_owner
                .get(p as usize / 64)
                .is_some_and(|w| w >> (p % 64) & 1 == 1);
            let Some(name) = self.symbol(p).filter(|_| !stored && p != me) else {
                continue;
            };
            let i = self.impact(p)?;
            if has(&i.confirmed) || has(&i.possible) || has(&i.unknown) {
                // `?dynamic` is the position every unresolved call points at, not a crate.
                if name.starts_with('?') {
                    reaches_dynamic = true;
                } else {
                    outside.insert(name.split("::").next().unwrap_or(name).to_owned());
                }
            }
        }
        if !outside.is_empty() {
            limits.push(ReachLimit::CalleesOutsideStore {
                crates: outside.into_iter().collect(),
            });
        }
        if reaches_dynamic {
            limits.push(ReachLimit::DynamicCalls {
                store_dynamic_calls: self.coverage().dynamic_calls,
            });
        }
        let mut hits = Vec::new();
        let mut unplaced = BTreeSet::new();
        // One impact per distinct subject: who reaches the body that holds the effect.
        let mut graded: Vec<(&str, Option<ReachGrade>)> = Vec::new();
        for e in effects.rows().iter().filter(|e| keep(&e.kind)) {
            let subject = e.subject.as_str();
            let grade = match graded.iter().find(|(s, _)| *s == subject) {
                Some((_, g)) => *g,
                None => {
                    let g = match self.pos(subject) {
                        None => {
                            unplaced.insert(subject);
                            None
                        }
                        Some(p) if p == me => Some(ReachGrade::Own),
                        Some(p) => {
                            let i = self.impact(p)?;
                            if has(&i.confirmed) {
                                Some(ReachGrade::Confirmed)
                            } else if has(&i.possible) {
                                Some(ReachGrade::Possible)
                            } else if has(&i.unknown) {
                                Some(ReachGrade::Unknown)
                            } else {
                                None
                            }
                        }
                    };
                    graded.push((subject, g));
                    g
                }
            };
            if let Some(grade) = grade {
                hits.push(ReachedEffect { effect: e, grade });
            }
        }
        limits.extend(unplaced.into_iter().map(|s| ReachLimit::Unplaced {
            subject: s.to_owned(),
        }));
        Ok(Some(Reached { hits, limits }))
    }
}
