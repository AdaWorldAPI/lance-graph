//! A coverage-aware impact report: [`Codegraph::impact`] plus what the answer could not see.
//!
//! [`Codegraph::impact`] answers in symbol positions, and that stays as it is. A caller in a
//! crate the store never loaded has no position, so it cannot appear in [`Graded::unknown`]
//! (that would be a fabricated entry). This module reports such callers as missing *coverage*
//! instead: an [`ImpactReport`] carries the unchanged [`Graded`] answer together with every
//! [`CoverageGap`] that applies to it.
//!
//! Some coverage facts cannot be read off the rows: which targets were extracted, which crates
//! depend on the symbol's crate, and whether a source file changed since extraction. The caller
//! declares those in an [`ExtractionScope`]. A fact it does not declare becomes a gap
//! ([`CoverageGap::ConsumersNotDeclared`], [`CoverageGap::FreshnessNotChecked`]), never an
//! assumption.
//!
//! An empty gap list means no *known* gap within the declared scope. Like
//! [`crate::diag::Coverage::no_known_gap`], it is never a proof that the harvest saw everything,
//! and the declared scope is only as complete as the caller made it.

use std::collections::BTreeSet;
use std::path::Path;

use lance_graph_contract::hash::fnv1a;

use crate::graph::{Codegraph, Pos, RunStatus};
use crate::query::{Graded, QueryRefusal};

/// Which compilation targets the store's crates were extracted from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Targets {
    /// Library targets only: test targets, `#[cfg(test)]` modules, examples and benches were
    /// not compiled, so their calls are unobserved.
    LibraryOnly,
    /// Library and test targets.
    LibraryAndTests,
}

/// The crates that depend on the queried symbol's crate, as the caller knows them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dependents<'a> {
    /// Not declared: callers outside the store are unknown.
    NotDeclared,
    /// These crates depend on the symbol's crate (an empty slice means none do).
    Declared(&'a [&'a str]),
}

/// How the store was produced, as far as the rows cannot say it themselves.
#[derive(Clone, Copy, Debug)]
pub struct ExtractionScope<'a> {
    /// Targets compiled for every crate in the store.
    pub targets: Targets,
    /// Reverse dependencies of the queried symbol's crate.
    pub dependents: Dependents<'a>,
    /// The store directory (`<crate>.files.tsv`) and the source root its paths are relative to.
    /// `None`: freshness is not checked.
    pub sources: Option<(&'a Path, &'a Path)>,
}

/// Something the answer could not observe. Each gap names its cause.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoverageGap {
    /// Declared dependents whose crates are not in the store; their callers are unobserved.
    ConsumersNotInStore {
        /// Crate names, sorted.
        crates: Vec<String>,
    },
    /// The symbol is public and its dependents were not declared, so callers outside the store
    /// may exist.
    ConsumersNotDeclared,
    /// Only library targets were extracted; callers in tests are unobserved.
    TestTargetsNotExtracted,
    /// Callers reach the symbol only through trait dispatch (`possible`), which the compiler did
    /// not resolve to this implementation.
    UnresolvedDispatch {
        /// Positions in `graded.possible`.
        possible: u32,
    },
    /// A dynamic call or function pointer may reach the symbol (`graded.unknown`).
    DynamicCalls {
        /// Positions in `graded.unknown`.
        unknown: u32,
    },
    /// A store crate's CRATE row is not `COMPLETE`.
    PartialCrate {
        /// The crate name.
        krate: String,
    },
    /// A store crate carries extraction residuals (e.g. `BODY_NOT_STRUCTURED`): facts ruff_charon
    /// could not represent, any of which may hide a call to the symbol.
    Residuals {
        /// The crate name.
        krate: String,
        /// Residuals over every kind.
        count: u32,
    },
    /// A store crate was read without a `<crate>.residuals.tsv`: what its harvest missed is unknown.
    ResidualsNotRead {
        /// The crate name.
        krate: String,
    },
    /// Source files whose content no longer matches the store.
    StaleSource {
        /// Paths relative to the source root, sorted.
        files: Vec<String>,
    },
    /// Source files the store indexes that are absent from the source root.
    MissingSource {
        /// Paths relative to the source root, sorted.
        files: Vec<String>,
    },
    /// A store crate has no `<crate>.files.tsv` to check freshness against.
    NoFileList {
        /// The crate name.
        krate: String,
    },
    /// No source root was given, so freshness was not checked.
    FreshnessNotChecked,
}

/// The impact of a symbol and the coverage behind it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImpactReport {
    /// Exactly [`Codegraph::impact`].
    pub graded: Graded,
    /// Every crate in the store with its extractor status, sorted by name.
    pub observed: Vec<(String, RunStatus)>,
    /// What the answer could not see; empty only when nothing known is missing.
    pub gaps: Vec<CoverageGap>,
}

impl ImpactReport {
    /// No known gap within the declared scope. Not a proof of completeness.
    pub fn no_known_gap(&self) -> bool {
        self.gaps.is_empty()
    }
}

/// An impact answer, or the statement that the symbol is not in the store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImpactAnswer {
    /// The symbol has a position.
    Found(ImpactReport),
    /// No position. Not "no impact": the symbol may exist in an unloaded crate.
    SymbolNotInStore {
        /// The symbol asked for.
        symbol: String,
        /// The crates the store holds.
        observed: Vec<String>,
    },
}

fn popcount(mask: &[u64]) -> u32 {
    mask.iter().map(|w| w.count_ones()).sum()
}

impl Codegraph {
    /// [`Codegraph::impact`] of `symbol`, with the coverage gaps that apply to the answer.
    pub fn impact_report(
        &self,
        symbol: &str,
        scope: &ExtractionScope<'_>,
    ) -> Result<ImpactAnswer, QueryRefusal> {
        let observed = self.crate_list();
        let Some(pos) = self.pos(symbol) else {
            return Ok(ImpactAnswer::SymbolNotInStore {
                symbol: symbol.to_string(),
                observed: observed.into_iter().map(|(c, _)| c).collect(),
            });
        };
        let graded = self.impact(pos)?;
        let gaps = self.gaps(pos, &graded, &observed, scope);
        Ok(ImpactAnswer::Found(ImpactReport {
            graded,
            observed,
            gaps,
        }))
    }

    fn gaps(
        &self,
        pos: Pos,
        graded: &Graded,
        observed: &[(String, RunStatus)],
        scope: &ExtractionScope<'_>,
    ) -> Vec<CoverageGap> {
        let mut gaps = Vec::new();
        let public = {
            let w = (pos / 64) as usize;
            self.public()
                .get(w)
                .is_some_and(|word| word >> (pos % 64) & 1 == 1)
        };
        let in_store: BTreeSet<&str> = observed.iter().map(|(c, _)| c.as_str()).collect();
        match scope.dependents {
            Dependents::Declared(deps) => {
                let missing: BTreeSet<String> = deps
                    .iter()
                    .filter(|d| !in_store.contains(**d))
                    .map(|d| d.to_string())
                    .collect();
                if !missing.is_empty() {
                    gaps.push(CoverageGap::ConsumersNotInStore {
                        crates: missing.into_iter().collect(),
                    });
                }
            }
            Dependents::NotDeclared if public => gaps.push(CoverageGap::ConsumersNotDeclared),
            Dependents::NotDeclared => {}
        }
        if scope.targets == Targets::LibraryOnly {
            gaps.push(CoverageGap::TestTargetsNotExtracted);
        }
        let possible = popcount(&graded.possible);
        if possible > 0 {
            gaps.push(CoverageGap::UnresolvedDispatch { possible });
        }
        let unknown = popcount(&graded.unknown);
        if unknown > 0 {
            gaps.push(CoverageGap::DynamicCalls { unknown });
        }
        for (krate, status) in observed {
            if *status == RunStatus::Partial {
                gaps.push(CoverageGap::PartialCrate {
                    krate: krate.clone(),
                });
            }
            match self.crate_residuals(krate) {
                Some(Some(r)) if r.total() > 0 => gaps.push(CoverageGap::Residuals {
                    krate: krate.clone(),
                    count: r.total(),
                }),
                Some(None) => gaps.push(CoverageGap::ResidualsNotRead {
                    krate: krate.clone(),
                }),
                _ => {}
            }
        }
        match scope.sources {
            None => gaps.push(CoverageGap::FreshnessNotChecked),
            Some((store, root)) => gaps.extend(freshness(store, root, observed)),
        }
        gaps
    }
}

/// The rows of a `#schema` TSV, by column name; `None` when the file is absent or has no header.
fn read_schema_rows(path: &Path) -> Option<Vec<std::collections::HashMap<String, String>>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut cols: Option<Vec<String>> = None;
    let mut rows = Vec::new();
    for line in text.lines() {
        if let Some(h) = line.strip_prefix("#schema ") {
            cols = Some(h.split('\t').map(str::to_string).collect());
        } else if !line.is_empty() && !line.starts_with('#') {
            let cols = cols.as_ref()?;
            rows.push(
                cols.iter()
                    .cloned()
                    .zip(line.split('\t').map(str::to_string))
                    .collect(),
            );
        }
    }
    cols.map(|_| rows)
}

/// Stale and missing source files across every store crate. The whole answer is folded from
/// every crate's rows, so a change anywhere in the store can move it.
fn freshness(store: &Path, root: &Path, observed: &[(String, RunStatus)]) -> Vec<CoverageGap> {
    let (mut stale, mut missing, mut gaps) = (BTreeSet::new(), BTreeSet::new(), Vec::new());
    for (krate, _) in observed {
        let Some(rows) = read_schema_rows(&store.join(format!("{krate}.files.tsv"))) else {
            gaps.push(CoverageGap::NoFileList {
                krate: krate.clone(),
            });
            continue;
        };
        for row in rows {
            let (Some(path), Some(want)) = (row.get("path"), row.get("fnv64")) else {
                continue;
            };
            match std::fs::read(root.join(path)) {
                Err(_) => {
                    missing.insert(path.clone());
                }
                Ok(bytes) if format!("{:016x}", fnv1a(&bytes)) != *want => {
                    stale.insert(path.clone());
                }
                Ok(_) => {}
            }
        }
    }
    if !stale.is_empty() {
        gaps.push(CoverageGap::StaleSource {
            files: stale.into_iter().collect(),
        });
    }
    if !missing.is_empty() {
        gaps.push(CoverageGap::MissingSource {
            files: missing.into_iter().collect(),
        });
    }
    gaps
}
