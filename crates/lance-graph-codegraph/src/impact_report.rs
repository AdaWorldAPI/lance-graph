//! Coverage-aware impact: [`Codegraph::impact`] plus a typed account of what
//! the answer could not see.
//!
//! [`Codegraph::impact`] answers over symbol positions. A caller the store
//! never observed has no position, so an empty answer cannot tell "nothing
//! depends on this" from "the dependants were never extracted". This report
//! keeps the impact masks exactly as `impact` returns them and adds a list of
//! [`Gap`]s, each a fact the store or the declared [`Scope`] actually holds.
//! Missing coverage is never written into `Graded.unknown`, which stays the set
//! of observed functions a dynamic call may reach.
//!
//! [`ImpactReport::closed_in_scope`] is the only "nothing else" claim this
//! module makes, and it is local: no gap is known *within the declared scope*.
//! It is never a statement about code outside the stored crates.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::artifact::{fnv64_hex, ResidualCounts};
use crate::graph::{Codegraph, Pos};
use crate::query::{Graded, QueryRefusal};

/// Which targets of one crate were extracted into the store. The store does
/// not record this; the caller declares it from how the extractor was run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Targets {
    /// The library target.
    pub lib: bool,
    /// Integration tests and `#[cfg(test)]` unit tests.
    pub tests: bool,
    /// Examples.
    pub examples: bool,
    /// Benches.
    pub benches: bool,
}

impl Targets {
    /// The library target only (Charon's default `charon cargo` run).
    pub const LIB_ONLY: Targets = Targets {
        lib: true,
        tests: false,
        examples: false,
        benches: false,
    };

    /// Every target kind.
    pub const ALL: Targets = Targets {
        lib: true,
        tests: true,
        examples: true,
        benches: true,
    };

    fn missing(self) -> Targets {
        Targets {
            lib: !self.lib,
            tests: !self.tests,
            examples: !self.examples,
            benches: !self.benches,
        }
    }

    fn any(self) -> bool {
        self.lib || self.tests || self.examples || self.benches
    }
}

/// What the caller asserts about the analysis beyond the store's own rows.
#[derive(Debug, Clone, Default)]
pub struct Scope {
    /// The store directory, for `<crate>.files.tsv`.
    pub store: Option<PathBuf>,
    /// The source root the recorded paths are relative to. `None` means
    /// freshness was not checked, which is itself a gap.
    pub source_root: Option<PathBuf>,
    /// Extracted targets per stored crate name. A stored crate without an
    /// entry is `TargetsUndeclared`.
    pub targets: Vec<(String, Targets)>,
    /// The caller asserts that every crate which can call the symbol's crate
    /// is in the store (for example an unpublished crate whose whole
    /// dependent set was extracted). Without this, a public symbol's external
    /// callers are unobserved.
    pub consumers_closed: bool,
}

/// Whether the queried symbol has a position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolState {
    /// The symbol is in the store.
    Found(Pos),
    /// No stored row names the symbol.
    Missing {
        /// The symbol's first path segment names a stored crate: the crate
        /// was read and does not define or mention it. When false, the symbol
        /// lies outside every stored crate and nothing was looked at.
        owner_stored: bool,
    },
}

/// One reason the impact answer may be incomplete. Every variant comes from
/// a store fact or a declared scope fact; none is inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gap {
    /// The symbol is `PUBLIC` and the consumer set is not declared closed:
    /// crates outside the store can call it. `observed_consumers` lists the
    /// stored crates other than the owner that already call into it.
    ExternalConsumersUnobserved {
        /// The symbol's crate.
        owner: String,
        /// Other stored crates with a confirmed caller in the answer.
        observed_consumers: Vec<String>,
    },
    /// Targets of the owning crate that were not extracted, so callers there
    /// (for example test functions) have no positions.
    TargetsNotExtracted {
        /// The crate.
        krate: String,
        /// The target kinds that were not extracted.
        missing: Targets,
    },
    /// No target declaration for a stored crate the answer depends on.
    TargetsUndeclared {
        /// The crate.
        krate: String,
    },
    /// A dynamic call (`UNRESOLVED_CALL`) may reach the symbol or something
    /// it reaches; its target was not named by the compiler.
    UnresolvedDispatch {
        /// `UNRESOLVED_CALL` rows in the store.
        dynamic_calls: u32,
    },
    /// The owning crate's harvest recorded residuals, did not finish, or has
    /// no residuals file.
    HarvestIncomplete {
        /// The crate.
        krate: String,
        /// Residual counts (zero when only the run status is the cause).
        residuals: ResidualCounts,
        /// The CRATE row is not `COMPLETE`.
        partial: bool,
        /// No `<crate>.residuals.tsv` was read.
        no_residuals_file: bool,
    },
    /// A recorded source file changed or vanished since extraction.
    StaleSource {
        /// The crate.
        krate: String,
        /// The changed or missing files, as recorded.
        files: Vec<String>,
    },
    /// The crate has no `<crate>.files.tsv`, so freshness cannot be checked.
    FilesUnrecorded {
        /// The crate.
        krate: String,
    },
    /// No source root was given; freshness was not checked.
    FreshnessUnchecked,
}

/// [`Codegraph::impact`] with its coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactReport {
    /// Exactly what [`Codegraph::impact`] returns (empty masks when missing).
    pub graded: Graded,
    /// Whether the symbol has a position.
    pub symbol: SymbolState,
    /// Every known reason the answer may be incomplete.
    pub gaps: Vec<Gap>,
}

impl ImpactReport {
    /// The symbol is in the store and no gap is known within the declared
    /// scope. This is a local statement: it never claims anything about
    /// crates, targets or revisions outside the scope.
    pub fn closed_in_scope(&self) -> bool {
        matches!(self.symbol, SymbolState::Found(_)) && self.gaps.is_empty()
    }
}

fn owner_of(symbol: &str) -> &str {
    symbol.split("::").next().unwrap_or(symbol)
}

fn has(mask: &[u64], p: Pos) -> bool {
    mask.get(p as usize / 64)
        .is_some_and(|w| w >> (p % 64) & 1 == 1)
}

fn bits(mask: &[u64]) -> impl Iterator<Item = Pos> + '_ {
    (0..mask.len() * 64)
        .filter(|&p| mask[p / 64] >> (p % 64) & 1 == 1)
        .filter_map(|p| Pos::try_from(p).ok())
}

/// Recorded files of `krate` whose digest differs under `root`, or `None`
/// when the crate has no files table.
fn stale_files(store: &Path, root: &Path, krate: &str) -> Option<Vec<String>> {
    let table = std::fs::read_to_string(store.join(format!("{krate}.files.tsv"))).ok()?;
    let mut stale = Vec::new();
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let mut f = line.split('\t');
        let (Some(path), Some(digest)) = (f.next(), f.next()) else {
            stale.push(line.to_owned());
            continue;
        };
        match std::fs::read(root.join(path)) {
            Ok(bytes) if fnv64_hex(&bytes) == digest => {}
            _ => stale.push(path.to_owned()),
        }
    }
    Some(stale)
}

impl Codegraph {
    /// [`Codegraph::impact`] of `symbol`, with the gaps of `scope`.
    ///
    /// Gaps are reported for the symbol's own crate and for every stored crate
    /// holding a function in the answer. The masks are those of `impact`;
    /// this method never adds a position for an unobserved caller.
    pub fn impact_report(&self, symbol: &str, scope: &Scope) -> Result<ImpactReport, QueryRefusal> {
        let stored = self.stored_crates();
        let Some(g) = self.pos(symbol) else {
            let z = vec![0u64; self.words()];
            return Ok(ImpactReport {
                graded: Graded {
                    confirmed: z.clone(),
                    possible: z.clone(),
                    unknown: z,
                },
                symbol: SymbolState::Missing {
                    owner_stored: stored.contains(owner_of(symbol)),
                },
                gaps: Vec::new(),
            });
        };
        let graded = self.impact(g)?;
        let owner = owner_of(symbol).to_owned();
        let mut gaps = Vec::new();

        // Crates the answer depends on: the owner, plus every stored crate
        // holding a graded function.
        let mut involved: BTreeSet<String> = BTreeSet::new();
        if stored.contains(owner.as_str()) {
            involved.insert(owner.clone());
        }
        let mut consumers: BTreeSet<String> = BTreeSet::new();
        for m in [&graded.confirmed, &graded.possible, &graded.unknown] {
            for p in bits(m) {
                if let Some(s) = self.symbol(p) {
                    let k = owner_of(s);
                    if stored.contains(k) {
                        involved.insert(k.to_owned());
                    }
                }
            }
        }
        for p in bits(&graded.confirmed) {
            if let Some(s) = self.symbol(p) {
                let k = owner_of(s);
                if k != owner && stored.contains(k) {
                    consumers.insert(k.to_owned());
                }
            }
        }

        if has(self.public(), g) && !scope.consumers_closed {
            gaps.push(Gap::ExternalConsumersUnobserved {
                owner: owner.clone(),
                observed_consumers: consumers.into_iter().collect(),
            });
        }

        for k in &involved {
            match scope.targets.iter().find(|(n, _)| n == k) {
                None => gaps.push(Gap::TargetsUndeclared { krate: k.clone() }),
                Some((_, t)) => {
                    let missing = t.missing();
                    if missing.any() {
                        gaps.push(Gap::TargetsNotExtracted {
                            krate: k.clone(),
                            missing,
                        });
                    }
                }
            }
        }

        let dynamic_calls = self.coverage().dynamic_calls;
        let unknown_reached = graded.unknown.iter().any(|&w| w != 0);
        if unknown_reached || (dynamic_calls > 0 && self.dynamic_may_reach(g)) {
            gaps.push(Gap::UnresolvedDispatch { dynamic_calls });
        }

        for k in &involved {
            let (residuals, no_file) = self.residuals_of_crate(k);
            let partial = self.crate_status(k) == Some(crate::graph::RunStatus::Partial);
            if residuals.total() > 0 || no_file || partial {
                gaps.push(Gap::HarvestIncomplete {
                    krate: k.clone(),
                    residuals,
                    partial,
                    no_residuals_file: no_file,
                });
            }
        }

        match (&scope.store, &scope.source_root) {
            (Some(store), Some(root)) => {
                for k in &involved {
                    match stale_files(store, root, k) {
                        None => gaps.push(Gap::FilesUnrecorded { krate: k.clone() }),
                        Some(files) if !files.is_empty() => gaps.push(Gap::StaleSource {
                            krate: k.clone(),
                            files,
                        }),
                        Some(_) => {}
                    }
                }
            }
            _ => gaps.push(Gap::FreshnessUnchecked),
        }

        Ok(ImpactReport {
            graded,
            symbol: SymbolState::Found(g),
            gaps,
        })
    }

    /// Whether a dynamic call may reach `g`: `g` is an impl or referenced.
    fn dynamic_may_reach(&self, g: Pos) -> bool {
        has(self.impl_subjects(), g) || has(self.referenced(), g)
    }
}
