//! A coverage-aware impact report: [`Codegraph::impact`] plus a typed account of what the
//! answer could not see.
//!
//! [`Codegraph::impact`] answers in symbol positions, and that stays as it is. A caller the
//! store never observed has no position, so an empty answer cannot tell "nothing depends on
//! this" from "the dependants were never extracted". This report keeps the impact masks exactly
//! as `impact` returns them and adds a list of [`Gap`]s. Missing coverage is never written into
//! [`Graded::unknown`], which stays the set of observed functions a dynamic call may reach.
//!
//! Some coverage facts cannot be read off the rows: which targets were extracted, which crates
//! depend on the symbol's crate, which regions the extraction skipped, and whether a source file
//! changed since extraction. The caller declares those in a [`Scope`]. A fact it does not
//! declare becomes a gap, never an assumption; a fact it does declare is trusted.
//!
//! [`ImpactReport::closed_in_scope`] is the only "nothing else" claim this module makes, and it
//! is local: no gap is known *within the declared scope*. Like
//! [`crate::diag::Coverage::no_known_gap`], it is never a proof that the harvest saw everything.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use lance_graph_contract::hash::fnv1a;

use crate::artifact::ResidualCounts;
use crate::graph::{Codegraph, Pos, RunStatus};
use crate::query::{Graded, QueryRefusal};

/// Which targets of one crate were extracted into the store. The store does not record this;
/// the caller declares it from how the extractor was run.
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

/// The crates that can call into the queried symbol's crate, as the caller knows them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Consumers {
    /// Not declared: callers outside the store are unknown.
    #[default]
    NotDeclared,
    /// Exactly these crates depend on the symbol's crate (an empty list means none do). One
    /// that is not in the store is reported by name.
    Declared(Vec<String>),
    /// Every dependent crate is in the store, without naming them (for example an unpublished
    /// crate whose whole dependent set was extracted). Every stored crate is then checked.
    AllInStore,
}

/// What the caller asserts about the analysis beyond the store's own rows.
#[derive(Debug, Clone, Default)]
pub struct Scope {
    /// The store directory, for `<crate>.files.tsv`.
    pub store: Option<PathBuf>,
    /// The source root the recorded paths are relative to. Without both `store` and
    /// `source_root`, freshness is not checked, which is itself a gap.
    pub source_root: Option<PathBuf>,
    /// Extracted targets per stored crate name. A crate the answer depends on without an entry
    /// is [`Gap::TargetsUndeclared`].
    pub targets: Vec<(String, Targets)>,
    /// The crates that can call into the symbol's crate.
    pub consumers: Consumers,
    /// Code of a stored crate the extraction did not translate, per crate name: modules passed
    /// to Charon as opaque, cargo features left off, cfg branches not taken. The store cannot
    /// see what is inside, so a caller, callee or definition there has no position.
    pub excluded: Vec<(String, Vec<String>)>,
}

/// Whether the queried symbol has a position with current rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolState {
    /// The symbol is in the store.
    Found(Pos),
    /// No current row names the symbol. Not "no impact".
    Missing {
        /// The symbol's first path segment names a stored crate: the crate was read and does
        /// not define or mention it. When false, the symbol lies outside every stored crate
        /// and nothing was looked at.
        owner_stored: bool,
    },
}

/// One reason the impact answer may be incomplete. Every variant comes from a store fact or a
/// declared scope fact; none is inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gap {
    /// The symbol, or something that reaches it, is `PUBLIC` and the consumers were not
    /// declared: crates outside the store can call it.
    ExternalConsumersUnobserved {
        /// The symbol's crate.
        owner: String,
        /// Other stored crates with a confirmed caller in the answer, sorted.
        observed_consumers: Vec<String>,
    },
    /// Declared consumers whose crates are not in the store; their callers are unobserved.
    ConsumersNotInStore {
        /// Crate names, sorted.
        crates: Vec<String>,
    },
    /// Targets of a crate the answer depends on that were not extracted, so callers there (for
    /// example test functions) have no positions.
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
    /// The declared scope excludes regions of the crate from extraction. Calls inside them have
    /// no rows, and a symbol defined only there is missing without being absent.
    RegionsExcluded {
        /// The crate.
        krate: String,
        /// The excluded regions, as declared.
        regions: Vec<String>,
    },
    /// Callers reach the symbol through trait dispatch (`graded.possible`), which the compiler
    /// did not resolve to this implementation.
    TraitDispatch {
        /// Positions in `graded.possible`.
        possible: u32,
    },
    /// A dynamic call or function pointer may reach the symbol or something it reaches; its
    /// target was not named by the compiler.
    DynamicCalls {
        /// Positions in `graded.unknown`.
        unknown: u32,
        /// `UNRESOLVED_CALL` rows in the whole store.
        store_dynamic_calls: u32,
    },
    /// A crate's harvest recorded residuals, did not finish, or has no residuals file.
    HarvestIncomplete {
        /// The crate.
        krate: String,
        /// Residual counts (zero when only the run status or the missing file is the cause).
        residuals: ResidualCounts,
        /// The CRATE row is absent or not `COMPLETE`.
        partial: bool,
        /// No `<crate>.residuals.tsv` was read.
        no_residuals_file: bool,
    },
    /// Recorded source files whose content no longer matches the store.
    StaleSource {
        /// The crate.
        krate: String,
        /// Paths relative to the source root, sorted.
        files: Vec<String>,
    },
    /// Recorded source files that are absent from the source root.
    MissingSource {
        /// The crate.
        krate: String,
        /// Paths relative to the source root, sorted.
        files: Vec<String>,
    },
    /// Rust files under a directory the extraction read from that it did not read itself:
    /// modules behind a disabled feature or cfg, other targets' roots, files never declared as
    /// modules. Their code has no rows.
    SourceFilesUnread {
        /// The crate.
        krate: String,
        /// The unread `.rs` files, relative to the source root, sorted.
        files: Vec<String>,
    },
    /// The crate has no `<crate>.files.tsv`, so freshness cannot be checked.
    FilesUnrecorded {
        /// The crate.
        krate: String,
    },
    /// The crate's `<crate>.files.tsv` has no `#schema` line or a row without `path` or `fnv64`,
    /// so its sources could not be compared.
    InvalidFileList {
        /// The crate.
        krate: String,
    },
    /// No store directory or source root was given; freshness was not checked.
    FreshnessUnchecked,
}

/// [`Codegraph::impact`] with its coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactReport {
    /// Exactly what [`Codegraph::impact`] returns (empty masks when the symbol is missing).
    pub graded: Graded,
    /// Whether the symbol has a position with current rows.
    pub symbol: SymbolState,
    /// Every crate in the store with its extractor status, sorted by name.
    pub observed: Vec<(String, RunStatus)>,
    /// Every known reason the answer may be incomplete.
    pub gaps: Vec<Gap>,
}

impl ImpactReport {
    /// The symbol is in the store and no gap is known within the declared scope. This is a
    /// local statement: it never claims anything about crates, targets or revisions outside
    /// the scope, and it is not a proof of completeness.
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

fn popcount(mask: &[u64]) -> u32 {
    mask.iter().map(|w| w.count_ones()).sum()
}

/// What one crate's file list says about its sources.
#[derive(Default)]
struct FileGaps {
    stale: Vec<String>,
    missing: Vec<String>,
    unread: Vec<String>,
    invalid: bool,
}

/// The file gaps of `krate`, or `None` when the crate has no files table.
///
/// Stale: a recorded file whose digest differs under `root`. Missing: a recorded file that is
/// not there. Unread: a `.rs` file below the top directory of a recorded path that the table
/// does not list; a recorded file at the root names no directory, so nothing is scanned for it.
fn file_gaps(store: &Path, root: &Path, krate: &str) -> Option<FileGaps> {
    let table = std::fs::read_to_string(store.join(format!("{krate}.files.tsv"))).ok()?;
    let mut out = FileGaps::default();
    let mut cols: Option<(usize, usize)> = None;
    let mut recorded = BTreeSet::new();
    let mut tops = BTreeSet::new();
    for line in table.lines() {
        if let Some(h) = line.strip_prefix("#schema ") {
            let names: Vec<&str> = h.split('\t').collect();
            cols = names
                .iter()
                .position(|c| *c == "path")
                .zip(names.iter().position(|c| *c == "fnv64"));
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((path_col, hash_col)) = cols else {
            out.invalid = true;
            continue;
        };
        let fields: Vec<&str> = line.split('\t').collect();
        let (Some(path), Some(want)) = (fields.get(path_col), fields.get(hash_col)) else {
            out.invalid = true;
            continue;
        };
        recorded.insert((*path).to_owned());
        if let Some((top, _)) = path.split_once('/') {
            if top != "." && top != ".." {
                tops.insert(top.to_owned());
            }
        }
        match std::fs::read(root.join(path)) {
            Err(_) => out.missing.push((*path).to_owned()),
            Ok(bytes) if format!("{:016x}", fnv1a(&bytes)) != *want => {
                out.stale.push((*path).to_owned());
            }
            Ok(_) => {}
        }
    }
    if cols.is_none() {
        out.invalid = true;
    }
    for top in &tops {
        rust_files(root, top, &mut out.unread);
    }
    out.unread.retain(|f| !recorded.contains(f));
    out.stale.sort();
    out.missing.sort();
    out.unread.sort();
    Some(out)
}

/// Every `.rs` file below `root/rel`, as paths relative to `root`.
fn rust_files(root: &Path, rel: &str, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(root.join(rel)) else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let path = format!("{rel}/{name}");
        match e.file_type() {
            Ok(t) if t.is_dir() => rust_files(root, &path, out),
            Ok(t) if t.is_file() && name.ends_with(".rs") => out.push(path),
            _ => {}
        }
    }
}

/// Freshness and unread-file gaps of `krates` under the declared store and source root;
/// [`Gap::FreshnessUnchecked`] when either is missing.
fn freshness_gaps<'a>(
    krates: impl IntoIterator<Item = &'a String>,
    scope: &Scope,
    gaps: &mut Vec<Gap>,
) {
    let (Some(store), Some(root)) = (&scope.store, &scope.source_root) else {
        gaps.push(Gap::FreshnessUnchecked);
        return;
    };
    for k in krates {
        let Some(f) = file_gaps(store, root, k) else {
            gaps.push(Gap::FilesUnrecorded { krate: k.clone() });
            continue;
        };
        if f.invalid {
            gaps.push(Gap::InvalidFileList { krate: k.clone() });
        }
        if !f.stale.is_empty() {
            gaps.push(Gap::StaleSource {
                krate: k.clone(),
                files: f.stale,
            });
        }
        if !f.missing.is_empty() {
            gaps.push(Gap::MissingSource {
                krate: k.clone(),
                files: f.missing,
            });
        }
        if !f.unread.is_empty() {
            gaps.push(Gap::SourceFilesUnread {
                krate: k.clone(),
                files: f.unread,
            });
        }
    }
}

/// The declared-scope gaps of one stored crate: its targets and its excluded regions.
fn scope_gaps(k: &str, scope: &Scope, gaps: &mut Vec<Gap>) {
    match scope.targets.iter().find(|(n, _)| n == k) {
        None => gaps.push(Gap::TargetsUndeclared {
            krate: k.to_owned(),
        }),
        Some((_, t)) => {
            let missing = t.missing();
            if missing.any() {
                gaps.push(Gap::TargetsNotExtracted {
                    krate: k.to_owned(),
                    missing,
                });
            }
        }
    }
    if let Some((_, regions)) = scope.excluded.iter().find(|(n, r)| n == k && !r.is_empty()) {
        gaps.push(Gap::RegionsExcluded {
            krate: k.to_owned(),
            regions: regions.clone(),
        });
    }
}

impl Codegraph {
    /// [`Codegraph::impact`] of `symbol`, with the gaps of `scope`.
    ///
    /// Gaps are reported for the crates the answer depends on: the symbol's own crate, every
    /// stored crate holding a function in the answer, and the stored crates that can be
    /// consumers. Those are the declared ones, or every stored crate when the consumers are not
    /// named and the symbol or one of its callers is public. The masks are those of `impact`; this method never adds a position for an
    /// unobserved caller.
    pub fn impact_report(&self, symbol: &str, scope: &Scope) -> Result<ImpactReport, QueryRefusal> {
        let observed = self.crate_list();
        let stored: BTreeSet<&str> = observed.iter().map(|(c, _)| c.as_str()).collect();
        let owner = owner_of(symbol).to_owned();
        let owner_stored = stored.contains(owner.as_str());
        let mut gaps = Vec::new();

        // A position alone is not enough: it survives a refresh that removed every row.
        let Some(g) = self.pos(symbol).filter(|_| self.has_rows(symbol)) else {
            // A missing symbol in a stored crate is absent only from what was extracted:
            // undeclared or missing targets, excluded regions, an incomplete harvest and unread
            // files are each a place it may still be defined.
            if owner_stored {
                scope_gaps(&owner, scope, &mut gaps);
                self.harvest_gaps(&owner, &observed, &mut gaps);
                freshness_gaps([&owner], scope, &mut gaps);
            }
            let z = vec![0u64; self.words()];
            return Ok(ImpactReport {
                graded: Graded {
                    confirmed: z.clone(),
                    possible: z.clone(),
                    unknown: z,
                },
                symbol: SymbolState::Missing { owner_stored },
                observed,
                gaps,
            });
        };
        let graded = self.impact(g)?;

        // Crates the answer depends on: the owner, plus every stored crate holding a graded
        // function.
        let mut involved: BTreeSet<String> = BTreeSet::new();
        if owner_stored {
            involved.insert(owner.clone());
        }
        for m in [&graded.confirmed, &graded.possible, &graded.unknown] {
            for p in bits(m) {
                if let Some(k) = self.symbol(p).map(owner_of) {
                    if stored.contains(k) {
                        involved.insert(k.to_owned());
                    }
                }
            }
        }
        let observed_consumers: BTreeSet<String> = bits(&graded.confirmed)
            .filter_map(|p| self.symbol(p).map(owner_of))
            .filter(|k| *k != owner && stored.contains(k))
            .map(str::to_owned)
            .collect();

        // The consumer frontier: the symbol, or anything that reaches it, is public. External
        // callers of a public caller are affected too.
        let frontier_public = has(self.public(), g)
            || [&graded.confirmed, &graded.possible, &graded.unknown]
                .iter()
                .any(|m| m.iter().zip(self.public()).any(|(a, b)| a & b != 0));
        match &scope.consumers {
            Consumers::NotDeclared => {
                if frontier_public {
                    // Any stored crate may be a consumer too, so each one's coverage counts.
                    involved.extend(stored.iter().map(|k| (*k).to_owned()));
                    gaps.push(Gap::ExternalConsumersUnobserved {
                        owner: owner.clone(),
                        observed_consumers: observed_consumers.into_iter().collect(),
                    });
                }
            }
            Consumers::Declared(deps) => {
                // A declared consumer may call from code it did not extract and so have no
                // position here; its own coverage decides the answer.
                let mut absent = BTreeSet::new();
                for d in deps {
                    if stored.contains(d.as_str()) {
                        involved.insert(d.clone());
                    } else {
                        absent.insert(d.clone());
                    }
                }
                if !absent.is_empty() {
                    gaps.push(Gap::ConsumersNotInStore {
                        crates: absent.into_iter().collect(),
                    });
                }
            }
            Consumers::AllInStore => {
                if frontier_public {
                    involved.extend(stored.iter().map(|k| (*k).to_owned()));
                }
            }
        }

        for k in &involved {
            scope_gaps(k, scope, &mut gaps);
        }

        let possible = popcount(&graded.possible);
        if possible > 0 {
            gaps.push(Gap::TraitDispatch { possible });
        }
        let unknown = popcount(&graded.unknown);
        let store_dynamic_calls = self.coverage().dynamic_calls;
        if unknown > 0 || (store_dynamic_calls > 0 && self.dynamic_may_reach(g)) {
            gaps.push(Gap::DynamicCalls {
                unknown,
                store_dynamic_calls,
            });
        }

        for k in &involved {
            self.harvest_gaps(k, &observed, &mut gaps);
        }
        freshness_gaps(&involved, scope, &mut gaps);

        Ok(ImpactReport {
            graded,
            symbol: SymbolState::Found(g),
            observed,
            gaps,
        })
    }

    /// [`Gap::HarvestIncomplete`] for `k` when its harvest recorded residuals, did not finish,
    /// or has no residuals file.
    fn harvest_gaps(&self, k: &str, observed: &[(String, RunStatus)], gaps: &mut Vec<Gap>) {
        let (residuals, no_residuals_file) = match self.crate_residuals(k) {
            Some(Some(r)) => (r, false),
            _ => (ResidualCounts::default(), true),
        };
        let partial = observed
            .iter()
            .any(|(c, s)| c == k && *s == RunStatus::Partial);
        if residuals.total() > 0 || no_residuals_file || partial {
            gaps.push(Gap::HarvestIncomplete {
                krate: k.to_owned(),
                residuals,
                partial,
                no_residuals_file,
            });
        }
    }

    /// Whether a dynamic call may reach `g`: `g` is an impl or referenced.
    fn dynamic_may_reach(&self, g: Pos) -> bool {
        has(self.impl_subjects(), g) || has(self.referenced(), g)
    }
}
