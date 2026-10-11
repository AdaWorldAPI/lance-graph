//! A version history as owned data: what persisting a [`VersionStore`]
//! writes, and what restoring one reads.
//!
//! The image holds the store's inputs, never its derived planes:
//!
//! * each root as the [`Observation`] it was built from. Snapshot lanes are
//!   derived and lossy (a nickname keeps only its comparison form there), and
//!   their layout grows as planes are added;
//! * the dictionary as raw values in [`ValueId`](crate::ValueId) order.
//!   Comparison keys and routing aliases are recomputed by interning, so the
//!   same order reproduces every id the changes refer to;
//! * every [`Version`] verbatim: a simulated version's delta is what the
//!   rule produced, so restoring never re-runs a rule;
//! * tags and recorded verdicts.
//!
//! No I/O here: a storage crate writes and reads the image.
//!
//! [`VersionStore`]: crate::VersionStore

use crate::snapshot::{BuildError, Observation};
use crate::store::REVERT;
use crate::store::{SimError, VersionStore};
use crate::view::ApplyError;
use crate::view::View;
use ogar_dir_sim::ValueId;
use ogar_dir_sim::{Change, EvidenceRef, Origin, RuleId, Version, VersionId, Violation};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// A version history as owned data ([`crate::VersionStore::image`],
/// [`crate::VersionStore::restore`]).
#[derive(Clone, Debug, Default)]
pub struct StoreImage {
    /// The dictionary: raw values in id order.
    pub values: Vec<String>,
    /// Each root and the observation it was built from, by version id.
    pub roots: Vec<(VersionId, Observation)>,
    /// Every version, in id order.
    pub versions: Vec<Version>,
    /// Tags, sorted by name.
    pub tags: Vec<(String, VersionId)>,
    /// Recorded verdicts, by version id.
    pub verdicts: Vec<(VersionId, Vec<Violation>)>,
}

/// Why an image could not be restored. Nothing is half-restored: the error
/// is returned instead of a store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestoreError {
    /// Value `index` appears twice in the dictionary, so ids would shift.
    DuplicateValue(usize),
    /// Versions must be listed densely in id order; `index` holds another id.
    VersionOrder {
        /// Position in the list.
        index: usize,
        /// The id found there.
        found: VersionId,
    },
    /// A root version without its observation.
    MissingRoot(VersionId),
    /// An observation for a version that is not a root.
    NotARoot(VersionId),
    /// A version derived from a version listed after it, or not at all.
    UnknownParent(VersionId),
    /// A root's observation does not build.
    Build(VersionId, BuildError),
    /// A version's delta does not apply to its parent.
    Apply(VersionId, ApplyError),
    /// Building the roots interned a value the dictionary did not list, so
    /// the image's ids are not the ones its changes were made with.
    IncompleteDictionary,
    /// A tag or verdict names a version the image does not hold.
    UnknownVersion(VersionId),
    /// Simulated versions whose rule the registry does not know, all of
    /// them, in version order. A live store needs every rule resolved.
    UnknownRules(Vec<UnknownRule>),
}

/// A rule's identity as persisted: its name as text and its version.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoredRule {
    /// The rule's stable name.
    pub name: String,
    /// The rule's version.
    pub version: u16,
}

impl From<RuleId> for StoredRule {
    fn from(r: RuleId) -> Self {
        Self {
            name: r.name.to_string(),
            version: r.version,
        }
    }
}

/// What produced a version, as persisted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoredOrigin {
    /// Read from reality.
    Observed {
        /// Which observer.
        source: String,
        /// When it was read (unix ms).
        observed_at_ms: i64,
    },
    /// Produced by a rule.
    Simulated {
        /// The rule.
        rule: StoredRule,
        /// The input it ran on.
        evidence: Vec<EvidenceRef>,
    },
}

/// A version as persisted: [`Version`] with its rule as text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredVersion {
    /// This version.
    pub id: VersionId,
    /// The version it was derived from.
    pub parent: Option<VersionId>,
    /// What produced it.
    pub origin: StoredOrigin,
    /// The changes it introduced, verbatim.
    pub delta: Vec<Change>,
}

impl From<Version> for StoredVersion {
    fn from(v: Version) -> Self {
        Self {
            id: v.id,
            parent: v.parent,
            origin: match v.origin {
                Origin::Observed {
                    source,
                    observed_at_ms,
                } => StoredOrigin::Observed {
                    source,
                    observed_at_ms,
                },
                Origin::Simulated { rule, evidence } => StoredOrigin::Simulated {
                    rule: rule.into(),
                    evidence,
                },
            },
            delta: v.delta,
        }
    }
}

/// A [`StoreImage`] as a storage crate writes it: every rule as text, so
/// nothing in it borrows from the program that wrote it.
#[derive(Clone, Debug, Default)]
pub struct StoredImage {
    /// The dictionary: raw values in id order.
    pub values: Vec<String>,
    /// Each root and the observation it was built from.
    pub roots: Vec<(VersionId, Observation)>,
    /// Every version, in id order.
    pub versions: Vec<StoredVersion>,
    /// Tags, sorted by name.
    pub tags: Vec<(String, VersionId)>,
    /// Recorded verdicts, by version id.
    pub verdicts: Vec<(VersionId, Vec<Violation>)>,
}

impl From<StoreImage> for StoredImage {
    fn from(i: StoreImage) -> Self {
        Self {
            values: i.values,
            roots: i.roots,
            versions: i.versions.into_iter().map(Into::into).collect(),
            tags: i.tags,
            verdicts: i.verdicts,
        }
    }
}

/// The rules a restore may resolve a [`StoredRule`] to.
///
/// A stored rule resolves only to the registered rule with the same name
/// and the same version. Another version of the same rule is a different
/// behaviour, so it is never taken in its place, and nothing is guessed
/// from a name alone. [`REVERT`] is always registered: the store itself
/// records it.
#[derive(Clone, Debug)]
pub struct RuleRegistry {
    rules: BTreeSet<RuleId>,
}

impl Default for RuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleRegistry {
    /// A registry holding the store's own rule, [`REVERT`].
    pub fn new() -> Self {
        Self {
            rules: BTreeSet::from([REVERT]),
        }
    }

    /// Register `rule`.
    #[must_use]
    pub fn with(mut self, rule: RuleId) -> Self {
        self.rules.insert(rule);
        self
    }

    /// The registered rule with exactly `rule`'s name and version.
    pub fn resolve(&self, rule: &StoredRule) -> Option<RuleId> {
        self.rules
            .iter()
            .find(|r| r.name == rule.name && r.version == rule.version)
            .copied()
    }
}

/// A simulated version whose rule a [`RuleRegistry`] does not know.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownRule {
    /// The version.
    pub version: VersionId,
    /// Its rule as stored.
    pub rule: StoredRule,
}

impl StoredImage {
    /// The image with every rule resolved through `rules`, or every version
    /// whose rule is unknown.
    ///
    /// # Errors
    ///
    /// [`RestoreError::UnknownRules`], listing each unresolved version.
    pub fn resolve(self, rules: &RuleRegistry) -> Result<StoreImage, RestoreError> {
        let (image, unknown) = self.resolve_with(rules);
        if unknown.is_empty() {
            Ok(image)
        } else {
            Err(RestoreError::UnknownRules(unknown))
        }
    }

    /// Resolve each rule; an unknown one is recorded as [`UNRESOLVED`] and
    /// returned in the list.
    fn resolve_with(self, rules: &RuleRegistry) -> (StoreImage, Vec<UnknownRule>) {
        let mut unknown = Vec::new();
        let versions = self
            .versions
            .into_iter()
            .map(|v| {
                let origin = match v.origin {
                    StoredOrigin::Observed {
                        source,
                        observed_at_ms,
                    } => Origin::Observed {
                        source,
                        observed_at_ms,
                    },
                    StoredOrigin::Simulated { rule, evidence } => {
                        let resolved = rules.resolve(&rule).unwrap_or_else(|| {
                            unknown.push(UnknownRule {
                                version: v.id,
                                rule: rule.clone(),
                            });
                            UNRESOLVED
                        });
                        Origin::Simulated {
                            rule: resolved,
                            evidence,
                        }
                    }
                };
                Version {
                    id: v.id,
                    parent: v.parent,
                    origin,
                    delta: v.delta,
                }
            })
            .collect();
        (
            StoreImage {
                values: self.values,
                roots: self.roots,
                versions,
                tags: self.tags,
                verdicts: self.verdicts,
            },
            unknown,
        )
    }
}

/// The rule a [`History`] records for a version whose stored rule it could
/// not resolve. Never registered and never resolved to: it marks "not
/// known here", and the stored identity stays in [`History::rule`].
const UNRESOLVED: RuleId = RuleId {
    name: "Unresolved",
    version: 0,
};

/// A restored history that can only be read.
///
/// For inspecting a history whose rules are not all known to this program:
/// every version's delta is stored verbatim, so reading a version never
/// needs its rule. The store inside cannot be reached mutably, so nothing
/// can be observed, simulated, reverted or promoted on top of an unknown
/// rule. [`History::rule`] gives each simulated version's rule as stored.
#[derive(Debug)]
pub struct History {
    store: VersionStore,
    rules: BTreeMap<VersionId, StoredRule>,
    unresolved: Vec<UnknownRule>,
}

impl History {
    /// Restore `image` for reading, resolving what `rules` knows.
    ///
    /// # Errors
    ///
    /// [`RestoreError`] other than `UnknownRules`: the image is damaged.
    pub fn restore(image: StoredImage, rules: &RuleRegistry) -> Result<Self, RestoreError> {
        let stored: BTreeMap<VersionId, StoredRule> = image
            .versions
            .iter()
            .filter_map(|v| match &v.origin {
                StoredOrigin::Simulated { rule, .. } => Some((v.id, rule.clone())),
                StoredOrigin::Observed { .. } => None,
            })
            .collect();
        let (image, unresolved) = image.resolve_with(rules);
        Ok(Self {
            store: VersionStore::restore(image)?,
            rules: stored,
            unresolved,
        })
    }

    /// The versions whose rule was not resolved, in version order.
    pub fn unresolved(&self) -> &[UnknownRule] {
        &self.unresolved
    }

    /// A simulated version's rule as stored; `None` for an observation or
    /// an unknown version.
    pub fn rule(&self, v: VersionId) -> Option<&StoredRule> {
        self.rules.get(&v)
    }

    /// The version record. For an unresolved version the origin's rule is a
    /// placeholder named `"Unresolved"`; [`History::rule`] has the stored one.
    pub fn version(&self, v: VersionId) -> Option<&Version> {
        self.store.version(v)
    }

    /// The directory as of `v`.
    ///
    /// # Errors
    ///
    /// [`SimError::UnknownVersion`].
    pub fn view(&self, v: VersionId) -> Result<View<'_>, SimError> {
        self.store.view(v)
    }

    /// The changes from `a` to `b`.
    ///
    /// # Errors
    ///
    /// As [`VersionStore::diff`].
    pub fn diff(&self, a: VersionId, b: VersionId) -> Result<Vec<Change>, SimError> {
        self.store.diff(a, b)
    }

    /// The version a tag names.
    pub fn tag(&self, name: &str) -> Option<VersionId> {
        self.store.tag(name)
    }

    /// The verdict recorded for `v`.
    pub fn verdict(&self, v: VersionId) -> Option<&[Violation]> {
        self.store.verdict(v)
    }

    /// The text of a dictionary value.
    pub fn value(&self, v: ValueId) -> Option<&str> {
        self.store.value(v)
    }

    /// A value's id, if the history holds it.
    pub fn lookup(&self, s: &str) -> Option<ValueId> {
        self.store.lookup(s)
    }
}
