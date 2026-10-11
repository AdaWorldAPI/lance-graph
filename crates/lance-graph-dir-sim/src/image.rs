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
use crate::view::ApplyError;
use ogar_dir_sim::{Version, VersionId, Violation};

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
}
