//! Which AD objects have their mailbox in Exchange Online, from OGAR's
//! hybrid correspondence fold (`ogar_dir_core::correspond`).
//!
//! A remote mailbox is an AD object whose recipient type says the mailbox
//! lives in the cloud. Whether it does is a separate observation: Exchange
//! Online has a mailbox whose `ExternalDirectoryObjectId` is the Entra id
//! the AD object synchronized to. The fold relates the three families on
//! their GUIDs and never on an address:
//!
//! * forward: the AD source anchor equals the Entra `onPremisesImmutableId`;
//! * backsync: AD `msDS-ExternalDirectoryObjectId` equals the Entra id;
//! * cloud: the Exchange Online `ExternalDirectoryObjectId` equals the
//!   Entra id.
//!
//! [`CloudMailboxes::from_fold`] keeps an AD object when the cloud witness
//! holds and neither AD witness contradicts the Entra object it reached.
//! `Ruler::from_fold` places an Entra id only when the forward anchor or
//! backsync reached exactly that object, so the cloud witness already
//! implies an AD witness; and it leaves contested rows empty (a shared
//! anchor, several targets, a split Entra row, several mailboxes), so an
//! ambiguous match is never read as a mailbox.

use ogar_dir_core::correspond::{fold_aligned, Index, Lanes, Output, Planes, Ruler};
use ogar_dir_core::Guid128;

/// The AD objects (by `objectGUID`) with exactly one Exchange Online
/// mailbox, sorted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CloudMailboxes {
    owners: Vec<Guid128>,
}

impl CloudMailboxes {
    /// Read a completed fold: `out` must be the result of
    /// `correspond::fold` over `lanes` and `index`.
    pub fn from_fold(lanes: &Lanes, index: &Index, out: &Output) -> Self {
        let ruler = Ruler::from_fold(lanes, index, out);
        let mut planes = Planes::for_ruler(&ruler);
        fold_aligned(&ruler, &mut planes);
        let bit = |plane: &[u64], i: usize| plane[i / 64] >> (i % 64) & 1 == 1;
        let mut owners: Vec<Guid128> = (0..ruler.len())
            .filter(|&i| {
                bit(&planes.cloud, i)
                    && !bit(&planes.forward_conflict, i)
                    && !bit(&planes.backsync_conflict, i)
            })
            .map(|i| lanes.ad.owner[i])
            .collect();
        owners.sort_unstable();
        owners.dedup();
        Self { owners }
    }

    /// Whether the AD object `g` has its mailbox in Exchange Online.
    pub fn contains(&self, g: &Guid128) -> bool {
        self.owners.binary_search(g).is_ok()
    }

    /// The AD objects, sorted.
    pub fn owners(&self) -> &[Guid128] {
        &self.owners
    }
}
