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
//!
//! The mailbox's immutable identity is its `ExchangeGuid`. On-premises it is
//! `msExchMailboxGuid`, and for a remote mailbox created with
//! `Enable-RemoteMailbox` it starts **empty**: only backsync fills it, later.
//! The cloud value is read from Exchange Online by `ExternalDirectoryObjectId`
//! ([`CloudMailboxes::with_exchange_guids`]). [`CloudMailboxes::mailbox_guid`]
//! compares the two:
//!
//! * empty on-premises: awaiting backsync; mail is delivered, but the
//!   mailbox cannot be migrated;
//! * equal: migratable;
//! * different: the on-premises value blocks provisioning the cloud
//!   mailbox, so nothing is delivered to it.

use crate::view::View;
use ogar_dir_core::correspond::{fold_aligned, Index, Lanes, Output, Planes, Ruler};
use ogar_dir_core::Guid128;
use ogar_dir_sim::Recipient;

/// The AD objects (by `objectGUID`) with exactly one Exchange Online
/// mailbox, sorted, with the Entra id each one's mailbox links to
/// (`ExternalDirectoryObjectId`) and, once read, that mailbox's
/// `ExchangeGuid`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CloudMailboxes {
    owners: Vec<Guid128>,
    entra: Vec<Guid128>,
    exchange_guid: Vec<Option<Guid128>>,
}

/// How a remote mailbox's on-premises `ExchangeGuid` relates to its Exchange
/// Online mailbox ([`CloudMailboxes::mailbox_guid`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MailboxGuid {
    /// Not a remote mailbox: the cloud comparison does not apply.
    NotRemote,
    /// Exchange Online has no mailbox for the object.
    NoCloudMailbox,
    /// The Exchange Online mailbox's `ExchangeGuid` was not read.
    CloudUnread,
    /// On-premises `msExchMailboxGuid` is still empty (`Enable-RemoteMailbox`
    /// leaves it so until backsync fills it). Delivered; not migratable.
    AwaitingBacksync {
        /// The cloud mailbox's `ExchangeGuid`.
        cloud: Guid128,
    },
    /// Both sides carry the same `ExchangeGuid`: migratable.
    Matched(Guid128),
    /// The values differ: the on-premises value blocks provisioning the
    /// cloud mailbox. Not delivered; not migratable.
    Mismatch {
        /// On-premises `msExchMailboxGuid`.
        on_premises: Guid128,
        /// The cloud mailbox's `ExchangeGuid`.
        cloud: Guid128,
    },
}

impl MailboxGuid {
    /// Whether the mailbox can be migrated: only when both sides carry the
    /// same `ExchangeGuid`.
    pub fn migratable(&self) -> bool {
        matches!(self, Self::Matched(_))
    }
}

impl CloudMailboxes {
    /// Read a completed fold: `out` must be the result of
    /// `correspond::fold` over `lanes` and `index`.
    pub fn from_fold(lanes: &Lanes, index: &Index, out: &Output) -> Self {
        let ruler = Ruler::from_fold(lanes, index, out);
        let mut planes = Planes::for_ruler(&ruler);
        fold_aligned(&ruler, &mut planes);
        let bit = |plane: &[u64], i: usize| plane[i / 64] >> (i % 64) & 1 == 1;
        let mut rows: Vec<(Guid128, Guid128)> = (0..ruler.len())
            .filter(|&i| {
                bit(&planes.cloud, i)
                    && !bit(&planes.forward_conflict, i)
                    && !bit(&planes.backsync_conflict, i)
            })
            .map(|i| (lanes.ad.owner[i], ruler.entra_id[i]))
            .collect();
        rows.sort_unstable();
        rows.dedup();
        let (owners, entra): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        let exchange_guid = vec![None; owners.len()];
        Self {
            owners,
            entra,
            exchange_guid,
        }
    }

    /// Attach the Exchange Online mailboxes' `ExchangeGuid`, each keyed by
    /// its `ExternalDirectoryObjectId`. An id listed with two different
    /// values is ambiguous and stays unread.
    pub fn with_exchange_guids(mut self, cloud: &[(Guid128, Guid128)]) -> Self {
        let mut sorted = cloud.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        for (i, entra) in self.entra.iter().enumerate() {
            let lo = sorted.partition_point(|(e, _)| e < entra);
            let hi = sorted.partition_point(|(e, _)| e <= entra);
            self.exchange_guid[i] = match &sorted[lo..hi] {
                [(_, g)] => Some(*g),
                _ => None,
            };
        }
        self
    }

    /// Whether the AD object `g` has its mailbox in Exchange Online.
    pub fn contains(&self, g: &Guid128) -> bool {
        self.owners.binary_search(g).is_ok()
    }

    /// The AD objects, sorted.
    pub fn owners(&self) -> &[Guid128] {
        &self.owners
    }

    /// How `g`'s on-premises `ExchangeGuid` relates to its Exchange Online
    /// mailbox.
    pub fn mailbox_guid(&self, v: &View<'_>, g: &Guid128) -> MailboxGuid {
        if !matches!(
            v.node_state(g).and_then(|s| s.recipient),
            Some(Recipient::RemoteMailbox(_))
        ) {
            return MailboxGuid::NotRemote;
        }
        let Ok(i) = self.owners.binary_search(g) else {
            return MailboxGuid::NoCloudMailbox;
        };
        let Some(cloud) = self.exchange_guid[i] else {
            return MailboxGuid::CloudUnread;
        };
        match v.exchange_guid(g) {
            None => MailboxGuid::AwaitingBacksync { cloud },
            Some(on_premises) if on_premises == cloud => MailboxGuid::Matched(cloud),
            Some(on_premises) => MailboxGuid::Mismatch { on_premises, cloud },
        }
    }

    /// Whether mail to `g` is delivered to it, with the cloud observed: it
    /// must be a mail recipient ([`View::is_mail_recipient`]), and a remote
    /// mailbox must also have its mailbox in Exchange Online, not blocked by
    /// a mismatched on-premises `ExchangeGuid`. An on-premises mailbox and a
    /// group are decided by the view alone.
    ///
    /// This is the rule [`crate::validate::address_recipient_in`] applies
    /// to the holder of an address; a consumer that also expands groups
    /// applies it to each member, so a list never reaches a remote mailbox
    /// that does not exist.
    pub fn delivers_to(&self, v: &View<'_>, g: &Guid128) -> bool {
        v.is_mail_recipient(g)
            && match v.node_state(g).and_then(|s| s.recipient) {
                Some(Recipient::RemoteMailbox(_)) => {
                    self.contains(g)
                        && !matches!(self.mailbox_guid(v, g), MailboxGuid::Mismatch { .. })
                }
                _ => true,
            }
    }
}
