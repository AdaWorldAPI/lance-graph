//! Licences as DR64 planes over a version's users.
//!
//! The Graph licence pages that `ogar_az::license` decodes become, per tenant,
//! at most [`MAX_LICENSE_STATES`] licence-state codes. A code is one
//! `(sku, via, state)` triple: the SKU, whether the path is direct or through a
//! group, and the reported state. Each code is one bit plane over the version's
//! user ordinals, so one DR64 tile holds every licence state of a tenant, and a
//! licence question is a Quack `Filter` over resident planes, like every other
//! population query here.
//!
//! What the planes keep apart, by construction:
//!
//! * **Not read is not none.** [`LicensePlanes`] has a read plane. A user whose
//!   `licenseAssignmentStates` was absent or `null`, or who is on no licence
//!   page at all, is not read; a user read with an empty list is read and
//!   holds no code. Every holder query is gated by the read plane.
//! * **A path is not a seat.** A user who holds one SKU directly and through a
//!   group sets two codes and is one holder of that SKU.
//! * **Overflow is refused.** A 65th distinct code is
//!   [`LicenseError::TooManyStates`], never a dropped or merged state.
//! * **Capacity stays exact.** Purchased and consumed units are the decoded
//!   `i64` counts, held beside the planes; they are not lane values.
//!
//! The assigning group's identity is not part of a code (that would multiply
//! codes by groups); several group paths to one SKU in one state set one bit.

use lance_graph_mask_risc::{words_for, Foreign, Planes};
use lance_graph_quack::{Agg, Filter, Mask};
use ogar_az::license::{LicensesPage, PathState, SkuCapacity, SkuPage};
use ogar_dir_core::Guid128;

use crate::exec::{self, Kept};
use crate::{UserOrdinal, View};

/// Licence-state codes per tenant: one DR64 tile.
pub const MAX_LICENSE_STATES: usize = 64;

/// How a licence reaches a user.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Via {
    /// Assigned to the user directly.
    Direct,
    /// Inherited from a group.
    Group,
}

/// One licence-state code: a SKU, how it is assigned, and its reported state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LicenseState {
    /// The SKU.
    pub sku: Guid128,
    /// Direct or through a group.
    pub via: Via,
    /// The reported state, as decoded.
    pub state: PathState,
}

/// Why licence planes were not built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LicenseError {
    /// More distinct licence-state codes than one DR64 tile holds.
    TooManyStates {
        /// Distinct codes seen before the limit was passed.
        found: usize,
        /// [`MAX_LICENSE_STATES`].
        limit: usize,
    },
    /// A licence page names a user the version does not have.
    UnknownUser(Guid128),
    /// A user appears on more than one licence page entry.
    DuplicateUser(Guid128),
    /// A SKU appears more than once in the capacity pages.
    DuplicateSku(Guid128),
}

/// A tenant's licences over one version's users.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LicensePlanes {
    n_users: usize,
    /// The users that exist in this version; a deleted user keeps its slot
    /// but is not here.
    live: Vec<u64>,
    read: Vec<u64>,
    states: Vec<LicenseState>,
    planes: Vec<Vec<u64>>,
    capacity: Vec<SkuCapacity>,
}

impl LicensePlanes {
    /// Build the planes from decoded licence and capacity pages.
    ///
    /// # Errors
    ///
    /// [`LicenseError`]: too many codes, a user the version does not know, a
    /// user listed twice, or a SKU listed twice in the capacity pages.
    pub fn build(
        v: &View<'_>,
        licenses: &[LicensesPage],
        skus: &[SkuPage],
    ) -> Result<Self, LicenseError> {
        let n_users = v.users_len();
        let words = words_for(n_users);
        let mut read = vec![0u64; words];
        // A user listed on a page, whether or not its licences were read.
        let mut seen = vec![0u64; words];
        let mut live = vec![0u64; words];
        for row in 0..n_users {
            let o = UserOrdinal(u16::try_from(row).expect("user ordinals are u16"));
            if v.user_guid(o).is_some() {
                live[row / 64] |= 1 << (row % 64);
            }
        }
        let mut states: Vec<LicenseState> = Vec::new();
        let mut planes: Vec<Vec<u64>> = Vec::new();
        for u in licenses.iter().flat_map(|p| &p.users) {
            let o = v
                .user_ordinal(&u.user)
                .ok_or(LicenseError::UnknownUser(u.user))?;
            let row = usize::from(o.0);
            if seen[row / 64] & (1 << (row % 64)) != 0 {
                return Err(LicenseError::DuplicateUser(u.user));
            }
            seen[row / 64] |= 1 << (row % 64);
            let Some(paths) = &u.paths else {
                continue;
            };
            read[row / 64] |= 1 << (row % 64);
            for p in paths {
                let code = LicenseState {
                    sku: p.sku,
                    via: if p.assigned_by_group.is_some() {
                        Via::Group
                    } else {
                        Via::Direct
                    },
                    state: p.state.clone(),
                };
                let i = match states.iter().position(|s| *s == code) {
                    Some(i) => i,
                    None => {
                        if states.len() == MAX_LICENSE_STATES {
                            return Err(LicenseError::TooManyStates {
                                found: states.len() + 1,
                                limit: MAX_LICENSE_STATES,
                            });
                        }
                        states.push(code);
                        planes.push(vec![0u64; words]);
                        states.len() - 1
                    }
                };
                planes[i][row / 64] |= 1 << (row % 64);
            }
        }
        let mut capacity: Vec<SkuCapacity> = Vec::new();
        for s in skus.iter().flat_map(|p| &p.skus) {
            if capacity.iter().any(|c| c.sku == s.sku) {
                return Err(LicenseError::DuplicateSku(s.sku));
            }
            capacity.push(s.clone());
        }
        Ok(Self {
            n_users,
            live,
            read,
            states,
            planes,
            capacity,
        })
    }

    /// The licence-state codes, in first-seen order; the index is the plane.
    pub fn states(&self) -> &[LicenseState] {
        &self.states
    }

    /// The decoded capacity of `sku`, if its capacity was read.
    pub fn capacity(&self, sku: &Guid128) -> Option<&SkuCapacity> {
        self.capacity.iter().find(|c| c.sku == *sku)
    }

    /// Users whose licences were read (an empty list included).
    pub fn read(&self) -> Kept {
        self.run(Filter::plane(Mask(0)))
    }

    /// Users whose licences were not read.
    pub fn not_read(&self) -> Kept {
        self.run(Filter::and([
            Filter::plane(Mask(1)),
            Filter::negate(Filter::plane(Mask(0))),
        ]))
    }

    /// Read users holding `sku` by any path in any state. Each holder is one
    /// row however many paths it has.
    pub fn holders(&self, sku: &Guid128) -> Kept {
        self.run(self.holders_filter(sku))
    }

    /// How many read users hold `sku` by any path in any state.
    pub fn holder_count(&self, sku: &Guid128) -> usize {
        exec::count(
            self.holders_filter(sku),
            &self.planes_with(&self.live).planes(),
            &Foreign::NONE,
        )
    }

    /// Read users holding licence-state code `code` (an index into
    /// [`Self::states`]); `None` for an index past the codes.
    pub fn with_state(&self, code: usize) -> Option<Kept> {
        (code < self.states.len()).then(|| {
            self.run(Filter::and([
                Filter::plane(Mask(0)),
                Filter::plane(Mask(Self::plane_mask(code))),
            ]))
        })
    }

    /// Plane 0 is the read plane, plane 1 the version's existing users, then
    /// one per code.
    fn plane_mask(code: usize) -> u16 {
        u16::try_from(code + 2).expect("at most 64 codes")
    }

    fn holders_filter(&self, sku: &Guid128) -> Filter {
        let held: Vec<Filter> = self
            .states
            .iter()
            .enumerate()
            .filter(|(_, s)| s.sku == *sku)
            .map(|(i, _)| Filter::plane(Mask(Self::plane_mask(i))))
            .collect();
        if held.is_empty() {
            // No code names the SKU: no row holds it.
            return Filter::and([
                Filter::plane(Mask(0)),
                Filter::negate(Filter::plane(Mask(0))),
            ]);
        }
        Filter::and([Filter::plane(Mask(0)), Filter::or(held)])
    }

    fn planes_with<'a>(&'a self, all: &'a [u64]) -> PlaneSet<'a> {
        let mut masks: Vec<&'a [u64]> = Vec::with_capacity(self.planes.len() + 2);
        masks.push(&self.read);
        masks.push(all);
        masks.extend(self.planes.iter().map(Vec::as_slice));
        PlaneSet {
            n_rows: self.n_users,
            masks,
        }
    }

    fn run(&self, filter: Filter) -> Kept {
        let set = self.planes_with(&self.live);
        exec::keep(
            &exec::program(filter, Agg::Rows),
            &set.planes(),
            &Foreign::NONE,
        )
    }
}

/// The resident planes one licence query reads.
struct PlaneSet<'a> {
    n_rows: usize,
    masks: Vec<&'a [u64]>,
}

impl PlaneSet<'_> {
    fn planes(&self) -> Planes<'_> {
        Planes {
            n_rows: self.n_rows,
            masks: &self.masks,
            lanes: &[],
        }
    }
}
