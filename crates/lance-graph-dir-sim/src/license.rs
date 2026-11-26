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
//!
//! Corrections for the DT64 binding (D-DIRSIM-LICENSE-DT64-1, Gate 0):
//!
//! * **A code's index is its content, not its arrival.** Codes are ordered by
//!   `(sku, via, state)`, so reordering Graph pages or users gives the same
//!   code at the same index. Each code also has a stable identity,
//!   [`LicensePlanes::predicate_iri`], which a DT64 binding resolves through;
//!   a later observation that adds a code moves indices, and a binding made
//!   before is refused, not re-pointed.
//! * **The plane is an accelerator, not the evidence.** The decoded pages
//!   are owned here, beside the planes. [`LicensePlanes::paths`] returns a
//!   user's typed assignment paths (with `assignedByGroup`), and
//!   [`LicensePlanes::effective`] the separately collected `assignedLicenses`
//!   SKUs; the two are never equated.
//! * **A partial collection says so.** [`LicensePlanes::coverage`] reads the
//!   pages' `@odata.nextLink`s: a link nobody followed is
//!   [`Coverage::Partial`], never complete coverage.

use lance_graph_mask_risc::{words_for, Foreign, Planes};
use lance_graph_quack::{Agg, Filter, Mask};
use ogar_az::license::{LicensesPage, PathState, SkuCapacity, SkuPage};
use ogar_dir_core::Guid128;

use ogar_az::license::{EffectiveSku, LicensePath, Origin};
use ogar_dir_sim::VersionId;

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
    /// The path did not report `assignedByGroup`: its origin is unknown, and
    /// it is neither direct nor through a group.
    Unreported,
}

impl Via {
    /// How `p` reaches its user, from the origin OGAR decoded.
    pub fn of(p: &LicensePath) -> Via {
        match p.origin {
            Some(Origin::Direct) => Via::Direct,
            Some(Origin::Group(_)) => Via::Group,
            None => Via::Unreported,
        }
    }
}

/// One licence-state code: a SKU, how it is assigned, and its reported state.
///
/// Ordered by `(sku, via, state)`, with states as `Active < Disabled < Error
/// < Other(text)`: that order is the code index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LicenseState {
    /// The SKU.
    pub sku: Guid128,
    /// Direct or through a group.
    pub via: Via,
    /// The reported state, as decoded.
    pub state: PathState,
}

impl LicenseState {
    fn key(&self) -> ([u8; 16], u8, u8, &str) {
        let (rank, text) = match &self.state {
            PathState::Active => (0, ""),
            PathState::Disabled => (1, ""),
            PathState::Error => (2, ""),
            PathState::Other(t) => (3, t.as_str()),
        };
        let via = match self.via {
            Via::Direct => 0,
            Via::Group => 1,
            Via::Unreported => 2,
        };
        (self.sku.0, via, rank, text)
    }

    /// The code's stable identity, the predicate a DT64 field binds. The
    /// state is the Graph text as read (`Other` can never hold `Active`,
    /// `Disabled` or `Error`, so the text is unique) and comes last, so a
    /// colon inside it cannot be confused with a separator.
    fn iri(&self) -> String {
        let via = match self.via {
            Via::Direct => "direct",
            Via::Group => "group",
            Via::Unreported => "unreported",
        };
        let state = match &self.state {
            PathState::Active => "Active",
            PathState::Disabled => "Disabled",
            PathState::Error => "Error",
            PathState::Other(t) => t.as_str(),
        };
        format!(
            "urn:graph:licenseAssignmentState:{}:{via}:{state}",
            self.sku
        )
    }

    fn matches(&self, p: &LicensePath) -> bool {
        p.sku == self.sku && p.state == self.state && Via::of(p) == self.via
    }
}

/// Whether the licence pages are one whole collection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// No page was given.
    NotCollected,
    /// One unbroken `@odata.nextLink` chain from the first page to the last,
    /// covering every page once (or a single page that links nowhere).
    Complete,
    /// A `@odata.nextLink` was not followed: no page ends the collection, or
    /// a link names a page that is not here.
    Partial,
    /// The pages are not one run: more than one end, more than one start, a
    /// cycle, or a page outside the chain.
    Ambiguous,
    /// More than one page and no request URLs: the pages alone cannot show
    /// that no page between them is missing (see
    /// [`LicensePlanes::build_fetched`]).
    Unverified,
}

impl Coverage {
    /// From the pages' links alone. A missing middle page leaves the links
    /// unchanged, so more than one page with exactly one end is
    /// `Unverified`, never `Complete`.
    fn of<'a>(links: impl Iterator<Item = &'a Option<String>>) -> Coverage {
        let (mut pages, mut ends) = (0usize, 0usize);
        for l in links {
            pages += 1;
            if l.is_none() {
                ends += 1;
            }
        }
        match (pages, ends) {
            (0, _) => Coverage::NotCollected,
            (1, 1) => Coverage::Complete,
            (_, 0) => Coverage::Partial,
            (_, 1) => Coverage::Unverified,
            _ => Coverage::Ambiguous,
        }
    }

    /// From the chain: page `k + 1` is the page whose request URL is page
    /// `k`'s `@odata.nextLink`, in any delivery order.
    fn chain(requested: &[String], links: &[&Option<String>]) -> Coverage {
        let n = requested.len();
        if n == 0 {
            return Coverage::NotCollected;
        }
        let page_of = |url: &str| requested.iter().position(|r| r == url);
        if (0..n).any(|i| page_of(&requested[i]) != Some(i)) {
            return Coverage::Ambiguous; // one URL requested twice
        }
        // The first page: the one no other page links to.
        let mut starts = (0..n).filter(|&i| {
            !links
                .iter()
                .any(|l| l.as_deref() == Some(requested[i].as_str()))
        });
        let (Some(first), None) = (starts.next(), starts.next()) else {
            return Coverage::Ambiguous;
        };
        let mut seen = vec![false; n];
        let mut at = first;
        loop {
            seen[at] = true;
            match links[at] {
                None => break,
                Some(next) => match page_of(next) {
                    None => return Coverage::Partial,
                    Some(j) if seen[j] => return Coverage::Ambiguous,
                    Some(j) => at = j,
                },
            }
        }
        if seen.iter().all(|&s| s) {
            Coverage::Complete
        } else {
            Coverage::Ambiguous
        }
    }
}

/// What the source holds for one user's licence property.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence<'a, T> {
    /// The user is on no licence page.
    NotListed,
    /// The user is listed and the property was absent or `null`.
    NotRead,
    /// The property as read; an empty slice is "read, none".
    Read(&'a [T]),
}

/// Why a code was not explained for a user.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExplainRefusal {
    /// No code at that index.
    NoSuchCode(usize),
    /// The user is on no licence page.
    NotListed,
    /// The user's assignment paths were not read.
    NotRead,
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
    /// The version whose users the planes are over.
    version: VersionId,
    n_users: usize,
    /// The users that exist in this version; a deleted user keeps its slot
    /// but is not here.
    live: Vec<u64>,
    read: Vec<u64>,
    states: Vec<LicenseState>,
    planes: Vec<Vec<u64>>,
    /// Parallel to `states`: each code's stable identity.
    iris: Vec<Box<str>>,
    capacity: Vec<SkuCapacity>,
    /// The decoded pages, owned: the evidence the planes summarise.
    pages: Vec<LicensesPage>,
    sku_pages: Vec<SkuPage>,
    /// Per user ordinal: `(page, entry)` of its listing, if listed.
    listed: Vec<Option<(u32, u32)>>,
    /// Parallel to `pages`: each page's request URL, when the caller kept it.
    requested: Option<Vec<String>>,
}

impl LicensePlanes {
    /// Build the planes from decoded licence and capacity pages over the users
    /// of `v`, which is version `version`. The pages are kept as the evidence.
    ///
    /// # Errors
    ///
    /// [`LicenseError`]: too many codes, a user the version does not know, a
    /// user listed twice, or a SKU listed twice in the capacity pages.
    pub fn build(
        v: &View<'_>,
        version: VersionId,
        licenses: Vec<LicensesPage>,
        skus: Vec<SkuPage>,
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
        let mut listed: Vec<Option<(u32, u32)>> = vec![None; n_users];
        let entries = licenses
            .iter()
            .enumerate()
            .flat_map(|(pi, p)| p.users.iter().enumerate().map(move |(ui, u)| (pi, ui, u)));
        for (pi, ui, u) in entries {
            let o = v
                .user_ordinal(&u.user)
                .ok_or(LicenseError::UnknownUser(u.user))?;
            let row = usize::from(o.0);
            if seen[row / 64] & (1 << (row % 64)) != 0 {
                return Err(LicenseError::DuplicateUser(u.user));
            }
            seen[row / 64] |= 1 << (row % 64);
            listed[row] = Some((
                u32::try_from(pi).expect("pages fit u32"),
                u32::try_from(ui).expect("entries fit u32"),
            ));
            let Some(paths) = &u.paths else {
                continue;
            };
            read[row / 64] |= 1 << (row % 64);
            for p in paths {
                let code = LicenseState {
                    sku: p.sku,
                    via: Via::of(p),
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
        // The code index is the code's content: sort the (at most 64) codes
        // and move each plane with its code.
        let mut order: Vec<usize> = (0..states.len()).collect();
        order.sort_by(|&a, &b| states[a].key().cmp(&states[b].key()));
        let mut taken: Vec<Option<(LicenseState, Vec<u64>)>> =
            states.into_iter().zip(planes).map(Some).collect();
        let (states, planes): (Vec<LicenseState>, Vec<Vec<u64>>) = order
            .iter()
            .map(|&i| taken[i].take().expect("each index once"))
            .unzip();
        let iris = states.iter().map(|s| s.iri().into_boxed_str()).collect();
        Ok(Self {
            version,
            n_users,
            live,
            read,
            states,
            planes,
            iris,
            capacity,
            pages: licenses,
            sku_pages: skus,
            listed,
            requested: None,
        })
    }

    /// The licence-state codes in `(sku, via, state)` order; the index is
    /// the plane. Reordering pages or users does not move a code.
    pub fn states(&self) -> &[LicenseState] {
        &self.states
    }

    /// The version whose users the planes are over.
    pub fn version(&self) -> VersionId {
        self.version
    }

    /// Users of the version (slots, deleted ones included).
    pub fn users_len(&self) -> usize {
        self.n_users
    }

    /// The stable identity of code `code`: the predicate a DT64 field binds.
    pub fn predicate_iri(&self, code: usize) -> Option<&str> {
        self.iris.get(code).map(|s| &**s)
    }

    /// The code whose identity is `iri`. At most 64 string compares; no
    /// allocation.
    pub fn code_of(&self, iri: &str) -> Option<usize> {
        self.iris.iter().position(|s| &**s == iri)
    }

    /// Code `code`'s plane over the user ordinals, if it exists: one bit per
    /// read user holding the code.
    pub fn plane(&self, code: usize) -> Option<&[u64]> {
        self.planes.get(code).map(Vec::as_slice)
    }

    /// [`Self::build`] over pages fetched with their request URLs: `(url,
    /// page)` in any order. The URLs let [`Self::coverage`] follow the
    /// `@odata.nextLink` chain and see a missing page.
    ///
    /// # Errors
    ///
    /// As for [`Self::build`].
    pub fn build_fetched(
        v: &View<'_>,
        version: VersionId,
        licenses: Vec<(String, LicensesPage)>,
        skus: Vec<SkuPage>,
    ) -> Result<Self, LicenseError> {
        let (requested, pages): (Vec<String>, Vec<LicensesPage>) = licenses.into_iter().unzip();
        let mut planes = Self::build(v, version, pages, skus)?;
        planes.requested = Some(requested);
        Ok(planes)
    }

    /// Whether the licence pages are one whole collection. Without request
    /// URLs more than one page is at best [`Coverage::Unverified`].
    pub fn coverage(&self) -> Coverage {
        match &self.requested {
            Some(urls) => {
                let links: Vec<&Option<String>> = self.pages.iter().map(|p| &p.next_link).collect();
                Coverage::chain(urls, &links)
            }
            None => Coverage::of(self.pages.iter().map(|p| &p.next_link)),
        }
    }

    /// Whether the capacity pages are one whole collection.
    pub fn capacity_coverage(&self) -> Coverage {
        Coverage::of(self.sku_pages.iter().map(|p| &p.next_link))
    }

    fn listing(&self, o: UserOrdinal) -> Option<&ogar_az::license::UserLicenses> {
        let (pi, ui) = (*self.listed.get(usize::from(o.0))?)?;
        self.pages
            .get(pi as usize)
            .and_then(|p| p.users.get(ui as usize))
    }

    /// The user's `licenseAssignmentStates` as decoded: every path with its
    /// assigning group. This is the evidence; the planes only summarise it.
    pub fn paths(&self, o: UserOrdinal) -> Evidence<'_, LicensePath> {
        match self.listing(o) {
            None => Evidence::NotListed,
            Some(u) => match &u.paths {
                None => Evidence::NotRead,
                Some(p) => Evidence::Read(p),
            },
        }
    }

    /// The user's `assignedLicenses` (effective SKUs) as decoded. A separate
    /// property with its own collection state: it is never derived from, or
    /// substituted for, the assignment paths.
    pub fn effective(&self, o: UserOrdinal) -> Evidence<'_, EffectiveSku> {
        match self.listing(o) {
            None => Evidence::NotListed,
            Some(u) => match &u.effective {
                None => Evidence::NotRead,
                Some(e) => Evidence::Read(e),
            },
        }
    }

    /// The assignment paths behind code `code` for user `o`: for a group
    /// code, one per assigning group. Borrowed from the evidence.
    ///
    /// # Errors
    ///
    /// [`ExplainRefusal`]: no such code, or the user's paths are not held.
    pub fn explain(
        &self,
        o: UserOrdinal,
        code: usize,
    ) -> Result<impl Iterator<Item = &LicensePath>, ExplainRefusal> {
        let state = self
            .states
            .get(code)
            .ok_or(ExplainRefusal::NoSuchCode(code))?;
        match self.paths(o) {
            Evidence::NotListed => Err(ExplainRefusal::NotListed),
            Evidence::NotRead => Err(ExplainRefusal::NotRead),
            Evidence::Read(p) => Ok(p.iter().filter(move |x| state.matches(x))),
        }
    }

    /// How many assignment paths to `sku` the read users of this version
    /// have. Counted over the evidence; a holder with two paths counts two
    /// here and once in [`Self::holder_count`]. Two paths are not two seats.
    pub fn path_count(&self, sku: &Guid128) -> usize {
        (0..self.n_users)
            .filter(|&row| self.live[row / 64] & (1 << (row % 64)) != 0)
            .filter_map(|row| match self.paths(UserOrdinal(row as u16)) {
                Evidence::Read(p) => Some(p.iter().filter(|x| x.sku == *sku).count()),
                _ => None,
            })
            .sum()
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
