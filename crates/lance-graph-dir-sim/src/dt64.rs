//! A version's users through the DT64 lane socket, and dir-sim results as DR
//! selections.
//!
//! ```text
//!   VersionStore::view(v) ──► DirLanes  (LaneSource + SelectionSource)
//!   users_with_key / subtree / … ──► Kept ──select──► (sel, Axis)
//!                                    Axis = DR64 | DR256 | n × DR256
//!   BindStore::bind_lanes ──► LaneBinding::selected(field, sel)
//!                         ──► (UserLane, &[u32])  read lane.get(i) per i
//! ```
//!
//! **A cell is a virtual tuple.** A [`UserLane`] holds no column. Reading row
//! `i` asks the version's [`View`] for that user's attribute, so a base value,
//! an overridden value and a user the version created all read correctly, and
//! nothing is copied into a second object universe. `A[i]:B[i]` is two such
//! reads at one known `i`; there is no tuple, row object or join.
//!
//! **The DR axis follows the survivors.** A selection's axis is
//! `Axis::for_survivors(kept.len())`: up to 64 rows one DR64 tile, up to 256 one
//! DR256 tile, more `ceil(n / 256)` DR256 tiles (512 rows are two). The count is
//! a popcount of the result bitmap; choosing the axis decodes no row. The
//! selection's position list, the socket's form (`SelectionSource`), is built
//! once per selection, bounded by its survivors, never per field or per read.
//!
//! **One version, one frame store.** A [`DirLanes`] is built over one version
//! and owns the frame store its frames come from. A binding made over another
//! version's `DirLanes` is refused (`ForeignFrame`) before any lane is
//! borrowed, so a reader cannot read one version's selection against another
//! version's cells.
//!
//! `Scope` is a consistency check between reader, lanes and selections, as
//! everywhere in DT64; it is not authorization.

use lance_graph_contract::ontology::FieldRef;
use lance_graph_dt64::dr::{Axis, Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneSource, SchemaRev, SelAddr, SelectionSource};
use ogar_dir_sim::{Attribute, VersionId};

use crate::{Kept, UserOrdinal, View};

/// Field 0 of the user lanes: the UPN value.
pub const F_UPN: u8 = 0;
/// Field 1 of the user lanes: the primary SMTP value.
pub const F_SMTP: u8 = 1;

/// The schema revision these lanes serve.
pub const LANE_REV: SchemaRev = SchemaRev(1);

/// One user attribute of one version, read cell by cell. It holds no values.
#[derive(Clone, Copy)]
pub struct UserLane<'a, 's> {
    view: &'a View<'s>,
    attr: Attribute,
}

impl UserLane<'_, '_> {
    /// The attribute's value id for the user at `row`; `None` when the user
    /// has no such value, or `row` is past the version's users.
    pub fn get(&self, row: u32) -> Option<u32> {
        let g = self.view.user_guid(UserOrdinal(u16::try_from(row).ok()?))?;
        self.view.attr(&g, self.attr).map(|v| v.0)
    }
}

/// Why a result was not admitted as a selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectRefusal {
    /// The result was computed over another population size than this
    /// version's users (a group result, or another version's users).
    NotUsers {
        /// Rows of the result.
        rows: usize,
        /// Users of this version.
        users: usize,
    },
    /// More selections than a `u16` id addresses.
    Full,
}

/// A selection the source holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selected {
    /// Its id for [`lance_graph_dt64::LaneBinding::selected`].
    pub sel: u16,
    /// The DR axis its survivors take.
    pub axis: Axis,
}

/// A version's users as a DT64 lane and selection source.
pub struct DirLanes<'a, 's> {
    view: &'a View<'s>,
    version: VersionId,
    class: u32,
    scope: Scope,
    frames: FrameStore,
    selections: Vec<Vec<u32>>,
}

impl<'a, 's> DirLanes<'a, 's> {
    /// The users of `view` (version `version`), bound as class `class` for
    /// reader `scope`.
    pub fn new(view: &'a View<'s>, version: VersionId, class: u32, scope: Scope) -> Self {
        Self {
            view,
            version,
            class,
            scope,
            frames: FrameStore::new(FrameRevisions {
                source: version.0,
                relation_map: 0,
                coordinates: version.0,
            }),
            selections: Vec::new(),
        }
    }

    /// The row order the lanes and selections are in: this version's user
    /// ordinals.
    pub fn space(&self) -> CoordSpace {
        CoordSpace {
            population: u64::from(self.class),
            coordinates: self.version.0,
        }
    }

    /// The DR axis the version's whole user population takes.
    pub fn identity_axis(&self) -> Axis {
        Axis::for_survivors(self.view.users_len())
    }

    /// Hold `kept` (a result over this version's users) as a selection.
    ///
    /// # Errors
    ///
    /// [`SelectRefusal`]: the result is over another population, or the
    /// source holds `u16::MAX + 1` selections already.
    pub fn select(&mut self, kept: &Kept) -> Result<Selected, SelectRefusal> {
        let users = self.view.users_len();
        if kept.universe() != users {
            return Err(SelectRefusal::NotUsers {
                rows: kept.universe(),
                users,
            });
        }
        let sel = u16::try_from(self.selections.len()).map_err(|_| SelectRefusal::Full)?;
        let axis = Axis::for_survivors(kept.len());
        // Rows are user ordinals, at most 65,536, so each fits a `u32`.
        self.selections
            .push(kept.rows().into_iter().map(|r| r as u32).collect());
        Ok(Selected { sel, axis })
    }
}

impl LaneSource for DirLanes<'_, '_> {
    type Lane<'b>
        = UserLane<'b, 'b>
    where
        Self: 'b;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        if class != self.class || rev != LANE_REV || field > F_SMTP {
            return None;
        }
        Some(LaneAddr {
            space: self.space(),
            lane: u16::from(field),
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<UserLane<'_, '_>> {
        if addr.space != self.space() {
            return None;
        }
        let attr = match addr.lane {
            0 => Attribute::Upn,
            1 => Attribute::PrimarySmtp,
            _ => return None,
        };
        Some(UserLane {
            view: self.view,
            attr,
        })
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

impl SelectionSource for DirLanes<'_, '_> {
    fn selection_of(&self, sel: u16, _frame: &Frame) -> Option<SelAddr> {
        self.selections.get(usize::from(sel))?;
        Some(SelAddr {
            space: self.space(),
            sel,
            scope: self.scope,
        })
    }

    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        if addr.space != self.space() {
            return None;
        }
        self.selections
            .get(usize::from(addr.sel))
            .map(Vec::as_slice)
    }
}
