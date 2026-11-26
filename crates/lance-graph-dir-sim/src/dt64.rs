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
//!
//! **Licence-state codes through the same socket** (D-DIRSIM-LICENSE-DT64-1).
//! [`LicenseLanes`] serves a version's [`LicensePlanes`] in the row order of
//! that version's [`DirLanes`]: one lane per licence-state code (at most 64),
//! over the version's user ordinals (any number of DR tiles). The code axis
//! and the user axis are two axes; nothing builds a user × code × group
//! table. A lane is a membership plane ([`StatePlane`]), not a value lane:
//! read a row with [`StatePlane::contains`], or hand its words to a mask-RISC
//! program as a mask. A field resolves to a code by the code's identity (its
//! `predicate_iri`), never by position, so a code that moved is a refused
//! binding, not a re-pointed one. The assignment paths behind a bit stay with
//! the planes' owner ([`LicensePlanes::explain`]); the borrowed plane carries
//! none.

use lance_graph_contract::ontology::FieldRef;
use lance_graph_dt64::dr::{Axis, Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneSource, SchemaRev, SelAddr, SelectionSource};
use ogar_dir_core::DN_LEVELS;
use ogar_dir_sim::{Attribute, VersionId};

use lance_graph_contract::class_view::{ClassId, ClassView};
use lance_graph_contract::ontology::DisplayTemplate;

use crate::license::{Coverage, LicensePlanes};
use crate::{Kept, UserOrdinal, View};

/// Field 0 of the user lanes: the UPN value.
pub const F_UPN: u8 = 0;
/// Field 1 of the user lanes: the primary SMTP value.
pub const F_SMTP: u8 = 1;

/// The predicate of the UPN lane: a basis field with this identity binds it,
/// at whatever position the basis puts it.
pub const UPN_IRI: &str = "dir:userPrincipalName";
/// The predicate of the primary SMTP lane.
pub const SMTP_IRI: &str = "dir:primarySmtp";

/// Field 2 of the user lanes: Path1, the first OU level below the naming
/// context. Path`k` is field `F_PATH1 + k - 1`, up to [`DN_LEVELS`] levels.
pub const F_PATH1: u8 = 2;

/// The predicate prefix of the OU path lanes: `dir:ouPath/1` is Path1 (the
/// level directly below the naming context), `dir:ouPath/16` the deepest.
pub const PATH_IRI_PREFIX: &str = "dir:ouPath/";

/// The schema revision these lanes serve. The path lanes are additive: a
/// basis without them binds exactly the lanes it did before.
pub const LANE_REV: SchemaRev = SchemaRev(1);

/// The OU path level a predicate names (`dir:ouPath/3` is level 3), if any:
/// the one parser of the path predicates, for this crate's lanes and for
/// every consumer that reads them.
///
/// One spelling per level: decimal digits with no sign and no leading zero,
/// 1..=16. `dir:ouPath/01`, `/+1`, `/0` and `/17` name no level. It
/// allocates nothing: a lane check calls it for every bound path field.
pub fn path_level(iri: &str) -> Option<u8> {
    let digits = iri.strip_prefix(PATH_IRI_PREFIX)?;
    if digits.starts_with('0') || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let k: u8 = digits.parse().ok()?;
    (1..=DN_LEVELS as u8).contains(&k).then_some(k)
}

/// What a user lane reads.
#[derive(Clone, Copy)]
enum Read {
    Attr(Attribute),
    /// The level code at this 0-based depth of the user's location.
    Path(usize),
}

/// One user column of one version, read cell by cell. It holds no values.
///
/// A path lane reads the user's location (`Dn128`) at one level. Its value is
/// that level's code, which identifies an OU among its siblings only: Path`k`
/// means something together with Path1..Path`k-1`, as a coordinate, and is
/// never a name. The name is the `OuDictionary`'s, at render time. A user with
/// no location, or a shallower one, has no value at that level.
#[derive(Clone, Copy)]
pub struct UserLane<'a, 's> {
    view: &'a View<'s>,
    read: Read,
}

impl UserLane<'_, '_> {
    /// The cell's value for the user at `row`; `None` when the user has no
    /// such value, or `row` is past the version's users.
    pub fn get(&self, row: u32) -> Option<u32> {
        let g = self.view.user_guid(UserOrdinal(u16::try_from(row).ok()?))?;
        match self.read {
            Read::Attr(a) => self.view.attr(&g, a).map(|v| v.0),
            Read::Path(d) => {
                let dn = self.view.location(&g)??;
                dn.codes().get(d).map(|&c| u32::from(c))
            }
        }
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
    /// No licence-state code at that index.
    NoSuchCode(usize),
}

/// Hold `kept` as one more position list of `selections`.
fn hold(
    selections: &mut Vec<Vec<u32>>,
    kept: &Kept,
    users: usize,
) -> Result<Selected, SelectRefusal> {
    if kept.universe() != users {
        return Err(SelectRefusal::NotUsers {
            rows: kept.universe(),
            users,
        });
    }
    let sel = u16::try_from(selections.len()).map_err(|_| SelectRefusal::Full)?;
    let axis = Axis::for_survivors(kept.len());
    // Rows are user ordinals, at most 65,536, so each fits a `u32`.
    selections.push(kept.rows().into_iter().map(|r| r as u32).collect());
    Ok(Selected { sel, axis })
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
        hold(&mut self.selections, kept, self.view.users_len())
    }

    /// The version the lanes are over.
    pub fn version(&self) -> VersionId {
        self.version
    }

    /// The version's user slots (deleted users included).
    pub fn users_len(&self) -> usize {
        self.view.users_len()
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
        _field: u8,
        basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        if class != self.class || rev != LANE_REV {
            return None;
        }
        // By identity, never by position: a basis may order its fields
        // differently, and an unknown predicate has no lane.
        let lane = match basis.predicate_iri.as_str() {
            UPN_IRI => F_UPN,
            SMTP_IRI => F_SMTP,
            iri => F_PATH1 + path_level(iri)? - 1,
        };
        Some(LaneAddr {
            space: self.space(),
            lane: u16::from(lane),
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<UserLane<'_, '_>> {
        if addr.space != self.space() {
            return None;
        }
        let read = match addr.lane {
            0 => Read::Attr(Attribute::Upn),
            1 => Read::Attr(Attribute::PrimarySmtp),
            l => {
                let d = usize::from(l.checked_sub(u16::from(F_PATH1))?);
                if d >= DN_LEVELS {
                    return None;
                }
                Read::Path(d)
            }
        };
        Some(UserLane {
            view: self.view,
            read,
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

/// The field basis of the user class: UPN, primary SMTP, then the OU path
/// levels Path1..Path`levels`, each named by its lane's identity.
///
/// The naming context (the domain) is the version's scope, one value for the
/// whole version, so it is not a row field.
pub struct UserBasis {
    class: ClassId,
    fields: Vec<FieldRef>,
}

impl UserBasis {
    /// The basis for the view class of `class` with `levels` path levels
    /// (clamped to [`DN_LEVELS`]). Built once per binding, cold.
    pub fn new(class: u32, levels: usize) -> Self {
        let mut fields = vec![
            FieldRef::new(UPN_IRI, "UPN"),
            FieldRef::new(SMTP_IRI, "Primary SMTP"),
        ];
        fields.extend(
            (1..=levels.min(DN_LEVELS))
                .map(|k| FieldRef::new(format!("{PATH_IRI_PREFIX}{k}"), format!("Path{k}"))),
        );
        Self {
            class: lance_graph_dt64::view_class(class),
            fields,
        }
    }

    /// Fields in the basis.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the basis has no field (it always has UPN and SMTP).
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl ClassView for UserBasis {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == self.class {
            &self.fields
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

// ---- Licence-state codes (D-DIRSIM-LICENSE-DT64-1) ---------------------------

/// The schema revision the licence-state lanes serve.
pub const LICENSE_REV: SchemaRev = SchemaRev(1);

/// The field basis of a licence-state class: one field per code of a
/// [`LicensePlanes`], in its code order, each named by the code's identity.
///
/// A store binds the class over this basis once and never re-binds it
/// (D5): when a later observation changes the codes, its basis differs, and
/// a binding made over the old one is refused rather than re-pointed.
pub struct LicenseBasis {
    class: ClassId,
    fields: Vec<FieldRef>,
}

impl LicenseBasis {
    /// The basis of `planes`' codes for the view class of `class`. Built once
    /// per binding, cold.
    pub fn of(planes: &LicensePlanes, class: u32) -> Self {
        let fields = (0..planes.states().len())
            .filter_map(|c| planes.predicate_iri(c))
            .map(|iri| FieldRef::new(iri, iri))
            .collect();
        Self {
            class: lance_graph_dt64::view_class(class),
            fields,
        }
    }

    /// Fields in the basis: the number of codes.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the basis has no field.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl ClassView for LicenseBasis {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == self.class {
            &self.fields
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// One licence-state code's plane over a version's user ordinals, borrowed
/// from the planes' owner. A membership mask, one bit per user: never index
/// it as a value lane.
#[derive(Clone, Copy, Debug)]
pub struct StatePlane<'a> {
    words: &'a [u64],
    n_rows: usize,
}

impl<'a> StatePlane<'a> {
    /// Whether user `row` holds the code (and its licences were read). A row
    /// past the version's users holds nothing.
    pub fn contains(&self, row: u32) -> bool {
        let r = row as usize;
        r < self.n_rows && (self.words[r / 64] >> (r % 64)) & 1 == 1
    }

    /// The plane's words, for a mask-RISC program's masks.
    pub fn words(&self) -> &'a [u64] {
        self.words
    }

    /// Rows the plane covers: the version's user slots.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
}

/// Why licence planes were not served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LicenseSourceRefusal {
    /// The planes are over another version than the directory lanes.
    OtherVersion {
        /// The planes' version.
        planes: VersionId,
        /// The directory lanes' version.
        lanes: VersionId,
    },
    /// The planes cover another number of user slots.
    OtherPopulation {
        /// The planes' user slots.
        planes: usize,
        /// The directory lanes' user slots.
        users: usize,
    },
    /// The licence pages are not one whole collection, so the planes cannot
    /// stand for the version's licences.
    Incomplete(Coverage),
}

/// A version's licence planes as a DT64 lane and selection source, in the
/// row order of that version's [`DirLanes`].
pub struct LicenseLanes<'p> {
    planes: &'p LicensePlanes,
    class: u32,
    space: CoordSpace,
    scope: Scope,
    frames: FrameStore,
    selections: Vec<Vec<u32>>,
}

impl<'p> LicenseLanes<'p> {
    /// Serve `planes` as class `class` for reader `scope`, in `dir`'s row
    /// order.
    ///
    /// # Errors
    ///
    /// [`LicenseSourceRefusal`]: the planes are over another version or
    /// population than `dir`, or their pages are not one whole collection.
    pub fn over(
        dir: &DirLanes<'_, '_>,
        planes: &'p LicensePlanes,
        class: u32,
        scope: Scope,
    ) -> Result<Self, LicenseSourceRefusal> {
        if planes.version() != dir.version() {
            return Err(LicenseSourceRefusal::OtherVersion {
                planes: planes.version(),
                lanes: dir.version(),
            });
        }
        if planes.users_len() != dir.users_len() {
            return Err(LicenseSourceRefusal::OtherPopulation {
                planes: planes.users_len(),
                users: dir.users_len(),
            });
        }
        match planes.coverage() {
            Coverage::Complete => {}
            other => return Err(LicenseSourceRefusal::Incomplete(other)),
        }
        let version = planes.version().0;
        Ok(Self {
            planes,
            class,
            space: dir.space(),
            scope,
            frames: FrameStore::new(FrameRevisions {
                source: version,
                relation_map: 0,
                coordinates: version,
            }),
            selections: Vec::new(),
        })
    }

    /// The row order: the directory lanes' user ordinals.
    pub fn space(&self) -> CoordSpace {
        self.space
    }

    /// Hold the users with licence-state code `code` as a selection.
    ///
    /// # Errors
    ///
    /// [`SelectRefusal::NoSuchCode`], or the source is full.
    pub fn select_state(&mut self, code: usize) -> Result<Selected, SelectRefusal> {
        let kept = self
            .planes
            .with_state(code)
            .ok_or(SelectRefusal::NoSuchCode(code))?;
        hold(&mut self.selections, &kept, self.planes.users_len())
    }

    /// Hold `kept` (a result over this version's users) as a selection.
    ///
    /// # Errors
    ///
    /// [`SelectRefusal`] as for [`DirLanes::select`].
    pub fn select(&mut self, kept: &Kept) -> Result<Selected, SelectRefusal> {
        hold(&mut self.selections, kept, self.planes.users_len())
    }
}

impl LaneSource for LicenseLanes<'_> {
    type Lane<'b>
        = StatePlane<'b>
    where
        Self: 'b;

    /// The code whose identity is `basis.predicate_iri`. The field position
    /// is not read: two bases can order their fields differently.
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        if class != self.class || rev != LICENSE_REV {
            return None;
        }
        let code = self.planes.code_of(basis.predicate_iri.as_str())?;
        Some(LaneAddr {
            space: self.space,
            lane: u16::try_from(code).ok()?,
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<StatePlane<'_>> {
        if addr.space != self.space {
            return None;
        }
        Some(StatePlane {
            words: self.planes.plane(usize::from(addr.lane))?,
            n_rows: self.planes.users_len(),
        })
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

impl SelectionSource for LicenseLanes<'_> {
    /// Only under this source's current frame.
    fn selection_of(&self, sel: u16, frame: &Frame) -> Option<SelAddr> {
        self.frames.check(frame).ok()?;
        self.selections.get(usize::from(sel))?;
        Some(SelAddr {
            space: self.space,
            sel,
            scope: self.scope,
        })
    }

    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        if addr.space != self.space {
            return None;
        }
        self.selections
            .get(usize::from(addr.sel))
            .map(Vec::as_slice)
    }
}
