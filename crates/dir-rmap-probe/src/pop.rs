//! Populations over the user axis, bound to one register version (arm Q).
//!
//! A [`Pop`] is one bit per user: bit `i` of tile `p` is user `p * 256 + i`.
//! Populations from the same register line up by position, so filters chain
//! as word-wise AND / AND-NOT and answer with a count. A population carries
//! the witness of the snapshot it was derived from. Combining or folding
//! populations from different snapshots is refused ([`Refusal::Stale`]).
//!
//! A population is a cache of derived state, never a source. It is rebuilt
//! or repaired from the source rows ([`Prepared`]) of the version it claims.
//!
//! * [`Qr::population`] builds one over every tile; [`Qr::population_in`]
//!   only over a [`TeamFocus`]'s tiles, through the per-slot tile pointer
//!   tables ([`Runs`]).
//! * [`TeamFocus`] binds one group's identity (a group ordinal, stable across
//!   versions) to the tiles and rows of its effective members. The tile mask
//!   is locality, not identity: a move between tiles changes the mask, never
//!   the group.
//! * [`Pop::plane`] lends the bits as one `n_rows`-long mask plane, so
//!   `lance-graph-relalg` folds them (`Rel::Plane`) with mask-risc, without a
//!   copy.
//! * [`Pop::repair`] brings a population to a new version: the slot set is
//!   diffed against the rebound register; slots that gained the property
//!   scatter their members in; the members of slots that lost it, and every
//!   user whose direct memberships changed, are RE-DERIVED from their direct
//!   groups. A user who loses one supporting slot keeps the bit while
//!   another remains. Clearing the members of a removed slot instead would be
//!   wrong whenever a second path exists.

use crate::focus::{Row, SLOTS};
use crate::prep::{Prepared, Refusal, Witness};
use crate::qr::{Qr, Runs, row_bit, set};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pop {
    pub witness: Witness,
    pub tiles: Vec<Row>,
    /// The tile mask the population was built for, `None` for every tile.
    /// Outside it the bits are not derived, so a restricted population is
    /// exact only within these tiles, and a repair must keep the same mask.
    pub kept: Option<Vec<u64>>,
}

#[inline]
fn tile_on(tiles: &[u64], p: usize) -> bool {
    tiles[p >> 6] >> (p & 63) & 1 == 1
}

impl Pop {
    pub fn check(&self, current: &Witness) -> Result<(), Refusal> {
        if self.witness == *current {
            Ok(())
        } else {
            Err(Refusal::Stale)
        }
    }

    pub fn and(mut self, o: &Pop) -> Result<Pop, Refusal> {
        o.check(&self.witness)?;
        for (a, b) in self.tiles.iter_mut().zip(&o.tiles) {
            for k in 0..4 {
                a[k] &= b[k];
            }
        }
        Ok(self)
    }

    pub fn and_not(mut self, o: &Pop) -> Result<Pop, Refusal> {
        o.check(&self.witness)?;
        for (a, b) in self.tiles.iter_mut().zip(&o.tiles) {
            for k in 0..4 {
                a[k] &= !b[k];
            }
        }
        Ok(self)
    }

    pub fn or(mut self, o: &Pop) -> Result<Pop, Refusal> {
        o.check(&self.witness)?;
        for (a, b) in self.tiles.iter_mut().zip(&o.tiles) {
            for k in 0..4 {
                a[k] |= b[k];
            }
        }
        Ok(self)
    }

    pub fn count(&self) -> u64 {
        self.tiles
            .iter()
            .flatten()
            .map(|w| u64::from(w.count_ones()))
            .sum()
    }

    /// `self AND o`, counted, without building the intersection.
    pub fn count_and(&self, o: &Pop) -> Result<u64, Refusal> {
        o.check(&self.witness)?;
        let mut n = 0;
        for (a, b) in self.tiles.iter().zip(&o.tiles) {
            for k in 0..4 {
                n += u64::from((a[k] & b[k]).count_ones());
            }
        }
        Ok(n)
    }

    /// The bits as one mask plane over `n_rows` users (`ceil(n_rows / 64)`
    /// words), borrowed. Bits past `n_rows` are zero by construction.
    pub fn plane(&self, n_rows: usize) -> &[u64] {
        &self.tiles.as_flattened()[..n_rows.div_ceil(64)]
    }

    pub fn bytes(&self) -> usize {
        self.tiles.len() * 32
    }

    /// Brings this population, built from `slots0` against the previous
    /// version, to the version of `qr1` / `b1`, whose register gives the
    /// population's slot set as `slots1`. `touched` are the users whose
    /// direct memberships changed between the versions. With `tiles`, only
    /// those tiles are kept (a focus-restricted population).
    ///
    /// Refuses when the slot binding changed (the security chain gained or
    /// lost a group): the old slot sets do not address the same groups, so
    /// the population is rebuilt instead. Refuses ([`Refusal::OutOfFocus`])
    /// a `tiles` other than the mask the population was built for: tiles
    /// leaving the mask would keep stale bits and tiles joining it would stay
    /// empty under the new version. A different focus is a rebuild.
    pub fn repair(
        &mut self,
        qr0: &Qr,
        qr1: &Qr,
        b1: &Prepared,
        (slots0, slots1): (&Row, &Row),
        touched: &[u16],
        tiles: Option<&[u64]>,
    ) -> Result<Repair, Refusal> {
        self.check(&qr0.witness)?;
        b1.check(&qr1.witness)?;
        if qr0.ids != qr1.ids || self.tiles.len() != qr1.pages() {
            return Err(Refusal::Stale);
        }
        if self.kept.as_deref() != tiles {
            return Err(Refusal::OutOfFocus);
        }
        let keep = |u: usize| tiles.is_none_or(|t| tile_on(t, u / SLOTS));
        let mut out = Repair::default();
        for (s, &g) in qr1.ids.iter().enumerate() {
            let (was, is) = (row_bit(slots0, s), row_bit(slots1, s));
            if is && !was {
                out.slots_added += 1;
                for &u in b1.group_users.row(g as usize) {
                    let u = u as usize;
                    if keep(u) {
                        set(&mut self.tiles[u / SLOTS], u % SLOTS);
                        out.scattered += 1;
                    }
                }
            } else if was && !is {
                out.slots_removed += 1;
                for &u in b1.group_users.row(g as usize) {
                    if keep(u as usize) {
                        out.rederived += 1;
                        self.rederive(qr1, b1, slots1, u as usize);
                    }
                }
            }
        }
        for &u in touched {
            if keep(u as usize) {
                out.rederived += 1;
                self.rederive(qr1, b1, slots1, u as usize);
            }
        }
        self.witness = qr1.witness.clone();
        Ok(out)
    }

    /// One user's bit from its direct groups against `slots`.
    fn rederive(&mut self, qr: &Qr, b: &Prepared, slots: &Row, u: usize) {
        let mut on = false;
        for &d in b.user_groups.row(u) {
            if let Some(s) = qr.slot(d) {
                on |= row_bit(slots, s);
            }
        }
        let w = &mut self.tiles[u / SLOTS][(u % SLOTS) >> 6];
        let m = 1u64 << (u % 64);
        if on {
            *w |= m;
        } else {
            *w &= !m;
        }
    }
}

/// What a repair did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Repair {
    pub slots_added: usize,
    pub slots_removed: usize,
    /// Members of added slots set.
    pub scattered: usize,
    /// Users re-derived from their direct groups.
    pub rederived: usize,
}

/// One group's identity bound to the tiles and rows of its effective
/// members, for one version.
#[derive(Clone, Debug)]
pub struct TeamFocus {
    /// The group's ordinal: the identity, stable across versions.
    pub group: u16,
    /// Tile mask: the tiles holding at least one member.
    pub tiles: Vec<u64>,
    /// The members (direct or through nesting).
    pub rows: Pop,
}

impl TeamFocus {
    pub fn tiles_selected(&self) -> usize {
        self.tiles.iter().map(|w| w.count_ones() as usize).sum()
    }
}

impl Qr {
    /// Binds the focus of security group `g`: its effective members and the
    /// tiles they occupy. Reads only the member lists of `g` and the groups
    /// nested in it. Refuses a group outside the chain.
    pub fn team_focus(&self, b: &Prepared, g: u16) -> Result<TeamFocus, Refusal> {
        let slots = self.slots_reaching(g).ok_or(Refusal::OutOfFocus)?;
        let rows = self.population(b, &slots)?;
        let mut tiles = vec![0u64; self.pages().div_ceil(64)];
        for (p, t) in rows.tiles.iter().enumerate() {
            if t.iter().any(|&w| w != 0) {
                tiles[p >> 6] |= 1 << (p & 63);
            }
        }
        Ok(TeamFocus {
            group: g,
            tiles,
            rows,
        })
    }

    /// The users holding one of `slots` directly, only in the focus's tiles
    /// (other tiles stay empty). Reads the selected tiles' runs of each slot
    /// through `runs`, building a slot's pointer table on first use.
    pub fn population_in(
        &self,
        b: &Prepared,
        slots: &Row,
        focus: &TeamFocus,
        runs: &mut Runs,
    ) -> Result<Pop, Refusal> {
        b.check(&self.witness)?;
        focus.rows.check(&self.witness)?;
        self.check(&runs.witness)?;
        let mut tiles = vec![[0u64; 4]; self.pages()];
        for (s, &g) in self.ids.iter().enumerate() {
            if !row_bit(slots, s) {
                continue;
            }
            let members = b.group_users.row(g as usize);
            let off = runs.index(self, b, s);
            for (p, t) in tiles.iter_mut().enumerate() {
                if tile_on(&focus.tiles, p) {
                    for &u in &members[off[p] as usize..off[p + 1] as usize] {
                        set(t, u as usize % SLOTS);
                    }
                }
            }
        }
        Ok(Pop {
            witness: self.witness.clone(),
            tiles,
            kept: Some(focus.tiles.clone()),
        })
    }
}
