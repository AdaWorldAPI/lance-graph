//! Arm Q: nested groups as a DR256 register, users x groups as a bit matrix.
//!
//! The security chain's groups are bound to 256 local slots once (the
//! global-to-local register). Everything after that is a 256 x 256 bit
//! matrix over slots, one `Row` (4 words) per row:
//!
//! * `nest`: slot -> parent slots (security child -> security parent);
//! * `reach`: the reflexive transitive closure of `nest`, computed inside the
//!   tile by Warshall (`reach[i] |= reach[k]` wherever bit `k` of `reach[i]`
//!   is set, for every `k` in order). No closure row is read from outside the
//!   register, so the closure is complete only because EVERY security group
//!   is bound: a chain with more than 256 security groups is refused
//!   ([`Refusal::TooLarge`]), never truncated;
//! * `role`: per slot, the roles its reach carries.
//!
//! [`Qr::bind`] builds only the register. Users are never resolved all at
//! once: nobody asks for every user's nested groups in one go. A page of 256
//! users (a [`Page`]: those users' rows of the membership matrix, the "QR
//! code", plus their roles) is resolved when a question names it
//! ([`Qr::page`], [`Pages`]).
//!
//! Every inner loop runs over slot or user coordinates and tests a bit; no
//! loop scans set bits for a position (the D-RU-6 address rule). Translating
//! a group ordinal to a slot is counted ([`Qr::translations`],
//! [`Page::translations`]). It is frozen-codebook decoding over the same
//! snapshot, and its contract-level witness is still missing (ADMISSION B,
//! `ISS-ADMISSION-PREP-BINDING`).

use crate::focus::{Row, SLOTS};
use crate::pop::Pop;
use crate::prep::{Prepared, Refusal, Witness, bit};

#[inline]
pub(crate) fn row_bit(r: &Row, s: usize) -> bool {
    r[s >> 6] >> (s & 63) & 1 == 1
}

#[inline]
pub(crate) fn set(r: &mut Row, s: usize) {
    r[s >> 6] |= 1 << (s & 63);
}

#[inline]
fn or(a: &mut Row, b: &Row) {
    for k in 0..4 {
        a[k] |= b[k];
    }
}

/// The register: the security chain bound to slots and closed in the tile.
pub struct Qr {
    pub witness: Witness,
    /// Users in the snapshot; pages are `users.div_ceil(256)`.
    pub users: usize,
    /// Slot -> group ordinal.
    pub ids: Vec<u16>,
    /// Group ordinal -> slot + 1; 0 = not a security group.
    slot_of: Vec<u16>,
    /// Slot -> parent slots.
    pub nest: [Row; SLOTS],
    /// Slot -> every slot it reaches upward, itself included.
    pub reach: [Row; SLOTS],
    /// Slot -> roles carried by its reach.
    pub role: [u64; SLOTS],
    /// Role -> the slots that carry it (`role` transposed).
    pub role_slots: [Row; 64],
    /// Ordinal -> slot lookups done while binding the register.
    pub translations: usize,
}

/// One page of 256 users: their rows of the membership matrix and their
/// roles.
pub struct Page {
    /// Row `i`: user `page * 256 + i`'s direct security groups as slots.
    pub member: [Row; SLOTS],
    /// Roles of user `page * 256 + i`.
    pub roles: [u64; SLOTS],
    /// Ordinal -> slot lookups done while resolving this page.
    pub translations: usize,
}

impl Qr {
    /// Binds the security chain to slots and closes it in the tile. Resolves
    /// no user. Refuses a stale `b` and more than 256 security groups.
    pub fn bind(b: &Prepared, current: &Witness) -> Result<Qr, Refusal> {
        b.check(current)?;
        let ids: Vec<u16> = (0..b.groups)
            .filter(|&g| bit(&b.security, g))
            .map(|g| g as u16)
            .collect();
        if ids.len() > SLOTS {
            return Err(Refusal::TooLarge);
        }
        let mut slot_of = vec![0u16; b.groups];
        for (s, &g) in ids.iter().enumerate() {
            slot_of[g as usize] = s as u16 + 1;
        }
        let mut translations = 0;
        let mut nest = [[0u64; 4]; SLOTS];
        for &(c, p) in &b.nested {
            translations += 2;
            let (c, p) = (slot_of[c as usize], slot_of[p as usize]);
            if c != 0 && p != 0 {
                set(&mut nest[c as usize - 1], p as usize - 1);
            }
        }
        let n = ids.len();
        let reach = closure(&nest, n);
        let mut role = [0u64; SLOTS];
        for (i, r) in role.iter_mut().enumerate().take(n) {
            for (k, &g) in ids.iter().enumerate() {
                if row_bit(&reach[i], k) {
                    *r |= b.grants[g as usize];
                }
            }
        }
        let mut role_slots = [[0u64; 4]; 64];
        for (s, &rs) in role.iter().enumerate().take(n) {
            for (r, row) in role_slots.iter_mut().enumerate() {
                if rs >> r & 1 == 1 {
                    set(row, s);
                }
            }
        }
        Ok(Qr {
            witness: b.witness.clone(),
            users: b.users,
            role_slots,
            ids,
            slot_of,
            nest,
            reach,
            role,
            translations,
        })
    }

    pub fn check(&self, current: &Witness) -> Result<(), Refusal> {
        if self.witness == *current {
            Ok(())
        } else {
            Err(Refusal::Stale)
        }
    }

    pub fn slot(&self, g: u16) -> Option<usize> {
        match self.slot_of[g as usize] {
            0 => None,
            s => Some(s as usize - 1),
        }
    }

    pub fn pages(&self) -> usize {
        self.users.div_ceil(SLOTS)
    }

    /// Resolves page `p`: its users' membership rows, then their roles as a
    /// fold over the page's columns. The users holding role `r` are the OR of
    /// the columns of the slots whose role set contains `r`. Refuses a `b`
    /// from another snapshot.
    pub fn page(&self, b: &Prepared, p: usize) -> Result<Page, Refusal> {
        b.check(&self.witness)?;
        let base = p * SLOTS;
        let end = self.users.min(base + SLOTS);
        let mut member = [[0u64; 4]; SLOTS];
        let mut cols = [[0u64; 4]; SLOTS];
        let mut translations = 0;
        for u in base..end {
            for &d in b.user_groups.row(u) {
                translations += 1;
                if let Some(s) = self.slot(d) {
                    set(&mut member[u - base], s);
                    set(&mut cols[s], u - base);
                }
            }
        }
        let mut holders = [[0u64; 4]; 64];
        for (s, c) in cols.iter().enumerate().take(self.ids.len()) {
            for (r, h) in holders.iter_mut().enumerate() {
                if self.role[s] >> r & 1 == 1 {
                    or(h, c);
                }
            }
        }
        let mut roles = [0u64; SLOTS];
        for (i, o) in roles.iter_mut().enumerate().take(end - base) {
            for (r, h) in holders.iter().enumerate() {
                *o |= u64::from(row_bit(h, i)) << r;
            }
        }
        Ok(Page {
            member,
            roles,
            translations,
        })
    }

    /// Effective security groups as slots: a membership row times `reach`.
    pub fn effective(&self, member: &Row) -> Row {
        let mut out = [0u64; 4];
        for k in 0..self.ids.len() {
            if row_bit(member, k) {
                or(&mut out, &self.reach[k]);
            }
        }
        out
    }

    /// Roles from a membership row and the slot roles (the row-times-vector
    /// product a page resolves by columns).
    pub fn roles_live(&self, member: &Row) -> u64 {
        let mut out = 0;
        for (k, &r) in self.role.iter().enumerate().take(self.ids.len()) {
            if row_bit(member, k) {
                out |= r;
            }
        }
        out
    }

    /// The slots whose reach contains security group `g`: a user is
    /// effectively in `g` iff one of its direct slots is in this set. `None`
    /// when `g` is outside the chain.
    pub fn slots_reaching(&self, g: u16) -> Option<Row> {
        let t = self.slot(g)?;
        let mut out = [0u64; 4];
        for (s, r) in self.reach.iter().enumerate().take(self.ids.len()) {
            if row_bit(r, t) {
                set(&mut out, s);
            }
        }
        Some(out)
    }

    /// Masked count over several tiles at once: how many users, in the tiles
    /// selected by `tiles` (one bit per page) and the rows selected by
    /// `rows[p]` inside tile `p`, hold one of the slots in `slots` directly.
    ///
    /// Each selected slot's group is read once (its direct-member list), and
    /// every member landing in a selected tile sets its bit in that tile of
    /// `scratch`. The answer is the popcount of each selected tile AND its
    /// row mask; no page is resolved and no per-user value is built.
    /// `scratch` has one `Row` per page and is left zeroed.
    pub fn count_masked(
        &self,
        b: &Prepared,
        slots: &Row,
        tiles: &[u64],
        rows: &[Row],
        scratch: &mut [Row],
    ) -> Result<u64, Refusal> {
        b.check(&self.witness)?;
        for (s, &g) in self.ids.iter().enumerate() {
            if row_bit(slots, s) {
                for &u in b.group_users.row(g as usize) {
                    let u = u as usize;
                    let p = u / SLOTS;
                    // Branchless: the tile's selection bit is shifted into
                    // place, so an unselected tile receives an OR of zero.
                    // Member lists are in ordinal order, which is random with
                    // respect to tiles, so a branch here mispredicts.
                    let sel = tiles[p >> 6] >> (p & 63) & 1;
                    let i = u % SLOTS;
                    scratch[p][i >> 6] |= sel << (i & 63);
                }
            }
        }
        let mut n = 0u64;
        for (p, (acc, m)) in scratch.iter_mut().zip(rows).enumerate() {
            if tiles[p >> 6] >> (p & 63) & 1 == 1 {
                for k in 0..4 {
                    n += u64::from((acc[k] & m[k]).count_ones());
                }
                *acc = [0; 4];
            }
        }
        Ok(n)
    }

    /// Bytes retained by the register; pages are counted separately.
    pub fn bytes(&self) -> usize {
        self.ids.len() * 2 + self.slot_of.len() * 2 + SLOTS * (32 + 32 + 8) + 64 * 32
    }
}

impl Page {
    /// Roles of the page's user `i`. Unchecked hot path.
    #[inline]
    pub fn roles_of(&self, i: usize) -> u64 {
        self.roles[i]
    }

    pub const BYTES: usize = SLOTS * (32 + 8);
}

/// Pages resolved on first use.
pub struct Pages {
    /// The register version the pages were resolved against. A cached page
    /// never answers for another version.
    pub witness: Witness,
    pages: Vec<Option<Box<Page>>>,
    /// Pages resolved so far.
    pub resolved: usize,
}

impl Pages {
    pub fn new(qr: &Qr) -> Pages {
        Pages {
            witness: qr.witness.clone(),
            pages: (0..qr.pages()).map(|_| None).collect(),
            resolved: 0,
        }
    }

    /// The page holding user `u`, resolved if it was not. Refuses a register
    /// or source rows of another version than the cache's, before returning
    /// a cached page or resolving a new one.
    pub fn of(&mut self, qr: &Qr, b: &Prepared, u: u16) -> Result<&Page, Refusal> {
        qr.check(&self.witness)?;
        b.check(&self.witness)?;
        let p = u as usize / SLOTS;
        if self.pages[p].is_none() {
            self.pages[p] = Some(Box::new(qr.page(b, p)?));
            self.resolved += 1;
        }
        Ok(self.pages[p].as_deref().expect("resolved above"))
    }

    /// Roles of user `u`, resolving its page on first use.
    pub fn roles_of(&mut self, qr: &Qr, b: &Prepared, u: u16) -> Result<u64, Refusal> {
        Ok(self.of(qr, b, u)?.roles_of(u as usize % SLOTS))
    }

    /// Checked: is `u` effectively in security group `g`?
    pub fn ask(
        &mut self,
        qr: &Qr,
        b: &Prepared,
        current: &Witness,
        u: u16,
        g: u16,
    ) -> Result<bool, Refusal> {
        qr.check(current)?;
        let s = qr.slot(g).ok_or(Refusal::OutOfFocus)?;
        let row = self.of(qr, b, u)?.member[u as usize % SLOTS];
        Ok(row_bit(&qr.effective(&row), s))
    }
}

/// Reflexive transitive closure of `nest` over the first `n` slots, by
/// Warshall inside the tile.
fn closure(nest: &[Row; SLOTS], n: usize) -> [Row; SLOTS] {
    let mut r = *nest;
    for (i, row) in r.iter_mut().enumerate().take(n) {
        set(row, i);
    }
    for k in 0..n {
        let rk = r[k];
        for row in r.iter_mut().take(n) {
            if row_bit(row, k) {
                or(row, &rk);
            }
        }
    }
    r
}

/// Per-slot tile pointers into the slot's sorted direct-member list: for
/// slot `s`, `runs[s][p]..runs[s][p + 1]` are its members in tile `p`.
/// Built for a slot the first time a question names it; a query then visits
/// only the selected tiles' runs.
pub struct Runs {
    /// The version the tables were read from; a table never outlives it.
    pub witness: Witness,
    runs: Vec<Option<Box<[u32]>>>,
    /// Slots whose pointer table has been built.
    pub built: usize,
}

impl Runs {
    pub fn new(qr: &Qr) -> Runs {
        Runs {
            witness: qr.witness.clone(),
            runs: (0..qr.ids.len()).map(|_| None).collect(),
            built: 0,
        }
    }

    pub(crate) fn index(&mut self, qr: &Qr, b: &Prepared, s: usize) -> &[u32] {
        if self.runs[s].is_none() {
            let pages = qr.pages();
            let mut off = vec![0u32; pages + 1];
            for &u in b.group_users.row(qr.ids[s] as usize) {
                off[u as usize / SLOTS + 1] += 1;
            }
            for p in 0..pages {
                off[p + 1] += off[p];
            }
            self.runs[s] = Some(off.into_boxed_slice());
            self.built += 1;
        }
        self.runs[s].as_deref().expect("built above")
    }

    /// Same answer as [`Qr::count_masked`], visiting only selected tiles:
    /// per tile, the runs of the slots in `slots` are ORed into one row,
    /// ANDed with the tile's row mask and counted.
    pub fn count(
        &mut self,
        qr: &Qr,
        b: &Prepared,
        slots: &Row,
        tiles: &[u64],
        rows: &[Row],
    ) -> Result<u64, Refusal> {
        b.check(&qr.witness)?;
        qr.check(&self.witness)?;
        let n = qr.ids.len();
        for s in 0..n {
            if row_bit(slots, s) {
                self.index(qr, b, s);
            }
        }
        let mut total = 0u64;
        for (p, m) in rows.iter().enumerate().take(qr.pages()) {
            if tiles[p >> 6] >> (p & 63) & 1 == 0 {
                continue;
            }
            let mut acc = [0u64; 4];
            for s in 0..n {
                if row_bit(slots, s) {
                    let off = self.runs[s].as_deref().expect("indexed above");
                    let members = b.group_users.row(qr.ids[s] as usize);
                    for &u in &members[off[p] as usize..off[p + 1] as usize] {
                        set(&mut acc, u as usize % SLOTS);
                    }
                }
            }
            for k in 0..4 {
                total += u64::from((acc[k] & m[k]).count_ones());
            }
        }
        Ok(total)
    }

    pub fn bytes(&self) -> usize {
        self.runs.iter().flatten().map(|r| r.len() * 4).sum()
    }
}

impl Qr {
    /// Every user of the snapshot.
    pub fn everyone(&self) -> Pop {
        let mut tiles = vec![[0u64; 4]; self.pages()];
        for (p, t) in tiles.iter_mut().enumerate() {
            let n = self.users.saturating_sub(p * SLOTS).min(SLOTS);
            for (k, w) in t.iter_mut().enumerate() {
                let bits = n.saturating_sub(k * 64).min(64);
                *w = if bits == 64 { !0 } else { (1u64 << bits) - 1 };
            }
        }
        Pop {
            witness: self.witness.clone(),
            tiles,
            kept: None,
        }
    }

    /// The users holding one of `slots` directly. With `slots` from
    /// [`Qr::slots_reaching`] that is the users effectively in the group;
    /// from [`Qr::role_slots`], the users holding the role. Each slot's
    /// direct-member list is read once and scattered by position.
    pub fn population(&self, b: &Prepared, slots: &Row) -> Result<Pop, Refusal> {
        b.check(&self.witness)?;
        let mut tiles = vec![[0u64; 4]; self.pages()];
        for (s, &g) in self.ids.iter().enumerate() {
            if row_bit(slots, s) {
                for &u in b.group_users.row(g as usize) {
                    let u = u as usize;
                    set(&mut tiles[u / SLOTS], u % SLOTS);
                }
            }
        }
        Ok(Pop {
            witness: self.witness.clone(),
            tiles,
            kept: None,
        })
    }
}
