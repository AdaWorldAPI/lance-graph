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
//! * `role`: per slot, the roles its reach carries;
//! * the membership matrix ("QR code"): users in pages of 256 rows, each row
//!   the user's direct security groups as slots. A user's effective groups
//!   are its row times `reach`; its roles are resolved once per bind.
//!
//! Every inner loop runs over slot coordinates and tests a bit; no loop
//! scans set bits for a position (the D-RU-6 address rule). The one
//! translation from a group ordinal to a slot happens while binding, and is
//! counted ([`Qr::translations`]). It is frozen-codebook decoding over the
//! same snapshot, and its contract-level witness is still missing
//! (ADMISSION B, `ISS-ADMISSION-PREP-BINDING`).

use crate::focus::{Row, SLOTS};
use crate::prep::{Prepared, Refusal, Witness, bit};

#[inline]
fn row_bit(r: &Row, s: usize) -> bool {
    r[s >> 6] >> (s & 63) & 1 == 1
}

#[inline]
fn set(r: &mut Row, s: usize) {
    r[s >> 6] |= 1 << (s & 63);
}

#[inline]
fn or(a: &mut Row, b: &Row) {
    for k in 0..4 {
        a[k] |= b[k];
    }
}

pub struct Qr {
    pub witness: Witness,
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
    /// User ordinal -> direct security groups as slots (the membership
    /// matrix, page `u / 256`, row `u % 256`).
    pub member: Vec<Row>,
    /// User ordinal -> roles, resolved once per bind by [`page_roles`].
    pub roles: Vec<u64>,
    /// Ordinal -> slot lookups done while binding.
    pub translations: usize,
}

impl Qr {
    /// Binds the security chain to slots and resolves every user's roles.
    /// Refuses a stale `b` and more than 256 security groups.
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
        let mut slot = |g: u16| {
            translations += 1;
            match slot_of[g as usize] {
                0 => None,
                s => Some(s as usize - 1),
            }
        };
        let mut nest = [[0u64; 4]; SLOTS];
        for &(c, p) in &b.nested {
            if let (Some(c), Some(p)) = (slot(c), slot(p)) {
                set(&mut nest[c], p);
            }
        }
        let mut member = vec![[0u64; 4]; b.users];
        for (u, row) in member.iter_mut().enumerate() {
            for &d in b.user_groups.row(u) {
                if let Some(s) = slot(d) {
                    set(row, s);
                }
            }
        }
        let mut grant = [0u64; SLOTS];
        for (s, &g) in ids.iter().enumerate() {
            grant[s] = b.grants[g as usize];
        }

        let n = ids.len();
        let reach = closure(&nest, n);
        let mut role = [0u64; SLOTS];
        for i in 0..n {
            for (k, &g) in grant.iter().enumerate().take(n) {
                if row_bit(&reach[i], k) {
                    role[i] |= g;
                }
            }
        }
        let roles = page_roles(b, &ids, &role, b.users);
        Ok(Qr {
            witness: b.witness.clone(),
            ids,
            slot_of,
            nest,
            reach,
            role,
            member,
            roles,
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

    /// Roles of user `u`. Unchecked hot path: the caller checked the witness.
    #[inline]
    pub fn roles_of(&self, u: u16) -> u64 {
        self.roles[u as usize]
    }

    /// Roles of user `u` recomputed from its membership row and the slot
    /// roles (the row-times-vector product `bind` resolves once).
    pub fn roles_live(&self, u: u16) -> u64 {
        roles_of_row(&self.member[u as usize], &self.role, self.ids.len())
    }

    /// User `u`'s effective security groups as slots: its membership row
    /// times `reach`.
    pub fn effective(&self, u: u16) -> Row {
        let m = &self.member[u as usize];
        let mut out = [0u64; 4];
        for k in 0..self.ids.len() {
            if row_bit(m, k) {
                or(&mut out, &self.reach[k]);
            }
        }
        out
    }

    /// Checked: is `u` effectively in security group `g`?
    pub fn ask(&self, current: &Witness, u: u16, g: u16) -> Result<bool, Refusal> {
        self.check(current)?;
        let s = self.slot(g).ok_or(Refusal::OutOfFocus)?;
        Ok(row_bit(&self.effective(u), s))
    }

    /// Bytes retained: register maps, the three slot tiles, the membership
    /// matrix and the resolved roles.
    pub fn bytes(&self) -> usize {
        self.ids.len() * 2
            + self.slot_of.len() * 2
            + SLOTS * (32 + 32 + 8)
            + self.member.len() * 32
            + self.roles.len() * 8
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

/// Every user's roles, page by page, as a fold over the membership matrix's
/// columns: for a page of 256 users, the users holding role `r` are the OR of
/// the columns of the slots whose role set contains `r`. Column `s` of a page
/// is security group `ids[s]`'s direct users in that page, read from the
/// group's own member row.
fn page_roles(b: &Prepared, ids: &[u16], role: &[u64; SLOTS], users: usize) -> Vec<u64> {
    let pages = users.div_ceil(SLOTS);
    let mut cols = vec![[[0u64; 4]; SLOTS]; pages];
    for (s, &g) in ids.iter().enumerate() {
        for &u in b.group_users.row(g as usize) {
            let u = u as usize;
            set(&mut cols[u / SLOTS][s], u % SLOTS);
        }
    }
    let mut out = vec![0u64; users];
    for (p, col) in cols.iter().enumerate() {
        let mut holders = [[0u64; 4]; 64];
        for (s, c) in col.iter().enumerate().take(ids.len()) {
            for (r, h) in holders.iter_mut().enumerate() {
                if role[s] >> r & 1 == 1 {
                    or(h, c);
                }
            }
        }
        let base = p * SLOTS;
        for (i, o) in out[base..users.min(base + SLOTS)].iter_mut().enumerate() {
            for (r, h) in holders.iter().enumerate() {
                *o |= u64::from(row_bit(h, i)) << r;
            }
        }
    }
    out
}

fn roles_of_row(m: &Row, role: &[u64; SLOTS], n: usize) -> u64 {
    let mut out = 0;
    for (k, &r) in role.iter().enumerate().take(n) {
        if row_bit(m, k) {
            out |= r;
        }
    }
    out
}
