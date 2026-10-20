//! Arms B and B+: membership laid out by ordinal, prepared once.
//!
//! **B** ([`Prepared`]): user -> direct groups and group -> direct users as
//! CSR, the nesting pairs as child -> parents and parent -> children CSR, and
//! the group property planes. Walks read a CSR row at a reached coordinate.
//!
//! **B+** ([`Compiled`]): B plus one closure row per group and chain (every
//! group, the security chain, the mail chain), computed once by a fixpoint
//! over the nesting CSR in the style of #1634's walk_mask, and the roles a
//! group's SID chain carries. A user's effective groups are then the OR of
//! the closure rows of the user's direct groups.
//!
//! Source of the edges: dir-sim's membership lanes are `pub(crate)`, and
//! this probe changes no production code, so B is built from the same
//! records dir-sim ingested (plus the same change list), translated through
//! dir-sim's public ordinals. A production B would read the lanes; its
//! build would skip that translation, so the build times here are an upper
//! bound. The equivalence tests check B against dir-sim's own walks.
//!
//! Address translation (GUID -> ordinal, fixture id -> ordinal) happens only
//! in preparation, counted. Whether preparation may translate at all is
//! `ISS-ADMISSION-PREP-BINDING`, still open.

use crate::fixture::Fixture;
use lance_graph_dir_sim::{groups_where, GroupProperty, GroupWhere, View};
use ndarray::simd::mask_or_assign;
use ogar_dir_sim::VersionId;

pub fn words(n: usize) -> usize {
    n.div_ceil(64)
}

#[inline]
pub fn bit(p: &[u64], i: usize) -> bool {
    p[i >> 6] >> (i & 63) & 1 == 1
}

#[inline]
pub fn set(p: &mut [u64], i: usize) {
    p[i >> 6] |= 1 << (i & 63);
}

/// The set bits of a plane, ascending.
pub fn ones(p: &[u64]) -> impl Iterator<Item = usize> + '_ {
    p.iter().enumerate().flat_map(|(wi, &w)| {
        let mut b = w;
        std::iter::from_fn(move || {
            if b == 0 {
                return None;
            }
            let i = b.trailing_zeros() as usize;
            b &= b - 1;
            Some(wi * 64 + i)
        })
    })
}

/// What a prepared structure was built against. Ordinals are valid inside
/// one version only, so any difference refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Witness {
    pub version: VersionId,
    pub users: usize,
    pub groups: usize,
}

impl Witness {
    pub fn of(version: VersionId, v: &View<'_>) -> Self {
        Witness {
            version,
            users: v.users_len(),
            groups: v.groups_len(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Built against another version.
    Stale,
    /// The question names a group outside the focus.
    OutOfFocus,
}

/// A compressed row table: `row(i)` is sorted.
#[derive(Clone, Debug)]
pub struct Csr {
    off: Vec<u32>,
    val: Vec<u16>,
}

impl Csr {
    /// From `(row, value)` pairs; duplicates are kept once.
    pub fn build(rows: usize, pairs: &mut [(u16, u16)]) -> Csr {
        pairs.sort_unstable();
        let mut off = vec![0u32; rows + 1];
        let mut val = Vec::with_capacity(pairs.len());
        let mut last = None;
        for &(r, v) in pairs.iter() {
            if last == Some((r, v)) {
                continue;
            }
            last = Some((r, v));
            off[r as usize + 1] += 1;
            val.push(v);
        }
        for i in 0..rows {
            off[i + 1] += off[i];
        }
        Csr { off, val }
    }
    #[inline]
    pub fn row(&self, i: usize) -> &[u16] {
        &self.val[self.off[i] as usize..self.off[i + 1] as usize]
    }
    pub fn bytes(&self) -> usize {
        self.off.len() * 4 + self.val.len() * 2
    }
}

/// Fixture id -> dir-sim ordinal, read once through the public API.
pub struct Ordinals {
    pub user: Vec<u16>,
    pub group: Vec<u16>,
}

impl Ordinals {
    pub fn of(f: &Fixture, v: &View<'_>) -> Self {
        Ordinals {
            user: f
                .users
                .iter()
                .map(|g| v.user_ordinal(g).expect("fixture user exists").0)
                .collect(),
            group: f
                .groups
                .iter()
                .map(|g| v.group_ordinal(g).expect("fixture group exists").0)
                .collect(),
        }
    }
}

/// Which nesting chain a walk follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    /// Every group (`members_transitive`, `groups_transitive`).
    All,
    /// Security groups only (`security_identifiers`).
    Security,
    /// Mail-enabled groups only.
    Mail,
}

pub const CHAINS: [Chain; 3] = [Chain::All, Chain::Security, Chain::Mail];

/// Arm B.
pub struct Prepared {
    pub witness: Witness,
    pub users: usize,
    pub groups: usize,
    pub user_groups: Csr,
    pub group_users: Csr,
    /// `(child, parent)`, sorted: the source of `up` and `down`.
    pub nested: Vec<(u16, u16)>,
    pub up: Csr,
    pub down: Csr,
    /// Group property planes, from dir-sim's own `groups_where`.
    pub security: Vec<u64>,
    pub mail: Vec<u64>,
    /// Roles granted to each group, as a bit set (role ids < 64).
    pub grants: Vec<u64>,
}

impl Prepared {
    pub fn build(
        v: &View<'_>,
        witness: Witness,
        ord: &Ordinals,
        direct: &[(u32, u32)],
        nested: &[(u32, u32)],
        grants: &[(u32, u8)],
    ) -> Self {
        let (users, groups) = (v.users_len(), v.groups_len());
        let mut ug: Vec<(u16, u16)> = direct
            .iter()
            .map(|&(u, g)| (ord.user[u as usize], ord.group[g as usize]))
            .collect();
        let mut gu: Vec<(u16, u16)> = ug.iter().map(|&(u, g)| (g, u)).collect();
        let mut nest: Vec<(u16, u16)> = nested
            .iter()
            .map(|&(c, p)| (ord.group[c as usize], ord.group[p as usize]))
            .collect();
        nest.sort_unstable();
        nest.dedup();
        let plane = |p: GroupProperty| {
            let mut out = vec![0u64; words(groups)];
            for i in groups_where(v, &GroupWhere::Is(p)).rows() {
                set(&mut out, i);
            }
            out
        };
        let mut g = vec![0u64; groups];
        for &(gr, role) in grants {
            g[ord.group[gr as usize] as usize] |= 1 << role;
        }
        let (up, down) = nest_csr(groups, &nest);
        Prepared {
            witness,
            users,
            groups,
            user_groups: Csr::build(users, &mut ug),
            group_users: Csr::build(groups, &mut gu),
            nested: nest,
            up,
            down,
            security: plane(GroupProperty::SecurityEnabled),
            mail: plane(GroupProperty::MailEnabled),
            grants: g,
        }
    }

    pub fn check(&self, current: &Witness) -> Result<(), Refusal> {
        if self.witness == *current {
            Ok(())
        } else {
            Err(Refusal::Stale)
        }
    }

    pub fn admit(&self, chain: Chain) -> Option<&[u64]> {
        match chain {
            Chain::All => None,
            Chain::Security => Some(&self.security),
            Chain::Mail => Some(&self.mail),
        }
    }

    /// Direct membership: one binary search in the user's row.
    pub fn is_member(&self, u: u16, g: u16) -> bool {
        self.user_groups.row(u as usize).binary_search(&g).is_ok()
    }

    /// Groups `u` is in through `chain`, as a plane over group ordinals.
    pub fn groups_of(&self, u: u16, chain: Chain, out: &mut [u64], queue: &mut Vec<u16>) {
        let admit = self.admit(chain);
        let ok = |g: u16| admit.is_none_or(|a| bit(a, g as usize));
        out.fill(0);
        queue.clear();
        for &d in self.user_groups.row(u as usize) {
            if ok(d) && !bit(out, d as usize) {
                set(out, d as usize);
                queue.push(d);
            }
        }
        while let Some(g) = queue.pop() {
            for &p in self.up.row(g as usize) {
                if ok(p) && !bit(out, p as usize) {
                    set(out, p as usize);
                    queue.push(p);
                }
            }
        }
    }

    /// Users in `g` through `chain`, as a plane over user ordinals.
    pub fn members_of(
        &self,
        g: u16,
        chain: Chain,
        reached: &mut [u64],
        out: &mut [u64],
        queue: &mut Vec<u16>,
    ) {
        let admit = self.admit(chain);
        let ok = |g: u16| admit.is_none_or(|a| bit(a, g as usize));
        reached.fill(0);
        out.fill(0);
        queue.clear();
        if !ok(g) {
            return;
        }
        set(reached, g as usize);
        queue.push(g);
        while let Some(h) = queue.pop() {
            for &u in self.group_users.row(h as usize) {
                set(out, u as usize);
            }
            for &c in self.down.row(h as usize) {
                if ok(c) && !bit(reached, c as usize) {
                    set(reached, c as usize);
                    queue.push(c);
                }
            }
        }
    }

    /// Roles through the SID chain.
    pub fn roles_of(&self, u: u16, scratch: &mut [u64], queue: &mut Vec<u16>) -> u64 {
        self.groups_of(u, Chain::Security, scratch, queue);
        ones(scratch).fold(0, |a, g| a | self.grants[g])
    }

    /// Rebuild the user and group membership tables after a change list.
    pub fn set_direct(&mut self, ord: &Ordinals, direct: &[(u32, u32)]) {
        let mut ug: Vec<(u16, u16)> = direct
            .iter()
            .map(|&(u, g)| (ord.user[u as usize], ord.group[g as usize]))
            .collect();
        let mut gu: Vec<(u16, u16)> = ug.iter().map(|&(u, g)| (g, u)).collect();
        self.user_groups = Csr::build(self.users, &mut ug);
        self.group_users = Csr::build(self.groups, &mut gu);
    }

    /// Add one nesting pair and rebuild the nesting tables.
    pub fn add_nesting(&mut self, c: u16, p: u16) {
        if let Err(i) = self.nested.binary_search(&(c, p)) {
            self.nested.insert(i, (c, p));
        }
        let (up, down) = nest_csr(self.groups, &self.nested);
        self.up = up;
        self.down = down;
    }

    pub fn bytes(&self) -> usize {
        self.user_groups.bytes()
            + self.group_users.bytes()
            + self.nested.len() * 4
            + self.up.bytes()
            + self.down.bytes()
            + (self.security.len() + self.mail.len() + self.grants.len()) * 8
    }
}

fn nest_csr(groups: usize, nested: &[(u16, u16)]) -> (Csr, Csr) {
    let mut up: Vec<(u16, u16)> = nested.to_vec();
    let mut down: Vec<(u16, u16)> = nested.iter().map(|&(c, p)| (p, c)).collect();
    (Csr::build(groups, &mut up), Csr::build(groups, &mut down))
}

/// Arm B+.
pub struct Compiled {
    pub b: Prepared,
    /// Words per closure row.
    pub w: usize,
    /// Per chain: row `g` = the groups `g` reaches upward, `g` included
    /// when it passes the chain.
    pub up: [Vec<u64>; 3],
    /// Per chain: row `g` = the groups reachable downward from `g`.
    pub down: [Vec<u64>; 3],
    /// Roles carried by the SID chain starting at each group.
    pub role_sec: Vec<u64>,
    /// Fixpoint passes of the last full compile, per chain (up, down).
    pub passes: [(usize, usize); 3],
    /// `b.up` / `b.down` no longer match `b.nested` (after `add_nesting`).
    pub b_nest_stale: bool,
}

fn ci(chain: Chain) -> usize {
    chain as usize
}

/// Reflexive closure over `adj`, restricted to admitted groups: a fixpoint
/// of `row[g] |= row[n]` for every admitted neighbour `n` of an admitted `g`.
/// Rows are read only at coordinates named by the CSR, as #1634 reads
/// `delta` at the adjacency's set bits.
fn closure(n: usize, w: usize, adj: &Csr, admit: Option<&[u64]>) -> (Vec<u64>, usize) {
    let ok = |g: usize| admit.is_none_or(|a| bit(a, g));
    let mut c = vec![0u64; n * w];
    for g in 0..n {
        if ok(g) {
            set(&mut c[g * w..g * w + w], g);
        }
    }
    let mut passes = 0;
    loop {
        passes += 1;
        let mut changed = false;
        for g in 0..n {
            if !ok(g) {
                continue;
            }
            for &p in adj.row(g) {
                let p = p as usize;
                if p == g || !ok(p) {
                    continue;
                }
                for k in 0..w {
                    let v = c[p * w + k];
                    let old = c[g * w + k];
                    if old | v != old {
                        c[g * w + k] = old | v;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            return (c, passes);
        }
    }
}

impl Compiled {
    pub fn compile(b: Prepared) -> Self {
        let n = b.groups;
        let w = words(n);
        let mut up: [Vec<u64>; 3] = Default::default();
        let mut down: [Vec<u64>; 3] = Default::default();
        let mut passes = [(0, 0); 3];
        for chain in CHAINS {
            let admit = b.admit(chain);
            let (u, pu) = closure(n, w, &b.up, admit);
            let (d, pd) = closure(n, w, &b.down, admit);
            up[ci(chain)] = u;
            down[ci(chain)] = d;
            passes[ci(chain)] = (pu, pd);
        }
        let mut c = Compiled {
            b,
            w,
            up,
            down,
            role_sec: vec![0; n],
            passes,
            b_nest_stale: false,
        };
        for g in 0..n {
            c.refresh_roles(g);
        }
        c
    }

    fn refresh_roles(&mut self, g: usize) {
        let w = self.w;
        let row = &self.up[ci(Chain::Security)][g * w..g * w + w];
        self.role_sec[g] = ones(row).fold(0, |a, h| a | self.b.grants[h]);
    }

    pub fn row(&self, chain: Chain, g: usize) -> &[u64] {
        &self.up[ci(chain)][g * self.w..g * self.w + self.w]
    }

    pub fn down_row(&self, chain: Chain, g: usize) -> &[u64] {
        &self.down[ci(chain)][g * self.w..g * self.w + self.w]
    }

    /// A user's groups through `chain`: OR of its direct groups' rows.
    pub fn groups_of(&self, u: u16, chain: Chain, out: &mut [u64]) {
        out.fill(0);
        for &d in self.b.user_groups.row(u as usize) {
            mask_or_assign(out, self.row(chain, d as usize));
        }
    }

    /// Users in `g` through `chain`: the direct users of every group in
    /// `g`'s downward closure row.
    pub fn members_of(&self, g: u16, chain: Chain, out: &mut [u64]) {
        out.fill(0);
        for h in ones(self.down_row(chain, g as usize)) {
            for &u in self.b.group_users.row(h) {
                set(out, u as usize);
            }
        }
    }

    pub fn roles_of(&self, u: u16) -> u64 {
        self.b
            .user_groups
            .row(u as usize)
            .iter()
            .fold(0, |a, &d| a | self.role_sec[d as usize])
    }

    /// Add one nesting pair without recompiling: for every chain both ends
    /// pass, each descendant of `c` gains the ancestors of `p` and each
    /// ancestor of `p` gains the descendants of `c` (the transitive-closure
    /// insertion, exact for reflexive rows, cycles included).
    pub fn add_nesting(&mut self, c: u16, p: u16) {
        let (c, p) = (c as usize, p as usize);
        let w = self.w;
        for chain in CHAINS {
            if let Some(a) = self.b.admit(chain) {
                if !bit(a, c) || !bit(a, p) {
                    continue;
                }
            }
            let k = ci(chain);
            let anc_p = self.up[k][p * w..p * w + w].to_vec();
            let desc_c = self.down[k][c * w..c * w + w].to_vec();
            for x in ones(&desc_c) {
                mask_or_assign(&mut self.up[k][x * w..x * w + w], &anc_p);
            }
            for y in ones(&anc_p) {
                mask_or_assign(&mut self.down[k][y * w..y * w + w], &desc_c);
            }
            if chain == Chain::Security {
                for x in ones(&desc_c) {
                    self.refresh_roles(x);
                }
            }
        }
        if let Err(i) = self.b.nested.binary_search(&(c as u16, p as u16)) {
            self.b.nested.insert(i, (c as u16, p as u16));
        }
        // B+ never reads B's nesting CSR after compiling; it is rebuilt from
        // `nested` by `recompile`. Until then it is stale, so B's own walks
        // must use their own `Prepared`, not `self.b`.
        self.b_nest_stale = true;
    }

    /// Full recompile from the current pairs: the reference the incremental
    /// insertion is checked against, and the only path for a removal.
    pub fn recompile(mut self) -> Self {
        let (up, down) = nest_csr(self.b.groups, &self.b.nested);
        self.b.up = up;
        self.b.down = down;
        Compiled::compile(self.b)
    }

    /// Bytes of the closure rows and role table, B's tables excluded.
    pub fn closure_bytes(&self) -> usize {
        (self.up.iter().chain(self.down.iter()).map(Vec::len).sum::<usize>() + self.role_sec.len())
            * 8
    }
}
