//! # `class_view` — the class as a META lookup that flies ABOVE the SoA.
//!
//! ## The XML-parse framing
//!
//! Today OGIT (`lance-graph-ontology::OntologyRegistry`) is a **hashtable doing
//! single lookups**: `uri → row`, `entity_type_id → row` — one key, one value,
//! O(1), leaf. That is the *single* lookup. What a class needs is a **meta
//! lookup**: `class_id → the whole shape` — the ordered field set, labels,
//! template, and the presence-bit basis. The class composes many leaf lookups
//! into one shape — the way an XSD schema composes element declarations.
//!
//! ```text
//!   SoA row          =  the XML document   (agnostic bytes, no meaning)
//!   class / ObjectView =  the XSD schema     (the shape: which fields, in order)
//!   ClassView (this) =  the parser+schema  (projects row → typed view, late-bound)
//!   FieldMask        =  which optional elements are present  (structural)
//!   askama template  =  the XSLT            (renders the projected view)
//! ```
//!
//! ## Classes fly as a meta-DTO ABOVE the SoA — the SoA stays agnostic
//!
//! The load-bearing rule (`cognitive-risc-classes.md`:39 "the meta-DTO resolves;
//! it does not store"; `cognitive-risc-core.md` invariant #1 "nothing semantic in
//! the register file"): the SoA row carries **only** `class_id` + a presence
//! [`FieldMask`] + agnostic columns. **Zero labels in the bytes.** The
//! labels / template / DOLCE-category are resolved *at projection time* by the
//! flying meta-DTO from the OGIT cache — never hand-rolled onto the row.
//!
//! That makes the presence/semantics split (C2) fall out for free:
//! - **bit = presence** — structural, lives on the SoA ("field N is populated").
//! - **bit → field → label → template** — semantic resolution, lives in the
//!   meta-DTO *above* the SoA. A bit NEVER means "field N behaves differently."
//!
//! ## Layering (dependency inversion, same shape as `MailboxSoaView`)
//!
//! - **contract (here, zero-dep):** the agnostic surface — [`FieldMask`] presence
//!   bits + the [`ClassView`] resolver *trait*. Extends the existing
//!   [`crate::ontology::ObjectView`] (the per-class ordered field set = the bit
//!   basis), does not duplicate it.
//! - **ontology (one layer up):** *implements* [`ClassView`] — the "parser" that
//!   walks the class shape and resolves labels late from the OGIT hashmap.
//! - **render (a consumer):** reads the projected view + mask, picks the askama
//!   template, skips off-bits.

use crate::ontology::{DisplayTemplate, FieldRef};
use std::hash::{Hash, Hasher};

/// Per-row class discriminator — the Cognitive-RISC `class_id` / `shape_id`.
///
/// A `u16` (≤ 65,535 shape-families; OD-CLASSID-WIDTH ratified). It is a
/// *discriminator*, never a content hash — it stays OUTSIDE the CAM identity
/// layer (`I-VSA-IDENTITIES`: never hashed-as-content, never superposed). Reuses
/// the width of the existing [`crate::soa_view::MailboxSoaView::class_id`] accessor.
pub type ClassId = u16;

/// A class's **presence bitmask** — one bit per field of its class
/// [`ObjectView`](crate::ontology::ObjectView), set iff that field is populated
/// on a given instance.
///
/// The instance's *delta from its class* (`cognitive-risc-classes.md`:48), as
/// **pure presence bits**. Bit position `N` = the `N`-th field in the class's
/// ordered field list — stable + append-only (N3): once instances persist, a
/// field's bit position never moves and retired bits are never reused. Zero-dep
/// (`u64`, no `bitflags`); mask width is bounded by the *class's* field count
/// (dozens), never the entity union.
///
/// **Presence, NEVER semantics (C2).** `has(n)` answers "is field n populated
/// here"; it must never gate "field n means something different here."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct FieldMask(pub u64);

impl FieldMask {
    /// The empty mask (no fields populated).
    pub const EMPTY: Self = Self(0);

    /// Maximum addressable field positions in one `u64` mask.
    pub const MAX_FIELDS: u32 = 64;

    /// Build a mask from the populated field positions. Positions `>= MAX_FIELDS`
    /// (64) are **ignored** — NOT folded onto a valid bit. Folding (`& 63`) would
    /// alias position 64 onto bit 0 and silently corrupt the presence contract for
    /// an oversized class shape (Codex P2 on #441); ignoring keeps the no-panic
    /// property without misrepresenting which fields are present.
    pub const fn from_positions(positions: &[u8]) -> Self {
        let mut bits = 0u64;
        let mut i = 0;
        while i < positions.len() {
            if (positions[i] as u32) < Self::MAX_FIELDS {
                bits |= 1u64 << positions[i];
            }
            i += 1;
        }
        Self(bits)
    }

    /// Set field position `n` as populated. `n >= MAX_FIELDS` (64) is a no-op
    /// (NOT folded — see [`from_positions`](FieldMask::from_positions)).
    #[inline]
    pub const fn with(self, n: u8) -> Self {
        if (n as u32) < Self::MAX_FIELDS {
            Self(self.0 | (1u64 << n))
        } else {
            self
        }
    }

    /// Is field position `n` populated? (presence — C2). `n >= MAX_FIELDS` (64) is
    /// always `false` — an out-of-range field is never "present" (NOT folded onto
    /// a valid bit).
    #[inline]
    pub const fn has(self, n: u8) -> bool {
        (n as u32) < Self::MAX_FIELDS && self.0 & (1u64 << n) != 0
    }

    /// Number of populated fields.
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Is nothing populated?
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The full mask — every addressable field position present. The
    /// "no projection constraint" default for an RBAC role that has not
    /// narrowed its view (lance-graph-rbac `PermissionSpec::projection`).
    pub const FULL: Self = Self(u64::MAX);

    /// Bitwise intersection — the field positions present in BOTH masks.
    #[inline]
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Bitwise union — the field positions present in EITHER mask. The fold an
    /// RBAC kernel uses to combine the projections a user's several granting
    /// roles each permit (a user sees the union of the columns any of their
    /// roles may see).
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Do the two masks share NO field position? RBAC uses this to assert
    /// two roles project **distinct** views of the same class — e.g. a
    /// research projection must be disjoint from the identifier fields
    /// (`classid :: role :: membership`, where the role is the projection).
    #[inline]
    pub const fn is_disjoint(self, other: Self) -> bool {
        self.0 & other.0 == 0
    }

    /// Inherit a parent class's presence into this mask — the **mask-inherits-as-
    /// delta** of the HHTL `subClassOf` walk (`wikidata-hhtl-load.md`). A child
    /// IS-A its parent, so its mask carries every field the parent declares
    /// present PLUS its own `delta`: a bitwise union. N3 stable positions mean the
    /// parent's bits never move — the child only adds (multi-parent
    /// "flying-family" facets are orthogonal bits in this same mask, never a
    /// second path). Read `parent.inherit(own_delta)` → the child's full mask;
    /// the union is commutative, so the direction is documentation, not a
    /// constraint. See [`crate::hhtl`].
    #[inline]
    pub const fn inherit(self, delta: FieldMask) -> FieldMask {
        FieldMask(self.0 | delta.0)
    }
}

impl From<u64> for FieldMask {
    /// Additive convenience alongside the untouched tuple constructor
    /// `FieldMask(bits)` — `FieldMask::from(bits)` is equivalent, never a
    /// replacement (Ruling c: nothing about `FieldMask` changes).
    #[inline]
    fn from(bits: u64) -> Self {
        Self(bits)
    }
}

/// Backward-compatible widening companion to [`FieldMask`] for classes with
/// MORE than [`FieldMask::MAX_FIELDS`] (64) fields (Ruling c,
/// `D-FIELDMASK-WIDENING`; account.move/Odoo — 109 fields — is the motivating
/// case).
///
/// **Non-footgun by construction: `FieldMask` above is completely untouched.**
/// An earlier design considered swapping `FieldMask`'s internals for an
/// enum (`Repr::Small(u64) | Repr::Wide(Box<[u64]>)`) so ONE type covered both
/// tiers. That was rejected: `FieldMask::with`/`has`/`count`/`is_empty`/
/// `intersect`/`union`/`inherit` all take `self` **by value** and rely on
/// `FieldMask: Copy` so the same variable can be reused after a method call
/// (e.g. [`ClassProjection::next`] reads `self.mask.has(..)` out of a
/// `&mut self` field every iteration; [`FieldMask::from_positions`] folds
/// `with` over a slice). A `Box`-bearing repr cannot be `Copy`, so that design
/// would have broken every such call site — the exact footgun Ruling (c)
/// forbids. Keeping `FieldMask` byte-for-byte identical and adding this
/// sibling type instead means the acceptance criterion ("every existing
/// u64 constructor/call site compiles and behaves identically") holds
/// trivially: nothing about `FieldMask` changed.
///
/// **N3 stability across the pair:** bit position `n` denotes the same
/// logical field in both types — `WideFieldMask`'s positions 0..63 read
/// bit-for-bit identically to a `FieldMask` over the same class, the widening
/// only adds positions >= 64, never moves 0..63.
///
/// **Allocates only on demand:** internally `Small(u64)` (bit-identical to
/// `FieldMask`, zero heap) until a position >= 64 is set, at which point it
/// promotes once to `Wide(Box<[u64]>)` (chunk `k` = bits `64k..64k+63`). A
/// mask that never crosses the 64 boundary never allocates.
///
/// **When to use which:** reach for [`FieldMask`] for classes with <= 64
/// fields; reach for `WideFieldMask` only when a class's field count may
/// exceed 64.
///
/// `PartialEq`/`Eq`/`Hash` are hand-written (see the impls below), NOT
/// derived: equality must be representation-independent (a `Wide` value
/// whose high chunks are all zero must compare equal to, and hash
/// identically with, the equivalent `Small` value), which a structural
/// derive over `WideRepr` cannot express.
#[derive(Debug, Clone)]
pub struct WideFieldMask(WideRepr);

#[derive(Debug, Clone)]
enum WideRepr {
    /// <= 64 positions — zero-allocation, bit-identical to `FieldMask(u64)`.
    Small(u64),
    /// > 64 positions. `chunks[k]` holds bits `64k..64k+63`.
    Wide(Box<[u64]>),
}

impl Default for WideFieldMask {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl From<u64> for WideFieldMask {
    fn from(bits: u64) -> Self {
        Self(WideRepr::Small(bits))
    }
}

impl From<FieldMask> for WideFieldMask {
    /// Promote a `FieldMask` into the wide-tier vocabulary — always lands in
    /// `Small` (no allocation): every `FieldMask` bit is already `< 64`.
    fn from(m: FieldMask) -> Self {
        Self(WideRepr::Small(m.0))
    }
}

impl WideFieldMask {
    /// The empty mask (no fields populated). Zero-allocation (`Small`).
    pub const EMPTY: Self = Self(WideRepr::Small(0));

    /// Set field position `n` as populated. Promotes `Small` → `Wide` only
    /// when `n >= 64` (or the mask already promoted); a mask that never sets
    /// a position >= 64 never allocates.
    #[must_use]
    pub fn with(self, n: u8) -> Self {
        let chunk = (n / 64) as usize;
        let bit = 1u64 << (n % 64);
        match self.0 {
            WideRepr::Small(bits) if chunk == 0 => Self(WideRepr::Small(bits | bit)),
            WideRepr::Small(bits) => {
                let mut v = vec![0u64; chunk + 1];
                v[0] = bits;
                v[chunk] |= bit;
                Self(WideRepr::Wide(v.into_boxed_slice()))
            }
            WideRepr::Wide(v) => {
                let mut v = if chunk >= v.len() {
                    let mut grown = vec![0u64; chunk + 1];
                    grown[..v.len()].copy_from_slice(&v);
                    grown
                } else {
                    v.into_vec()
                };
                v[chunk] |= bit;
                Self(WideRepr::Wide(v.into_boxed_slice()))
            }
        }
    }

    /// Build a mask from populated field positions — the wide-tier sibling of
    /// [`FieldMask::from_positions`]; folds [`with`](Self::with) over
    /// `positions` (same allocate-on-demand contract).
    #[must_use]
    pub fn from_positions(positions: &[u8]) -> Self {
        positions.iter().fold(Self::EMPTY, |m, &p| m.with(p))
    }

    /// Is field position `n` populated?
    #[inline]
    #[must_use]
    pub fn has(&self, n: u8) -> bool {
        let chunk = (n / 64) as usize;
        let bit = 1u64 << (n % 64);
        match &self.0 {
            WideRepr::Small(bits) => chunk == 0 && bits & bit != 0,
            WideRepr::Wide(v) => v.get(chunk).is_some_and(|w| w & bit != 0),
        }
    }

    /// Number of populated fields.
    #[must_use]
    pub fn count(&self) -> u32 {
        match &self.0 {
            WideRepr::Small(bits) => bits.count_ones(),
            WideRepr::Wide(v) => v.iter().map(|w| w.count_ones()).sum(),
        }
    }

    /// Is nothing populated?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match &self.0 {
            WideRepr::Small(bits) => *bits == 0,
            WideRepr::Wide(v) => v.iter().all(|&w| w == 0),
        }
    }

    /// The mask's current representable capacity in bit positions. `64` while
    /// `Small` (mirrors [`FieldMask::MAX_FIELDS`]); grows in steps of 64 once
    /// promoted to `Wide`. Reports capacity, not [`count`](Self::count).
    #[must_use]
    pub fn max_fields(&self) -> u32 {
        match &self.0 {
            WideRepr::Small(_) => FieldMask::MAX_FIELDS,
            WideRepr::Wide(v) => (v.len() as u32) * 64,
        }
    }

    /// The full mask over exactly `field_count` positions (`0..field_count`
    /// all present) — the wide-tier sibling of [`FieldMask::FULL`], which is
    /// only meaningful for a *fixed* 64-wide class; a wide class's "all
    /// fields" sentinel must instead know how many fields it has.
    #[must_use]
    pub fn full_for(field_count: usize) -> Self {
        if field_count == 0 {
            return Self::EMPTY;
        }
        if field_count <= 64 {
            let bits = if field_count == 64 {
                u64::MAX
            } else {
                (1u64 << field_count) - 1
            };
            return Self(WideRepr::Small(bits));
        }
        let chunks = field_count.div_ceil(64);
        let mut v = vec![u64::MAX; chunks];
        let rem = field_count % 64;
        if rem != 0 {
            v[chunks - 1] = (1u64 << rem) - 1;
        }
        Self(WideRepr::Wide(v.into_boxed_slice()))
    }

    /// Bitwise intersection — field positions present in BOTH masks. Folds
    /// tier-agnostically: a `Small`∩`Wide` pair reads the `Small` side's
    /// missing high chunks as `0` (never aliases a low bit onto a high one).
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        match (&self.0, &other.0) {
            (WideRepr::Small(a), WideRepr::Small(b)) => Self(WideRepr::Small(a & b)),
            _ => self.zip_fold(other, |a, b| a & b),
        }
    }

    /// Bitwise union — field positions present in EITHER mask. Same
    /// tier-agnostic fold as [`intersect`](Self::intersect).
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        match (&self.0, &other.0) {
            (WideRepr::Small(a), WideRepr::Small(b)) => Self(WideRepr::Small(a | b)),
            _ => self.zip_fold(other, |a, b| a | b),
        }
    }

    /// Chunk-wise fold across tiers (only reached when at least one side is
    /// already `Wide`, so the allocation is not new spend for that side).
    ///
    /// Normalizes the result before returning: trailing all-zero chunks are
    /// trimmed, and a result that now fits in a single chunk demotes back to
    /// `Small` (V-L P0: an un-normalized `Wide` result — e.g. intersecting a
    /// mask that has bits >= 64 down to only bits < 64 — must still compare
    /// equal to, and hash identically with, the equivalent `Small` mask).
    fn zip_fold(&self, other: &Self, f: impl Fn(u64, u64) -> u64) -> Self {
        let a = self.chunks_view();
        let b = other.chunks_view();
        let len = a.len().max(b.len());
        let mut v = vec![0u64; len];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = f(
                a.get(i).copied().unwrap_or(0),
                b.get(i).copied().unwrap_or(0),
            );
        }
        while v.last() == Some(&0) {
            v.pop();
        }
        if v.len() <= 1 {
            return Self(WideRepr::Small(v.first().copied().unwrap_or(0)));
        }
        Self(WideRepr::Wide(v.into_boxed_slice()))
    }

    fn chunks_view(&self) -> Vec<u64> {
        match &self.0 {
            WideRepr::Small(bits) => vec![*bits],
            WideRepr::Wide(v) => v.to_vec(),
        }
    }

    /// Chunk `i` of this mask, `0` past the end — the tier-agnostic accessor
    /// the canonical-equality/hash impls fold over.
    #[inline]
    fn chunk_at(&self, i: usize) -> u64 {
        match &self.0 {
            WideRepr::Small(bits) => {
                if i == 0 {
                    *bits
                } else {
                    0
                }
            }
            WideRepr::Wide(v) => v.get(i).copied().unwrap_or(0),
        }
    }

    /// The number of chunks in this mask's *canonical* form: the raw chunk
    /// count with trailing all-zero chunks trimmed. Two masks denote the same
    /// field set, regardless of tier, iff they agree on this length AND on
    /// every chunk below it — this is the representation-independent view
    /// `PartialEq`/`Hash` are built over.
    fn canonical_len(&self) -> usize {
        let raw = match &self.0 {
            WideRepr::Small(_) => 1,
            WideRepr::Wide(v) => v.len(),
        };
        let mut n = raw;
        while n > 0 && self.chunk_at(n - 1) == 0 {
            n -= 1;
        }
        n
    }
}

impl PartialEq for WideFieldMask {
    /// Representation-independent equality: two masks are equal iff they
    /// denote the same populated field set, regardless of whether either
    /// side is `Small` or `Wide` (or, for two `Wide` values, regardless of
    /// their chunk-vector lengths) — trailing all-zero chunks never affect
    /// equality (V-L P0).
    fn eq(&self, other: &Self) -> bool {
        let len = self.canonical_len();
        if len != other.canonical_len() {
            return false;
        }
        (0..len).all(|i| self.chunk_at(i) == other.chunk_at(i))
    }
}

impl Eq for WideFieldMask {}

impl Hash for WideFieldMask {
    /// Must stay consistent with [`PartialEq`]: hashes exactly the canonical
    /// (trailing-zero-trimmed) chunk sequence, length-prefixed like a slice's
    /// `Hash` impl, so `a == b` implies `hash(a) == hash(b)` regardless of
    /// which tier either side happens to be stored in.
    fn hash<H: Hasher>(&self, state: &mut H) {
        let len = self.canonical_len();
        len.hash(state);
        for i in 0..len {
            self.chunk_at(i).hash(state);
        }
    }
}

/// Positions in [`WideFieldMask`] are `u8`, so a mask can address at most 256
/// fields. The doctrine (lance-graph #651 body) is explicit: capacity beyond
/// 256 is headroom, never license — a class whose universe genuinely exceeds
/// 256 fields is an OGAR-SOC (separation-of-concerns) split signal, not a case
/// to widen the mask type further. [`WideFieldMask::from_universe_present`]
/// refuses loudly rather than silently truncating or wrapping such a universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WideMaskCapError {
    /// `universe.len()` exceeded the 256-position cap that `u8` positions
    /// impose on [`WideFieldMask`].
    UniverseExceedsSocCap {
        /// The offending universe size.
        fields: usize,
    },
}

impl std::fmt::Display for WideMaskCapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UniverseExceedsSocCap { fields } => write!(
                f,
                "universe of {fields} fields exceeds the 256-field cap (u8 positions) \
                 — an OGAR-SOC split signal, not a mask to widen further"
            ),
        }
    }
}

impl std::error::Error for WideMaskCapError {}

impl WideFieldMask {
    /// Mint the projection mask from a domain-blind `universe`/`present` pair:
    /// bit `i` is set iff `universe[i]` ∈ `present` — the same sort-agnostic
    /// membership rule odoo-rs's `od_ontology::view_mask::mint_wide_mask` uses,
    /// so any consumer (Rails/ERB `ViewFieldSet` included) mints the identical
    /// mask from the identical inputs. Positions are the universe index (`u8`).
    ///
    /// **One more brick, not a parallel path.** This is a plain constructor
    /// stacked on top of the existing kit — it computes the populated
    /// positions and hands them to [`Self::from_positions`] (which itself
    /// folds [`Self::with`] over the slice); no bit-fiddling is reimplemented
    /// here. The `Self` it returns composes exactly like any other
    /// `WideFieldMask`: keep stacking `.with(...)`, [`Self::union`],
    /// [`Self::intersect`], etc. on the result.
    ///
    /// # Errors
    ///
    /// Returns [`WideMaskCapError::UniverseExceedsSocCap`] if
    /// `universe.len() > 256` — `WideFieldMask` positions are `u8`, so a larger
    /// universe cannot be addressed at all. This is a loud refusal, never a
    /// silent drop or truncation.
    pub fn from_universe_present(
        universe: &[&str],
        present: &[&str],
    ) -> Result<Self, WideMaskCapError> {
        if universe.len() > 256 {
            return Err(WideMaskCapError::UniverseExceedsSocCap {
                fields: universe.len(),
            });
        }
        let present_set: std::collections::BTreeSet<&str> = present.iter().copied().collect();
        #[allow(clippy::cast_possible_truncation)] // guarded: universe.len() <= 256 above
        let positions: Vec<u8> = universe
            .iter()
            .enumerate()
            .filter(|(_, field)| present_set.contains(*field))
            .map(|(i, _)| i as u8)
            .collect();
        Ok(Self::from_positions(&positions))
    }
}

/// One recompute edge in a class's **compute DAG**: field position `target` is
/// (re)computed from the field positions in `inputs`.
///
/// Harvest-sourced — `target` is the `emitted_by` field, `inputs` are its
/// `depends_on` precedents (Odoo `@api.depends`, an Excel formula's referenced
/// cells, a chess-eval feature's inputs). All fields are `&'static` so a
/// generated `const DAG: &[ComputeEdge] = &[..]` compiles (the harvest IS the
/// manifest — mirrors [`crate::codegen_manifest::MethodSig`] /
/// [`crate::action::ActionDef`]). Positions index the class's [`FieldMask`]
/// (0..[`FieldMask::MAX_FIELDS`]), matching [`ClassView::fields`].
///
/// This is the Core home for recompute *dispatch* (`E-EXCEL-SHADER-PROJECTION` /
/// `probe-excel-compute-dag-v1`): the manifest lives ABOVE the SoA (resolution
/// metadata, stores nothing on the row); no adapter carries its own `@api.depends`
/// table (`core-first-transcode-doctrine` — that would be the Adapter-State-Leak).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeEdge {
    /// Field position this edge recomputes (the `emitted_by` target).
    pub target: u8,
    /// Field positions this target reads (its `depends_on` precedents).
    pub inputs: &'static [u8],
}

/// Whether a class's `compute_dag` is **acyclic** — the registry-build gate.
///
/// A cyclic recompute DAG (a formula loop `A=B+1, B=A+1`, or a `@api.depends`
/// cycle) MUST be rejected at registry-build: it has no topological order and
/// would never converge. Returns `false` on any cycle (incl. a self-loop
/// `target ∈ inputs`). Considers only positions `< FieldMask::MAX_FIELDS`;
/// out-of-range targets/inputs are ignored (no panic, mirrors
/// [`FieldMask::from_positions`]). Allocation-free (≤ 64 positions).
#[must_use]
pub fn compute_dag_is_acyclic(edges: &[ComputeEdge]) -> bool {
    const N: usize = FieldMask::MAX_FIELDS as usize; // 64
                                                     // deps[t] = bitmask of in-range positions that target `t` depends on.
    let mut deps = [0u64; N];
    let mut is_target = 0u64;
    for e in edges {
        if (e.target as usize) >= N {
            continue;
        }
        is_target |= 1u64 << e.target;
        for &inp in e.inputs {
            if (inp as usize) < N {
                deps[e.target as usize] |= 1u64 << inp;
            }
        }
    }
    // Kahn: leaves (non-targets) are resolved; peel any target whose deps are all
    // resolved. If a round makes no progress with targets remaining → cycle.
    let mut resolved = !is_target;
    let mut remaining = is_target;
    loop {
        if remaining == 0 {
            return true;
        }
        let mut progressed = false;
        let mut r = remaining;
        while r != 0 {
            let t = r.trailing_zeros();
            r &= r - 1;
            if deps[t as usize] & !resolved == 0 {
                resolved |= 1u64 << t;
                remaining &= !(1u64 << t);
                progressed = true;
            }
        }
        if !progressed {
            return false; // stuck with targets remaining → cycle
        }
    }
}

/// A valid **recompute order** for a class's `compute_dag` — the target field
/// positions in an order where every target appears after all targets it
/// (transitively) depends on. `None` if the DAG is cyclic (no topological order).
///
/// Leaves (positions that are only inputs, never targets) are not included — they
/// are the already-present values a recompute reads. Within one Kahn round the
/// resolved targets are mutually independent, so any order among them is valid.
/// The consumer recomputes targets in this order, each gated by the cycle-aware
/// `write_row` (`probe-excel-compute-dag-v1` Inc 2). Allocation = one `Vec` of
/// the target positions (≤ 64).
#[must_use]
pub fn compute_dag_topo_order(edges: &[ComputeEdge]) -> Option<Vec<u8>> {
    const N: usize = FieldMask::MAX_FIELDS as usize; // 64
    let mut deps = [0u64; N];
    let mut is_target = 0u64;
    for e in edges {
        if (e.target as usize) >= N {
            continue;
        }
        is_target |= 1u64 << e.target;
        for &inp in e.inputs {
            if (inp as usize) < N {
                deps[e.target as usize] |= 1u64 << inp;
            }
        }
    }
    let mut resolved = !is_target;
    let mut remaining = is_target;
    let mut order = Vec::with_capacity(is_target.count_ones() as usize);
    loop {
        if remaining == 0 {
            return Some(order);
        }
        let mut progressed = false;
        let mut r = remaining;
        while r != 0 {
            let t = r.trailing_zeros();
            r &= r - 1;
            if deps[t as usize] & !resolved == 0 {
                resolved |= 1u64 << t;
                remaining &= !(1u64 << t);
                order.push(t as u8);
                progressed = true;
            }
        }
        if !progressed {
            return None; // cycle
        }
    }
}

/// The class as a **meta lookup that flies above the SoA** — the resolver trait.
///
/// An implementor (in `lance-graph-ontology`, over the OGIT cache) is the
/// "parser+schema": given a `class_id` it resolves the class's ordered field set,
/// labels, DOLCE category, and render template — all LATE-bound from the cache,
/// none stored on the SoA row. The contract owns only the *vocabulary*; the cache
/// owns the *answers* (dependency inversion, like `PlannerContract`/`MailboxSoaView`).
///
/// "Single lookup" (leaf, today) vs "meta lookup" (the class, this trait): a
/// single lookup is `uri → row`; a meta lookup is `class_id → shape`, composing
/// many leaf lookups into one projected view.
pub trait ClassView {
    /// The class's ordered field set — the bit basis. Position `i` in this slice
    /// is the stable [`FieldMask`] bit `i` (N3 append-only). This IS the
    /// per-class [`ObjectView`](crate::ontology::ObjectView)'s `fields`.
    fn fields(&self, class: ClassId) -> &[FieldRef];

    /// Which askama template renders this class.
    fn template(&self, class: ClassId) -> DisplayTemplate;

    /// The DOLCE upper-category of this class, RESOLVED from the ontology cache
    /// (not a stored enum on the row — OD-DOLCE "use the ontology cache"). Returned
    /// as the cache's opaque category id; the consumer maps it to its own enum.
    fn dolce_category_id(&self, class: ClassId) -> u8;

    /// The label of field position `n` in `class`, resolved late from the cache
    /// (locale resolution is the consumer's job). `None` if `n` is out of range.
    fn field_label(&self, class: ClassId, n: u8) -> Option<&str> {
        self.fields(class).get(n as usize).map(|f| f.label.as_str())
    }

    /// The class's field count (mask width). Must be `<= FieldMask::MAX_FIELDS`.
    #[inline]
    fn field_count(&self, class: ClassId) -> usize {
        self.fields(class).len()
    }

    /// Project an instance: iterate `(field, populated?)` pairs in class order,
    /// gating each field by the presence `mask`. This is the render surface — the
    /// consumer skips off-bits (`cognitive-risc-classes.md`:49). The SoA supplied
    /// only `(class, mask)`; the labels come from the cache, above the SoA.
    fn project<'a>(&'a self, class: ClassId, mask: FieldMask) -> ClassProjection<'a> {
        ClassProjection {
            fields: self.fields(class),
            mask,
            pos: 0,
        }
    }

    /// The **render rows** for an instance: only the populated `(label, predicate)`
    /// pairs, off-bits skipped (`cognitive-risc-classes.md`:49). This is the
    /// template-agnostic render surface — an askama/jinja per-class template iterates
    /// these rows; the engine choice (F3, askama) lives in the deferred render crate.
    ///
    /// Presence-only (C2): a row appears iff its bit is set; the mask NEVER changes a
    /// row's meaning, only its presence. The labels are the meta-DTO's late resolution
    /// (above the SoA), the mask is the SoA's structural delta.
    fn render_rows<'a>(&'a self, class: ClassId, mask: FieldMask) -> Vec<RenderRow<'a>> {
        self.project(class, mask)
            .filter(|(_, present)| *present)
            .map(|(f, _)| RenderRow {
                label: f.label.as_str(),
                predicate: f.predicate_iri.as_str(),
            })
            .collect()
    }

    /// The **value render rows** for an instance — the value-projected sibling of
    /// [`render_rows`](ClassView::render_rows): each populated field paired with the
    /// byte it reads from the node's V3 content-blind 12-byte facet payload.
    ///
    /// Field position `i` binds to facet byte `i`
    /// (`.claude/v3/soa_layout/le-contract.md` §3 — the 12-byte payload is a *dumb
    /// byte register the ClassView projects*; the class picks which reading, never a
    /// slot in the byte). A [`ValueRow`] is emitted for each position that is BOTH
    /// mask-present (C2 presence) AND `< 12` (facet-backed).
    ///
    /// **Positions `>= 12` are value-slab fields, out of facet scope — skipped
    /// here** (never folded onto a valid byte, mirroring the
    /// [`FieldMask::MAX_FIELDS`] 64-guard style). A class whose field set exceeds the
    /// 12 facet bytes carries the overflow fields in the 480-byte value slab
    /// (`ValueSchema` tenants), read through a different surface; those never surface
    /// as [`ValueRow`]s.
    ///
    /// Presence-only (C2): the `mask` gates which rows exist; the facet byte a row
    /// carries is agnostic — its meaning is the late-resolved `label` / `predicate`,
    /// never the byte itself. The SoA supplied only `(class, mask, facet)`.
    fn facet_rows<'a>(
        &'a self,
        class: ClassId,
        mask: FieldMask,
        facet: &[u8; 12],
    ) -> Vec<ValueRow<'a>> {
        let mut rows = Vec::new();
        for (i, f) in self.fields(class).iter().enumerate() {
            // Facet-backed positions only (< 12), and only where the mask is set.
            // `i >= 12` is a value-slab field: skipped, NEVER folded onto byte i&11.
            if i < 12 && mask.has(i as u8) {
                rows.push(ValueRow {
                    label: f.label.as_str(),
                    predicate: f.predicate_iri.as_str(),
                    position: i as u8,
                    value: facet[i],
                });
            }
        }
        rows
    }

    /// The `is_a` (subClassOf) parent of `class`, resolved by the implementor from
    /// its taxonomy cache (OGIT / HHTL `subClassOf`). The taxonomy hop the
    /// zero-fallback render ladder ([`resolve_render_class`](ClassView::resolve_render_class))
    /// walks.
    ///
    /// Default `None` — **no taxonomy**: an implementor without a `subClassOf` cache
    /// leaves every class an orphan, and `resolve_render_class` degenerates to
    /// "return the class itself" (still monotonic — the ladder never fails).
    #[inline]
    fn is_a_parent(&self, _class: ClassId) -> Option<ClassId> {
        None
    }

    /// Resolve the class whose card actually renders `class` — the **zero-fallback
    /// render ladder**: bespoke card → nearest ancestor's card → (caller renders the
    /// generic facet dump).
    ///
    /// Walks [`is_a_parent`](ClassView::is_a_parent) while the current class has an
    /// EMPTY field set ([`fields`](ClassView::fields)`.is_empty()`), returning the
    /// first class with a non-empty field set (its bespoke or inherited card). If the
    /// walk reaches the taxonomy root (no parent) or hits the depth cap without
    /// finding fields, it returns the ORIGINAL `class` — **monotonic: the ladder
    /// never fails**. An empty field set on the returned class is the caller's signal
    /// to render the generic facet dump (the last rung — dump the raw
    /// [`facet_rows`](ClassView::facet_rows) with no bespoke labels).
    ///
    /// **Cycle- and depth-safe, zero-dep:** a hard cap of 16 hops bounds the walk,
    /// and an on-stack visited array rejects a `subClassOf` cycle
    /// (`A is_a B is_a A`) without a `HashSet`. Either guard hit → return the
    /// original class (never an infinite loop, never a panic).
    fn resolve_render_class(&self, class: ClassId) -> ClassId {
        const MAX_HOPS: usize = 16;
        // Seeded with `class` so an unwritten tail can never spuriously match.
        let mut visited: [ClassId; MAX_HOPS] = [class; MAX_HOPS];
        let mut current = class;
        let mut depth = 0usize;
        loop {
            if !self.fields(current).is_empty() {
                return current; // bespoke (depth 0) or nearest ancestor card
            }
            if depth >= MAX_HOPS {
                return class; // depth cap → caller renders the generic facet dump
            }
            visited[depth] = current;
            depth += 1;
            let parent = match self.is_a_parent(current) {
                Some(p) => p,
                None => return class, // taxonomy root, no card → generic facet dump
            };
            if visited[..depth].contains(&parent) {
                return class; // subClassOf cycle → generic facet dump
            }
            current = parent;
        }
    }

    /// Which edge-codec flavor this class reads its node edge block with.
    ///
    /// Default is
    /// [`EdgeCodecFlavor::CoarseOnly`](crate::canonical_node::EdgeCodecFlavor::CoarseOnly)
    /// — the canon zero-fallback reading (each edge byte is a palette index). An
    /// implementor overrides this to let a class opt into residue or PQ fidelity.
    /// This is *selection only*: every flavor shares the SAME byte layout, so the
    /// choice never changes `NODE_ROW_STRIDE` (canon "registry-resolved via
    /// `classid → ClassView`", never a stride change).
    #[inline]
    fn edge_codec_flavor(&self, _class: ClassId) -> crate::canonical_node::EdgeCodecFlavor {
        crate::canonical_node::EdgeCodecFlavor::CoarseOnly
    }

    /// Which value-slab schema preset this class materialises in
    /// [`NodeRow::value`](crate::canonical_node::NodeRow::value).
    ///
    /// **TEMPORARY (POC default, 2026-06-15):** returns
    /// [`ValueSchema::Full`](crate::canonical_node::ValueSchema::Full) — every
    /// *unconfigured* class (incl. the default `classid 0x0000_0000`) materialises
    /// the whole value slab so downstream consumers (tesseract-rs / woa-rs /
    /// medcare-rs / q2) can transcode against a fully-populated `NodeRow` POC.
    /// Specialisation is **opt-IN, not opt-out**: a consumer that needs to save
    /// memory mints a class that overrides this to a smaller preset (`Cognitive` /
    /// `Compressed` / `Bootstrap`); a consumer that needs denser/specialised data
    /// mints a *separate* class. Selection only: every preset carves within the
    /// reserved 480-byte value slab, so the choice never changes `NODE_ROW_STRIDE`
    /// (canon "registry-resolved via `classid → ClassView`", never a stride change)
    /// — flipping the default is layout-preserving and a one-line revert to
    /// `Bootstrap` (the canon zero-fallback) before merge. The type-level
    /// [`ValueSchema::default()`](crate::canonical_node::ValueSchema) stays
    /// `Bootstrap`, so the substrate zero-fallback semantics are untouched; only
    /// the class→schema *resolution* default is Full.
    #[inline]
    fn value_schema(&self, _class: ClassId) -> crate::canonical_node::ValueSchema {
        // TEMPORARY POC default — see doc above. Revert to `ValueSchema::Bootstrap`
        // (canon zero-fallback) before merge. No invention: `Full` activates the
        // already-existing, already-tested 9 ValueTenants (helix-48 / turbovec /
        // signed / fingerprint / …), it adds no new property.
        crate::canonical_node::ValueSchema::Full
    }

    /// The class's **recompute DAG** — the topological manifest of which fields
    /// recompute from which (the `emitted_by` + `depends_on` harvest), the Core
    /// home for computed-field dispatch (Odoo `@api.depends`, Excel formulas,
    /// chess-eval features; `probe-excel-compute-dag-v1`).
    ///
    /// Default `&[]` — the zero-fallback: an unconfigured class has no computed
    /// fields (mirrors `compute_dag`'s no-panic siblings). An implementor returns
    /// a generated `const &[ComputeEdge]`; the registry MUST validate it with
    /// [`compute_dag_is_acyclic`] at build (a cyclic DAG is rejected, never
    /// recomputed). Layout-preserving: resolution metadata above the SoA, stores
    /// nothing on the row, never a `NODE_ROW_STRIDE`/`ENVELOPE_LAYOUT_VERSION`
    /// change. The instance recompute that consumes this is gated per-cell by the
    /// cycle-aware `write_row` (`E-SOA-CYCLE-OWNERSHIP`).
    #[inline]
    fn compute_dag(&self, _class: ClassId) -> &[ComputeEdge] {
        &[]
    }
}

/// One populated field to render — the late-resolved `label` + its `predicate` key.
/// Produced only for set bits (off-bits are skipped), so a template never branches
/// on presence (C2): it just iterates the rows it is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderRow<'a> {
    /// The display label, resolved late from the OGIT cache (above the SoA).
    pub label: &'a str,
    /// The field's predicate IRI (the stable key behind the label).
    pub predicate: &'a str,
}

/// The **value-projected sibling of [`RenderRow`]** — one populated field paired
/// with the byte it reads from the node's V3 content-blind 12-byte facet payload.
///
/// Where [`RenderRow`] carries only the late-resolved `(label, predicate)` of a
/// present field, `ValueRow` additionally carries the raw facet byte at that field's
/// position. Position `i` binds to facet byte `i` per
/// `.claude/v3/soa_layout/le-contract.md` §3: the 12-byte payload is a **dumb byte
/// register**, and the ClassView picks the reading — the byte's meaning is never in
/// the byte, only in the class's field (`label` / `predicate`) at that position.
///
/// Presence-only C2 still holds: the [`FieldMask`] gates which rows *exist*, never
/// what a byte *means*. Produced only for [`facet_rows`](ClassView::facet_rows)
/// positions that are both mask-present and facet-backed (`< 12`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRow<'a> {
    /// The display label, resolved late from the OGIT cache (above the SoA).
    pub label: &'a str,
    /// The field's predicate IRI (the stable key behind the label).
    pub predicate: &'a str,
    /// The field position — binds to facet byte `position` (§3 dumb byte register).
    pub position: u8,
    /// The raw facet byte at `position` (agnostic; the class picks its reading).
    pub value: u8,
}

/// An iterator over a class's fields paired with their presence bit — the
/// projected view a render template consumes (off-bits are still yielded with
/// `present = false` so the template can `{% if present %}`-skip them).
pub struct ClassProjection<'a> {
    fields: &'a [FieldRef],
    mask: FieldMask,
    pos: usize,
}

impl<'a> Iterator for ClassProjection<'a> {
    /// `(field, present)` — `present` is the C2 presence bit, never a semantics bit.
    type Item = (&'a FieldRef, bool);

    fn next(&mut self) -> Option<Self::Item> {
        let f = self.fields.get(self.pos)?;
        let present = self.mask.has(self.pos as u8);
        self.pos += 1;
        Some((f, present))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{DisplayTemplate, FieldRef};
    use std::collections::HashMap;

    /// A tiny in-contract ClassView fake — proves the trait is satisfiable and the
    /// meta-DTO projects above an agnostic (class, mask) input, no labels stored.
    ///
    /// `extra` + `parents` extend it for the value-projection + is_a-walk surface:
    /// `extra` holds field sets for classes other than the special-cased invoice
    /// (class 7); `parents` is the `subClassOf` taxonomy `resolve_render_class` walks.
    #[derive(Default)]
    struct FakeClasses {
        // class 7 = a 3-field shape ("invoice": amount, tax, partner)
        invoice: Vec<FieldRef>,
        // per-class field sets for non-7 classes (empty vec / absent = empty shape).
        extra: HashMap<ClassId, Vec<FieldRef>>,
        // is_a (subClassOf) edges: child -> parent.
        parents: HashMap<ClassId, ClassId>,
    }

    impl FakeClasses {
        fn new() -> Self {
            Self {
                invoice: vec![
                    FieldRef::new("amount_total", "Total"),
                    FieldRef::new("amount_tax", "Tax"),
                    FieldRef::new("partner_id", "Partner"),
                ],
                ..Default::default()
            }
        }

        /// Register a class's field set (test builder for the is_a / facet surface).
        fn with_class(mut self, class: ClassId, fields: Vec<FieldRef>) -> Self {
            self.extra.insert(class, fields);
            self
        }

        /// Register an `is_a` (subClassOf) edge `child -> parent`.
        fn with_isa(mut self, child: ClassId, parent: ClassId) -> Self {
            self.parents.insert(child, parent);
            self
        }
    }

    impl ClassView for FakeClasses {
        fn fields(&self, class: ClassId) -> &[FieldRef] {
            match class {
                7 => &self.invoice,
                c => self.extra.get(&c).map_or(&[], |v| v.as_slice()),
            }
        }
        fn template(&self, _class: ClassId) -> DisplayTemplate {
            DisplayTemplate::Detail
        }
        fn dolce_category_id(&self, _class: ClassId) -> u8 {
            0 // Endurant, resolved from the cache in the real impl
        }
        fn is_a_parent(&self, class: ClassId) -> Option<ClassId> {
            self.parents.get(&class).copied()
        }
    }

    // ── compute_dag (probe-excel-compute-dag-v1, Inc 0) ──────────────────────

    /// Default `compute_dag` is the zero-fallback empty manifest (no computed
    /// fields for an unconfigured class).
    #[test]
    fn compute_dag_default_is_empty() {
        let c = FakeClasses {
            invoice: vec![],
            ..Default::default()
        };
        assert!(c.compute_dag(7).is_empty());
        assert!(c.compute_dag(0).is_empty());
    }

    /// `const`-constructible manifest — the exact shape a generated
    /// `const DAG: &[ComputeEdge]` emits (a chain: f2 = g(f1), f1 = h(f0)).
    const SAMPLE_DAG: &[ComputeEdge] = &[
        ComputeEdge {
            target: 1,
            inputs: &[0],
        },
        ComputeEdge {
            target: 2,
            inputs: &[1],
        },
    ];

    #[test]
    fn compute_dag_acyclic_chain_passes() {
        assert!(
            compute_dag_is_acyclic(SAMPLE_DAG),
            "a dependency chain f0→f1→f2 is acyclic"
        );
        assert!(compute_dag_is_acyclic(&[]), "empty dag is acyclic");
        // a target reading a non-computed leaf is fine
        assert!(compute_dag_is_acyclic(&[ComputeEdge {
            target: 5,
            inputs: &[3, 4]
        }]));
    }

    #[test]
    fn compute_dag_cycle_is_rejected() {
        // f0 = g(f1), f1 = h(f0) — a 2-cycle, no topological order.
        let two_cycle = &[
            ComputeEdge {
                target: 0,
                inputs: &[1],
            },
            ComputeEdge {
                target: 1,
                inputs: &[0],
            },
        ];
        assert!(
            !compute_dag_is_acyclic(two_cycle),
            "a formula loop must be rejected at registry-build"
        );
        // self-loop f0 = g(f0)
        assert!(!compute_dag_is_acyclic(&[ComputeEdge {
            target: 0,
            inputs: &[0]
        }]));
        // 3-cycle f0→f1→f2→f0
        assert!(!compute_dag_is_acyclic(&[
            ComputeEdge {
                target: 1,
                inputs: &[0]
            },
            ComputeEdge {
                target: 2,
                inputs: &[1]
            },
            ComputeEdge {
                target: 0,
                inputs: &[2]
            },
        ]));
    }

    #[test]
    fn compute_dag_out_of_range_positions_ignored() {
        // target/inputs >= MAX_FIELDS (64) are ignored, never folded → no panic,
        // no false cycle (mirrors FieldMask::from_positions).
        assert!(compute_dag_is_acyclic(&[
            ComputeEdge {
                target: 64,
                inputs: &[0]
            }, // ignored target
            ComputeEdge {
                target: 5,
                inputs: &[200]
            }, // input ignored → leaf-only target
        ]));
    }

    #[test]
    fn compute_dag_topo_order_respects_dependencies() {
        // chain f0→f1→f2: f1 must come before f2; f0 is a leaf, not emitted.
        let order = compute_dag_topo_order(SAMPLE_DAG).expect("acyclic has an order");
        assert_eq!(
            order.len(),
            2,
            "two targets (f1, f2); f0 is a read-only leaf"
        );
        let pos1 = order.iter().position(|&t| t == 1).unwrap();
        let pos2 = order.iter().position(|&t| t == 2).unwrap();
        assert!(pos1 < pos2, "f1 recomputed before its dependent f2");
        // empty manifest → empty order, not None.
        assert_eq!(compute_dag_topo_order(&[]), Some(vec![]));
    }

    #[test]
    fn compute_dag_topo_order_none_on_cycle() {
        // a 2-cycle has no topological order — None, matching is_acyclic == false.
        let two_cycle = &[
            ComputeEdge {
                target: 0,
                inputs: &[1],
            },
            ComputeEdge {
                target: 1,
                inputs: &[0],
            },
        ];
        assert!(compute_dag_topo_order(two_cycle).is_none());
        assert!(!compute_dag_is_acyclic(two_cycle));
    }

    #[test]
    fn compute_dag_topo_order_diamond() {
        // f3 = g(f1, f2); f1 = h(f0); f2 = k(f0). f0 leaf. f1,f2 before f3.
        let diamond = &[
            ComputeEdge {
                target: 1,
                inputs: &[0],
            },
            ComputeEdge {
                target: 2,
                inputs: &[0],
            },
            ComputeEdge {
                target: 3,
                inputs: &[1, 2],
            },
        ];
        let order = compute_dag_topo_order(diamond).expect("acyclic");
        let p = |t: u8| order.iter().position(|&x| x == t).unwrap();
        assert!(
            p(1) < p(3) && p(2) < p(3),
            "both precedents before the join"
        );
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn field_mask_is_presence_bits() {
        let m = FieldMask::from_positions(&[0, 2]); // amount + partner populated, tax absent
        assert!(m.has(0) && !m.has(1) && m.has(2));
        assert_eq!(m.count(), 2);
        assert!(!m.is_empty() && FieldMask::EMPTY.is_empty());
        assert_eq!(
            FieldMask::EMPTY.with(1).with(1),
            FieldMask::from_positions(&[1])
        );

        // Out-of-range positions are IGNORED, never folded onto a valid bit
        // (Codex P2 #441): position 64 must NOT alias to bit 0.
        assert_eq!(
            FieldMask::from_positions(&[64]),
            FieldMask::EMPTY,
            "position 64 must be ignored, not aliased to bit 0"
        );
        assert!(
            !FieldMask::EMPTY.with(64).has(0),
            "with(64) must not set bit 0"
        );
        assert!(
            !FieldMask::from_positions(&[0]).has(64),
            "has(64) must be false, not bit-0 aliased"
        );
        // In-range bit 0 unaffected by the out-of-range guard.
        assert!(FieldMask::from_positions(&[0, 64]).has(0));
        assert_eq!(
            FieldMask::from_positions(&[0, 64]).count(),
            1,
            "only the in-range bit 0 is set"
        );
    }

    #[test]
    fn field_mask_inherit_is_nondestructive_union() {
        // inherit = bitwise OR — a child IS-A its parent: it carries the parent's
        // present fields PLUS its own delta (focused cover, CodeRabbit #442).
        let parent = FieldMask::from_positions(&[0, 2]);
        let delta = FieldMask::from_positions(&[1, 2]); // bit 2 overlaps
        let child = parent.inherit(delta);
        assert_eq!(
            child,
            FieldMask(parent.0 | delta.0),
            "inherit is the bitwise union"
        );
        assert!(child.has(0) && child.has(1) && child.has(2));
        assert_eq!(
            child.count(),
            3,
            "the overlapping bit is not double-counted"
        );
        // EMPTY is the identity, both directions; the union is commutative.
        assert_eq!(parent.inherit(FieldMask::EMPTY), parent);
        assert_eq!(FieldMask::EMPTY.inherit(parent), parent);
        assert_eq!(parent.inherit(delta), delta.inherit(parent), "commutative");
        // FieldMask is Copy — neither operand is mutated by inherit.
        assert_eq!(parent, FieldMask::from_positions(&[0, 2]));
        assert_eq!(delta, FieldMask::from_positions(&[1, 2]));
    }

    #[test]
    fn meta_dto_projects_above_agnostic_class_mask() {
        let classes = FakeClasses::new();
        // The SoA supplied ONLY (class_id=7, mask) — no labels. The meta-DTO
        // resolves the labels from above.
        let mask = FieldMask::from_positions(&[0, 2]); // tax (pos 1) is off
        let projected: Vec<(&str, bool)> = classes
            .project(7, mask)
            .map(|(f, present)| (f.label.as_str(), present))
            .collect();
        assert_eq!(
            projected,
            vec![("Total", true), ("Tax", false), ("Partner", true)],
            "labels come from the cache above the SoA; presence comes from the mask"
        );
        // The render template skips off-bits: only present fields surface.
        let rendered: Vec<&str> = classes
            .project(7, mask)
            .filter(|(_, present)| *present)
            .map(|(f, _)| f.label.as_str())
            .collect();
        assert_eq!(rendered, vec!["Total", "Partner"], "off-bit (Tax) skipped");
    }

    #[test]
    fn field_label_resolves_late_from_class_not_row() {
        let classes = FakeClasses::new();
        assert_eq!(classes.field_label(7, 1), Some("Tax"));
        assert_eq!(classes.field_label(7, 9), None); // out of range
        assert_eq!(classes.field_count(7), 3);
        assert_eq!(classes.field_count(999), 0); // unknown class
    }

    #[test]
    fn render_rows_skips_off_bits_presence_only() {
        let classes = FakeClasses::new();
        // Tax (pos 1) is off → it must NOT produce a render row (C2: off-bits skipped).
        let rows = classes.render_rows(7, FieldMask::from_positions(&[0, 2]));
        assert_eq!(rows.len(), 2, "only the 2 populated fields render");
        assert_eq!(
            rows[0],
            RenderRow {
                label: "Total",
                predicate: "amount_total"
            }
        );
        assert_eq!(
            rows[1],
            RenderRow {
                label: "Partner",
                predicate: "partner_id"
            }
        );
        // Empty mask → zero rows (no template branch needed, just an empty iteration).
        assert!(classes.render_rows(7, FieldMask::EMPTY).is_empty());
        // Full mask → all 3 rows, in class order (the bit basis).
        let all = classes.render_rows(7, FieldMask::from_positions(&[0, 1, 2]));
        assert_eq!(
            all.iter().map(|r| r.label).collect::<Vec<_>>(),
            vec!["Total", "Tax", "Partner"]
        );
    }

    #[test]
    fn value_schema_default_is_full_temporary_poc() {
        // TEMPORARY (2026-06-15 POC): the blanket ClassView default materialises the
        // FULL value slab so consumers (tesseract-rs / woa-rs / medcare-rs / q2)
        // transcode against a populated NodeRow. Specialisation is opt-IN (override
        // to a smaller preset). When the POC phase ends, revert the default to
        // `ValueSchema::Bootstrap` AND this test together.
        use crate::canonical_node::{EdgeCodecFlavor, ValueSchema};
        let classes = FakeClasses::new();
        // The default class (classid 0x0000_0000) and any unconfigured class both
        // resolve to Full while the POC default is active.
        assert_eq!(classes.value_schema(0), ValueSchema::Full);
        assert_eq!(classes.value_schema(7), ValueSchema::Full);
        // The edge-codec axis is SEPARATE and untouched (still the CoarseOnly
        // zero-fallback) — only the value slab flipped to Full.
        assert_eq!(classes.edge_codec_flavor(0), EdgeCodecFlavor::CoarseOnly);
        // The TYPE-level default is unchanged: substrate zero-fallback stays Bootstrap.
        assert_eq!(ValueSchema::default(), ValueSchema::Bootstrap);
    }

    // ── facet value projection + is_a-walk (unified ClassView render) ────────────

    /// `facet_rows` binds field position `i` to facet byte `i`, skips mask-off
    /// positions, and skips value-slab positions `>= 12` (never folded onto byte
    /// `i & 11`). A 14-field class with a mix of present positions proves all three.
    #[test]
    fn facet_rows_binds_position_to_byte_skips_off_and_over_twelve() {
        // A 14-field class (positions 0..13) — 12 facet-backed + 2 value-slab.
        let big: Vec<FieldRef> = (0u8..14)
            .map(|n| FieldRef::new(format!("p{n}"), format!("L{n}")))
            .collect();
        let classes = FakeClasses::new().with_class(20, big);
        // facet byte i = 100 + i, so the byte carried is unambiguous per position.
        let facet: [u8; 12] = std::array::from_fn(|i| 100 + i as u8);
        // Present: 0 and 2 (facet-backed), 1 is OFF, 12 and 13 are value-slab (>=12).
        let mask = FieldMask::from_positions(&[0, 2, 12, 13]);
        let rows = classes.facet_rows(20, mask, &facet);
        // Only the two facet-backed present positions survive: 0 and 2.
        assert_eq!(
            rows.len(),
            2,
            "off-bit (1) and value-slab bits (12,13) skipped"
        );
        assert_eq!(
            rows[0],
            ValueRow {
                label: "L0",
                predicate: "p0",
                position: 0,
                value: 100
            }
        );
        assert_eq!(
            rows[1],
            ValueRow {
                label: "L2",
                predicate: "p2",
                position: 2,
                value: 102
            },
            "position 2 reads facet byte 2, not folded/aliased"
        );
        // Empty mask → zero rows.
        assert!(classes.facet_rows(20, FieldMask::EMPTY, &facet).is_empty());
        // A mask of ONLY value-slab positions (>=12) → zero facet rows.
        assert!(
            classes
                .facet_rows(20, FieldMask::from_positions(&[12, 13]), &facet)
                .is_empty(),
            "positions >= 12 are value-slab fields, out of facet scope"
        );
    }

    /// FULL mask on a 3-field class yields exactly 3 value rows, each carrying its
    /// own facet byte in class order (the bit basis).
    #[test]
    fn facet_rows_full_mask_three_field_class() {
        let classes = FakeClasses::new(); // class 7 = invoice (3 fields)
        let facet: [u8; 12] = std::array::from_fn(|i| (i as u8) * 3); // 0,3,6,9,...
        let rows = classes.facet_rows(7, FieldMask::FULL, &facet);
        assert_eq!(
            rows.len(),
            3,
            "all three invoice fields are facet-backed & present"
        );
        assert_eq!(
            rows.iter()
                .map(|r| (r.label, r.position, r.value))
                .collect::<Vec<_>>(),
            vec![("Total", 0, 0), ("Tax", 1, 3), ("Partner", 2, 6)],
        );
    }

    /// `resolve_render_class`: a child with an empty field set resolves UP the is_a
    /// chain to the nearest ancestor that HAS a card (the middle rung of the ladder).
    #[test]
    fn resolve_render_class_walks_is_a_to_ancestor_card() {
        // class 30 is empty; 30 is_a 7 (invoice, which has fields) → resolves to 7.
        let classes = FakeClasses::new().with_isa(30, 7);
        assert!(
            classes.fields(30).is_empty(),
            "child 30 has no bespoke card"
        );
        assert_eq!(
            classes.resolve_render_class(30),
            7,
            "empty child falls through to its is_a ancestor's card"
        );
        // A class that ALREADY has a card resolves to itself (top rung, no walk).
        assert_eq!(classes.resolve_render_class(7), 7);
    }

    /// An orphan (no is_a parent, no fields) resolves to ITSELF — the monotonic
    /// bottom rung: the caller renders the generic facet dump.
    #[test]
    fn resolve_render_class_orphan_returns_itself() {
        let classes = FakeClasses::new(); // no taxonomy edges for class 99
        assert!(classes.fields(99).is_empty());
        assert_eq!(
            classes.resolve_render_class(99),
            99,
            "no parent + no card → return original (generic facet dump signal)"
        );
    }

    /// A `subClassOf` cycle (A is_a B is_a A, both empty) terminates and returns the
    /// ORIGINAL class — the ladder never loops, never fails.
    #[test]
    fn resolve_render_class_cycle_terminates_at_original() {
        let classes = FakeClasses::new().with_isa(40, 41).with_isa(41, 40);
        assert!(classes.fields(40).is_empty() && classes.fields(41).is_empty());
        assert_eq!(
            classes.resolve_render_class(40),
            40,
            "a 2-cycle of empty classes returns the original, no infinite loop"
        );
        // A self-loop (A is_a A, empty) likewise returns the original.
        let selfloop = FakeClasses::new().with_isa(50, 50);
        assert_eq!(selfloop.resolve_render_class(50), 50);
    }

    /// A chain of 20 empty parents exceeds the 16-hop cap: the walk stops and returns
    /// the original class (depth cap respected, no panic).
    #[test]
    fn resolve_render_class_depth_cap_returns_original() {
        // 200 is_a 201 is_a ... is_a 220 — 21 empty classes, 20 edges, none carry a card.
        let mut classes = FakeClasses::new();
        for c in 200u16..220 {
            classes = classes.with_isa(c, c + 1);
        }
        assert_eq!(
            classes.resolve_render_class(200),
            200,
            "> 16 hops of empty ancestors → cap hit → return original (generic dump)"
        );
    }

    /// Regression canary: the additive facet-value / is_a surface must NOT perturb
    /// `render_rows` (C2 presence-only, late label/predicate resolution). The
    /// value-projected `facet_rows` surfaces the SAME present positions, plus bytes.
    #[test]
    fn render_rows_unchanged_by_facet_addition_regression_canary() {
        let classes = FakeClasses::new();
        let mask = FieldMask::from_positions(&[0, 2]); // Tax (pos 1) off
                                                       // render_rows: exact pre-existing shape (label + predicate, off-bit skipped).
        let rows = classes.render_rows(7, mask);
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            RenderRow {
                label: "Total",
                predicate: "amount_total"
            }
        );
        assert_eq!(
            rows[1],
            RenderRow {
                label: "Partner",
                predicate: "partner_id"
            }
        );
        // facet_rows is the value-projected sibling: same present set, same order.
        let facet: [u8; 12] = std::array::from_fn(|i| i as u8);
        let vrows = classes.facet_rows(7, mask, &facet);
        assert_eq!(
            vrows.iter().map(|r| r.label).collect::<Vec<_>>(),
            rows.iter().map(|r| r.label).collect::<Vec<_>>(),
            "facet_rows presents the identical present fields render_rows does"
        );
        assert_eq!(
            vrows.iter().map(|r| r.position).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(
            vrows.iter().map(|r| r.value).collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    // ── WideFieldMask widening (Ruling c, D-FIELDMASK-WIDENING) ───────────────
    // `FieldMask` above is UNTOUCHED by the widening — these tests prove (1)
    // every pre-existing u64 call shape still compiles/behaves identically,
    // and (2) the new sibling type covers the >64 case the old type cannot.

    /// (L-1 acceptance #1) Every existing `FieldMask` u64 constructor/method
    /// shape still compiles and round-trips exactly as before the widening
    /// landed — the non-footgun acceptance criterion, pinned directly.
    #[test]
    fn u64_constructors_unchanged() {
        // Tuple construction (the exact call shape OGAR's render crate uses).
        let m = FieldMask(0b101);
        assert!(m.has(0) && !m.has(1) && m.has(2));
        // The new `From<u64>` is ADDITIVE sugar, not a replacement: it agrees
        // with tuple construction bit-for-bit.
        assert_eq!(FieldMask::from(0b101u64), m);
        // EMPTY / FULL / MAX_FIELDS unchanged.
        assert_eq!(FieldMask::EMPTY, FieldMask(0));
        assert_eq!(FieldMask::FULL, FieldMask(u64::MAX));
        assert_eq!(FieldMask::MAX_FIELDS, 64);
        // `FieldMask` is still `Copy`: `m` is usable after being read into `n`
        // with no `.clone()` — if the widening had swapped `FieldMask`'s
        // internals for a `Box`-bearing enum, this would fail to compile.
        let n = m;
        assert_eq!(m, n);
        assert_eq!(m.count(), 2);
        // `with`/`from_positions`/out-of-range-ignored semantics unchanged.
        assert_eq!(FieldMask::EMPTY.with(0).with(2), m);
        assert_eq!(FieldMask::from_positions(&[0, 2]), m);
        assert!(!FieldMask::EMPTY.with(64).has(0), "with(64) still a no-op");
    }

    /// (L-1 acceptance #2) A 70-field mask sets and reads back position 65 —
    /// the exact case `FieldMask` cannot represent (SPEC-2 (i) pins the old
    /// type's silent drop at this same position).
    #[test]
    fn wide_class_positions_beyond_64_are_representable() {
        let mask = WideFieldMask::EMPTY.with(0).with(65);
        assert!(mask.has(0));
        assert!(
            mask.has(65),
            "position 65 must be representable, not dropped"
        );
        assert!(!mask.has(1) && !mask.has(64));
        assert_eq!(mask.count(), 2);
        assert!(mask.max_fields() >= 66);

        // A full 70-field class, built position-by-position, keeps every bit.
        let seventy = WideFieldMask::from_positions(&(0u8..70).collect::<Vec<_>>());
        for i in 0..70u8 {
            assert!(seventy.has(i), "position {i} must be present");
        }
        assert_eq!(seventy.count(), 70);
    }

    /// A mask that never crosses the 64 boundary stays in the zero-allocation
    /// `Small` repr — the "never allocates" half of the allocate-on-demand
    /// contract.
    #[test]
    fn small_mask_never_allocates() {
        let mask = WideFieldMask::from_positions(&[0, 5, 63]);
        assert!(
            matches!(mask.0, WideRepr::Small(_)),
            "a <= 64 mask must stay Small (zero heap), not promote to Wide"
        );
        assert_eq!(mask.max_fields(), FieldMask::MAX_FIELDS);

        // Crossing 64 promotes exactly once, and only then.
        let wide = mask.with(64);
        assert!(
            matches!(wide.0, WideRepr::Wide(_)),
            "setting position 64 must promote to Wide"
        );
    }

    /// `intersect`/`union` combine masks correctly regardless of which side
    /// (or both) has promoted to `Wide` — no bit is aliased or dropped across
    /// the tier boundary.
    #[test]
    fn intersect_union_across_tiers() {
        let small = WideFieldMask::from_positions(&[0, 2]); // stays Small
        let wide = WideFieldMask::from_positions(&[2, 70]); // promotes to Wide

        let inter = small.intersect(&wide);
        assert!(inter.has(2) && !inter.has(0) && !inter.has(70));
        assert_eq!(inter.count(), 1);

        let uni = small.union(&wide);
        assert!(uni.has(0) && uni.has(2) && uni.has(70));
        assert_eq!(uni.count(), 3);

        // Order independence (both folds are commutative).
        assert_eq!(small.intersect(&wide), wide.intersect(&small));
        assert_eq!(small.union(&wide), wide.union(&small));

        // Wide ∩/∪ Wide (both already promoted) also folds correctly.
        let wide2 = WideFieldMask::from_positions(&[70, 71]);
        let wide_inter = wide.intersect(&wide2);
        assert!(wide_inter.has(70) && !wide_inter.has(71) && !wide_inter.has(2));
    }

    /// `full_for(field_count)` is the wide-tier sibling of `FieldMask::FULL`:
    /// every position `< field_count` present, nothing at or beyond it.
    #[test]
    fn full_for_wide_class_emits_all() {
        // account.move-scale case: 109 fields, all present.
        let full = WideFieldMask::full_for(109);
        for i in 0..109u8 {
            assert!(
                full.has(i),
                "position {i} must be present under full_for(109)"
            );
        }
        assert!(
            !full.has(109),
            "field_count is exclusive of its own boundary"
        );
        assert_eq!(full.count(), 109);

        // A <= 64 field_count stays in the Small tier (agrees with
        // FieldMask::FULL's bit pattern when the class happens to be 64-wide).
        let full64 = WideFieldMask::full_for(64);
        assert!(matches!(full64.0, WideRepr::Small(bits) if bits == u64::MAX));
        assert_eq!(full64, WideFieldMask::from(u64::MAX));

        // 0 fields → EMPTY, not a panic.
        assert_eq!(WideFieldMask::full_for(0), WideFieldMask::EMPTY);
    }

    /// Promoting a `FieldMask` into `WideFieldMask` (e.g. a caller migrating a
    /// <=64 class onto the wide API) preserves every bit and stays `Small`.
    #[test]
    fn field_mask_promotes_into_wide_field_mask_losslessly() {
        let narrow = FieldMask::from_positions(&[1, 3, 60]);
        let promoted: WideFieldMask = narrow.into();
        assert!(matches!(promoted.0, WideRepr::Small(_)));
        for i in 0..64u8 {
            assert_eq!(promoted.has(i), narrow.has(i), "bit {i} must round-trip");
        }
    }

    /// (V-L P0 regression) `intersect`/`union` results whose high chunks
    /// collapse away (down to <= 64 populated positions, or fewer significant
    /// chunks than either input) must compare equal to — and hash identically
    /// with — the canonical `Small`/`Wide` mask built directly from the same
    /// position set. Before this fix, `zip_fold` always returned an
    /// un-normalized `Wide`, so e.g. `from_positions(&[2, 70]).intersect(&
    /// from_positions(&[2]))` was logically `{2}` but compared UNEQUAL to
    /// `from_positions(&[2])` and hashed differently — a footgun for any
    /// `HashSet`/`HashMap` keyed on `WideFieldMask` (RBAC folds, DTO dedup).
    #[test]
    fn intersect_result_collapsing_to_small_equals_canonical_and_hashes_identically() {
        use std::collections::hash_map::DefaultHasher;

        fn hash_of(m: &WideFieldMask) -> u64 {
            let mut h = DefaultHasher::new();
            m.hash(&mut h);
            h.finish()
        }

        // intersect: {2, 70} ∩ {2} = {2}, but the left side is Wide — the
        // fold must not leave a phantom high chunk behind.
        let wide = WideFieldMask::from_positions(&[2, 70]);
        let small = WideFieldMask::from_positions(&[2]);
        let inter = wide.intersect(&small);
        let canonical = WideFieldMask::from_positions(&[2]);

        assert!(matches!(inter.0, WideRepr::Small(_)), "collapsed intersect result must demote to Small, not stay a Wide value with an all-zero high chunk");
        assert_eq!(inter, canonical);
        assert_eq!(canonical, inter, "PartialEq must be symmetric");
        assert_eq!(hash_of(&inter), hash_of(&canonical));

        // union: also exercised, since it shares `zip_fold`.
        let a = WideFieldMask::from_positions(&[70]);
        let b = WideFieldMask::from_positions(&[70]);
        // Force the fold path (both sides already Wide) rather than the
        // Small/Small fast path in `union`.
        let uni = a.union(&b);
        let canonical_uni = WideFieldMask::from_positions(&[70]);
        assert_eq!(uni, canonical_uni);
        assert_eq!(hash_of(&uni), hash_of(&canonical_uni));

        // Wide-vs-Wide with different underlying chunk-vector *lengths* that
        // denote the same field set: {2, 70} built directly (2 chunks) vs a
        // 3-chunk Wide value ({2, 70, 200} minus {200}) that collapses back
        // down to the same 2 significant chunks.
        let direct_two_chunks = WideFieldMask::from_positions(&[2, 70]);
        let three_chunk_source = WideFieldMask::from_positions(&[2, 70, 200]);
        let only_200 = WideFieldMask::from_positions(&[200]);
        // {2, 70, 200} minus {200}, expressed as intersect-with-complement via
        // union/xor-free construction: intersect against the union of the
        // low two chunks' full range plus nothing at chunk 3, i.e. simply
        // intersect the 3-chunk value against a mask that has {2, 70} set
        // (which itself is 2 chunks) — this drives zip_fold's `len =
        // a.len().max(b.len())` down from 4 chunks (chunk index 3 for
        // position 200) to the 2 significant chunks that remain non-zero.
        let three_chunk_minus_200 = three_chunk_source.intersect(&direct_two_chunks);
        assert!(!only_200.is_empty()); // sanity: 200 really set a bit somewhere
        assert_eq!(three_chunk_minus_200, direct_two_chunks);
        assert_eq!(hash_of(&three_chunk_minus_200), hash_of(&direct_two_chunks));
    }

    // ── from_universe_present minter (Q6, odoo-rs view_mask relocation) ──────

    /// A universe of 257 fields exceeds the `u8`-position cap — refused loudly,
    /// mirroring odoo-rs's `mint_wide_mask` cap error.
    #[test]
    fn from_universe_present_rejects_universe_over_256() {
        let universe: Vec<String> = (0..257).map(|i| format!("f{i:03}")).collect();
        let universe_refs: Vec<&str> = universe.iter().map(String::as_str).collect();
        let err = WideFieldMask::from_universe_present(&universe_refs, &[]).unwrap_err();
        assert_eq!(err, WideMaskCapError::UniverseExceedsSocCap { fields: 257 });
    }

    /// Membership agrees with the raw `from_positions` construction on a mixed
    /// 70-field universe — mirrors `wide_mask_agrees_with_mask_words` in
    /// odoo-rs's `view_mask.rs`.
    #[test]
    fn from_universe_present_agrees_with_from_positions() {
        let universe: Vec<String> = (0..70).map(|i| format!("f{i:02}")).collect();
        let universe_refs: Vec<&str> = universe.iter().map(String::as_str).collect();
        let present = ["f00", "f63", "f69"];

        let minted = WideFieldMask::from_universe_present(&universe_refs, &present)
            .expect("universe within cap");
        let direct = WideFieldMask::from_positions(&[0, 63, 69]);

        assert_eq!(minted, direct);
        assert_eq!(minted.count(), 3);
        for i in 0..universe.len() as u8 {
            assert_eq!(minted.has(i), direct.has(i));
        }
    }
}
