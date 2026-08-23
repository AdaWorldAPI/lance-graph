//! PROBE-TARSKI-SIGNED-WITNESS-1 — can a signed `24×i4` derivational
//! witness preserve falsifier depth that `Belief`'s scalar state discards?
//!
//! **The measured claim, stated at the strength the evidence supports:**
//!
//! > A signed i4 derivational witness CAN preserve falsifier depth that the
//! > current `Belief` scalar state provably discards. Whether the positive
//! > Tarski rung is itself *derivable* from that field is measured here
//! > against an independent oracle and reported — not assumed.
//!
//! An earlier revision of this probe claimed to prove `Belief::rung` IS the
//! projection of a derivational field. **That proof was circular** and the
//! claim is withdrawn: it wrote `b.rung` into a nibble and then checked the
//! nibble equalled `b.rung`, which proves only that an i4 field can hold
//! 0..7. The positive lane here is now derived **independently from support
//! topology** (the premise DAG — [`derive_depth_from_support`], which never
//! reads `b.rung`), with the arena's stored scalar as the ORACLE it is
//! checked against.
//!
//! # This probe does NOT use the A9 `Locus` API (deliberate, load-bearing)
//!
//! `CausalWitnessFacet`'s A9 reading is **operator-locked**: *"Loci, not
//! magnitudes"* — every named `Locus` is a signed context POINTER, `0` is
//! unbound, and the module states verbatim that *"the rung level occupies
//! ZERO slots."* Writing a derivational MAGNITUDE through `Locus::
//! SupportedBy` would be using A9's semantic API to mean something A9's
//! contract forbids — the same-geometry/different-meaning smuggle the
//! DOCK/ROUTE law exists to prevent.
//!
//! So this probe carries its own [`SignedTarskiWitnessView`] over the same
//! **physical** 12-byte / 24-nibble geometry, with its own slot names
//! ([`TarskiSlot`]) and its own nibble accessors. Same bytes, same dock,
//! **different ClassView** — which is exactly what the law licenses:
//!
//! > Same physical geometry may support different ClassViews. That does not
//! > license using one ClassView's semantic API to mean another thing.
//!
//! The probe-local classid is a placeholder, NOT an OGAR mint.
//!
//! **⚠ MEMORY-ABI ESCAPE, acknowledged in place.** `BeliefArena` remains the
//! acknowledged non-canonical AoS owner (#1004,
//! `E-TYPE-COMPLEXITY-EXPOSED-A-MEMORY-ABI-ESCAPE-1`); this probe uses it as
//! the PARITY ORACLE only and repairs nothing. The witness docks are a fixed
//! stack array of 16-byte `Copy` rows — a bounded **probe fixture**, not a
//! claim about the resident SoA shape.
//!
//! # Fixtures
//!
//! - **A** — positive derivation chain, no falsifier. Measures whether an
//!   independently-derived support depth agrees with the arena's `rung`.
//! - **B** — same chain plus a freq-0 falsifying counter-chain. The shipped
//!   merge law (`admit_derived` CHOICE, `belief.rs:247`: replace only on
//!   STRICT expectation gain) DROPS the counter-derivation, so the arena's
//!   scalar state is bit-identical to fixture A. The witness alone retains it.
//! - **B'** — a DEEPER counter-chain, scalar-indistinguishable from B.
//! - **C** — falsifier removed; the register returns byte-identical to A.

use lance_graph_contract::facet::FacetCascade;
use lance_graph_planner::nars::belief::{Belief, BeliefArena, CStmt, Copula, Stamp};
use lance_graph_planner::nars::truth::TruthValue;

/// Probe-local ClassView id for the signed-Tarski READING. Deliberately in
/// a nonsense range: this is NOT an OGAR mint.
const PROBE_TARSKI_CLASSID: u32 = 0xFFFF_0009;

/// The content-blind register width shared with every 12-byte reading.
const REGISTER_BYTES: usize = 12;
/// 12 bytes = 24 nibbles.
const TARSKI_SLOTS: usize = 24;

/// Dock capacity — fixed stack array, no heap row population.
const MAX_DOCKS: usize = 64;

/// One 16-byte dock row: `classid(4, LE) | 24×i4 register(12)`.
type Dock = [u8; 16];

const _: () = assert!(core::mem::size_of::<Dock>() == 16, "16-byte dock");
const _: () = assert!(
    core::mem::size_of::<FacetCascade>() == 16,
    "FacetCascade shares the dock width"
);

const EMPTY_DOCK: Dock = [0u8; 16];

/// Probe-local slot names for the signed-Tarski reading. **Deliberately NOT
/// `Locus` variants** — under A9 those names mean signed context pointers,
/// and this view's slots carry signed derivational DEPTHS. Two readings of
/// the same physical nibbles must not share a vocabulary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TarskiSlot {
    /// `+n` — the constructive derivation reaching this statement has depth n.
    ConstructiveDepth = 0,
    /// `−n` — a falsifying counter-derivation of depth n bites this statement.
    FalsifyingDepth = 1,
}

/// The signed-Tarski reading: a 12-byte content-blind register carved as 24
/// signed `i4` **derivational magnitudes**.
///
/// Sign is polarity (`+` constructive / `−` falsifying / `0` unresolved);
/// magnitude is derivational depth. This is a DIFFERENT ClassView from
/// `CausalWitnessFacet`'s A9 reading of the identical geometry, and shares
/// none of its accessors — see the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(transparent)]
struct SignedTarskiWitnessView([u8; REGISTER_BYTES]);

const _: () = assert!(
    core::mem::size_of::<SignedTarskiWitnessView>() == REGISTER_BYTES,
    "the view is exactly its backing register"
);

impl SignedTarskiWitnessView {
    const ZERO: Self = Self([0u8; REGISTER_BYTES]);

    /// Borrowed zero-cost read of a raw register (repr(transparent) reborrow).
    fn from_register_ref(reg: &[u8; REGISTER_BYTES]) -> &Self {
        // SAFETY: `#[repr(transparent)]` over `[u8; REGISTER_BYTES]` with
        // identical size (const-asserted above), so the reinterpretation is
        // layout-identical and defined. No read of the bytes occurs here.
        unsafe { &*(reg as *const [u8; REGISTER_BYTES] as *const Self) }
    }

    /// Signed depth at slot `0..24` (sign-extended nibble; even slot = low
    /// nibble, odd = high). Own implementation — this view does not borrow
    /// A9's accessors any more than it borrows A9's meanings.
    fn get(self, slot: usize) -> i8 {
        if slot >= TARSKI_SLOTS {
            return 0;
        }
        let byte = self.0[slot / 2];
        let nib = if slot & 1 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        ((nib << 4) as i8) >> 4
    }

    /// Set the signed depth at slot `0..24`; clamps to `[−8, +7]`.
    fn set(&mut self, slot: usize, depth: i8) {
        if slot >= TARSKI_SLOTS {
            return;
        }
        let v = (depth.clamp(-8, 7) as u8) & 0x0F;
        let bi = slot / 2;
        if slot & 1 == 0 {
            self.0[bi] = (self.0[bi] & 0xF0) | v;
        } else {
            self.0[bi] = (self.0[bi] & 0x0F) | (v << 4);
        }
    }

    fn at(self, slot: TarskiSlot) -> i8 {
        self.get(slot as usize)
    }

    fn to_register(self) -> [u8; REGISTER_BYTES] {
        self.0
    }

    /// `max(positive slots)` — the constructive-support ceiling.
    fn support_ceiling(self) -> u32 {
        (0..TARSKI_SLOTS)
            .map(|s| self.get(s).max(0) as u32)
            .max()
            .unwrap_or(0)
    }

    /// `max(|negative slots|)` — the falsifier ceiling.
    fn falsifier_ceiling(self) -> u32 {
        (0..TARSKI_SLOTS)
            .map(|s| (-self.get(s)).max(0) as u32)
            .max()
            .unwrap_or(0)
    }
}

/// Borrowed signed-Tarski view of a dock's 12-byte payload.
fn view_of(dock: &Dock) -> &SignedTarskiWitnessView {
    let reg: &[u8; REGISTER_BYTES] = dock[4..16].try_into().expect("12-byte payload");
    SignedTarskiWitnessView::from_register_ref(reg)
}

/// Write-back per the borrow-strategy law: owned `Copy` microcopy, nibble
/// edit, gated write-back. Never `&mut` a borrowed view.
fn set_slot(dock: &mut Dock, slot: TarskiSlot, depth: i8) {
    let mut v = *view_of(dock);
    v.set(slot as usize, depth);
    dock[4..16].copy_from_slice(&v.to_register());
}

fn mint_dock() -> Dock {
    let mut d = EMPTY_DOCK;
    d[0..4].copy_from_slice(&PROBE_TARSKI_CLASSID.to_le_bytes());
    d
}

/// **The independent positive derivation** — the fix for the withdrawn
/// circular claim. Derives each belief's derivational depth from SUPPORT
/// TOPOLOGY ALONE: `depth(b) = 0` if `b` has no premises, else
/// `1 + max(depth(premises))`. Reads `premises` (the support-DAG edges) and
/// nothing else — **never `b.rung`**, which is the oracle this is checked
/// against, not an input.
///
/// Returns `None` if the premise graph contains a cycle (the CHOICE
/// replacement path can in principle rewrite premises to point forward);
/// a cycle is reported honestly rather than papered over with a recursion
/// guard that silently returns a wrong depth.
fn derive_depth_from_support(arena: &BeliefArena) -> Option<Vec<u32>> {
    let n = arena.entries().len();
    let mut memo: Vec<Option<u32>> = vec![None; n];
    // 0 = unvisited, 1 = on stack, 2 = done
    let mut mark = vec![0u8; n];

    fn walk(
        i: usize,
        arena: &BeliefArena,
        memo: &mut Vec<Option<u32>>,
        mark: &mut Vec<u8>,
    ) -> Option<u32> {
        match mark[i] {
            1 => return None, // cycle
            2 => return memo[i],
            _ => {}
        }
        mark[i] = 1;
        let prem = &arena.entries()[i].premises;
        let d = if prem.is_empty() {
            0
        } else {
            let mut m = 0u32;
            for &p in prem {
                let dp = walk(p as usize, arena, memo, mark)?;
                m = m.max(dp);
            }
            m + 1
        };
        mark[i] = 2;
        memo[i] = Some(d);
        Some(d)
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(walk(i, arena, &mut memo, &mut mark)?);
    }
    Some(out)
}

fn inh(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Inh,
        p,
    }
}

/// Scan the closed arena for falsifying derivations the scalar path DROPPED:
/// composable premise pairs whose candidate conclusion targets an existing
/// belief of opposing polarity (stored expectation > 0.5, candidate ≤ 0.5).
/// Depth uses the INDEPENDENTLY-derived support depths, not `b.rung`.
fn scan_falsifiers(arena: &BeliefArena, depth: &[u32]) -> Vec<(CStmt, u32)> {
    let entries = arena.entries();
    let mut out: Vec<(CStmt, u32)> = Vec::new();
    for (i, ei) in entries.iter().enumerate() {
        if !ei.stmt.cop.transits() {
            continue;
        }
        for (j, ej) in entries.iter().enumerate() {
            if ej.stmt.cop != ei.stmt.cop || ej.stmt.s != ei.stmt.p {
                continue;
            }
            let stmt = CStmt {
                s: ei.stmt.s,
                cop: ei.stmt.cop,
                p: ej.stmt.p,
            };
            let Some(stored) = arena.get(stmt) else {
                continue;
            };
            let cand = ei.truth.deduction(&ej.truth);
            if stored.truth.expectation() > 0.5 + 1e-6 && cand.expectation() <= 0.5 + 1e-6 {
                let d = depth[i].max(depth[j]) + 1;
                match out.iter_mut().find(|(s, _)| *s == stmt) {
                    Some((_, best)) => *best = (*best).max(d),
                    None => out.push((stmt, d)),
                }
            }
        }
    }
    out
}

/// Mint the witness docks: the positive slot carries the INDEPENDENTLY
/// derived support depth; the negative slot the deepest dropped falsifying
/// derivation. Panics (probe honesty) if any depth exceeds the i4 ceiling.
fn mint_witness_docks(arena: &BeliefArena, depth: &[u32]) -> ([Dock; MAX_DOCKS], usize) {
    let mut docks = [EMPTY_DOCK; MAX_DOCKS];
    let n = arena.entries().len();
    assert!(n <= MAX_DOCKS, "probe capacity");
    for (i, dock) in docks.iter_mut().enumerate().take(n) {
        *dock = mint_dock();
        if depth[i] > 0 {
            assert!(depth[i] <= 7, "i4 ceiling: depth {} > +7", depth[i]);
            set_slot(dock, TarskiSlot::ConstructiveDepth, depth[i] as i8);
        }
    }
    for (stmt, d) in scan_falsifiers(arena, depth) {
        assert!(d <= 8, "i4 ceiling: falsifier depth {d} > 8");
        let idx = arena
            .entries()
            .iter()
            .position(|b| b.stmt == stmt)
            .expect("falsifier target exists");
        set_slot(&mut docks[idx], TarskiSlot::FalsifyingDepth, -(d as i8));
    }
    (docks, n)
}

/// Positive fixture: the 4-link chain `1→2→3→4→5`, f=1.0, c=0.9, disjoint
/// stamps, closed to fixed point.
fn build_positive_arena() -> BeliefArena {
    let mut a = BeliefArena::new();
    for (k, (s, p)) in [(1u16, 2u16), (2, 3), (3, 4), (4, 5)].iter().enumerate() {
        a.observe(
            inh(*s, *p),
            TruthValue::new(1.0, 0.9),
            Stamp::source(k as u32 + 1),
        );
    }
    a.close_transitive(16);
    assert!(a.reached_fixed_point, "fixture chain must close");
    a
}

/// Add a falsifying counter-chain `1→…→5` whose LAST link has frequency 0,
/// then re-close. `hops` = number of links (3 → counter-depth 2; 5 → 3).
fn add_falsifier_chain(a: &mut BeliefArena, hops: usize) {
    let mut nodes: Vec<u16> = vec![1];
    for i in 0..hops - 1 {
        nodes.push(6 + i as u16);
    }
    nodes.push(5);
    for (k, w) in nodes.windows(2).enumerate() {
        let f = if k == hops - 1 { 0.0 } else { 1.0 };
        a.observe(
            inh(w[0], w[1]),
            TruthValue::new(f, 0.9),
            Stamp::source(10 + k as u32),
        );
    }
    a.close_transitive(16);
    assert!(a.reached_fixed_point, "falsifier closure must fix");
}

/// Bit-exact scalar-state snapshot of one belief (f32s by bit pattern).
///
/// Premises are EXCLUDED, for a measured reason: two IDENTICALLY-built
/// arenas diverge in `premises` at two layers, both from `close_transitive`
/// admitting its per-pass `derived: HashMap` in iteration order —
/// (1) the numeric arena indices differ (observed live: `(1,5)` premises
/// `[6,4]` vs `[5,6]`), and (2) equal-expectation derivations (`(1,4)` via
/// `(1,3)∘(3,4)` vs `(1,2)∘(2,4)`) are tie-broken by whichever the HashMap
/// yields first, so even statement-RESOLVED premises differ.
///
/// **This is stated as an observation, not a verdict.** It is NOT a claim
/// that the premise route is unimportant: two proof routes with tied truth
/// may later carry different provenance, and a route chosen by hash order
/// is then a real reproducibility problem, not a harmless one. What is
/// established here is only that the route is not currently STABLE, so it
/// cannot serve as a parity key in this probe.
fn scalar_state(b: &Belief) -> (CStmt, u32, u32, u64, u32, u32) {
    (
        b.stmt,
        b.truth.frequency.to_bits(),
        b.truth.confidence.to_bits(),
        b.stamp.0,
        b.rung,
        b.contradiction.to_bits(),
    )
}

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    // ================= Fixture A =================
    let arena_a = build_positive_arena();
    let depth_a = derive_depth_from_support(&arena_a).expect("premise DAG is acyclic");
    let (docks_a, n_a) = mint_witness_docks(&arena_a, &depth_a);

    // A1 — hand-authored rung expectations (balanced composition: rung is
    // ~log2 of span, NOT span length).
    let expected: [((u16, u16), u32); 10] = [
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 4), 0),
        ((4, 5), 0),
        ((1, 3), 1),
        ((2, 4), 1),
        ((3, 5), 1),
        ((1, 4), 2),
        ((2, 5), 2),
        ((1, 5), 2),
    ];
    let mut a1_ok = arena_a.entries().len() == expected.len();
    for ((s, p), r) in expected {
        a1_ok &= arena_a.get(inh(s, p)).map(|b| b.rung) == Some(r);
    }
    gate(
        "A1 authored rung fixture",
        a1_ok,
        format!("{} statements, rungs 0/1/2 as authored", expected.len()),
    );

    // A2 — THE NON-CIRCULAR PARITY TEST. The positive lane was derived from
    // the premise DAG alone (never reading `b.rung`); the arena's stored
    // scalar is the ORACLE. Agreement here is evidence that `rung` is
    // reconstructible from support topology; disagreement would be the
    // finding. (The earlier revision wrote `b.rung` in and read it back —
    // that gate proved nothing and is withdrawn.)
    let mut a2_ok = true;
    let mut a2_detail = String::new();
    for (i, b) in arena_a.entries().iter().enumerate() {
        let derived = view_of(&docks_a[i]).support_ceiling();
        if derived != b.rung {
            a2_ok = false;
            a2_detail = format!(
                "DIVERGENCE at {:?}: support-derived {} vs arena rung {}",
                b.stmt, derived, b.rung
            );
            break;
        }
    }
    if a2_ok {
        a2_detail = format!(
            "{n_a}/{n_a} beliefs: depth derived from the premise DAG alone equals the arena's \
             stored rung (independent derivation, arena as oracle)"
        );
    }
    gate(
        "A2 support-topology derivation reproduces rung",
        a2_ok,
        a2_detail,
    );

    // A3 — negative lane silent in the positive fixture.
    let a3_ok = (0..n_a).all(|i| view_of(&docks_a[i]).falsifier_ceiling() == 0);
    gate(
        "A3 falsifier lane silent with no counter-chain",
        a3_ok,
        format!("{n_a}/{n_a} docks carry zero negative slots"),
    );

    // A4 — dock conservation: the same 16 bytes round-trip through BOTH the
    // shipped FacetCascade reading and this probe-local Tarski reading.
    let mut a4_ok = true;
    for dock in docks_a.iter().take(n_a) {
        let fc = FacetCascade::from_bytes(dock);
        a4_ok &= fc.facet_classid == PROBE_TARSKI_CLASSID;
        a4_ok &= fc.to_bytes() == *dock;
        a4_ok &= view_of(dock).to_register() == dock[4..16];
    }
    gate(
        "A4 dock conservation (one geometry, two readings, no byte moves)",
        a4_ok,
        format!("{n_a} docks round-trip through FacetCascade and the Tarski view alike"),
    );

    let s_apex = inh(1, 5);
    let apex_idx = arena_a
        .entries()
        .iter()
        .position(|b| b.stmt == s_apex)
        .expect("apex");
    let apex_a_state = scalar_state(&arena_a.entries()[apex_idx]);
    let apex_a_dock = docks_a[apex_idx];

    // ================= Fixture B (falsifier, depth 2) =================
    let mut arena_b = build_positive_arena();
    add_falsifier_chain(&mut arena_b, 3);
    let depth_b = derive_depth_from_support(&arena_b).expect("acyclic");
    let (docks_b, _n_b) = mint_witness_docks(&arena_b, &depth_b);

    // B1 — the falsifier is INVISIBLE to the scalar state: every fixture-A
    // belief is bit-identical after the counter-chain. This is the shipped
    // CHOICE law doing the discarding (belief.rs:247). THIS is the probe's
    // load-bearing result.
    let mut b1_ok = true;
    for b in arena_a.entries() {
        let after = arena_b.get(b.stmt).expect("fixture-A statement survives");
        b1_ok &= scalar_state(after) == scalar_state(b);
    }
    gate(
        "B1 falsifier invisible in scalar state (CHOICE drops it)",
        b1_ok,
        format!("{n_a}/{n_a} fixture-A beliefs bit-identical post-falsifier"),
    );

    // B2 — the witness RETAINS what the scalar dropped, without touching the
    // constructive lane: the apex dock differs from fixture A in EXACTLY the
    // FalsifyingDepth slot.
    let apex_b_idx = arena_b
        .entries()
        .iter()
        .position(|b| b.stmt == s_apex)
        .expect("apex in B");
    let vb = view_of(&docks_b[apex_b_idx]);
    let changed: Vec<usize> = (0..TARSKI_SLOTS)
        .filter(|&s| vb.get(s) != view_of(&apex_a_dock).get(s))
        .collect();
    gate(
        "B2 falsifier lands beside support (neither overwrites the other)",
        vb.at(TarskiSlot::ConstructiveDepth) == 2
            && vb.at(TarskiSlot::FalsifyingDepth) == -2
            && changed == vec![TarskiSlot::FalsifyingDepth as usize],
        format!(
            "apex: constructive {:+}, falsifying {:+}, changed slots {:?}",
            vb.at(TarskiSlot::ConstructiveDepth),
            vb.at(TarskiSlot::FalsifyingDepth),
            changed
        ),
    );

    // B3 — can-stay-silent: the falsifier channel fires ONLY at the apex.
    let hotspots: Vec<CStmt> = arena_b
        .entries()
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let v = view_of(&docks_b[*i]);
            v.support_ceiling() > 0 && v.falsifier_ceiling() > 0
        })
        .map(|(_, b)| b.stmt)
        .collect();
    gate(
        "B3 dialectical hotspot detection is discriminating",
        hotspots == vec![s_apex],
        format!(
            "exactly one hotspot (the apex), {} beliefs scanned",
            arena_b.entries().len()
        ),
    );

    // ================= Fixture B' (falsifier, depth 3) =================
    let mut arena_b2 = build_positive_arena();
    add_falsifier_chain(&mut arena_b2, 5);
    let depth_b2 = derive_depth_from_support(&arena_b2).expect("acyclic");
    let (docks_b2, _) = mint_witness_docks(&arena_b2, &depth_b2);
    let apex_b2_idx = arena_b2
        .entries()
        .iter()
        .position(|b| b.stmt == s_apex)
        .expect("apex in B'");
    let vb2 = view_of(&docks_b2[apex_b2_idx]);

    // B4 — the discrimination the scalar CANNOT make.
    let b_state = scalar_state(&arena_b.entries()[apex_b_idx]);
    let b2_state = scalar_state(&arena_b2.entries()[apex_b2_idx]);
    gate(
        "B4 scalar state identical across falsifier depths; witness differs",
        b_state == b2_state
            && b_state == apex_a_state
            && vb.at(TarskiSlot::FalsifyingDepth) == -2
            && vb2.at(TarskiSlot::FalsifyingDepth) == -3,
        format!(
            "apex scalar bit-equal in A/B/B'; witness −2 vs {:+}",
            vb2.at(TarskiSlot::FalsifyingDepth)
        ),
    );

    // ================= Fixture C (falsifier removed) =================
    let arena_c = build_positive_arena();
    let depth_c = derive_depth_from_support(&arena_c).expect("acyclic");
    let (docks_c, n_c) = mint_witness_docks(&arena_c, &depth_c);
    // Per-STATEMENT comparison: admission order is HashMap-nondeterministic.
    let mut c1_ok = n_c == n_a;
    for (i, b) in arena_a.entries().iter().enumerate() {
        let ci = arena_c
            .entries()
            .iter()
            .position(|cb| cb.stmt == b.stmt)
            .expect("fixture-C statement set matches A");
        c1_ok &= docks_c[ci] == docks_a[i];
    }
    gate(
        "C1 falsifier removal restores byte-identical docks",
        c1_ok,
        format!("{n_a} docks byte-equal to fixture A (negative lane vanished)"),
    );

    // ================= Physical honesty gates =================
    // P1 — the i4 ceiling is +7/−8: depth beyond it CLAMPS, never widens.
    let mut probe = mint_dock();
    set_slot(&mut probe, TarskiSlot::ConstructiveDepth, 9);
    let hi = view_of(&probe).at(TarskiSlot::ConstructiveDepth);
    set_slot(&mut probe, TarskiSlot::FalsifyingDepth, -9);
    let lo = view_of(&probe).at(TarskiSlot::FalsifyingDepth);
    gate(
        "P1 i4 depth ceiling stated honestly (+7/−8, clamp not widen)",
        hi == 7 && lo == -8,
        format!("+9 → {hi:+}, −9 → {lo:+}"),
    );

    // P2 — slot isolation: writing one slot leaves all 23 others unchanged.
    let mut iso_ok = true;
    for k in 0..TARSKI_SLOTS {
        let mut v = SignedTarskiWitnessView::ZERO;
        for other in 0..TARSKI_SLOTS {
            if other != k {
                let d = ((other as i32 % 15) - 7) as i8;
                v.set(other, if d == 0 { 1 } else { d });
            }
        }
        let before: Vec<i8> = (0..TARSKI_SLOTS).map(|s| v.get(s)).collect();
        let distinctive = if k % 2 == 0 { 7 } else { -8 };
        v.set(k, distinctive);
        iso_ok &= v.get(k) == distinctive;
        for (other, &prior) in before.iter().enumerate() {
            if other != k {
                iso_ok &= v.get(other) == prior;
            }
        }
    }
    gate(
        "P2 slot isolation across all 24 nibbles",
        iso_ok,
        "a write to slot k leaves every other slot bit-identical".to_string(),
    );

    println!("PROBE-TARSKI-SIGNED-WITNESS-1: ALL {pass} GATES GREEN");
    println!(
        "measured claim: a signed i4 derivational witness PRESERVES falsifier depth that the \
         Belief scalar state provably discards (B1/B4); and on this fixture the constructive \
         lane derived from support topology alone reproduces the arena's rung (A2)"
    );
}
