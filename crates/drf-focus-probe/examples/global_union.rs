//! D-DR256 lab probe: 64 chaotically mapped 64x64 registers -> ONE global
//! 256x256 Boolean mask (65,536 bits, 1,024 u64 words, 8 KiB). Lab only.
//!
//! A register is one `SealedMap` (`ord[slot]` = the slot's canonical global
//! address, u16) plus one 4,096-bit local selection `L[r]`. The global set is
//! `G[g] = OR_{r,s} L[r][s] AND (ord_r[s] == g)`, a set union: membership only,
//! never a count, never provenance, never SQL K3 (inputs are TRUE-only masks).
//!
//! Arms, all producing the same 8 KiB mask from the same inputs:
//!   A  oracle: BTreeSet of resolved addresses, then bits from the set.
//!   B1 ndarray `mask_scatter_or_u32`, 64 calls unioning into one buffer.
//!   B2 mask-risc, ONE program over the 64 registers laid out as one 262,144-row
//!      population, `Terminal::ScatterOrU32 { out_rows: 65_536 }` (the
//!      executor zeroes its `Out::Mask`, so 64 programs could not union).
//!   C  `G AND h`, h = TERNLOG(X, Y, Z) of three global planes: h is a `Keep`
//!      over the 65,536-row universe; then per slot `Gather` of h through the
//!      map under `L` and `ScatterOrU32`. The scattered mask is never an input:
//!      `Terminal::ScatterOrU32`'s survival condition (ir.rs) forbids it.
//!
//! Run: `cargo run --release --example global_union` (from this crate).
//! Timings are ns per run, median of 31 batches, reported, not gated.
use drf_focus_probe::membership::BoundMap;
use drf_focus_probe::{MapId, Refusal, Scope, SealedMap, SLOTS, SourceId, TILE_WORDS};
use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::fuse::ternlog_imm;
use lance_graph_mask_risc::{
    scratch_words_for, words_for, Foreign, ForeignPlane, LaneRef, MaskOp, Operand, Out, Planes,
    Program, Terminal, Value,
};
use ndarray::simd::mask_scatter_or_u32;
use std::collections::BTreeSet;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const U: usize = 65_536;
const GW: usize = U / 64;
const R: usize = 64;
const POP: usize = R * SLOTS;
const S0: Operand = Operand::Scratch(0);

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fisher-Yates over 0..65536, seeded.
fn permutation(seed: u64) -> Vec<u16> {
    let mut p: Vec<u16> = (0..U as u32).map(|x| x as u16).collect();
    for i in (1..U).rev() {
        let j = (mix(seed ^ i as u64) % (i as u64 + 1)) as usize;
        p.swap(i, j);
    }
    p
}

struct Fx {
    name: &'static str,
    maps: Vec<Arc<SealedMap>>,
    /// The 64 local selections, register r at words [r*64, r*64+64).
    local: Vec<u64>,
}

fn source() -> SourceId {
    SourceId { dataset: 7, branch: 0, version: 3, index_seal: 0xD256 }
}

fn map_of(r: usize, ord: Vec<u16>) -> Arc<SealedMap> {
    let ord: Box<[u16; SLOTS]> = ord.into_boxed_slice().try_into().expect("4096 slots");
    Arc::new(SealedMap::new(MapId(r as u64), source().index_seal, ord))
}

/// Every register an independent shuffled injection into the 65,536 universe.
fn chaotic_maps(seed: u64) -> Vec<Arc<SealedMap>> {
    (0..R).map(|r| map_of(r, permutation(seed ^ (r as u64) << 20)[..SLOTS].to_vec())).collect()
}

fn set(local: &mut [u64], r: usize, s: usize) {
    local[r * TILE_WORDS + s / 64] |= 1 << (s % 64);
}

/// Select exactly the given global addresses: each address is assigned to a
/// register and written into a random free slot of that register's map.
fn place(name: &'static str, targets: &[u16], seed: u64) -> Fx {
    let mut ords: Vec<Vec<u16>> =
        (0..R).map(|r| permutation(seed ^ 0xA5A5 ^ (r as u64) << 20)[..SLOTS].to_vec()).collect();
    let mut local = vec![0u64; R * TILE_WORDS];
    let per = targets.len().div_ceil(R);
    for (r, chunk) in targets.chunks(per.max(1)).enumerate() {
        let slots = permutation(seed ^ 0x51 ^ (r as u64) << 24);
        for (&g, &s) in chunk.iter().zip(slots.iter().filter(|&&s| (s as usize) < SLOTS)) {
            ords[r][s as usize] = g;
            set(&mut local, r, s as usize);
        }
    }
    Fx { name, maps: ords.into_iter().enumerate().map(|(r, o)| map_of(r, o)).collect(), local }
}

// ------------------------------------------------------------------ witness

/// Bind every register through the existing `BoundMap::bind` (seal and range),
/// then require ONE source and ONE scope across all 64 before any union: a
/// union of addresses from two sources or scopes is not one global set.
fn bind_all(
    maps: &[Arc<SealedMap>],
    sources: &[SourceId],
    scopes: &[Scope],
    universe: usize,
) -> Result<Vec<BoundMap>, Refusal> {
    let mut out = Vec::with_capacity(maps.len());
    for ((m, &src), &sc) in maps.iter().zip(sources).zip(scopes) {
        if src != sources[0] {
            return Err(Refusal::DomainMismatch { map: src, set: sources[0] });
        }
        if sc != scopes[0] {
            return Err(Refusal::ScopeMismatch { map: sc, set: scopes[0] });
        }
        out.push(BoundMap::bind(m.clone(), src, sc, universe)?);
    }
    Ok(out)
}

/// The u32 index lane the scatter/gather primitives take: the u16 ordinals,
/// widened once (a disposable derived lane, 1 MiB for 64 registers).
fn widen(maps: &[Arc<SealedMap>], lane: &mut [u32]) {
    for (r, m) in maps.iter().enumerate() {
        for (d, &o) in lane[r * SLOTS..(r + 1) * SLOTS].iter_mut().zip(m.ord.iter()) {
            *d = u32::from(o);
        }
    }
}

// ------------------------------------------------------------------ arms

fn arm_a(fx: &Fx) -> Vec<u64> {
    let mut s = BTreeSet::new();
    for (r, m) in fx.maps.iter().enumerate() {
        for slot in 0..SLOTS {
            if fx.local[r * TILE_WORDS + slot / 64] >> (slot % 64) & 1 == 1 {
                s.insert(m.ord[slot]);
            }
        }
    }
    let mut g = vec![0u64; GW];
    for a in s {
        g[a as usize / 64] |= 1 << (a % 64);
    }
    g
}

fn arm_b1(fx: &Fx, lane: &[u32], g: &mut [u64]) {
    g.fill(0);
    for r in 0..R {
        mask_scatter_or_u32(
            &fx.local[r * TILE_WORDS..(r + 1) * TILE_WORDS],
            &lane[r * SLOTS..(r + 1) * SLOTS],
            g,
            U,
        );
    }
}

fn scatter_program() -> Program {
    Program::new(vec![], Terminal::ScatterOrU32 { mask: Operand::Plane(0), lane: 0, out_rows: U as u32 })
}

fn arm_b2(p: &Program, fx: &Fx, lane: &[u32], buf: &mut [u64], g: &mut [u64]) -> Value {
    let masks: [&[u64]; 1] = [&fx.local];
    let lanes = [LaneRef::U32(lane)];
    let planes = Planes { n_rows: POP, masks: &masks, lanes: &lanes };
    let mut sc = Scratch::over(buf, words_for(POP), p.scratch_slots as usize).expect("scratch");
    execute_into(p, &planes, &Foreign::NONE, &mut sc, Out::Mask(g)).expect("scatter runs")
}

/// h = TERNLOG(X, Y, Z) over three global planes, elected with `Keep`.
fn h_program(imm: u8) -> Program {
    Program::new(
        vec![MaskOp::Ternlog {
            imm,
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            c: Operand::Plane(2),
            dst: 0,
        }],
        Terminal::Keep { mask: S0 },
    )
}

fn arm_h(p: &Program, xyz: &[Vec<u64>; 3], buf: &mut [u64]) -> Vec<u64> {
    let masks: [&[u64]; 3] = [&xyz[0], &xyz[1], &xyz[2]];
    let planes = Planes { n_rows: U, masks: &masks, lanes: &[] };
    let mut sc = Scratch::over(buf, words_for(U), p.scratch_slots as usize).expect("scratch");
    execute_into(p, &planes, &Foreign::NONE, &mut sc, Out::None).expect("h runs");
    sc.slot(0).expect("kept").to_vec()
}

/// `G AND h`: gather h through each map under L, then scatter. h is a foreign
/// plane over the global universe; G itself is never read.
fn and_program() -> Program {
    Program::new(
        vec![MaskOp::Gather { lane: 0, foreign: 0, under: Some(Operand::Plane(0)), dst: 0 }],
        Terminal::ScatterOrU32 { mask: S0, lane: 0, out_rows: U as u32 },
    )
}

fn arm_c(p: &Program, fx: &Fx, lane: &[u32], h: &[u64], buf: &mut [u64], g: &mut [u64]) {
    let masks: [&[u64]; 1] = [&fx.local];
    let lanes = [LaneRef::U32(lane)];
    let planes = Planes { n_rows: POP, masks: &masks, lanes: &lanes };
    let fp = [ForeignPlane { words: h, rows: U }];
    let foreign = Foreign { planes: &fp, lanes: &[] };
    let mut sc = Scratch::over(buf, words_for(POP), p.scratch_slots as usize).expect("scratch");
    execute_into(p, &planes, &foreign, &mut sc, Out::Mask(g)).expect("and runs");
}

// ------------------------------------------------------------------ helpers

fn pop(w: &[u64]) -> u32 {
    w.iter().map(|x| x.count_ones()).sum()
}

fn time(mut f: impl FnMut()) -> f64 {
    let mut med = Vec::new();
    for _ in 0..31 {
        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        med.push(t.elapsed().as_nanos() as f64 / reps as f64);
    }
    med.sort_by(|a, b| a.partial_cmp(b).unwrap());
    med[med.len() / 2]
}

fn plane(seed: u64, density_per_256: u64) -> Vec<u64> {
    let mut w = vec![0u64; GW];
    for g in 0..U {
        if mix(seed ^ g as u64) % 256 < density_per_256 {
            w[g / 64] |= 1 << (g % 64);
        }
    }
    w
}

fn main() {
    println!(
        "tier: {}",
        if cfg!(target_feature = "avx512f") { "avx512f" } else { "below v4: timings not comparable to v4 runs" }
    );
    let universe = U;
    let scope = Scope(1);
    let src = source();

    // ---------------------------------------------------------- fixtures
    let mut fixtures = Vec::new();
    // chaotic: 64 shuffled injections, 50 % local selection each
    {
        let maps = chaotic_maps(0xC4A0);
        let mut local = vec![0u64; R * TILE_WORDS];
        for (i, w) in local.iter_mut().enumerate() {
            *w = mix(0x50 ^ i as u64);
        }
        fixtures.push(Fx { name: "chaotic-50%", maps, local });
    }
    // clustered 64x64 block of the 256x256 grid: x 64..128, y 128..192
    let block: Vec<u16> = (128..192u32)
        .flat_map(|y| (64..128u32).map(move |x| (y * 256 + x) as u16))
        .collect();
    fixtures.push(place("clustered-64x64", &block, 0xB10C));
    // scattered: 4,096 distinct random addresses
    let scattered: Vec<u16> = permutation(0x5CA7)[..4096].to_vec();
    fixtures.push(place("scattered-4096", &scattered, 0x5CA7));
    // empty
    fixtures.push(Fx { name: "empty", maps: chaotic_maps(0xE), local: vec![0; R * TILE_WORDS] });
    // full: registers 0..16 partition the universe; the rest are shuffled
    {
        let mut maps = chaotic_maps(0xF);
        for (r, m) in maps.iter_mut().enumerate().take(16) {
            *m = map_of(r, (r * SLOTS..(r + 1) * SLOTS).map(|g| g as u16).collect());
        }
        fixtures.push(Fx { name: "full", maps, local: vec![u64::MAX; R * TILE_WORDS] });
    }

    let mut lane = vec![0u32; POP];
    let mut g_b1 = vec![0u64; GW];
    let mut g_b2 = vec![0u64; GW];
    let p_scatter = scatter_program();
    let mut buf_pop =
        vec![0u64; scratch_words_for(words_for(POP), (p_scatter.scratch_slots as usize).max(1)).unwrap()];

    println!("csv,fixture,selected_slots,global_bits,nonzero_words,parity_b1,parity_b2,bind_ns,widen_ns,oracle_ns,b1_ns,b2_ns");
    for fx in &fixtures {
        let sources = vec![src; R];
        let scopes = vec![scope; R];
        bind_all(&fx.maps, &sources, &scopes, universe).expect("binds");
        widen(&fx.maps, &mut lane);
        let want = arm_a(fx);
        arm_b1(fx, &lane, &mut g_b1);
        let v = arm_b2(&p_scatter, fx, &lane, &mut buf_pop, &mut g_b2);
        assert!(matches!(v, Value::Scattered));
        let (ok1, ok2) = (g_b1 == want, g_b2 == want);
        assert!(ok1 && ok2, "{}: parity", fx.name);

        let t_bind = time(|| {
            black_box(bind_all(&fx.maps, &sources, &scopes, universe).unwrap());
        });
        let t_widen = time(|| widen(&fx.maps, black_box(&mut lane)));
        let t_a = time(|| {
            black_box(arm_a(fx));
        });
        let t_b1 = time(|| arm_b1(fx, &lane, black_box(&mut g_b1)));
        let t_b2 = time(|| {
            black_box(arm_b2(&p_scatter, fx, &lane, &mut buf_pop, &mut g_b2));
        });
        println!(
            "csv,{},{},{},{},{},{},{:.0},{:.0},{:.0},{:.0},{:.0}",
            fx.name,
            pop(&fx.local),
            pop(&want),
            want.iter().filter(|&&w| w != 0).count(),
            ok1,
            ok2,
            t_bind,
            t_widen,
            t_a,
            t_b1,
            t_b2
        );
    }

    // ---------------------------------------------------------- composition C
    let imm = ternlog_imm(|x, y, z| x && (y || !z));
    let xyz = [plane(0x11, 128), plane(0x22, 96), plane(0x33, 160)];
    let p_h = h_program(imm);
    let mut buf_u = vec![0u64; scratch_words_for(words_for(U), p_h.scratch_slots as usize).unwrap()];
    let p_and = and_program();
    let mut buf_and =
        vec![0u64; scratch_words_for(words_for(POP), p_and.scratch_slots as usize).unwrap()];
    println!("csv_c,fixture,h_bits,g_and_h_bits,parity,h_ns,gather_scatter_ns");
    for fx in &fixtures {
        widen(&fx.maps, &mut lane);
        let h = arm_h(&p_h, &xyz, &mut buf_u);
        let want_h: Vec<u64> =
            (0..GW).map(|i| xyz[0][i] & (xyz[1][i] | !xyz[2][i])).collect();
        assert_eq!(h, want_h, "h = TERNLOG(X, Y, Z)");
        let g = arm_a(fx);
        let want: Vec<u64> = g.iter().zip(&h).map(|(a, b)| a & b).collect();
        let mut got = vec![0u64; GW];
        arm_c(&p_and, fx, &lane, &h, &mut buf_and, &mut got);
        let ok = got == want;
        assert!(ok, "{}: G AND h parity", fx.name);
        let t_h = time(|| {
            black_box(arm_h(&p_h, &xyz, &mut buf_u));
        });
        let t_c = time(|| arm_c(&p_and, fx, &lane, &h, &mut buf_and, black_box(&mut got)));
        println!("csv_c,{},{},{},{},{:.0},{:.0}", fx.name, pop(&h), pop(&want), ok, t_h, t_c);
    }

    controls(&src, scope);
}

/// The membership and refusal cases (§3 of the brief), each asserted.
fn controls(src: &SourceId, scope: Scope) {
    let s = *src;
    let id = |r: usize| map_of(r, (0..SLOTS).map(|g| g as u16).collect());

    // 1. identity: one register, identity map -> G is L zero-extended.
    let mut local = vec![0u64; R * TILE_WORDS];
    for w in 0..TILE_WORDS {
        local[w] = mix(0x1D ^ w as u64);
    }
    let fx = Fx { name: "identity", maps: (0..R).map(id).collect(), local: local.clone() };
    let g = arm_a(&fx);
    assert_eq!(&g[..TILE_WORDS], &local[..TILE_WORDS]);
    assert!(g[TILE_WORDS..].iter().all(|&w| w == 0));

    // 3 + 8. collisions / same entity: two registers select slots that both
    // resolve to 777 -> exactly one bit, and the union is not a count.
    let mut o3: Vec<u16> = (0..SLOTS).map(|g| g as u16).collect();
    let mut o9: Vec<u16> = (0..SLOTS).map(|g| (g + 8192) as u16).collect();
    o3[10] = 777;
    o9[2000] = 777;
    let mut maps: Vec<Arc<SealedMap>> = (0..R).map(id).collect();
    maps[3] = map_of(3, o3);
    maps[9] = map_of(9, o9);
    let mut local = vec![0u64; R * TILE_WORDS];
    set(&mut local, 3, 10);
    set(&mut local, 9, 2000);
    let fx = Fx { name: "same-entity", maps, local };
    let g = arm_a(&fx);
    assert_eq!(pop(&g), 1, "two local coordinates of one entity are one member");
    assert!(pop(&fx.local) > pop(&g), "Boolean union collapsed multiplicity 2 -> 1 (not a count)");

    // 9. equal slot numbers, different entities: slot 5 of registers 0 and 1.
    let mut o0: Vec<u16> = (0..SLOTS).map(|g| g as u16).collect();
    let mut o1: Vec<u16> = (0..SLOTS).map(|g| (g + 4096) as u16).collect();
    o0[5] = 100;
    o1[5] = 200;
    let mut maps: Vec<Arc<SealedMap>> = (0..R).map(id).collect();
    maps[0] = map_of(0, o0);
    maps[1] = map_of(1, o1);
    let mut local = vec![0u64; R * TILE_WORDS];
    set(&mut local, 0, 5);
    set(&mut local, 1, 5);
    let fx = Fx { name: "same-slot", maps, local };
    let g = arm_a(&fx);
    assert_eq!(pop(&g), 2, "same slot number, two entities: no false merge");
    let slot_keyed = (fx.local[0] | fx.local[TILE_WORDS]).count_ones();
    assert_eq!(slot_keyed, 1, "a slot-number union would have merged them");

    // 7. refusals, before any union.
    let maps: Vec<Arc<SealedMap>> = (0..R).map(id).collect();
    let scopes = vec![scope; R];
    let mut stale = maps.clone();
    stale[17] = Arc::new(SealedMap::new(MapId(17), s.index_seal ^ 1, maps[17].ord.clone()));
    assert!(matches!(bind_all(&stale, &vec![s; R], &scopes, U), Err(Refusal::StaleMap { .. })));
    let mut wide = maps.clone();
    wide[40] = map_of(40, (0..SLOTS).map(|g| (g + 61_440) as u16).collect());
    assert!(matches!(
        bind_all(&wide, &vec![s; R], &scopes, 61_440),
        Err(Refusal::OrdinalOutOfRange { .. })
    ));
    let mut versions = vec![s; R];
    versions[5].version += 1;
    assert!(matches!(bind_all(&maps, &versions, &scopes, U), Err(Refusal::DomainMismatch { .. })));
    let mut sc = scopes.clone();
    sc[63] = Scope(2);
    assert!(matches!(bind_all(&maps, &vec![s; R], &sc, U), Err(Refusal::ScopeMismatch { .. })));
    println!("controls: identity, same-entity, same-slot, stale seal, out-of-universe, version, scope -> all hold");
}
