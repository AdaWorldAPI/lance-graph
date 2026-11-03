//! D-GLOBAL-WHERE P1, Part 1: falsifiers F1..F21 over working sets P (4 DR256
//! tiles, 1,000 ids) and Q (2 tiles, 400 ids).
//!
//! `--disable dupcheck|framecheck|bindguard` compiles the named guard out of the
//! arm-C index / the binding guard; the named check must then report FAIL.
//! Exit 1 on any failed check. Lab only; nothing here is production.
//!
//! Fixture lanes: the fixture keeps `val: id -> (a, b)` (the SOURCE, oracle
//! side). Lane arrays the folds read are built at preparation by
//! `WorkingSet::gather_lane` in coordinate order; for a set built from a list
//! the coordinate is the list position, so a lane is the source lane in list
//! order. Folds never look an identity up.

use std::collections::{HashMap, HashSet};

use d_global_where_p1_probe::*;
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Frame, FrameRefusal, LocateRefusal, MapRead, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, MethodBinding, MethodId, MethodRef, Refusal, SchemaRev, SigRev, ValueDemand,
};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const A_LT: i32 = 300;
const B_LT: i32 = 500;
const NOVAL: i32 = 1000;

struct Tally {
    pass: u32,
    fail: u32,
}
impl Tally {
    fn check(&mut self, id: &str, ok: bool, detail: String) {
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
        println!("{id} {} {detail}", if ok { "PASS" } else { "FAIL" });
    }
}

struct Fx {
    p: Vec<u64>,
    q: Vec<u64>,
    all: HashSet<u64>,
    val: HashMap<u64, (i32, i32)>,
    edges: Vec<(u64, u64)>,
}
impl Fx {
    fn a(&self, g: u64) -> i32 {
        self.val.get(&g).map_or(NOVAL, |v| v.0)
    }
    fn b(&self, g: u64) -> i32 {
        self.val.get(&g).map_or(NOVAL, |v| v.1)
    }
}

fn fixture() -> Fx {
    let mut rng = Rng::new(SEED);
    let mut ids: Vec<u64> = Vec::new();
    let mut all = HashSet::new();
    while ids.len() < 1400 {
        let g = rng.below(1 << 32);
        if all.insert(g) {
            ids.push(g);
        }
    }
    let p = ids[..1000].to_vec();
    let q = ids[1000..].to_vec();
    let mut val = HashMap::new();
    for &g in &ids {
        val.insert(g, (rng.below(1000) as i32, rng.below(1000) as i32));
    }
    let mut edges = Vec::new();
    for (i, &s) in ids.iter().enumerate() {
        let deg = rng.below(9) as usize; // mean 4
        let mut seen: HashSet<u64> = HashSet::new();
        let mut tries = 0;
        while seen.len() < deg && tries < 100 {
            tries += 1;
            let d = if i < 1000 && rng.unit() >= 0.08 {
                p[rng.below(1000) as usize]
            } else {
                q[rng.below(400) as usize]
            };
            if d != s && seen.insert(d) {
                edges.push((s, d));
            }
        }
    }
    Fx { p, q, all, val, edges }
}

fn sets(fx: &Fx) -> (WorkingSet, WorkingSet) {
    (
        WorkingSet::build(&fx.p, 1).expect("P builds"),
        WorkingSet::build(&fx.q, 2).expect("Q builds"),
    )
}

fn bind(fx: &Fx, s: &[&WorkingSet], arm: Arm, a_lt: i32, b_lt: i32) -> Binding {
    Binding::prepare(s, &fx.edges, arm, &|g| fx.a(g), &|g| fx.b(g), a_lt, b_lt).expect("bind")
}

fn orc(fx: &Fx, a_lt: i32, b_lt: i32, del: &[(u64, u64)]) -> Oracle {
    oracle(&fx.edges, &|g| fx.a(g) < a_lt, &|g| fx.b(g) < b_lt, del)
}

fn scratch(b: &Binding) -> FoldScratch {
    let sizes: Vec<usize> = b.graphs.iter().map(|g| g.n).collect();
    FoldScratch::new(&sizes)
}

/// Every route of every workload; returns the CSR-route answers and the list
/// of internal route disagreements.
fn folds(b: &Binding) -> (Oracle, Vec<String>) {
    let mut sc = scratch(b);
    let (g, a, bm) = (&b.graphs, &b.a, &b.b);
    let mut bad = Vec::new();
    let w2d = w2(g, a, bm);
    let w2a = w2_and(g, a, bm, &mut sc);
    let w2x = w2_xor(g, a, bm, &mut sc);
    if w2a != w2d || w2x != w2d {
        bad.push(format!("W2 direct {w2d} and {w2a} xor {w2x}"));
    }
    let w3c = w3_csr(g, a, bm);
    let w3f = w3_front(g, a, bm, &mut sc);
    if w3c != w3f {
        bad.push(format!("W3 csr {w3c} front {w3f}"));
    }
    let w4c = w4_csr(g, a, bm, &mut sc);
    let w4f = w4_front(g, a, bm, &mut sc);
    let w4t = TileMatrix::build(g).map(|mut t| {
        t.load(a, bm);
        t.fold()
    });
    if w4c != w4f || w4t.is_some_and(|t| t != w4c) {
        bad.push(format!("W4 csr {w4c} front {w4f} tile {w4t:?}"));
    }
    let w5v = w5(g, a, bm, &mut sc);
    let w6v = w6(g, a, None).expect("no overflow");
    let w6bv = w6(g, a, Some(bm)).expect("no overflow");
    (
        Oracle {
            w2: w2d,
            w3: w3c,
            w4: w4c,
            w5: w5v,
            w6: w6v,
            w6b: w6bv,
        },
        bad,
    )
}

fn lookups_of(ws: &WorkingSet, f: &Frame) -> u64 {
    ws.tile_ids
        .iter()
        .filter_map(|&t| ws.table.map(f, t).ok())
        .map(|m| m.lookups())
        .sum()
}

fn global_of(ws: &WorkingSet, f: &Frame, c: usize) -> u64 {
    let tw = ws.tw();
    ws.table
        .map(f, ws.tile_ids[c / tw])
        .expect("tile")
        .global((c % tw) as u8)
        .expect("local in range")
}

fn refs(v: &[WorkingSet]) -> Vec<&WorkingSet> {
    v.iter().collect()
}

fn fresh_ids(fx: &Fx, n: usize, salt: u64) -> Vec<u64> {
    let mut rng = Rng::new(SEED ^ salt);
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    while out.len() < n {
        let g = rng.below(1 << 32);
        if !fx.all.contains(&g) && seen.insert(g) {
            out.push(g);
        }
    }
    out
}

fn method(id: u32) -> MethodBinding {
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("m{id}"),
        on: WideFieldMask::from_positions(&[0]),
        when: None,
        reads: WideFieldMask::from_positions(&[0]),
        effects: vec![],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::MapOnly,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let disable: Option<String> = args.iter().position(|a| a == "--disable").and_then(|i| args.get(i + 1).cloned());
    let dup_off = disable.as_deref() == Some("dupcheck");
    let frame_off = disable.as_deref() == Some("framecheck");
    let guard_off = disable.as_deref() == Some("bindguard");
    if let Some(d) = &disable {
        println!("# --disable {d}: the named guard is compiled out; its check must FAIL");
    }
    let cflags = CFlags {
        disable_dupcheck: dup_off,
        disable_framecheck: frame_off,
    };
    let fx = fixture();
    let mut t = Tally { pass: 0, fail: 0 };
    let p_edges: Vec<(u64, u64)> = {
        let ps: HashSet<u64> = fx.p.iter().copied().collect();
        fx.edges.iter().copied().filter(|(s, _)| ps.contains(s)).collect()
    };
    let cross = fx.edges.iter().filter(|(s, d)| fx.p.contains(s) && fx.q.contains(d)).count();
    println!(
        "# fixture: P={} ids Q={} ids edges={} P-source edges={} P->Q edges={}",
        fx.p.len(),
        fx.q.len(),
        fx.edges.len(),
        p_edges.len(),
        cross
    );

    // ---- F1: G->L->G and L->G->L parity, all three arms --------------------
    {
        let (p, q) = sets(&fx);
        let mut bad = 0usize;
        let mut n = 0usize;
        for (ws, ids) in [(&p, &fx.p), (&q, &fx.q)] {
            let f = ws.seal();
            let tw = ws.tw();
            let idx = CoordIndex::build_with(ws, &f, cflags).expect("C builds");
            let mut oa = vec![0u32; ids.len()];
            let mut ob = vec![0u32; ids.len()];
            let mut oc = vec![0u32; ids.len()];
            route_batch_a_in(&ws.table, &f, &ws.tile_ids, tw, ids, &mut oa).expect("A");
            route_batch_b_in(&ws.table, &f, &ws.tile_ids, tw, ids, &mut ob).expect("B");
            route_batch_c(&idx, &f, ids, &mut oc).expect("C");
            for (i, &g) in ids.iter().enumerate() {
                n += 1;
                for out in [&oa, &ob, &oc] {
                    let c = out[i] as usize;
                    // G -> L -> G
                    if out[i] >= COORD_AMBIGUOUS || global_of(ws, &f, c) != g {
                        bad += 1;
                    }
                }
            }
            // L -> G -> L over every occupied coordinate
            for c in 0..ids.len() {
                let g = global_of(ws, &f, c);
                for out in [
                    ws.coord_a(&f, g).map(|v| v as u32).unwrap_or(u32::MAX),
                    idx.lookup(&f, g).unwrap_or(u32::MAX),
                ] {
                    if out as usize != c {
                        bad += 1;
                    }
                }
            }
        }
        t.check("F1", bad == 0, format!("{n} ids x 3 arms G->L->G and L->G->L, mismatches={bad}"));
    }

    // ---- F2: arms A, B, C identical for every edge endpoint of P ------------
    {
        let (p, _q) = sets(&fx);
        let f = p.seal();
        let ep: Vec<u64> = p_edges.iter().flat_map(|&(s, d)| [s, d]).collect();
        let idx = CoordIndex::build_with(&p, &f, cflags).expect("C");
        let (mut oa, mut ob, mut oc) = (vec![0u32; ep.len()], vec![0u32; ep.len()], vec![0u32; ep.len()]);
        route_batch_a_in(&p.table, &f, &p.tile_ids, p.tw(), &ep, &mut oa).expect("A");
        route_batch_b_in(&p.table, &f, &p.tile_ids, p.tw(), &ep, &mut ob).expect("B");
        route_batch_c(&idx, &f, &ep, &mut oc).expect("C");
        let unmapped = oa.iter().filter(|&&c| c == COORD_UNMAPPED).count();
        t.check(
            "F2",
            oa == ob && oa == oc && unmapped > 0,
            format!(
                "{} endpoints: A==B {} A==C {}; {} endpoints are Q ids (Unmapped in P, same refusal in all arms)",
                ep.len(),
                oa == ob,
                oa == oc,
                unmapped
            ),
        );
    }

    // ---- F3: locate calls (counted twice: by the routine and by the maps) ---
    {
        let (p, _q) = sets(&fx);
        let f = p.seal();
        let tiles = p.tile_ids.len() as u64;
        let ep: Vec<u64> = p_edges.iter().flat_map(|&(s, d)| [s, d]).collect();
        let mut out = vec![0u32; ep.len()];
        let l0 = lookups_of(&p, &f);
        let sa = route_batch_a_in(&p.table, &f, &p.tile_ids, p.tw(), &ep, &mut out).expect("A");
        let l1 = lookups_of(&p, &f);
        let sb = route_batch_b_in(&p.table, &f, &p.tile_ids, p.tw(), &ep, &mut out).expect("B");
        let l2 = lookups_of(&p, &f);
        let idx = CoordIndex::build_with(&p, &f, cflags).expect("C");
        let l3 = lookups_of(&p, &f);
        let sc = route_batch_c(&idx, &f, &ep, &mut out).expect("C");
        let l4 = lookups_of(&p, &f);
        let distinct = ep.iter().copied().collect::<HashSet<_>>().len() as u64;
        let e2 = ep.len() as u64;
        let ok = sa.locate_calls == e2
            && l1 - l0 == e2 * tiles
            && sb.locate_calls == distinct
            && sb.distinct == distinct
            && l2 - l1 == distinct * tiles
            && l3 == l2
            && sc.locate_calls == 0
            && l4 == l3;
        t.check(
            "F3",
            ok,
            format!(
                "A calls={} (2*edges={e2}, map lookups {}) B calls={} (distinct={distinct}, map lookups {}) C query calls={} (map lookups {}, build lookups {})",
                sa.locate_calls,
                l1 - l0,
                sb.locate_calls,
                l2 - l1,
                sc.locate_calls,
                l4 - l3,
                l3 - l2
            ),
        );
    }

    // ---- F4: id in two tiles: Ambiguous in all arms --------------------------
    {
        let mut tiles: Vec<Vec<u64>> = fx.p.chunks(256).map(|c| c.to_vec()).collect();
        let dup = fx.p[0];
        tiles[3].push(dup);
        let ws = WorkingSet::from_tiles(lance_graph_dt64::Width::W256, tiles, 7).expect("dup set");
        let f = ws.seal();
        let a = ws.coord_a(&f, dup);
        let mut ob = [0u32];
        route_batch_b_in(&ws.table, &f, &ws.tile_ids, ws.tw(), &[dup], &mut ob).expect("B");
        let idx = CoordIndex::build_with(&ws, &f, cflags).expect("C");
        let c = idx.lookup(&f, dup);
        let other = fx.p[5];
        let fine = ws.coord_a(&f, other).ok() == idx.lookup(&f, other).ok().map(|v| v as usize);
        t.check(
            "F4",
            a == Err(LocateRefusal::Ambiguous) && ob[0] == COORD_AMBIGUOUS && c == Err(CRefusal::Ambiguous) && fine,
            format!("A={a:?} B={} C={c:?} (an unrelated id still agrees: {fine})", ob[0]),
        );
    }

    // ---- F5: id in no tile: Unmapped everywhere --------------------------------
    {
        let (p, _q) = sets(&fx);
        let f = p.seal();
        let g = u64::MAX - 5;
        let idx = CoordIndex::build_with(&p, &f, cflags).expect("C");
        let mut ob = [0u32];
        route_batch_b_in(&p.table, &f, &p.tile_ids, p.tw(), &[g], &mut ob).expect("B");
        let (a, c) = (p.coord_a(&f, g), idx.lookup(&f, g));
        t.check(
            "F5",
            a == Err(LocateRefusal::Unmapped) && ob[0] == COORD_UNMAPPED && c == Err(CRefusal::Unmapped),
            format!("A={a:?} B={} C={c:?}", ob[0]),
        );
    }

    // ---- F6: foreign frame --------------------------------------------------
    {
        let (p, q) = sets(&fx);
        let fp = p.seal();
        let fq = q.seal();
        let idx = CoordIndex::build_with(&p, &fp, cflags).expect("C");
        let a = p.coord_a(&fq, fx.p[0]);
        let c = idx.lookup(&fq, fx.p[0]);
        t.check(
            "F6",
            a == Err(LocateRefusal::Read(MapRead::Stale(FrameRefusal::ForeignFrame))) && c == Err(CRefusal::StaleIndex),
            format!("A={a:?} C={c:?}"),
        );
    }

    // ---- F7: historical frame -----------------------------------------------
    {
        let (mut p, _q) = sets(&fx);
        let f1 = p.seal();
        let tile0: Vec<u64> = fx.p[..256].to_vec();
        let old: Vec<usize> = tile0.iter().map(|&g| p.coord_a(&f1, g).expect("old")).collect();
        let c1 = CoordIndex::build_with(&p, &f1, cflags).expect("C1");
        let mut rev = tile0.clone();
        rev.reverse();
        p.replace_tile(0, &rev).expect("replace");
        let f2 = p.seal();
        let a_old = tile0.iter().map(|&g| p.coord_a(&f1, g).ok()).collect::<Vec<_>>()
            == old.iter().map(|&c| Some(c)).collect::<Vec<_>>();
        let c_old = tile0.iter().map(|&g| c1.lookup(&f1, g).ok().map(|v| v as usize)).collect::<Vec<_>>()
            == old.iter().map(|&c| Some(c)).collect::<Vec<_>>();
        let c_stale = c1.lookup(&f2, tile0[0]);
        let c2 = CoordIndex::build_with(&p, &f2, cflags).expect("C2");
        let mut same = true;
        for &g in &fx.p {
            same &= p.coord_a(&f2, g).ok().map(|v| v as u32) == c2.lookup(&f2, g).ok();
        }
        let moved = tile0.iter().zip(&old).filter(|&(&g, &o)| p.coord_a(&f2, g).ok() != Some(o)).count();
        t.check(
            "F7",
            a_old && c_old && c_stale == Err(CRefusal::StaleIndex) && same && moved > 0,
            format!(
                "under f1: A old {a_old} C old {c_old}; C(f1) asked with f2 -> {c_stale:?}; C rebuilt==A under f2 {same}; {moved} ids moved"
            ),
        );
    }

    // ---- F8: extension keeps coordinates (generations, not digests) ---------
    {
        let (mut p, q) = sets(&fx);
        let fo = p.seal();
        let old_tiles = p.tile_ids.clone();
        let old_coords: Vec<usize> = fx.p.iter().map(|&g| p.coord_a(&fo, g).expect("old")).collect();
        let bo = bind(&fx, &[&p, &q], Arm::A, A_LT, B_LT);
        let w2_old = w2(&bo.graphs, &bo.a, &bo.b);
        let gen3 = p.table.generation_count(3);
        let ev3 = p.table.event_count(3);
        let vold_digest = p.table.map(&fo, 3).expect("t3").digest();
        let added = fresh_ids(&fx, 40, 0xE8);
        let touched = p.extend(&added).expect("extend");
        let fnw = p.seal();
        let (vo, vn) = (p.table.map(&fo, 3).expect("t3 old"), p.table.map(&fnw, 3).expect("t3 new"));
        let ext = vn.extends(vo);
        let digest_changed = vn.digest() != vold_digest;
        let gen_same = p.table.generation_count(3) == gen3 && p.table.event_count(3) == ev3 + 1;
        let kept = fx.p.iter().zip(&old_coords).all(|(&g, &c)| {
            p.coord_a(&fnw, g).ok() == Some(c) && coord_a_in(&p.table, &fo, &old_tiles, p.tw(), g).ok() == Some(c)
        });
        let bn = bind(&fx, &[&p, &q], Arm::A, A_LT, B_LT);
        let mut restricted = bn.a.clone();
        ndarray::simd::mask_and_assign(&mut restricted[0], &p.valid_mask(&fo));
        let w2_new = w2(&bn.graphs, &restricted, &bn.b);
        let stale = bo.guard(&[&p, &q], true).is_err();
        t.check(
            "F8",
            ext && digest_changed && gen_same && kept && w2_old == w2_new && stale && touched == 2,
            format!(
                "tiles changed={touched} extends={ext} digest changed={digest_changed} (a digest guard would refuse) generations same={gen_same} old coords kept={kept} W2 old={w2_old} new-restricted={w2_new} old binding is stale to the frame guard={stale}"
            ),
        );
    }

    // ---- F9: replacement invalidates the binding -----------------------------
    {
        let (mut p, q) = sets(&fx);
        let f1 = p.seal();
        let c1 = CoordIndex::build_with(&p, &f1, cflags).expect("C1");
        let b1 = bind(&fx, &[&p, &q], Arm::A, A_LT, B_LT);
        let mut rev: Vec<u64> = fx.p[..256].to_vec();
        rev.reverse();
        p.replace_tile(0, &rev).expect("replace");
        let b2 = bind(&fx, &[&p, &q], Arm::A, A_LT, B_LT);
        let truth = orc(&fx, A_LT, B_LT, &[]).w2;
        let fresh = w2(&b2.graphs, &b2.a, &b2.b);
        let mut stale = b1.clone();
        stale.a = b2.a.clone();
        stale.b = b2.b.clone();
        let wrong = w2(&stale.graphs, &stale.a, &stale.b);
        let guarded = b1.guard(&[&p, &q], !guard_off);
        let f2 = p.seal();
        let c_stale = c1.lookup(&f2, fx.p[0]);
        t.check(
            "F9",
            fresh == truth
                && wrong != truth
                && matches!(guarded, Err(BindError::Stale(0)))
                && c_stale == Err(CRefusal::StaleIndex),
            format!(
                "oracle W2={truth} fresh f2 binding={fresh} reused f1 binding without the check={wrong} (WRONG: {}) guarded -> {guarded:?} C(f1) with f2 -> {c_stale:?}",
                wrong != truth
            ),
        );
    }

    // ---- F10: tile removal -----------------------------------------------------
    {
        let (mut p, _q) = sets(&fx);
        let f1 = p.seal();
        let old_tiles = p.tile_ids.clone();
        let g = fx.p[512]; // tile 2
        let under_old = coord_a_in(&p.table, &f1, &old_tiles, p.tw(), g);
        p.remove_tile(2);
        let f2 = p.seal();
        let new_tiles: Vec<u32> = vec![0, 1, 3];
        let under_new = coord_a_in(&p.table, &f2, &new_tiles, p.tw(), g);
        let under_old_list = coord_a_in(&p.table, &f2, &old_tiles, p.tw(), g);
        let again = coord_a_in(&p.table, &f1, &old_tiles, p.tw(), g);
        let ci = CoordIndex::build_in(&p.table, &new_tiles, p.tw(), &f2, cflags).expect("C new");
        let c_new = ci.lookup(&f2, g);
        t.check(
            "F10",
            under_old == Ok(512) && under_new == Err(LocateRefusal::Unmapped) && again == Ok(512) && c_new == Err(CRefusal::Unmapped),
            format!(
                "old frame {under_old:?}; new frame with the reduced tile list {under_new:?} (C {c_new:?}); new frame with the old list {under_old_list:?}; old frame again {again:?}"
            ),
        );
    }

    // ---- F11: presence-only update ----------------------------------------------
    {
        let (p, q) = sets(&fx);
        let mut b = bind(&fx, &[&p, &q], Arm::C, A_LT, B_LT);
        let idx: Vec<CoordIndex> = [&p, &q]
            .iter()
            .zip(&b.frames)
            .map(|(ws, f)| CoordIndex::build_with(ws, f, cflags).expect("C"))
            .collect();
        let frames = b.frames.clone();
        let (base, bad0) = folds(&b);
        // an effective deletion: removes an edge of an A source whose removal moves W2
        let mut eff: Option<(u64, u64)> = None;
        for &(s, d) in &fx.edges {
            if fx.a(s) >= A_LT {
                continue;
            }
            let (Some(sn), Some(dn)) = (route_global_c(&idx, &frames, s), route_global_c(&idx, &frames, d)) else {
                continue;
            };
            let mut g2 = b.graphs.clone();
            if g2[nr_set(sn)].delete(nr_coord(sn), tgt_pack(nr_set(sn), dn)) && w2(&g2, &b.a, &b.b) != base.w2 {
                eff = Some((s, d));
                break;
            }
        }
        let (s1, d1) = eff.expect("an effective deletion exists");
        let un = fx
            .edges
            .iter()
            .copied()
            .find(|&(s, d)| fx.a(s) >= A_LT && (s, d) != (s1, d1) && fx.p.contains(&s))
            .expect("unrelated edge");
        let before_lookups = lookups_of(&p, &frames[0]) + lookups_of(&q, &frames[1]);
        let (sn, dn) = (route_global_c(&idx, &frames, s1).unwrap(), route_global_c(&idx, &frames, d1).unwrap());
        let deleted1 = b.delete(nr_set(sn), nr_coord(sn), dn);
        let (after1, bad1) = folds(&b);
        let (sn2, dn2) = (route_global_c(&idx, &frames, un.0).unwrap(), route_global_c(&idx, &frames, un.1).unwrap());
        let deleted2 = b.delete(nr_set(sn2), nr_coord(sn2), dn2);
        let (after2, bad2) = folds(&b);
        let after_lookups = lookups_of(&p, &frames[0]) + lookups_of(&q, &frames[1]);
        let o1 = orc(&fx, A_LT, B_LT, &[(s1, d1)]);
        let o2 = orc(&fx, A_LT, B_LT, &[(s1, d1), un]);
        let current = b.guard(&[&p, &q], true).is_ok();
        let ok = deleted1
            && deleted2
            && bad0.is_empty()
            && bad1.is_empty()
            && bad2.is_empty()
            && after1 == o1
            && after2 == o2
            && after1.w2 < base.w2
            && after2.w2 == after1.w2
            && after_lookups == before_lookups
            && current;
        t.check(
            "F11",
            ok,
            format!(
                "W2 {} -> {} (oracle {}) after the effective deletion; unrelated deletion leaves W2 {} (all workloads == oracle: {}); map lookups during folds+updates: {}; frame still current {current}",
                base.w2,
                after1.w2,
                o1.w2,
                after2.w2,
                after2 == o2,
                after_lookups - before_lookups
            ),
        );
    }

    // ---- F12: cross-working-set edge ----------------------------------------------
    {
        let (p, q) = sets(&fx);
        let b = bind(&fx, &[&p, &q], Arm::A, A_LT, B_LT);
        let truth = orc(&fx, A_LT, B_LT, &[]).w2;
        let right = w2(&b.graphs, &b.a, &b.b);
        // Restricted to cross-set edges, where the decoding is what is under test:
        // x in A (source set) with a cross edge to a B member. `right_cross` decodes
        // the target through its own set; `wrong` reads the coordinate as if it
        // belonged to the source set (coordinate coincidence).
        let (mut right_cross, mut wrong) = (0i64, 0i64);
        for (si, c) in b.graphs.iter().enumerate() {
            for_each_bit(&b.a[si], |x| {
                let (mut r, mut w) = (false, false);
                for e in c.off[x] as usize..c.off[x + 1] as usize {
                    if bit(&c.valid, e) && c.tgt[e] & CROSS != 0 {
                        let n = tgt_unpack(si, c.tgt[e]);
                        r |= bit(&b.b[nr_set(n)], nr_coord(n));
                        w |= bit(&b.b[si], nr_coord(n));
                    }
                }
                right_cross += i64::from(r);
                wrong += i64::from(w);
            });
        }
        let qset: HashSet<u64> = fx.q.iter().copied().collect();
        let pset: HashSet<u64> = fx.p.iter().copied().collect();
        let truth_cross = fx
            .p
            .iter()
            .filter(|&&x| {
                fx.a(x) < A_LT
                    && fx.edges.iter().any(|&(s, d)| s == x && qset.contains(&d) && !pset.contains(&d) && fx.b(d) < B_LT)
            })
            .count() as i64;
        let cross_edges: usize = b.graphs[0].tgt.iter().filter(|&&x| x & CROSS != 0).count();
        t.check(
            "F12",
            cross_edges > 0 && right == truth && right_cross == truth_cross && truth_cross > 0 && wrong != truth_cross,
            format!(
                "{cross_edges} P->Q edges, B held in Q coordinates: W2={right} oracle={truth}; over cross edges only: DR-decoded {right_cross} oracle {truth_cross}, coordinate coincidence {wrong} (WRONG)"
            ),
        );
    }

    // ---- F13: same id, different coordinates ------------------------------------------
    {
        let n = fx.p.len();
        let mut rot = fx.p.clone();
        rot.rotate_left(7);
        let w1 = WorkingSet::build(&fx.p, 11).expect("w1");
        let w2s = WorkingSet::build(&rot, 12).expect("w2");
        let (f1, f2) = (w1.seal(), w2s.seal());
        let g = fx.p[10];
        let (c1, c2) = (w1.coord_a(&f1, g).expect("c1"), w2s.coord_a(&f2, g).expect("c2"));
        let back = global_of(&w1, &f1, c1) == g && global_of(&w2s, &f2, c2) == g;
        let (v1, v2) = (w1.valid_mask(&f1), w2s.valid_mask(&f2));
        let (l1, l2) = (
            w1.gather_lane(&f1, |g| fx.a(g)),
            w2s.gather_lane(&f2, |g| fx.a(g)),
        );
        let (mut m1, mut m2) = (vec![0u64; words_for(w1.n_coords())], vec![0u64; words_for(w2s.n_coords())]);
        virtual_where(&l1, A_LT, &v1, &mut m1);
        virtual_where(&l2, A_LT, &v2, &mut m2);
        let mut proj = vec![0u64; m2.len()];
        for_each_bit(&m1, |c| {
            let gg = global_of(&w1, &f1, c);
            set_bit(&mut proj, w2s.coord_a(&f2, gg).expect("in w2"));
        });
        t.check(
            "F13",
            c1 != c2 && back && proj == m2 && m1 != m2,
            format!(
                "id {g:#x} is coordinate {c1} in one set and {c2} in the other (both resolve back); A selection: {n} ids, DR-aligned equality {}; bitwise coincidence equality {} (WRONG)",
                proj == m2,
                m1 == m2
            ),
        );
    }

    // ---- F14: reused binding == freshly prepared binding, Q = 64 -----------------------
    {
        let (p, q) = sets(&fx);
        let mut reused = bind(&fx, &[&p, &q], Arm::C, A_LT, 100);
        let mut bad = 0usize;
        for k in 0..64 {
            let tb = 100 + 10 * k;
            reused.rewhere_b(tb);
            let fresh = bind(&fx, &[&p, &q], Arm::C, A_LT, tb);
            let same_struct = reused.graphs == fresh.graphs
                && reused.a == fresh.a
                && reused.b == fresh.b
                && reused.valid == fresh.valid;
            let (ra, _) = folds(&reused);
            let (fa, _) = folds(&fresh);
            if !same_struct || ra != fa || ra != orc(&fx, A_LT, tb, &[]) {
                bad += 1;
            }
        }
        t.check("F14", bad == 0, format!("64 reuses (B parameter changed each time, only the WHERE re-run) vs 64 fresh bindings: {bad} differ"));
    }

    // ---- F15: EXO-46 shared selection -----------------------------------------------------
    {
        const CLASS: u32 = 0x7F0B_0001;
        const OTHER: u32 = 0x7F0B_0002;
        let schema = ClassSchema {
            schema_rev: SchemaRev(1),
            field_count: 2,
            extent: Extent::DR64_CUBE,
        };
        let mut store = BindStore::new();
        store.register(CLASS, schema, vec![method(1), method(2)]).expect("class");
        store.register(OTHER, schema, vec![method(1)]).expect("other class");
        let (m1, m2, mo) = (method(1), method(2), method(1));
        let (r1, r2, ro) = (MethodRef::of(CLASS, &m1), MethodRef::of(CLASS, &m2), MethodRef::of(OTHER, &mo));
        let (permit1, permit2) = (vec![r1.clone()], vec![r2.clone()]);
        let (mut p, _q) = sets(&fx);
        let mut b = bind(&fx, &[&p], Arm::A, A_LT, B_LT);
        let scope = Scope { id: 1 };
        let sel = SharedSelection {
            class: CLASS,
            frame: p.seal(),
            scope,
            criteria: A_LT as u64,
            mask: b.a[0].clone(),
        };
        let cur = p.seal();
        // a
        let ra = sel.admit(&store, &r1, &permit1, &cur, scope);
        let rb = sel.admit(&store, &r2, &permit2, &cur, scope);
        let a_ok = match (ra, rb) {
            (Ok(x), Ok(y)) => std::ptr::eq(x.as_ptr(), y.as_ptr()) && w2_borrowed(&b.graphs, x, &b.b) == w2_borrowed(&b.graphs, y, &b.b),
            _ => false,
        };
        // b, c, d, e
        let b_ref = sel.admit(&store, &r2, &permit1, &cur, scope);
        let c_unbound = sel.admit(
            &store,
            &MethodRef {
                class: 0x7F0B_00FF,
                ..r1.clone()
            },
            &[],
            &cur,
            scope,
        );
        let c_other = sel.admit(&store, &ro, &[ro.clone()], &cur, scope);
        let d_sig = sel.admit(
            &store,
            &MethodRef {
                sig_rev: SigRev(2),
                ..r1.clone()
            },
            &permit1,
            &cur,
            scope,
        );
        let e_scope = sel.admit(&store, &r1, &permit1, &cur, Scope { id: 2 });
        // g: presence update does not change the mask; the fold reflects it
        let mask_before = sel.mask.clone();
        let w2_before = w2_borrowed(&b.graphs, &sel.mask, &b.b);
        let mut deleted = false;
        'outer: for x in 0..b.graphs[0].n {
            if !bit(&sel.mask, x) {
                continue;
            }
            for e in b.graphs[0].off[x] as usize..b.graphs[0].off[x + 1] as usize {
                let mut g2 = b.graphs.clone();
                clear_bit(&mut g2[0].valid, e);
                if w2_borrowed(&g2, &sel.mask, &b.b) != w2_before {
                    let dst = tgt_unpack(0, b.graphs[0].tgt[e]);
                    deleted = b.delete(0, x, dst);
                    break 'outer;
                }
            }
        }
        let w2_after = w2_borrowed(&b.graphs, &sel.mask, &b.b);
        let g_ok = deleted && sel.mask == mask_before && w2_after != w2_before;
        // i: zero allocation in admit + fold
        let (a0, c0) = (alloc_bytes(), alloc_calls());
        let r = sel.admit(&store, &r1, &permit1, &cur, scope);
        let v = match r {
            Ok(m) => w2_borrowed(&b.graphs, m, &b.b),
            Err(_) => -1,
        };
        let (a1, c1) = (alloc_bytes(), alloc_calls());
        std::hint::black_box(v);
        // f: stale frame after an advance
        let mut rev: Vec<u64> = fx.p[..256].to_vec();
        rev.reverse();
        p.replace_tile(0, &rev).expect("replace");
        let cur2 = p.seal();
        let f_stale = sel.admit(&store, &r1, &permit1, &cur2, scope);
        t.check("F15a", a_ok, "M1 and M2, each with its own permit, admit the SAME selection (same mask address, same fold answer)".into());
        t.check("F15b", b_ref == Err(ShareRefusal::NotPermitted), format!("M2 with only M1's permit -> {b_ref:?}"));
        t.check(
            "F15c",
            matches!(c_unbound, Err(ShareRefusal::Bind(Refusal::UnknownClass(_))))
                && matches!(c_other, Err(ShareRefusal::ClassMismatch { .. })),
            format!("unbound class -> {c_unbound:?}; other bound class -> {c_other:?}"),
        );
        t.check(
            "F15d",
            matches!(d_sig, Err(ShareRefusal::Bind(Refusal::SignatureRevisionMismatch { .. }))),
            format!("stale sig rev -> {d_sig:?}"),
        );
        t.check("F15e", e_scope == Err(ShareRefusal::Scope), format!("another scope -> {e_scope:?}"));
        t.check("F15f", f_stale == Err(ShareRefusal::StaleFrame), format!("stale frame after advance -> {f_stale:?}"));
        t.check(
            "F15g",
            g_ok,
            format!("mask bit-identical before/after the presence update; fold {w2_before} -> {w2_after}"),
        );
        t.check(
            "F15h",
            std::ptr::eq(sel.admit(&store, &r1, &permit1, &cur, scope).map(|m| m.as_ptr()).unwrap_or(std::ptr::null()), sel.mask.as_ptr()),
            "admit returns a borrow of the selection's own mask (&[u64] tied to &self): no mask is passed or copied between programs".into(),
        );
        t.check("F15i", a1 == a0 && c1 == c0, format!("admit + fold allocated {} bytes in {} calls", a1 - a0, c1 - c0));
    }

    // ---- F16: AND/XOR/FOLD parity with the oracle -----------------------------------------
    {
        let (p, q) = sets(&fx);
        let mut bad: Vec<String> = Vec::new();
        let mut n = 0;
        for arm in Arm::ALL {
            for &(al, bl) in &[(300, 500), (100, 200), (700, 900)] {
                let b = bind(&fx, &[&p, &q], arm, al, bl);
                let (got, routes) = folds(&b);
                let want = orc(&fx, al, bl, &[]);
                n += 1;
                if got != want || !routes.is_empty() {
                    bad.push(format!("arm {} a<{al} b<{bl}: got {got:?} want {want:?} routes {routes:?}", arm.name()));
                }
            }
        }
        let o = orc(&fx, 300, 500, &[]);
        t.check(
            "F16",
            bad.is_empty(),
            format!(
                "{n} (arm x params) runs, W2 direct/AND/XOR, W3 csr/front, W4 csr/front/tile, W5, W6 (plain and B) == oracle; sample W2={} W3={} W4={} W5={} W6={} W6b={}{}",
                o.w2,
                o.w3,
                o.w4,
                o.w5,
                o.w6,
                o.w6b,
                bad.first().map(|s| format!(" FIRST BAD: {s}")).unwrap_or_default()
            ),
        );
    }

    // ---- F17: folds allocate 0 bytes ------------------------------------------------------
    {
        let (p, q) = sets(&fx);
        let b = bind(&fx, &[&p, &q], Arm::A, A_LT, B_LT);
        let mut sc = scratch(&b);
        let mut tm = TileMatrix::build(&b.graphs).expect("<= 4096");
        let mut names = Vec::new();
        let mut run = |name: &'static str, f: &mut dyn FnMut() -> i64| {
            let (a0, c0) = (alloc_bytes(), alloc_calls());
            let v = f();
            let (a1, c1) = (alloc_bytes(), alloc_calls());
            std::hint::black_box(v);
            if a1 != a0 || c1 != c0 {
                names.push(format!("{name}:{}B/{}calls", a1 - a0, c1 - c0));
            }
        };
        let (g, a, bm) = (&b.graphs, &b.a, &b.b);
        run("w2", &mut || w2(g, a, bm));
        run("w2_and", &mut || w2_and(g, a, bm, &mut sc));
        run("w2_xor", &mut || w2_xor(g, a, bm, &mut sc));
        run("w3_csr", &mut || w3_csr(g, a, bm));
        run("w3_front", &mut || w3_front(g, a, bm, &mut sc));
        run("w4_csr", &mut || w4_csr(g, a, bm, &mut sc));
        run("w4_front", &mut || w4_front(g, a, bm, &mut sc));
        run("w4_tile", &mut || {
            tm.load(a, bm);
            tm.fold()
        });
        run("w5", &mut || w5(g, a, bm, &mut sc));
        run("w6", &mut || w6(g, a, Some(bm)).unwrap_or(-1));
        t.check(
            "F17",
            names.is_empty(),
            format!("W2/W3/W4(csr,front,tile)/W5/W6 folds: allocations {names:?}; no tuple vector (folds read masks, offsets and targets only; signatures carry no Vec of tuples)"),
        );
    }

    // ---- F18: the duplicate check is load-bearing --------------------------------------------
    {
        let mut tiles: Vec<Vec<u64>> = fx.p.chunks(256).map(|c| c.to_vec()).collect();
        let dup = fx.p[0];
        tiles[3].push(dup);
        let ws = WorkingSet::from_tiles(lance_graph_dt64::Width::W256, tiles, 8).expect("dup set");
        let f = ws.seal();
        let on = CoordIndex::build_with(&ws, &f, CFlags::default()).expect("on").lookup(&f, dup);
        let off = CoordIndex::build_with(
            &ws,
            &f,
            CFlags {
                disable_dupcheck: true,
                ..CFlags::default()
            },
        )
        .expect("off")
        .lookup(&f, dup);
        t.check(
            "F18",
            on == Err(CRefusal::Ambiguous) && off.is_ok(),
            format!("duplicate check on -> {on:?}; compiled out -> {off:?} (a silent wrong coordinate, so F4 fails under --disable dupcheck)"),
        );
    }

    // ---- F19: the frame check is load-bearing ---------------------------------------------------
    {
        let (mut p, _q) = sets(&fx);
        let f1 = p.seal();
        let on = CoordIndex::build_with(&p, &f1, CFlags::default()).expect("on");
        let off = CoordIndex::build_with(
            &p,
            &f1,
            CFlags {
                disable_framecheck: true,
                ..CFlags::default()
            },
        )
        .expect("off");
        let mut rev: Vec<u64> = fx.p[..256].to_vec();
        rev.reverse();
        p.replace_tile(0, &rev).expect("replace");
        let f2 = p.seal();
        let g = fx.p[0];
        let (r_on, r_off, truth) = (on.lookup(&f2, g), off.lookup(&f2, g), p.coord_a(&f2, g).ok().map(|v| v as u32));
        t.check(
            "F19",
            r_on == Err(CRefusal::StaleIndex) && r_off.is_ok() && r_off.ok() != truth,
            format!("frame check on -> {r_on:?}; compiled out -> {r_off:?} but the truth under f2 is {truth:?} (so F6/F7/F9 fail under --disable framecheck)"),
        );
    }

    // ---- F20: the binding guard is load-bearing ------------------------------------------------------
    {
        let (mut p, q) = sets(&fx);
        let b1 = bind(&fx, &[&p, &q], Arm::A, A_LT, B_LT);
        let mut rev: Vec<u64> = fx.p[..256].to_vec();
        rev.reverse();
        p.replace_tile(0, &rev).expect("replace");
        let b2 = bind(&fx, &[&p, &q], Arm::A, A_LT, B_LT);
        let truth = orc(&fx, A_LT, B_LT, &[]).w2;
        let on = b1.guard(&[&p, &q], true);
        let off = b1.guard(&[&p, &q], false);
        let mut stale = b1.clone();
        stale.a = b2.a.clone();
        stale.b = b2.b.clone();
        let silent = w2(&stale.graphs, &stale.a, &stale.b);
        t.check(
            "F20",
            matches!(on, Err(BindError::Stale(0))) && off.is_ok() && silent != truth,
            format!("guard on -> {on:?}; compiled out -> {off:?} and the reused binding answers {silent} where the truth is {truth} (silent wrong answer, so F9 fails under --disable bindguard)"),
        );
    }

    // ---- F21: an A-side selection change extends the set --------------------------------------------------
    {
        let q = WorkingSet::build(&fx.q, 2).expect("Q");
        let old_ids = &fx.p[..800];
        let added = &fx.p[800..];
        let mut ext = WorkingSet::build(old_ids, 21).expect("old");
        let fo = ext.seal();
        let old_tiles = ext.tile_ids.clone();
        let old_coords: Vec<usize> = old_ids.iter().map(|&g| ext.coord_a(&fo, g).expect("old")).collect();
        let t0 = std::time::Instant::now();
        let touched = ext.extend(added).expect("extend");
        let t_ext = t0.elapsed().as_nanos() as f64 / 1000.0;
        let fn_ = ext.seal();
        let mut full = WorkingSet::build(&fx.p, 22).expect("full");
        let t1 = std::time::Instant::now();
        full = {
            drop(full);
            WorkingSet::build(&fx.p, 22).expect("rebuild")
        };
        let t_full = t1.elapsed().as_nanos() as f64 / 1000.0;
        let ff = full.seal();
        let kept = old_ids
            .iter()
            .zip(&old_coords)
            .all(|(&g, &c)| ext.coord_a(&fn_, g).ok() == Some(c) && coord_a_in(&ext.table, &fo, &old_tiles, ext.tw(), g).ok() == Some(c));
        let all_same = fx.p.iter().all(|&g| ext.coord_a(&fn_, g).ok() == full.coord_a(&ff, g).ok());
        let extends = ext.table.map(&fn_, 3).expect("t3").extends(ext.table.map(&fo, 3).expect("t3 old"));
        let be = bind(&fx, &[&ext, &q], Arm::C, A_LT, B_LT);
        let bf = bind(&fx, &[&full, &q], Arm::C, A_LT, B_LT);
        let (ae, rb_e) = folds(&be);
        let (af, rb_f) = folds(&bf);
        let want = orc(&fx, A_LT, B_LT, &[]);
        let same_graph = be.graphs == bf.graphs && be.a == bf.a && be.b == bf.b;
        t.check(
            "F21",
            kept && all_same && extends && same_graph && ae == af && ae == want && rb_e.is_empty() && rb_f.is_empty() && touched == 1,
            format!(
                "extend by {} ids touched {touched} tile(s) via MapTable::advance; TileMap::extends={extends}; old coordinates kept={kept}; every coordinate == full rebuild {all_same}; binding structure equal {same_graph}; all workloads equal rebuild and oracle {}; extend {} vs full rebuild {} (preparation: reported, not gate)",
                added.len(),
                ae == af && ae == want,
                fmt_us(t_ext, true),
                fmt_us(t_full, true),
            ),
        );
    }

    // ======================= multi-domain composition (F22..F29) =======================
    // A domain is a working set; D domains over one corpus resolve cross-domain
    // edges through the global id (arm routing under each domain's own frame).
    let all: Vec<u64> = fx.p.iter().chain(fx.q.iter()).copied().collect();
    let split = |ids: &[u64], d: usize, base: u64| -> Vec<WorkingSet> {
        let per = ids.len().div_ceil(d);
        ids.chunks(per)
            .enumerate()
            .map(|(i, c)| WorkingSet::build(c, base + i as u64).expect("domain"))
            .collect()
    };
    let restrict = |a: &[Vec<u64>], k: usize| -> Vec<Vec<u64>> {
        a.iter()
            .enumerate()
            .map(|(i, m)| if i == k { m.clone() } else { vec![0u64; m.len()] })
            .collect()
    };
    let want_all = orc(&fx, A_LT, B_LT, &[]);

    // ---- F22: one / two / three domains == one domain over the union ----------------
    {
        let mut bad = Vec::new();
        let mut line = Vec::new();
        for (d, arm) in [(1usize, Arm::A), (2, Arm::B), (3, Arm::C)] {
            let doms = split(&all, d, 100 * d as u64);
            let b = bind(&fx, &refs(&doms), arm, A_LT, B_LT);
            let (got, routes) = folds(&b);
            line.push(format!("D={d} arm {} W2={} W5={} W6={}", arm.name(), got.w2, got.w5, got.w6));
            if got != want_all || !routes.is_empty() || b.stats.dropped_edges != 0 {
                bad.push(format!("D={d}: {got:?} dropped {}", b.stats.dropped_edges));
            }
        }
        t.check(
            "F22",
            bad.is_empty(),
            format!("EXISTS/COUNT/DISTINCT/Bag == one domain == oracle for 1, 2, 3 domains ({}); {bad:?}", line.join("; ")),
        );
    }

    // ---- F23: an edge crossing a domain boundary --------------------------------------
    {
        let doms = split(&all, 2, 200); // P | Q
        let cross_edges: Vec<(u64, u64)> = fx.edges.iter().copied().filter(|(s, d)| fx.p.contains(s) && fx.q.contains(d)).collect();
        let b = Binding::prepare(&refs(&doms), &cross_edges, Arm::C, &|g| fx.a(g), &|g| fx.b(g), A_LT, B_LT).expect("bind");
        let (got, routes) = folds(&b);
        let want = oracle(&cross_edges, &|g| fx.a(g) < A_LT, &|g| fx.b(g) < B_LT, &[]);
        // decode one cross edge by hand: target is (domain 1, the coordinate Q gave the id)
        let (x, y) = cross_edges[0];
        let fq = doms[1].seal();
        let xc = doms[0].coord_a(&doms[0].seal(), x).expect("x");
        let want_n = noderef(1, doms[1].coord_a(&fq, y).expect("y"));
        let found = (b.graphs[0].off[xc] as usize..b.graphs[0].off[xc + 1] as usize)
            .any(|e| b.graphs[0].tgt[e] & CROSS != 0 && tgt_unpack(0, b.graphs[0].tgt[e]) == want_n);
        t.check(
            "F23",
            got == want && routes.is_empty() && found && b.stats.dropped_edges == 0 && !cross_edges.is_empty(),
            format!("{} cross-domain edges only: all workloads == oracle {}; edge {x:#x}->{y:#x} stored as CROSS | (domain 1, coord {})", cross_edges.len(), got == want, nr_coord(want_n)),
        );
    }

    // ---- F24: a path crossing three domains ----------------------------------------------
    {
        let doms = split(&all, 3, 300);
        let (d0, d1, d2) = (&all[..467], &all[467..934], &all[934..]);
        let x = d0.iter().copied().find(|&g| fx.a(g) < A_LT).expect("x");
        let y = d1[0];
        let z = d2.iter().copied().find(|&g| fx.b(g) < B_LT).expect("z");
        let w = d0.iter().copied().find(|&g| g != x && fx.b(g) < B_LT).expect("w");
        let path = vec![(x, y), (y, z), (z, w)];
        let run = |edges: &[(u64, u64)]| {
            let b = Binding::prepare(&refs(&doms), edges, Arm::B, &|g| fx.a(g), &|g| fx.b(g), A_LT, B_LT).expect("bind");
            let (got, routes) = folds(&b);
            let want = oracle(edges, &|g| fx.a(g) < A_LT, &|g| fx.b(g) < B_LT, &[]);
            (got, want, routes)
        };
        let (g3, w3, r3) = run(&path);
        let (g2, w2o, r2) = run(&path[..2]);
        let (g1, w1o, r1) = run(&path[..1]);
        t.check(
            "F24",
            g3 == w3 && g2 == w2o && g1 == w1o && r3.is_empty() && r2.is_empty() && r1.is_empty() && g3.w4 == 1 && g2.w3 == 1 && g1.w3 == 0,
            format!("x(domain 0)->y(domain 1)->z(domain 2)->w(domain 0): W4 over 3 hops={} (oracle {}), W3 over 2 hops={} and {} with only the first hop; csr, front and tile routes agree", g3.w4, w3.w4, g2.w3, g1.w3),
        );
    }

    // ---- F25: the same global id in two selections -------------------------------------------
    {
        let d2 = split(&all, 2, 400);
        let d3 = split(&all, 3, 500);
        let b2 = bind(&fx, &refs(&d2), Arm::A, A_LT, B_LT);
        let b3 = bind(&fx, &refs(&d3), Arm::A, A_LT, B_LT);
        let both_oracle = all.iter().filter(|&&g| fx.a(g) < A_LT && fx.b(g) < B_LT).count() as i64;
        let mut tmp2: Vec<Vec<u64>> = b2.a.clone();
        let mut tmp3: Vec<Vec<u64>> = b3.a.clone();
        let c2 = and_count(&b2.a, &b2.b, &mut tmp2);
        let c3 = and_count(&b3.a, &b3.b, &mut tmp3);
        let g = all.iter().copied().find(|&g| fx.a(g) < A_LT && fx.b(g) < B_LT).expect("an id in both");
        let (f2, f3) = (d2.iter().map(|s| s.seal()).collect::<Vec<_>>(), d3.iter().map(|s| s.seal()).collect::<Vec<_>>());
        let loc = |doms: &[WorkingSet], fr: &[Frame]| doms.iter().enumerate().find_map(|(i, ws)| ws.coord_a(&fr[i], g).ok().map(|c| (i, c)));
        let (l2, l3) = (loc(&d2, &f2), loc(&d3, &f3));
        let both_bits = |b: &Binding, l: (usize, usize)| bit(&b.a[l.0], l.1) && bit(&b.b[l.0], l.1);
        t.check(
            "F25",
            c2 == both_oracle && c3 == both_oracle && l2.is_some() && l3.is_some() && both_bits(&b2, l2.unwrap()) && both_bits(&b3, l3.unwrap()),
            format!("id {g:#x} is in selection A and selection B: coordinate {l2:?} under 2 domains, {l3:?} under 3; |A AND B| = {c2} = {c3} = oracle {both_oracle}"),
        );
    }

    // ---- F26: a duplicate id across incorrectly configured domains is refused --------------------
    {
        let good = split(&all, 3, 600);
        let ok_union = union_check(&refs(&good));
        let overlapping = vec![
            WorkingSet::build(&all[..700], 610).expect("d1"),
            WorkingSet::build(&all[400..1100], 611).expect("d2"),
        ];
        let bad_union = union_check(&refs(&overlapping));
        // listed together in one table: Ambiguous from locate
        let mut tiles: Vec<Vec<u64>> = all[..510].chunks(255).map(|c| c.to_vec()).collect();
        tiles[1].push(all[3]);
        let together = WorkingSet::from_tiles(lance_graph_dt64::Width::W256, tiles, 612).expect("set");
        let amb = together.coord_a(&together.seal(), all[3]);
        t.check(
            "F26",
            ok_union == Ok(all.len()) && matches!(bad_union, Err(g) if all[400..700].contains(&g)) && amb == Err(LocateRefusal::Ambiguous),
            format!("disjoint 3 domains: union check {ok_union:?}; domains overlapping on 300 ids: {bad_union:?}; duplicate listed in one table -> {amb:?}"),
        );
    }

    // ---- F27: a deleted cross-domain edge ----------------------------------------------------------
    {
        let doms = split(&all, 2, 700);
        let mut b = bind(&fx, &refs(&doms), Arm::C, A_LT, B_LT);
        let idx: Vec<CoordIndex> = doms.iter().zip(&b.frames).map(|(ws, f)| CoordIndex::build_with(ws, f, cflags).expect("C")).collect();
        let frames = b.frames.clone();
        let (base, _) = folds(&b);
        let (q_valid0, p_valid0) = (b.graphs[1].valid_edges(), b.graphs[0].valid_edges());
        let mut pick: Option<(u64, u64)> = None;
        for &(s, d) in &fx.edges {
            if !(fx.p.contains(&s) && fx.q.contains(&d) && fx.a(s) < A_LT) {
                continue;
            }
            let (Some(sn), Some(dn)) = (route_global_c(&idx, &frames, s), route_global_c(&idx, &frames, d)) else { continue };
            let mut g2 = b.graphs.clone();
            if g2[nr_set(sn)].delete(nr_coord(sn), tgt_pack(nr_set(sn), dn)) && w2(&g2, &b.a, &b.b) != base.w2 {
                pick = Some((s, d));
                break;
            }
        }
        let (s, d) = pick.expect("an effective cross-domain deletion");
        let (sn, dn) = (route_global_c(&idx, &frames, s).unwrap(), route_global_c(&idx, &frames, d).unwrap());
        let done = b.delete(nr_set(sn), nr_coord(sn), dn);
        let (after, routes) = folds(&b);
        let want = orc(&fx, A_LT, B_LT, &[(s, d)]);
        t.check(
            "F27",
            done && after == want && routes.is_empty() && after.w2 < base.w2 && b.graphs[1].valid_edges() == q_valid0 && b.graphs[0].valid_edges() == p_valid0 - 1,
            format!("deleted {s:#x}->{d:#x} (domain 0 -> domain 1): W2 {} -> {} (oracle {}); domain 0 lost one edge, domain 1 untouched; frames unchanged", base.w2, after.w2, want.w2),
        );
    }

    // ---- F28: historical replay across domains ---------------------------------------------------------
    {
        let mut doms = split(&all, 2, 800);
        let (f_old0, f_old1) = (doms[0].seal(), doms[1].seal());
        let tiles_old: Vec<Vec<u32>> = doms.iter().map(|d| d.tile_ids.clone()).collect();
        let b1 = bind(&fx, &refs(&doms), Arm::A, A_LT, B_LT);
        for (k, d) in doms.iter_mut().enumerate() {
            let f = d.seal();
            let mut tile0: Vec<u64> = d.table.map(&f, 0).expect("t0").globals().to_vec();
            tile0.reverse();
            d.replace_tile(0, &tile0).expect("replace");
            let _ = k;
        }
        let fr_old = [f_old0, f_old1];
        // every cross target of b1 still decodes to the right global under the OLD frames
        let mut decode_ok = true;
        let mut cross_n = 0usize;
        for (si, c) in b1.graphs.iter().enumerate() {
            for x in 0..c.n {
                for e in c.off[x] as usize..c.off[x + 1] as usize {
                    if c.tgt[e] & CROSS != 0 {
                        cross_n += 1;
                        let n = tgt_unpack(si, c.tgt[e]);
                        let tw = doms[nr_set(n)].tw();
                        let g = doms[nr_set(n)].table.map(&fr_old[nr_set(n)], tiles_old[nr_set(n)][nr_coord(n) / tw]).ok().and_then(|v| v.global((nr_coord(n) % tw) as u8));
                        let src_g = doms[si].table.map(&fr_old[si], tiles_old[si][x / doms[si].tw()]).ok().and_then(|v| v.global((x % doms[si].tw()) as u8));
                        decode_ok &= g.is_some() && src_g.is_some() && fx.edges.contains(&(src_g.unwrap(), g.unwrap()));
                    }
                }
            }
        }
        // under the CURRENT frames the same coordinates mean other ids
        let cur: Vec<Frame> = doms.iter().map(|d| d.seal()).collect();
        let moved = (0..256usize).filter(|&c| doms[0].table.map(&fr_old[0], 0).unwrap().global(c as u8) != doms[0].table.map(&cur[0], 0).unwrap().global(c as u8)).count();
        let stale = b1.guard(&refs(&doms), true);
        let (old_ans, _) = folds(&b1);
        t.check(
            "F28",
            decode_ok && cross_n > 0 && moved > 0 && matches!(stale, Err(BindError::Stale(_))) && old_ans == want_all,
            format!("{cross_n} cross-domain targets of the old binding all decode to the right ids under the old frames of BOTH domains; {moved} coordinates of tile 0 mean other ids under the new frames; the old binding is stale to the guard ({stale:?}) and still answers its own frames (== oracle {})", old_ans == want_all),
        );
    }

    // ---- F29: aggregate composition laws ---------------------------------------------------------------------
    {
        let doms = split(&all, 3, 900);
        let b = bind(&fx, &refs(&doms), Arm::C, A_LT, B_LT);
        let g = &b.graphs;
        let mut sc = scratch(&b);
        let parts: Vec<Vec<Vec<u64>>> = (0..3).map(|k| restrict(&b.a, k)).collect();
        // EXISTS = OR of parts
        let any_full = w2(g, &b.a, &b.b) > 0;
        let any_or = parts.iter().any(|pk| w2(g, pk, &b.b) > 0);
        let empty_b: Vec<Vec<u64>> = b.b.iter().map(|m| vec![0u64; m.len()]).collect();
        let none_or = parts.iter().any(|pk| w2(g, pk, &empty_b) > 0) || w2(g, &b.a, &empty_b) > 0;
        // COUNT = checked sum over DISJOINT contributions
        let w2_parts: Vec<i64> = parts.iter().map(|pk| w2(g, pk, &b.b)).collect();
        let sum_w2 = checked_sum(&w2_parts);
        let full_w2 = w2(g, &b.a, &b.b);
        // DISTINCT = exact union; the per-domain sum is WRONG
        let w5_parts: Vec<i64> = parts.iter().map(|pk| w5(g, pk, &b.b, &mut sc)).collect();
        let sum_w5 = checked_sum(&w5_parts);
        let full_w5 = w5(g, &b.a, &b.b, &mut sc);
        let mut uni: Vec<Vec<u64>> = b.b.iter().map(|m| vec![0u64; m.len()]).collect();
        for pk in &parts {
            step(g, pk, &mut sc.m1);
            for si in 0..uni.len() {
                ndarray::simd::mask_or_assign(&mut uni[si], &sc.m1[si]);
            }
        }
        let exact_w5 = and_count(&uni, &b.b, &mut sc.m2);
        // Bag: multiplicities preserved, checked i64
        let w6_parts: Vec<i64> = parts.iter().map(|pk| w6(g, pk, None).expect("fits")).collect();
        let w6b_parts: Vec<i64> = parts.iter().map(|pk| w6(g, pk, Some(&b.b)).expect("fits")).collect();
        let (sum_w6, sum_w6b) = (checked_sum(&w6_parts), checked_sum(&w6b_parts));
        let overflow = checked_sum(&[i64::MAX, 1]);
        // overlapping domains evaluated independently: sums are WRONG, the exact union is right
        let ov: Vec<WorkingSet> = vec![
            WorkingSet::build(&all[..700], 910).expect("d1"),
            WorkingSet::build(&all[400..1100], 911).expect("d2"),
            WorkingSet::build(&all[800..], 912).expect("d3"),
        ];
        let member: Vec<HashSet<u64>> = [&all[..700], &all[400..1100], &all[800..]].iter().map(|s| s.iter().copied().collect()).collect();
        let e_union: Vec<(u64, u64)> = fx.edges.iter().copied().filter(|(s, d)| member.iter().any(|m| m.contains(s) && m.contains(d))).collect();
        let want_u = oracle(&e_union, &|g| fx.a(g) < A_LT, &|g| fx.b(g) < B_LT, &[]);
        let (mut s2, mut s5, mut s6) = (0i64, 0i64, 0i64);
        let (mut xs, mut ys): (Vec<u64>, Vec<u64>) = (Vec::new(), Vec::new());
        for ws in &ov {
            let bk = Binding::prepare(&[ws], &fx.edges, Arm::C, &|g| fx.a(g), &|g| fx.b(g), A_LT, B_LT).expect("domain binding");
            let mut sck = scratch(&bk);
            let fr = ws.seal();
            s2 += w2(&bk.graphs, &bk.a, &bk.b);
            s5 += w5(&bk.graphs, &bk.a, &bk.b, &mut sck);
            s6 += w6(&bk.graphs, &bk.a, None).expect("fits");
            semijoin(&bk.graphs, &bk.b, &mut sck.m1);
            ndarray::simd::mask_and(&bk.a[0], &sck.m1[0], &mut sck.m2[0]);
            for_each_bit(&sck.m2[0], |c| xs.push(global_of(ws, &fr, c)));
            step(&bk.graphs, &bk.a, &mut sck.m1);
            ndarray::simd::mask_and(&sck.m1[0], &bk.b[0], &mut sck.m2[0]);
            for_each_bit(&sck.m2[0], |c| ys.push(global_of(ws, &fr, c)));
        }
        xs.sort_unstable();
        xs.dedup();
        ys.sort_unstable();
        ys.dedup();
        let ok = any_full == any_or
            && !none_or
            && sum_w2 == Some(full_w2)
            && full_w2 == want_all.w2
            && sum_w5 != Some(full_w5)
            && exact_w5 == full_w5
            && full_w5 == want_all.w5
            && sum_w6 == Some(want_all.w6)
            && sum_w6b == Some(want_all.w6b)
            && overflow.is_none()
            && union_check(&refs(&ov)).is_err()
            && s2 != want_u.w2
            && s5 != want_u.w5
            && s6 != want_u.w6
            && xs.len() as i64 == want_u.w2
            && ys.len() as i64 == want_u.w5;
        t.check(
            "F29",
            ok,
            format!(
                "disjoint domains: EXISTS=OR {any_or}; COUNT sum {sum_w2:?} = {full_w2}; DISTINCT sum of per-domain counts {sum_w5:?} is WRONG, exact union {exact_w5} = {full_w5}; Bag sums {sum_w6:?}/{sum_w6b:?} = oracle {}/{} (i64 checked, overflow detected {}). OVERLAPPING domains evaluated alone: sums COUNT {s2} DISTINCT {s5} BAG {s6} are WRONG vs oracle {}/{}/{}; exact union by global id {}/{} is right",
                want_all.w6,
                want_all.w6b,
                overflow.is_none(),
                want_u.w2,
                want_u.w5,
                want_u.w6,
                xs.len(),
                ys.len()
            ),
        );
    }

    println!("falsify_p1: {} of {} checks failed", t.fail, t.pass + t.fail);
    if t.fail > 0 {
        std::process::exit(1);
    }
}
