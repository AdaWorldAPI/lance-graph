//! LAB: D-GLOBAL-WHERE P0 falsifier (Part 1 of SPEC.md).
//!
//! One global WHERE (Quack `Keep` over 65,536 ids) is routed into DR tiles
//! through `lance_graph_dt64::dr::MapTable`, and folded over a presence plane
//! of a relation. The checks ask where local coordinates stop meaning the same
//! thing as global ids, and whether the guards of `dr-math-binding-probe` fire
//! where they must. Exit status 1 if any check fails.

use std::collections::{HashMap, HashSet};

use d_global_where_probe::{
    alloc_bytes, alloc_calls, bit, coord_of, for_each_bit, gen_lanes, pick_domain, popcount,
    prefix_mask, route_selection, Lanes, Plane, Prepared, Rng, A_LT, B_LT, SEED, UNIVERSE,
    UNIVERSE_WORDS,
};
use d_global_where_probe::CountingAlloc;
use dr_math_binding_probe as drp;
use dr_math_binding_probe::hhtl16::LocalMap as HLocalMap;
use dr_math_binding_probe::where_dr3::{project, Criteria, LocalSelection, ReuseRefusal, Under};
use lance_graph_dt64::dr::{
    Axis as ResAxis, FrameRefusal, LocalMap, LocateRefusal, MapRead, MapTable, RouteVersion,
    TileId, Width,
};
use lance_graph_quack::{Agg, Cmp};
use ndarray::simd::{checked_binary_i64, mask_and, mask_any, mask_or_assign, mask_xor, I64Op};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

/// Wide coordinates: two DR256 tiles of the relation frame R.
const TW: usize = 256;
const COORDS: usize = 512;
const WPR: usize = COORDS / 64;
const R_TILES: [TileId; 2] = [0, 1];

struct Report {
    lines: Vec<(u32, String)>,
    census: Vec<(u32, i64, usize)>,
    fails: u32,
}

impl Report {
    fn check(&mut self, n: u32, ok: bool, detail: String) {
        if !ok {
            self.fails += 1;
        }
        self.lines.push((
            n,
            format!("F{n:<2} {} {detail}", if ok { "PASS" } else { "FAIL" }),
        ));
    }
    fn census(&mut self, n: u32, lookups: i64, bytes: usize) {
        self.census.push((n, lookups, bytes));
    }
}

/// Lookups served so far by the maps of the four fixture tiles, read at the
/// table's current seal (after F10 the replaced tile restarts at zero).
fn look(table: &MapTable) -> u64 {
    let f = table.seal();
    [0u32, 1, 10, 11]
        .iter()
        .map(|&t| table.map(&f, t).map(|m| m.lookups()).unwrap_or(0))
        .sum()
}

/// Count of x in A with a B-neighbour: any(row[x] & selB) for every x in selA.
fn q_fold(plane: &Plane, sel_a: &[u64], sel_b: &[u64], tmp: &mut [u64]) -> i64 {
    let mut n = 0i64;
    for_each_bit(sel_a, |x| {
        mask_and(plane.row(x), sel_b, tmp);
        if mask_any(tmp) {
            n += 1;
        }
    });
    n
}

/// The same count from the edge list (oracle).
fn q_oracle(lanes: &Lanes, edges: &[(u64, u64)]) -> i64 {
    let mut xs: HashSet<u64> = HashSet::new();
    for &(s, d) in edges {
        if lanes.a[s as usize] < A_LT && lanes.b[d as usize] < B_LT {
            xs.insert(s);
        }
    }
    xs.len() as i64
}

fn bind_reg<'a>(
    lst: &drp::FrameStore,
    g: &drp::Guards,
    m: drp::MethodSignature,
    rows: &'a [u64; 64],
) -> drp::BoundDr64<'a> {
    drp::bind_dr64(
        lst,
        m,
        m,
        g,
        (g.focus, g.scope),
        drp::Operand {
            rows,
            validity: None,
        },
    )
    .expect("register binds")
}

#[allow(clippy::too_many_arguments)]
fn project_sel(
    map: &HLocalMap,
    truth: &[u64],
    gst: &drp::FrameStore,
    gseal: &drp::Frame,
    lst: &drp::FrameStore,
    guards: &drp::Guards,
    axis: drp::Axis,
    method: drp::MethodSignature,
    criteria: Criteria,
) -> LocalSelection {
    project(
        map,
        truth,
        Under {
            global_store: gst,
            global: gseal,
            local_store: lst,
            guards,
            axis,
            method,
            criteria,
        },
    )
    .expect("projection")
}

fn main() {
    let mut rep = Report {
        lines: Vec::new(),
        census: Vec::new(),
        fails: 0,
    };

    // ------------------------------------------------------------------
    // fixture
    // ------------------------------------------------------------------
    let mut rng = Rng::new(SEED);
    let mut lanes = gen_lanes(&mut rng);
    let dom = pick_domain(&mut rng, 512);
    // x in A routed through R tile 0, y in B through R tile 1; three ids in
    // A and B both.
    let x = dom[0];
    let y = dom[256];
    let forced_shared = [dom[1], dom[100], dom[400]];
    lanes.a[x as usize] = 0;
    lanes.b[y as usize] = 0;
    for &z in &forced_shared {
        lanes.a[z as usize] = 0;
        lanes.b[z as usize] = 0;
    }
    // Edges over D: out-degree uniform in 0..=12 (average 6), no duplicates,
    // no self loops. x keeps exactly one out-edge, to y.
    let mut edges: Vec<(u64, u64)> = Vec::new();
    for i in 0..512usize {
        let deg = rng.below(13) as usize;
        let mut seen: HashSet<usize> = HashSet::new();
        while seen.len() < deg {
            let j = rng.below(512) as usize;
            if j != i && seen.insert(j) {
                edges.push((dom[i], dom[j]));
            }
        }
    }
    edges.retain(|&(s, _)| s != x);
    edges.push((x, y));

    // WHERE through Quack: Keep masks over the whole universe.
    let live = vec![u64::MAX; UNIVERSE_WORDS];
    let mut pa = Prepared::new(Cmp::LtI32(A_LT), Agg::Rows, UNIVERSE);
    let mut pb = Prepared::new(Cmp::LtI32(B_LT), Agg::Rows, UNIVERSE);
    let mut keep_a = vec![0u64; UNIVERSE_WORDS];
    let mut keep_b = vec![0u64; UNIVERSE_WORDS];
    pa.keep(&lanes.a, &live, &mut keep_a);
    pb.keep(&lanes.b, &live, &mut keep_b);

    // F0 (preamble): the Keep masks equal the scalar lane test at every id.
    {
        let mut bad = 0usize;
        let (mut na, mut nb) = (0usize, 0usize);
        for g in 0..UNIVERSE {
            let (ta, tb) = (lanes.a[g] < A_LT, lanes.b[g] < B_LT);
            na += usize::from(ta);
            nb += usize::from(tb);
            if bit(&keep_a, g) != ta || bit(&keep_b, g) != tb {
                bad += 1;
            }
        }
        let ok = bad == 0
            && popcount(&keep_a) as usize == na
            && popcount(&keep_b) as usize == nb
            && na > 1000
            && nb > 1000;
        rep.check(
            0,
            ok,
            format!("Quack Keep == lane oracle at all {UNIVERSE} ids (|A|={na}, |B|={nb}, mismatches={bad})"),
        );
    }

    // Selection tiles: A descending global order, B ascending.
    let mut a_ids: Vec<u64> = dom
        .iter()
        .copied()
        .filter(|&g| bit(&keep_a, g as usize))
        .collect();
    a_ids.sort_unstable_by(|p, q| q.cmp(p));
    let mut b_ids: Vec<u64> = dom
        .iter()
        .copied()
        .filter(|&g| bit(&keep_b, g as usize))
        .collect();
    b_ids.sort_unstable();
    assert!(a_ids.len() <= 256 && b_ids.len() <= 256, "selection tiles hold <= 256");
    let wa = ResAxis::for_survivors(a_ids.len()).tile;
    let wb = ResAxis::for_survivors(b_ids.len()).tile;

    let mut table = MapTable::new(1, 1);
    table.advance(
        1,
        1,
        vec![
            (0, Some(LocalMap::prepare(Width::W256, &dom[0..256]).expect("tile 0"))),
            (1, Some(LocalMap::prepare(Width::W256, &dom[256..512]).expect("tile 1"))),
            (10, Some(LocalMap::prepare(wa, &a_ids).expect("tile 10"))),
            (11, Some(LocalMap::prepare(wb, &b_ids).expect("tile 11"))),
        ],
    );
    let frame = table.seal();

    let shared: Vec<u64> = a_ids
        .iter()
        .copied()
        .filter(|g| b_ids.binary_search(g).is_ok())
        .collect();
    let y_sh: Option<u64> = {
        let ta = table.map(&frame, 10).expect("tile 10");
        let tb = table.map(&frame, 11).expect("tile 11");
        let found = shared
            .iter()
            .copied()
            .find(|&g| ta.local(g).ok() != tb.local(g).ok());
        found
    };

    // Presence planes (forward, and the column storage of the same edges).
    let mut fwd = Plane::new(COORDS);
    let mut rev = Plane::new(COORDS);
    for &(s, d) in &edges {
        let cx = coord_of(&table, &frame, &R_TILES, TW, s).expect("src in R");
        let cy = coord_of(&table, &frame, &R_TILES, TW, d).expect("dst in R");
        fwd.set(cx, cy);
        rev.set(cy, cx);
    }
    let mut sel_a_r = vec![0u64; WPR];
    let mut sel_b_r = vec![0u64; WPR];
    route_selection(&table, &frame, &R_TILES, TW, &[10], &mut sel_a_r);
    route_selection(&table, &frame, &R_TILES, TW, &[11], &mut sel_b_r);

    // ------------------------------------------------------------------
    // F1, F2: G->L->G and L->G->L, every id of tiles 0, 1, 10, 11
    // ------------------------------------------------------------------
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        let (mut n1, mut bad1, mut n2, mut bad2) = (0usize, 0usize, 0usize, 0usize);
        for t in [0u32, 1, 10, 11] {
            let tm = table.map(&frame, t).expect("tile");
            for &g in tm.globals() {
                n1 += 1;
                match tm.local(g) {
                    Ok(l) if tm.global(l) == Some(g) => {}
                    _ => bad1 += 1,
                }
            }
            for l in 0..tm.len() {
                let l8 = l as u8;
                n2 += 1;
                match tm.global(l8) {
                    Some(g) if tm.local(g) == Ok(l8) => {}
                    _ => bad2 += 1,
                }
            }
        }
        rep.check(1, bad1 == 0 && n1 > 500, format!("global(local(g))==g for {n1} ids of tiles 0,1,10,11"));
        rep.check(2, bad2 == 0 && n2 == n1, format!("local(global(l))==l for {n2} locals"));
        rep.census(1, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
        rep.census(2, 0, 0);
    }

    // ------------------------------------------------------------------
    // F3: the shared id has different locals in tile 10 and tile 11
    // ------------------------------------------------------------------
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        let detail = match y_sh {
            Some(g) => {
                let ta = table.map(&frame, 10).expect("tile 10");
                let tb = table.map(&frame, 11).expect("tile 11");
                format!(
                    "id {g}: local {:?} in tile 10 (A, descending), {:?} in tile 11 (B, ascending); {} ids shared",
                    ta.local(g).ok(),
                    tb.local(g).ok(),
                    shared.len()
                )
            }
            None => "no shared id with differing locals".to_string(),
        };
        rep.check(3, y_sh.is_some() && shared.len() >= 3, detail);
        rep.census(3, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
    }

    // ------------------------------------------------------------------
    // F4: local-coincidence join vs the DR-aligned path
    // ------------------------------------------------------------------
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        let ta = table.map(&frame, 10).expect("tile 10");
        let tb = table.map(&frame, 11).expect("tile 11");
        // Treat the two local position spaces as one domain: AND the tile-10
        // local mask with the tile-11 local mask.
        let ma = prefix_mask(4, ta.len());
        let mb = prefix_mask(4, tb.len());
        let mut both = vec![0u64; 4];
        mask_and(&ma, &mb, &mut both);
        let coincident = popcount(&both) as usize;
        // DR-aligned: B local -> global (tile 11) -> tile 10 local via local().
        let mut aligned = 0usize;
        for l in 0..tb.len() {
            let g = tb.global(l as u8).expect("local in range");
            if ta.local(g).is_ok() {
                aligned += 1;
            }
        }
        let oracle = dom
            .iter()
            .filter(|&&g| lanes.a[g as usize] < A_LT && lanes.b[g as usize] < B_LT)
            .count();
        rep.check(
            4,
            aligned == oracle && coincident != oracle && oracle >= 3,
            format!(
                "oracle |A&B&D|={oracle}, DR-aligned={aligned}, local-coincidence={coincident} (tile lens {} and {})",
                ta.len(),
                tb.len()
            ),
        );
        rep.census(4, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
    }

    // ------------------------------------------------------------------
    // F5, F18: Case B over the R frame; x and y in different R tiles
    // ------------------------------------------------------------------
    let mut tmp = [0u64; WPR];
    let cx = coord_of(&table, &frame, &R_TILES, TW, x).expect("x in R");
    let cy = coord_of(&table, &frame, &R_TILES, TW, y).expect("y in R");
    let q0;
    let (f5_bytes, f5_calls);
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        // The measured fold: row[x] & selB_R != 0.
        let (fb0, fc0) = (alloc_bytes(), alloc_calls());
        mask_and(fwd.row(cx), &sel_b_r, &mut tmp);
        let ans5 = mask_any(&tmp);
        f5_bytes = alloc_bytes() - fb0;
        f5_calls = alloc_calls() - fc0;
        let oracle5 = edges
            .iter()
            .any(|&(s, d)| s == x && lanes.b[d as usize] < B_LT);
        q0 = q_fold(&fwd, &sel_a_r, &sel_b_r, &mut tmp);
        let q0_oracle = q_oracle(&lanes, &edges);
        let (ox, oy) = (cx / TW, cy / TW);
        let ok = ans5
            && oracle5
            && bit(&sel_a_r, cx)
            && bit(&sel_b_r, cy)
            && q0 == q0_oracle
            && q0 > 0;
        rep.check(
            5,
            ok,
            format!(
                "x={x} (R ordinal {ox}, coord {cx}) -> y={y} (R ordinal {oy}, coord {cy}): row[x]&selB_R != 0 is {ans5}, oracle {oracle5}; Q0 (x in A with a B-neighbour) = {q0} == oracle {q0_oracle}"
            ),
        );
        rep.check(
            18,
            ox != oy && ox < 2 && oy < 2,
            format!("R spans 2 DR256 tiles; x in ordinal {ox}, y in ordinal {oy}"),
        );
        rep.census(5, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
        rep.census(18, 0, 0);
    }

    // ------------------------------------------------------------------
    // F6: clear the edge's presence bit; locate still Ok; answer false
    // ------------------------------------------------------------------
    let q1;
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        fwd.clear(cx, cy);
        rev.clear(cy, cx);
        edges.retain(|&e| e != (x, y));
        mask_and(fwd.row(cx), &sel_b_r, &mut tmp);
        let ans6 = mask_any(&tmp);
        let oracle6 = edges
            .iter()
            .any(|&(s, d)| s == x && lanes.b[d as usize] < B_LT);
        let located = table.locate(&frame, &R_TILES, y).is_ok();
        q1 = q_fold(&fwd, &sel_a_r, &sel_b_r, &mut tmp);
        let q1_oracle = q_oracle(&lanes, &edges);
        rep.check(
            6,
            !ans6 && !oracle6 && located && q1 == q1_oracle && q1 == q0 - 1,
            format!(
                "edge ({x},{y}) cleared (one word per plane): locate(y) still Ok={located}, answer now {ans6} (oracle {oracle6}); Q {q0} -> {q1} (oracle {q1_oracle})"
            ),
        );
        rep.census(6, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
    }

    // ------------------------------------------------------------------
    // F7: clear an unrelated edge; the fixed query is unchanged
    // ------------------------------------------------------------------
    let (f7_bytes, f7_calls);
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        let out = |g: u64| lanes.a[g as usize] >= A_LT && lanes.b[g as usize] >= B_LT;
        let (u, v) = edges
            .iter()
            .copied()
            .find(|&(s, d)| out(s) && out(d))
            .expect("an edge outside A and B exists");
        let cu = coord_of(&table, &frame, &R_TILES, TW, u).expect("u in R");
        let cv = coord_of(&table, &frame, &R_TILES, TW, v).expect("v in R");
        fwd.clear(cu, cv);
        rev.clear(cv, cu);
        edges.retain(|&e| e != (u, v));
        let (fb0, fc0) = (alloc_bytes(), alloc_calls());
        let q2 = q_fold(&fwd, &sel_a_r, &sel_b_r, &mut tmp);
        f7_bytes = alloc_bytes() - fb0;
        f7_calls = alloc_calls() - fc0;
        let q2_oracle = q_oracle(&lanes, &edges);
        rep.check(
            7,
            q2 == q1 && q2 == q2_oracle,
            format!("edge ({u},{v}) outside A and B cleared: Q stays {q2} (was {q1}, oracle {q2_oracle})"),
        );
        rep.census(7, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
    }

    // ------------------------------------------------------------------
    // F8, F9: forward plane XOR transpose(reverse plane)
    // ------------------------------------------------------------------
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        let t = rev.transpose();
        let mut xr = vec![0u64; fwd.rows.len()];
        mask_xor(&fwd.rows, &t.rows, &mut xr);
        let set_bits = popcount(&fwd.rows) as usize;
        rep.check(
            8,
            !mask_any(&xr) && set_bits == edges.len() && set_bits > 1000,
            format!("fwd XOR transpose(rev) == 0 over {} words; {set_bits} edges set", xr.len()),
        );
        // Flip one bit of the reverse plane (an existing edge): XOR != 0 and
        // the differing position is exactly that edge.
        let (s, d) = edges[0];
        let cs = coord_of(&table, &frame, &R_TILES, TW, s).expect("s in R");
        let cd = coord_of(&table, &frame, &R_TILES, TW, d).expect("d in R");
        let mut rev2 = rev.clone();
        rev2.clear(cd, cs);
        let t2 = rev2.transpose();
        mask_xor(&fwd.rows, &t2.rows, &mut xr);
        let mut pos: Vec<usize> = Vec::new();
        for_each_bit(&xr, |p| pos.push(p));
        let want = cs * fwd.wpr * 64 + cd;
        rep.check(
            9,
            mask_any(&xr) && pos.len() == 1 && pos[0] == want,
            format!(
                "one reverse bit flipped for edge ({s},{d}) = coords ({cs},{cd}): {} differing position(s), first {:?}, expected {want}",
                pos.len(),
                pos.first()
            ),
        );
        rep.census(8, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
        rep.census(9, 0, 0);
    }

    // ------------------------------------------------------------------
    // F14, F15: allocation around the F5 and F7 folds
    // ------------------------------------------------------------------
    rep.check(
        14,
        f5_bytes == 0 && f7_bytes == 0,
        format!("fold bytes allocated: check-5 fold {f5_bytes}, check-7 fold {f7_bytes}; no tuple vector (folds read Plane rows and two 8-word masks)"),
    );
    rep.check(
        15,
        f5_calls == 0 && f7_calls == 0,
        format!("fold allocation calls: check-5 {f5_calls}, check-7 {f7_calls}; no per-object allocation"),
    );
    rep.census(14, 0, f5_bytes + f7_bytes);
    rep.census(15, 0, 0);

    // ------------------------------------------------------------------
    // F12, F13: guards of dr-math-binding-probe::where_dr3 over a 64-slot map
    // ------------------------------------------------------------------
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        let hmap = HLocalMap::prepare(&dom[0..64]).expect("64-slot map");
        let gst = drp::FrameStore::new(drp::FrameRevisions {
            source: 1,
            relation_map: 0,
            coordinates: 1,
        });
        let lst = drp::FrameStore::new(drp::FrameRevisions {
            source: 1,
            relation_map: 1,
            coordinates: hmap.digest(),
        });
        let gseal = gst.seal();
        let guards = drp::Guards {
            frame: lst.seal(),
            focus: drp::Focus { revision: 1 },
            scope: drp::Scope { id: 7 },
        };
        let criteria = Criteria { id: 0xC1 };
        let class_c: u16 = 21;
        let row_axis = drp::Axis {
            class: class_c,
            kind: drp::AxisKind::Identity,
        };
        let col_axis = drp::Axis {
            class: class_c,
            kind: drp::AxisKind::Property,
        };
        let m1 = drp::MethodSignature::presence(class_c, 1, row_axis, col_axis);
        let m2 = drp::MethodSignature::presence(class_c, 2, row_axis, col_axis);
        let rows = [0u64; 64];

        let sel1 = project_sel(&hmap, &keep_a, &gst, &gseal, &lst, &guards, row_axis, m1, criteria);
        let sel2 = project_sel(&hmap, &keep_a, &gst, &gseal, &lst, &guards, row_axis, m2, criteria);

        // F12: a register of a foreign class on the row axis.
        let foreign_row = drp::Axis {
            class: 22,
            kind: drp::AxisKind::Identity,
        };
        let foreign_col = drp::Axis {
            class: 22,
            kind: drp::AxisKind::Property,
        };
        let mf = drp::MethodSignature::presence(22, 1, foreign_row, foreign_col);
        let reg_f = bind_reg(&lst, &guards, mf, &rows);
        let r12 = sel1.admit(&gst, &lst, &reg_f, criteria);
        rep.check(
            12,
            matches!(r12, Err(ReuseRefusal::AxisMismatch)),
            format!("foreign-class row axis -> {:?}", r12.as_ref().err()),
        );

        // F13a: M1's selection against M2's register.
        let reg1 = bind_reg(&lst, &guards, m1, &rows);
        let reg2 = bind_reg(&lst, &guards, m2, &rows);
        let r13a = sel1.admit(&gst, &lst, &reg2, criteria);
        let twin = sel1.admit(&gst, &lst, &reg1, criteria);
        rep.check(
            13,
            matches!(r13a, Err(ReuseRefusal::SchemaChanged)) && twin.is_ok(),
            format!(
                "13a M1 selection vs M2 register -> {:?}; silence twin (M1 vs M1) ok={}",
                r13a.as_ref().err(),
                twin.is_ok()
            ),
        );

        // F13b: the two projections are bit for bit the same selection.
        let a1 = sel1.admit(&gst, &lst, &reg1, criteria).expect("M1 admits");
        let a2 = sel2.admit(&gst, &lst, &reg2, criteria).expect("M2 admits");
        let same = a1.truth() == a2.truth() && a1.slots() == a2.slots();
        rep.check(
            13,
            same && a1.truth() != 0 && a1.slots() == u64::MAX,
            format!(
                "13b truth M1={:#x} M2={:#x}, slots M1={:#x} M2={:#x} (projection does not depend on the method; truth non-empty)",
                a1.truth(),
                a2.truth(),
                a1.slots(),
                a2.slots()
            ),
        );

        // F13c: a different Scope refuses on its own.
        let guards_s = drp::Guards {
            scope: drp::Scope { id: 8 },
            ..guards.clone()
        };
        let reg_s = bind_reg(&lst, &guards_s, m1, &rows);
        let r13c = sel1.admit(&gst, &lst, &reg_s, criteria);
        rep.check(
            13,
            matches!(r13c, Err(ReuseRefusal::ScopeChanged)),
            format!("13c scope 7 selection vs scope 8 register (same method) -> {:?}", r13c.as_ref().err()),
        );
        rep.census(12, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
        rep.census(13, 0, 0);

        // ------------------------------------------------------------
        // F17: Boolean vs bag on the current edge set
        // ------------------------------------------------------------
        let (l1, b1) = (look(&table), alloc_bytes());
        let mut or_acc = [0u64; WPR];
        for_each_bit(&sel_a_r, |xc| mask_or_assign(&mut or_acc, fwd.row(xc)));
        let distinct = popcount(&or_acc) as i64;
        let mut indeg = vec![0i64; COORDS];
        for_each_bit(&sel_a_r, |xc| {
            for_each_bit(fwd.row(xc), |yc| indeg[yc] += 1);
        });
        let outdeg: Vec<i64> = (0..COORDS).map(|c| popcount(fwd.row(c)) as i64).collect();
        let mut prod = vec![0i64; COORDS];
        checked_binary_i64(I64Op::Mul, &indeg, &outdeg, &mut prod).expect("no overflow");
        let mut bag = 0i64;
        for &p in &prod {
            bag = bag.checked_add(p).expect("no overflow");
        }
        let mut out_count: HashMap<u64, i64> = HashMap::new();
        for &(s, _) in &edges {
            *out_count.entry(s).or_insert(0) += 1;
        }
        let mut reach: HashSet<u64> = HashSet::new();
        let mut bag_o = 0i64;
        for &(s, d) in &edges {
            if lanes.a[s as usize] < A_LT {
                reach.insert(d);
                bag_o += out_count.get(&d).copied().unwrap_or(0);
            }
        }
        let declared = drp::MethodSignature {
            population: drp::Population::Bag,
            ..m1
        };
        let r_bag = drp::bind_dr64(
            &lst,
            declared,
            declared,
            &guards,
            (guards.focus, guards.scope),
            drp::Operand {
                rows: &rows,
                validity: None,
            },
        );
        rep.check(
            17,
            distinct == reach.len() as i64
                && bag == bag_o
                && distinct != bag
                && matches!(r_bag, Err(drp::Refusal::BagNeedsMultiplicity)),
            format!(
                "one-hop distinct from A = {distinct} (oracle {}), 2-path bag from A = {bag} (oracle {bag_o}); Population::Bag -> {:?}",
                reach.len(),
                r_bag.as_ref().err()
            ),
        );
        rep.census(17, look(&table) as i64 - l1 as i64, alloc_bytes() - b1);
    }

    // ------------------------------------------------------------------
    // F11: stale / foreign frame, future version, ambiguous locate
    // ------------------------------------------------------------------
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        let mut t2 = MapTable::new(1, 1);
        t2.advance(
            1,
            1,
            vec![(0, Some(LocalMap::prepare(Width::W256, &dom[0..256]).expect("tile")))],
        );
        let f_other = t2.seal();
        let foreign = table.map(&f_other, 0);
        let foreign_ok = matches!(foreign, Err(MapRead::Stale(FrameRefusal::ForeignFrame)));
        let future = table.diagnose_at(RouteVersion(table.current_version().0 + 7), 0);
        let future_ok = matches!(future, Err(MapRead::FutureVersion { .. }));
        let now = table.seal();
        let ambiguous = y_sh.map(|g| table.locate(&now, &[10, 11], g));
        let ambiguous_ok = matches!(ambiguous, Some(Err(LocateRefusal::Ambiguous)));
        let twin_ok = y_sh.is_some_and(|g| {
            table.locate(&now, &[10], g).is_ok() && table.locate(&now, &R_TILES, g).is_ok()
        });
        rep.check(
            11,
            foreign_ok && future_ok && ambiguous_ok && twin_ok,
            format!(
                "foreign frame -> Stale(ForeignFrame)={foreign_ok}; future version -> FutureVersion={future_ok}; locate in [10,11] of the shared id -> Ambiguous={ambiguous_ok}; silence twin (single tile) Ok={twin_ok}"
            ),
        );
        rep.census(11, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
    }

    // ------------------------------------------------------------------
    // F10: historical frame replays; a replacement is a new generation
    // ------------------------------------------------------------------
    {
        let (l0, b0) = (look(&table), alloc_bytes());
        let f1 = table.seal();
        let old_globals: Vec<u64> = table.map(&f1, 10).expect("tile 10 at f1").globals().to_vec();
        let rev_before = table.revisions();
        let mut reordered = old_globals.clone();
        reordered.reverse();
        let reordered_differs = reordered != old_globals;
        table.advance(
            1,
            1,
            vec![(10, Some(LocalMap::prepare(wa, &reordered).expect("replacement")))],
        );
        let f2 = table.seal();
        let at_f1: Vec<u64> = table.map(&f1, 10).expect("tile 10 at f1").globals().to_vec();
        let at_f2: Vec<u64> = table.map(&f2, 10).expect("tile 10 at f2").globals().to_vec();
        let rev_after = table.revisions();
        let first_old = old_globals.first().copied();
        let local_old = first_old.and_then(|g| table.map(&f1, 10).ok().and_then(|m| m.local(g).ok()));
        let local_new = first_old.and_then(|g| table.map(&f2, 10).ok().and_then(|m| m.local(g).ok()));
        let gens = table.generation_count(10);
        rep.check(
            10,
            reordered_differs
                && at_f1 == old_globals
                && at_f2 == reordered
                && local_old != local_new
                && rev_before.coordinates != rev_after.coordinates
                && gens == 2,
            format!(
                "f1 still reads the old tile 10 (first id local {local_old:?}); f2 reads the new (local {local_new:?}); coordinates revision {:#x} -> {:#x}; generations of tile 10 = {gens}",
                rev_before.coordinates, rev_after.coordinates
            ),
        );
        rep.census(10, look(&table) as i64 - l0 as i64, alloc_bytes() - b0);
    }

    // ------------------------------------------------------------------
    // F16: structure (manual)
    // ------------------------------------------------------------------
    rep.check(
        16,
        true,
        "types defined by the probe: CountingAlloc, Rng, Lanes, Plane (bit matrix over DR coordinates), Prepared (lowered Quack program + scratch), Report (LAB); identities only via MapTable/LocalMap; id lists (dom, a_ids, b_ids, edges) are fixture/oracle input, read by no fold [manual]".to_string(),
    );

    // ------------------------------------------------------------------
    // report
    // ------------------------------------------------------------------
    rep.lines.sort_by_key(|(n, _)| *n);
    for (_, line) in &rep.lines {
        println!("{line}");
    }
    rep.census.sort_by_key(|(n, _, _)| *n);
    println!("--- census (per check: TileMap lookups on tiles 0,1,10,11; bytes allocated incl. fixture and formatting) ---");
    for (n, lookups, bytes) in &rep.census {
        println!("census F{n:<2} lookups={lookups} bytes={bytes}");
    }
    let total = rep.lines.len();
    println!(
        "falsify: {} of {total} checks failed",
        rep.fails
    );
    if rep.fails > 0 {
        std::process::exit(1);
    }
}
