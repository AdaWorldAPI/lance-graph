//! D-VIRTUAL-ADDRESS-ALGEBRA-1: the composition table, the execution trace
//! of a cancelling chain before and after normalisation, and the cost of
//! each route to one virtual property. Lab only.
//!
//! Every arm's answer is compared with a row loop before it is timed, and a
//! warm arm that allocates aborts the run.
//!
//! `cargo run --release -p virtual-address-lab --example routes`

use lance_graph_mask_risc::exec::materialize_rows;
use lance_graph_mask_risc::{
    execute_into, execute_sel, Foreign, Out, Planes, Scratch, SelFold, Value,
};
use lance_graph_quack::{lower, Agg, Col, Filter, Mask, Query};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;
use virtual_address_lab::grid::{Grid, World};
use virtual_address_lab::physical;
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::route::{plan_sum_where_eq, sum_where_eq_cell_by_cell, AcrossPlan};
use virtual_address_lab::view::{Trace, VirtualTable};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
// SAFETY: forwards to the system allocator unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn counted<T>(f: impl FnOnce() -> T) -> (T, u64, u64, f64) {
    let (a, b) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    let t = Instant::now();
    let v = f();
    let us = t.elapsed().as_secs_f64() * 1e6;
    (v, ALLOCS.load(Relaxed) - a, BYTES.load(Relaxed) - b, us)
}

/// Median of 21 samples of `k` calls, ns per call. Aborts on a warm allocation.
fn warm<T>(arm: &str, rows: usize, k: u32, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let (_, allocs, _, _) = counted(|| black_box(f()));
    assert_eq!(allocs, 0, "{arm} allocates when warm");
    let mut samples: Vec<f64> = (0..21)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..k {
                black_box(f());
            }
            t.elapsed().as_nanos() as f64 / f64::from(k)
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{rows}\t{arm}\t{:.1}\t{:.3}\t0",
        samples[10] / 1e3,
        samples[10] / rows as f64
    );
    samples[10]
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn main() {
    println!(
        "# avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );

    // ---------------------------------------------------- composition table
    println!("# composition: row THEN column");
    print!("then");
    for g in Recipe::ALL {
        print!("\t{}", g.name());
    }
    println!();
    for g in Recipe::ALL {
        print!("{}", g.name());
        for h in Recipe::ALL {
            print!("\t{}", g.then(h).name());
        }
        println!();
    }

    // ------------------------------------------------------------- fixture
    const LANES: usize = 32;
    const LEN: usize = 262_144;
    let grid = Grid::new(LANES, LEN, |l, p| {
        (mix((l * 1_000_003 + p) as u64) % 1_000) as i32
    });
    let world = World::new(LANES);
    let binding = world.bind(&grid).unwrap();
    let (sum_col, where_col, v) = (5usize, 7usize, 123);
    let want: i64 = (0..LEN)
        .filter(|&p| grid.cell(where_col, p) == v)
        .map(|p| i64::from(grid.cell(sum_col, p)))
        .sum();
    println!(
        "# tile: {LANES} lanes x {LEN} rows, {} source-owned bytes; demand SUM(c{sum_col}) WHERE c{where_col} = {v} -> {want}",
        grid.source_bytes()
    );

    // ---------------------------------------- trace before / after normalising
    let chain = [Recipe::ROTATE90, Recipe::ROTATE270];
    let trace = Trace::default();
    let entity = VirtualTable::bound(&world, &binding, &grid, Recipe::TRANSPOSE, &trace).unwrap();
    let ((mut plan, table), a, b, us) = counted(|| {
        let table = entity.through(Recipe::normalize(&chain));
        (
            plan_sum_where_eq(&table, sum_col, where_col, v).unwrap(),
            table,
        )
    });
    assert_eq!(plan.sum(&table), Ok(want));
    println!("# trace, normalised: chain {:?} -> {}; counters [resolutions, reads, gathers, transforms, copied, folds] = {:?}; plan built in {us:.1} us with {a} allocations, {b} B (program and scratch)",
        chain.map(Recipe::name), Recipe::normalize(&chain).name(), trace.snapshot());

    let moved = Trace::default();
    let ((got, held), a, b, us) = counted(|| {
        let m = physical::apply(&physical::of(&grid, &moved), Recipe::TRANSPOSE, &moved);
        let out = physical::apply_chain(&m, &chain, &moved);
        let got: i64 = out
            .iter()
            .filter(|row| row[where_col] == v)
            .map(|row| i64::from(row[sum_col]))
            .sum();
        (got, out.len() * out[0].len() * 4)
    });
    assert_eq!(got, want);
    println!("# trace, every step applied: counters = {:?}; {us:.0} us, {a} allocations, {b} B allocated, final view {held} B", moved.snapshot());

    println!("rows\tarm\tmedian_us\tns_per_row\twarm_allocs");
    // ------------------------------------------------- A: contiguous lanes
    let direct = warm("A_direct_lanes_entity_view", LEN, 30, || {
        plan.sum(&entity).unwrap()
    });
    let cancelled = warm("A_chain_cancelled_to_identity", LEN, 30, || {
        plan.sum(&table).unwrap()
    });
    let turned = VirtualTable::bound(&world, &binding, &grid, Recipe::ROTATE90, &trace).unwrap();
    let mut turned_plan = plan_sum_where_eq(&turned, sum_col, where_col, v).unwrap();
    assert_eq!(turned_plan.sum(&turned), Ok(want));
    assert!(turned_plan.reversal_cancelled);
    let reversed = warm("A_rows_reversed_reversal_dropped", LEN, 30, || {
        turned_plan.sum(&turned).unwrap()
    });

    // ------------------------------------------- F: the materialised view
    let scrap = Trace::default();
    let (built, a, b, us) = counted(|| turned.materialize());
    println!("# F cold: the Rotate90 view built in {us:.0} us, {a} allocation(s), {b} B, {} values copied", built.len());
    let _ = scrap;
    let f_fold = warm("F_materialised_view_scalar_fold", LEN, 10, || {
        let mut s = 0i64;
        for row in built.chunks_exact(LANES) {
            if row[where_col] == v {
                s += i64::from(row[sum_col]);
            }
        }
        s
    });
    assert_eq!(
        built
            .chunks_exact(LANES)
            .filter(|r| r[where_col] == v)
            .map(|r| i64::from(r[sum_col]))
            .sum::<i64>(),
        want
    );

    // --------------------------- C/D/E: one property over a given selection
    let lanes = entity.operands();
    let planes_of = |masks| Planes {
        n_rows: LEN,
        masks,
        lanes,
    };
    let sum_mask = lower(&Query {
        filter: Filter::plane(Mask(0)),
        agg: Agg::SumI32(Col(sum_col as u16)),
    })
    .unwrap();
    let mut scratch = Scratch::for_program(&sum_mask, LEN).unwrap();
    for one_in in [1024u64, 64, 4, 1] {
        let mut words = vec![0u64; LEN.div_ceil(64)];
        for p in (0..LEN).filter(|&p| mix(0xABC ^ p as u64).is_multiple_of(one_in)) {
            words[p / 64] |= 1u64 << (p % 64);
        }
        let (rows, _, bytes, us) = counted(|| materialize_rows(&words, LEN));
        let positions: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
        let want_sel: i64 = positions
            .iter()
            .map(|&p| i64::from(grid.cell(sum_col, p as usize)))
            .sum();
        println!("# selection 1/{one_in}: {} rows; mask {} B; position list {} B (built from the mask in {us:.0} us, {bytes} B allocated)", positions.len(), words.len() * 8, positions.len() * 4);
        let masks: [&[u64]; 1] = [&words];
        let planes = planes_of(&masks);
        assert_eq!(
            execute_into(&sum_mask, &planes, &Foreign::NONE, &mut scratch, Out::None),
            Ok(Value::SumI64(want_sel))
        );
        let e = warm(&format!("E_mask_1/{one_in}"), LEN, 30, || {
            execute_into(&sum_mask, &planes, &Foreign::NONE, &mut scratch, Out::None).unwrap()
        });
        let fold = SelFold::SumI32 {
            lane: sum_col as u16,
        };
        let bare = planes_of(&[]);
        assert_eq!(
            execute_sel(fold, &bare, &positions),
            Ok(Value::SumI64(want_sel))
        );
        let d = warm(&format!("D_positions_1/{one_in}"), LEN, 30, || {
            execute_sel(fold, &bare, &positions).unwrap()
        });
        println!(
            "# selection 1/{one_in}: positions / mask = {:.2}; gathers per call {}",
            d / e,
            positions.len()
        );
    }

    // C: an arbitrary reorder. A whole-population sum ignores it; a prefix
    // in the new order is a position list.
    let (order, _, bytes, us) = counted(|| {
        let mut order: Vec<u32> = (0..LEN as u32).collect();
        order.sort_by_key(|&r| mix(0x5EED ^ u64::from(r)));
        order
    });
    println!(
        "# C cold: an arbitrary row order (a sort) in {us:.0} us, {bytes} B allocated, {} B held",
        order.len() * 4
    );
    let fold = SelFold::SumI32 {
        lane: sum_col as u16,
    };
    let bare = planes_of(&[]);
    for k in [256usize, 16_384, LEN] {
        let want_k: i64 = order[..k]
            .iter()
            .map(|&p| i64::from(grid.cell(sum_col, p as usize)))
            .sum();
        assert_eq!(
            execute_sel(fold, &bare, &order[..k]),
            Ok(Value::SumI64(want_k))
        );
        warm(&format!("C_reorder_first_{k}_gathered"), k, 30, || {
            execute_sel(fold, &bare, &order[..k]).unwrap()
        });
    }
    let all: i64 = (0..LEN).map(|p| i64::from(grid.cell(sum_col, p))).sum();
    assert_eq!(execute_sel(fold, &bare, &order), Ok(Value::SumI64(all)));
    let full = vec![u64::MAX; LEN / 64];
    let masks: [&[u64]; 1] = [&full];
    let planes = planes_of(&masks);
    assert_eq!(
        execute_into(&sum_mask, &planes, &Foreign::NONE, &mut scratch, Out::None),
        Ok(Value::SumI64(all))
    );
    warm("C_reorder_whole_sum_order_dropped", LEN, 30, || {
        execute_into(&sum_mask, &planes, &Foreign::NONE, &mut scratch, Out::None).unwrap()
    });

    // -------------------------------------- B: a column that crosses lanes
    const N: usize = 256;
    let square = Grid::new(N, N, |l, p| (mix((l * 4099 + p) as u64) % 50) as i32);
    let sworld = World::new(N);
    let sbinding = sworld.bind(&square).unwrap();
    let strace = Trace::default();
    let lane_matrix =
        VirtualTable::bound(&sworld, &sbinding, &square, Recipe::IDENTITY, &strace).unwrap();
    let want_b: i64 = (0..N)
        .filter(|&l| square.cell(l, 20) == 4)
        .map(|l| i64::from(square.cell(l, 9)))
        .sum();
    let mut across = AcrossPlan::sum_where_eq(&square, 4).unwrap();
    assert_eq!(across.run(&square, 9, 20), Ok(want_b));
    assert_eq!(
        sum_where_eq_cell_by_cell(&lane_matrix, 9, 20, 4),
        Some(want_b)
    );
    println!("# square tile {N} x {N}: a view column is position p of every lane ({N} rows)");
    let b_strided = warm("B_strided_slab_unadmitted", N, 5_000, || {
        across.run(&square, 9, 20).unwrap()
    });
    let b_cells = warm("B_cell_by_cell_bound", N, 5_000, || {
        sum_where_eq_cell_by_cell(&lane_matrix, 9, 20, 4).unwrap()
    });
    let sentity =
        VirtualTable::bound(&sworld, &sbinding, &square, Recipe::TRANSPOSE, &strace).unwrap();
    let mut splan = plan_sum_where_eq(&sentity, 9, 20, 4).unwrap();
    let b_lane = warm("B_reference_one_lane_same_rows", N, 5_000, || {
        splan.sum(&sentity).unwrap()
    });

    println!(
        "# ratios: cancelled/direct {:.2}; reversed/direct {:.2}; materialised fold/direct {:.2}; strided/lane {:.2}; cell-by-cell/lane {:.2}",
        cancelled / direct,
        reversed / direct,
        f_fold / direct,
        b_strided / b_lane,
        b_cells / b_lane
    );
}
