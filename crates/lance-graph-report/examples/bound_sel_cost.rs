//! Cost of the witnessed selection path (D-DR-SELECT-K-SV-2), term by term.
//!
//! N = 65,536 rows, one `i32` lane, a source-issued Top-K selection by a key
//! (K = 64 and 256). Prints one TSV table on stdout. Run with
//! `cargo run --release --example bound_sel_cost -p lance-graph-report`.
//!
//! Terms (medians, ns):
//! - `produce`: the source builds the Top-K position list (once per frame).
//! - `bind`: `BindStore::bind_lanes` (cold, once per frame).
//! - `witness`: `LaneBinding::selected` — the check and both borrows.
//! - `fold`: `execute_sel` over the borrowed operands, no witness.
//! - `witnessed_fold`: `fold_bound_selection` — witness and fold, per call.
//! - `mask_fold`: the mask program over the same rows as a set (built once),
//!   for the selections where a set says the same thing.
//!
//! The Top-K order is the source's own claim; no order witness is consulted
//! and COUNT / SUM do not depend on it.

use std::hint::black_box;
use std::time::Instant;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRequest, LaneSource,
    RouteAt, SchemaRev, SelAddr, SelectionSource,
};
use lance_graph_mask_risc::{
    execute_into, execute_sel, words_for, Foreign, LaneRef, Operand, Out, Planes, Program, Scratch,
    SelFold, Terminal, Value,
};
use lance_graph_report::{fold_bound_selection, BoundFold};

const N: usize = 65_536;
const CLASS: u32 = 0x0C1A_0001;
const VIEW_CLASS: ClassId = 0x0C1A;
const SPACE: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
const READER: Scope = Scope { id: 7 };
const WARMUP: usize = 20;
const RUNS: usize = 201;

struct View([FieldRef; 1]);

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == VIEW_CLASS {
            &self.0
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

struct Source {
    frames: FrameStore,
    values: Vec<i32>,
    key: Vec<i32>,
    sels: Vec<Vec<u32>>,
}

impl Source {
    /// The K rows with the smallest key, ties by row: what a Top-K hands over.
    fn top_k(&self, k: usize) -> Vec<u32> {
        let mut rows: Vec<u32> = (0..N as u32).collect();
        rows.select_nth_unstable_by_key(k, |&r| (self.key[r as usize], r));
        rows.truncate(k);
        rows.sort_unstable_by_key(|&r| (self.key[r as usize], r));
        rows
    }
}

impl LaneSource for Source {
    type Lane<'a> = LaneRef<'a>;
    fn lane_of(
        &self,
        class: u32,
        _: SchemaRev,
        field: u8,
        _: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        self.frames.check(frame).ok()?;
        (class == CLASS && field == 0).then_some(LaneAddr {
            space: SPACE,
            lane: 0,
            scope: READER,
        })
    }
    fn borrow(&self, _: &LaneAddr) -> Option<LaneRef<'_>> {
        Some(LaneRef::I32(&self.values))
    }
    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

impl SelectionSource for Source {
    fn selection_of(&self, sel: u16, frame: &Frame) -> Option<SelAddr> {
        self.frames.check(frame).ok()?;
        self.sels.get(usize::from(sel)).map(|_| SelAddr {
            space: SPACE,
            sel,
            scope: READER,
        })
    }
    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        self.sels.get(usize::from(addr.sel)).map(Vec::as_slice)
    }
}

fn median_ns<T>(mut f: impl FnMut() -> T) -> u128 {
    for _ in 0..WARMUP {
        black_box(f());
    }
    let mut t = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let s = Instant::now();
        black_box(f());
        t.push(s.elapsed().as_nanos());
    }
    t.sort_unstable();
    t[RUNS / 2]
}

fn main() {
    let mut x = 0x5EED_0002u64;
    let mut next = || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut source = Source {
        frames: FrameStore::new(FrameRevisions {
            source: 1,
            relation_map: 1,
            coordinates: 1,
        }),
        values: (0..N).map(|_| (next() % 2001) as i32 - 1000).collect(),
        key: (0..N).map(|_| (next() % 1_000_000) as i32).collect(),
        sels: Vec::new(),
    };
    let view = View([FieldRef::new("ord:amount", "Amount")]);
    let mut store = BindStore::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 1,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .unwrap();
    let route = RouteAt {
        position: 5,
        version: store.route(5, CLASS).unwrap(),
    };
    let fields = WideFieldMask::from_positions(&[0]);

    println!(
        "# N={N} medians of {RUNS} after {WARMUP} warm-up, ns; produce and bind are once per frame"
    );
    println!("k\tproduce_ns\tbind_ns\twitness_ns\tfold_sum_ns\tfold_count_ns\twitnessed_fold_sum_ns\tmask_build_ns\tmask_fold_sum_ns\ttotal_r1_ns\ttotal_r100_ns\ttotal_r100_recheck_ns\tposition_bytes");
    for (sel, k) in [(0u16, 64usize), (1, 256)] {
        let produce = median_ns(|| source.top_k(k));
        source.sels.push(source.top_k(k));
        let frame = source.frames.seal();
        let request = || LaneRequest {
            route,
            rev: SchemaRev(1),
            fields: &fields,
            frame: &frame,
            scope: READER,
            selection: SPACE,
        };
        let bind = median_ns(|| {
            store
                .bind_lanes(&view, &source.frames, &source, request())
                .unwrap()
        });
        let binding: LaneBinding = store
            .bind_lanes(&view, &source.frames, &source, request())
            .unwrap();

        let witness = median_ns(|| {
            binding
                .selected(&store, &view, &source.frames, &source, 0, sel)
                .unwrap()
        });
        let (lane, positions) = binding
            .selected(&store, &view, &source.frames, &source, 0, sel)
            .unwrap();
        let lanes = [lane];
        let planes = Planes {
            n_rows: N,
            masks: &[],
            lanes: &lanes,
        };
        // Oracle before timing: plain indexing.
        let want: i64 = positions
            .iter()
            .map(|&p| i64::from(source.values[p as usize]))
            .sum();
        assert_eq!(
            execute_sel(SelFold::SumI32 { lane: 0 }, &planes, positions),
            Ok(Value::SumI64(want))
        );
        let fold_sum = median_ns(|| execute_sel(SelFold::SumI32 { lane: 0 }, &planes, positions));
        let fold_count =
            median_ns(|| execute_sel(SelFold::CountGtI32 { lane: 0, t: 0 }, &planes, positions));
        let witnessed = median_ns(|| {
            fold_bound_selection(
                &binding,
                &store,
                &view,
                &source.frames,
                &source,
                0,
                sel,
                BoundFold::SumI32,
            )
        });
        assert_eq!(
            fold_bound_selection(
                &binding,
                &store,
                &view,
                &source.frames,
                &source,
                0,
                sel,
                BoundFold::SumI32
            )
            .ok(),
            Some(Value::SumI64(want))
        );

        // The same rows as a set, through the mask program (a Top-K has no repeats).
        let build = |pos: &[u32]| {
            let mut m = vec![0u64; words_for(N)];
            for &p in pos {
                m[p as usize / 64] |= 1u64 << (p % 64);
            }
            m
        };
        let mask_build = median_ns(|| build(positions));
        let mask = build(positions);
        let masks: [&[u64]; 1] = [&mask];
        let mplanes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        let program = Program::new(
            vec![],
            Terminal::MaskedSumI32 {
                mask: Operand::Plane(0),
                lane: 0,
            },
        );
        let mut scratch = Scratch::for_program(&program, N).unwrap();
        assert_eq!(
            execute_into(&program, &mplanes, &Foreign::NONE, &mut scratch, Out::None),
            Ok(Value::SumI64(want))
        );
        let mask_fold =
            median_ns(|| execute_into(&program, &mplanes, &Foreign::NONE, &mut scratch, Out::None));

        println!(
            "{k}\t{produce}\t{bind}\t{witness}\t{fold_sum}\t{fold_count}\t{witnessed}\t{mask_build}\t{mask_fold}\t{}\t{}\t{}\t{}",
            produce + bind + witnessed,
            produce + bind + witness + 100 * fold_sum,
            produce + bind + 100 * witnessed,
            positions.len() * 4,
        );
    }
}
