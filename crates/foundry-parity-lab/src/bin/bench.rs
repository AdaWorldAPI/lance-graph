//! D-FOUNDRY-PARITY-1 bench: median of 41 warm runs per stage, nanoseconds.
//! Any stage at or above one millisecond is printed as a FAILED GATE, never
//! as a result (CLAUDE.md, "no millisecond results").

use std::hint::black_box;
use std::time::Instant;

use foundry_parity_lab::arm_a::Objects;
use foundry_parity_lab::arm_b::{demanded_fields, Guards, ItemQuery};
use foundry_parity_lab::bind::{World, READER, SPACE_A};
use foundry_parity_lab::fixture::STORES;
use foundry_parity_lab::source::{Network, Order};
use lance_graph_dt64::LaneSource;

const RUNS: usize = 41;

fn median(mut f: impl FnMut()) -> u128 {
    f();
    let mut t: Vec<u128> = (0..RUNS)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_nanos()
        })
        .collect();
    t.sort_unstable();
    t[RUNS / 2]
}

fn report(name: &str, ns: u128) {
    if ns >= 1_000_000 {
        println!("{name:<44} FAILED GATE (>= 1 ms; the path left the fold)");
    } else {
        println!("{name:<44} {ns:>10} ns");
    }
}

fn main() {
    let net = Network::ingest(Order::Identity);
    let fields = demanded_fields(0, true, true);
    println!(
        "item 0, {STORES} stores, {} demanded of 211 fields",
        fields.len()
    );
    println!("source resident bytes            {}", net.resident_bytes());
    println!(
        "arm A object bytes (second copy) {}",
        Objects::build(false).bytes()
    );
    println!("arm B bytes copied from source   0 (scratch: 2 words)");
    println!();

    report(
        "cold: World::new (register + route)",
        median(|| {
            black_box(World::new());
        }),
    );
    let world = World::new();
    report(
        "cold: bind_lanes (demanded fields)",
        median(|| {
            black_box(world.bind(&net, &fields, SPACE_A, READER).unwrap());
        }),
    );
    let binding = world.bind(&net, &fields, SPACE_A, READER).unwrap();
    report(
        "cold: open (borrow_all + pointer table)",
        median(|| {
            black_box(
                ItemQuery::open(0, &binding, &world.store, &world.view, net.frames(), &net)
                    .unwrap(),
            );
        }),
    );
    let q = ItemQuery::open(0, &binding, &world.store, &world.view, net.frames(), &net).unwrap();
    report(
        "warm: CanCook(q=1) one program, 64 stores",
        median(|| {
            black_box(q.cook_mask(black_box(1), Guards::ALL).unwrap());
        }),
    );
    report(
        "warm: CanFulfill(q=2, 45 min)",
        median(|| {
            black_box(q.fulfill_mask(2, 45).unwrap());
        }),
    );
    report(
        "warm: per-store answers (layer-cake max)",
        median(|| {
            black_box(q.answers(1, Guards::ALL).unwrap());
        }),
    );
    report(
        "warm: NetworkCapacity (Σ_q Count)",
        median(|| {
            black_box(q.network_capacity().unwrap());
        }),
    );
    report(
        "warm: network cost q=3 (Σ_k MaskedSum)",
        median(|| {
            black_box(q.network_cost(3).unwrap());
        }),
    );
    println!();
    report(
        "arm A: build 64 objects",
        median(|| {
            black_box(Objects::build(false));
        }),
    );
    let objects = Objects::build(false);
    report(
        "arm A: CanCook(q=1) over 64 objects",
        median(|| {
            for s in 0..STORES {
                black_box(objects.can_cook(s, 0, 1));
            }
        }),
    );
    report(
        "arm A: NetworkCapacity",
        median(|| {
            black_box(objects.network_capacity(0));
        }),
    );
}
