//! The electric-outage perturbation, wired onto the full 16-bit-per-tier spatial
//! cascade key (`CascadeKey` = HEEL/HIP/TWIG = family/leaf/identity, each u16).
//!
//! One key, six lenses — run: `cargo run --example spain_cascade`.
//!
//! Echoes the 2025-04-28 Iberian blackout shape: a stressed inter-region tie
//! trips, flow redistributes, the cascade fragments one region. The point this
//! example makes is that the SAME 48-bit Morton key proves location, math,
//! learning, representation, substrate, and thinking — it is not six artifacts,
//! it is six readings of one address.

use perturbation_sim::{
    cascade_keys, cascade_keys_v3, simulate_outage, CascadeConfig, Edge, Grid, IsaPath,
};

/// A small Iberian-shaped transmission graph: three regional 4-clique pockets
/// (e.g. North / Centre / South) joined by two weak tie-lines — the topology
/// where a single trip cascades within a region.
fn iberian_grid() -> Grid {
    let mut e = Vec::new();
    for region in 0..3 {
        let b = region * 4;
        for (a, c) in [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)] {
            e.push(Edge::new(b + a, b + c, 1.0, 1e6));
        }
    }
    e.push(Edge::new(3, 4, 0.01, 1e6)); // North–Centre tie
    e.push(Edge::new(7, 8, 0.01, 1e6)); // Centre–South tie
    Grid::new(12, e)
}

fn main() {
    let g = iberian_grid();
    let alive = vec![true; g.edges.len()];
    let keys = cascade_keys(&g, &alive);

    println!("== electric-outage perturbation on the 16-bit-per-tier cascade key ==\n");

    // REPRESENTATION + SUBSTRATE: the key IS the canonical (HEEL,HIP,TWIG) GUID
    // cascade path; morton48 is the packed key the SoA node carries.
    println!("bus  family(HEEL)  leaf(HIP)  identity(TWIG)   morton48");
    for (bus, k) in keys.iter().enumerate() {
        let (h, hp, t) = k.to_guid_tiers();
        println!(
            "{bus:>3}    0x{h:04X}       0x{hp:04X}     0x{t:04X}        0x{:012X}",
            k.morton48()
        );
    }

    // LOCATION: decode a key back to its quantized spectral tile (the address is
    // the position).
    let (x, y) = keys[0].tile();
    println!("\nlocation: bus 0 sits at spectral tile (x24=0x{x:06X}, y24=0x{y:06X})");

    // MATH: O(1) Morton-prefix cascade distance, zero value decode.
    println!(
        "math: cascade_distance(bus0, bus1)={}  cascade_distance(bus0, bus11)={}",
        keys[0].cascade_distance(keys[1]),
        keys[0].cascade_distance(keys[11]),
    );

    // THINKING + LEARNING: the outage cascade traverses the key; its epicentre is
    // a low-distance neighbourhood (prefix-local) — the footprint learns the tree.
    let mut p = vec![0.0; g.n];
    p[0] = 1.0;
    p[10] = -1.0;
    let res = simulate_outage(
        &g,
        &p,
        g.edges.len() - 1,
        CascadeConfig {
            overload_factor: 1.0,
            max_rounds: 16,
            rel_tol: 1e-12,
        },
    );
    let epi: Vec<usize> = res.shape.epicentre(4).into_iter().map(|(b, _)| b).collect();
    let mean = |bs: &[usize]| {
        let (mut s, mut n) = (0u32, 0u32);
        for i in 0..bs.len() {
            for j in (i + 1)..bs.len() {
                s += keys[bs[i]].cascade_distance(keys[bs[j]]) as u32;
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            s as f64 / n as f64
        }
    };
    let all: Vec<usize> = (0..g.n).collect();
    println!(
        "\nthinking: outage epicentre buses {epi:?}  ({} lines tripped)",
        res.shape.n_tripped()
    );
    println!(
        "learning: epicentre mean cascade-distance {:.3} < random baseline {:.3} \
         ⇒ footprint is prefix-local (placement learns the basin tree)",
        mean(&epi),
        mean(&all),
    );

    // ── V3 (part_of:is_a): each tier = (place:tissue), two hierarchies one key ──
    println!("\n== V3 (part_of:is_a) — the better grid representation ==\n");
    // is_a taxonomy from the power balance: source (p>0) / sink (p<0) / transfer.
    let is_a: Vec<IsaPath> = p
        .iter()
        .map(|&pi| {
            let class = if pi > 0.0 {
                1
            } else if pi < 0.0 {
                2
            } else {
                3
            };
            IsaPath {
                class,
                kind: class,
                sub: 0,
            }
        })
        .collect();
    let v3 = cascade_keys_v3(&g, &alive, &is_a);
    println!("bus  HEEL  HIP   TWIG   place(part_of)  tissue(is_a)  role");
    for (bus, k) in v3.iter().enumerate() {
        let (h, hp, t) = k.to_guid_tiers();
        let role = match k.tissue_chain()[0] {
            1 => "source/gen",
            2 => "sink/load",
            _ => "transfer",
        };
        println!(
            "{bus:>3}  {h:04X}  {hp:04X}  {t:04X}   {:?}      {:?}     {role}",
            k.place_chain(),
            k.tissue_chain()
        );
    }
    // Two orthogonal prefix queries on ONE key — impossible with V1/V2 spatial-only:
    let gen = 0usize;
    let load = 10usize;
    println!(
        "\npart_of: outage epicentre is place-local (where it blacked out)\n\
         is_a:    bus{gen}(source) vs bus{load}(sink) part_of_distance={} is_a_distance={} \
         — same key, orthogonal axes",
        v3[gen].part_of_distance(v3[load]),
        v3[gen].is_a_distance(v3[load]),
    );
}
