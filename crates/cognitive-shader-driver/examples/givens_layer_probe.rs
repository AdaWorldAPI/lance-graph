//! Givens layer: hydration writes thought 0, hypotheses write thoughts 1..63.
//!
//! Pre-registered in `.claude/plans/ce64-hypothesis-layers-v1.md`
//! ("Givens layer: hydration is thought 0").
//!
//! - A world is a dense class tile: given facts `(row, lane, cell)` and at most
//!   one declared relation per row, both in arbitrary declaration order.
//! - Hydration buckets the facts by row and lane (so the declaration order
//!   cannot change the bytes) and stamps each row's CausalEdge64 bits 59..63
//!   as one `EpistemicState5`: topology from the ontology, certification
//!   capped by the proofs hydration ran (membership: Associated; a relation
//!   whose endpoints and named intermediate resolve: Related).
//! - A hypothesis write offering a different value for a given's lane is
//!   refused and recorded as a contradiction; the given's own value is
//!   agreement and is not stored. Hypotheses merge higher-thought-wins.
//! - One read-only givens table per (classid, sealed version) behind a
//!   `LazyLock` registry; the read path never hydrates.
//!
//! Run: `cargo run --release --example givens_layer_probe` (tier printed).
//! Tests: `cargo test --example givens_layer_probe`.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use causal_edge::CausalEdge64;
use lance_graph_contract::epistemic_state5::{Certification3, Epi5Gen, EpistemicState5, Topology2};

const LANES: usize = 30;
const ALL_LANES: u32 = (1 << LANES) - 1;
const CLASSID: u32 = 0x0C0C_0000;
/// Every thought but the givens layer.
const HYPOTHESES: u64 = !1;
type Cell = [u8; 16];
type Lanes = [Cell; LANES];

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn cell(a: u64) -> Cell {
    let b = mix(a);
    let mut c = [0u8; 16];
    c[..8].copy_from_slice(&a.to_le_bytes());
    c[8..].copy_from_slice(&b.to_le_bytes());
    c
}

/// The value thought `t` offers for `lane` of `row` (t = 0: the given).
fn val(t: usize, row: u32, lane: usize) -> Cell {
    cell(mix(row as u64 ^ ((t as u64) << 48) ^ ((lane as u64) << 40)))
}

struct BitIter(u64);
impl Iterator for BitIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let i = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(i)
    }
}

// ── The world (the ontology as declared) ──────────────────────────────────

/// How a relation reaches its target, as the ontology declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Via {
    Direct,
    /// Through a named intermediate row (which may not resolve).
    Named(u32),
    /// Through an intermediate the ontology does not name.
    Unnamed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Relation {
    from: u32,
    to: u32,
    via: Via,
    /// What the ontology text claims. Never trusted by hydration.
    claimed: Certification3,
}

#[derive(Clone, Debug)]
struct World {
    n: usize,
    facts: Vec<(u32, u8, Cell)>,
    relations: Vec<Relation>,
}

/// A deterministic world: 0..3 given lanes per row; three rows in four declare
/// a relation; one target in nine lies outside the class; every claim from
/// Open to Causes appears.
fn world(n: usize, seed: u64) -> World {
    let mut facts = Vec::new();
    let mut relations = Vec::new();
    let outside = (n + n / 8) as u64;
    for r in 0..n as u32 {
        let h = mix(seed ^ r as u64);
        let mut taken = 0u32;
        for i in 0..(h % 4) {
            let lane = (mix(h ^ i) % LANES as u64) as u32;
            if taken >> lane & 1 == 0 {
                taken |= 1 << lane;
                facts.push((r, lane as u8, val(0, r, lane as usize)));
            }
        }
        let g = mix(h);
        if !g.is_multiple_of(4) {
            let via = match (g >> 8) % 3 {
                0 => Via::Direct,
                1 => Via::Named((mix(g ^ 1) % outside) as u32),
                _ => Via::Unnamed,
            };
            relations.push(Relation {
                from: r,
                to: (mix(g ^ 2) % outside) as u32,
                via,
                claimed: Certification3::ALL[((g >> 16) % 6) as usize],
            });
        }
    }
    World {
        n,
        facts,
        relations,
    }
}

/// The same world, every declaration list reversed.
#[cfg(test)]
fn reversed(w: &World) -> World {
    let mut v = w.clone();
    v.facts.reverse();
    v.relations.reverse();
    v
}

// ── Hydration: the only writer of layer 0 ─────────────────────────────────

/// The read-only givens of one (classid, sealed version).
#[derive(Debug, PartialEq, Eq)]
struct Givens {
    /// Given lanes per row.
    lanes: Vec<u32>,
    /// Start of each row's cells in `cells` (row r: `start[r]..start[r+1]`).
    start: Vec<u32>,
    /// Cells, per row ascending by lane.
    cells: Vec<Cell>,
    /// Per row: CausalEdge64 with only bits 59..63 written.
    edge: Vec<u64>,
}

impl Givens {
    fn cell(&self, row: usize, lane: usize) -> Cell {
        let have = self.lanes[row];
        self.cells[self.start[row] as usize + (have & ((1u32 << lane) - 1)).count_ones() as usize]
    }
}

/// Topology as the ontology declares it; a named intermediate that does not
/// resolve is known to exist but not identified.
fn topology(w: &World, rel: &Relation) -> Topology2 {
    match rel.via {
        Via::Direct => Topology2::Direct,
        Via::Named(i) if (i as usize) < w.n => Topology2::IndirectKnown,
        Via::Named(_) | Via::Unnamed => Topology2::IndirectUnknown,
    }
}

/// The strongest certification the proofs hydration runs can earn.
/// Membership of both endpoints: Associated. The relation resolves (and a named
/// intermediate resolves): Related. Nothing higher is tested here.
fn proof_ceiling(w: &World, rel: &Relation) -> Certification3 {
    let member = (rel.from as usize) < w.n && (rel.to as usize) < w.n;
    if !member {
        return Certification3::Open;
    }
    let intermediate_resolves = match rel.via {
        Via::Named(i) => (i as usize) < w.n,
        Via::Direct | Via::Unnamed => true,
    };
    if intermediate_resolves {
        Certification3::Related
    } else {
        Certification3::Associated
    }
}

/// The state hydration writes for one row.
fn state_of(w: &World, rel: Option<&Relation>) -> EpistemicState5 {
    match rel {
        None => EpistemicState5::new(Epi5Gen::V1, Topology2::Unknown, Certification3::Open),
        Some(r) => EpistemicState5::new(Epi5Gen::V1, topology(w, r), proof_ceiling(w, r)),
    }
}

fn hydrate(w: &World) -> Givens {
    // Bucket by row, then place each cell at its lane rank: the bytes depend
    // on (row, lane) only, never on declaration order.
    let mut lanes = vec![0u32; w.n];
    for &(r, l, _) in &w.facts {
        let bit = 1u32 << l;
        assert!(lanes[r as usize] & bit == 0, "duplicate given ({r}, {l})");
        lanes[r as usize] |= bit;
    }
    let mut start = Vec::with_capacity(w.n + 1);
    start.push(0u32);
    for &l in &lanes {
        start.push(start.last().unwrap() + l.count_ones());
    }
    let mut cells = vec![[0u8; 16]; *start.last().unwrap() as usize];
    for &(r, l, c) in &w.facts {
        let r = r as usize;
        cells[start[r] as usize + (lanes[r] & ((1u32 << l) - 1)).count_ones() as usize] = c;
    }
    let mut rel: Vec<Option<&Relation>> = vec![None; w.n];
    for x in &w.relations {
        let slot = &mut rel[x.from as usize];
        assert!(slot.is_none(), "row {} declares two relations", x.from);
        *slot = Some(x);
    }
    let edge = rel
        .iter()
        .map(|r| CausalEdge64(0).with_epistemic_raw5(state_of(w, *r).raw()).0)
        .collect();
    Givens {
        lanes,
        start,
        cells,
        edge,
    }
}

// ── The registry: one table per (classid, sealed version) ─────────────────

type Registry = Mutex<HashMap<(u32, u64), &'static Givens>>;
static REGISTRY: LazyLock<Registry> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// At seal: hydrate once from the resident world and publish the table.
/// Sealing the same key again returns the table already published.
fn seal(classid: u32, version: u64, w: &World) -> &'static Givens {
    let mut reg = REGISTRY.lock().unwrap();
    reg.entry((classid, version))
        .or_insert_with(|| Box::leak(Box::new(hydrate(w))))
}

/// The read path: a published table, or nothing. Never hydrates.
fn sealed(classid: u32, version: u64) -> Option<&'static Givens> {
    REGISTRY.lock().unwrap().get(&(classid, version)).copied()
}

// ── The tunnel: givens plus hypothesis layers ──────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct Contradiction {
    row: u32,
    thought: u8,
    lanes: u32,
    /// The refused cells, ascending by lane.
    offered: Vec<Cell>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct WriteReport {
    accepted: u32,
    agreed: u32,
    contradicted: u32,
}

#[derive(Debug, PartialEq, Eq)]
enum WriteError {
    /// Layer 0 is written by hydration only.
    GivensLayer,
}

#[derive(Clone, Copy)]
struct Entry {
    thought: u8,
    lanes: u32,
    start: u32,
}

struct Tunnel<'g> {
    givens: &'g Givens,
    /// Per row: bit t set when thought t has a stored cell (bit 0: givens).
    written: Vec<u64>,
    entries: Vec<Vec<Entry>>,
    payload: Vec<Cell>,
    contradictions: Vec<Contradiction>,
}

impl<'g> Tunnel<'g> {
    fn new(givens: &'g Givens) -> Self {
        let n = givens.lanes.len();
        Self {
            givens,
            written: givens.lanes.iter().map(|&l| u64::from(l != 0)).collect(),
            entries: vec![Vec::new(); n],
            payload: Vec::new(),
            contradictions: Vec::new(),
        }
    }

    /// Offer `cells` (ascending by lane) for `lanes` of `row` under `thought`.
    fn write(
        &mut self,
        thought: u8,
        row: u32,
        lanes: u32,
        cells: &[Cell],
    ) -> Result<WriteReport, WriteError> {
        if thought == 0 {
            return Err(WriteError::GivensLayer);
        }
        assert!(thought < 64 && lanes & !ALL_LANES == 0);
        assert_eq!(cells.len(), lanes.count_ones() as usize);
        let r = row as usize;
        let given = self.givens.lanes[r];
        let mut rep = WriteReport::default();
        let mut offered = Vec::new();
        let first = self.payload.len() as u32;
        for (i, lane) in BitIter(lanes as u64).enumerate() {
            if given >> lane & 1 == 0 {
                rep.accepted |= 1 << lane;
                self.payload.push(cells[i]);
            } else if self.givens.cell(r, lane) == cells[i] {
                rep.agreed |= 1 << lane;
            } else {
                rep.contradicted |= 1 << lane;
                offered.push(cells[i]);
            }
        }
        if rep.contradicted != 0 {
            self.contradictions.push(Contradiction {
                row,
                thought,
                lanes: rep.contradicted,
                offered,
            });
        }
        if rep.accepted != 0 {
            // Keep the row's entries ascending by thought; a later write of the
            // same thought lands after the earlier one, so it wins on read.
            let e = &mut self.entries[r];
            let at = e.partition_point(|x| x.thought <= thought);
            e.insert(
                at,
                Entry {
                    thought,
                    lanes: rep.accepted,
                    start: first,
                },
            );
            self.written[r] |= 1 << thought;
        }
        Ok(rep)
    }

    /// The merged view of `row` under a thought selection, restricted to `lanes`.
    fn read(&self, row: usize, thoughts: u64, lanes: u32, out: &mut Lanes) -> u32 {
        if self.written[row] & thoughts == 0 {
            return 0;
        }
        let mut m = 0;
        if thoughts & 1 == 1 {
            let mm = self.givens.lanes[row] & lanes;
            for lane in BitIter(mm as u64) {
                out[lane] = self.givens.cell(row, lane);
            }
            m |= mm;
        }
        for e in &self.entries[row] {
            if thoughts >> e.thought & 1 == 0 {
                continue;
            }
            let mm = e.lanes & lanes;
            for lane in BitIter(mm as u64) {
                out[lane] = self.payload
                    [e.start as usize + (e.lanes & ((1u32 << lane) - 1)).count_ones() as usize];
            }
            m |= mm;
        }
        m
    }

    /// Rows with at least one stored cell under the selection.
    fn count(&self, thoughts: u64) -> u64 {
        self.written.iter().filter(|&&w| w & thoughts != 0).count() as u64
    }

    /// Does `row` hold a stored cell under the selection?
    #[cfg(test)]
    fn count_row(&self, row: usize, thoughts: u64) -> u64 {
        u64::from(self.written[row] & thoughts != 0)
    }
}

/// Hypotheses for thoughts 1..=t: about a third of the rows, four lanes each,
/// own values; some land on given lanes.
fn hypotheses(n: usize, t: u8, seed: u64) -> Vec<(u8, u32, u32, Vec<Cell>)> {
    let mut out = Vec::new();
    for th in 1..=t {
        for r in 0..n as u32 {
            let h = mix(seed ^ ((th as u64) << 40) ^ r as u64);
            if !h.is_multiple_of(3) {
                continue;
            }
            let mut lanes = 0u32;
            for i in 0..4 {
                lanes |= 1 << (mix(h ^ i) % LANES as u64);
            }
            let cells = BitIter(lanes as u64)
                .map(|l| val(th as usize, r, l))
                .collect();
            out.push((th, r, lanes, cells));
        }
    }
    out
}

fn ns_per(elements: u64, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64 / elements.max(1) as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[2]
}

fn main() {
    const N: usize = 65_536;
    println!(
        "tier: {}  (dense class tile, {N} rows, {LANES} value lanes)",
        tier()
    );
    let w = world(N, 7);
    let hyd = ns_per(N as u64, || {
        black_box(hydrate(&w));
    });
    let g = seal(CLASSID, 1, &w);
    assert!(std::ptr::eq(g, sealed(CLASSID, 1).expect("sealed")));
    let hyps = hypotheses(N, 63, 11);
    let offered: u64 = hyps.iter().map(|h| h.2.count_ones() as u64).sum();
    let mut tun = Tunnel::new(g);
    let t0 = Instant::now();
    let mut rep = WriteReport::default();
    for (th, r, l, c) in &hyps {
        let x = tun.write(*th, *r, *l, c).expect("hypothesis layer");
        rep.accepted += x.accepted.count_ones();
        rep.agreed += x.agreed.count_ones();
        rep.contradicted += x.contradicted.count_ones();
    }
    let write = t0.elapsed().as_nanos() as f64 / offered as f64;
    let mut r = 12345u64;
    let probes: Vec<usize> = (0..N)
        .map(|_| {
            r = mix(r);
            (r % N as u64) as usize
        })
        .collect();
    let mut out = [[0u8; 16]; LANES];
    let read_with = ns_per(N as u64, || {
        for &p in &probes {
            black_box(tun.read(p, u64::MAX, ALL_LANES, &mut out));
            black_box(&out);
        }
    });
    let read_without = ns_per(N as u64, || {
        for &p in &probes {
            black_box(tun.read(p, HYPOTHESES, ALL_LANES, &mut out));
            black_box(&out);
        }
    });
    let count_with = ns_per(N as u64, || {
        black_box(tun.count(u64::MAX));
    });
    let count_without = ns_per(N as u64, || {
        black_box(tun.count(HYPOTHESES));
    });
    println!(
        "hydration {hyd:.2} ns/row; hypothesis write {write:.2} ns/offered cell \
         ({offered} offered: {} accepted, {} agreed, {} contradicted, {} contradiction records)",
        rep.accepted,
        rep.agreed,
        rep.contradicted,
        tun.contradictions.len()
    );
    println!(
        "merged point read, all 30 lanes: with givens {read_with:.2} ns, without {read_without:.2} ns; \
         count rows: with givens {count_with:.3} ns/row ({}), without {count_without:.3} ns/row ({})",
        tun.count(u64::MAX),
        tun.count(HYPOTHESES)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::epistemic_state5::fact;
    use std::collections::BTreeMap;

    /// Two values for one given lane contradict each other; hydration refuses
    /// the world instead of letting declaration order pick one.
    #[test]
    #[should_panic(expected = "duplicate given")]
    fn hydration_refuses_two_values_for_one_given_lane() {
        let w = World {
            n: 2,
            facts: vec![(1, 3, val(0, 1, 3)), (1, 3, val(1, 1, 3))],
            relations: Vec::new(),
        };
        let _ = hydrate(&w);
    }

    /// A row declares at most one relation; a second is refused, not
    /// overwritten.
    #[test]
    #[should_panic(expected = "two relations")]
    fn hydration_refuses_two_relations_from_one_row() {
        let rel = |to| Relation {
            from: 0,
            to,
            via: Via::Direct,
            claimed: Certification3::Related,
        };
        let w = World {
            n: 3,
            facts: Vec::new(),
            relations: vec![rel(1), rel(2)],
        };
        let _ = hydrate(&w);
    }

    const N: usize = 3000;

    /// Brute-force givens straight from the declarations.
    fn given_oracle(w: &World) -> BTreeMap<(u32, usize), Cell> {
        w.facts
            .iter()
            .map(|&(r, l, c)| ((r, l as usize), c))
            .collect()
    }

    /// Brute-force merge: givens if bit 0 is selected; hypotheses in thought
    /// order, higher wins, never on a given lane.
    fn read_oracle(
        w: &World,
        hyps: &[(u8, u32, u32, Vec<Cell>)],
        row: u32,
        thoughts: u64,
        lanes: u32,
    ) -> BTreeMap<usize, Cell> {
        let given = given_oracle(w);
        let mut out = BTreeMap::new();
        if thoughts & 1 == 1 {
            for (&(r, l), &c) in &given {
                if r == row && lanes >> l & 1 == 1 {
                    out.insert(l, c);
                }
            }
        }
        let mut sorted: Vec<_> = hyps.iter().filter(|h| h.1 == row).collect();
        sorted.sort_by_key(|h| h.0);
        for (th, _, l, c) in sorted {
            if thoughts >> th & 1 == 0 {
                continue;
            }
            for (i, lane) in BitIter(*l as u64).enumerate() {
                if lanes >> lane & 1 == 1 && !given.contains_key(&(row, lane)) {
                    out.insert(lane, c[i]);
                }
            }
        }
        out
    }

    fn load<'g>(g: &'g Givens, hyps: &[(u8, u32, u32, Vec<Cell>)]) -> Tunnel<'g> {
        let mut t = Tunnel::new(g);
        for (th, r, l, c) in hyps {
            t.write(*th, *r, *l, c).unwrap();
        }
        t
    }

    /// (a) Same world, same bytes, whatever order the declarations arrive in.
    #[test]
    fn a_two_hydrations_of_one_world_are_identical() {
        let w = world(N, 3);
        let a = hydrate(&w);
        let b = hydrate(&w);
        let c = hydrate(&reversed(&w));
        assert_eq!(a.lanes.len(), N);
        assert!(a.cells.len() > N / 2, "the fixture has givens");
        assert_ne!(w.facts, reversed(&w).facts, "the reversal reorders");
        assert_eq!(a, b);
        assert_eq!(a, c, "declaration order changed the givens");
        // And the bytes are the declared facts.
        for (&(r, l), &cell) in &given_oracle(&w) {
            assert_eq!(a.lanes[r as usize] >> l & 1, 1);
            assert_eq!(a.cell(r as usize, l), cell);
        }
        let declared: u32 = given_oracle(&w).len() as u32;
        assert_eq!(
            a.lanes.iter().map(|l| l.count_ones()).sum::<u32>(),
            declared
        );
    }

    /// (b) A different value for a given's lane is refused and recorded.
    #[test]
    fn b_a_hypothesis_contradicting_a_given_is_refused_and_recorded() {
        let w = world(N, 3);
        let g = hydrate(&w);
        let row = (0..N).find(|&r| g.lanes[r].count_ones() >= 2).unwrap() as u32;
        let given_lane = g.lanes[row as usize].trailing_zeros() as usize;
        let free_lane = (!g.lanes[row as usize]).trailing_zeros() as usize;
        let lanes = (1u32 << given_lane) | (1 << free_lane);
        let offered = val(9, row, given_lane);
        assert_ne!(offered, g.cell(row as usize, given_lane));
        let mut cells = vec![offered, val(9, row, free_lane)];
        if free_lane < given_lane {
            cells.swap(0, 1);
        }
        let mut t = Tunnel::new(&g);
        let rep = t.write(9, row, lanes, &cells).unwrap();
        assert_eq!(rep.contradicted, 1 << given_lane);
        assert_eq!(rep.accepted, 1 << free_lane);
        assert_eq!(rep.agreed, 0);
        assert_eq!(
            t.contradictions,
            vec![Contradiction {
                row,
                thought: 9,
                lanes: 1 << given_lane,
                offered: vec![offered],
            }]
        );
        // The given is still what every read sees; the free lane merged.
        let mut out = [[0u8; 16]; LANES];
        let m = t.read(row as usize, u64::MAX, ALL_LANES, &mut out);
        assert_eq!(out[given_lane], g.cell(row as usize, given_lane));
        assert_eq!(out[free_lane], val(9, row, free_lane));
        assert_eq!(m, g.lanes[row as usize] | (1 << free_lane));
        // Without the givens selected, the refused value does not appear either.
        let m = t.read(row as usize, 1 << 9, ALL_LANES, &mut out);
        assert_eq!(m, 1 << free_lane);
        // Every read over a whole hypothesis load matches the oracle.
        let hyps = hypotheses(N, 7, 5);
        let t = load(&g, &hyps);
        assert!(!t.contradictions.is_empty(), "the fixture contradicts");
        for row in 0..N as u32 {
            for thoughts in [u64::MAX, 0b1010_1110, 1] {
                let want = read_oracle(&w, &hyps, row, thoughts, ALL_LANES);
                let m = t.read(row as usize, thoughts, ALL_LANES, &mut out);
                assert_eq!(
                    m.count_ones() as usize,
                    want.len(),
                    "row {row} {thoughts:#b}"
                );
                for (&l, &c) in &want {
                    assert_eq!(m >> l & 1, 1);
                    assert_eq!(out[l], c, "row {row} lane {l}");
                }
            }
        }
    }

    /// (b) twin: a load that never disagrees with a given stays silent, and
    /// agreement is not stored.
    #[test]
    fn b_twin_agreeing_and_disjoint_writes_raise_no_contradiction() {
        let w = world(N, 3);
        let g = hydrate(&w);
        let mut hyps = hypotheses(N, 7, 5);
        let mut agreeing = 0u32;
        for (_, r, l, c) in hyps.iter_mut() {
            for (i, lane) in BitIter(*l as u64).enumerate() {
                if g.lanes[*r as usize] >> lane & 1 == 1 {
                    c[i] = g.cell(*r as usize, lane);
                    agreeing += 1;
                }
            }
        }
        assert!(agreeing > 100, "the twin must hit given lanes");
        let mut t = Tunnel::new(&g);
        let mut agreed = 0;
        let mut accepted = 0;
        for (th, r, l, c) in &hyps {
            let rep = t.write(*th, *r, *l, c).unwrap();
            assert_eq!(rep.contradicted, 0);
            assert_eq!(rep.agreed | rep.accepted, *l);
            assert_eq!(rep.agreed & rep.accepted, 0);
            agreed += rep.agreed.count_ones();
            accepted += rep.accepted.count_ones();
        }
        assert!(t.contradictions.is_empty());
        assert_eq!(agreed, agreeing);
        assert_eq!(t.payload.len() as u32, accepted, "agreement is not stored");
        // Higher thought wins between hypotheses.
        let mut out = [[0u8; 16]; LANES];
        let mut shared = 0;
        for row in 0..N as u32 {
            let want = read_oracle(&w, &hyps, row, HYPOTHESES, ALL_LANES);
            let m = t.read(row as usize, HYPOTHESES, ALL_LANES, &mut out);
            assert_eq!(m.count_ones() as usize, want.len());
            for (&l, &c) in &want {
                assert_eq!(out[l], c);
            }
            let writers = hyps
                .iter()
                .filter(|h| h.1 == row && h.2 & !g.lanes[row as usize] != 0)
                .count();
            shared += usize::from(writers > 1);
        }
        assert!(shared > 0, "the fixture has lanes written by two thoughts");
    }

    /// Layer 0 has one writer.
    #[test]
    fn only_hydration_writes_layer_zero() {
        let g = hydrate(&world(N, 3));
        let mut t = Tunnel::new(&g);
        assert_eq!(t.write(0, 1, 1, &[[7u8; 16]]), Err(WriteError::GivensLayer));
        assert!(t.payload.is_empty());
    }

    /// (c) Counts exclude layer 0 when bit 0 is outside the selection.
    #[test]
    fn c_counts_ignore_layer_zero_outside_the_selection() {
        let w = world(N, 3);
        let g = hydrate(&w);
        let hyps = hypotheses(N, 7, 5);
        let t = load(&g, &hyps);
        let mut out = [[0u8; 16]; LANES];
        let mut checked = 0;
        for thoughts in [u64::MAX, HYPOTHESES, 1, 0b1000_0100, 0b1000_0101] {
            let want = (0..N as u32)
                .filter(|&r| !read_oracle(&w, &hyps, r, thoughts, ALL_LANES).is_empty())
                .count() as u64;
            assert_eq!(t.count(thoughts), want, "{thoughts:#b}");
            let by_read = (0..N)
                .filter(|&r| t.read(r, thoughts, ALL_LANES, &mut out) != 0)
                .count() as u64;
            assert_eq!(by_read, want);
            checked += 1;
        }
        assert_eq!(checked, 5);
        let only_given = (0..N)
            .filter(|&r| g.lanes[r] != 0 && t.count_row(r, HYPOTHESES) == 0)
            .count();
        assert!(only_given > 0, "rows carrying only givens exist");
        assert!(t.count(u64::MAX) > t.count(HYPOTHESES));
    }

    /// (d) Certification never exceeds the proofs hydration ran.
    #[test]
    fn d_certification_is_capped_by_the_executed_proof() {
        let w = world(N, 3);
        let g = hydrate(&w);
        let mut claimed_causes_resolved = 0;
        let mut seen = [0usize; 6];
        for rel in &w.relations {
            let raw = CausalEdge64(g.edge[rel.from as usize]).epistemic_raw5();
            let s = EpistemicState5::decode(Epi5Gen::V1, raw).unwrap();
            seen[s.certification().code() as usize] += 1;
            // Recompute the proofs here, independently of `proof_ceiling`.
            let member = (rel.from as usize) < w.n && (rel.to as usize) < w.n;
            let resolves = member
                && match rel.via {
                    Via::Named(i) => (i as usize) < w.n,
                    _ => true,
                };
            let earned = if resolves {
                Certification3::Related
            } else if member {
                Certification3::Associated
            } else {
                Certification3::Open
            };
            assert_eq!(s.certification(), earned, "{rel:?}");
            assert!(!s.asserts(fact::SUPPORTS));
            if rel.claimed == Certification3::Causes && resolves {
                claimed_causes_resolved += 1;
            }
            assert_eq!(s.topology(), topology(&w, rel));
        }
        assert!(
            claimed_causes_resolved > 10,
            "the ontology claims Causes on resolvable relations"
        );
        assert!(
            seen[0] > 0 && seen[1] > 0 && seen[2] > 0,
            "every earned level appears: {seen:?}"
        );
        // Only bits 59..63 are written; rows without a relation are Unknown x Open.
        let related: std::collections::HashSet<u32> = w.relations.iter().map(|r| r.from).collect();
        for (r, &e) in g.edge.iter().enumerate() {
            assert_eq!(e & !causal_edge::layout::EPISTEMIC_MASK, 0, "row {r}");
            if !related.contains(&(r as u32)) {
                let s = state_of(&w, None);
                assert_eq!(CausalEdge64(e).epistemic_raw5(), s.raw());
            }
        }
    }

    /// The registry: published at seal, shared afterwards, never built on read.
    #[test]
    fn the_registry_publishes_once_per_sealed_version() {
        let w = world(N, 3);
        let key = 0x0D0D_0000;
        assert!(sealed(key, 1).is_none(), "the read path does not hydrate");
        let a = seal(key, 1, &w);
        assert!(std::ptr::eq(a, sealed(key, 1).unwrap()));
        assert!(
            std::ptr::eq(a, seal(key, 1, &w)),
            "a second seal reuses the table"
        );
        assert_eq!(*a, hydrate(&w), "recomputable");
        let b = seal(key, 2, &w);
        assert!(!std::ptr::eq(a, b));
        assert!(sealed(key + 1, 1).is_none());
    }
}
