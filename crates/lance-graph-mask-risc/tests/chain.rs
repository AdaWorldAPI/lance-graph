//! `MaskOp::GatherChain` — chained hops folded into one pass — against the
//! row-at-a-time oracle AND against an independent row loop written here
//! from the fixture's own lanes, so neither crate path grades itself.
//!
//! Shape: `line.partner -> partner.country -> country`, with a test at each
//! depth (`partner.status == 1`, `country.region != 2`). No mask over
//! `partner` or `country` is built anywhere in the executed program.

use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::reference::reference_execute_into;
use lance_graph_mask_risc::{
    scratch_words_for, words_for, ExecError, Foreign, ForeignPlane, Hop, HopTest, LaneRef, MaskOp,
    Operand, Out, Planes, Pred, Program, Terminal, Value, MAX_CHAIN_HOPS, MAX_HOP_TESTS,
};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

const S0: Operand = Operand::Scratch(0);
const S1: Operand = Operand::Scratch(1);

/// Narrow tiles, so fixtures of a few thousand rows span several of them.
const TEST_TILE_WORDS: usize = 8;

const ROWS: [usize; 6] = [1, 63, 64, 65, 700, 3000];

struct Fixture {
    /// line → partner key (some out of range).
    partner_of: Vec<u32>,
    /// line's own status, for the gate.
    line_status: Vec<u32>,
    partner_rows: usize,
    partner_live: Vec<u64>,
    partner_status: Vec<u32>,
    /// partner → country key (some out of range).
    country_of: Vec<u32>,
    country_rows: usize,
    country_live: Vec<u64>,
    country_region: Vec<u32>,
}

fn live_bits(rows: usize, s: &mut u64) -> Vec<u64> {
    let mut w: Vec<u64> = (0..words_for(rows)).map(|_| lcg(s) | lcg(s)).collect();
    if !rows.is_multiple_of(64) {
        if let Some(last) = w.last_mut() {
            *last &= (1u64 << (rows % 64)) - 1;
        }
    }
    w
}

impl Fixture {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = seed ^ n as u64;
        let (partner_rows, country_rows) = (90usize, 12usize);
        let mut partner_of: Vec<u32> = (0..n).map(|_| (lcg(&mut s) % 95) as u32).collect();
        // A guaranteed out-of-range key at depth 1.
        if let Some(last) = partner_of.last_mut() {
            *last = 93;
        }
        let line_status = (0..n).map(|_| (lcg(&mut s) % 3) as u32).collect();
        let partner_status = (0..partner_rows)
            .map(|_| (lcg(&mut s) % 2) as u32)
            .collect();
        let mut country_of: Vec<u32> = (0..partner_rows)
            .map(|_| (lcg(&mut s) % 14) as u32)
            .collect();
        // A guaranteed out-of-range key at depth 2.
        country_of[0] = 13;
        let country_region = (0..country_rows)
            .map(|_| (lcg(&mut s) % 3) as u32)
            .collect();
        Self {
            partner_of,
            line_status,
            partner_rows,
            partner_live: live_bits(partner_rows, &mut s),
            partner_status,
            country_of,
            country_rows,
            country_live: live_bits(country_rows, &mut s),
            country_region,
        }
    }

    /// The query, written as a row loop over the fixture's own lanes.
    fn holds(&self, i: usize) -> bool {
        let bit = |w: &[u64], k: usize| (w[k / 64] >> (k % 64)) & 1 == 1;
        let p = self.partner_of[i] as usize;
        if p >= self.partner_rows || !bit(&self.partner_live, p) || self.partner_status[p] != 1 {
            return false;
        }
        let c = self.country_of[p] as usize;
        c < self.country_rows && bit(&self.country_live, c) && self.country_region[c] != 2
    }
}

fn chain() -> Vec<Hop> {
    vec![
        Hop {
            live: 0,
            tests: vec![HopTest {
                lane: 0,
                value: 1,
                negate: false,
            }],
            next: Some(1),
        },
        Hop {
            live: 1,
            tests: vec![HopTest {
                lane: 2,
                value: 2,
                negate: true,
            }],
            next: None,
        },
    ]
}

fn run(
    p: &Program,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
) -> (Result<Value, ExecError>, Result<Value, ExecError>) {
    let words = words_for(planes.n_rows).clamp(1, TEST_TILE_WORDS);
    let slots = p.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
    let mut scratch = Scratch::over(&mut buf, words, slots).expect("carves");
    (
        execute_into(p, planes, foreign, &mut scratch, Out::None),
        reference_execute_into(p, planes, foreign, Out::None),
    )
}

/// FAILS IF the executor or the oracle disagrees with the row loop on the
/// two-hop chain, gated or ungated, on any tile boundary.
/// Non-vacuous: the answer is neither empty nor every row, and both depths
/// reject something.
#[test]
fn a_two_hop_chain_matches_the_oracle_and_the_row_loop() {
    let mut kept = 0usize;
    let mut total = 0usize;
    for &n in &ROWS {
        let fx = Fixture::new(n, 0xC4A1);
        let lanes = [LaneRef::U32(&fx.partner_of), LaneRef::U32(&fx.line_status)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let fplanes = [
            ForeignPlane {
                words: &fx.partner_live,
                rows: fx.partner_rows,
            },
            ForeignPlane {
                words: &fx.country_live,
                rows: fx.country_rows,
            },
        ];
        let flanes = [
            LaneRef::U32(&fx.partner_status),
            LaneRef::U32(&fx.country_of),
            LaneRef::U32(&fx.country_region),
        ];
        let foreign = Foreign {
            planes: &fplanes,
            lanes: &flanes,
        };

        let want = (0..n).filter(|&i| fx.holds(i)).count();
        let ungated = Program::with_chains(
            vec![MaskOp::GatherChain {
                lane: 0,
                chain: 0,
                under: None,
                dst: 0,
            }],
            Terminal::Count { mask: S0 },
            vec![chain()],
        );
        let (got, oracle) = run(&ungated, &planes, &foreign);
        assert_eq!(got, Ok(Value::Count(want)), "n={n}: executor, ungated");
        assert_eq!(oracle, Ok(Value::Count(want)), "n={n}: oracle, ungated");

        // Gate: line_status == 0, then the chain only where it holds.
        let want_g = (0..n)
            .filter(|&i| fx.line_status[i] == 0 && fx.holds(i))
            .count();
        let gated = Program::with_chains(
            vec![
                MaskOp::Pred {
                    pred: Pred::EqU32 { lane: 1, v: 0 },
                    under: None,
                    dst: 0,
                },
                MaskOp::GatherChain {
                    lane: 0,
                    chain: 0,
                    under: Some(S0),
                    dst: 1,
                },
            ],
            Terminal::Count { mask: S1 },
            vec![chain()],
        );
        let (got, oracle) = run(&gated, &planes, &foreign);
        assert_eq!(got, Ok(Value::Count(want_g)), "n={n}: executor, gated");
        assert_eq!(oracle, Ok(Value::Count(want_g)), "n={n}: oracle, gated");

        kept += want;
        total += n;
    }
    assert!(
        kept > 0 && kept * 3 < total,
        "vacuous fixture: kept {kept} of {total}"
    );
}

/// FAILS IF a chain of one hop with no tests is not exactly `Gather`.
#[test]
fn one_bare_hop_is_gather() {
    let n = 700;
    let fx = Fixture::new(n, 7);
    let lanes = [LaneRef::U32(&fx.partner_of)];
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let fplanes = [ForeignPlane {
        words: &fx.partner_live,
        rows: fx.partner_rows,
    }];
    let foreign = Foreign {
        planes: &fplanes,
        lanes: &[],
    };
    let chained = Program::with_chains(
        vec![MaskOp::GatherChain {
            lane: 0,
            chain: 0,
            under: None,
            dst: 0,
        }],
        Terminal::Count { mask: S0 },
        vec![vec![Hop {
            live: 0,
            tests: vec![],
            next: None,
        }]],
    );
    let plain = Program::new(
        vec![MaskOp::Gather {
            lane: 0,
            foreign: 0,
            under: None,
            dst: 0,
        }],
        Terminal::Count { mask: S0 },
    );
    let (a, _) = run(&chained, &planes, &foreign);
    let (b, _) = run(&plain, &planes, &foreign);
    assert_eq!(a, b);
    assert!(matches!(a, Ok(Value::Count(c)) if c > 0));
}

/// FAILS IF a malformed chain reaches the facade (whose shape checks panic)
/// instead of being refused, identically, by both paths.
#[test]
fn malformed_chains_are_refused_by_both_paths() {
    let n = 65;
    let fx = Fixture::new(n, 3);
    let lanes = [LaneRef::U32(&fx.partner_of)];
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let fplanes = [
        ForeignPlane {
            words: &fx.partner_live,
            rows: fx.partner_rows,
        },
        ForeignPlane {
            words: &fx.country_live,
            rows: fx.country_rows,
        },
    ];
    let short = vec![0u32; 3];
    let flanes = [
        LaneRef::U32(&fx.partner_status),
        LaneRef::U32(&fx.country_of),
        LaneRef::U32(&fx.country_region),
        LaneRef::U32(&short),
    ];
    let foreign = Foreign {
        planes: &fplanes,
        lanes: &flanes,
    };
    let op = |chain| MaskOp::GatherChain {
        lane: 0,
        chain,
        under: None,
        dst: 0,
    };
    let refused = |chains: Vec<Vec<Hop>>, chain: u16, want: ExecError| {
        let p = Program::with_chains(vec![op(chain)], Terminal::Count { mask: S0 }, chains);
        let (got, oracle) = run(&p, &planes, &foreign);
        assert_eq!(got, Err(want), "executor");
        assert_eq!(oracle, Err(want), "oracle");
    };
    let hop = |live, next| Hop {
        live,
        tests: vec![],
        next,
    };

    refused(vec![chain()], 1, ExecError::ChainOutOfRange(1));
    refused(vec![vec![]], 0, ExecError::ChainShape { chain: 0, hop: 0 });
    // A hop before the last without its key, and a key after the last.
    refused(
        vec![vec![hop(0, None), hop(1, None)]],
        0,
        ExecError::ChainShape { chain: 0, hop: 0 },
    );
    refused(
        vec![vec![hop(0, Some(1)), hop(1, Some(1))]],
        0,
        ExecError::ChainShape { chain: 0, hop: 1 },
    );
    refused(
        vec![vec![hop(0, None); MAX_CHAIN_HOPS + 1]],
        0,
        ExecError::ChainShape { chain: 0, hop: 0 },
    );
    refused(
        vec![vec![Hop {
            live: 0,
            tests: vec![
                HopTest {
                    lane: 0,
                    value: 0,
                    negate: false
                };
                MAX_HOP_TESTS + 1
            ],
            next: None,
        }]],
        0,
        ExecError::ChainShape { chain: 0, hop: 0 },
    );
    refused(vec![vec![hop(2, None)]], 0, ExecError::ForeignOutOfRange(2));
    // A test lane shorter than its hop table.
    refused(
        vec![vec![Hop {
            live: 0,
            tests: vec![HopTest {
                lane: 3,
                value: 0,
                negate: false,
            }],
            next: None,
        }]],
        0,
        ExecError::LenMismatch {
            what: "chain lane",
            expected: fx.partner_rows,
            found: 3,
        },
    );
}
