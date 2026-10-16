//! Cartesian addressing of grammar: Haspelmath's semantic maps as two
//! presence planes, checked through quack and mask-risc.
//!
//! Source: M. Haspelmath, "The Geometry of Grammatical Meaning: Semantic Maps
//! and Cross-Linguistic Comparison", in Tomasello (ed.), *The New Psychology
//! of Language*, vol. 2, 2003, pp. 211–242.
//!
//! A semantic map is a graph over grammatical FUNCTIONS (direction,
//! recipient, purpose, …). Its one testable claim: every gram of every
//! language (English `to`, French `à`, …) covers a CONTIGUOUS region of the
//! map. A gram that covers a disconnected set falsifies the map.
//!
//! # Layout
//!
//! - Every function has one coordinate; every gram has one coordinate. Names
//!   are labels, used only to build the planes and print results.
//! - `G` (gram × function): bit `g · S + f` = gram `g` is used in function `f`.
//!   A parsed occurrence — an AST node of gram `g` in function `f` — sets
//!   exactly that bit; the paper's numbered examples are such occurrences.
//! - `E` (function × function): bit `h · S + f` = the map links `h` and `f`
//!   (both directions; the map is undirected).
//! - Contiguity of gram `g`: seed the lowest function of `g`'s row, then
//!   repeatedly add every function `h` of the row with
//!   `Any(And(Plane(E), Plane(R_rep)))` over row `h`, where `R` is the set
//!   reached so far, tiled into every row segment. Contiguous iff the fixpoint
//!   equals the gram's row. Every compared pair sits at one coordinate.
//!
//! ```text
//! cargo run -p lance-graph-quack --example semantic_map_probe
//! ```

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{words_for, Foreign, Out, Planes, Program, Value};
use lance_graph_quack::{lower, Agg, Filter, Mask, Query};

fn set(words: &mut [u64], i: usize) {
    words[i / 64] |= 1 << (i % 64);
}

fn get(words: &[u64], i: usize) -> bool {
    words[i / 64] >> (i % 64) & 1 == 1
}

/// One semantic map: function labels and its links.
struct Map {
    name: &'static str,
    functions: &'static [&'static str],
    links: &'static [(&'static str, &'static str)],
}

/// A gram: its language-particular label and the functions it covers.
struct Gram {
    label: &'static str,
    functions: &'static [&'static str],
}

impl Map {
    fn coord(&self, f: &str) -> usize {
        self.functions
            .iter()
            .position(|x| *x == f)
            .unwrap_or_else(|| panic!("{}: unknown function {f:?}", self.name))
    }
}

/// The two planes for one map and a set of grams.
struct Space {
    n: usize,
    w: usize,
    stride: usize,
    e: Vec<u64>,
    g: Vec<u64>,
    program: Program,
}

impl Space {
    fn new(map: &Map, grams: &[Gram]) -> Self {
        let n = map.functions.len();
        let w = words_for(n);
        let stride = w * 64;
        let mut e = vec![0u64; words_for(n * stride)];
        for &(a, b) in map.links {
            let (a, b) = (map.coord(a), map.coord(b));
            set(&mut e, a * stride + b);
            set(&mut e, b * stride + a);
        }
        let mut g = vec![0u64; words_for(grams.len() * stride).max(1)];
        for (gi, gram) in grams.iter().enumerate() {
            for f in gram.functions {
                set(&mut g, gi * stride + map.coord(f));
            }
        }
        let program = lower(&Query {
            filter: Filter::And(vec![Filter::Plane(Mask(0)), Filter::Plane(Mask(1))]),
            agg: Agg::Any,
        })
        .expect("lowers");
        Space {
            n,
            w,
            stride,
            e,
            g,
            program,
        }
    }

    /// The gram's row, `w` words.
    fn row(&self, gi: usize) -> Vec<u64> {
        self.g[gi * self.w..(gi + 1) * self.w].to_vec()
    }

    /// Reachable part of the gram's row from its lowest function, staying
    /// inside the row. `steps = None` runs to the fixpoint.
    fn reach(&self, gi: usize, steps: Option<usize>) -> Vec<u64> {
        let row = self.row(gi);
        let mut r = vec![0u64; self.w];
        if let Some(first) = (0..self.n).find(|&f| get(&row, f)) {
            set(&mut r, first);
        }
        let n_rows = self.n * self.stride;
        let mut r_rep = vec![0u64; words_for(n_rows)];
        let mut done = 0;
        loop {
            if steps.is_some_and(|k| done >= k) {
                break;
            }
            for h in 0..self.n {
                r_rep[h * self.w..(h + 1) * self.w].copy_from_slice(&r);
            }
            let masks: [&[u64]; 2] = [&self.e, &r_rep];
            let planes = Planes {
                n_rows,
                masks: &masks,
                lanes: &[],
            };
            let mut next = r.clone();
            for h in 0..self.n {
                if !get(&row, h) || get(&r, h) {
                    continue;
                }
                let mut s = Scratch::for_program(&self.program, n_rows).expect("scratch");
                let v = execute_extent(
                    &self.program,
                    &planes,
                    &Foreign::NONE,
                    &mut s,
                    Out::None,
                    h * self.stride..(h + 1) * self.stride,
                )
                .expect("runs");
                if matches!(v, Value::Bool(true)) {
                    set(&mut next, h);
                }
            }
            if next == r {
                break;
            }
            r = next;
            done += 1;
        }
        r
    }

    fn contiguous(&self, gi: usize) -> bool {
        self.reach(gi, None) == self.row(gi)
    }
}

/// Returns the labels of the grams that are NOT contiguous on the map.
fn check(map: &Map, grams: &[Gram]) -> Vec<&'static str> {
    let s = Space::new(map, grams);
    grams
        .iter()
        .enumerate()
        .filter(|(gi, _)| !s.contiguous(*gi))
        .map(|(_, g)| g.label)
        .collect()
}

// ── Fig. 8.1: typical dative functions ────────────────────────────────────

const DATIVE: Map = Map {
    name: "dative (Fig. 8.1)",
    functions: &[
        "direction",
        "purpose",
        "recipient",
        "experiencer",
        "predicative possessor",
        "external possessor",
        "beneficiary",
        "judicantis",
    ],
    links: &[
        ("direction", "recipient"),
        ("direction", "purpose"),
        ("recipient", "experiencer"),
        ("recipient", "predicative possessor"),
        ("recipient", "beneficiary"),
        ("predicative possessor", "external possessor"),
        ("beneficiary", "external possessor"),
        ("beneficiary", "judicantis"),
    ],
};

const DATIVE_GRAMS: &[Gram] = &[
    // (2a–d), Fig. 8.1
    Gram {
        label: "English to",
        functions: &["direction", "recipient", "experiencer", "purpose"],
    },
    // Fig. 8.2: no purpose, but predicative possession (Ce chien est à moi)
    Gram {
        label: "French à",
        functions: &[
            "direction",
            "recipient",
            "experiencer",
            "predicative possessor",
        ],
    },
    // Fig. 8.5 and its text: no direction, no predicative possession
    Gram {
        label: "French dative clitics",
        functions: &[
            "recipient",
            "experiencer",
            "beneficiary",
            "external possessor",
            "judicantis",
        ],
    },
];

// ── Fig. 8.6: indefinite pronoun functions ───────────────────────────────

const INDEFINITE: Map = Map {
    name: "indefinites (Fig. 8.6)",
    functions: &[
        "specific known",
        "specific unknown",
        "irrealis non-specific",
        "question",
        "conditional",
        "indirect negation",
        "comparative",
        "direct negation",
        "free choice",
    ],
    links: &[
        ("specific known", "specific unknown"),
        ("specific unknown", "irrealis non-specific"),
        ("irrealis non-specific", "question"),
        ("irrealis non-specific", "conditional"),
        ("question", "indirect negation"),
        ("conditional", "comparative"),
        ("indirect negation", "comparative"),
        ("indirect negation", "direct negation"),
        ("comparative", "free choice"),
    ],
};

const INDEFINITE_GRAMS: &[Gram] = &[
    // Fig. 8.7
    Gram {
        label: "English some-",
        functions: &[
            "specific known",
            "specific unknown",
            "irrealis non-specific",
            "question",
            "conditional",
        ],
    },
    Gram {
        label: "English any-",
        functions: &[
            "question",
            "conditional",
            "indirect negation",
            "comparative",
            "direct negation",
            "free choice",
        ],
    },
    // Fig. 8.8 and (8)–(11): -nibud' in irrealis, questions, conditionals;
    // not in negation, not free choice
    Gram {
        label: "Russian -nibud'",
        functions: &["irrealis non-specific", "question", "conditional"],
    },
];

// ── Fig. 8.9: reflexive and middle functions ─────────────────────────────

const MIDDLE: Map = Map {
    name: "reflexive/middle (Fig. 8.9)",
    functions: &[
        "full reflexive",
        "grooming/body motion",
        "naturally reciprocal",
        "anticausative",
        "deobjective",
        "potential passive",
        "passive",
    ],
    links: &[
        ("full reflexive", "grooming/body motion"),
        ("grooming/body motion", "naturally reciprocal"),
        ("grooming/body motion", "anticausative"),
        ("anticausative", "deobjective"),
        ("anticausative", "potential passive"),
        ("potential passive", "passive"),
    ],
};

const MIDDLE_GRAMS: &[Gram] = &[
    // Fig. 8.10, (13)
    Gram {
        label: "French se",
        functions: &[
            "full reflexive",
            "grooming/body motion",
            "naturally reciprocal",
            "anticausative",
            "potential passive",
        ],
    },
    // Fig. 8.10, (14)–(15)
    Gram {
        label: "Russian -sja",
        functions: &[
            "grooming/body motion",
            "naturally reciprocal",
            "anticausative",
            "deobjective",
            "potential passive",
            "passive",
        ],
    },
    // Fig. 8.11
    Gram {
        label: "Turkish -In",
        functions: &["grooming/body motion"],
    },
    Gram {
        label: "Turkish -Iş",
        functions: &["naturally reciprocal"],
    },
    Gram {
        label: "Turkish -Il",
        functions: &["anticausative", "potential passive", "passive"],
    },
];

// ── Fig. 8.12: instrumental and related functions ────────────────────────

const INSTRUMENTAL: Map = Map {
    name: "instrumental (Fig. 8.12)",
    functions: &[
        "recipient",
        "beneficiary",
        "cause",
        "co-agent",
        "conjunctive",
        "comitative",
        "instrumental",
        "passive",
        "source",
    ],
    links: &[
        ("recipient", "beneficiary"),
        ("beneficiary", "cause"),
        ("recipient", "co-agent"),
        ("co-agent", "comitative"),
        ("conjunctive", "comitative"),
        ("comitative", "instrumental"),
        ("instrumental", "passive"),
        ("passive", "source"),
        ("cause", "passive"),
        ("cause", "source"),
    ],
};

const INSTRUMENTAL_GRAMS: &[Gram] = &[
    // Fig. 8.13
    Gram {
        label: "English with",
        functions: &["co-agent", "comitative", "instrumental"],
    },
    Gram {
        label: "French par",
        functions: &["passive", "cause"],
    },
    Gram {
        label: "German aus",
        functions: &["source", "cause"],
    },
    Gram {
        label: "German von",
        functions: &["passive", "source"],
    },
    // Fig. 8.14, (21a–g)
    Gram {
        label: "Seychelles Creole ek",
        functions: &[
            "instrumental",
            "comitative",
            "conjunctive",
            "passive",
            "source",
            "cause",
            "recipient",
            "co-agent",
        ],
    },
    // Fig. 8.15
    Gram {
        label: "Russian instrumental",
        functions: &["instrumental", "passive"],
    },
    Gram {
        label: "Russian s",
        functions: &["co-agent", "comitative", "conjunctive"],
    },
];

// ── (4): choosing between three arrangements of one sub-map ──────────────

const SUB_FUNCTIONS: &[&str] = &["direction", "recipient", "purpose"];

const ARRANGE_A: Map = Map {
    name: "(4a) purpose – direction – recipient",
    functions: SUB_FUNCTIONS,
    links: &[("purpose", "direction"), ("direction", "recipient")],
};
const ARRANGE_B: Map = Map {
    name: "(4b) direction – purpose – recipient",
    functions: SUB_FUNCTIONS,
    links: &[("direction", "purpose"), ("purpose", "recipient")],
};
const ARRANGE_C: Map = Map {
    name: "(4c) direction – recipient – purpose",
    functions: SUB_FUNCTIONS,
    links: &[("direction", "recipient"), ("recipient", "purpose")],
};
/// Fig. 8.4: the vacuous map, every function linked to every other.
const VACUOUS: Map = Map {
    name: "Fig. 8.4 vacuous",
    functions: SUB_FUNCTIONS,
    links: &[
        ("direction", "recipient"),
        ("direction", "purpose"),
        ("recipient", "purpose"),
    ],
};

const SUB_GRAMS: &[Gram] = &[
    Gram {
        label: "English to",
        functions: &["direction", "recipient", "purpose"],
    },
    // à Adam, à Leipzig; not purpose
    Gram {
        label: "French à",
        functions: &["direction", "recipient"],
    },
    // Anna ging zum Spielen; Ich gehe zu Anna; not recipient
    Gram {
        label: "German zu",
        functions: &["direction", "purpose"],
    },
];

// ── (23): the Turkish old present as a doughnut ──────────────────────────

const TENSE: Map = Map {
    name: "(23) habitual – progressive – future",
    functions: &["habitual", "progressive", "future"],
    links: &[("habitual", "progressive"), ("progressive", "future")],
};

const TENSE_GRAMS: &[Gram] = &[
    Gram {
        label: "German Present",
        functions: &["habitual", "progressive", "future"],
    },
    Gram {
        label: "English Progressive",
        functions: &["progressive", "future"],
    },
    // ousted from progressive by -Iyor: the hole in the doughnut
    Gram {
        label: "Turkish old present",
        functions: &["habitual", "future"],
    },
];

fn main() {
    let tier = if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else {
        "not v4"
    };
    println!("tier: {tier}");

    // 1. Every gram the chapter draws lies in one contiguous region.
    for (map, grams) in [
        (&DATIVE, DATIVE_GRAMS),
        (&INDEFINITE, INDEFINITE_GRAMS),
        (&MIDDLE, MIDDLE_GRAMS),
        (&INSTRUMENTAL, INSTRUMENTAL_GRAMS),
    ] {
        let broken = check(map, grams);
        println!(
            "{}: {} grams, non-contiguous {broken:?}",
            map.name,
            grams.len()
        );
        assert!(broken.is_empty(), "{} is falsified by {broken:?}", map.name);
    }

    // 2. The arrangement argument of (4): French and German eliminate (4b)
    //    and (4c); (4a) survives; the vacuous map admits everything.
    let mut survivors = vec![];
    for map in [&ARRANGE_A, &ARRANGE_B, &ARRANGE_C, &VACUOUS] {
        let broken = check(map, SUB_GRAMS);
        println!("{}: non-contiguous {broken:?}", map.name);
        if broken.is_empty() {
            survivors.push(map.name);
        }
    }
    assert_eq!(survivors, [ARRANGE_A.name, VACUOUS.name]);
    assert_eq!(check(&ARRANGE_B, SUB_GRAMS), ["French à"]);
    assert_eq!(check(&ARRANGE_C, SUB_GRAMS), ["German zu"]);

    // 3. A gram with a hole is caught: the Turkish old present.
    let broken = check(&TENSE, TENSE_GRAMS);
    println!("{}: non-contiguous {broken:?}", TENSE.name);
    assert_eq!(broken, ["Turkish old present"]);

    // 4. Disable: with no closure steps only the seed is reached, so every
    //    gram that covers more than one function must read as broken.
    let s = Space::new(&DATIVE, DATIVE_GRAMS);
    let broken0 = (0..DATIVE_GRAMS.len())
        .filter(|&gi| s.reach(gi, Some(0)) != s.row(gi))
        .count();
    println!(
        "disable (0 steps) on {}: {broken0} of {} broken",
        DATIVE.name,
        DATIVE_GRAMS.len()
    );
    assert_eq!(broken0, DATIVE_GRAMS.len());
}
