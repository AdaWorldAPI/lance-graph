//! D-MIST-NATIVE-0: Mistaber's Kripke holding rules as a least fixpoint over
//! the existing DR64 Boolean relation register.
//!
//! Plan: `.claude/plans/d-mistaber-native-v1.md`. Reference answers: Clingo
//! golden files exported by `AdaWorldAPI/mistaber-rs` (`goldens/SOURCE`).
//!
//! # The fixpoint
//!
//! Upstream `kripke_rules.lp` makes `holds` recursive through
//! `holds_via(inherited(Parent), P, Child)`, and every negation in that rule
//! is over INPUT relations (`override`, `asserts`, `endorses` at the child).
//! The program is therefore stratified and its answer is the least fixpoint
//!
//! ```text
//! H₀   = (asserts ∪ endorses) ∧ world
//! Hₖ₊₁ = H₀ ∪ ((Hₖ ∘ children) ∧ ¬Block)        Block = asserts ∪ endorses ∪ overridden
//! ```
//!
//! with `H` a proposition × world relation and `children` the world × world
//! relation `parent → child`. Each step is one
//! [`compose`](dr_math_binding_probe::relcomp::compose) plus an AND-NOT: no new
//! algebra. The two lineages (`holds_via_rule`'s source rule and
//! `inheritance_source`'s origin) are the same fixpoint gated by the final
//! `H`, one per rule or origin.
//!
//! # The holdings lens
//!
//! Holdings are read through one lens whose fields are [`HOLDINGS_FIELDS`].
//! A predicate is a [`WideFieldMask`] selection over it: `holds/2` selects
//! proposition and world, `holds_via/3` adds the mode, `holds_via_rule/5`
//! adds both rules. Selecting fewer fields can make two lens rows equal, so
//! the projection is [`Projection::Set`] (Datalog: one fact) or
//! [`Projection::Bag`] (one row per lens row). Only `Set` is Mistaber's
//! semantics; `Bag` exists to show where an alias-only reading goes wrong.
//!
//! # Not covered, refused
//!
//! Anything that is not one fixpoint of facts: a fact kind outside the Kripke
//! inputs, more than 64 worlds, propositions or rules (one register), and a
//! selection that is not one of the three predicates.
#![forbid(unsafe_code)]

use dr_math_binding_probe::relcomp::{Rows, Work, compose};
use lance_graph_contract::class_view::WideFieldMask;

/// The slot count of one register axis.
pub const SLOTS: usize = 64;

/// The holdings lens's fields, by position.
pub const HOLDINGS_FIELDS: [&str; 5] = [
    "proposition",
    "world",
    "mode",
    "immediate_rule",
    "source_rule",
];
const F_PROP: u8 = 0;
const F_WORLD: u8 = 1;
const F_MODE: u8 = 2;
const F_IMMEDIATE: u8 = 3;
const F_SOURCE: u8 = 4;

/// The immediate rule upstream attributes every inherited holding to.
const INHERIT_RULE: &str = "r_kripke_inherit";

/// Why an input or a selection is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A fact kind the Kripke rules do not read.
    UnknownFact(String),
    /// A known fact kind with the wrong number of arguments.
    Arity(String),
    /// More than [`SLOTS`] distinct values on one axis.
    TooMany(&'static str),
    /// A golden line that is neither header, fact nor atom.
    Format(String),
    /// A field selection that is not `holds/2`, `holds_via/3` or
    /// `holds_via_rule/5`.
    Selection,
}

/// How a narrower selection treats rows that become equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// One row per distinct value of the selected fields (Datalog).
    Set,
    /// One row per lens row.
    Bag,
}

#[derive(Default)]
struct Interner(Vec<String>);

impl Interner {
    fn id(&mut self, name: &str, axis: &'static str) -> Result<usize, Refusal> {
        if let Some(i) = self.0.iter().position(|n| n == name) {
            return Ok(i);
        }
        if self.0.len() == SLOTS {
            return Err(Refusal::TooMany(axis));
        }
        self.0.push(name.to_owned());
        Ok(self.0.len() - 1)
    }
}

/// A golden file: its fixture id, input facts and expected atoms.
pub struct Golden {
    pub fixture_id: String,
    pub facts: Vec<Vec<String>>,
    pub atoms: Vec<String>,
}

/// Parse one golden TSV.
pub fn parse_golden(text: &str) -> Result<Golden, Refusal> {
    let mut g = Golden {
        fixture_id: String::new(),
        facts: Vec::new(),
        atoms: Vec::new(),
    };
    for line in text.lines() {
        let mut cols = line.split('\t');
        match cols.next() {
            Some("# fixture") => g.fixture_id = cols.next().unwrap_or_default().to_owned(),
            Some(h) if h.starts_with('#') => {}
            Some("fact") => g.facts.push(cols.map(str::to_owned).collect()),
            Some("atom") => g.atoms.push(cols.collect::<Vec<_>>().join("\t")),
            _ => return Err(Refusal::Format(line.to_owned())),
        }
    }
    Ok(g)
}

fn or(a: &Rows, b: &Rows) -> Rows {
    std::array::from_fn(|i| a[i] | b[i])
}

fn and(a: &Rows, b: &Rows) -> Rows {
    std::array::from_fn(|i| a[i] & b[i])
}

fn and_not(a: &Rows, b: &Rows) -> Rows {
    std::array::from_fn(|i| a[i] & !b[i])
}

fn and_row(a: &Rows, m: u64) -> Rows {
    std::array::from_fn(|i| a[i] & m)
}

fn bits(mut x: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (x != 0).then(|| {
            let b = x.trailing_zeros() as usize;
            x &= x - 1;
            b
        })
    })
}

/// The Kripke inputs, one register per relation.
pub struct Kb {
    worlds: Interner,
    props: Interner,
    rules: Interner,
    world_set: u64,
    asserts: Rows,
    endorses: Rows,
    overridden: Rows,
    /// Row `parent`, bit `child`: `accessible(child, parent)`.
    children: Rows,
    asserts_rule: Vec<Rows>,
    endorses_rule: Vec<Rows>,
}

impl Kb {
    /// Load fact tuples (`kind, args…`). Refuses any fact kind the Kripke
    /// rules do not read.
    pub fn load(facts: &[Vec<String>]) -> Result<Self, Refusal> {
        let mut kb = Kb {
            worlds: Interner::default(),
            props: Interner::default(),
            rules: Interner::default(),
            world_set: 0,
            asserts: [0; SLOTS],
            endorses: [0; SLOTS],
            overridden: [0; SLOTS],
            children: [0; SLOTS],
            asserts_rule: Vec::new(),
            endorses_rule: Vec::new(),
        };
        for f in facts {
            let (kind, args) = f.split_first().ok_or(Refusal::Format(String::new()))?;
            let want = match kind.as_str() {
                "world" => 1,
                "asserts" | "endorses" | "accessible" => 2,
                "override" | "asserts_rule" | "endorses_rule" => 3,
                _ => return Err(Refusal::UnknownFact(kind.clone())),
            };
            if args.len() != want {
                return Err(Refusal::Arity(kind.clone()));
            }
            match kind.as_str() {
                "world" => kb.world_set |= 1 << kb.worlds.id(&args[0], "world")?,
                "accessible" => {
                    let child = kb.worlds.id(&args[0], "world")?;
                    let parent = kb.worlds.id(&args[1], "world")?;
                    kb.children[parent] |= 1 << child;
                }
                _ => {
                    let w = kb.worlds.id(&args[0], "world")?;
                    let p = kb.props.id(&args[1], "proposition")?;
                    match kind.as_str() {
                        "asserts" => kb.asserts[p] |= 1 << w,
                        "endorses" => kb.endorses[p] |= 1 << w,
                        "override" => kb.overridden[p] |= 1 << w,
                        _ => {
                            let r = kb.rules.id(&args[2], "rule")?;
                            for v in [&mut kb.asserts_rule, &mut kb.endorses_rule] {
                                v.resize(kb.rules.0.len(), [0; SLOTS]);
                            }
                            let table = if kind == "asserts_rule" {
                                &mut kb.asserts_rule
                            } else {
                                &mut kb.endorses_rule
                            };
                            table[r][p] |= 1 << w;
                        }
                    }
                }
            }
        }
        Ok(kb)
    }

    /// Solve: the holds fixpoint, then both lineages.
    pub fn solve(&self) -> Holdings<'_> {
        let mut work = Work::default();
        let block = or(&or(&self.asserts, &self.endorses), &self.overridden);
        let direct = and_row(&or(&self.asserts, &self.endorses), self.world_set);
        let (holds, steps) = lfp(&direct, None, &self.children, &block, &mut work);
        let lineage =
            |base: &Rows, work: &mut Work| lfp(base, Some(&holds), &self.children, &block, work).0;
        let sources = (0..self.rules.0.len())
            .map(|r| {
                let base = and_row(
                    &or(&self.asserts_rule[r], &self.endorses_rule[r]),
                    self.world_set,
                );
                lineage(&base, &mut work)
            })
            .collect();
        let origins = (0..self.worlds.0.len())
            .map(|o| lineage(&and_row(&direct, 1 << o), &mut work))
            .collect();
        Holdings {
            kb: self,
            block,
            direct,
            holds,
            sources,
            origins,
            steps,
            work,
        }
    }
}

/// `X = base ∪ ((gate(X) ∘ children) ∧ ¬block)` to its least fixpoint.
/// `gate = None` gates by `X` itself; `Some(h)` gates by `X ∧ h`.
fn lfp(
    base: &Rows,
    gate: Option<&Rows>,
    children: &Rows,
    block: &Rows,
    work: &mut Work,
) -> (Rows, usize) {
    let mut cur = *base;
    for step in 1..=SLOTS + 1 {
        let from = gate.map_or(cur, |h| and(&cur, h));
        let next = or(base, &and_not(&compose(&from, children, work), block));
        if next == cur {
            return (cur, step);
        }
        cur = next;
    }
    unreachable!(
        "each step extends every proposition by one hop; 64 worlds have no path longer than 63 hops"
    )
}

/// One holding mode: its cells, and per rule the cells that rule supports.
struct Mode {
    label: String,
    cells: Rows,
    rules: Vec<Rows>,
}

impl Mode {
    /// Lens rows at `(p, w)`: one per supporting rule, one if the holding has
    /// none, zero if the mode does not hold there.
    fn rows(&self, p: usize, w: usize) -> usize {
        if self.cells[p] >> w & 1 == 0 {
            return 0;
        }
        self.rules
            .iter()
            .filter(|r| r[p] >> w & 1 == 1)
            .count()
            .max(1)
    }
}

/// The solved holdings lens.
pub struct Holdings<'a> {
    kb: &'a Kb,
    block: Rows,
    direct: Rows,
    holds: Rows,
    /// Per rule: the (proposition, world) cells whose holding traces to it.
    sources: Vec<Rows>,
    /// Per origin world: the cells whose holding traces to it.
    origins: Vec<Rows>,
    /// Fixpoint steps of `holds`, the last one confirming no change.
    pub steps: usize,
    /// Row ORs the compose kernel performed.
    pub work: Work,
}

impl Holdings<'_> {
    fn w(&self, i: usize) -> &str {
        &self.kb.worlds.0[i]
    }

    fn p(&self, i: usize) -> &str {
        &self.kb.props.0[i]
    }

    /// `holds_via(inherited(parent), p, ·)` as a world mask.
    fn inherited_from(&self, parent: usize, p: usize) -> u64 {
        if self.holds[p] >> parent & 1 == 0 {
            return 0;
        }
        self.kb.children[parent] & !self.block[p]
    }

    /// Every mode with its cells and, per rule, the cells that rule supports
    /// under that mode.
    fn modes(&self) -> Vec<Mode> {
        let kb = self.kb;
        let n = kb.props.0.len();
        let direct = |table: &Rows, rules: &[Rows]| Mode {
            cells: and_row(table, kb.world_set),
            rules: rules
                .iter()
                .map(|r| and(r, &and_row(table, kb.world_set)))
                .collect(),
            label: String::new(),
        };
        let mut out = vec![
            Mode {
                label: "asserted".to_owned(),
                ..direct(&kb.asserts, &kb.asserts_rule)
            },
            Mode {
                label: "endorsed".to_owned(),
                ..direct(&kb.endorses, &kb.endorses_rule)
            },
        ];
        for par in 0..kb.worlds.0.len() {
            let cells: Rows = std::array::from_fn(|p| {
                if p < n {
                    self.inherited_from(par, p)
                } else {
                    0
                }
            });
            let rules = self
                .sources
                .iter()
                .map(|src| {
                    std::array::from_fn(|p| if src[p] >> par & 1 == 1 { cells[p] } else { 0 })
                })
                .collect();
            out.push(Mode {
                label: format!("inherited({})", self.w(par)),
                cells,
                rules,
            });
        }
        out
    }

    /// How many lens rows hold `p` in `w`: the multiplicity a bag projection
    /// to `holds/2` reports. A lens row is one (mode, source rule); a holding
    /// with no rule is one row with empty rule fields.
    pub fn bag_multiplicity(&self, prop: &str, world: &str) -> usize {
        let (Some(p), Some(w)) = (
            self.kb.props.0.iter().position(|n| n == prop),
            self.kb.worlds.0.iter().position(|n| n == world),
        ) else {
            return 0;
        };
        self.modes().iter().map(|m| m.rows(p, w)).sum()
    }

    /// Materialize one predicate as atoms: a [`WideFieldMask`] selection over
    /// [`HOLDINGS_FIELDS`]. For the test oracle only.
    pub fn materialize_selection(
        &self,
        sel: &WideFieldMask,
        proj: Projection,
    ) -> Result<Vec<String>, Refusal> {
        let has = |f| sel.has(f);
        let base = has(F_PROP) && has(F_WORLD);
        let mut out = Vec::new();
        match (
            base,
            has(F_MODE),
            has(F_IMMEDIATE),
            has(F_SOURCE),
            sel.count(),
        ) {
            (true, false, false, false, 2) => match proj {
                Projection::Set => {
                    for p in 0..self.kb.props.0.len() {
                        for w in bits(self.holds[p]) {
                            out.push(format!("holds({},{})", self.p(p), self.w(w)));
                        }
                    }
                }
                Projection::Bag => {
                    for m in self.modes() {
                        for (p, &row) in m.cells.iter().enumerate().take(self.kb.props.0.len()) {
                            for w in bits(row) {
                                for _ in 0..m.rows(p, w) {
                                    out.push(format!("holds({},{})", self.p(p), self.w(w)));
                                }
                            }
                        }
                    }
                }
            },
            (true, true, false, false, 3) => {
                for m in self.modes() {
                    for (p, &row) in m.cells.iter().enumerate().take(self.kb.props.0.len()) {
                        for w in bits(row) {
                            let copies = match proj {
                                Projection::Set => 1,
                                Projection::Bag => m.rows(p, w),
                            };
                            for _ in 0..copies {
                                out.push(format!(
                                    "holds_via({},{},{})",
                                    m.label,
                                    self.p(p),
                                    self.w(w)
                                ));
                            }
                        }
                    }
                }
            }
            (true, true, true, true, 5) => self.rule_atoms(&mut out),
            _ => return Err(Refusal::Selection),
        }
        Ok(out)
    }

    fn rule_atoms(&self, out: &mut Vec<String>) {
        let kb = self.kb;
        for (r, rule) in kb.rules.0.iter().enumerate() {
            for (mode, table) in [
                ("asserted", &kb.asserts_rule[r]),
                ("endorsed", &kb.endorses_rule[r]),
            ] {
                for (p, &row) in table.iter().enumerate().take(kb.props.0.len()) {
                    for w in bits(row & kb.world_set) {
                        out.push(format!(
                            "holds_via_rule({mode},{},{},{rule},{rule})",
                            self.p(p),
                            self.w(w)
                        ));
                    }
                }
            }
            for par in 0..kb.worlds.0.len() {
                for p in 0..kb.props.0.len() {
                    if self.sources[r][p] >> par & 1 == 1 {
                        for w in bits(self.inherited_from(par, p)) {
                            out.push(format!(
                                "holds_via_rule(inherited({}),{},{},{INHERIT_RULE},{rule})",
                                self.w(par),
                                self.p(p),
                                self.w(w)
                            ));
                        }
                    }
                }
            }
        }
    }

    /// Every Kripke atom the golden files carry, sorted. `proj` applies to
    /// `holds/2`, the only selection where rows can coincide.
    pub fn materialize_atoms(&self, proj: Projection) -> Vec<String> {
        let sel = |f: &[u8]| WideFieldMask::from_positions(f);
        let mut out = Vec::new();
        for (s, pr) in [
            (sel(&[F_PROP, F_WORLD]), proj),
            (sel(&[F_PROP, F_WORLD, F_MODE]), Projection::Set),
            (
                sel(&[F_PROP, F_WORLD, F_MODE, F_IMMEDIATE, F_SOURCE]),
                Projection::Set,
            ),
        ] {
            out.extend(
                self.materialize_selection(&s, pr)
                    .expect("a supported selection"),
            );
        }
        let kb = self.kb;
        for p in 0..kb.props.0.len() {
            let inherited: u64 = (0..kb.worlds.0.len())
                .map(|par| self.inherited_from(par, p))
                .fold(0, |a, b| a | b);
            let moderate: u64 = bits(self.direct[p])
                .map(|par| self.inherited_from(par, p))
                .fold(0, |a, b| a | b);
            let strong = self.direct[p];
            for (level, m) in [
                ("strong", strong),
                ("moderate", moderate),
                ("weak", inherited & !strong & !moderate),
            ] {
                for w in bits(m) {
                    out.push(format!(
                        "holding_strength({level},{},{})",
                        self.p(p),
                        self.w(w)
                    ));
                }
            }
            for (o, cells) in self.origins.iter().enumerate() {
                for w in bits(cells[p]) {
                    out.push(format!(
                        "inheritance_source({},{},{})",
                        self.p(p),
                        self.w(w),
                        self.w(o)
                    ));
                }
            }
            for w in bits(kb.overridden[p]) {
                out.push(format!("overridden({},{})", self.p(p), self.w(w)));
            }
        }
        out.sort();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_goldens() -> Vec<Golden> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens");
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "tsv"))
            .collect();
        paths.sort();
        paths
            .iter()
            .map(|p| parse_golden(&std::fs::read_to_string(p).unwrap()).unwrap())
            .collect()
    }

    /// The probe's lineage already lives on three 64 x 64 x 64 index spaces,
    /// each a stack of 64 x 64 registers (row = proposition, bit = world)
    /// along a third axis:
    ///
    /// - V[parent]: `holds_via(inherited(parent), p, w)`, built on demand;
    /// - O[origin]: `inheritance_source(p, w, origin)`;
    /// - L[rule]:  the source rule of `holds_via_rule`.
    ///
    /// Contracting V or O over its third axis must give back the 64 x 64
    /// `holds` register, and each cell must read the same through its HHTL16
    /// address.
    #[test]
    fn the_third_axis_contracts_back_to_holds() {
        use dr_math_binding_probe::hhtl16;
        use lance_graph_contract::morton8x8::Morton8x8;
        let (mut two_parents, mut two_origins) = (false, false);
        for g in all_goldens() {
            let kb = Kb::load(&g.facts).unwrap();
            let h = kb.solve();
            let nw = kb.worlds.0.len();
            let v: Vec<Rows> = (0..nw)
                .map(|par| std::array::from_fn(|p| h.inherited_from(par, p)))
                .collect();
            for p in 0..SLOTS {
                let via_parent = v.iter().fold(h.direct[p], |a, s| a | s[p]);
                assert_eq!(via_parent, h.holds[p], "{} V", g.fixture_id);
                let via_origin = h.origins.iter().fold(0, |a, s| a | s[p]);
                assert_eq!(via_origin, h.holds[p], "{} O", g.fixture_id);
                for w in 0..nw {
                    let parents = v.iter().filter(|s| s[p] >> w & 1 == 1).count();
                    let origins = h.origins.iter().filter(|s| s[p] >> w & 1 == 1).count();
                    two_parents |= parents >= 2;
                    two_origins |= origins >= 2;
                }
            }
            for cube in [&v, &h.origins, &h.sources] {
                for slice in cube.iter() {
                    for p in 0..kb.props.0.len() {
                        for w in 0..nw {
                            let code = Morton8x8::from_xy(p as u8, w as u8);
                            let bit = slice[p] >> w & 1 == 1;
                            assert_eq!(hhtl16::cell(slice, code), Ok(bit));
                        }
                    }
                }
            }
        }
        // Anti-vacuity: the third axis carries more than one entry somewhere,
        // so contraction is a real OR and not a copy of one slice.
        assert!(two_parents, "no cell with two parents");
        assert!(two_origins, "no cell with two origins");
    }

    #[test]
    fn lineage_never_leaves_the_held_cells() {
        // F5: r_orphan has a rule companion but no assertion, so p holds
        // nowhere. Its lineage must stay at its own cell, not flow to a.
        let g = parse_golden(include_str!("../goldens/F5-gates.tsv")).unwrap();
        let kb = Kb::load(&g.facts).unwrap();
        let h = kb.solve();
        let r = kb.rules.0.iter().position(|n| n == "r_orphan").unwrap();
        let p = kb.props.0.iter().position(|n| n == "p").unwrap();
        let base = kb.worlds.0.iter().position(|n| n == "base").unwrap();
        assert_eq!(h.holds[p], 0);
        assert_eq!(h.sources[r][p], 1 << base);
        // Every lineage cell outside the base lies inside holds.
        for s in h.sources.iter().chain(&h.origins) {
            for (i, row) in s.iter().enumerate() {
                let base_row = (0..kb.rules.0.len())
                    .map(|r| (kb.asserts_rule[r][i] | kb.endorses_rule[r][i]) & kb.world_set)
                    .fold(h.direct[i], |a, b| a | b);
                assert_eq!(row & !base_row & !h.holds[i], 0);
            }
        }
    }
}
