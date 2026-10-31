//! The membrane: the only module that holds names.
//!
//! Names enter here (the golden parse, the fact tuples) and leave here (the
//! atom renderer). Inside, a node is an address in an HHTL16 row. The
//! [`NameTable`] is never held by [`Kb`] or [`Holdings`].

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::morton8x8::Morton8x8;

use crate::rows::{
    COLUMNS, Concepts, MAX_PARENTS, Population, address, bind_populations, blank_row, write_columns,
};
use crate::{
    F_IMMEDIATE, F_MODE, F_PROP, F_SOURCE, F_WORLD, Holdings, Kb, Mode, ModeTag, Projection,
    Refusal, bits,
};

/// The immediate rule upstream attributes every inherited holding to.
const INHERIT_RULE: &str = "r_kripke_inherit";

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

/// The three populations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pop {
    World,
    Proposition,
    Rule,
}

impl Pop {
    fn axis(self) -> &'static str {
        match self {
            Pop::World => "world",
            Pop::Proposition => "proposition",
            Pop::Rule => "rule",
        }
    }
}

/// Names by first-appearance ordinal, one list per population. Lives only at
/// the membrane.
#[derive(Default)]
pub struct NameTable {
    worlds: Vec<String>,
    props: Vec<String>,
    rules: Vec<String>,
}

impl NameTable {
    fn list(&self, pop: Pop) -> &Vec<String> {
        match pop {
            Pop::World => &self.worlds,
            Pop::Proposition => &self.props,
            Pop::Rule => &self.rules,
        }
    }

    /// First-appearance ordinal of `name`. Refuses only when the ordinal
    /// space is exhausted; the 64-slot limit belongs to the local map.
    fn ordinal(&mut self, pop: Pop, name: &str) -> Result<u16, Refusal> {
        let list = match pop {
            Pop::World => &mut self.worlds,
            Pop::Proposition => &mut self.props,
            Pop::Rule => &mut self.rules,
        };
        if let Some(i) = list.iter().position(|n| n == name) {
            return Ok(i as u16);
        }
        if list.len() >= usize::from(u16::MAX) {
            return Err(Refusal::TooMany(pop.axis()));
        }
        list.push(name.to_owned());
        Ok((list.len() - 1) as u16)
    }

    pub fn address_of(&self, pop: Pop, name: &str) -> Option<Morton8x8> {
        let i = self.list(pop).iter().position(|n| n == name)?;
        Some(address(i as u16))
    }

    pub fn name(&self, pop: Pop, a: Morton8x8) -> &str {
        &self.list(pop)[usize::from(a.code()) - 1]
    }
}

/// One input relation, over addresses. Accessibility is not here: it lives
/// only in world rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    Declared {
        world: Morton8x8,
    },
    Asserts {
        world: Morton8x8,
        prop: Morton8x8,
    },
    Endorses {
        world: Morton8x8,
        prop: Morton8x8,
    },
    Overridden {
        world: Morton8x8,
        prop: Morton8x8,
    },
    AssertsRule {
        world: Morton8x8,
        prop: Morton8x8,
        rule: Morton8x8,
    },
    EndorsesRule {
        world: Morton8x8,
        prop: Morton8x8,
        rule: Morton8x8,
    },
}

/// Facts taken in: three populations of rows and the non-accessibility
/// relations over their addresses.
pub struct Intake {
    pub worlds: Population,
    pub props: Population,
    pub rules: Population,
    pub relations: Vec<Relation>,
}

/// [`intake_under`] with the lab concepts.
pub fn intake(facts: &[Vec<String>]) -> Result<(Intake, NameTable), Refusal> {
    intake_under(facts, Concepts::LAB)
}

/// Take fact tuples (`kind, args...`) in under `c`. Refuses any fact kind the
/// Kripke rules do not read, and a world with more than [`MAX_PARENTS`]
/// parents.
pub fn intake_under(facts: &[Vec<String>], c: Concepts) -> Result<(Intake, NameTable), Refusal> {
    let mut names = NameTable::default();
    let mut relations = Vec::new();
    let mut parents: Vec<Vec<u16>> = Vec::new();
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
            "world" => {
                let w = names.ordinal(Pop::World, &args[0])?;
                relations.push(Relation::Declared { world: address(w) });
            }
            "accessible" => {
                let child = names.ordinal(Pop::World, &args[0])?;
                let parent = names.ordinal(Pop::World, &args[1])?;
                if parents.len() <= usize::from(child) {
                    parents.resize(usize::from(child) + 1, Vec::new());
                }
                let ps = &mut parents[usize::from(child)];
                if !ps.contains(&parent) {
                    ps.push(parent);
                }
            }
            _ => {
                let w = names.ordinal(Pop::World, &args[0])?;
                let p = names.ordinal(Pop::Proposition, &args[1])?;
                let (world, prop) = (address(w), address(p));
                relations.push(match kind.as_str() {
                    "asserts" => Relation::Asserts { world, prop },
                    "endorses" => Relation::Endorses { world, prop },
                    "override" => Relation::Overridden { world, prop },
                    _ => {
                        let rule = address(names.ordinal(Pop::Rule, &args[2])?);
                        if kind == "asserts_rule" {
                            Relation::AssertsRule { world, prop, rule }
                        } else {
                            Relation::EndorsesRule { world, prop, rule }
                        }
                    }
                });
            }
        }
    }
    parents.resize(names.worlds.len(), Vec::new());
    let [wl, pl, rl] = bind_populations(&c)?;
    let mut worlds = Population {
        lanes: wl,
        rows: Vec::new(),
    };
    for (o, ps) in parents.iter().enumerate() {
        if ps.len() > MAX_PARENTS {
            return Err(Refusal::TooManyParents { parents: ps.len() });
        }
        let cols: Vec<Morton8x8> = std::iter::once(address(o as u16))
            .chain(ps.iter().map(|&p| address(p)))
            .collect();
        debug_assert!(cols.len() <= COLUMNS);
        let mut row = blank_row();
        write_columns(&wl, &mut row, &cols);
        worlds.rows.push(row);
    }
    let single = |lanes, n: usize| {
        let mut p = Population {
            lanes,
            rows: Vec::new(),
        };
        for o in 0..n {
            let mut row = blank_row();
            write_columns(&lanes, &mut row, &[address(o as u16)]);
            p.rows.push(row);
        }
        p
    };
    let props = single(pl, names.props.len());
    let rules = single(rl, names.rules.len());
    Ok((
        Intake {
            worlds,
            props,
            rules,
            relations,
        },
        names,
    ))
}

impl Mode {
    fn label(&self, names: &NameTable, kb: &Kb) -> String {
        match self.tag {
            ModeTag::Asserted => "asserted".to_owned(),
            ModeTag::Endorsed => "endorsed".to_owned(),
            ModeTag::Inherited(par) => {
                format!(
                    "inherited({})",
                    names.name(Pop::World, kb.world_address(par))
                )
            }
        }
    }
}

impl Holdings<'_> {
    fn wn<'n>(&self, names: &'n NameTable, i: usize) -> &'n str {
        names.name(Pop::World, self.kb.world_address(i))
    }

    fn pn<'n>(&self, names: &'n NameTable, i: usize) -> &'n str {
        names.name(Pop::Proposition, self.kb.prop_address(i))
    }

    /// How many lens rows hold `prop` in `world`: the multiplicity a bag
    /// projection to `holds/2` reports. A lens row is one (mode, source
    /// rule); a holding with no rule is one row with empty rule fields.
    pub fn bag_multiplicity(&self, names: &NameTable, prop: &str, world: &str) -> usize {
        let local = |pop, map: &dyn Fn(u64) -> Option<u8>, name: &str| {
            names
                .address_of(pop, name)
                .and_then(|a| map(u64::from(a.code())))
        };
        let kb = self.kb;
        let (Some(p), Some(w)) = (
            local(Pop::Proposition, &|g| kb.prop_map.local(g).ok(), prop),
            local(Pop::World, &|g| kb.world_map.local(g).ok(), world),
        ) else {
            return 0;
        };
        self.modes()
            .iter()
            .map(|m| m.rows(usize::from(p), usize::from(w)))
            .sum()
    }

    /// Materialize one predicate as atoms: a [`WideFieldMask`] selection over
    /// [`HOLDINGS_FIELDS`](crate::HOLDINGS_FIELDS). For the test oracle only.
    pub fn materialize_selection(
        &self,
        names: &NameTable,
        sel: &WideFieldMask,
        proj: Projection,
    ) -> Result<Vec<String>, Refusal> {
        let has = |f| sel.has(f);
        let base = has(F_PROP) && has(F_WORLD);
        let kb = self.kb;
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
                    for p in 0..kb.prop_count() {
                        for w in bits(self.holds[p]) {
                            out.push(format!(
                                "holds({},{})",
                                self.pn(names, p),
                                self.wn(names, w)
                            ));
                        }
                    }
                }
                Projection::Bag => {
                    for m in self.modes() {
                        for (p, &row) in m.cells.iter().enumerate().take(kb.prop_count()) {
                            for w in bits(row) {
                                for _ in 0..m.rows(p, w) {
                                    out.push(format!(
                                        "holds({},{})",
                                        self.pn(names, p),
                                        self.wn(names, w)
                                    ));
                                }
                            }
                        }
                    }
                }
            },
            (true, true, false, false, 3) => {
                for m in self.modes() {
                    let label = m.label(names, kb);
                    for (p, &row) in m.cells.iter().enumerate().take(kb.prop_count()) {
                        for w in bits(row) {
                            let copies = match proj {
                                Projection::Set => 1,
                                Projection::Bag => m.rows(p, w),
                            };
                            for _ in 0..copies {
                                out.push(format!(
                                    "holds_via({},{},{})",
                                    label,
                                    self.pn(names, p),
                                    self.wn(names, w)
                                ));
                            }
                        }
                    }
                }
            }
            (true, true, true, true, 5) => self.rule_atoms(names, &mut out),
            _ => return Err(Refusal::Selection),
        }
        Ok(out)
    }

    fn rule_atoms(&self, names: &NameTable, out: &mut Vec<String>) {
        let kb = self.kb;
        for r in 0..kb.rule_count() {
            let rule = names.name(Pop::Rule, kb.rule_address(r));
            for (mode, table) in [
                ("asserted", &kb.asserts_rule[r]),
                ("endorsed", &kb.endorses_rule[r]),
            ] {
                for (p, &row) in table.iter().enumerate().take(kb.prop_count()) {
                    for w in bits(row & kb.world_set) {
                        out.push(format!(
                            "holds_via_rule({mode},{},{},{rule},{rule})",
                            self.pn(names, p),
                            self.wn(names, w)
                        ));
                    }
                }
            }
            for par in 0..kb.world_count() {
                for p in 0..kb.prop_count() {
                    if self.sources[r][p] >> par & 1 == 1 {
                        for w in bits(self.inherited_from(par, p)) {
                            out.push(format!(
                                "holds_via_rule(inherited({}),{},{},{INHERIT_RULE},{rule})",
                                self.wn(names, par),
                                self.pn(names, p),
                                self.wn(names, w)
                            ));
                        }
                    }
                }
            }
        }
    }

    /// Every Kripke atom the golden files carry, sorted. `proj` applies to
    /// `holds/2`, the only selection where rows can coincide.
    pub fn materialize_atoms(&self, names: &NameTable, proj: Projection) -> Vec<String> {
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
                self.materialize_selection(names, &s, pr)
                    .expect("a supported selection"),
            );
        }
        let kb = self.kb;
        for p in 0..kb.prop_count() {
            let inherited: u64 = (0..kb.world_count())
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
                        self.pn(names, p),
                        self.wn(names, w)
                    ));
                }
            }
            for (o, cells) in self.origins.iter().enumerate() {
                for w in bits(cells[p]) {
                    out.push(format!(
                        "inheritance_source({},{},{})",
                        self.pn(names, p),
                        self.wn(names, w),
                        self.wn(names, o)
                    ));
                }
            }
            for w in bits(kb.overridden[p]) {
                out.push(format!(
                    "overridden({},{})",
                    self.pn(names, p),
                    self.wn(names, w)
                ));
            }
        }
        out.sort();
        out
    }
}
