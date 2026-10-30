//! D-MIST-NATIVE-0/1a: Mistaber's Kripke holding rules as a least fixpoint
//! over the existing DR64 Boolean relation register, on HHTL16 node rows.
//!
//! Plan: `.claude/plans/d-mistaber-native-v1.md`. Reference answers: Clingo
//! golden files exported by `AdaWorldAPI/mistaber-rs` (`goldens/SOURCE`).
//!
//! # Identity
//!
//! A node is an address, never a name. Each population (world, proposition,
//! rule) is a set of [`rows::NodeRow`](lance_graph_contract::canonical_node::NodeRow)s
//! in the HHTL16 format of [`rows`]; names live only in [`membrane`].
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
//! relation `parent → child`, derived from the parent addresses in the world
//! rows. Each step is one DR64-bound composition plus an AND-NOT: no new
//! algebra. The two lineages (`holds_via_rule`'s source rule and
//! `inheritance_source`'s origin) are the same fixpoint gated by the final
//! `H`, one per rule or origin.
//!
//! # The holdings lens
//!
//! Holdings are read through one lens whose fields are [`HOLDINGS_FIELDS`].
//! A predicate is a [`WideFieldMask`](lance_graph_contract::class_view::WideFieldMask) selection over it: `holds/2` selects
//! proposition and world, `holds_via/3` adds the mode, `holds_via_rule/5`
//! adds both rules. Selecting fewer fields can make two lens rows equal, so
//! the projection is [`Projection::Set`] (Datalog: one fact) or
//! [`Projection::Bag`] (one row per lens row). Only `Set` is Mistaber's
//! semantics; `Bag` exists to show where an alias-only reading goes wrong.
//!
//! # Not covered, refused
//!
//! Anything that is not one fixpoint of facts: a fact kind outside the Kripke
//! inputs, more than 64 worlds, propositions or rules (one register), more
//! than 15 parents of one world (one row), and a selection that is not one of
//! the three predicates.
#![forbid(unsafe_code)]

pub mod membrane;
pub mod rows;

use dr_math_binding_probe::hhtl16::{LocalMap, MapRefusal};
use dr_math_binding_probe::relcomp::{Chain, RelExpr, Rows, Work};
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Scope, bind_dr64,
};
use lance_graph_contract::canonical_node::NodeRow;
use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_contract::register128::RegisterLanes;

pub use membrane::{Golden, Intake, NameTable, Pop, Relation, intake, intake_under, parse_golden};
use rows::{COLUMNS, Population, UNBOUND, columns, identity};

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

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 1 };

/// Why an input or a selection is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A fact kind the Kripke rules do not read.
    UnknownFact(String),
    /// A known fact kind with the wrong number of arguments.
    Arity(String),
    /// More than [`SLOTS`] distinct values on one axis.
    TooMany(&'static str),
    /// The same address twice in one population.
    Duplicate(&'static str),
    /// An address no row of the population holds.
    Unmapped(&'static str),
    /// A world with more parents than one row holds.
    TooManyParents { parents: usize },
    /// The register rails could not be bound.
    Register(lance_graph_contract::hotplug::ActivationDrift),
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

/// Types that hold addresses and numbers, never names.
#[doc(hidden)]
pub trait AddressOnly {}

macro_rules! address_only {
    ($($t:ty),* $(,)?) => { $(impl AddressOnly for $t {})* };
}
address_only!(
    u8,
    u16,
    u32,
    u64,
    usize,
    bool,
    Morton8x8,
    NodeRow,
    RegisterLanes,
    LocalMap,
    FrameStore,
    Work,
    ModeTag,
    Population,
    Kb,
    Mode,
);
impl<T: AddressOnly, const N: usize> AddressOnly for [T; N] {}
impl<T: AddressOnly> AddressOnly for Vec<T> {}
impl<T: AddressOnly> AddressOnly for Option<T> {}
impl<T: AddressOnly + ?Sized> AddressOnly for &T {}
impl AddressOnly for Holdings<'_> {}

/// Compiles only for address-only types.
///
/// ```compile_fail
/// mistaber_kripke_probe::assert_address_only::<String>();
/// ```
///
/// ```compile_fail
/// mistaber_kripke_probe::assert_address_only::<Vec<String>>();
/// ```
#[doc(hidden)]
pub fn assert_address_only<T: AddressOnly + ?Sized>() {}

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

fn prepare(p: &Population, axis: &'static str) -> Result<LocalMap, Refusal> {
    let ids: Vec<u64> = p
        .rows
        .iter()
        .map(|r| u64::from(identity(&p.lanes, r).code()))
        .collect();
    LocalMap::prepare(&ids).map_err(|e| match e {
        MapRefusal::TooMany => Refusal::TooMany(axis),
        MapRefusal::Duplicate => Refusal::Duplicate(axis),
        _ => Refusal::Unmapped(axis),
    })
}

fn local(map: &LocalMap, a: Morton8x8, axis: &'static str) -> Result<usize, Refusal> {
    map.local(u64::from(a.code()))
        .map(usize::from)
        .map_err(|_| Refusal::Unmapped(axis))
}

fn global(map: &LocalMap, local: usize) -> Morton8x8 {
    let g = map.global(local as u8).expect("a mapped local position");
    Morton8x8::from_code(g as u16)
}

/// The Kripke inputs: three populations of rows, their local maps, and one
/// register per relation.
pub struct Kb {
    worlds: Population,
    props: Population,
    rules: Population,
    world_map: LocalMap,
    prop_map: LocalMap,
    rule_map: LocalMap,
    world_set: u64,
    asserts: Rows,
    endorses: Rows,
    overridden: Rows,
    /// Row `parent`, bit `child`: derived from the parent addresses in the
    /// world rows, nowhere else.
    children: Rows,
    asserts_rule: Vec<Rows>,
    endorses_rule: Vec<Rows>,
    store: FrameStore,
}

impl Kb {
    /// Load fact tuples (`kind, args…`). Refuses any fact kind the Kripke
    /// rules do not read.
    pub fn load(facts: &[Vec<String>]) -> Result<(Self, NameTable), Refusal> {
        let (taken, names) = intake(facts)?;
        Ok((Self::assemble(taken)?, names))
    }

    /// Map each population to local positions and fill the relation
    /// registers. The 64-slot limit is enforced here, by the local maps.
    pub fn assemble(i: Intake) -> Result<Self, Refusal> {
        let world_map = prepare(&i.worlds, "world")?;
        let prop_map = prepare(&i.props, "proposition")?;
        let rule_map = prepare(&i.rules, "rule")?;
        let mut kb = Kb {
            store: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: world_map.digest(),
            }),
            world_set: 0,
            asserts: [0; SLOTS],
            endorses: [0; SLOTS],
            overridden: [0; SLOTS],
            children: [0; SLOTS],
            asserts_rule: vec![[0; SLOTS]; rule_map.len()],
            endorses_rule: vec![[0; SLOTS]; rule_map.len()],
            worlds: i.worlds,
            props: i.props,
            rules: i.rules,
            world_map,
            prop_map,
            rule_map,
        };
        kb.derive_children()?;
        for rel in &i.relations {
            match *rel {
                Relation::Declared { world } => {
                    kb.world_set |= 1 << local(&kb.world_map, world, "world")?;
                }
                Relation::Asserts { world, prop } => {
                    let (lw, lp) = kb.cell(world, prop)?;
                    kb.asserts[lp] |= 1 << lw;
                }
                Relation::Endorses { world, prop } => {
                    let (lw, lp) = kb.cell(world, prop)?;
                    kb.endorses[lp] |= 1 << lw;
                }
                Relation::Overridden { world, prop } => {
                    let (lw, lp) = kb.cell(world, prop)?;
                    kb.overridden[lp] |= 1 << lw;
                }
                Relation::AssertsRule { world, prop, rule } => {
                    let (lw, lp) = kb.cell(world, prop)?;
                    let lr = local(&kb.rule_map, rule, "rule")?;
                    kb.asserts_rule[lr][lp] |= 1 << lw;
                }
                Relation::EndorsesRule { world, prop, rule } => {
                    let (lw, lp) = kb.cell(world, prop)?;
                    let lr = local(&kb.rule_map, rule, "rule")?;
                    kb.endorses_rule[lr][lp] |= 1 << lw;
                }
            }
        }
        Ok(kb)
    }

    /// Local `(world, proposition)` positions of one cell.
    fn cell(&self, world: Morton8x8, prop: Morton8x8) -> Result<(usize, usize), Refusal> {
        Ok((
            local(&self.world_map, world, "world")?,
            local(&self.prop_map, prop, "proposition")?,
        ))
    }

    /// The only writer of `children`: every bound column of a world row after
    /// column 0 is a parent of the world in column 0.
    fn derive_children(&mut self) -> Result<(), Refusal> {
        for row in &self.worlds.rows {
            let cols = columns(&self.worlds.lanes, row);
            let lw = local(&self.world_map, cols[0], "world")?;
            for a in cols[1..COLUMNS].iter().filter(|a| **a != UNBOUND) {
                let lp = local(&self.world_map, *a, "world")?;
                self.children[lp] |= 1 << lw;
            }
        }
        Ok(())
    }

    pub fn world_address(&self, local: usize) -> Morton8x8 {
        global(&self.world_map, local)
    }

    pub fn prop_address(&self, local: usize) -> Morton8x8 {
        global(&self.prop_map, local)
    }

    pub fn rule_address(&self, local: usize) -> Morton8x8 {
        global(&self.rule_map, local)
    }

    pub fn world_count(&self) -> usize {
        self.world_map.len()
    }

    pub fn prop_count(&self) -> usize {
        self.prop_map.len()
    }

    pub fn rule_count(&self) -> usize {
        self.rules.rows.len()
    }

    /// Solve: the holds fixpoint, then both lineages.
    pub fn solve(&self) -> Holdings<'_> {
        let (wc, pc) = (self.worlds.lanes.concept(), self.props.lanes.concept());
        let w_axis = Axis {
            class: wc,
            kind: AxisKind::Identity,
        };
        let p_axis = Axis {
            class: pc,
            kind: AxisKind::Property,
        };
        let m_access = MethodSignature::presence(wc, 0, w_axis, w_axis);
        let m_holds = MethodSignature::presence(pc, 0, p_axis, w_axis);
        let m_asserts = MethodSignature::presence(pc, 1, p_axis, w_axis);
        let m_endorses = MethodSignature::presence(pc, 2, p_axis, w_axis);
        let m_overridden = MethodSignature::presence(pc, 3, p_axis, w_axis);
        let guards = Guards {
            frame: self.store.seal(),
            focus: FOCUS,
            scope: SCOPE,
        };
        let access = bind(&self.store, &guards, m_access, &self.children);
        let asserts = bind(&self.store, &guards, m_asserts, &self.asserts);
        let endorses = bind(&self.store, &guards, m_endorses, &self.endorses);
        let overridden = bind(&self.store, &guards, m_overridden, &self.overridden);
        let mut work = Work::default();
        let block = or(&or(asserts.rows, endorses.rows), overridden.rows);
        let direct = and_row(&or(asserts.rows, endorses.rows), self.world_set);
        let hop = |from: &Rows, work: &mut Work| -> Rows {
            hop(&self.store, &guards, m_holds, &access, from, work)
        };
        let (holds, steps) = lfp(&direct, None, &hop, &block, &mut work);
        let lineage = |base: &Rows, work: &mut Work| lfp(base, Some(&holds), &hop, &block, work).0;
        let sources = (0..self.rule_count())
            .map(|r| {
                let base = and_row(
                    &or(&self.asserts_rule[r], &self.endorses_rule[r]),
                    self.world_set,
                );
                lineage(&base, &mut work)
            })
            .collect();
        let origins = (0..self.world_count())
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

/// Bind one register through the DR64 admission.
fn bind<'a>(
    store: &FrameStore,
    guards: &Guards,
    m: MethodSignature,
    rows: &'a Rows,
) -> BoundDr64<'a> {
    bind_dr64(
        store,
        m,
        m,
        guards,
        (FOCUS, SCOPE),
        Operand {
            rows,
            validity: None,
        },
    )
    .expect("a register of this store binds under its own frame")
}

/// `from ∘ children` as a typed two-operand chain.
fn hop(
    store: &FrameStore,
    guards: &Guards,
    m_holds: MethodSignature,
    children: &BoundDr64<'_>,
    from: &Rows,
    work: &mut Work,
) -> Rows {
    let chain = Chain::new(
        store,
        vec![bind(store, guards, m_holds, from), children.clone()],
    )
    .expect("holds ∘ children is typed");
    let (rows, w) = chain
        .eval(&RelExpr::comp(RelExpr::Leaf(0), RelExpr::Leaf(1)))
        .expect("the order covers the chain");
    work.row_ors += w.row_ors;
    rows
}

/// `X = base ∪ ((gate(X) ∘ children) ∧ ¬block)` to its least fixpoint.
/// `gate = None` gates by `X` itself; `Some(h)` gates by `X ∧ h`.
fn lfp(
    base: &Rows,
    gate: Option<&Rows>,
    hop: &dyn Fn(&Rows, &mut Work) -> Rows,
    block: &Rows,
    work: &mut Work,
) -> (Rows, usize) {
    let mut cur = *base;
    for step in 1..=SLOTS + 1 {
        let from = gate.map_or(cur, |h| and(&cur, h));
        let next = or(base, &and_not(&hop(&from, work), block));
        if next == cur {
            return (cur, step);
        }
        cur = next;
    }
    unreachable!(
        "each step extends every proposition by one hop; 64 worlds have no path longer than 63 hops"
    )
}

/// Which holding a [`Mode`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModeTag {
    Asserted,
    Endorsed,
    /// Inherited from the world at this local position.
    Inherited(usize),
}

/// One holding mode: its cells, and per rule the cells that rule supports.
struct Mode {
    tag: ModeTag,
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
        let n = kb.prop_count();
        let direct = |tag, table: &Rows, rules: &[Rows]| Mode {
            tag,
            cells: and_row(table, kb.world_set),
            rules: rules
                .iter()
                .map(|r| and(r, &and_row(table, kb.world_set)))
                .collect(),
        };
        let mut out = vec![
            direct(ModeTag::Asserted, &kb.asserts, &kb.asserts_rule),
            direct(ModeTag::Endorsed, &kb.endorses, &kb.endorses_rule),
        ];
        for par in 0..kb.world_count() {
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
                tag: ModeTag::Inherited(par),
                cells,
                rules,
            });
        }
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

    fn fact(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    fn local_of(map: &LocalMap, names: &NameTable, pop: Pop, name: &str) -> usize {
        let a = names.address_of(pop, name).expect(name);
        usize::from(map.local(u64::from(a.code())).expect(name))
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
            let (kb, _names) = Kb::load(&g.facts).unwrap();
            let h = kb.solve();
            let nw = kb.world_count();
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
                    for p in 0..kb.prop_count() {
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
        let (kb, names) = Kb::load(&g.facts).unwrap();
        let h = kb.solve();
        let r = local_of(&kb.rule_map, &names, Pop::Rule, "r_orphan");
        let p = local_of(&kb.prop_map, &names, Pop::Proposition, "p");
        let base = local_of(&kb.world_map, &names, Pop::World, "base");
        assert_eq!(h.holds[p], 0);
        assert_eq!(h.sources[r][p], 1 << base);
        // Every lineage cell outside the base lies inside holds.
        for s in h.sources.iter().chain(&h.origins) {
            for (i, row) in s.iter().enumerate() {
                let base_row = (0..kb.rule_count())
                    .map(|r| (kb.asserts_rule[r][i] | kb.endorses_rule[r][i]) & kb.world_set)
                    .fold(h.direct[i], |a, b| a | b);
                assert_eq!(row & !base_row & !h.holds[i], 0);
            }
        }
    }

    /// Destructures each struct exhaustively (no `..`), so adding a field
    /// that is not address-only, or any field at all, stops compiling here.
    fn fields_are_address_only(kb: &Kb, h: &Holdings<'_>, m: &Mode, p: &Population) {
        fn ok<T: AddressOnly + ?Sized>(_: &T) {}
        let Kb {
            worlds,
            props,
            rules,
            world_map,
            prop_map,
            rule_map,
            world_set,
            asserts,
            endorses,
            overridden,
            children,
            asserts_rule,
            endorses_rule,
            store,
        } = kb;
        ok(worlds);
        ok(props);
        ok(rules);
        ok(world_map);
        ok(prop_map);
        ok(rule_map);
        ok(world_set);
        ok(asserts);
        ok(endorses);
        ok(overridden);
        ok(children);
        ok(asserts_rule);
        ok(endorses_rule);
        ok(store);
        let Holdings {
            kb,
            block,
            direct,
            holds,
            sources,
            origins,
            steps,
            work,
        } = h;
        ok(kb);
        ok(block);
        ok(direct);
        ok(holds);
        ok(sources);
        ok(origins);
        ok(steps);
        ok(work);
        let Mode { tag, cells, rules } = m;
        ok(tag);
        ok(cells);
        ok(rules);
        let Population { lanes, rows } = p;
        ok(lanes);
        ok(rows);
    }

    /// FAILS IF a String, a `NameTable` or a `Golden` becomes a field of
    /// `Kb`, `Holdings`, `Mode` or `Population` (compile error), or the bound
    /// stops rejecting `String` (the two compile_fail doctests on
    /// `assert_address_only`).
    #[test]
    fn the_solved_state_holds_no_string() {
        let g = all_goldens()
            .into_iter()
            .find(|g| g.fixture_id == "F1-modes")
            .unwrap();
        let (kb, _names) = Kb::load(&g.facts).unwrap();
        let h = kb.solve();
        let modes = h.modes();
        fields_are_address_only(&kb, &h, &modes[0], &kb.worlds);
        assert!(modes.len() > 2, "inherited modes exist");
    }

    /// FAILS IF a world with two parents does not get both as in-row
    /// addresses that derive to exactly the old accessibility register.
    #[test]
    fn a_world_with_two_parents_derives_the_old_accessibility() {
        let facts = vec![
            fact(&["world", "base"]),
            fact(&["world", "mechaber"]),
            fact(&["world", "rema"]),
            fact(&["world", "ashk_ah"]),
            fact(&["accessible", "mechaber", "base"]),
            fact(&["accessible", "rema", "base"]),
            fact(&["accessible", "ashk_ah", "mechaber"]),
            fact(&["accessible", "ashk_ah", "rema"]),
            fact(&["asserts", "mechaber", "p"]),
            fact(&["asserts", "rema", "q"]),
        ];
        let oracle = |facts: &[Vec<String>], label: &str| {
            let (kb, names) = Kb::load(facts).unwrap();
            let mut pairs = std::collections::BTreeSet::new();
            for f in facts.iter().filter(|f| f[0] == "accessible") {
                let c = local_of(&kb.world_map, &names, Pop::World, &f[1]);
                let p = local_of(&kb.world_map, &names, Pop::World, &f[2]);
                assert_eq!(kb.children[p] >> c & 1, 1, "{label}: {} <- {}", f[1], f[2]);
                pairs.insert((c, p));
            }
            let total: u32 = kb.children.iter().map(|r| r.count_ones()).sum();
            assert_eq!(total as usize, pairs.len(), "{label}");
            kb
        };
        let kb = oracle(&facts, "ashk_ah");
        let (_, names) = Kb::load(&facts).unwrap();
        let atoms = kb.solve().materialize_atoms(&names, Projection::Set);
        for a in [
            "holds(p,ashk_ah)",
            "holds(q,ashk_ah)",
            "holds_via(inherited(rema),q,ashk_ah)",
        ] {
            assert!(atoms.contains(&a.to_owned()), "{a}");
        }
        let mut with_edges = 0;
        for g in all_goldens() {
            let kb = oracle(&g.facts, &g.fixture_id);
            with_edges += usize::from(kb.children.iter().any(|r| *r != 0));
        }
        assert!(with_edges > 0, "some golden has accessibility");

        // Accessibility is not a relation: this match stops compiling if an
        // accessible variant is added.
        let (taken, _) = intake(&facts).unwrap();
        for rel in &taken.relations {
            match rel {
                Relation::Declared { .. }
                | Relation::Asserts { .. }
                | Relation::Endorses { .. }
                | Relation::Overridden { .. }
                | Relation::AssertsRule { .. }
                | Relation::EndorsesRule { .. } => {}
            }
        }
    }
}
