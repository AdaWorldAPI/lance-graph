//! `selection` — the **RAIL WALK** + **NAMED-VIEW REGISTRY**: nested selection
//! over the object graph expressed as *masks keyed by view*, with **NO query
//! document**.
//!
//! This is the GraphQL analogue built the substrate's way. There is no
//! materialized selection tree, no recursive `Selection` struct, no serde. The
//! mechanism is the operator-ruled one:
//!
//! - A relationship is a **field POSITION** in a class's existing field basis —
//!   bit `i` of the flat [`WideFieldMask`] can be a *rail* (`part_of:is_a`, a
//!   `u8:u8` pair per `.claude/v3/soa_layout/le-contract.md` §3 L1). Setting that
//!   bit means "follow this rail".
//! - At the rail's target, the target's OWN `classid` resolves its own view and
//!   mask — the same `classid → ClassView → mask` lookup everything uses.
//! - Nested selection = **masks keyed by view**; the recursion is THE WALK
//!   ITSELF: at each node, AND the view's mask with the node's presence, emit the
//!   present fields, follow the set rail-bearing bits, recurse. Termination: a
//!   node whose present∩view mask yields no rail target is a leaf.
//! - A **named view** is a [`NamedView`] = `(ClassId, WideFieldMask,
//!   DisplayTemplate)` recipe — a persisted mask constant (GraphQL fragment +
//!   persisted query in one). Fragment spread / composition =
//!   [`ViewRegistry::union_of`], bitwise OR via the EXISTING
//!   [`WideFieldMask::union`] op.
//!
//! ## Carving-verification finding (mandatory pre-write check)
//!
//! **Question:** can "position `i` of class `C` is rail-bearing" be *derived* from
//! the existing carving / read-mode machinery?  **Answer: NO — it cannot.**
//!
//! - [`facet::CascadeShape`](crate::facet::CascadeShape) only GROUPS the 12
//!   content-blind cascade bytes into `(group, level)` — `6×2` / `4×3` / `3×4`,
//!   pure index math (`crate::facet`, `CascadeShape::index`/`group_of`). It is by
//!   construction "content-blind: only the CONSUMER decides what the 8:8 means"
//!   (`facet.rs` module docs, ~lines 31-35 and 344-393). It never marks a group
//!   as a relationship rather than a scalar.
//! - [`canonical_node::ReadMode`](crate::canonical_node::ReadMode) carries exactly
//!   `{ tail_variant, value_schema, edge_codec }` (`canonical_node.rs` ~lines
//!   1192-1202); [`classid_read_mode`](crate::canonical_node::classid_read_mode)
//!   (`canonical_node.rs` ~line 1402) resolves only those three axes. None of them
//!   names rail positions.
//! - `.claude/v3/soa_layout/le-contract.md` §3 (rows L1-L3) says a `6×(8:8)`
//!   payload *may* be read as `part_of:is_a` rails, but "The reading is ALWAYS
//!   selected by the classview (slot purity §2) — never by inspecting payload
//!   bytes, never by convention-in-code" (le-contract.md ~lines 130-131). Which
//!   positions are rails is therefore a per-class ClassView *focus-lens* property,
//!   not a derivable of `CascadeShape` or `ReadMode`.
//! - The [`ClassView`](crate::class_view::ClassView) trait itself exposes no
//!   rail-position accessor (`class_view.rs`).
//!
//! **Consequence** (per the task's documented condition — "if no, the walker takes
//! rail-position knowledge via a minimal additive trait method … do NOT change any
//! existing trait method"): rail knowledge enters the walk through the NEW
//! [`RailGraph`] trait's [`rail_target`](RailGraph::rail_target) — dependency
//! inversion in the shape of [`PlannerContract`](crate::plan) /
//! [`MailboxSoaView`](crate::soa_view): the contract owns the *vocabulary* (view
//! masks, the walk), the consumer owns the *graph* and thus which `(key,
//! position)` pairs resolve to another node. **No existing type/trait/signature is
//! changed, and NOT ONE method is added to `ClassView`.** This module is purely
//! additive.
//!
//! ## Zero-dep
//!
//! std only ([`std::collections::HashSet`] for the cycle-guard visited set). No
//! serde anywhere. The walk allocates nothing beyond the visited set and whatever
//! the caller's `visit` closure keeps.

use crate::class_view::{ClassId, ClassView, WideFieldMask};
use crate::ontology::DisplayTemplate;
use std::collections::HashSet;

/// Identifier of a [`NamedView`] in a [`ViewRegistry`] — a plain `Vec` index.
///
/// **Width justification (`u16`).** A `ViewId` names one recipe in the fragment
/// library; `u16` gives 65 536 named views, which mirrors the
/// [`ClassId`](crate::class_view::ClassId) cardinality ceiling (also `u16`) — a
/// view is at most a per-class-per-projection constant, so the library is bounded
/// by (classes × a handful of projections each), comfortably inside `u16`. Keeping
/// it `u16` also keeps [`NamedView`] small and lets a consumer store a view id in
/// the same width it stores a class id. It is a registry index, never a content
/// hash — it stays outside the CAM identity layer (`I-VSA-IDENTITIES`).
pub type ViewId = u16;

/// A **named view** — the fragment-library recipe: which class, which fields
/// ([`WideFieldMask`]), and which render template.
///
/// This is "GraphQL fragment + persisted query in one": a persisted mask constant
/// keyed by `class`, plus the [`DisplayTemplate`] a renderer picks. The mask's set
/// bits are the *selected* field positions; some of those positions are rails
/// (followed at walk time via [`RailGraph::rail_target`]), the rest are leaf
/// fields (emitted). Composition of two views is [`WideFieldMask::union`] (see
/// [`ViewRegistry::union_of`]) — fragment spread is a bitwise OR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedView {
    /// The class this view projects. A view is applied to a node only when the
    /// node's [`class_of`](RailGraph::class_of) matches (the walk resolves the
    /// view per node's own classid — the canon `classid → view` lookup).
    pub class: ClassId,
    /// The selected field positions — the persisted mask constant.
    pub mask: WideFieldMask,
    /// Which template renders the projected view.
    pub template: DisplayTemplate,
}

impl NamedView {
    /// A view recipe over `class` selecting `mask`, rendered by `template`.
    #[must_use]
    pub fn new(class: ClassId, mask: WideFieldMask, template: DisplayTemplate) -> Self {
        Self {
            class,
            mask,
            template,
        }
    }
}

/// The **fragment library** — a `Vec`-indexed registry of [`NamedView`] recipes.
///
/// Plain `Vec`, no `HashMap` ceremony: a [`ViewId`] IS the index. `register`
/// appends and returns the id; `get` resolves it; `union_of` composes several
/// views' masks into one (fragment spread = bitwise OR over the EXISTING
/// [`WideFieldMask::union`]).
#[derive(Debug, Clone, Default)]
pub struct ViewRegistry {
    views: Vec<NamedView>,
}

impl ViewRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { views: Vec::new() }
    }

    /// Register a view, returning its [`ViewId`] (its index). Ids are assigned
    /// densely from `0`; a registered id never moves (append-only, like the
    /// N3-stable [`WideFieldMask`] bit positions).
    ///
    /// # Panics
    ///
    /// If the registry already holds [`u16::MAX`]` + 1` views (the `ViewId` space
    /// is exhausted) — a hard refusal rather than a silently-wrapping index.
    pub fn register(&mut self, view: NamedView) -> ViewId {
        let id = self.views.len();
        assert!(
            id <= ViewId::MAX as usize,
            "ViewRegistry: ViewId space (u16) exhausted"
        );
        self.views.push(view);
        id as ViewId
    }

    /// Resolve a [`ViewId`] to its [`NamedView`], or `None` if out of range.
    #[must_use]
    pub fn get(&self, id: ViewId) -> Option<&NamedView> {
        self.views.get(id as usize)
    }

    /// Number of registered views.
    #[must_use]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// **Fragment spread** — the composed [`WideFieldMask`] selecting every field
    /// any of `ids` selects: the bitwise OR of their masks, via the EXISTING
    /// [`WideFieldMask::union`]. Out-of-range ids contribute
    /// [`WideFieldMask::EMPTY`] (skipped), never a panic — composing an unknown
    /// fragment simply adds nothing. Order-independent (union is commutative +
    /// associative), so the composed mask is deterministic regardless of `ids`
    /// order.
    #[must_use]
    pub fn union_of(&self, ids: &[ViewId]) -> WideFieldMask {
        ids.iter()
            .fold(WideFieldMask::EMPTY, |acc, &id| match self.get(id) {
                Some(v) => acc.union(&v.mask),
                None => acc,
            })
    }
}

/// The minimal generic **node surface** the rail walk traverses — dependency
/// inversion in the shape of [`PlannerContract`](crate::plan): the contract owns
/// the walk vocabulary, the consumer owns the graph.
///
/// The consumer implements this over whatever holds its nodes (a
/// [`NodeRow`](crate::canonical_node::NodeRow) SoA, an in-memory adjacency map,
/// …). The three methods are exactly what the walk needs and nothing more:
///
/// - [`class_of`](Self::class_of) — the node's `classid`, so the walk resolves
///   *its own* view + mask (the canon `classid → ClassView → mask` lookup, applied
///   per node — the target's own classid governs the target's projection).
/// - [`present_mask`](Self::present_mask) — the node's populated-field presence
///   bits (C2 presence, the SoA's structural delta). The walk AND-s this with the
///   view mask so it emits only fields that are BOTH selected AND present.
/// - [`rail_target`](Self::rail_target) — the rail resolver: given a `(key,
///   position)`, the node this field position rails to, or `None` if that position
///   is a leaf field (not a rail). **This is where rail-position knowledge enters
///   the walk** — see the module-level carving-verification finding.
pub trait RailGraph {
    /// The node handle. `Copy + Eq + Hash` so it can seed the cycle-guard set with
    /// no allocation per node beyond the set itself.
    type Key: Copy + Eq + core::hash::Hash;

    /// The `classid` of `key` — resolves which view/mask governs this node.
    fn class_of(&self, key: Self::Key) -> ClassId;

    /// The presence [`WideFieldMask`] of `key`: which field positions are
    /// populated on this instance (C2 presence). AND-ed with the view mask so a
    /// hop emits exactly the present, selected fields.
    fn present_mask(&self, key: Self::Key) -> WideFieldMask;

    /// The node `key`'s field `position` rails to — `Some(child)` if this position
    /// is a rail (`part_of:is_a`, an object reference), `None` if it is a leaf
    /// field. The consumer owns this map, so it owns which positions are rails.
    fn rail_target(&self, key: Self::Key, position: u8) -> Option<Self::Key>;
}

/// One emitted field of the walk — renderer-neutral. The consumer feeds `(key,
/// position)` to the EXISTING render path
/// ([`ClassView::facet_rows`](crate::class_view::ClassView::facet_rows) /
/// [`render_rows`](crate::class_view::ClassView::render_rows)); this module invents
/// no render-row type. `class` and `depth` are carried because the walk already
/// has them (the node's own classid governs its projection; depth is the nesting
/// level, root = 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldVisit<K> {
    /// The node this field belongs to.
    pub key: K,
    /// The node's class (its own `classid` resolved the view).
    pub class: ClassId,
    /// The field position (binds to the class's [`FieldRef`](crate::ontology::FieldRef)
    /// / facet byte at that position — the consumer's render path resolves it).
    pub position: u8,
    /// Nesting depth: root = 0, a rail hop adds 1.
    pub depth: usize,
}

/// **The rail walk.** Starting at `root`, resolve each node's view via its own
/// `classid`, AND the view's mask with the node's presence, emit each present
/// selected field (ascending position order — deterministic), then follow the set
/// rail-bearing bits and recurse.
///
/// - `graph` — the [`RailGraph`] the consumer owns.
/// - `class_view` — the [`ClassView`]; used to bound each hop's field iteration to
///   the class's declared field count (a class has exactly `field_count` fields).
///   The consumer additionally uses it downstream to render each [`FieldVisit`].
/// - `registry` — the [`ViewRegistry`] fragment library.
/// - `view_binding` — per-class view selection: `class → Option<ViewId>`. A class
///   with no bound view (`None`) is not projected (its subtree is pruned).
/// - `root` — the starting node.
/// - `max_depth` — the explicit descent cap: the root is depth `0`; a child is
///   visited only while its depth `<= max_depth`. `max_depth == 0` walks the root
///   only. Clamped to [`MAX_WALK_DEPTH`] — the walk recurses one frame per rail
///   hop, and a stack overflow aborts the process (it cannot be caught), so the
///   ceiling holds regardless of caller input.
/// - `visit` — the renderer-neutral sink, called once per emitted `(key,
///   position)`.
///
/// **Termination is guaranteed three ways:** the **path-local** cycle guard (a
/// key already on the CURRENT recursion stack is not re-entered, so a cyclic
/// `part_of` graph cannot loop — while a key reachable through several selected
/// rails is walked once **per path**, emitting at each path's own depth; a
/// global visited-set would let the first DFS path win and emit shared DAG
/// targets at the wrong depth), the explicit `max_depth` cap, and natural leaf
/// termination (a node whose present∩view mask yields no
/// [`rail_target`](RailGraph::rail_target) recurses nowhere). Fields are
/// emitted in ascending position order and rails followed in ascending position
/// order, so the whole traversal is deterministic.
///
/// **View/class agreement is enforced:** a binding that returns a view whose
/// [`NamedView::class`] differs from the node's own class is pruned fail-closed
/// — field positions are defined per class, and applying a foreign class's mask
/// would emit unrelated positions.
pub fn walk_rails<G, V, F>(
    graph: &G,
    class_view: &V,
    registry: &ViewRegistry,
    view_binding: impl Fn(ClassId) -> Option<ViewId>,
    root: G::Key,
    max_depth: usize,
    visit: &mut F,
) where
    G: RailGraph,
    V: ClassView,
    F: FnMut(FieldVisit<G::Key>),
{
    let mut stack: HashSet<G::Key> = HashSet::new();
    walk_node(
        graph,
        class_view,
        registry,
        &view_binding,
        root,
        0,
        max_depth.min(MAX_WALK_DEPTH),
        &mut stack,
        visit,
    );
}

/// Hard sanity ceiling on [`walk_rails`]'s `max_depth`, independent of caller
/// input. The walk recurses one call frame per rail hop; a long acyclic chain
/// under a generous `max_depth` (e.g. `usize::MAX`) would otherwise grow the
/// call stack unboundedly, and a stack overflow ABORTS the process — unlike a
/// panic it cannot be caught or unwound. 64 is far above any sanctioned
/// projection depth (the facet path is 12 levels; rails span 6 positions) while
/// keeping worst-case recursion trivially stack-safe.
pub const MAX_WALK_DEPTH: usize = 64;

/// Positions are `u8`, so at most 256 field positions exist per class — the same
/// cap [`WideFieldMask`] imposes. The iteration bound is `min(field_count, 256)`.
const MAX_POSITIONS: usize = 256;

#[allow(clippy::too_many_arguments)]
fn walk_node<G, V, B, F>(
    graph: &G,
    class_view: &V,
    registry: &ViewRegistry,
    view_binding: &B,
    key: G::Key,
    depth: usize,
    max_depth: usize,
    stack: &mut HashSet<G::Key>,
    visit: &mut F,
) where
    G: RailGraph,
    V: ClassView,
    B: Fn(ClassId) -> Option<ViewId>,
    F: FnMut(FieldVisit<G::Key>),
{
    if depth > max_depth {
        return;
    }
    // PATH-LOCAL cycle guard (codex P2 on #776): only keys on the CURRENT
    // recursion stack are pruned — a true cycle terminates, while a key
    // reachable through several selected rails (a DAG) is walked once per
    // path, at each path's own depth. A global visited-set would let the
    // first DFS path win, emitting shared targets at the wrong depth and
    // silently pruning subtrees reachable via a shorter path. The key is
    // removed on EVERY exit below (unwind), never left behind.
    if !stack.insert(key) {
        return;
    }

    'body: {
        let class = graph.class_of(key);
        let Some(view_id) = view_binding(class) else {
            break 'body; // no view bound for this class → prune its subtree
        };
        let Some(view) = registry.get(view_id) else {
            break 'body; // dangling view id → prune (never a panic)
        };
        // View/class agreement (codex P2 on #776): field positions are defined
        // per class; a stale or cross-class binding must not apply a foreign
        // class's mask to this node. Prune fail-closed.
        if view.class != class {
            break 'body;
        }

        // AND the view's selection mask with the node's presence: emit only fields
        // that are BOTH selected by the view AND populated on this instance (C2
        // presence).
        let effective = view.mask.intersect(&graph.present_mask(key));

        // Bound iteration to the class's declared field count (a class has exactly
        // `field_count` fields), capped at the u8 position ceiling.
        let field_count = class_view.field_count(class).min(MAX_POSITIONS);

        // Emit present, selected fields in ascending position order (deterministic).
        for pos in 0..field_count {
            let p = pos as u8;
            if effective.has(p) {
                visit(FieldVisit {
                    key,
                    class,
                    position: p,
                    depth,
                });
            }
        }

        // Follow the set rail-bearing bits in ascending position order
        // (deterministic), recursing into each rail's target. A present, selected
        // position that resolves to a `rail_target` is a rail; one that resolves to
        // `None` is a leaf field (already emitted above, not recursed).
        for pos in 0..field_count {
            let p = pos as u8;
            if effective.has(p) {
                if let Some(child) = graph.rail_target(key, p) {
                    walk_node(
                        graph,
                        class_view,
                        registry,
                        view_binding,
                        child,
                        depth + 1,
                        max_depth,
                        stack,
                        visit,
                    );
                }
            }
        }
    }

    // Unwind the path-local guard: this key may be revisited via another path.
    stack.remove(&key);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_view::FieldMask;
    use crate::ontology::{DisplayTemplate, FieldRef};
    use std::collections::HashMap;

    // ── A tiny in-memory ClassView + RailGraph ───────────────────────────────
    //
    // Two independent shapes prove the walk generalizes:
    //   WorkPackage(1) --assignee--> User(2)          (a hot work-item graph)
    //   Wall(3)        --part_of --> Storey(4)         (a mereology rail)
    // The cyclic-graph test reuses Wall(3) as A part_of B, B part_of A.

    const WORK_PACKAGE: ClassId = 1;
    const USER: ClassId = 2;
    const WALL: ClassId = 3;
    const STOREY: ClassId = 4;

    struct TestClasses {
        fields: HashMap<ClassId, Vec<FieldRef>>,
    }

    impl TestClasses {
        fn new() -> Self {
            let mut fields = HashMap::new();
            // WorkPackage: subject(0), assignee(1) = RAIL, status(2)
            fields.insert(
                WORK_PACKAGE,
                vec![
                    FieldRef::new("wp:subject", "Subject"),
                    FieldRef::new("wp:assignee", "Assignee"),
                    FieldRef::new("wp:status", "Status"),
                ],
            );
            // User: name(0), email(1) — no rails (leaf)
            fields.insert(
                USER,
                vec![
                    FieldRef::new("user:name", "Name"),
                    FieldRef::new("user:email", "Email"),
                ],
            );
            // Wall: material(0), part_of(1) = RAIL
            fields.insert(
                WALL,
                vec![
                    FieldRef::new("wall:material", "Material"),
                    FieldRef::new("wall:part_of", "Part of"),
                ],
            );
            // Storey: level(0) — leaf
            fields.insert(STOREY, vec![FieldRef::new("storey:level", "Level")]);
            Self { fields }
        }
    }

    impl ClassView for TestClasses {
        fn fields(&self, class: ClassId) -> &[FieldRef] {
            self.fields.get(&class).map_or(&[], |v| v.as_slice())
        }
        fn template(&self, _class: ClassId) -> DisplayTemplate {
            DisplayTemplate::Detail
        }
        fn dolce_category_id(&self, _class: ClassId) -> u8 {
            0
        }
    }

    /// A node = (class, presence mask). Rails keyed by (node, position) → target.
    struct TestGraph {
        nodes: HashMap<u32, (ClassId, WideFieldMask)>,
        rails: HashMap<(u32, u8), u32>,
    }

    impl TestGraph {
        fn new() -> Self {
            Self {
                nodes: HashMap::new(),
                rails: HashMap::new(),
            }
        }
        fn node(&mut self, key: u32, class: ClassId, present: &[u8]) {
            self.nodes.insert(
                key,
                (
                    class,
                    WideFieldMask::from(FieldMask::from_positions(present)),
                ),
            );
        }
        fn rail(&mut self, from: u32, position: u8, to: u32) {
            self.rails.insert((from, position), to);
        }
    }

    impl RailGraph for TestGraph {
        type Key = u32;
        fn class_of(&self, key: u32) -> ClassId {
            self.nodes.get(&key).map_or(0, |(c, _)| *c)
        }
        fn present_mask(&self, key: u32) -> WideFieldMask {
            self.nodes
                .get(&key)
                .map_or(WideFieldMask::EMPTY, |(_, m)| m.clone())
        }
        fn rail_target(&self, key: u32, position: u8) -> Option<u32> {
            self.rails.get(&(key, position)).copied()
        }
    }

    /// Build the standard registry + binding: one view per class.
    /// `user_view` deliberately masks ONLY name(0) — email(1) is present but NOT
    /// selected, so the walk must not emit it (proves masking selects a subset).
    fn registry() -> (ViewRegistry, HashMap<ClassId, ViewId>) {
        let mut reg = ViewRegistry::new();
        let mut binding = HashMap::new();
        binding.insert(
            WORK_PACKAGE,
            reg.register(NamedView::new(
                WORK_PACKAGE,
                WideFieldMask::from_positions(&[0, 1, 2]),
                DisplayTemplate::Detail,
            )),
        );
        binding.insert(
            USER,
            reg.register(NamedView::new(
                USER,
                WideFieldMask::from_positions(&[0]), // name only, NOT email
                DisplayTemplate::Card,
            )),
        );
        binding.insert(
            WALL,
            reg.register(NamedView::new(
                WALL,
                WideFieldMask::from_positions(&[0, 1]),
                DisplayTemplate::Detail,
            )),
        );
        binding.insert(
            STOREY,
            reg.register(NamedView::new(
                STOREY,
                WideFieldMask::from_positions(&[0]),
                DisplayTemplate::Summary,
            )),
        );
        (reg, binding)
    }

    fn collect<G: RailGraph<Key = u32>, V: ClassView>(
        graph: &G,
        cv: &V,
        reg: &ViewRegistry,
        binding: &HashMap<ClassId, ViewId>,
        root: u32,
        max_depth: usize,
    ) -> Vec<(u32, ClassId, u8, usize)> {
        let mut out = Vec::new();
        walk_rails(
            graph,
            cv,
            reg,
            |c| binding.get(&c).copied(),
            root,
            max_depth,
            &mut |v: FieldVisit<u32>| out.push((v.key, v.class, v.position, v.depth)),
        );
        out
    }

    #[test]
    fn nested_walk_emits_exactly_the_masked_fields_per_hop() {
        let cv = TestClasses::new();
        let (reg, binding) = registry();
        let mut g = TestGraph::new();
        // WorkPackage 10, all 3 fields present; assignee(1) rails to User 20.
        g.node(10, WORK_PACKAGE, &[0, 1, 2]);
        g.node(20, USER, &[0, 1]); // BOTH name+email present…
        g.rail(10, 1, 20);

        let got = collect(&g, &cv, &reg, &binding, 10, 8);
        assert_eq!(
            got,
            vec![
                // WorkPackage hop (depth 0): subject, assignee, status — all selected+present
                (10, WORK_PACKAGE, 0, 0),
                (10, WORK_PACKAGE, 1, 0),
                (10, WORK_PACKAGE, 2, 0),
                // User hop (depth 1): name ONLY — email(1) present but NOT in the view mask
                (20, USER, 0, 1),
            ],
            "each hop emits exactly (view.mask ∩ presence); email is pruned by the mask"
        );
    }

    #[test]
    fn presence_and_view_are_anded() {
        let cv = TestClasses::new();
        let (reg, binding) = registry();
        let mut g = TestGraph::new();
        // WorkPackage with assignee(1) NOT present → not emitted, and its rail is
        // not followed (so no User hop) even though the rail edge exists.
        g.node(10, WORK_PACKAGE, &[0, 2]); // subject + status only
        g.node(20, USER, &[0, 1]);
        g.rail(10, 1, 20);

        let got = collect(&g, &cv, &reg, &binding, 10, 8);
        assert_eq!(
            got,
            vec![(10, WORK_PACKAGE, 0, 0), (10, WORK_PACKAGE, 2, 0)],
            "an absent rail position is neither emitted nor followed"
        );
    }

    #[test]
    fn leaf_view_terminates() {
        let cv = TestClasses::new();
        let (reg, binding) = registry();
        let mut g = TestGraph::new();
        // A lone User: its view mask sets no rail bit → leaf, no recursion.
        g.node(20, USER, &[0, 1]);

        let got = collect(&g, &cv, &reg, &binding, 20, 8);
        assert_eq!(
            got,
            vec![(20, USER, 0, 0)],
            "a leaf view emits its selected fields and stops"
        );
    }

    #[test]
    fn second_shape_part_of_rail_walks() {
        let cv = TestClasses::new();
        let (reg, binding) = registry();
        let mut g = TestGraph::new();
        // Wall 30 --part_of(1)--> Storey 40
        g.node(30, WALL, &[0, 1]);
        g.node(40, STOREY, &[0]);
        g.rail(30, 1, 40);

        let got = collect(&g, &cv, &reg, &binding, 30, 8);
        assert_eq!(
            got,
            vec![(30, WALL, 0, 0), (30, WALL, 1, 0), (40, STOREY, 0, 1),],
            "the walk is shape-agnostic: part_of rails the same way assignee does"
        );
    }

    #[test]
    fn cyclic_graph_does_not_hang() {
        let cv = TestClasses::new();
        let (reg, binding) = registry();
        let mut g = TestGraph::new();
        // A part_of B, B part_of A — a cycle.
        g.node(30, WALL, &[0, 1]);
        g.node(31, WALL, &[0, 1]);
        g.rail(30, 1, 31);
        g.rail(31, 1, 30);

        // A large depth cap: only the cycle guard can stop this. If it hangs, the
        // test times out; if the guard works, each node is walked exactly once.
        let got = collect(&g, &cv, &reg, &binding, 30, 1_000);
        assert_eq!(
            got,
            vec![
                (30, WALL, 0, 0),
                (30, WALL, 1, 0),
                (31, WALL, 0, 1),
                (31, WALL, 1, 1),
            ],
            "the path-local cycle guard terminates the cycle; no infinite loop"
        );
    }

    #[test]
    fn depth_cap_is_respected() {
        let cv = TestClasses::new();
        let (reg, binding) = registry();
        let mut g = TestGraph::new();
        // A chain Wall 1 → Wall 2 → Wall 3 → Storey 4 via part_of(1).
        g.node(1, WALL, &[0, 1]);
        g.node(2, WALL, &[0, 1]);
        g.node(3, WALL, &[0, 1]);
        g.node(4, STOREY, &[0]);
        g.rail(1, 1, 2);
        g.rail(2, 1, 3);
        g.rail(3, 1, 4);

        // max_depth = 1 → root (depth 0) + one hop (depth 1), nothing deeper.
        let got = collect(&g, &cv, &reg, &binding, 1, 1);
        assert_eq!(
            got,
            vec![
                (1, WALL, 0, 0),
                (1, WALL, 1, 0),
                (2, WALL, 0, 1),
                (2, WALL, 1, 1),
            ],
            "depth cap 1 stops after one rail hop; nodes 3 and 4 are not reached"
        );

        // max_depth = 0 → root only.
        let root_only = collect(&g, &cv, &reg, &binding, 1, 0);
        assert_eq!(root_only, vec![(1, WALL, 0, 0), (1, WALL, 1, 0)]);
    }

    /// Falsifier for the PATH-LOCAL cycle guard (codex P2 on #776): a diamond
    /// DAG — root rails to A and B, both rail to the SAME target C. A global
    /// visited-set would emit C once (first DFS path wins) and silently prune
    /// the second path; the path-local guard emits C once PER PATH, each at
    /// that path's own depth.
    #[test]
    fn diamond_dag_emits_shared_target_once_per_path() {
        let cv = TestClasses::new();
        let (reg, binding) = registry();
        let mut g = TestGraph::new();
        // The diamond needs TWO rail positions on the root; WORK_PACKAGE has
        // three fields, and rail-ness is graph-side (rail_target), not
        // schema-side — so positions 1 and 2 both rail here.
        g.node(10, WORK_PACKAGE, &[0, 1, 2]);
        g.node(30, WALL, &[0, 1]); // A
        g.node(31, WALL, &[0, 1]); // B
        g.node(40, STOREY, &[0]); // C — the shared target
        g.rail(10, 1, 30); // root → A
        g.rail(10, 2, 31); // root → B
        g.rail(30, 1, 40); // A → C
        g.rail(31, 1, 40); // B → C

        let got = collect(&g, &cv, &reg, &binding, 10, 8);
        let c_visits: Vec<_> = got.iter().filter(|(k, ..)| *k == 40).collect();
        assert_eq!(
            c_visits,
            vec![&(40, STOREY, 0, 2), &(40, STOREY, 0, 2)],
            "the shared DAG target is emitted once per path, each at its own depth \
             — a global visited-set would emit it exactly once"
        );
    }

    /// Falsifier for view/class agreement (codex P2 on #776): a binding that
    /// resolves a class to a view registered for a DIFFERENT class is pruned
    /// fail-closed — the foreign mask must not be applied to this node.
    #[test]
    fn cross_class_view_binding_is_pruned() {
        let cv = TestClasses::new();
        let mut reg = ViewRegistry::new();
        // The only registered view belongs to WORK_PACKAGE…
        let wp_view = reg.register(NamedView::new(
            WORK_PACKAGE,
            WideFieldMask::from_positions(&[0, 1, 2]),
            DisplayTemplate::Detail,
        ));
        let mut g = TestGraph::new();
        g.node(20, USER, &[0, 1]);

        // …but the binding hands it out for USER too (stale/cross-class binding).
        let mut out = Vec::new();
        walk_rails(
            &g,
            &cv,
            &reg,
            |_| Some(wp_view),
            20,
            8,
            &mut |v: FieldVisit<u32>| out.push((v.key, v.class, v.position, v.depth)),
        );
        assert!(
            out.is_empty(),
            "a view whose class differs from the node's class is pruned fail-closed, \
             not applied as a foreign mask"
        );
    }

    /// The hard recursion ceiling (CodeRabbit on #776): a rail chain longer
    /// than [`MAX_WALK_DEPTH`] under `max_depth == usize::MAX` stops at the
    /// ceiling instead of growing the call stack without bound.
    #[test]
    fn max_depth_is_clamped_to_walk_ceiling() {
        let cv = TestClasses::new();
        let (reg, binding) = registry();
        let mut g = TestGraph::new();
        // A linear Wall chain twice the ceiling: 0 →part_of→ 1 →…→ 2·MAX.
        let len = (MAX_WALK_DEPTH * 2) as u32;
        for i in 0..=len {
            g.node(i, WALL, &[0, 1]);
            if i > 0 {
                g.rail(i - 1, 1, i);
            }
        }

        let got = collect(&g, &cv, &reg, &binding, 0, usize::MAX);
        let deepest = got.iter().map(|&(.., d)| d).max().unwrap();
        assert_eq!(
            deepest, MAX_WALK_DEPTH,
            "an unbounded caller max_depth is clamped to MAX_WALK_DEPTH"
        );
        // The node AT the ceiling is emitted; the one past it is not.
        assert!(got.iter().any(|&(k, ..)| k == MAX_WALK_DEPTH as u32));
        assert!(!got.iter().any(|&(k, ..)| k == MAX_WALK_DEPTH as u32 + 1));
    }

    #[test]
    fn fragment_or_composes() {
        // Two fragments over the same class, disjoint masks; union_of = OR.
        let mut reg = ViewRegistry::new();
        let a = reg.register(NamedView::new(
            WORK_PACKAGE,
            WideFieldMask::from_positions(&[0, 2]),
            DisplayTemplate::Card,
        ));
        let b = reg.register(NamedView::new(
            WORK_PACKAGE,
            WideFieldMask::from_positions(&[1, 3]),
            DisplayTemplate::Detail,
        ));
        let composed = reg.union_of(&[a, b]);
        assert_eq!(composed, WideFieldMask::from_positions(&[0, 1, 2, 3]));
        assert_eq!(composed.count(), 4);

        // Order-independent (union is commutative).
        assert_eq!(reg.union_of(&[b, a]), composed);
        // An out-of-range id contributes nothing (no panic).
        assert_eq!(reg.union_of(&[a, 999, b]), composed);
        // Empty spread = empty mask.
        assert!(reg.union_of(&[]).is_empty());
    }

    #[test]
    fn registry_register_get_roundtrip() {
        let mut reg = ViewRegistry::new();
        assert!(reg.is_empty());
        let id0 = reg.register(NamedView::new(
            USER,
            WideFieldMask::from_positions(&[0]),
            DisplayTemplate::Card,
        ));
        let id1 = reg.register(NamedView::new(
            WALL,
            WideFieldMask::from_positions(&[0, 1]),
            DisplayTemplate::Detail,
        ));
        assert_eq!((id0, id1), (0, 1));
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.get(id0).unwrap().class, USER);
        assert_eq!(reg.get(id1).unwrap().template, DisplayTemplate::Detail);
        assert!(reg.get(2).is_none());
    }

    #[test]
    fn traversal_is_deterministic() {
        let cv = TestClasses::new();
        let (reg, binding) = registry();
        let mut g = TestGraph::new();
        // A WorkPackage that rails to two Users (positions 0 and 1 both rails here,
        // via a bespoke rail map) — proves ascending-position order is stable.
        g.node(10, WORK_PACKAGE, &[0, 1, 2]);
        g.node(20, USER, &[0]);
        g.node(21, USER, &[0]);
        g.rail(10, 0, 21); // subject(0) rails to User 21
        g.rail(10, 1, 20); // assignee(1) rails to User 20

        let a = collect(&g, &cv, &reg, &binding, 10, 8);
        let b = collect(&g, &cv, &reg, &binding, 10, 8);
        assert_eq!(a, b, "repeated walks are byte-identical");
        // Rails followed in ascending position order: position 0's target (21)
        // recurses before position 1's target (20).
        let user_order: Vec<u32> = a
            .iter()
            .filter(|(_, c, _, _)| *c == USER)
            .map(|(k, _, _, _)| *k)
            .collect();
        assert_eq!(user_order, vec![21, 20], "ascending rail-position order");
    }
}
