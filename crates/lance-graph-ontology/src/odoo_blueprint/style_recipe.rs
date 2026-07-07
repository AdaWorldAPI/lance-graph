// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Style-recipe derivation — the **interpretation step** from typed Odoo
//! SoA (`OdooEntity` / `OdooMethod` / `OdooField` / `OdooDecorator`) into a
//! cognitive-fingerprint [`OdooStyleRecipe`] suitable for SoC synergy
//! compilation and downstream Op-codegen.
//!
//! # Where this fits in the pipeline
//!
//! ```text
//!   Odoo source
//!     │  (Stage 1, PR #426 — odoo-blueprint-extractor)
//!     ▼
//!   Typed Rust SoA  ← extracted/{account,sale,…}.rs (OdooEntity[ ])
//!     │  (THIS MODULE — interpretation, no new triplets stored)
//!     ▼
//!   OdooStyleRecipe[ ]  ← cognitive fingerprint per method
//!     │              (atom weights + regulatory anchors + dispatch hints)
//!     ▼
//!   Askama bucket templates (next commit) → Rust Ops + const recipes
//!     │
//!     ▼
//!   PaletteCompose SpMV at runtime (cognitive-shader-driver path)
//! ```
//!
//! # The "business logic stays in the triplets, you have to interpret it"
//! rule, concretely
//!
//! - The triplets (in `lance_graph::graph::spo::odoo_ontology` + the typed
//!   SoA above) are the lossless source. We do NOT store a `has_recipe`
//!   triple back in the graph — the recipe is *re-derived* deterministically
//!   every codegen run. That's the "interpretation" half of the rule.
//! - The recipe is Odoo-specific (D-Atoms like `EmitAmount`, `FiscalCtx`,
//!   `Onchange` are Odoo idioms). It lives in `lance-graph-ontology`'s
//!   `odoo_blueprint` because that's where Odoo-static interpretation
//!   belongs. A Rails frontend will write its own `style_recipe.rs`
//!   targeting the same SoC compiler.
//!
//! # D-Atom catalogue
//!
//! 12 basis vectors over which methods project (see [`DAtom`]). The
//! diagram excerpt in the architecture brief showed 9; we extended by 3
//! to cover the Odoo-specific dispatch shape (Onchange cascade,
//! Compute-vs-Validate split, Helper utility). Adding a 13th atom is one
//! variant here, one classification arm in [`derive_style_recipe`], and
//! one column in the downstream synergy matrix.
//!
//! # Determinism
//!
//! Pure function from `(&OdooEntity, &OdooMethod)` to [`OdooStyleRecipe`]. No
//! allocation beyond the recipe's own `Vec`s. Atom order in the output is
//! `DAtom` declaration order (the matching is by enum discriminant, not
//! by hash). `recipe_id` is a content-addressed FNV-1a over the sorted
//! atom-weight tuples — stable across runs, stable across machines.
//!
//! # Naming — NOT the contract `StyleRecipe`
//!
//! [`OdooStyleRecipe`] is deliberately named to NOT collide with
//! `lance_graph_contract::recipe::StyleRecipe`. They are different layers:
//!
//! - `contract::recipe::StyleRecipe` — a RUNTIME cognition object over the
//!   canonical 33-TSV / `I4x32` atom basis; composes into personas;
//!   dispatches through `cognitive-shader-driver` (reduces to a dot
//!   product). It is a thinking-style fingerprint.
//! - `odoo_blueprint::OdooStyleRecipe` (this type) — a CODEGEN-TIME IR over
//!   a SEPARATE, Odoo-specific 12-`DAtom` basis. Owned `String`/`Vec`,
//!   never a runtime SoA row.
//!
//! The two `DAtom`/`Atom` bases must NEVER be fused: per
//! `atom-basis-inventory.md`, **business is not a canonical atom** — it
//! rides as an OGIT/`Marking::Financial` sidecar. The Odoo `DAtom` basis
//! is a domain codegen basis, not a 13th canonical TSV dimension.
//!
//! # `recipe_id` is NOT an OGIT identity (FNV exemption)
//!
//! `E-CODEBOOK-INHERITS-FROM-OGIT` (EPIPHANIES.md) bans FNV-seeded /
//! hashed IDs for **identity** — every row identity must resolve through
//! `OntologyRegistry` (OGIT URI → stable codebook code). `recipe_id` is
//! exempt because it is NOT an identity: it is an ephemeral
//! content-addressed *collapse key* the dispatcher uses to deduplicate
//! structurally-identical recipes at codegen time. It is never stored in
//! the graph, never crosses a mailbox boundary, and never names a row.
//! Two methods sharing a `recipe_id` share a generated Op body — that is
//! the intended (and only) semantic. If `recipe_id` ever became a stored
//! or transmitted identity, this exemption would no longer hold and it
//! would have to route through `OntologyRegistry` instead.

use crate::odoo_blueprint::{
    OdooEntity, OdooField, OdooFieldKind, OdooMethod, OdooMethodKind, OdooReturnKind,
    OdooSemanticRole,
};

// ---------------------------------------------------------------------------
// DAtom — the basis vector catalogue
// ---------------------------------------------------------------------------

/// One basis vector of the cognitive-fingerprint space.
///
/// A method's [`OdooStyleRecipe`] is a sparse weighted vector over these
/// atoms. The 12 atoms span the Odoo-method dispatch axis; downstream SoC
/// synergy compilation projects them into the runtime palette.
///
/// Adding a 13th atom is a deliberate change: one variant here, one arm
/// in [`derive_style_recipe`], one entry in [`DAtom::ALL`], and one
/// column in the synergy matrix.
///
/// # Invariant
///
/// [`DAtom::ALL`] MUST be in declaration order — [`atom_idx`] casts the
/// enum discriminant to a `usize` index into a `[u8; 12]` array. The
/// `all_matches_discriminant_order` test pins this invariant; if a new
/// variant is added in the middle without updating `ALL`, that test
/// catches the drift before the silent indexing failure ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DAtom {
    /// Structural identity — every method gets weight 1 here. Anchors the
    /// recipe to a non-empty vector even when all other atoms miss.
    Entity,
    /// Regulatory rule reference — set when the parent entity's
    /// `provenance.regulation_iri` is non-empty (e.g. UStG §12, EU VAT).
    Law,
    /// Fiscal/state context — set when the parent entity carries a state
    /// machine the method participates in (period locks, GoBD).
    FiscalCtx,
    /// Writes a `Money`-typed field — derived from the computed field's
    /// `OdooFieldKind::Monetary` or `OdooReturnKind::Money`.
    EmitAmount,
    /// Reads/applies a rate field — set when a field with
    /// `OdooSemanticRole::Tax` is referenced (VAT rate, currency rate).
    ApplyRate,
    /// Reads/writes a quantity field — `OdooSemanticRole::Quantity` or
    /// `OdooFieldKind::Integer/Float` in a quantity context.
    Quantity,
    /// Reads/writes a money field — `OdooSemanticRole::Money` or
    /// `OdooFieldKind::Monetary`. Distinct from [`DAtom::EmitAmount`]
    /// (which marks the WRITE direction).
    Money,
    /// State-transition trigger — set when the method has non-empty
    /// `triggers` or appears as an `OdooTransition::trigger`.
    Event,
    /// Mutation action — [`OdooMethodKind::Action`] or
    /// [`OdooReturnKind::Action`].
    Action,
    /// Derivation — [`OdooMethodKind::Compute`] or
    /// [`OdooMethodKind::Inverse`].
    Compute,
    /// Guard / validator — [`OdooMethodKind::Constrain`] or any method
    /// that raises (in the SPO sense).
    Validate,
    /// `@api.onchange` cascade — [`OdooMethodKind::Onchange`].
    Onchange,
}

impl DAtom {
    /// All atoms in declaration order. Drives histogram tests, dispatch
    /// loops, and (later) the synergy-matrix column order.
    pub const ALL: [DAtom; 12] = [
        DAtom::Entity,
        DAtom::Law,
        DAtom::FiscalCtx,
        DAtom::EmitAmount,
        DAtom::ApplyRate,
        DAtom::Quantity,
        DAtom::Money,
        DAtom::Event,
        DAtom::Action,
        DAtom::Compute,
        DAtom::Validate,
        DAtom::Onchange,
    ];

    /// Stable snake_case identifier — used in codegen output paths,
    /// recipe-id derivation, and cross-language interop. Never reformat.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            DAtom::Entity => "entity",
            DAtom::Law => "law",
            DAtom::FiscalCtx => "fiscal_ctx",
            DAtom::EmitAmount => "emit_amount",
            DAtom::ApplyRate => "apply_rate",
            DAtom::Quantity => "quantity",
            DAtom::Money => "money",
            DAtom::Event => "event",
            DAtom::Action => "action",
            DAtom::Compute => "compute",
            DAtom::Validate => "validate",
            DAtom::Onchange => "onchange",
        }
    }
}

// ---------------------------------------------------------------------------
// OdooStyleRecipe — the cognitive fingerprint
// ---------------------------------------------------------------------------

/// Per-method cognitive fingerprint — a sparse weighted vector over
/// [`DAtom`], plus regulatory anchors and dispatch hints.
///
/// Owned `String`s + `Vec`s by design: this is the codegen-layer
/// representation. Runtime PaletteCompose reads the const projection
/// emitted by the askama templates, not this struct.
///
/// `recipe_id` is a content-addressed digest — two methods that project
/// to the same atom-weight set share the same recipe (a useful collapse
/// for the dispatcher).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OdooStyleRecipe {
    /// Fully-qualified method id: `"account.move._compute_amount"`.
    pub method_id: String,
    /// Sorted, deduplicated `(atom, weight)` tuples. Sorted by atom enum
    /// order; zero-weight atoms are NOT included.
    pub atoms: Vec<(DAtom, u8)>,
    /// IRI list lifted verbatim from the parent entity's
    /// `provenance.regulation_iri`. The downstream codegen uses these to
    /// emit `// per UStG §12` / `// per EU VAT Dir 2006/112/EC` comments
    /// alongside the const recipe.
    pub regulation_iris: Vec<String>,
    /// The method's return shape — drives function-signature codegen
    /// (`-> Money`, `-> Recordset`, `-> Action`, …).
    pub return_kind: OdooReturnKind,
    /// Content-addressed recipe digest. FNV-1a over the sorted
    /// `(atom_id, weight)` tuples. Stable across runs and machines.
    /// The dispatcher uses this as the `RecipeId(0x…)` const value.
    pub recipe_id: u32,
}

impl OdooStyleRecipe {
    /// Whether this recipe carries a regulatory-anchor signal. Codegen
    /// uses this to decide whether to emit the `// per <law>` doc-comment.
    #[must_use]
    pub fn has_law(&self) -> bool {
        self.atoms.iter().any(|(a, _)| *a == DAtom::Law)
    }
}

// ---------------------------------------------------------------------------
// Derivation — the deterministic projection
// ---------------------------------------------------------------------------

/// Project one method into its [`OdooStyleRecipe`] given its parent entity.
///
/// # Derivation rules (priority cascade, atoms accumulate)
///
/// 1. **`Entity = 1`** — every method gets the structural anchor.
/// 2. Method `kind` → primary axis weight:
///    - `Compute` / `Inverse` → `Compute = 4`
///    - `Constrain` → `Validate = 8`
///    - `Onchange` → `Onchange = 6`
///    - `Action` → `Action = 7`
///    - `Cron` → `Action = 4` (scheduled mutation, lower than user action)
///    - `Helper` / `Override` / `ApiModel` / `ApiModelCreateMulti` → no
///      atom (Entity=1 still anchors)
/// 3. `return_kind`:
///    - `Money` → `Money = 6` AND `EmitAmount = 7`
///    - `Number` → `Quantity = 4`
///    - `Action` → `Action += 2` (boost; the method emits an action dict)
///    - others → no atom
/// 4. `triggers` non-empty → `Event = 5`
/// 5. Cross-reference: walk parent entity's fields; for every field where
///    `field.computed == Some(method.name)`:
///    - `field.kind == Monetary` → `EmitAmount = max(7)`, `Money = max(6)`
///    - `field.semantic_role == Money` → `Money = max(6)`
///    - `field.semantic_role == Quantity` → `Quantity = max(5)`
///    - `field.semantic_role == Tax` → `ApplyRate = max(8)`
///    - `field.semantic_role == Status` → `FiscalCtx = max(5)`
/// 6. `entity.provenance.regulation_iri` non-empty →
///    `Law = 8` AND record IRIs.
/// 7. `entity.state_machine` is `Some` AND method.name is referenced by a
///    transition (either as trigger or check) → `FiscalCtx = 6`,
///    `Event += 2` (boost on transition methods).
///
/// All weights are `max`-merged (the strongest signal wins per atom);
/// zero-weight atoms drop out of the output `Vec`.
#[must_use]
pub fn derive_style_recipe(entity: &OdooEntity, method: &OdooMethod) -> OdooStyleRecipe {
    let mut weights: [u8; 12] = [0; 12];

    // 1. Structural anchor
    weights[atom_idx(DAtom::Entity)] = 1;

    // 2. Method kind → primary axis
    match method.kind {
        OdooMethodKind::Compute | OdooMethodKind::Inverse => {
            bump(&mut weights, DAtom::Compute, 4);
        }
        OdooMethodKind::Constrain => {
            bump(&mut weights, DAtom::Validate, 8);
        }
        OdooMethodKind::Onchange => {
            bump(&mut weights, DAtom::Onchange, 6);
        }
        OdooMethodKind::Action => {
            bump(&mut weights, DAtom::Action, 7);
        }
        OdooMethodKind::Cron => {
            bump(&mut weights, DAtom::Action, 4);
        }
        OdooMethodKind::Helper
        | OdooMethodKind::Override
        | OdooMethodKind::ApiModel
        | OdooMethodKind::ApiModelCreateMulti => {
            // Entity=1 anchors; no specialised axis.
        }
    }

    // 3. Return kind
    match method.return_kind {
        OdooReturnKind::Money => {
            bump(&mut weights, DAtom::Money, 6);
            bump(&mut weights, DAtom::EmitAmount, 7);
        }
        OdooReturnKind::Number => {
            bump(&mut weights, DAtom::Quantity, 4);
        }
        OdooReturnKind::Action => {
            bump(&mut weights, DAtom::Action, 2);
        }
        OdooReturnKind::Unit
        | OdooReturnKind::Self_
        | OdooReturnKind::Record
        | OdooReturnKind::Recordset
        | OdooReturnKind::Boolean
        | OdooReturnKind::Date
        | OdooReturnKind::Dict => {}
    }

    // 4. Triggers
    if !method.triggers.is_empty() {
        bump(&mut weights, DAtom::Event, 5);
    }

    // 5. Field cross-reference — what does this method compute?
    for field in entity.fields {
        if field.computed != Some(method.name) {
            continue;
        }
        accumulate_field_atoms(&mut weights, field);
    }

    // 6. Regulatory anchor
    let regulation_iris: Vec<String> = entity
        .provenance
        .regulation_iri
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    if !regulation_iris.is_empty() {
        bump(&mut weights, DAtom::Law, 8);
    }

    // 7. State-machine participation
    if let Some(sm) = entity.state_machine {
        let participates = sm.transitions.iter().any(|t| {
            t.trigger == method.name || t.guards.contains(&method.name)
        });
        if participates {
            bump(&mut weights, DAtom::FiscalCtx, 6);
            bump(&mut weights, DAtom::Event, 2);
        }
    }

    // Compose the sparse vector (atoms in DAtom::ALL order).
    let atoms: Vec<(DAtom, u8)> = DAtom::ALL
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| {
            let w = weights[i];
            if w > 0 { Some((a, w)) } else { None }
        })
        .collect();

    let method_id = format!("{}.{}", entity.model_name, method.name);
    let recipe_id = fnv1a_recipe(&atoms);

    OdooStyleRecipe {
        method_id,
        atoms,
        regulation_iris,
        return_kind: method.return_kind,
        recipe_id,
    }
}

/// Walk an entire corpus of entities and derive every method's recipe.
///
/// Output order: stable — sorted by `method_id` ascending. Two runs
/// produce byte-identical Vecs.
#[must_use]
pub fn derive_corpus_recipes(entities: &[&OdooEntity]) -> Vec<OdooStyleRecipe> {
    let mut recipes: Vec<OdooStyleRecipe> = entities
        .iter()
        .flat_map(|e| {
            e.methods
                .iter()
                .map(move |m| derive_style_recipe(e, m))
        })
        .collect();
    recipes.sort_by(|a, b| a.method_id.cmp(&b.method_id));
    recipes
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

#[inline]
const fn atom_idx(atom: DAtom) -> usize {
    // ALL is declared in DAtom variant order; the enum's discriminant
    // matches the array index.
    atom as usize
}

#[inline]
fn bump(weights: &mut [u8; 12], atom: DAtom, w: u8) {
    let i = atom_idx(atom);
    if weights[i] < w {
        weights[i] = w;
    }
}

/// Field-derived atom accumulation. Extracted as a helper because step 5
/// is the only one that walks an unbounded inner loop; keeps the cascade
/// in `derive_style_recipe` linear.
fn accumulate_field_atoms(weights: &mut [u8; 12], field: &OdooField) {
    // Field-kind signals
    if matches!(field.kind, OdooFieldKind::Monetary) {
        bump(weights, DAtom::EmitAmount, 7);
        bump(weights, DAtom::Money, 6);
    }

    // Semantic-role signals (drawn from the curated L-doc annotations)
    match field.semantic_role {
        OdooSemanticRole::Money => {
            bump(weights, DAtom::Money, 6);
        }
        OdooSemanticRole::Quantity => {
            bump(weights, DAtom::Quantity, 5);
        }
        OdooSemanticRole::Tax => {
            bump(weights, DAtom::ApplyRate, 8);
        }
        OdooSemanticRole::Status => {
            bump(weights, DAtom::FiscalCtx, 5);
        }
        OdooSemanticRole::Identity
        | OdooSemanticRole::Reference
        | OdooSemanticRole::Date
        | OdooSemanticRole::Policy
        | OdooSemanticRole::Document
        | OdooSemanticRole::Address
        | OdooSemanticRole::Audit
        | OdooSemanticRole::Other => {}
    }
}

/// FNV-1a 32-bit hash over the sorted `(atom_id, weight)` tuples. Used
/// as the content-addressed `recipe_id`. Two recipes with identical atom
/// vectors collide intentionally (the dispatcher collapses them).
fn fnv1a_recipe(atoms: &[(DAtom, u8)]) -> u32 {
    const OFFSET: u32 = 0x811c_9dc5;
    const PRIME: u32 = 0x0100_0193;
    let mut h = OFFSET;
    for (atom, w) in atoms {
        for b in atom.id().as_bytes() {
            h ^= u32::from(*b);
            h = h.wrapping_mul(PRIME);
        }
        h ^= u32::from(*w);
        h = h.wrapping_mul(PRIME);
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odoo_blueprint::{
        OdooConfidence, OdooEntityKind, OdooProvenance,
    };

    fn empty_entity() -> OdooEntity {
        OdooEntity {
            model_name: "test.model",
            kind: OdooEntityKind::Model,
            description: "test fixture",
            fields: &[],
            methods: &[],
            decorators: &[],
            state_machine: None,
            constraints: &[],
            provenance: OdooProvenance {
                l_doc: "test",
                l_doc_lines: (0, 0),
                odoo_source: &[],
                confidence: OdooConfidence::Curated,
                regulation_iri: &[],
            },
        }
    }

    fn method(name: &'static str, kind: OdooMethodKind, ret: OdooReturnKind) -> OdooMethod {
        OdooMethod {
            name,
            kind,
            return_kind: ret,
            triggers: &[],
        }
    }

    fn weight_of(recipe: &OdooStyleRecipe, atom: DAtom) -> u8 {
        recipe
            .atoms
            .iter()
            .find(|(a, _)| *a == atom)
            .map(|(_, w)| *w)
            .unwrap_or(0)
    }

    #[test]
    fn d_atom_ids_unique_and_stable() {
        let mut seen = std::collections::BTreeSet::new();
        for atom in DAtom::ALL {
            assert!(seen.insert(atom.id()), "duplicate id: {}", atom.id());
        }
        assert_eq!(seen.len(), 12);
    }

    /// Pin the invariant the [`DAtom`] docs promise: every variant appears
    /// in [`DAtom::ALL`] at the index equal to its enum discriminant. If a
    /// future change adds a variant in the middle of the enum but appends
    /// to `ALL`, `atom_idx` will silently mis-index `weights[..]`; this
    /// test catches that drift before it ships.
    #[test]
    fn all_matches_discriminant_order() {
        for (i, &atom) in DAtom::ALL.iter().enumerate() {
            assert_eq!(
                atom as usize, i,
                "DAtom::ALL[{i}] = {atom:?} but discriminant is {}",
                atom as usize,
            );
        }
    }

    #[test]
    fn every_recipe_carries_entity_anchor() {
        let e = empty_entity();
        let m = method("_helper", OdooMethodKind::Helper, OdooReturnKind::Unit);
        let r = derive_style_recipe(&e, &m);
        assert_eq!(weight_of(&r, DAtom::Entity), 1);
    }

    #[test]
    fn compute_method_gets_compute_atom() {
        let e = empty_entity();
        let m = method("_compute_x", OdooMethodKind::Compute, OdooReturnKind::Unit);
        let r = derive_style_recipe(&e, &m);
        assert_eq!(weight_of(&r, DAtom::Compute), 4);
        assert_eq!(weight_of(&r, DAtom::Validate), 0);
    }

    #[test]
    fn constrain_method_gets_validate_atom() {
        let e = empty_entity();
        let m = method("_check_x", OdooMethodKind::Constrain, OdooReturnKind::Unit);
        let r = derive_style_recipe(&e, &m);
        assert_eq!(weight_of(&r, DAtom::Validate), 8);
        assert_eq!(weight_of(&r, DAtom::Compute), 0);
    }

    #[test]
    fn money_return_emits_both_money_and_emit_amount() {
        let e = empty_entity();
        let m = method("_compute_total", OdooMethodKind::Compute, OdooReturnKind::Money);
        let r = derive_style_recipe(&e, &m);
        assert_eq!(weight_of(&r, DAtom::Money), 6);
        assert_eq!(weight_of(&r, DAtom::EmitAmount), 7);
        assert_eq!(weight_of(&r, DAtom::Compute), 4);
    }

    #[test]
    fn action_return_boosts_action_atom() {
        let e = empty_entity();
        // Action method whose return is also an action dict.
        let m = method("button_post", OdooMethodKind::Action, OdooReturnKind::Action);
        let r = derive_style_recipe(&e, &m);
        // 7 from kind + 2 from return, max-merged = 7 (max wins).
        assert_eq!(weight_of(&r, DAtom::Action), 7);
    }

    #[test]
    fn field_cross_reference_lifts_field_kind_atoms() {
        // An entity with a Monetary field computed by our method.
        static FIELDS: &[OdooField] = &[OdooField {
            name: "amount_total",
            kind: OdooFieldKind::Monetary,
            target: None,
            required: false,
            computed: Some("_compute_amount"),
            depends: &[],
            semantic_role: OdooSemanticRole::Money,
        }];
        let mut e = empty_entity();
        e.fields = FIELDS;

        let m = method("_compute_amount", OdooMethodKind::Compute, OdooReturnKind::Unit);
        let r = derive_style_recipe(&e, &m);
        // From Monetary kind on the field.
        assert_eq!(weight_of(&r, DAtom::EmitAmount), 7);
        assert_eq!(weight_of(&r, DAtom::Money), 6);
    }

    #[test]
    fn regulation_iri_lifts_law_atom_and_anchors() {
        static IRIS: &[&str] = &["ogit:law/de/UStG#§12", "ogit:law/eu/2006_112_EC"];
        let mut e = empty_entity();
        e.provenance.regulation_iri = IRIS;

        let m = method("_compute_tax", OdooMethodKind::Compute, OdooReturnKind::Money);
        let r = derive_style_recipe(&e, &m);
        assert_eq!(weight_of(&r, DAtom::Law), 8);
        assert_eq!(r.regulation_iris.len(), 2);
        assert!(r.has_law());
    }

    #[test]
    fn recipe_id_is_deterministic_and_collapses_identical_shapes() {
        let e = empty_entity();
        let m1 = method("_compute_x", OdooMethodKind::Compute, OdooReturnKind::Money);
        let m2 = method("_compute_y", OdooMethodKind::Compute, OdooReturnKind::Money);
        let r1 = derive_style_recipe(&e, &m1);
        let r2 = derive_style_recipe(&e, &m2);
        // Same atom shape → same recipe_id (the collapse the dispatcher exploits).
        assert_eq!(r1.recipe_id, r2.recipe_id);
        // Method ids still distinct.
        assert_ne!(r1.method_id, r2.method_id);
    }

    #[test]
    fn recipe_id_differs_when_atoms_differ() {
        let e = empty_entity();
        let m_compute = method("_compute_x", OdooMethodKind::Compute, OdooReturnKind::Unit);
        let m_check = method("_check_x", OdooMethodKind::Constrain, OdooReturnKind::Unit);
        let r_c = derive_style_recipe(&e, &m_compute);
        let r_v = derive_style_recipe(&e, &m_check);
        assert_ne!(r_c.recipe_id, r_v.recipe_id);
    }

    #[test]
    fn corpus_derivation_is_sorted_and_deterministic() {
        // Build a tiny two-method entity to verify the corpus walker.
        static METHODS: &[OdooMethod] = &[
            OdooMethod {
                name: "_compute_b",
                kind: OdooMethodKind::Compute,
                return_kind: OdooReturnKind::Unit,
                triggers: &[],
            },
            OdooMethod {
                name: "_compute_a",
                kind: OdooMethodKind::Compute,
                return_kind: OdooReturnKind::Unit,
                triggers: &[],
            },
        ];
        let mut e = empty_entity();
        e.methods = METHODS;
        let recipes = derive_corpus_recipes(&[&e]);
        assert_eq!(recipes.len(), 2);
        // Sorted by method_id ascending.
        assert!(recipes[0].method_id < recipes[1].method_id);
        // Re-run determinism.
        let again = derive_corpus_recipes(&[&e]);
        assert_eq!(recipes, again);
    }

    /// Shipped-corpus distribution: derive recipes across a sample of
    /// extracted entities and assert which atoms fire today.
    ///
    /// # Gap surfaced by this test
    ///
    /// The Stage-1 extractor (PR #426) populated structural fields
    /// (`OdooMethod::name`, `kind`, default `return_kind: Unit`,
    /// `triggers: &[]`) but the semantic enrichment that lights the
    /// financial atoms is mostly absent:
    ///
    /// - Most methods have `return_kind: Unit` (not `Money` / `Number` /
    ///   `Action`) → `Money` / `Quantity` / `Action`-via-return don't fire.
    /// - `OdooMethod::triggers` is empty across the extracted set →
    ///   `Event` doesn't fire.
    /// - Field `computed: Some(method_name)` cross-refs are sparse →
    ///   `EmitAmount` rarely fires, even though `Monetary` fields exist.
    /// - `OdooEntity::state_machine` is `None` on the extracted entities
    ///   → `FiscalCtx` + transition-boosted `Event` don't fire.
    ///
    /// **Atoms that DO fire today**: Entity, Compute, Validate, Onchange,
    /// Action (kind-driven), Law (where curated regulation_iri exists).
    ///
    /// **Closes the gap**: the Stage-2 extractor pass that populates
    /// `return_kind` from compute-method return type annotations + the
    /// L-doc enrichment that lifts field `semantic_role` to `Money` /
    /// `Quantity` / `Tax` per the curated lane docs.
    ///
    /// The test asserts both halves: lit atoms must include the
    /// kind-driven set, AND the unfit atoms must remain empty (so a
    /// future false-positive in the cascade surfaces immediately).
    #[test]
    fn shipped_corpus_resolves_kind_driven_atoms_today() {
        use crate::odoo_blueprint::extracted::{account, base, l10n_de, sale, stock};

        let entities: Vec<&OdooEntity> = vec![
            &account::EXT_ACCOUNT_MOVE,
            &account::EXT_ACCOUNT_MOVE_LINE,
            &account::EXT_ACCOUNT_ACCOUNT,
            &account::EXT_ACCOUNT_JOURNAL,
            &sale::EXT_SALE_ORDER,
            &sale::EXT_SALE_ORDER_LINE,
            &stock::EXT_STOCK_PICKING,
            &base::EXT_RES_PARTNER,
            &base::EXT_RES_COMPANY,
            &l10n_de::EXT_ACCOUNT_TAX,
        ];

        let recipes = derive_corpus_recipes(&entities);
        assert!(
            !recipes.is_empty(),
            "no methods derived from chosen entities — extracted data missing?"
        );

        let mut seen_atoms: std::collections::BTreeSet<DAtom> =
            std::collections::BTreeSet::new();
        for r in &recipes {
            for (a, _) in &r.atoms {
                seen_atoms.insert(*a);
            }
        }

        // Must-fire today (kind-driven + structural; covered by Stage-1
        // extractor output).
        for must in [
            DAtom::Entity,
            DAtom::Compute,
            DAtom::Validate,
            DAtom::Onchange,
            DAtom::Action,
        ] {
            assert!(
                seen_atoms.contains(&must),
                "kind-driven atom {must:?} not fired — cascade regression?",
            );
        }

        // Pin the Stage-2 gap: today's extracted data does NOT light
        // these atoms. When Stage-2 enrichment lands (return_kind /
        // semantic_role population), this set shrinks — flip the
        // assertions one by one as each atom starts firing.
        for stage2 in [
            DAtom::Money,
            DAtom::Quantity,
            DAtom::ApplyRate,
            DAtom::EmitAmount,
            DAtom::Event,
            DAtom::FiscalCtx,
        ] {
            assert!(
                !seen_atoms.contains(&stage2),
                "atom {stage2:?} fired unexpectedly — Stage-2 enrichment \
                 landed? Update this test (move from `stage2` to `must`).",
            );
        }
    }
}
