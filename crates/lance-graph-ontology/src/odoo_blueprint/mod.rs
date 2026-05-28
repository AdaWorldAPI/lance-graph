//! `odoo_blueprint` — typed Odoo entity DTOs as the substrate for the
//! `Odoo → OGIT → OWL → DOLCE → FIBU/FIBO` normalization chain.
//!
//! ## Why this module exists
//!
//! The lance-graph workspace has shipped most of the cognitive substrate:
//! OGIT families + Layer-2 axioms (PR #414/#416), OWL hydrators
//! (PR #407/#408), DOLCE classifier (PR #412), FIBU/FIBO alignment
//! (PR #416), 33-TSV atoms + 34 `Tactic` kernels (PR #411), `CausalEdge64`
//! (`causal-edge` crate), `cognitive-shader-driver`. The 25-savant roster
//! lives in [`lance_graph_contract::savants`]; the 25 AXIS-B evidence
//! contracts + 15 lane drafts are curated as **prose** under
//! `.claude/odoo/L*.md` + `.claude/odoo/savants/*.md`.
//!
//! What was missing: **typed Odoo entity DTOs that the inheritance chain
//! operates on**. Today every downstream layer (`classify_odoo`,
//! `hydrate_odoo`, DOLCE classifier, FIBU alignment) string-keys against
//! `model_name`. This module establishes the typed surface those layers
//! will consume in `D-ODOO-BP-1c/d/e`, and which the savant compositions
//! in `odoo-savant-reasoners-v2` Group F reach for through the normalized
//! chain.
//!
//! ## What lives here today (`D-ODOO-BP-1a`)
//!
//! - [`OdooEntity`] — one Odoo model captured as a typed const declaration
//!   carrying fields, methods, decorators, state machine, constraints,
//!   and provenance back to the L-doc line range + Odoo source paths.
//! - [`OdooField`] / [`OdooMethod`] / [`OdooDecorator`] /
//!   [`OdooStateMachine`] / [`OdooConstraint`] — the sub-types.
//! - [`OdooProvenance`] + [`OdooSourceRef`] — the audit trail per
//!   `I-VSA-IDENTITIES` ("content in typed registries, never bundled").
//! - The enums: [`OdooFieldKind`], [`OdooSemanticRole`], [`OdooMethodKind`],
//!   [`OdooReturnKind`], [`OdooDecoratorKind`], [`OdooStateSemantic`],
//!   [`OdooConstraintKind`], [`OdooConfidence`].
//!
//! ## What does NOT live here yet
//!
//! - Per-lane entity consts (15 lanes × ~5–30 entities each) — those land
//!   in `D-ODOO-BP-1b` as sub-modules `odoo_blueprint::l1`…`::l15`.
//! - Odoo source extraction (`D-ODOO-BP-1f`) lives in
//!   `tools/odoo-blueprint-extractor/` and emits candidate consts with
//!   [`OdooConfidence::Extracted`].
//! - JITson / recipe wiring (`D-ODOO-BP-1g`) lives in
//!   `lance-graph-contract::jit` and will take `&OdooEntity` as input.
//!
//! Plan: `.claude/plans/odoo-business-logic-blueprint-v1.md`.

// ─── Per-lane sub-modules (D-ODOO-BP-1b) ─────────────────────────────────
//
// One sub-module per L-doc (`.claude/odoo/L*.md`). Each carries an
// `ENTITIES: &[OdooEntity]` slice with all entities documented in that
// lane. Stubs land first (additive, all empty); the projection wave
// populates each from the L-doc prose.

pub mod l1;
pub mod l2;
pub mod l3;
pub mod l4;
pub mod l5;
pub mod l6;
pub mod l7;
pub mod l8;
pub mod l9;
pub mod l10;
pub mod l11;
pub mod l12;
pub mod l13;
pub mod l14;
pub mod l15;

// ─── Top-level entity ─────────────────────────────────────────────────────

/// Which ORM base class the entity inherits from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdooEntityKind {
    /// `class X(models.Model)` — persisted, has a backing table.
    Model,
    /// `class X(models.TransientModel)` — wizard/in-memory, no persistence.
    Transient,
    /// `class X(models.AbstractModel)` — mixin, no own table.
    Abstract,
}

/// One Odoo model captured as a typed const declaration.
///
/// Holds the full set of fields, methods, decorators, state machine, and
/// constraints from the prose curation (`.claude/odoo/L*.md`), projected
/// into a structure the inheritance chain (OGIT → OWL → DOLCE → FIBU/FIBO)
/// can operate over without string-keying against `model_name`.
#[derive(Debug, Clone, Copy)]
pub struct OdooEntity {
    /// Odoo model name (e.g. `"account.fiscal.position"`).
    pub model_name: &'static str,
    /// Which ORM base class this entity inherits from.
    pub kind: OdooEntityKind,
    /// One-line semantic intent — what this entity IS in business terms.
    pub description: &'static str,
    /// Fields declared on this model.
    pub fields: &'static [OdooField],
    /// Methods (`_compute_*`, `action_*`, `_check_*`, `_onchange_*`, …).
    pub methods: &'static [OdooMethod],
    /// Decorators (`@api.depends`, `@api.constrains`, `@api.onchange`, …).
    pub decorators: &'static [OdooDecorator],
    /// State machine if the model carries one (typically `state` field).
    pub state_machine: Option<&'static OdooStateMachine>,
    /// Constraints — DB-level ([`OdooConstraintKind::Sql`]), Python-level
    /// ([`OdooConstraintKind::Python`]), or domain-restricting.
    pub constraints: &'static [OdooConstraint],
    /// Audit provenance — L-doc + Odoo source + confidence band.
    pub provenance: OdooProvenance,
}

// ─── Sub-types ────────────────────────────────────────────────────────────

/// One field on an Odoo model.
#[derive(Debug, Clone, Copy)]
pub struct OdooField {
    pub name: &'static str,
    pub kind: OdooFieldKind,
    /// For relational fields, the target model name (e.g. `"res.partner"`).
    pub target: Option<&'static str>,
    pub required: bool,
    /// If this is a computed field, the `_compute_*` method name.
    pub computed: Option<&'static str>,
    /// `@api.depends(...)` targets the compute reacts to.
    pub depends: &'static [&'static str],
    /// Semantic role drawn from the L-doc — what this field MEANS in
    /// business terms beyond its [`OdooFieldKind`].
    pub semantic_role: OdooSemanticRole,
}

/// One method on an Odoo model.
#[derive(Debug, Clone, Copy)]
pub struct OdooMethod {
    pub name: &'static str,
    pub kind: OdooMethodKind,
    pub return_kind: OdooReturnKind,
    /// For [`OdooMethodKind::Action`] methods: the state-machine
    /// transition names they fire (matched against
    /// [`OdooTransition::trigger`]).
    pub triggers: &'static [&'static str],
}

/// One decorator on a method.
#[derive(Debug, Clone, Copy)]
pub struct OdooDecorator {
    pub kind: OdooDecoratorKind,
    /// Decorator arguments — for `@api.depends("a", "b")` the targets
    /// are `["a", "b"]`.
    pub targets: &'static [&'static str],
}

/// State machine on an Odoo model.
#[derive(Debug, Clone, Copy)]
pub struct OdooStateMachine {
    /// The state field name (typically `"state"`).
    pub state_field: &'static str,
    pub states: &'static [OdooState],
    pub transitions: &'static [OdooTransition],
}

/// One state in a state machine.
#[derive(Debug, Clone, Copy)]
pub struct OdooState {
    pub name: &'static str,
    pub semantic: OdooStateSemantic,
}

/// One transition in a state machine.
#[derive(Debug, Clone, Copy)]
pub struct OdooTransition {
    pub from: &'static str,
    pub to: &'static str,
    /// The method name that fires the transition (typically `action_*`).
    pub trigger: &'static str,
    /// `_check_*` method names that must pass before the transition fires.
    pub guards: &'static [&'static str],
}

/// One constraint on the model.
#[derive(Debug, Clone, Copy)]
pub struct OdooConstraint {
    pub kind: OdooConstraintKind,
    /// Human-readable constraint description.
    pub condition: &'static str,
    /// For [`OdooConstraintKind::Python`] constraints, the
    /// `@api.constrains` method name.
    pub source_method: Option<&'static str>,
}

// ─── Provenance ───────────────────────────────────────────────────────────

/// Audit provenance per `I-VSA-IDENTITIES` — every entity carries the
/// source it was derived from plus a confidence band.
#[derive(Debug, Clone, Copy)]
pub struct OdooProvenance {
    /// L-doc filename (e.g. `"L9-PARTNER-FISCALPOS.md"`).
    pub l_doc: &'static str,
    /// Line range in the L-doc that documents this entity.
    pub l_doc_lines: (u32, u32),
    /// Odoo source paths covering this entity (multi if the entity spans
    /// files via `_inherit`).
    pub odoo_source: &'static [OdooSourceRef],
    pub confidence: OdooConfidence,
    /// German tax/accounting law anchors that bind this entity's semantics —
    /// UStG / HGB / GoBD / AO / EN 16931 IRIs into the OGIT-inherited
    /// regulation codebook (per `E-CODEBOOK-INHERITS-FROM-OGIT`).
    /// Empty when the entity has no direct regulatory anchor.
    pub regulation_iri: &'static [&'static str],
}

/// One Odoo source-file reference within [`OdooProvenance`].
#[derive(Debug, Clone, Copy)]
pub struct OdooSourceRef {
    pub path: &'static str,
    pub line_range: (u32, u32),
}

// ─── Enums ────────────────────────────────────────────────────────────────

/// Odoo field kinds (mirrors the `odoo.fields.*` taxonomy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdooFieldKind {
    Char,
    Text,
    Boolean,
    Integer,
    Float,
    Monetary,
    Date,
    Datetime,
    Selection,
    Binary,
    Html,
    Many2one,
    One2many,
    Many2many,
    Reference,
    /// Computed field — uses [`OdooField::computed`] + `depends` for
    /// semantics.
    Computed,
    /// Property — partner-scoped Many2one with default.
    Property,
}

/// Semantic role beyond field kind — what the field MEANS in business
/// terms.
///
/// The L-docs surface roles like "identity" (the natural key), "policy"
/// (a decision input, e.g. `auto_apply`), "quantity" (a numeric measure),
/// "reference" (a relational pointer for evidence chains).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdooSemanticRole {
    Identity,
    Reference,
    Quantity,
    Date,
    Policy,
    Status,
    Money,
    Document,
    Address,
    Tax,
    Audit,
    Other,
}

/// Odoo method kinds — drawn from the `_compute_`, `_inverse_`, `_check_`,
/// `action_`, `_onchange_` conventions plus the `@api.model*` flavours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdooMethodKind {
    Compute,
    Inverse,
    Constrain,
    Onchange,
    Action,
    Cron,
    ApiModel,
    ApiModelCreateMulti,
    Override,
    Helper,
}

/// What an Odoo method returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdooReturnKind {
    Unit,
    Self_,
    Record,
    Recordset,
    Boolean,
    Number,
    Money,
    Date,
    Dict,
    Action,
}

/// Decorator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdooDecoratorKind {
    ApiDepends,
    ApiConstrains,
    ApiOnchange,
    ApiModel,
    ApiModelCreateMulti,
    ApiReturns,
    ApiAutovacuum,
}

/// Semantic of a state in a state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdooStateSemantic {
    Draft,
    Active,
    Posted,
    InProgress,
    Completed,
    Cancelled,
    Terminal,
}

/// Constraint kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdooConstraintKind {
    /// DB-level unique/not-null (in `_sql_constraints`).
    Sql,
    /// Python-level `@api.constrains` method.
    Python,
    /// Domain restricting (an Odoo `domain=...` on a Many2one or field).
    Domain,
}

/// Provenance confidence band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdooConfidence {
    /// Sourced from the human-curated L-docs (`D-ODOO-BP-1b`).
    Curated,
    /// Extracted automatically from Odoo source via Python's stdlib `ast`
    /// module (`D-ODOO-BP-1f` / `D-ODOO-EXT-2`).
    Extracted,
    /// Inferred but not yet validated against either source.
    Conjecture,
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_entity_compiles_as_const() {
        const FISCAL_POSITION: OdooEntity = OdooEntity {
            model_name: "account.fiscal.position",
            kind: OdooEntityKind::Model,
            description: "Tax mapping for partner / company combination",
            fields: &[OdooField {
                name: "country_id",
                kind: OdooFieldKind::Many2one,
                target: Some("res.country"),
                required: false,
                computed: None,
                depends: &[],
                semantic_role: OdooSemanticRole::Reference,
            }],
            methods: &[OdooMethod {
                name: "_get_fiscal_position",
                kind: OdooMethodKind::ApiModel,
                return_kind: OdooReturnKind::Record,
                triggers: &[],
            }],
            decorators: &[OdooDecorator {
                kind: OdooDecoratorKind::ApiModel,
                targets: &[],
            }],
            state_machine: None,
            constraints: &[],
            provenance: OdooProvenance {
                l_doc: "L9-PARTNER-FISCALPOS.md",
                l_doc_lines: (211, 259),
                odoo_source: &[OdooSourceRef {
                    path: "addons/account/models/account_fiscal_position.py",
                    line_range: (1, 200),
                }],
                confidence: OdooConfidence::Curated,
                regulation_iri: &[],
            },
        };
        assert_eq!(FISCAL_POSITION.model_name, "account.fiscal.position");
        assert_eq!(FISCAL_POSITION.fields.len(), 1);
        assert_eq!(FISCAL_POSITION.fields[0].kind, OdooFieldKind::Many2one);
        assert!(FISCAL_POSITION.state_machine.is_none());
        assert_eq!(
            FISCAL_POSITION.provenance.confidence,
            OdooConfidence::Curated
        );
    }

    #[test]
    fn state_machine_entity_compiles() {
        const INVOICE_STATE: OdooStateMachine = OdooStateMachine {
            state_field: "state",
            states: &[
                OdooState { name: "draft", semantic: OdooStateSemantic::Draft },
                OdooState { name: "posted", semantic: OdooStateSemantic::Posted },
                OdooState { name: "cancel", semantic: OdooStateSemantic::Cancelled },
            ],
            transitions: &[OdooTransition {
                from: "draft",
                to: "posted",
                trigger: "action_post",
                guards: &["_check_balanced", "_check_lock_date"],
            }],
        };
        assert_eq!(INVOICE_STATE.states.len(), 3);
        assert_eq!(INVOICE_STATE.transitions[0].trigger, "action_post");
        assert_eq!(INVOICE_STATE.transitions[0].guards.len(), 2);
    }

    #[test]
    fn empty_entity_compiles() {
        // Confirms the surface supports the zero case (no fields, methods,
        // decorators, state machine, or constraints) for entities still
        // being curated.
        const EMPTY: OdooEntity = OdooEntity {
            model_name: "test.model",
            kind: OdooEntityKind::Model,
            description: "test",
            fields: &[],
            methods: &[],
            decorators: &[],
            state_machine: None,
            constraints: &[],
            provenance: OdooProvenance {
                l_doc: "test.md",
                l_doc_lines: (0, 0),
                odoo_source: &[],
                confidence: OdooConfidence::Conjecture,
                regulation_iri: &[],
            },
        };
        assert_eq!(EMPTY.fields.len(), 0);
        assert_eq!(EMPTY.provenance.confidence, OdooConfidence::Conjecture);
    }
}
