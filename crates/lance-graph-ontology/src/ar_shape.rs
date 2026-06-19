// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `ar_shape` — minimal smoke convergence between Rails-shaped curators
//! (OpenSourceBilling, future Spree/Solidus, future Redmine/OpenProject)
//! and Odoo via the OGAR canonical-concept layer.
//!
//! # What this is
//!
//! The first concrete instance of the synergy-registry framing the doctrine
//! (`docs/OGAR_AR_SHAPE_ENDGAME.md` §2 corrections, dated 2026-06-19) names:
//!
//! - Per-curator labels (e.g. OSB `InvoiceLineItem.item_unit_cost` / Odoo
//!   `account.move.line.price_unit`) are **leaf detail** that hangs off the
//!   OGAR class-inheritance edge.
//! - The ≥2-curator promotion rule (`E-OGAR-AR-SHAPE-ENDGAME` §3) requires
//!   ≥2 independent curators to surface the SAME primitive under different
//!   syntactic forms before a `CanonicalConcept` is admitted.
//! - Claude Code owns convergence detection; OGAR stores only stable
//!   canonical results after code/tests prove the overlap (per operator
//!   smoke-pass directive, 2026-06-19).
//!
//! # The shape today
//!
//! Hand-built fixtures per the operator directive *"Prefer hand-built Class
//! fixtures for the first smoke test if full repository extraction is too
//! heavy"*. The fixtures are typed `Class` instances carrying:
//!
//! - `source_curator` (`OpenSourceBilling`, `Odoo`, …)
//! - `source_domain` (`Billing`, `Erp`, …)
//! - `curator_label` — the curator's own class name (`InvoiceLineItem` /
//!   `account.move.line`), kept verbatim as leaf detail.
//! - `shape: ClassShape` — the structural form the overlap detector
//!   compares (today: only `ClassShape::LineItem`).
//! - `inherits` — curator-side composition labels.
//!
//! The overlap detector (`overlap_commercial_line_item`) returns
//! `Some(CanonicalConcept::CommercialLineItem)` exactly when the two
//! fixtures (a) come from *different* curators (≥2-curator promotion rule)
//! and (b) share the structural `LineItem` shape (both carry parent-doc
//! reference, quantity, unit-price, ≥1 tax binding, and a label field).
//!
//! # Scope discipline
//!
//! - **One** `CanonicalConcept` today (`CommercialLineItem`). The minimal
//!   step per operator acceptance #4 ("if absent, add the minimal canonical
//!   class or slot needed").
//! - **No** Rails / Odoo syntax leaks into OGAR Core: the canonical
//!   concept is a name only; curator labels stay on the fixture side.
//! - **Additive only**: this module introduces no changes to existing
//!   ontology types and does not require any change to
//!   `lance-graph-contract`.
//! - Future curators (Spree, Solidus, Redmine, OpenProject, future SAP)
//!   plug in by adding a `SourceCurator` variant and a fixture; the
//!   detector is reusable as-is for the LineItem shape, and grows by adding
//!   sibling `overlap_*` functions per `CanonicalConcept`.

/// The high-level domain a curator belongs to. Used as a coarse filter
/// (e.g. ERP vs commercial document vs project tracking) before the
/// structural shape test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceDomain {
    /// Customer-facing billing apps. OpenSourceBilling sits here.
    Billing,
    /// Full ERP with accounting + posting + tax finalization. Odoo sits
    /// here.
    Erp,
    /// E-commerce / sales-order-shaped apps. Spree, Solidus sit here.
    Commerce,
    /// Project / task / time tracking apps. Redmine, OpenProject sit
    /// here.
    Project,
}

/// A specific curator (a concrete upstream codebase). Maps 1-1 to a
/// namespace prefix at the harvest seam (`open_source_billing:` / `odoo:`
/// / …). New variants are added as new curators come online.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceCurator {
    /// `AdaWorldAPI/open-source-billing` — Ruby/Rails AR billing app.
    OpenSourceBilling,
    /// Odoo ORM (Python). Sourced via `tools/odoo-blueprint-extractor`.
    Odoo,
    /// Spree commerce platform (Rails AR). Future.
    Spree,
    /// Solidus (Spree fork, Rails AR). Future.
    Solidus,
    /// Redmine PM (Rails AR). Future.
    Redmine,
    /// OpenProject PM (Rails AR). Future.
    OpenProject,
}

impl SourceCurator {
    /// The namespace prefix this curator emits at the harvest seam. Stable
    /// `&'static str` per workspace canon (E-OGAR-AR-SHAPE-ENDGAME §11.1
    /// Inc 3: adapter target ids are `&'static str`).
    #[must_use]
    pub const fn namespace_prefix(self) -> &'static str {
        match self {
            Self::OpenSourceBilling => "open_source_billing:",
            Self::Odoo => "odoo:",
            Self::Spree => "spree:",
            Self::Solidus => "solidus:",
            Self::Redmine => "redmine:",
            Self::OpenProject => "openproject:",
        }
    }
}

/// The OGAR canonical concept — what ≥2 curators must agree on to promote.
///
/// Append-only. Each variant lands ONLY after at least two independent
/// curator fixtures overlap on its structural shape AND tests pin the
/// detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalConcept {
    /// `CommercialLineItem` — a per-line entry on a commercial document
    /// (invoice / journal / sales order line) carrying
    /// quantity × unit_price + tax bindings + parent-doc ref + label.
    /// Promoted from `{ osb:InvoiceLineItem, odoo:account.move.line }`
    /// pair on 2026-06-19.
    CommercialLineItem,
}

/// A typed fixture for one curator's class declaration. Hand-built today;
/// future ruff-side extraction will emit these from real corpora.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Class {
    /// Which curator surfaced this class. Drives the ≥2-curator promotion
    /// rule (same-curator pairs cannot promote).
    pub source_curator: SourceCurator,
    /// The high-level domain. Coarse filter / observability.
    pub source_domain: SourceDomain,
    /// The curator's own name for the class. Kept verbatim. Leaf detail
    /// (per doctrine §2 correction 1).
    pub curator_label: &'static str,
    /// The structural form the overlap detector compares.
    pub shape: ClassShape,
    /// Curator-side composition / inheritance — Rails `acts_as_*` /
    /// `include` / STI parents; Odoo `_inherit` chains. Names verbatim.
    pub inherits: &'static [&'static str],
}

/// The structural form of a class. Today only `LineItem`; sibling variants
/// (Document, Tax, Payment, …) land as new `CanonicalConcept`s prove out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassShape {
    /// A per-line entry on a commercial document. Shared by Rails
    /// `InvoiceLineItem`, Odoo `account.move.line`, Spree `LineItem`,
    /// future SAP BSEG.
    LineItem(LineItemShape),
}

/// The structural fields a `LineItem`-shaped class must carry. Field
/// *names* are curator-specific (leaf detail); what matters for overlap is
/// that each slot is present (non-empty).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineItemShape {
    /// Curator label of the parent document (`Invoice`, `account.move`,
    /// `Order`).
    pub parent_doc: &'static str,
    /// Curator label of the item/product reference, if any (`Item`,
    /// `product.product`, `Variant`).
    pub item_ref: Option<&'static str>,
    /// Curator-side quantity field name (`item_quantity`, `quantity`).
    pub quantity_field: &'static str,
    /// Curator-side unit-price field name (`item_unit_cost`, `price_unit`,
    /// `price`).
    pub unit_price_field: &'static str,
    /// Curator-side tax references. OSB uses two named slots (`tax_1`,
    /// `tax_2`); Odoo uses one M2M (`tax_ids`); both are non-empty for a
    /// line-item shape that can be promoted.
    pub tax_refs: &'static [&'static str],
    /// Curator-side label / description field name (`item_description`,
    /// `name`).
    pub label_field: &'static str,
}

// ─── Overlap detection ──────────────────────────────────────────────────

/// Detect a `CanonicalConcept::CommercialLineItem` overlap between two
/// curator fixtures. Returns `Some(CommercialLineItem)` exactly when:
///
/// 1. The two fixtures come from *different* curators (≥2-curator
///    promotion rule — same-curator pairs cannot promote).
/// 2. Both fixtures carry `ClassShape::LineItem`.
/// 3. Both fixtures have non-empty values for every structural slot
///    (`parent_doc`, `quantity_field`, `unit_price_field`, ≥1 `tax_refs`,
///    `label_field`).
///
/// Symmetric: `overlap_commercial_line_item(a, b) ==
/// overlap_commercial_line_item(b, a)`.
///
/// Deterministic: re-running on the same pair returns the same result
/// (no duplicate emissions per operator acceptance #5).
#[must_use]
pub fn overlap_commercial_line_item(a: &Class, b: &Class) -> Option<CanonicalConcept> {
    if a.source_curator == b.source_curator {
        return None;
    }
    let (ClassShape::LineItem(la), ClassShape::LineItem(lb)) = (&a.shape, &b.shape);

    let has_shape = |s: &LineItemShape| {
        !s.parent_doc.is_empty()
            && !s.quantity_field.is_empty()
            && !s.unit_price_field.is_empty()
            && !s.tax_refs.is_empty()
            && !s.label_field.is_empty()
    };
    if has_shape(la) && has_shape(lb) {
        Some(CanonicalConcept::CommercialLineItem)
    } else {
        None
    }
}

// ─── Hand-built curator fixtures ────────────────────────────────────────

/// `open_source_billing:InvoiceLineItem` fixture. Sourced from
/// `AdaWorldAPI/open-source-billing` commit `61cd6ed` (2026-06-19),
/// `app/models/invoice_line_item.rb`.
///
/// Notable curator-side facts (preserved as leaf detail):
///
/// - `belongs_to :tax1` / `:tax2` with FKs `tax_1` / `tax_2` (max two
///   taxes per line vs Odoo's M2M).
/// - `acts_as_archival` / `acts_as_paranoid` (soft-delete).
/// - `after_destroy :recalculate_invoice_total` (denormalized parent).
#[must_use]
pub const fn osb_invoice_line_item() -> Class {
    Class {
        source_curator: SourceCurator::OpenSourceBilling,
        source_domain: SourceDomain::Billing,
        curator_label: "InvoiceLineItem",
        shape: ClassShape::LineItem(LineItemShape {
            parent_doc: "Invoice",
            item_ref: Some("Item"),
            quantity_field: "item_quantity",
            unit_price_field: "item_unit_cost",
            tax_refs: &["tax_1", "tax_2"],
            label_field: "item_description",
        }),
        inherits: &["ApplicationRecord"],
    }
}

/// `odoo:account.move.line` fixture. Field names per the Odoo canonical
/// `account/models/account_move_line.py` surface (already grounded in
/// `lance-graph-ontology::odoo_blueprint::structural` and matched against
/// the #527 corpus). Inherits `analytic.mixin`.
#[must_use]
pub const fn odoo_account_move_line() -> Class {
    Class {
        source_curator: SourceCurator::Odoo,
        source_domain: SourceDomain::Erp,
        curator_label: "account.move.line",
        shape: ClassShape::LineItem(LineItemShape {
            parent_doc: "account.move",
            item_ref: Some("product.product"),
            quantity_field: "quantity",
            unit_price_field: "price_unit",
            tax_refs: &["tax_ids"],
            label_field: "name",
        }),
        inherits: &["analytic.mixin"],
    }
}

// ─── OGIT canonical relation predicates ─────────────────────────────────
//
// Per `OGIT/SGO/sgo/verbs/{includes,isMemberOf,contains,isPartOf}.ttl`,
// OGIT defines the canonical relation predicate vocabulary for ALL
// curators. Each extractor (ruff_ruby_spo for Rails, spo_enrich.py for
// Odoo, future SAP) gets a small **codebook** that maps its
// extractor-specific predicates into the OGIT canonical names. OGAR
// then consumes the canonical predicates directly — synergy detection
// no longer has to know which extractor produced a triple.
//
// The codebook pattern dissolves the "predicate-vocabulary divergence"
// finding from `E-OGAR-AR-SHAPE-SMOKE-2`: each extractor stays free to
// emit its own native shape, and a per-extractor `translate_*_to_ogit`
// pass folds them into a unified canonical stream.

/// OGIT canonical relation predicates. The single shared vocabulary that
/// every curator's extractor codebook targets. See
/// `OGIT/SGO/sgo/verbs/*.ttl` for the authoritative `owl:ObjectProperty`
/// definitions.
pub mod ogit_relations {
    /// `ogit:includes` — *"Indicates if an entity includes something
    /// else."* One-to-many parent → children (`has_many`, `One2many`).
    pub const INCLUDES: &str = "ogit:includes";
    /// `ogit:isMemberOf` — *"An entity can be a member of another
    /// entity."* Many-to-one child → parent (`belongs_to`, `Many2one`).
    pub const IS_MEMBER_OF: &str = "ogit:isMemberOf";
    /// `ogit:contains` — *"This relationship indicates that something
    /// is part of something else."* Composition (`has_and_belongs_to_many`,
    /// `Many2many` from the composing side).
    pub const CONTAINS: &str = "ogit:contains";
    /// `ogit:isPartOf` — *"Indicates if an entity is part of another
    /// entity."* Inverse of `contains`.
    pub const IS_PART_OF: &str = "ogit:isPartOf";

    /// Returns `true` if the given predicate IRI is any of the four
    /// OGIT canonical relation predicates. Useful for direction-blind
    /// shape checks ("does class C have ANY relation to class T?").
    #[must_use]
    pub fn is_relation_predicate(p: &str) -> bool {
        matches!(p, INCLUDES | IS_MEMBER_OF | CONTAINS | IS_PART_OF)
    }
}

/// Codebook: translate Rails / `ruff_ruby_spo` extractor output into
/// OGIT-canonical relation triples.
///
/// The Rails extractor emits `(class, declares_association, class.assoc)`
/// alongside `(class.assoc, association_kind, "<kind>")` where kind is
/// one of `belongs_to` / `has_many` / `has_one` / `has_and_belongs_to_many`.
/// This codebook joins the two streams and emits one OGIT triple per
/// relation, with the canonical direction:
///
/// | Rails kind                    | OGIT predicate         |
/// |-------------------------------|------------------------|
/// | `belongs_to`                  | `ogit:isMemberOf`      |
/// | `has_many`                    | `ogit:includes`        |
/// | `has_one`                     | `ogit:includes`        |
/// | `has_and_belongs_to_many`     | `ogit:contains`        |
///
/// Output subject = the class IRI (with namespace prefix). Output
/// object = the original `<class>.<assoc>` field IRI (curator label
/// preserved as leaf detail per doctrine §2 correction 4).
///
/// Missing `association_kind` triple (older ndjson predating
/// `AdaWorldAPI/ruff#15`) defaults to `belongs_to` → `isMemberOf`,
/// preserving the conservative many-to-one assumption.
#[must_use]
pub fn translate_rails_to_ogit(triples: &[Triple]) -> Vec<Triple> {
    let mut kinds: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    for t in triples {
        if t.p == "association_kind" {
            kinds.insert(t.s.clone(), t.o.clone());
        }
    }

    let mut out = Vec::new();
    for t in triples {
        if t.p != "declares_association" {
            continue;
        }
        let kind = kinds
            .get(&t.o)
            .map(String::as_str)
            .unwrap_or("belongs_to");
        let predicate = match kind {
            "belongs_to" => ogit_relations::IS_MEMBER_OF,
            "has_many" | "has_one" => ogit_relations::INCLUDES,
            "has_and_belongs_to_many" => ogit_relations::CONTAINS,
            _ => continue,
        };
        out.push(Triple {
            s: t.s.clone(),
            p: predicate.to_string(),
            o: t.o.clone(),
        });
    }
    out
}

/// Codebook: translate Odoo / `spo_enrich.py` extractor output into
/// OGIT-canonical relation triples.
///
/// The Odoo extractor emits `(class.field, target, comodel.name)`
/// without an explicit field-kind sibling triple (today's `spo_enrich`
/// does not surface `Many2one` vs `One2many` vs `Many2many`). This
/// codebook conservatively defaults to **`ogit:isMemberOf`** — the
/// many-to-one assumption — because:
///
/// 1. Odoo's relational `target` is overwhelmingly `Many2one`
///    (every `*_id` field is a `Many2one`; `One2many` and `Many2many`
///    are the minority).
/// 2. For synergy detection, the direction-blind `is_relation_predicate`
///    check sees BOTH `isMemberOf` and `includes` as relations — the
///    detector doesn't care which one is emitted.
/// 3. A future Odoo-extractor extension can emit a sibling
///    `(class.field, field_kind, "Many2one"|"One2many"|"Many2many")`
///    triple; this codebook is then extended to dispatch on it.
///
/// Output subject = the class IRI (`<ns><class>`). Output object = the
/// comodel IRI under the same namespace, with `.` replaced by `_` to
/// match the workspace's underscored-IRI convention for Odoo class
/// identifiers (`account.tax` → `<ns>account_tax`).
#[must_use]
pub fn translate_odoo_to_ogit(triples: &[Triple], namespace_prefix: &str) -> Vec<Triple> {
    let mut out = Vec::new();
    for t in triples {
        if t.p != "target" {
            continue;
        }
        let Some(s_no_ns) = t.s.strip_prefix(namespace_prefix) else {
            continue;
        };
        let Some((class, _field)) = s_no_ns.split_once('.') else {
            continue;
        };
        let comodel_underscored = t.o.replace('.', "_");
        out.push(Triple {
            s: format!("{namespace_prefix}{class}"),
            p: ogit_relations::IS_MEMBER_OF.to_string(),
            o: format!("{namespace_prefix}{comodel_underscored}"),
        });
    }
    out
}

/// Find class IRIs in an OGIT-canonical triple set that look like a
/// `CommercialLineItem`. Walks **only** the OGIT canonical predicates
/// (`is_relation_predicate`), direction-blind — both `isMemberOf` (the
/// child→parent and `Many2one` case) and `includes` (the parent→children
/// and `has_many` case) count as "relation present."
///
/// Same `classify_line_item_signal` as the vocabulary-aware detector,
/// but here the curator-specific predicate names are gone: callers
/// pre-translate via `translate_rails_to_ogit` / `translate_odoo_to_ogit`,
/// then this single detector runs unchanged on either side.
///
/// **This is the "OGAR uses canonical" path** the user named: the
/// extractor codebooks fold curator vocabularies into OGIT canonical,
/// and detection runs on the canonical stream.
#[must_use]
pub fn classes_matching_commercial_line_item_shape_canonical(
    canonical_triples: &[Triple],
    namespace_prefix: &str,
) -> Vec<String> {
    let mut has_doc_assoc = std::collections::BTreeSet::<String>::new();
    let mut has_tax_assoc = std::collections::BTreeSet::<String>::new();

    for t in canonical_triples {
        if !ogit_relations::is_relation_predicate(&t.p) {
            continue;
        }
        let Some(class_iri) = t.s.strip_prefix(namespace_prefix) else {
            continue;
        };
        if class_iri.contains('.') {
            continue;
        }
        // Object is either `<ns><Class>.<assoc>` (Rails leaf) or
        // `<ns><comodel_underscored>` (Odoo translated). Strip ns;
        // the final segment after any `.` is the signal source.
        let Some(o_no_ns) = t.o.strip_prefix(namespace_prefix) else {
            continue;
        };
        let signal_source = o_no_ns.rsplit('.').next().unwrap_or(o_no_ns);

        match classify_line_item_signal(signal_source) {
            LineItemSignal::DocParent => {
                has_doc_assoc.insert(class_iri.to_string());
            }
            LineItemSignal::TaxBinding => {
                has_tax_assoc.insert(class_iri.to_string());
            }
            LineItemSignal::Other => {}
        }
    }

    has_doc_assoc.intersection(&has_tax_assoc).cloned().collect()
}

// ─── Triple-based detection on real ruff-harvested corpora ──────────────
//
// The hand-fixture path above remains as the structural CLAIM. The Triple
// path below is the EVIDENCE — it consumes real `Triple` ndjson harvested
// by `ruff_ruby_spo` (Rails side; merged via the openproject extractor
// landed in `AdaWorldAPI/ruff#26`) and the existing Odoo extractor's
// output already in this repo. Both emit the `{s, p, o, f, c}` wire shape
// (compatible with `lance_graph_contract::codegen_spine::Triple`); ar_shape
// reads only `(s, p, o)` to keep ndjson loading zero-dep.

/// A minimal triple as it appears in an `.ndjson` row from `ruff_ruby_spo`
/// or the Odoo `spo_enrich.py` extractor. Identity-only (`(s, p, o)`);
/// truth values `(f, c)` are present in the wire form but not needed for
/// shape detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Triple {
    /// Subject IRI (`openproject:InvoiceLineItem`, `odoo:account_move_line`).
    pub s: String,
    /// Predicate (`rdf:type`, `has_attribute`, `declares_association`, …).
    pub p: String,
    /// Object IRI / literal.
    pub o: String,
}

/// Errors from the minimal hand-rolled ndjson loader. Kept opaque +
/// small (no `thiserror`/`anyhow` — matches the lance-graph-contract
/// zero-dep ethos for in-line workspace types).
#[derive(Debug)]
pub struct LoadError {
    /// 1-based line number in the source ndjson file.
    pub line: usize,
    /// Human-readable reason.
    pub reason: &'static str,
}

impl core::fmt::Display for LoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ndjson line {}: {}", self.line, self.reason)
    }
}

impl std::error::Error for LoadError {}

/// Parse an ndjson byte buffer into `Vec<Triple>`. Each row is one JSON
/// object with `s`/`p`/`o` string fields (`f`/`c` ignored). Hand-rolled
/// to avoid pulling `serde_json` into the ontology crate solely for this
/// smoke; behaviour matches `ruff_spo_triplet::to_ndjson` round-trip on
/// the (s, p, o) identity columns.
///
/// Tolerant: empty lines are skipped. Strict on shape: a row missing any
/// of `s`/`p`/`o` returns `Err`.
pub fn load_triples_ndjson(bytes: &[u8]) -> Result<Vec<Triple>, LoadError> {
    let mut out = Vec::new();
    let text = core::str::from_utf8(bytes).map_err(|_| LoadError {
        line: 0,
        reason: "non-utf8 input",
    })?;
    for (idx, raw) in text.lines().enumerate() {
        let line = idx + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let s = extract_string_field(trimmed, "s").ok_or(LoadError {
            line,
            reason: "missing or malformed s",
        })?;
        let p = extract_string_field(trimmed, "p").ok_or(LoadError {
            line,
            reason: "missing or malformed p",
        })?;
        let o = extract_string_field(trimmed, "o").ok_or(LoadError {
            line,
            reason: "missing or malformed o",
        })?;
        out.push(Triple { s, p, o });
    }
    Ok(out)
}

/// Find the value of `"key":"<value>"` in a raw JSON row. Walks the
/// chars so common JSON escapes inside `s`/`p`/`o` (`\"`, `\\`, `\n`,
/// `\r`, `\t`, `\/`) are handled — Rails `validates` messages reach
/// `validation_param` triples with embedded `\"` and would otherwise
/// break a naïve `find('"')` early-terminator.
fn extract_string_field(row: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let start = row.find(&needle)? + needle.len();
    let mut out = String::new();
    let mut chars = row[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '/' => out.push('/'),
                other => {
                    // Unknown escape — preserve verbatim so the detector
                    // can still see the row without misparsing it.
                    out.push('\\');
                    out.push(other);
                }
            },
            '"' => return Some(out),
            other => out.push(other),
        }
    }
    None
}

/// Classify a relation-leaf hint into a `CommercialLineItem` signal.
/// `name` is either a Rails association-leaf (OSB: `invoice`, `tax1`,
/// `tax2`) or an Odoo comodel name (`account.move`, `account.tax`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineItemSignal {
    /// Hints at a document-parent association.
    DocParent,
    /// Hints at a tax-binding association.
    TaxBinding,
    /// Neither — the relation is something else (`currency`, `item`, …).
    Other,
}

fn classify_line_item_signal(name: &str) -> LineItemSignal {
    let lower = name.to_lowercase();
    // Tax check first — `account.tax` would also match `move` via the
    // `account.` prefix, so let tax win for explicitness on `tax_*`.
    if lower.contains("tax") {
        return LineItemSignal::TaxBinding;
    }
    if lower.contains("invoice") || lower.contains("move") || lower.contains("order") {
        return LineItemSignal::DocParent;
    }
    LineItemSignal::Other
}

/// Find class IRIs in a triple set that look like a `CommercialLineItem`:
/// the class has at least one relation to a **document parent**
/// (`invoice` / `move` / `order` in the leaf-or-comodel name) AND at
/// least one relation to a **tax binding** (`tax_1` / `tax_2` /
/// `tax_ids` / `account.tax`).
///
/// `namespace_prefix` is the curator's IRI prefix (`"openproject:"` /
/// `"odoo:"`). Returns class IRIs **with** the prefix stripped — the
/// per-curator label stays visible as leaf detail.
///
/// **Vocabulary-aware**: walks BOTH predicate shapes the workspace's
/// two extractors actually emit (this divergence is itself the next
/// finding — see `E-OGAR-AR-SHAPE-SMOKE-1` follow-up):
///
/// - **Rails / `ruff_ruby_spo`** uses `declares_association`. Subject is
///   the class IRI (`openproject:InvoiceLineItem`), object is the
///   field-IRI (`openproject:InvoiceLineItem.tax1`); the
///   association-leaf (`tax1`) carries the signal.
/// - **Odoo / `tools/odoo-blueprint-extractor`** uses `target`. Subject
///   is the field-IRI (`odoo:account_move_line.tax_ids`), object is the
///   plain comodel name (`account.tax`); the comodel name carries the
///   signal.
///
/// **The two extractors emit different predicate vocabularies even
/// though they describe the same AR-shape primitive** — corpus-level
/// evidence that the §11.1 Inc 4 curator-promotion probe needs either
/// (a) a small predicate-translation layer like this one, or (b) the
/// upstream alignment named by `E-AR-PROJECTION-CORRECTION-1` Phase 1
/// Option A (Odoo arms in the openproject-nexgen extractor, emitting
/// the unified Rails vocabulary). This detector takes path (a) as the
/// in-repo workaround.
#[must_use]
pub fn classes_matching_commercial_line_item_shape(
    triples: &[Triple],
    namespace_prefix: &str,
) -> Vec<String> {
    let mut has_doc_assoc = std::collections::BTreeSet::<String>::new();
    let mut has_tax_assoc = std::collections::BTreeSet::<String>::new();

    for t in triples {
        let (class_iri, signal_source) = match t.p.as_str() {
            "declares_association" => {
                // Rails-style: subject IS the class, object carries the
                // association leaf-name.
                let Some(class_iri) = t.s.strip_prefix(namespace_prefix) else {
                    continue;
                };
                if class_iri.contains('.') {
                    continue;
                }
                let Some(assoc_iri_without_ns) = t.o.strip_prefix(namespace_prefix) else {
                    continue;
                };
                let assoc_leaf = assoc_iri_without_ns.rsplit('.').next().unwrap_or("");
                (class_iri.to_string(), assoc_leaf.to_string())
            }
            "target" => {
                // Odoo-style: subject is `<class>.<field>`; object is the
                // comodel name (no namespace prefix), which carries the
                // signal directly.
                let Some(s_no_ns) = t.s.strip_prefix(namespace_prefix) else {
                    continue;
                };
                let Some((class_iri, _field)) = s_no_ns.split_once('.') else {
                    continue;
                };
                (class_iri.to_string(), t.o.clone())
            }
            _ => continue,
        };

        match classify_line_item_signal(&signal_source) {
            LineItemSignal::DocParent => {
                has_doc_assoc.insert(class_iri);
            }
            LineItemSignal::TaxBinding => {
                has_tax_assoc.insert(class_iri);
            }
            LineItemSignal::Other => {}
        }
    }

    has_doc_assoc.intersection(&has_tax_assoc).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The headline smoke per operator directive: OSB::InvoiceLineItem +
    /// Odoo::account.move.line surface the same primitive (a per-line
    /// commercial entry carrying qty × unit_price + tax + parent + label)
    /// → promote to `CommercialLineItem`.
    #[test]
    fn open_source_billing_invoice_line_and_odoo_move_line_overlap_as_commercial_line_item() {
        let osb = osb_invoice_line_item();
        let odoo = odoo_account_move_line();

        let forward = overlap_commercial_line_item(&osb, &odoo);
        assert_eq!(forward, Some(CanonicalConcept::CommercialLineItem));

        // Symmetric — order should not matter.
        let reverse = overlap_commercial_line_item(&odoo, &osb);
        assert_eq!(reverse, forward);
    }

    /// Regression for operator acceptance #5: detection is deterministic
    /// and re-running it must not register a second canonical concept.
    /// (Idempotence at the function level; registry-side idempotence
    /// would come from a `BTreeSet<CanonicalConcept>` upstream.)
    #[test]
    fn rails_billing_and_odoo_do_not_create_duplicate_canonical_concepts() {
        let osb = osb_invoice_line_item();
        let odoo = odoo_account_move_line();

        let first = overlap_commercial_line_item(&osb, &odoo);
        let second = overlap_commercial_line_item(&osb, &odoo);
        assert_eq!(first, second);
        assert!(matches!(first, Some(CanonicalConcept::CommercialLineItem)));
    }

    /// The ≥2-curator promotion rule is STRUCTURAL: comparing one
    /// curator's fixture against itself MUST NOT promote.
    #[test]
    fn same_curator_self_compare_does_not_promote() {
        let a = osb_invoice_line_item();
        let b = osb_invoice_line_item();
        assert_eq!(overlap_commercial_line_item(&a, &b), None);

        let c = odoo_account_move_line();
        let d = odoo_account_move_line();
        assert_eq!(overlap_commercial_line_item(&c, &d), None);
    }

    /// Curator-label divergence is part of the design — the field-NAMES
    /// differ (`item_unit_cost` vs `price_unit`), but the shape still
    /// promotes. The leaf detail stays visible on the fixture for
    /// adapter generation.
    #[test]
    fn curator_field_names_diverge_but_shape_still_promotes() {
        let osb = osb_invoice_line_item();
        let odoo = odoo_account_move_line();

        let ClassShape::LineItem(osb_shape) = osb.shape;
        let ClassShape::LineItem(odoo_shape) = odoo.shape;

        assert_ne!(osb_shape.unit_price_field, odoo_shape.unit_price_field);
        assert_ne!(osb_shape.quantity_field, odoo_shape.quantity_field);
        assert_ne!(osb_shape.label_field, odoo_shape.label_field);

        // …yet they overlap.
        assert_eq!(
            overlap_commercial_line_item(&osb, &odoo),
            Some(CanonicalConcept::CommercialLineItem),
        );
    }

    /// Namespace prefixes are stable `&'static str` per
    /// `E-OGAR-AR-SHAPE-ENDGAME` §11.1 Inc 3 (adapter target ids are
    /// `&'static str`). Lock the two curators we actually use today.
    #[test]
    fn namespace_prefixes_for_today_curators_are_stable() {
        assert_eq!(SourceCurator::OpenSourceBilling.namespace_prefix(), "open_source_billing:");
        assert_eq!(SourceCurator::Odoo.namespace_prefix(), "odoo:");
    }

    /// Empty structural slot (e.g. a malformed fixture with no tax_refs)
    /// must NOT promote — the overlap test is conservative on absent
    /// shape.
    #[test]
    fn empty_tax_refs_block_promotion() {
        let mut osb = osb_invoice_line_item();
        let ClassShape::LineItem(ref mut shape) = osb.shape;
        shape.tax_refs = &[];
        let odoo = odoo_account_move_line();

        assert_eq!(overlap_commercial_line_item(&osb, &odoo), None);
    }

    // ─── Triple-loader + harvest-driven detection tests ─────────────────

    /// The minimal ndjson parser round-trips a hand-built representative
    /// row in the exact shape the ruff/odoo extractors emit.
    #[test]
    fn load_triples_ndjson_round_trips_representative_row() {
        let raw = br#"{"s":"openproject:InvoiceLineItem","p":"declares_association","o":"openproject:InvoiceLineItem.tax1","f":0.95,"c":0.88}
{"s":"odoo:account_move_line","p":"rdf:type","o":"ogit:ObjectType","f":1.0,"c":1.0}
"#;
        let triples = load_triples_ndjson(raw).expect("parse");
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].s, "openproject:InvoiceLineItem");
        assert_eq!(triples[0].p, "declares_association");
        assert_eq!(triples[0].o, "openproject:InvoiceLineItem.tax1");
        assert_eq!(triples[1].s, "odoo:account_move_line");
    }

    /// The smoke that the operator pivot actually requested: run on the
    /// real OSB harvest (via `ruff_ruby_spo`) + the real Odoo harvest
    /// (workspace `odoo_ontology.spo.ndjson`). Assert each side surfaces
    /// the expected line-item class via structural signal, and that the
    /// pair becomes a synergy candidate.
    ///
    /// The OSB fixture lives in
    /// `tests/fixtures/osb_ruby_spo.ndjson` (~1 195 triples harvested
    /// from `AdaWorldAPI/open-source-billing@61cd6ed`). The Odoo file is
    /// the in-repo `crates/lance-graph/src/graph/spo/odoo_ontology.spo.ndjson`
    /// (~2.8 MB, the `#527`-regen corpus).
    ///
    /// The `openproject:` prefix on the OSB harvest is a **known
    /// artefact** of `ruff_ruby_spo::NAMESPACE` being a `const &str`
    /// (operator's "one tiny regex" point — fixable by a small upstream
    /// PR adding a parameterised `extract_with(path, ns)`). The detector
    /// takes the prefix as an argument so the test stays correct
    /// regardless.
    #[test]
    fn ruff_harvested_osb_and_odoo_corpora_surface_commercial_line_item_candidates() {
        let osb_bytes = include_bytes!("../tests/fixtures/osb_ruby_spo.ndjson");
        let odoo_bytes = include_bytes!(
            "../../lance-graph/src/graph/spo/odoo_ontology.spo.ndjson"
        );

        let osb = load_triples_ndjson(osb_bytes).expect("osb ndjson loads");
        let odoo = load_triples_ndjson(odoo_bytes).expect("odoo ndjson loads");

        // OSB harvest uses the (intentionally wrong-for-OSB) `openproject:`
        // prefix today; Odoo uses `odoo:`.
        let osb_candidates =
            classes_matching_commercial_line_item_shape(&osb, "openproject:");
        let odoo_candidates =
            classes_matching_commercial_line_item_shape(&odoo, "odoo:");

        // OSB must surface InvoiceLineItem (the strongest pair per
        // operator directive 2026-06-19).
        assert!(
            osb_candidates.iter().any(|c| c == "InvoiceLineItem"),
            "expected OSB to surface InvoiceLineItem; got {osb_candidates:?}",
        );

        // Odoo must surface account_move_line (the strongest pair on
        // the ERP side).
        assert!(
            odoo_candidates.iter().any(|c| c == "account_move_line"),
            "Odoo candidates missing account_move_line; got first 5: {:?}",
            odoo_candidates.iter().take(5).collect::<Vec<_>>(),
        );
    }

    /// Same detector run on the harvested OSB corpus must not promote a
    /// random model (e.g. `Currency`) — it's not LineItem-shaped (no
    /// doc-parent + tax-binding pair).
    #[test]
    fn ruff_harvested_osb_corpus_does_not_promote_non_line_item_classes() {
        let osb_bytes = include_bytes!("../tests/fixtures/osb_ruby_spo.ndjson");
        let osb = load_triples_ndjson(osb_bytes).expect("osb ndjson loads");
        let candidates =
            classes_matching_commercial_line_item_shape(&osb, "openproject:");
        // Currency / Client / Company are not LineItem-shaped — they
        // don't carry a tax association.
        for negative in ["Currency", "Client", "Company", "Project", "Payment"] {
            assert!(
                !candidates.iter().any(|c| c == negative),
                "{negative} must NOT promote as CommercialLineItem candidate \
                 (no tax association); got {candidates:?}",
            );
        }
    }

    // ─── OGIT canonical codebook tests ──────────────────────────────

    /// The four OGIT canonical relation predicates have stable IRIs
    /// matching `OGIT/SGO/sgo/verbs/*.ttl`. Lock them.
    #[test]
    fn ogit_relation_predicates_have_stable_canonical_iris() {
        assert_eq!(ogit_relations::INCLUDES, "ogit:includes");
        assert_eq!(ogit_relations::IS_MEMBER_OF, "ogit:isMemberOf");
        assert_eq!(ogit_relations::CONTAINS, "ogit:contains");
        assert_eq!(ogit_relations::IS_PART_OF, "ogit:isPartOf");

        assert!(ogit_relations::is_relation_predicate("ogit:isMemberOf"));
        assert!(ogit_relations::is_relation_predicate("ogit:includes"));
        assert!(ogit_relations::is_relation_predicate("ogit:contains"));
        assert!(ogit_relations::is_relation_predicate("ogit:isPartOf"));
        assert!(!ogit_relations::is_relation_predicate("declares_association"));
        assert!(!ogit_relations::is_relation_predicate("target"));
    }

    /// Rails codebook maps `belongs_to` → `isMemberOf` and `has_many` →
    /// `includes` via the joined `association_kind` triple. On the real
    /// OSB harvest, both directions appear (`Client.invoices: has_many`
    /// vs `InvoiceLineItem.invoice: belongs_to`).
    #[test]
    fn rails_codebook_translates_has_many_to_includes_and_belongs_to_to_is_member_of() {
        let triples = vec![
            // Rails: Client has_many :invoices
            Triple {
                s: "openproject:Client".into(),
                p: "declares_association".into(),
                o: "openproject:Client.invoices".into(),
            },
            Triple {
                s: "openproject:Client.invoices".into(),
                p: "association_kind".into(),
                o: "has_many".into(),
            },
            // Rails: InvoiceLineItem belongs_to :invoice
            Triple {
                s: "openproject:InvoiceLineItem".into(),
                p: "declares_association".into(),
                o: "openproject:InvoiceLineItem.invoice".into(),
            },
            Triple {
                s: "openproject:InvoiceLineItem.invoice".into(),
                p: "association_kind".into(),
                o: "belongs_to".into(),
            },
        ];
        let canonical = translate_rails_to_ogit(&triples);
        assert_eq!(canonical.len(), 2);
        let invoices = canonical.iter().find(|t| t.s == "openproject:Client").unwrap();
        assert_eq!(invoices.p, ogit_relations::INCLUDES);
        let parent = canonical
            .iter()
            .find(|t| t.s == "openproject:InvoiceLineItem")
            .unwrap();
        assert_eq!(parent.p, ogit_relations::IS_MEMBER_OF);
    }

    /// Odoo codebook conservatively maps `target` → `isMemberOf` (the
    /// Many2one-dominant default) and rewrites the subject to the class
    /// IRI plus the underscored comodel as the object.
    #[test]
    fn odoo_codebook_translates_target_to_is_member_of_with_underscored_comodel() {
        let triples = vec![
            Triple {
                s: "odoo:account_move_line.move_id".into(),
                p: "target".into(),
                o: "account.move".into(),
            },
            Triple {
                s: "odoo:account_move_line.tax_ids".into(),
                p: "target".into(),
                o: "account.tax".into(),
            },
        ];
        let canonical = translate_odoo_to_ogit(&triples, "odoo:");
        assert_eq!(canonical.len(), 2);
        for t in &canonical {
            assert_eq!(t.s, "odoo:account_move_line");
            assert_eq!(t.p, ogit_relations::IS_MEMBER_OF);
        }
        assert!(canonical.iter().any(|t| t.o == "odoo:account_move"));
        assert!(canonical.iter().any(|t| t.o == "odoo:account_tax"));
    }

    /// **The "OGAR uses canonical" smoke**: both codebooks fold their
    /// curator-specific vocabularies into OGIT canonical; the single
    /// `classes_matching_commercial_line_item_shape_canonical` detector
    /// runs unchanged on either side and surfaces the expected class
    /// (`InvoiceLineItem` on OSB; `account_move_line` on Odoo).
    #[test]
    fn ogit_canonical_detector_finds_line_item_classes_on_both_corpora() {
        let osb_bytes = include_bytes!("../tests/fixtures/osb_ruby_spo.ndjson");
        let odoo_bytes = include_bytes!(
            "../../lance-graph/src/graph/spo/odoo_ontology.spo.ndjson"
        );

        let osb_raw = load_triples_ndjson(osb_bytes).unwrap();
        let odoo_raw = load_triples_ndjson(odoo_bytes).unwrap();

        // Codebook pass — curator vocab → OGIT canonical.
        let osb_canonical = translate_rails_to_ogit(&osb_raw);
        let odoo_canonical = translate_odoo_to_ogit(&odoo_raw, "odoo:");

        // Single canonical detector runs on either side.
        let osb_cands = classes_matching_commercial_line_item_shape_canonical(
            &osb_canonical,
            "openproject:",
        );
        let odoo_cands = classes_matching_commercial_line_item_shape_canonical(
            &odoo_canonical,
            "odoo:",
        );

        assert!(
            osb_cands.iter().any(|c| c == "InvoiceLineItem"),
            "OSB canonical-detector candidates missing InvoiceLineItem; got {osb_cands:?}",
        );
        assert!(
            odoo_cands.iter().any(|c| c == "account_move_line"),
            "Odoo canonical-detector candidates missing account_move_line; got first 5: {:?}",
            odoo_cands.iter().take(5).collect::<Vec<_>>(),
        );
    }

    /// Hand-fixture detection (the `overlap_commercial_line_item` path
    /// committed in the prior smoke) and the harvest detection
    /// (`classes_matching_commercial_line_item_shape`) must agree on
    /// the OSB::InvoiceLineItem + Odoo::account_move_line pair. The
    /// hand fixture says yes; the corpus says yes; the doctrine line
    /// "labels are leaf detail, the SHAPE is what overlaps" stays true.
    #[test]
    fn hand_fixture_and_corpus_detection_agree_on_invoice_line_item_pair() {
        // Hand fixture → Some(CommercialLineItem)
        let hand = overlap_commercial_line_item(
            &osb_invoice_line_item(),
            &odoo_account_move_line(),
        );
        assert_eq!(hand, Some(CanonicalConcept::CommercialLineItem));

        // Corpus → InvoiceLineItem and account_move_line both appear as
        // candidates → the pair is a synergy row.
        let osb_bytes = include_bytes!("../tests/fixtures/osb_ruby_spo.ndjson");
        let odoo_bytes = include_bytes!(
            "../../lance-graph/src/graph/spo/odoo_ontology.spo.ndjson"
        );
        let osb = load_triples_ndjson(osb_bytes).unwrap();
        let odoo = load_triples_ndjson(odoo_bytes).unwrap();
        let osb_c =
            classes_matching_commercial_line_item_shape(&osb, "openproject:");
        let odoo_c =
            classes_matching_commercial_line_item_shape(&odoo, "odoo:");
        assert!(osb_c.iter().any(|c| c == "InvoiceLineItem"));
        assert!(odoo_c.iter().any(|c| c == "account_move_line"));
    }
}
