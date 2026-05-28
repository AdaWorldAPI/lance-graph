#!/usr/bin/env python3
"""Emit OGIT-conformant Turtle from ruff-py-dto NDJSON bundles.

Proof-of-concept for D-RPYDTO-6 (`unified-spo-nars-codegen-v1` / Direction-1
of `E-OWL-IS-THE-UNIVERSAL-INGRESS-1`). Reads the NDJSON bundles produced by
`ruff-py-dto harvest --config odoo.config.json --root <odoo/addons> --out <dir>`
and emits one Turtle file per family. Each method becomes an `ogit:Method`
resource carrying the 7-tuple (per `E-BUSINESS-LOGIC-IS-GRAMMAR-1`):

    (transitivity, temporal*, causal*, modal*, locative*, directionality,
     quantities*)

plus the dual-axis classification (AXIS-A / AXIS-B / HYBRID).

Heuristic axis assignment (refined by D-RPYDTO-5 + the proper TTL emitter in
D-RPYDTO-6):
  - `@api.depends`     → AXIS-A deterministic compute
  - `@api.constrains`  → AXIS-A deterministic validation
  - `@api.onchange`    → AXIS-B heuristic suggestion
  - default            → AXIS-A unless a heuristic-name marker matches

This is purely a substrate prototype; the real emitter integrates into
ruff-py-dto's TargetSpec layer.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OGIT = "https://ogit.adaworldapi.com"
PREFIXES = f"""@prefix ogit:    <{OGIT}/> .
@prefix odoo:    <{OGIT}/odoo/> .
@prefix axis:    <{OGIT}/axis/> .
@prefix verb:    <{OGIT}/grammar/verb/> .
@prefix tekamolo: <{OGIT}/grammar/tekamolo/> .
@prefix nars:    <{OGIT}/nars/> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
"""

# OWL 2 DL TBox preamble — addresses the failure mode Maria Keet documents in
# "Just Turtle and RDF vs OWL examples: the CPEV and FIBO" (keet blog,
# 2025-11-17): valid Turtle is not valid OWL DL.  Without these declarations,
# every `ogit:Method` rdf:type drops the ontology to OWL 2 Full (undeclared
# class), and every `ogit:methodName`/`axis:classification`/etc. is reported
# as an undeclared property by Protege / OWL Classifier.
#
# Design choice: use `owl:AnnotationProperty` for the metadata predicates
# (methodName, family, decorator, signature, bodyLines, axis:classification,
# verb:transitivity, tekamolo:causal, the delegation edges).  Annotation
# properties are exempt from OWL 2 DL global restrictions, so we get DL
# conformance without forcing logical-property semantics on metadata.  The
# axis + verb individuals get explicit `owl:NamedIndividual` declarations to
# avoid Protege's "punning" warnings.  No anonymous individuals (blank-node
# lists from `[...]` patterns) anywhere — those are the second failure mode
# Keet flags for CPEV.
OWL_TBOX = f"""### --- OWL 2 DL TBox: classes, properties, individuals ---

<{OGIT}/> a owl:Ontology ;
    rdfs:label "OGIT — Ontology of Grammar / Inference / Truth" ;
    rdfs:comment "Method-level vocabulary for porting Odoo and other Python business-logic codebases into the 7-tuple grammar of E-BUSINESS-LOGIC-IS-GRAMMAR-1." ;
    # Profile declaration per Keet 2025-11-15 ("No, an ontology isn't 'just RDF'"):
    # an ontology document must signal which OWL profile it conforms to.  We use
    # only class declarations, named-individual declarations, and annotation
    # properties (no class expressions, restrictions, cardinality, disjointness,
    # property characteristics).  That puts us inside the OWL 2 EL profile
    # (also satisfies OWL 2 QL and OWL 2 RL).  Turtle here is the serialization
    # syntax, not the structural spec; treat this as an OWL ontology, not an
    # "RDF ontology".
    owl:versionInfo "OWL 2 EL profile (serialized as Turtle 1.1)" .

# Method class
ogit:Method a owl:Class ;
    rdfs:label "Method" ;
    rdfs:comment "A decorated Python function / class method harvested from a framework source tree (Odoo @api.*, Flask @route, Django CBV, ...)." .

# Family is the business-object class (account_move, sale_order, ...).
# Family IRIs are first-class owl:Class instances in the odoo: namespace.
# Punning across class/individual is permitted in OWL 2 DL.

# Axis classification: three named individuals of a Classification class.
axis:Classification a owl:Class .
axis:Deterministic a owl:NamedIndividual, axis:Classification ;
    rdfs:label "Deterministic (AXIS-A)" ;
    rdfs:comment "Closed-form, regulation-implementing logic; outputs are functions of inputs without learned parameters." .
axis:Heuristic a owl:NamedIndividual, axis:Classification ;
    rdfs:label "Heuristic (AXIS-B)" ;
    rdfs:comment "Inferential / suggestion logic; ranks alternatives or proposes user actions on ambiguous inputs." .
axis:Hybrid a owl:NamedIndividual, axis:Classification ;
    rdfs:label "Hybrid" ;
    rdfs:comment "Mixed mode: deterministic over one slice + heuristic over another (e.g. @api.depends + @api.onchange on the same field)." .

# Verb transitivity classification.
verb:Transitivity a owl:Class .
verb:Transitive a owl:NamedIndividual, verb:Transitivity ;
    rdfs:comment "Method returns / mutates a recordset; the logic flows outward through assignment or return." .
verb:Intransitive a owl:NamedIndividual, verb:Transitivity ;
    rdfs:comment "Method terminates internally — typically @api.constrains validators that raise without returning." .

# Metadata annotation properties.
ogit:methodName a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:string .
ogit:matchId a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:string .
ogit:family a owl:AnnotationProperty ;
    rdfs:domain ogit:Method .
ogit:decorator a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:string .
ogit:signature a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:string .
ogit:bodyLines a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:integer .
axis:classification a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range axis:Classification .
verb:transitivity a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range verb:Transitivity .
tekamolo:causal a owl:AnnotationProperty ;
    rdfs:domain ogit:Method .

# DelegationTuple annotation properties (second-pass AST extractor output).
ogit:invokes a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range ogit:Method .
ogit:reads_field a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:string .
ogit:writes_field a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:string .
ogit:traverses_relation a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:string .
ogit:reads_env a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:boolean .
ogit:raises a owl:AnnotationProperty ;
    rdfs:domain ogit:Method ; rdfs:range xsd:string .

# External vocabulary anchors declared locally so a standalone consumer (no
# dcterms import) still passes "undeclared annotation property" checks.
dcterms:source a owl:AnnotationProperty .

### --- end TBox ---
"""

# Heuristic markers for AXIS-B (these names tend to indicate inferential logic
# in Odoo source — refined in D-RPYDTO-5 via the priority classifier).
HEURISTIC_NAME_RE = re.compile(
    r"^_?(?:get_fiscal_position|find|match|resolve|guess|suggest|classify|"
    r"propose|score|rank|detect|infer|recommend)\b"
)


def axis_for(decorators: list[str], method_name: str) -> str:
    """Return AXIS-A / AXIS-B / HYBRID per E-BUSINESS-LOGIC-IS-GRAMMAR-1."""
    deco_blob = " ".join(decorators)
    is_constrains = "@api.constrains" in deco_blob
    is_depends = "@api.depends" in deco_blob
    is_onchange = "@api.onchange" in deco_blob
    name_heuristic = bool(HEURISTIC_NAME_RE.match(method_name))
    if is_onchange and (is_depends or is_constrains):
        return "Hybrid"
    if is_onchange or name_heuristic:
        return "Heuristic"
    return "Deterministic"


def transitivity_for(method_name: str, body_source: str) -> str:
    """Transitive (mutates/returns) vs Intransitive (terminal/check)."""
    body = body_source or ""
    if method_name.startswith("_check_") or method_name.startswith("_validate_"):
        return "Intransitive"
    if "raise " in body and "return" not in body:
        return "Intransitive"
    return "Transitive"


def causal_refs(decorators: list[str], body_source: str) -> list[str]:
    """Extract regulation IRIs / German-fiscal markers from decorators+body.

    Heuristic only; the real path populates from L-doc curated regulation_iri
    + LLM literature extraction (E-LITERATURE-IS-INGRESS-3-1).
    """
    refs = []
    blob = " ".join(decorators) + " " + (body_source or "")
    # German fiscal markers (illustrative; D-RPYDTO-6 reads the regulation_iri
    # codebook for the authoritative list).
    markers = {
        "HGB": "regulation:HGB",
        "GoBD": "regulation:GoBD",
        "UStG": "regulation:UStG",
        "AO": "regulation:AO",
        "ELSTER": "regulation:ELSTER",
        "DATEV": "regulation:DATEV",
        "fiscalyear_lock_date": "regulation:HGB#239-Festschreibung",
        "tax_lock_date": "regulation:UStG#18-Voranmeldung",
        "restrictive_audit_trail": "regulation:GoBD#Unveraenderbarkeit",
        "Festschreibung": "regulation:HGB#239",
    }
    for marker, iri in markers.items():
        if marker in blob:
            refs.append(iri)
    return refs


def quote(s: str) -> str:
    """Turtle-safe triple-quoted string literal."""
    if s is None:
        return '""'
    # Backslash MUST be escaped first; otherwise Python line-continuations
    # `\<newline>` in the harvested body source become invalid Turtle escapes
    # (rdflib raises "bad escape"). After backslash doubling, triple-quotes
    # are the only other escape Turtle's `"""..."""` form needs.
    safe = s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return f'"""{safe}"""'


def emit_bundle(bundle: dict, addon: str | None = None) -> str:
    family = bundle["family"]
    fn = bundle["function_name"]
    file = bundle.get("file", "")
    line_start = bundle.get("line_start", 0)
    line_end = bundle.get("line_end", 0)
    body_lines = bundle.get("body_lines", 0)
    decorators = bundle.get("all_decorators", [])
    body_source = bundle.get("body_source", "")
    signature = bundle.get("signature", "")
    match_id = bundle.get("match_id", "")

    axis = axis_for(decorators, fn)
    transitivity = transitivity_for(fn, body_source)
    causal = causal_refs(decorators, body_source)

    # Subject IRI: ogit:odoo:<family>:<function_name>
    subject = f"odoo:{family}.{fn}"

    lines = []
    lines.append(f"### {family}.{fn}  [{axis} / {transitivity}]")
    lines.append(f"{subject} a ogit:Method ;")
    lines.append(f"    ogit:family odoo:{family} ;")
    lines.append(f'    ogit:methodName "{fn}" ;')
    lines.append(f"    ogit:matchId \"{match_id}\" ;")
    lines.append(f"    axis:classification axis:{axis} ;")
    lines.append(f"    verb:transitivity verb:{transitivity} ;")
    if decorators:
        deco_str = ", ".join(quote(d) for d in decorators)
        lines.append(f"    ogit:decorator {deco_str} ;")
    if signature:
        lines.append(f"    ogit:signature {quote(signature)} ;")
    lines.append(f"    ogit:bodyLines {body_lines} ;")
    for ref in causal:
        # Causal refs use a placeholder prefix; the real emitter resolves them
        # against the regulation_iri codebook.
        lines.append(f"    tekamolo:causal <{OGIT}/{ref.replace(':', '/')}> ;")
    # Body source as rdfs:comment (truncate above ~2KB to keep TTL readable).
    body_excerpt = body_source[:2048] + (" ..." if len(body_source) > 2048 else "")
    lines.append(f"    rdfs:comment {quote(body_excerpt)} ;")
    # Provenance.
    lines.append(f"    dcterms:source {quote(f'{file}:L{line_start}-{line_end}')} .")
    lines.append("")
    return "\n".join(lines)


def emit_family(family: str, ndjson_path: Path, out_path: Path) -> int:
    with ndjson_path.open() as f:
        bundles = [json.loads(line) for line in f if line.strip()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        out.write(f"# OGIT TTL for Odoo family `{family}` — {len(bundles)} methods\n")
        out.write(f"# Auto-emitted by bundles_to_ttl.py from ruff-py-dto NDJSON.\n")
        out.write(f"# Per E-BUSINESS-LOGIC-IS-GRAMMAR-1 + E-OWL-IS-THE-UNIVERSAL-INGRESS-1.\n\n")
        out.write(PREFIXES)
        out.write("\n")
        out.write(OWL_TBOX)
        out.write("\n")
        # Family IRI as owl:Class so `ogit:family odoo:<family>` is well-typed.
        out.write(f"odoo:{family} a owl:Class ;\n")
        out.write(f'    rdfs:label "{family}" .\n\n')
        for b in bundles:
            out.write(emit_bundle(b))
    return len(bundles)


def main():
    if len(sys.argv) < 3:
        print(
            "usage: bundles_to_ttl.py <bundles_dir> <out_dir>",
            file=sys.stderr,
        )
        sys.exit(1)
    bundles_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    families = 0
    for nd in sorted(bundles_dir.glob("*.ndjson")):
        family = nd.stem
        out_path = out_dir / f"{family}.ttl"
        n = emit_family(family, nd, out_path)
        total += n
        families += 1
    print(f"Emitted {total} methods across {families} families to {out_dir}")


if __name__ == "__main__":
    main()
