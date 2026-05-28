"""Pattern-match docstrings/comments/field-help against UStG/HGB/AO/GoBD/EN 16931
anchor table and return a deduped list of regulation IRIs.

All matching is case-insensitive substring search.
"""

from typing import List


# Anchor table: (pattern, iri)
# Order matters for readability but not for correctness (all matches are collected).
_ANCHOR_TABLE = [
    # UStG §15 — Vorsteuerabzug (input tax deduction)
    ("vorsteuer", "ogit:regulation/de/ustg/15"),
    ("input tax", "ogit:regulation/de/ustg/15"),
    ("purchase tax", "ogit:regulation/de/ustg/15"),
    # UStG §13 — Umsatzsteuer (sales tax / VAT)
    ("umsatzsteuer", "ogit:regulation/de/ustg/13"),
    ("vat", "ogit:regulation/de/ustg/13"),
    ("sales tax", "ogit:regulation/de/ustg/13"),
    # GoBD — audit trail / immutability
    ("gobd", "ogit:regulation/de/gobd"),
    ("audit trail", "ogit:regulation/de/gobd"),
    ("restrictive", "ogit:regulation/de/gobd"),
    ("inalterability", "ogit:regulation/de/gobd"),
    ("hash chain", "ogit:regulation/de/gobd"),
    # HGB §238 — Buchführungspflicht
    ("buchführungspflicht", "ogit:regulation/de/hgb/238"),
    ("books of account", "ogit:regulation/de/hgb/238"),
    ("buchfuehrungspflicht", "ogit:regulation/de/hgb/238"),
    # AO §146a — Verfahrensdokumentation
    ("verfahrensdokumentation", "ogit:regulation/de/ao/146a"),
    ("ao 146", "ogit:regulation/de/ao/146a"),
    # UStG §19 — Kleinunternehmer
    ("kleinunternehmer", "ogit:regulation/de/ustg/19"),
    ("small business", "ogit:regulation/de/ustg/19"),
    # EN 16931 / Peppol / UBL / e-invoice
    ("en 16931", "ogit:regulation/eu/en16931"),
    ("peppol", "ogit:regulation/eu/en16931"),
    ("ubl", "ogit:regulation/eu/en16931"),
    ("e-invoice", "ogit:regulation/eu/en16931"),
    ("einvoice", "ogit:regulation/eu/en16931"),
    ("factur-x", "ogit:regulation/eu/en16931"),
    ("xrechnung", "ogit:regulation/eu/en16931"),
    ("cii", "ogit:regulation/eu/en16931"),
    ("zugferd", "ogit:regulation/eu/en16931"),
]


def extract_regulation_iris(text: str) -> List[str]:
    """Scan *text* case-insensitively against the anchor table.

    Returns a deduped, sorted list of matching IRIs.
    """
    if not text:
        return []
    lower = text.lower()
    seen: set = set()
    result: List[str] = []
    for pattern, iri in _ANCHOR_TABLE:
        if pattern in lower and iri not in seen:
            seen.add(iri)
            result.append(iri)
    return result
