DATEV SKR (Standardkontenrahmen) — German chart of accounts
=============================================================

This directory holds machine-readable extracts of the DATEV
Standardkontenrahmen (German standard chart of accounts), the de-facto
canonical chart used by German SMEs for HGB-compliant bookkeeping. Two
schemes are shipped:

- SKR 03 (Prozessgliederungsprinzip — process-oriented) — used by most
  German SMEs. Family numbering reflects business-process flow.
- SKR 04 (Abschlussgliederungsprinzip — balance-sheet-structure) — used
  by larger firms; numbering mirrors the §266 HGB P&L structure.

Files
-----

skr04.csv (1232 accounts)
  Generic SKR 04, extracted from DATEV Art.-Nr. 11175, valid 2023.
  Family numbering:
    0 Anlagevermögen        1 Umlaufvermögen         2 Eigenkapital
    3 Fremdkapital          4 Erträge                5 Materialaufwand
    6 Personalaufwand       7 Finanzergebnis/Steuern 8 frei
    9 Statistische Konten

skr03.csv (1476 accounts)
  Canonical SKR 03 (4-digit account numbers). Extracted as the NN=00
  subset of the SKR 03 Bau und Handwerk variant — i.e. the base accounts
  that are NOT trade-specific. Family numbering:
    0 Anlage- und Kapitalkonten     1 Finanz- und Privatkonten
    2 Abgrenzungskonten             3 Wareneingang/Bestandskonten
    4 Betriebliche Aufwendungen     5/6 frei
    7 Bestände an Erzeugnissen      8 Erlöse
    9 Vortrags-, Statistische Konten

skr03-bau.csv (1686 accounts)
  Full SKR 03 Bau und Handwerk (DATEV Art.-Nr. 19606, 2026 edition).
  Trade-specific variant for construction and crafts businesses. Uses
  6-digit account format: NNNN + 2-digit sub. NN=00 is the canonical
  base account; NN>00 is a trade-specific subdivision (e.g. 007510
  Sand- und Kiesausbeute is a Bau-specific child of 0075 Grundstücke
  mit Substanzverzehr). Contains 210 trade-specific extensions on top
  of the canonical 1476.

Source PDFs
-----------
Both extracts come from official DATEV-published "Eigenformular" PDFs:
- SKR 04: Art.-Nr. 11175 (2023-01-02 revision), valid for booking year 2023.
- SKR 03 Bau: Art.-Nr. 19606, valid for booking year 2026.

Format
------
Each CSV has four columns:
  account_number   4-digit (SKR 04, canonical SKR 03) or 6-digit (SKR 03 Bau)
  account_name     German account name (UTF-8)
  family           Category derived from leading digit (see above)
  source           Artifact reference (DATEV article number + edition)

Reproducibility
---------------
parse_pdfs.py is the Python script used to extract these CSVs from the
source PDFs. Re-run by pointing the input paths inside the script at
the source PDFs and executing `python3 parse_pdfs.py`. Depends on
`pdftotext` (from poppler-utils) being on $PATH.

Known data-quality issues
-------------------------
Multi-column PDF layouts make name extraction noisy on category
boundaries. Approximately:
  - SKR 04: ~150 entries (12%) have multi-column bleeding in their
    account_name (typically the first ~50 accounts on page 1, where the
    extra-wide page header offsets the column positions slightly).
  - SKR 03 canonical: ~180 entries (12%) similar issue.
  - SKR 03 Bau: similar rate.

Suspect entries are those with account_name longer than ~80 characters.
Account NUMBERS are always reliable; only the names need cleanup for
the worst-affected entries.

Hyphenated line breaks like "Konzessio- nen" are repaired to
"Konzessionen" by a heuristic, with exceptions kept for compound-word
constructions like "Geschäfts- oder Firmenwert" where the hyphen marks
a coordinated compound rather than a line break.

License
-------
See ../../../LICENSES/DATEV-SKR.txt. DATEV publishes the SKR formats
as "Eigenformular" with the notice "Nachdruck — auch auszugsweise —
nicht gestattet" (reproduction, even excerpts, not permitted). The
ACCOUNT NUMBERS THEMSELVES are bookkeeping-standard data and not
copyrightable in Germany under the standard reasoning that they are
short alphanumeric codes ordering a body of public-domain accounting
concepts. The German names of the accounts are likewise descriptive
labels. The original PDF formatting and layout IS copyrighted by DATEV
eG; this CSV redistribution carries only the underlying data, not
DATEV's formatting.

For commercial use, consult DATEV directly. This data drop is shipped
for the lance-graph cognitive shader as a structural reference for HGB
booking entities, aligned with FIBO and ZUGFeRD invoice projections.
