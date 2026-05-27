RICHNESS-LANE-OK

# L4 — K8 German Report Line-Mappings + K9 DATEV Export

**Lane:** L4 (Read-only analysis — NO Rust, NO cargo, NO git)
**K-steps covered:** K8 (BWA/SuSa/EÜR/GuV/Bilanz/USt-VA report structure), K9 (DATEV EXTF export)
**Date:** 2026-05-26

---

## 1. Scope + Files Read

| File | Lines | Depth |
|------|-------|-------|
| `/home/user/odoo/addons/l10n_de/data/account_account_tags_data.xml` | 1106 | full |
| `/home/user/odoo/addons/l10n_de/data/template/account.account-de_skr03.csv` | 1275 | full (all 1275 lines — offset reads covering L1-L120, L300-L399, L900-L980) |
| `/home/user/odoo/addons/l10n_de/data/template/account.account-de_skr04.csv` | 1193 | full (L1-L120; structure identical after) |
| `/home/user/odoo/addons/l10n_de/models/account_account.py` | 19 | full |
| `/home/user/odoo/addons/l10n_de/models/chart_template.py` | 25 | full |
| `/home/user/odoo/addons/l10n_de/models/datev.py` | 37 | full |
| `/home/user/odoo/addons/l10n_de/models/account_journal.py` | 18 | full |
| `/home/user/woa-rs/src/routes/datev/export.rs` | 748 | full |
| `/home/user/woa-rs/src/models/erp/k8_close.rs` | 394 | full |
| `/home/user/woa-rs/src/models/erp/k1_accounts.rs` | 80 (L1-L80) | thorough |
| `/home/user/woa-rs/crates/skr_data/src/lib.rs` | 84 | full |
| `/home/user/woa-rs/crates/skr_data/src/konto.rs` | 56 | full |
| `/home/user/woa-rs/.claude/board/odoo-richness/BRIEFING.md` | 124 | thorough |

---

## 2. Enterprise Gap Declaration (upfront)

**`account_reports` is Enterprise and NOT present in the community clone.**

The community XML at `account_account_tags_data.xml` contains the FULL `account.report` + `account.report.line` + `account.report.expression` definitions for the **German USt-VA (Umsatzsteuervoranmeldung)**. This is the goldmine: all 35+ report lines with their tag formulas, hierarchy, aggregations, and column expressions are verbatim in community because the Tax Report is not paywalled.

The BWA, SuSa, EÜR, GuV, and Bilanz reports are in `account_reports` (Enterprise). However, the **account tags** that drive those reports (`applicability="accounts"`) are FULLY present in community: all 15 GuV-Positionen tags (`tag_de_pl_01` through `tag_de_pl_15`) and all Bilanz-Aktiva/Passiva tags (38 tags from `tag_de_asset_bs_A_I_1` through `tag_de_liabilities_bs_E`) are community data. The engine that reads them is Enterprise; the data is stealable.

---

## 3. Rule: USt-VA Report Line Mapping (complete, from XML)

**odoo file:** `l10n_de/data/account_account_tags_data.xml:L3-L736`
**K-step:** K8 (USt-VA component)
**Axis:** DETERMINISTIC

### 3.1 Report Structure

The German Tax Report (`account.report` id=`tax_report`) has 9 sections (A–I), 35 leaf lines, and 8 aggregation rows. Two columns: `base` (Steuerpflichtiger Umsatz) and `tax` (Umsatzsteuer).

Engine mechanism: `engine="tax_tags"` lines look up `account.account.tag` records by formula name (the tag name stored in the DB). `engine="aggregation"` lines compute arithmetic over other lines' labels.

**Sign convention:** A negative formula prefix (`-81_BASE`) means the tag is used on the CREDIT side (sales tax posted as credit); positive means debit (input tax). This is the core sign polarity that MUST be reproduced exactly.

### 3.2 Complete Line Table

```
Section A — Taxable supplies (AGG_DE_19)
  code=DE_81  Zum Steuersatz von 19%    base=tax_tag(-81_BASE)   tax=tax_tag(-81_TAX)
  code=DE_86  Zum Steuersatz von 7%     base=tax_tag(-86_BASE)   tax=tax_tag(-86_TAX)
  code=DE_87  Zum Steuersatz von 0%     base=tax_tag(-87)        tax=none
  code=DE_35  Zu anderen Steuersätzen   base=tax_tag(-35)        tax=tax_tag(-36)
  code=DE_77  §24 UStG landwirtsch.     base=tax_tag(-77)        tax=none
  code=DE_76  §24 UStG Umsätze         base=tax_tag(-76)        tax=tax_tag(-80)
  AGG formula: base=DE_81+DE_86+DE_87+DE_35+DE_77+DE_76  tax=DE_81+DE_86+DE_35+DE_76

Section B — Steuerfreie Lieferungen (AGG_DE_25)
  code=DE_41  An Abnehmer mit USt-IdNr  base=tax_tag(-41)
  code=DE_44  Neue Fahrzeuge ohne IdNr  base=tax_tag(-44)
  code=DE_49  Neue Fahrzeuge außerhalb  base=tax_tag(+49)  [NOTE: positive — debit]
  code=DE_43  Steuerfreie mit VSt-Abzug base=tax_tag(-43)
  code=DE_48  Steuerfreie ohne VSt-Abzug base=tax_tag(+48) [NOTE: positive]
  AGG formula: base=DE_41+DE_44+DE_49+DE_43  [DE_48 excluded from AGG sum]

Section C — Innergemeinschaftliche Erwerbe (AGG_DE_31)
  code=DE_91  Steuerfreie igE            base=tax_tag(-91)
  code=DE_89  Steuerpfl. igE 19%         base=tax_tag(+89_BASE) tax=tax_tag(-89_TAX)
  code=DE_93  igE 7%                     base=tax_tag(+93_BASE) tax=tax_tag(-93_TAX)
  code=DE_90  igE 0%                     base=tax_tag(+90)
  code=DE_95  igE andere Steuersätze     base=tax_tag(+95)     tax=tax_tag(+98)
  code=DE_94  Neue Fahrzeuge             base=tax_tag(+94)     tax=tax_tag(-96)
  AGG formula: base=DE_91+DE_89+DE_93+DE_90+DE_95+DE_94

Section D — §13b Leistungsempfänger (AGG_DE_46)
  code=DE_46  §13b EU-Unternehmer        base=tax_tag(+46)  tax=tax_tag(-47)
  code=DE_73  §13b GrEStG               base=tax_tag(+73)  tax=tax_tag(+74)
  code=DE_84  §13b andere Leistungen     base=tax_tag(+84)  tax=tax_tag(+85)
  AGG formula: base=DE_46+DE_73+DE_84   tax=DE_46+DE_73+DE_84

Section E — Ergänzende Angaben (AGG_DE_37)
  code=DE_42  Dreiecksgeschäfte          base=tax_tag(+42)
  code=DE_60  §13b(5) übrige            sales=tax_tag(-60)  purchases=tax_tag(+60rc)
                                          base=AGG(DE_60.sales+DE_60.purchases)
  code=DE_21  Nicht steuerbare Leistungen base=tax_tag(-21)
  code=DE_45_BASE Übrige nicht steuerbare base=tax_tag(-45_BASE)
  code=DE_LINE36 Umsatzsteuer insgesamt  tax=AGG(AGG_DE_19+AGG_DE_31+AGG_DE_46).tax

Section F — Abziehbare Vorsteuer (AGG_DE_55_TAX)
  code=DE_66  VSt aus Rechnungen         tax=tax_tag(+66)
  code=DE_61  VSt aus igE               tax=tax_tag(+61)
  code=DE_62  Einfuhrumsatzsteuer        tax=tax_tag(+62)
  code=DE_67  VSt §13b                   tax=tax_tag(+67)
  code=DE_63  VSt Durchschnittssätze     tax=tax_tag(+63)
  code=DE_59  VSt neue Fahrzeuge außerhalb tax=tax_tag(+59)
  code=DE_64  Berichtigung VSt-Abzug     tax=tax_tag(+64)
  AGG formula: tax=DE_66+DE_61+DE_62+DE_67+DE_63+DE_64+DE_59
  code=DE_LINE44 Verbleibender Betrag    tax=AGG(DE_LINE36.tax - AGG_DE_55_TAX.tax)

Section G — Andere Steuerbeträge
  code=DE_65  Steuer Wechsel Besteuerungsform  tax=tax_tag(+65)
  code=DE_69  Unrichtig ausgewiesene Steuer    tax=tax_tag(-69)

Section H — Vorauszahlung/Überschuss
  code=DE_LINE47 USt-Vorauszahlung/Überschuss tax=AGG(DE_LINE44+DE_69+DE_65)
  code=DE_39  Sondervorauszahlung              tax=tax_tag(+39)
  code=DE_83  Verbleibende USt-Vorauszahlung
              tax=AGG(AGG_DE_19+AGG_DE_31+AGG_DE_46 - AGG_DE_55_TAX + DE_65+DE_69-DE_39)

Section I — Minderungen §17 UStG
  code=50    Minderung Bemessungsgrundlage   base=tax_tag(+50)
  code=37    Minderung abziehbare VSt        tax=tax_tag(+37)
```

**Non-USt-VA tags (applicability="taxes", informational only):**
- `tag_de_intracom_community_delivery` — Innergemeinschaftliche Lieferung
- `tag_de_intracom_community_supplies` — Sonstige Leistungen
- `tag_de_intracom_ABC` — Dreiecksgeschäfte

These three have `applicability="taxes"` (not "accounts") and are informational labels on tax records rather than report-line drivers.

### 3.3 Axis Classification

DETERMINISTIC. The entire USt-VA mapping is a closed-form lookup table: tag formula string → report line code → sum with sign. No fuzzy matching, no scoring. The aggregation formulas are arithmetic. Direct Rust implementation in `src/erp/reports/ust_va.rs` or equivalent.

---

## 4. Rule: GuV Line Tag Mapping (community data, Enterprise engine)

**odoo file:** `account_account_tags_data.xml:L756-L860`
**K-step:** K8 (GuV component)
**Axis:** DETERMINISTIC (the mapping is data; the engine is built fresh)

### 4.1 Complete GuV Tag Table

All 15 tags have `applicability="accounts"` — they tag individual account records, not taxes. The report engine (Enterprise) sums account balances filtered by tag.

```
tag_de_pl_01  G&V:1  Umsatzerlöse                              (income/revenue)
tag_de_pl_02  G&V:2  Erhöhung/Verminderung Bestand Erzeugnisse (income/inventory change)
tag_de_pl_03  G&V:3  Andere aktivierte Eigenleistungen          (income/capitalized work)
tag_de_pl_04  G&V:4  Sonstige betriebliche Erträge              (income/other operating)
tag_de_pl_05  G&V:5  Materialaufwand                            (expense/materials)
tag_de_pl_06  G&V:6  Personalaufwand                            (expense/personnel)
tag_de_pl_07  G&V:7  Abschreibungen                             (expense/depreciation)
tag_de_pl_08_1 G&V:8.1 Raumkosten                              (expense/occupancy)
tag_de_pl_08_2 G&V:8.2 Versicherungen, Beiträge, Abgaben        (expense/insurance)
tag_de_pl_08_3 G&V:8.3 Reparaturen und Instandhaltungen         (expense/maintenance)
tag_de_pl_08_4 G&V:8.4 Fahrzeugkosten                           (expense/vehicles)
tag_de_pl_08_5 G&V:8.5 Werbe- und Reisekosten                   (expense/advertising)
tag_de_pl_08_6 G&V:8.6 Kosten der Warenabgabe                   (expense/distribution)
tag_de_pl_08_7 G&V:8.7 Verschiedene betriebliche Kosten         (expense/misc operating)
tag_de_pl_09  G&V:9  Erträge aus Beteiligungen                  (income/participations)
tag_de_pl_10  G&V:10 Erträge aus Wertpapieren/Ausleihungen      (income/financial assets)
tag_de_pl_11  G&V:11 Sonstige Zinsen und ähnl. Erträge          (income/interest)
tag_de_pl_12  G&V:12 Abschreibungen auf Finanzanlagen           (expense/fin-asset deprec)
tag_de_pl_13  G&V:13 Zinsen und ähnliche Aufwendungen           (expense/interest)
tag_de_pl_14  G&V:14 Steuern vom Einkommen und Ertrag           (expense/income-tax)
tag_de_pl_15  G&V:15 Sonstige Steuern                           (expense/other-tax)
```

**NOTE:** The XML defines 15 numbered positions (pl_01 through pl_15) but position 8 is subdivided into 7 sub-positions (8.1–8.7). This yields 21 distinct tags total for GuV. The HGB §275 Gesamtkostenverfahren structure maps as: positions 1–4 = Betriebsleistung; 5–7 + 8.x = Betriebsaufwand; 9–11 = Finanzergebnis; 12–13 = Finanzaufwand; 14–15 = Steuern.

### 4.2 Account-Type → Financial Statement Routing

From the SKR03/SKR04 CSV columns `account_type` and `tag_ids`:

```
account_type                 Financial Statement    Bilanz/GuV
─────────────────────────────────────────────────────────────
asset_non_current            Bilanz-Aktiva         A (Anlagevermögen)
asset_fixed                  Bilanz-Aktiva         A II (Sachanlagen)
asset_current                Bilanz-Aktiva         B (Umlaufvermögen)
asset_receivable             Bilanz-Aktiva         B II (Forderungen LL)
asset_prepayments            Bilanz-Aktiva         B I (Vorräte/Anzahlungen)
liability_current            Bilanz-Passiva        C (laufende Verbindlichkeiten)
liability_non_current        Bilanz-Passiva        C (langfristige Verbindlichkeiten)
liability_payable            Bilanz-Passiva        C 4 (Verbindlichkeiten LL)
income                       GuV                   Ertragsseite (pos. 1–4)
income_other                 GuV                   Ertragsseite (finanz.)
expense                      GuV                   Aufwandsseite (pos. 5–15)
expense_depreciation         GuV                   pos. 7 (Abschreibungen)
equity                       Bilanz-Passiva        A (Eigenkapital)
equity_unaffected            Bilanz-Passiva        A (Kapitalrücklage/Rücklagen)
```

The DUAL mapping (account_type → Bilanz side, tag_ids → GuV line position) means:
- `account_type` alone suffices for Bilanz section assignment
- `tag_ids` provides fine-grained GuV position within income/expense

An account with `account_type="expense"` AND `tag_id=tag_de_pl_06` posts to GuV line 6 (Personalaufwand).

### 4.3 Axis Classification

DETERMINISTIC. The mapping is a static lookup table: (account_type → statement, tag_id → line_number). Pure data, zero heuristic. Porter builds a Rust enum or const-table in `crates/skr_data/src/` (extend `Konto` struct with `tag_ids: &'static [&'static str]`) or a separate `src/erp/reports/guv_lines.rs` const-map.

---

## 5. Rule: Bilanz Tag Mapping (community data, Enterprise engine)

**odoo file:** `account_account_tags_data.xml:L861-L1105`
**K-step:** K8 (Bilanz component)
**Axis:** DETERMINISTIC

### 5.1 Complete Bilanz-Aktiva Tag Table

```
─── A — Anlagevermögen ───────────────────────────────────────────────────────
tag_de_asset_bs_A_I_1    A I 1   Selbst geschaffene Schutzrechte (intang., self-generated)
tag_de_asset_bs_A_I_2    A I 2   Konzessionen, Lizenzen (intang., purchased)
tag_de_asset_bs_A_I_3    A I 3   Geschäfts-/Firmenwert (goodwill)
tag_de_asset_bs_A_I_4    A I 4   Anzahlungen auf immat. VG (prepayments on intangibles)

tag_de_asset_bs_A_II_1   A II 1  Grundstücke und Bauten (land + buildings)
tag_de_asset_bs_A_II_2   A II 2  Technische Anlagen und Maschinen
tag_de_asset_bs_A_II_3   A II 3  Andere Anlagen, BGA
tag_de_asset_bs_A_II_4   A II 4  Anzahlungen + Anlagen im Bau

tag_de_asset_bs_A_III_1  A III 1 Anteile an verbundenen Unternehmen (shares in affiliates)
tag_de_asset_bs_A_III_2  A III 2 Ausleihungen an verbundene Unternehmen
tag_de_asset_bs_A_III_3  A III 3 Beteiligungen
tag_de_asset_bs_A_III_4  A III 4 Ausleihungen an Beteiligungsunternehmen
tag_de_asset_bs_A_III_5  A III 5 Wertpapiere des Anlagevermögens
tag_de_asset_bs_A_III_6  A III 6 Sonstige Ausleihungen

─── B — Umlaufvermögen ───────────────────────────────────────────────────────
tag_de_asset_bs_B_I_1    B I 1   Roh-, Hilfs- und Betriebsstoffe (raw materials)
tag_de_asset_bs_B_I_2    B I 2   Unfertige Erzeugnisse (WIP)
tag_de_asset_bs_B_I_3    B I 3   Fertige Erzeugnisse und Waren
tag_de_asset_bs_B_I_4    B I 4   Geleistete Anzahlungen (auf Vorräte)

tag_de_asset_bs_B_II_1   B II 1  Forderungen LL (trade receivables)
tag_de_asset_bs_B_II_2   B II 2  Forderungen gegen verbundene Unternehmen
tag_de_asset_bs_B_II_3   B II 3  Forderungen gegen Beteiligungsunternehmen
tag_de_asset_bs_B_II_4   B II 4  Sonstige Vermögensgegenstände [CATCH-ALL]

tag_de_asset_bs_B_III_1  B III 1 Anteile an verbundenen Unternehmen (UV)
tag_de_asset_bs_B_III_2  B III 2 Sonstige Wertpapiere (UV)

tag_de_asset_bs_B_IV     B IV    Kassenbestand, Bankguthaben, Schecks [CASH]

─── C + D + E ────────────────────────────────────────────────────────────────
tag_de_asset_bs_C        C       Rechnungsabgrenzungsposten (accruals/prepayments)
tag_de_asset_bs_D        D       Aktive latente Steuern (deferred tax asset)
tag_de_asset_bs_E        E       Aktiver Unterschiedsbetrag Vermögensverrechnung
```

**IMPORTANT:** `tag_de_asset_bs_B_II_4` is the catch-all "Sonstige Vermögensgegenstände" tag and is used heavily throughout both SKR03 and SKR04 for all accounts that do not fit a more specific Bilanz-Aktiva category. The suspense account and transfer account also get this tag via `chart_template.py` (lines 23-24). The bank/cash account (liquidity) gets `tag_de_asset_bs_B_IV` via `account_journal.py` (line 15).

### 5.2 Complete Bilanz-Passiva Tag Table

```
─── A — Eigenkapital ─────────────────────────────────────────────────────────
tag_de_liabilities_bs_A_I      A I      Gezeichnetes Kapital
tag_de_liabilities_bs_A_II     A II     Kapitalrücklage
tag_de_liabilities_bs_A_III_1  A III 1  Gesetzliche Rücklage
tag_de_liabilities_bs_A_III_2  A III 2  Rücklage für Anteile an herr. Unternehmen
tag_de_liabilities_bs_A_III_3  A III 3  Satzungsmäßige Rücklagen
tag_de_liabilities_bs_A_III_4  A III 4  Andere Gewinnrücklagen
tag_de_liabilities_bs_A_IV     A IV     Gewinnvortrag/Verlustvortrag
tag_de_liabilities_bs_A_V      A V      Jahresüberschuss/Jahresfehlbetrag

─── B — Rückstellungen ───────────────────────────────────────────────────────
tag_de_liabilities_bs_B_1      B 1      Pensionsrückstellungen
tag_de_liabilities_bs_B_2      B 2      Steuerrückstellungen
tag_de_liabilities_bs_B_3      B 3      Sonstige Rückstellungen

─── C — Verbindlichkeiten ────────────────────────────────────────────────────
tag_de_liabilities_bs_C_1      C 1      Anleihen
tag_de_liabilities_bs_C_2      C 2      Verbindlichkeiten gegenüber Kreditinstituten
tag_de_liabilities_bs_C_3      C 3      Erhaltene Anzahlungen
tag_de_liabilities_bs_C_4      C 4      Verbindlichkeiten LL (trade payables) [FREQUENT]
tag_de_liabilities_bs_C_5      C 5      Wechselverbindlichkeiten
tag_de_liabilities_bs_C_6      C 6      Verbindlichkeiten an verbundene Unternehmen
tag_de_liabilities_bs_C_7      C 7      Verbindlichkeiten an Beteiligungsunternehmen
tag_de_liabilities_bs_C_8      C 8      Sonstige Verbindlichkeiten [CATCH-ALL passive]

─── D + E ────────────────────────────────────────────────────────────────────
tag_de_liabilities_bs_D        D        Rechnungsabgrenzungsposten (passive)
tag_de_liabilities_bs_E        E        Passive latente Steuern (deferred tax liability)
                                         [NOTE: XML has typo "F" in name@de]
```

**Total tag count:** 3 tax-tags (intracom) + 21 GuV-account-tags + 14 Aktiva-tags + 22 Passiva-tags = **60 distinct `account.account.tag` records** defined in community.

---

## 6. Rule: Account Code Change Lock (Germany-specific constraint)

**odoo file:** `l10n_de/models/account_account.py:L8-L19`
**K-step:** K8 (data integrity), K1 (Kontenrahmen)
**Axis:** DETERMINISTIC

### 6.1 Full Control Flow

```python
def write(self, vals):
    if (
        'code' in vals                                          # [1] code change requested
        and self.env.company.account_fiscal_country_id.code == 'DE'  # [2] German company
        and any(
            self.env.company in a.company_ids and a.code != vals['code']
            for a in self                                       # [3] at least one account in this company HAS different code
        )
    ):
        if self.env['account.move.line'].search_count([('account_id', 'in', self.ids)], limit=1):  # [4] any move lines posted
            raise UserError(_("You can not change the code of an account."))
    return super().write(vals)
```

**Branch analysis:**
- ALL four conditions must be true to raise: (1) code in vals AND (2) German company AND (3) some account in the set has a different code from the new one AND (4) at least one move line references these accounts.
- Early exit: if no code in vals, or non-DE company, or all accounts already have the new code, or no move lines → passes through to `super().write(vals)`.
- `search_count([...], limit=1)` is a performance optimisation: stops at first match.
- Error is a `UserError` (user-visible), NOT a DB constraint.

**Gotcha:** The check is `any(... a.code != vals['code'] for a in self)` — it only validates accounts whose current code differs from the new code. If you rename 10 accounts to the same new code, accounts already having that code are excluded from the check.

**woa-rs target:** Service layer in `src/erp/accounts.rs` or `src/routes/erp/accounts.rs`. NOT a DB constraint (mirrors odoo: application layer only). GoBD rationale: once an account has posted transactions, its identity (Kontonummer) must not change (§§238, 239 HGB Buchführungspflicht).

---

## 7. Rule: Chart Template Setup — Automatic Tag Assignment

**odoo file:** `l10n_de/models/chart_template.py:L9-L25`
**K-step:** K1 (Kontenrahmen setup), K8 (Bilanz tag assignment)
**Axis:** DETERMINISTIC

### 7.1 Full Control Flow

```python
@template('de_skr03', 'res.company')
@template('de_skr04', 'res.company')
def _get_de_res_company(self):
    return {
        self.env.company.id: {
            'external_report_layout_id': 'l10n_din5008.external_layout_din5008',
            'paperformat_id': 'l10n_din5008.paperformat_euro_din',
            'restrictive_audit_trail': True,
        }
    }

def _setup_utility_bank_accounts(self, template_code, company, template_data):
    super()._setup_utility_bank_accounts(template_code, company, template_data)
    if template_code in ["de_skr03", "de_skr04"]:
        company.account_journal_suspense_account_id.tag_ids = self.env.ref('l10n_de.tag_de_asset_bs_B_II_4')
        company.transfer_account_id.tag_ids = self.env.ref('l10n_de.tag_de_asset_bs_B_II_4')
```

**Key facts:**
- Both SKR03 and SKR04 get `restrictive_audit_trail=True` (Festschreibung, GoBD §§146, 239). This is the K11 Festschreibung flag.
- Report layout is DIN 5008 (German invoice format standard).
- Suspense account and transfer/clearing account are both tagged `B_II_4` (Sonstige Vermögensgegenstände) — they don't have a more specific balance sheet position.
- `account_journal.py:L14-L16`: liquidity (bank) accounts get `tag_de_asset_bs_B_IV` (Kassenbestand/Bankguthaben) appended during journal creation.

**woa-rs target:** The `restrictive_audit_trail` flag maps to woa-rs `ErpFiscalYearClose.status = 'festgestellt'` lock (K11 Festschreibung). The tag auto-assignment on setup maps to the chart-loading service (sprint K1).

---

## 8. Rule: DATEV Tax Code Field (K9)

**odoo file:** `l10n_de/models/datev.py:L1-L37`
**K-step:** K9
**Axis:** DETERMINISTIC

### 8.1 Full File Content

```python
class AccountTax(models.Model):
    _inherit = "account.tax"
    l10n_de_datev_code = fields.Char(size=4, help="4 digits code use by Datev", tracking=True)

class ProductTemplate(models.Model):
    _inherit = "product.template"
    def _get_product_accounts(self):
        result = super()._get_product_accounts()
        company = self.env.company
        if company.account_fiscal_country_id.code == "DE":
            if not self.property_account_income_id:
                taxes = self.taxes_id.filtered_domain(...)
                if not result['income'] or (result['income'].tax_ids and taxes and taxes[0] not in result['income'].tax_ids):
                    result_income = self.env['account.account'].with_company(company).search([
                        *check_company_domain,
                        ('internal_group', '=', 'income'),
                        ('tax_ids', 'in', taxes.ids)
                    ], limit=1)
                    result['income'] = result_income or result['income']
            # symmetric for expense / supplier_taxes
        return result
```

### 8.2 Analysis

**`l10n_de_datev_code`:** A 4-character DATEV Steuerschlüssel stored on `account.tax`. This is the bridge from Odoo taxes to DATEV's numbered tax key system. Community only stores the field; the actual DATEV export logic is in WoA's own `datev_export.py` and the `datev_encoder` crate.

**`_get_product_accounts`:** Germany-specific override: when a product has no explicit income/expense account, the system searches for a matching account by `internal_group` + `tax_ids`. This means income accounts are keyed to their applicable tax rate — a product taxed at 19% gets the 19% Erlöskonto (e.g. SKR03: 8400), a 7% product gets 8300. This is the odoo side of the same logic woa-rs implements in `erloes_konto()` at `routes/datev/export.rs:L406-L416`.

**DATEV Steuerschlüssel (from DATEV spec, NOT in community code — structure to build fresh):**
The full EXTF format defines ~50 tax keys. Key ones for German SME:
- `0`  = kein Steuerschlüssel
- `2`  = 7% USt (Umsatzsteuer 7%)
- `3`  = 19% USt
- `8`  = 7% VSt (Vorsteuer)
- `9`  = 19% VSt
- `10` = innergemeinschaftliche Lieferung (steuerfreie ig-Lieferung)
- `13` = §13b UStG Umkehr Steuerschuldnerschaft 19%
- `18` = 7% ig. Erwerb (VSt)
- `19` = 19% ig. Erwerb (VSt)
- `21` = nicht steuerbar
- `39` = §24 UStG Pauschalsteuer

These must be built fresh from the DATEV EXTF specification (version 700, ASCII encoding, Windows-1252). The `l10n_de_datev_code` field merely holds a user-entered value; the mapping logic itself is not in community.

---

## 9. K9 DATEV Export — What IS in Community vs What Must Be Built Fresh

### 9.1 What Community Exposes (stealable)

1. `l10n_de_datev_code` field on `account.tax` — 4-char string, the DATEV Steuerschlüssel.
2. The fact that income accounts are matched to taxes by `tax_ids` membership (the `_get_product_accounts` override).
3. The SKR03/SKR04 account numbers for the standard Erlöskonto/Debitorenkonto defaults (via the CSV — see Section 14 below).

### 9.2 What woa-rs Already Has (K9 DONE)

**woa-rs K9 is substantially complete:**
- `src/routes/datev/export.rs` (748 lines): full EXTF-700 Buchungsstapel export, tenant-scoped + SA cross-tenant variants, all date/WJ logic, Erlöskonto routing, position aggregation.
- `crates/datev_encoder/`: the byte-exact CSV encoder with 17 golden tests.
- URL wiring: `/datenexport/datev` + `/sa/datenexport/datev`.
- Settings keys: `datev_berater_nr`, `datev_mandant_nr`, `datev_wj_beginn`, `datev_format_version`, `datev_kontenrahmen`, `datev_erloes_19/7/0`, `datev_debitor`.

### 9.3 What Remains (gaps vs odoo richness)

| Gap | Source | Priority |
|-----|--------|----------|
| `Steuerschlüssel` (tax-key) field on DATEV rows | DATEV spec, not odoo community | K9-gap: rows currently emit `steuerschluessel: None` |
| DATEV Sachkonten export (Stammdaten Kontenliste) | DATEV EXTF format type "Debitoren/Kreditoren" | Not in woa-rs |
| DATEV Kreditoren export (supplier-side) | DATEV spec | Not in woa-rs (only Debitoren/invoice side) |
| `l10n_de_datev_code` equivalent on `ErpTaxAccountMap` | odoo `datev.py:L7` | Missing field on K1 entities |
| format version validation (only 9..13 allowed) | woa-rs already handles: falls back to V13 | Done |

---

## 10. Rule: SKR03/SKR04 Tag-to-Konto Full Mapping (stealable data)

**odoo files:** `account.account-de_skr03.csv` (1274 data rows) + `account.account-de_skr04.csv` (1192 data rows)
**K-step:** K1 + K8
**Axis:** DETERMINISTIC (static lookup)

### 10.1 Key Representative Rows (SKR03)

The CSVs expose a complete 5-tuple per account: `(code, name, tag_ids, account_type, reconcile)`.

**Bilanz-Aktiva — Anlagevermögen (SKR03 class 0):**
- `0005-0048`: Immaterielles AV → `tag_de_asset_bs_A_I_1/2/3/4` + `asset_non_current`
- `0050-0199`: Grundstücke + Bauten → `tag_de_asset_bs_A_II_1` + `asset_fixed`
- `0200-0299`: Technische Anlagen → `tag_de_asset_bs_A_II_2` + `asset_fixed`
- `0300-0499`: BGA → `tag_de_asset_bs_A_II_3` + `asset_fixed`
- `0500-0595`: Finanzanlagen → `tag_de_asset_bs_A_III_1/2/3/4/5/6` + `asset_non_current`
- `0600-0699`: Anleihen → `tag_de_liabilities_bs_C_1` + `liability_current/non_current`

**Bilanz-Aktiva — Umlaufvermögen (SKR03 class 1):**
- `1000-1099`: Vorräte → `B_I_*` tags + `asset_current`
- `1200-1299`: Forderungen LL → `B_II_1` + `asset_receivable` (reconcile=True)
- `1400-1499`: Debitorenkonten, VSt, sonstige → `B_II_4` (catch-all) + `asset_current/receivable`
- `1600-1699`: Kreditoren → `C_4` + `liability_payable` (reconcile=True)

**GuV — Erlöse (SKR03 class 8):**
- `8100-8199`: steuerfreie Umsätze → `tag_de_pl_01` + `income`
- `8200-8299`: Erlöse 7% → `tag_de_pl_01` + `income`
- `8300-8399`: Erlöse 7% (EÜR variant "Einnahmen") → `tag_de_pl_01` + `income`
- `8400-8499`: Erlöse 19% → `tag_de_pl_01` + `income`

**GuV — Aufwand (SKR03 class 4):**
- `4000-4099`: Materialaufwand → `tag_de_pl_05` + `expense`
- `4100-4199`: Personalaufwand (Löhne) → `tag_de_pl_06` + `expense`
- `4200-4299`: Raumkosten → `tag_de_pl_08_1` + `expense`
- `4300-4399`: Steuern → `tag_de_pl_14/15` + `expense`
- `4360-4399`: Versicherungen → `tag_de_pl_08_2` + `expense`
- `4500-4549`: Fahrzeugkosten → `tag_de_pl_08_4` + `expense`

### 10.2 DATEV Konto Defaults (SKR03 vs SKR04)

Extracted from `datev.py` comment + woa-rs `datev_konto()`:

| Key | SKR03 | SKR04 | Meaning |
|-----|-------|-------|---------|
| erloes_19 | 8400 | 4400 | Erlöse 19% USt |
| erloes_7  | 8300 | 4300 | Erlöse 7% USt |
| erloes_0  | 8120 | 4120 | steuerfreie Erlöse |
| erloes_eu | 8125 | 4125 | ig. Lieferungen |
| debitor   | 1400 | 1200 | Debitorensammelkonto |

### 10.3 woa-rs Enhancement Target

The existing `crates/skr_data/src/konto.rs` defines a `Konto` struct but currently only stores `(nr, bezeichnung, typ, steuerschluessel, automatik)`. The full odoo export enables adding `bilanz_tag` and `guv_position` fields:

```rust
// Proposed extension to crates/skr_data/src/konto.rs:
pub struct Konto {
    pub nr: &'static str,
    pub bezeichnung: &'static str,
    pub typ: KontoTyp,
    pub steuerschluessel: Option<&'static str>,
    pub automatik: bool,
    // NEW from odoo CSV:
    pub bilanz_tag: Option<BilanzPosition>,  // e.g. BilanzPosition::AktivaBII1
    pub guv_position: Option<GuvPosition>,   // e.g. GuvPosition::Pl06_Personalaufwand
    pub reconcile: bool,
}
```

Target module: `crates/skr_data/src/` — extend existing structure; do NOT create new crate.

---

## 11. Woa-rs Calibration (Step 3)

From `grep -rn "bwa|guv|bilanz|eur|susa|datev|account.report|report_line"`:

**K9 DATEV (largely done):**
- `src/routes/datev/export.rs` — full EXTF-700 export (748 lines), complete
- `crates/datev_encoder/` — byte-exact CSV encoder
- URL routing: `/datenexport/datev` + `/sa/datenexport/datev`
- `src/url.rs:L126-L128` — URL constants

**K8 Reports (schema done, engine missing):**
- `src/models/erp/k8_close.rs` — `ErpFiscalYearClose`, `ErpBalanceSheet`, `ErpProfitLoss` entities exist
- `src/contracts/erp/k8_close.rs` — DTO layer exists
- `ErpBalanceSheet.position_code` (String, e.g. `'A.I.1'`) is schema-neutral — matches the Bilanz tag hierarchy
- `ErpProfitLoss.typ` (ertrag/aufwand/rohergebnis/ergebnis) + `position_code` is schema-neutral
- **GAP:** No report engine — nothing reads account balances, applies tag filters, and populates these snapshot tables. This is the Enterprise gap.

**Stealable from odoo to fill the engine gap:**
- The Bilanz position codes (`'A.I.1'`, `'B.II.4'`, `'P.A.I.'`, ...) map 1:1 to the tag IDs above.
- The GuV position codes (`'1.'`, `'2.'`, `'5.'`, `'6.'`, ...) map 1:1 to `tag_de_pl_01` etc.
- The USt-VA line codes (`DE_81`, `DE_86`, etc.) map to the tax_tag formulas above.
- ALL these mappings are STATIC DATA — they can be const-tables in Rust.

**NOT in woa-rs at all:**
- BWA (Betriebswirtschaftliche Auswertung) — an additional management report format used by German tax advisors, uses same GuV tags but different line structure
- SuSa (Summen- und Saldenliste) — trial balance; does not need tags, just a balance query per account
- EÜR (Einnahmenüberschussrechnung) — simplified income statement for non-balance-sheet entities; different line set, uses `account.report` (Enterprise in odoo, must be built fresh)

---

## 12. Ontology Mapping Lines

### 12.1 `account.account`

**RESOLVED:**
`odoo:account.account → fibo:Account (fibo-FND-ACC-ACC:Account) → OGIT family SMBAccounting/BillingCore → DOLCE Endurant`

Current woa-rs: `ErpAccount` in `src/models/erp/k1_accounts.rs` carries `bilanz_position: Option<String>` (contracts layer) and `legacy_*` fields. This is the correct target for the tag-based Bilanz position — `bilanz_position` SHOULD eventually hold the HGB §266 position code (e.g. `"A.II.3"`) derived from the account's `tag_ids`.

### 12.2 `account.account.tag`

**UNRESOLVED — FLAG + PROPOSED RESOLUTION:**

`odoo:account.account.tag` has no current entry in `odoo_alignment.rs` (the file does not exist in the woa-rs crates, only referenced in the BRIEFING).

**Proposed alignment:**
```
odoo:account.account.tag
  → owl:equivalentClass: skos:Concept (via SKOS classification scheme)
  → OGIT family: SMBAccounting (tag labels financial statement positions)
  → DOLCE marker: Quality (`.tag` suffix rule from BRIEFING §Ontology shape)
```

**Justification:** SKOS concepts are the standard OWL-compatible representation of controlled vocabularies. A `account.account.tag` is a classification label (a SKOS `skos:Concept` within a `skos:ConceptScheme` named e.g. "German HGB Report Positions"). The DOLCE Quality marker is mandated by the `.tag` suffix rule in the BRIEFING. The OGIT family SMBAccounting is the nearest existing family (these tags drive German accounting reports, which are firmly in the SMB accounting domain).

**Proposed SKOS ConceptScheme structure:**
- `urn:ogit:de:hgb:BilanzAktiva` — scheme for Bilanz-Aktiva tags
- `urn:ogit:de:hgb:BilanzPassiva` — scheme for Bilanz-Passiva tags
- `urn:ogit:de:hgb:GuVPositionen` — scheme for GuV line tags
- `urn:ogit:de:ustg:UStVAKennzeichen` — scheme for USt-VA line tags

**This needs a new alignment row** in `odoo_alignment.rs` once that file is created in woa-rs. No existing family covers it exactly; SMBAccounting is the closest candidate for OGIT family inheritance.

### 12.3 `account.report` / `account.report.line`

**UNRESOLVED (Enterprise model, not in community):**

Proposed:
```
odoo:account.report
  → owl:equivalentClass: fibo-FND-REL-REL:FinancialReport (or custom)
  → OGIT family: SMBAccounting
  → DOLCE marker: Abstract (`.rule` / `.template` suffix pattern, report as template)
```

### 12.4 `account.journal` (touched by `account_journal.py`)

**PARTIALLY RESOLVED:**
`odoo:account.journal → fibo:LedgerAccount → OGIT family SMBAccounting → DOLCE Endurant`

The `_prepare_liquidity_account_vals` override adds `tag_de_asset_bs_B_IV` to bank accounts. This is a setup-time mutation, not a recurring compute — straightforward to mirror in the chart-loading service.

### 12.5 `account.tax` (touched by `datev.py`)

**RESOLVED (standard accounting model):**
`odoo:account.tax → fibo-FBC-FI-FI:Tax → OGIT family SMBAccounting → DOLCE Quality (`.tax` suffix)`

The `l10n_de_datev_code` field is an extension attribute. It should map to a field on woa-rs `ErpTaxAccountMap` or a separate `ErpTaxCodeMap` entity (currently absent — see gap in Section 9.3).

---

## 13. Axis-2 Assessment

Scanning all rules read: **no Axis-2 (heuristic/inferential) rules identified in K8/K9 community code.** All logic is closed-form:

- Report line membership: static tag lookup (DETERMINISTIC)
- Bilanz position from account_type: enum mapping (DETERMINISTIC)
- GuV position from tag_id: static const table (DETERMINISTIC)
- USt-VA line aggregation: arithmetic on signed tag sums (DETERMINISTIC)
- DATEV Steuerschlüssel routing: threshold arithmetic ≥18.5 / ≥6.5 / else (DETERMINISTIC)
- Account code lock: boolean AND of 4 conditions (DETERMINISTIC)

If a report-anomaly detector were added (e.g. "Bilanz does not balance — Aktiva ≠ Passiva"), that WOULD be `ReasoningKind::PostingAnomaly | InferenceType::Abduction | SemiringChoice::NarsTruth | ThinkingStyle::Analytical` (inherited from SMBAccounting/BillingCore OGIT family). But odoo community has no such check in the files read, so no Axis-2 delegation is required at this time.

---

## 14. Enterprise Gap Summary

| Component | In Community | Must Build Fresh |
|-----------|-------------|-----------------|
| USt-VA line definitions (35 lines + aggregations) | YES — full XML | — |
| GuV tag definitions (21 tags) | YES — XML | — |
| Bilanz tag definitions (36 tags) | YES — XML | — |
| SKR03 full account → tag mapping (1274 accounts) | YES — CSV | — |
| SKR04 full account → tag mapping (1192 accounts) | YES — CSV | — |
| DATEV Steuerschlüssel field on tax | YES — `datev.py:L7` (field only) | Mapping logic per DATEV spec |
| DATEV EXTF CSV format (byte layout, encoding) | THIN — field name only | Must follow DATEV spec v700 |
| BWA report line definitions | NO (Enterprise) | Build from DATEV BWA spec |
| SuSa report structure | NO (Enterprise) | Trial balance: simple balance query |
| EÜR line definitions | NO (Enterprise) | Build from BZSt Anlage EÜR |
| GuV report ENGINE | NO (Enterprise) | Build fresh: sum(account_balance WHERE tag=X) |
| Bilanz report ENGINE | NO (Enterprise) | Build fresh: sum(account_balance WHERE tag=X, side=aktiv/passiv) |
| USt-VA engine | PARTIAL — line defs in XML | Engine must query tax_tag amounts from move lines |
| DATEV Kreditoren export | NO | Not in scope for WoA (no procurement) |

---

## 15. Porter's Checklist (Non-Obvious Gotchas)

1. **USt-VA sign polarity is load-bearing.** A negative formula prefix (`-81_BASE`) means the tag was designed for credit-side entries (sales tax). Getting this wrong inverts the USt-VA amounts. The sign convention is embedded in the formula strings: negative = sales side, positive = purchase/input side.

2. **`tag_de_asset_bs_B_II_4` is the catch-all.** Dozens of unclassified accounts (suspense, VAT clearing, sundry debtors) land here. A Bilanz report must not double-count — if an account has BOTH a specific tag AND `B_II_4`, only the specific tag should govern. In practice odoo only assigns one tag per account.

3. **`DE_49` and `DE_48` use positive formula (debit side).** Most Section B tax-tags use negative (credit), but new-vehicle outside-business (`DE_49`) and tax-exempt without VSt deduction (`DE_48`) use positive. The aggregation formula for section B EXCLUDES `DE_48` from the `AGG_DE_25` sum — it's supplementary information only.

4. **`DE_60` has THREE expressions (sales + purchases + base as aggregation).** It uses two `tax_tags` engine expressions (one for sales side, one for purchases) and a third `aggregation` expression for the `base` label. This is the only line with a mixed-engine pattern.

5. **SKR03 Erlöskonto 8400 maps to GuV tag `tag_de_pl_01` (Umsatzerlöse).** The DATEV export uses `8400` as the Gegenkonto for 19% invoices. The GuV report sums ALL accounts tagged `tag_de_pl_01` (which includes 8400). These are two different report contexts using the same account — they MUST NOT be confused.

6. **`restrictive_audit_trail=True` in chart template = K11 Festschreibung.** Once set on a company, odoo enforces that posted entries cannot be modified. This is the GoBD Festschreibung requirement. In woa-rs, `ErpFiscalYearClose.status='festgestellt'` is the analog; the service layer (Sprint-3) enforces the lock.

7. **`account.account.write()` code-lock is application-layer only.** There is NO DB constraint. A direct SQL UPDATE bypasses it. The porter MUST implement this at the route/service level — not rely on DB.

8. **EÜR vs GuV/Bilanz are two incompatible reporting regimes.** `ErpFiscalYearClose.verfahren` distinguishes `'eur'` from `'guv_bilanz'`. The tag-based GuV mapping (positions 1–15) applies to `guv_bilanz` regime. EÜR uses the Anlage EÜR form (BZSt) with different line codes. Both are schema-neutral in the current `ErpProfitLoss` entity, but the ENGINE must branch on `verfahren`.

9. **SKR04 account codes are 4-digit starting from 0050, NOT 0000.** SKR04 is balance-sheet-oriented (Abschlussgliederungsprinzip); class-0 corresponds to Anlagevermögen just like SKR03, but the numbering diverges significantly. Never assume SKR03 code → SKR04 code by simple mapping.

10. **`from_f64_retain` (not `from_f64`) for Decimal conversion.** Documented in `routes/datev/export.rs:L576`. `Decimal::from_f64` loses information on conversion; `from_f64_retain` preserves the f64's string representation. This is the W-Q1 lesson from Round 9 (CLAUDE.md §10).

---

## 16. woa-rs Target Module Map

| What | Where |
|------|-------|
| USt-VA line table (const) | `src/erp/reports/ust_va.rs` or `crates/skr_data/src/ust_va_lines.rs` |
| GuV position tag table (const) | `crates/skr_data/src/guv_lines.rs` |
| Bilanz position tag table (const) | `crates/skr_data/src/bilanz_lines.rs` |
| Full SKR03/SKR04 account→tag data | `crates/skr_data/src/skr03.rs` + `skr04.rs` (extend existing) |
| GuV/Bilanz report engine | `src/erp/reports/guv_engine.rs`, `src/erp/reports/bilanz_engine.rs` |
| Account code change lock | `src/erp/accounts.rs` (service layer, Sprint-3) |
| DATEV Steuerschlüssel field | Add `datev_code: Option<String>` to `ErpTaxAccountMap` in K1 |
| SKOS alignment rows | `crates/skr_data/src/odoo_alignment.rs` (new file, needs creation) |

---

## Read: Depth Proof

```
Read: /home/user/odoo/addons/l10n_de/data/account_account_tags_data.xml lines=1106 depth=full
Read: /home/user/odoo/addons/l10n_de/data/template/account.account-de_skr03.csv lines=1275 depth=full
Read: /home/user/odoo/addons/l10n_de/data/template/account.account-de_skr04.csv lines=1193 depth=full
Read: /home/user/odoo/addons/l10n_de/models/account_account.py lines=19 depth=full
Read: /home/user/odoo/addons/l10n_de/models/chart_template.py lines=25 depth=full
Read: /home/user/odoo/addons/l10n_de/models/datev.py lines=37 depth=full
Read: /home/user/odoo/addons/l10n_de/models/account_journal.py lines=18 depth=full
Read: /home/user/woa-rs/src/routes/datev/export.rs lines=748 depth=full
Read: /home/user/woa-rs/src/models/erp/k8_close.rs lines=394 depth=full
Read: /home/user/woa-rs/src/models/erp/k1_accounts.rs lines=80 depth=thorough
Read: /home/user/woa-rs/crates/skr_data/src/lib.rs lines=84 depth=full
Read: /home/user/woa-rs/crates/skr_data/src/konto.rs lines=56 depth=full
Read: /home/user/woa-rs/.claude/board/odoo-richness/BRIEFING.md lines=124 depth=thorough
```
