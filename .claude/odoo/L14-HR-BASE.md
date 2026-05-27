RICHNESS-LANE-OK

# Lane L14 — HR Base Data (Employee / Org / Contract Structure)

## Enterprise gap (read first)
`hr_payroll` is **absent** (`ls addons | grep payroll` → empty). No payslip/salary-rule/structure models. Community provides only data/structure hooks; the **payroll engine is built fresh** in woa-rs. This lane is the K13 data foundation, almost entirely AXIS-A (entity + structural rules), **0 Savant seeds**.

## Sources read
- hr/models/hr_employee.py : L1-1865 : full
- hr/models/hr_version.py : L1-700 : full
- hr/models/hr_department.py : L1-243 : full
- hr/models/hr_job.py : L1-94 : full
- hr/models/hr_contract_type.py, hr_payroll_structure_type.py, res_company.py : full

## Ontology rows — ALL hr.* resolve to `None`
| odoo class | owl pivot | OGIT family | DOLCE |
|---|---|---|---|
| `hr.employee` | vcard:Individual (work resource) | **None → propose 0x90 HRFoundation** (or inherit 0x80 via work_contact_id) | Endurant |
| `hr.department` | org:OrganizationalUnit | None → 0x90 | Endurant |
| `hr.job` | org:Role | None → 0x90 | Abstract |
| `hr.version` | temporal slice of vcard:Individual | None → 0x90 | Perdurant |
| `hr.contract.type` | schema:EmploymentType | None → 0x90 | Abstract |
| `hr.payroll.structure.type` | (stub; Enterprise boundary) | None | Abstract |

**Layer-2 alignment axiom needed**: mint `0x90 HRFoundation` family OR map `hr.employee` → 0x80 SmbFoundryCustomer (employee-as-internal-partner via work_contact_id). Decide at synthesis.

## Rules extracted (17 AXIS-A, 0 AXIS-B)

### R1 — current_version_id compute [AXIS-A]
- hr_employee.py:L524-539 — latest hr.version where date_version<=today desc, fallback earliest; stored; daily cron `_cron_update_current_version_id`.

### R2 — version date_start/date_end window [AXIS-A]
- hr_version.py:L561-580 — date_start = max(date_version, contract_date_start); date_end = min(next_version.date_version-1, contract_date_end). Effective validity window ≠ raw date_version. Compute on read (needs sibling versions).

### R3 — contract overlap constraint [AXIS-A]
- hr_version.py:L239-278 — no two versions overlap [contract_date_start, contract_date_end(NULL=open→date.max)]; start<=end; end NULL⇒start NOT NULL; partial unique (employee_id, date_version) WHERE active.

### R4 — gap-tolerant continuous occupation [AXIS-A]
- hr_employee.py:L459-485 — gap <4 days between versions = continuous (seniority); ≥4 breaks chain. date_end False ⇒ date(2100,1,1) for gap calc only.

### R5 — department complete_name hierarchy [AXIS-A]
- hr_department.py:L81-92 — recursive parent.complete_name + ' / ' + name; _parent_store; master_department_id = parent_path root; cycle check raises.

### R6 — dept manager → employee parent_id propagation [AXIS-A]
- hr_department.py:L129-147 — on manager change, update employees whose parent_id == old manager to new (direct members only; manual overrides untouched; exclude new manager).

### R7 — job headcount [AXIS-A]
- hr_job.py:L42-57 — no_of_employee = count active; expected = + no_of_recruitment; unique (name, company, department); no_of_recruitment>=0.

### R8 — employee→user→partner linkage + work-contact sync [AXIS-A]
- hr_employee.py:L85-94, L793-836 — user_id (unique per company), work_contact_id (res.partner, auto-created), user_partner_id; work_phone/email computed from work_contact when ≤1 linked employee; inverse pushes to partner; barcode unique [A-Za-z0-9]{≤18}; PIN digits-only; archive nulls others' parent_id/coach_id.

### R9 — coach default compute [AXIS-A]
- hr_employee.py:L806-814 — on parent_id change, coach_id=new manager if coach was old manager or empty.

### R10 — newly_hired (90-day) [AXIS-A]
- hr_employee.py:L421-430 — create_date > now-90d; override field via _get_new_hire_field.

### R11 — contract/work-permit expiry cron [AXIS-A]
- hr_employee.py:L1168-1200 — exact-date match (contract_date_end == today + notice_period[7], work_permit == today + [60]); schedule activity to hr_responsible.

### R12 — salary_distribution (sum-to-100 + rounding) [AXIS-A]
- hr_employee.py:L286-349 — JSON bank_account→{sequence, amount, amount_is_percentage}; % entries sum 100±0.0001; redistribute on add/remove (currency.round, last gets exact remainder); primary = lowest sequence.

### R13 — normalized wage (no payroll) [AXIS-A]
- hr_version.py:L475-486 — wage*12/52/hours_per_week; no calendar ⇒ wage (hourly); hours 0 ⇒ 0. Override point for fresh payroll engine.

### R14 — contract-template whitelist [AXIS-A]
- hr_version.py:L443-458 — copy only [job_id, department_id, contract_type_id, structure_type_id, wage, resource_calendar_id, hr_responsible_id].

### R15 — structure_type_id default [AXIS-A]
- hr_version.py:L534-550 — match country_id==company.country_id else generic (NULL). Payroll entry-point discriminant.

### R16 — version-period calendar query [AXIS-A]
- hr_employee.py:L1585-1710 — for [start,stop], collect version-slices active in window, clamp to [max(start,date_start), min(stop,date_end)], map to resource_calendar (per-version TZ). Critical for multi-schedule payroll periods (DE law).

### R17 — has_read_access dept ACL [AXIS-A]
- hr_department.py:L46-52 — non-HR users see departments they (or chain) manage; `child_of` on parent_path.

## Data hooks for fresh payroll engine
wage; structure_type_id→hr.payroll.structure.type (country discriminant); contract_type_id; resource_calendar_id (hours_per_week, tz); contract_date_start/end + trial_date_end; bank_account_ids + salary_distribution (SEPA split); ssnid + identification_id (statutory); marital + children + km_home_work (1.609 km factor — Lohnsteuer inputs); company notice periods; `_get_salary_costs_factor` stub=12.0 (DE may need 13/14 for Weihnachts/Urlaubsgeld).

## Open questions
1. hr.* OGIT family: mint 0x90 HRFoundation or inherit 0x80? (decide at synthesis).
2. hr.version temporal model: bi-temporal table vs snapshot-log; preserve date_start/end window semantics.
3. _parent_store dept tree → ltree column or recursive CTE.
4. structure_type_id.country_id = payroll ruleset discriminant; DE structure type required.
5. _get_salary_costs_factor 12 vs 13/14 (DE) — override in engine.
6. salary_distribution rounding via decimal_money (currency-aware, last-gets-remainder).

## Depth-proof footer
```
Read: hr/models/hr_employee.py lines=1865 depth=full
Read: hr/models/hr_version.py lines=700 depth=full
Read: hr/models/hr_department.py lines=243 depth=full
Read: hr/models/hr_job.py lines=94 depth=full
Read: hr/models/hr_contract_type.py lines=23 depth=full
Read: hr/models/hr_payroll_structure_type.py lines=19 depth=full
Read: hr/models/res_company.py lines=18 depth=full
```
