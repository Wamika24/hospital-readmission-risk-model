# Assumptions Register

Every financial and model assumption is listed with its source,
verified value, and break-even threshold from sensitivity analysis.
Last updated: 2026-05-09. Verified by: 13_business_case_sensitivity.py

---

## A1 — Cost per readmission

| Scenario | Value |
|---|---|
| Conservative | $13,000 |
| Base | $15,000 |
| Optimistic | $17,000 |

**Source:** Agency for Healthcare Research and Quality (AHRQ),
Statistical Brief #248, 2019. Average hospital stay involving
readmission: ~$15,200.

**Break-even:** Cost per readmission must exceed $5,147 for ROI to
remain positive. AHRQ benchmark of $15,000 is $9,853 above this floor.
Even at $8,000 (lowest tested), net savings remain $582,000.

---

## A2 — Intervention effectiveness rate

| Scenario | Value |
|---|---|
| Conservative | 15% |
| Base | 20% |
| Optimistic | 30% |

**Source:** Coleman EA et al. "The Care Transitions Intervention:
Results of a Randomized Controlled Trial." Archives of Internal
Medicine, 2006. Reported 20–25% reduction in 30-day readmissions
with structured transitional care.

**Break-even:** Effectiveness must exceed 6.9% for ROI to remain
positive. Base assumption of 20% is 13.1 percentage points above
this floor. At effectiveness of 10% (half the base assumption),
net savings are still $480,000.

---

## A3 — Outreach cost per patient

| Scenario | Value |
|---|---|
| Conservative | $200 (telephonic) |
| Base | $350 (mixed) |
| Optimistic | $500 (in-person) |

**Source:** AHRQ, "Preventing Readmissions Through Care Transitions."
Telephonic case management: $150–$250/patient. In-person transitional
care nurse visit: $400–$600/patient.

**Break-even:** Outreach cost must stay below $1,020/patient. Base
assumption of $350 has $670 of headroom. Even at $1,000/patient
(nearly 3× the base), net savings remain $60,000.

---

## A4 — Model precision at K

| K (monthly outreach) | Verified precision | Lift |
|---|---|---|
| 50 | 46.0% | 5.23× |
| 100 | 42.0% | 4.77× |
| 250 | 34.0% | 3.86× |
| 500 | 30.0% | 3.41× |

**Source:** Verified from catboost_readmission_model.cbm on
X_test.csv (14,303 patients). Script: 09_intervention_simulation.py.

**Break-even:** Precision must exceed 11.7% for ROI to remain positive.
Actual precision of 34.0% is 22.3 percentage points above this floor.

---

## A5 — Base readmission rate

**Value:** 8.80%

**Source:** UCI Diabetes 130-US Hospitals dataset (1999–2008),
cleaned cohort of 71,515 unique patients. 6,293 with readmit_30day_flag = 1.

**Note:** This rate reflects 30-day readmissions within the study
dataset after deduplication. National CMS all-cause 30-day readmission
rates are typically 15–20% and are not directly comparable due to
different population scope and encounter definitions.

---

## A6 — Scenario volumes

| Scenario | Monthly outreach | Annual outreach |
|---|---|---|
| Conservative | 100 patients | 1,200 |
| Base | 250 patients | 3,000 |
| Optimistic | 500 patients | 6,000 |

**Rationale:** Volumes reflect realistic care management team capacity.
A typical hospital care transitions team of 2–4 nurses can manage
100–250 post-discharge contacts per month. Optimistic scenario assumes
dedicated programme with expanded staffing.

---

## A7 — Kill condition

The only scenario in which the model loses money is if it performs
no better than random patient selection (precision = base rate = 8.8%).
At actual precision of 34.0%, the model is 3.86× better than random.

Sensitivity analysis confirms ROI is positive across all realistic
single-variable variations tested. The worst combined scenario
(random precision + low effectiveness + high outreach cost) produces
−$734,400 — which is the expected result of random outreach regardless
of model quality. The model itself does not create this loss; the loss
exists with or without the model if outreach costs exceed savings at
random precision.

---

## Disclaimer

All financial projections use published benchmark assumptions.
Actual savings depend on hospital census, payer mix, care team
capacity, and intervention protocol. This model should be validated
against hospital-specific data before operational use.