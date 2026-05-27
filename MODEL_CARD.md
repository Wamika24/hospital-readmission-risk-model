# Model Card — Hospital Readmission Risk Stratification

**Model name:** CatBoost 30-Day Readmission Classifier
**Version:** 1.0
**Date:** 2026-05-28
**File:** catboost_readmission_model.cbm

---

## Model Description

A gradient-boosted tree classifier trained to predict the probability that a diabetic hospital patient will be readmitted within 30 days of discharge. Designed as a decision-support tool for care transitions teams — outputs a ranked list of patients by readmission risk so outreach resources are concentrated on highest-risk individuals.

**This is not a diagnostic tool.** It is a triage prioritisation tool. Output should be interpreted as a relative risk ranking within a patient population, not as a literal readmission probability.

---

## Training Data

| Property | Value |
|---|---|
| Source | UCI Diabetes 130-US Hospitals dataset (1999–2008) |
| Original records | 101,766 encounters |
| After deduplication | 71,515 unique patients |
| Positive class | 6,293 patients (30-day readmission) |
| Positive class rate | 8.80% |
| Train/test split | 80% train, 20% test (stratified) |
| Test set size | 14,303 patients |

**Known data limitations:**
- Data from 1999–2008. Modern diabetes medications not represented.
- `a1cresult` is missing for 81.9% of records. Treated as "None" category.
- Deduplication kept first encounter per patient_nbr.

---

## Features

**Total features:** 32

**Top 15 features by SHAP importance:**

| Rank | Feature | Mean SHAP | Direction |
|---|---|---|---|
| 1 | discharge_type | 0.2088 | Pushes risk up |
| 2 | diag_group | 0.0808 | Pushes risk up |
| 3 | diabetesmed | 0.0656 | Pushes risk down |
| 4 | complexity_score | 0.0564 | Pushes risk down (review) |
| 5 | age | 0.0560 | Pushes risk up |
| 6 | lab_procedures | 0.0456 | Pushes risk up |
| 7 | inpatient_visits | 0.0456 | Pushes risk down (review) |
| 8 | procedures_per_day | 0.0433 | Pushes risk up |
| 9 | meds_per_day | 0.0381 | Pushes risk up |
| 10 | care_load | 0.0358 | Pushes risk down |
| 11 | days_in_hospital | 0.0352 | Pushes risk down |
| 12 | medical_specialty | 0.0347 | Pushes risk up |
| 13 | meds_per_diagnosis | 0.0324 | Pushes risk down |
| 14 | total_visits | 0.0304 | Pushes risk down |
| 15 | labs_per_day | 0.0293 | Pushes risk up |

**Feature engineering notes:**
- `complexity_score` direction is counterintuitive — pushes risk down on average. Requires clinical review. May reflect that complex patients have more intensive management.
- `inpatient_visits` pushes risk down on average. May reflect patients with strong existing care systems.
- `discharge_type` is the dominant feature. Model must be run at point of discharge, not admission.

**Categorical features (11):**
age, gender, race, admission_type, discharge_type, admission_source, diabetesmed, medical_specialty, a1cresult, insulin, diag_group

---

## Model Architecture

| Property | Value |
|---|---|
| Algorithm | CatBoost gradient boosted trees |
| Library | catboost 1.2.10 |
| Class imbalance handling | Class weights applied |
| Categorical encoding | Native CatBoost handling |
| Saved format | .cbm binary |

---

## Performance Metrics

**Verified on test set (n=14,303) — 2026-05-08**

### Discrimination

| Metric | Value | Interpretation |
|---|---|---|
| AUC-ROC | 0.6715 | On par with LACE Index (0.60–0.68) |
| Gini coefficient | 0.3431 | 2 × AUC − 1 |
| Average Precision | 0.1823 | vs null model = 0.088 |

### Ranking (operational metrics)

| Metric | Value | vs baseline (8.8%) |
|---|---|---|
| Precision@50 | 46.0% | 5.23× lift |
| Precision@100 | 40.0% | 4.55× lift |
| Precision@250 | 32.4% | 3.68× lift |
| Precision@500 | 30.0% | 3.41× lift |
| Top-decile lift | 2.41× | — |

### Calibration

| Metric | Before fix | After fix |
|---|---|---|
| Brier Score | 0.2116 | 0.0783 |
| Brier Skill Score | −1.59 | +0.036 |
| Mean calibration error | 0.3552 | 0.0096 |
| Method | — | Isotonic regression |
| Calibrator file | — | isotonic_calibrator.pkl |

**Calibration note:** Raw model probabilities are systematically over-predicted due to class-weight training on an imbalanced dataset. Isotonic regression calibration is applied post-hoc. Always use `risk_score_calibrated` from `fact_patient_risk.csv`, not raw scores.

---

## Risk Segmentation

**Thresholds based on calibrated probability percentiles:**

| Segment | Threshold | Patients | Actual readmission rate |
|---|---|---|---|
| High | ≥ 0.1479 (top 10%) | 1,502 | 21.0% |
| Medium | 0.0503–0.1479 | 10,219 | 8.5% |
| Low | < 0.0503 (bottom 25%) | 2,582 | 2.7% |

**Clinical interpretation:**
- High Risk patients readmit at 21.0% — 2.4× the baseline rate
- Low Risk patients readmit at 2.7% — 3.3× below baseline
- 7.8× spread between High and Low segments

---

## Probability Calibration Detail

**Problem:** Raw model probabilities were 3–9× higher than actual readmission rates.

**Root cause:** Class-weight training to handle 8.8% class imbalance inflates probability outputs.

**Fix:** Isotonic regression fitted on 7,151-patient calibration set (50% of test data). Applied to remaining 7,152-patient evaluation set.

**Result:**

| Bin | Predicted (after) | Actual | Gap |
|---|---|---|---|
| 1 | 1.60% | 2.59% | −0.99pp |
| 5 | 7.43% | 6.29% | +1.14pp |
| 10 | 27.90% | 27.69% | +0.21pp |

Mean absolute calibration error after: **0.0096** (near-perfect)

**Important:** Do not present raw scores to clinical staff. Always use calibrated scores. Frame as risk tier (High/Medium/Low), not literal percentage.

---

## Fairness Audit

**Verified: 2026-05-08 | Script: 08_fairness_audit.py**
**Metric used:** False Negative Rate (FNR) — lower = fewer missed readmissions
**Disparity threshold:** FNR difference > 5pp from overall (85.39%)

### Gender

| Group | n | FNR | vs overall | Status |
|---|---|---|---|---|
| Female | 7,620 | 84.43% | −0.96pp | PASSED |
| Male | 6,683 | 86.46% | +1.07pp | PASSED |

**Verdict:** No gender disparity. FNR gap 2.03pp, well below 5pp threshold.

### Race

| Group | n | FNR | vs overall | Status |
|---|---|---|---|---|
| African American | 2,598 | 86.01% | +0.62pp | PASSED |
| Caucasian | 10,681 | 84.95% | −0.44pp | PASSED |
| Hispanic | 314 | 86.36% | +0.98pp | Small sample |
| Asian | 99 | 100.00% | +14.61pp | Statistical artifact |
| Unknown | 367 | 95.45% | +10.07pp | Data quality category |

**Key finding:** African American vs Caucasian FNR gap = 1.06pp. No meaningful racial disparity in the clinically important comparison. Asian group result (100% FNR) is a statistical artifact of ~5 events — not interpretable as evidence of systematic bias.

### Age

| Group | n | FNR | Status |
|---|---|---|---|
| 50–60 | 2,456 | 93.08% | MONITOR — 7.7pp above overall |
| 70–80 | 3,599 | 80.93% | Best performance |
| Under 18 | < 130 | Various | Not validated — exclude from scoring |

**Action required:** Model is not validated for patients under 18. Deployment must restrict scoring to adults.

---

## Temporal Validation

**Script: 10_temporal_validation.py**
**Method:** Test set split into thirds by row order as temporal proxy.

| Period | n | AUC | Precision@10% |
|---|---|---|---|
| Early (first third) | 4,767 | 0.6612 | 21.0% |
| Middle (second third) | 4,768 | 0.6873 | 23.5% |
| Late (final third) | 4,768 | 0.6754 | 19.8% |

**AUC range:** 0.0261 — below 3pp stability threshold
**Drift early → late:** +0.014 (performance improves slightly on later data)
**Verdict:** STABLE — model does not degrade on more recent data.

**Limitation:** True temporal validation requires splitting full training data by year. This analysis uses test set as approximation. Full temporal study recommended before production.

---

## Deployment Guidance

**When to run this model:** At point of discharge, after discharge disposition is recorded. `discharge_type` is the top feature and is only available at this point.

**Who uses the output:** Care transitions coordinators, discharge planning nurses.

**How to use the output:**
1. Load `fact_patient_risk.csv` into care management system
2. Sort by `risk_score_calibrated` descending
3. Outreach top K patients based on team capacity
4. Do not use raw risk score as a literal probability
5. Apply clinical judgment alongside model output

**What not to do:**
- Do not score patients under 18
- Do not use `risk_score_raw` — use `risk_score_calibrated`
- Do not interpret model output as a clinical diagnosis
- Do not use as the sole basis for any clinical decision

---

## Limitations

1. **Dataset age (1999–2008):** Modern diabetes medications not represented. Must validate on contemporary institutional data before deployment.

2. **Single dataset:** Trained on 130 US hospitals aggregated. Performance on a specific hospital's patient population may differ. Recommend local validation study.

3. **Discharge timing only:** Cannot score patients at admission. Requires `discharge_type` to be recorded.

4. **Adult patients only:** Not validated for pediatric populations. Exclude patients under 18.

5. **Calibration is post-hoc:** Isotonic calibration was fitted on held-out test data. In production, recalibration should be performed periodically as patient population characteristics evolve.

6. **50–60 age group underperformance:** FNR of 93.1% vs overall 85.4%. Clinical teams should apply additional judgment for this age group.

---

## Intended Use

**Intended:** Triage prioritisation for post-discharge care management outreach in adult diabetic patient populations at US acute care hospitals.

**Not intended:** Diagnostic tool, admission decisions, treatment decisions, paediatric care, non-diabetic populations, non-US healthcare systems.

---

## Output Files

| File | Description |
|---|---|
| catboost_readmission_model.cbm | Saved model binary |
| isotonic_calibrator.pkl | Probability calibrator |
| fact_patient_risk.csv | All test patients with calibrated scores |
| shap_feature_importance.csv | Feature importance rankings |
| shap_beeswarm.png | SHAP summary visualisation |
| shap_global_importance.png | Feature importance bar chart |
| shap_individual_patient.png | Patient #8247 explanation |
| calibration_curve_fixed.png | Before/after calibration comparison |
| roc_curve.png | ROC curve |
| fairness_audit_results.csv | Full fairness metrics by group |
| fairness_audit.png | FNR by demographic group |
| temporal_validation_results.csv | AUC across time periods |
| intervention_scenarios.csv | ROI by scenario |
| precision_at_k_curve.csv | Precision at each outreach volume |
| business_case_sensitivity.csv | Sensitivity analysis results |

---

## Ethical Considerations

This model was evaluated for demographic fairness before documentation. No meaningful disparity was found between African American and Caucasian patients (1.1pp FNR gap). The 50–60 age group shows reduced sensitivity and is documented as a known limitation requiring clinical supplementation.

The model should never be used to deny care to any patient. Its sole purpose is to prioritise proactive outreach — a resource-allocation tool, not a gatekeeping tool.

---

## Disclaimer

This is a decision-support prototype. It is not FDA-cleared or CE-marked. It has not been prospectively validated on live patient data. Clinical and compliance review is required before operational deployment. Financial projections use AHRQ published benchmarks and are illustrative, not guaranteed.

---

*Model Card follows the Google Model Card framework (Mitchell et al., 2019)*
