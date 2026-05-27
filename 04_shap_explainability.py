# 04_shap_explainability.py
# Hospital Readmission Risk Stratification
# Path: C:\Users\LENOVO\Desktop\readmission_ai_free
# Run with: py 04_shap_explainability.py

import os
import sys
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Required for saving plots without a display window
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

print("=" * 60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("Hospital Readmission Risk Stratification")
print("=" * 60)

# ----------------------------------------------------------
# SECTION 1: FILE PATHS
# ----------------------------------------------------------
PROJECT_DIR = r"C:\Users\LENOVO\Desktop\readmission_ai_free"
MODEL_PATH  = os.path.join(PROJECT_DIR, "catboost_readmission_model.cbm")
XTEST_PATH  = os.path.join(PROJECT_DIR, "X_test.csv")
YTEST_PATH  = os.path.join(PROJECT_DIR, "y_test.csv")

# Verify all required files exist before doing anything
print("\nChecking required files...")
for label, path in [
    ("Model file", MODEL_PATH),
    ("X_test",     XTEST_PATH),
    ("y_test",     YTEST_PATH),
]:
    exists = os.path.exists(path)
    status = "FOUND" if exists else "MISSING"
    print(f"  {status}: {label} -> {path}")
    if not exists:
        print(f"\nERROR: {label} not found at {path}")
        print("Fix: Run 03_ml_rebuild_real.py first to generate this file.")
        sys.exit(1)

print("\nAll required files found. Proceeding...\n")

# ----------------------------------------------------------
# SECTION 2: LOAD MODEL AND DATA
# ----------------------------------------------------------
print("Loading model...")
model = CatBoostClassifier()
model.load_model(MODEL_PATH)
print(f"  Model loaded successfully")

print("Loading test data...")
X_test = pd.read_csv(XTEST_PATH)
y_test = pd.read_csv(YTEST_PATH).squeeze()

# squeeze() converts a 1-column DataFrame into a plain Series
if not isinstance(y_test, pd.Series):
    y_test = y_test.iloc[:, 0]

print(f"  X_test: {X_test.shape[0]:,} patients x {X_test.shape[1]} features")
print(f"  y_test: {len(y_test):,} labels")
print(f"  Positive class (readmitted): {y_test.sum():,} ({y_test.mean():.1%})")
print(f"\n  Features in dataset:")
for i, col in enumerate(X_test.columns, 1):
    print(f"    {i:2}. {col}")

# ----------------------------------------------------------
# SECTION 3: FIX CATEGORICAL NaN VALUES + PREDICT PROBABILITIES
# ----------------------------------------------------------
print("\nFixing categorical columns (converting NaN to string 'None')...")

# When X_test is saved to CSV and reloaded, NaN values in categorical
# columns become float NaN. CatBoost requires strings or integers only.
# This gets the exact categorical column names the model was trained with
# and converts any NaN in those columns to the string "None".

cat_feature_indices = model.get_cat_feature_indices()
model_feature_names = model.feature_names_

cat_feature_names = [model_feature_names[i] for i in cat_feature_indices]
print(f"  Categorical features in model: {cat_feature_names}")

for col in cat_feature_names:
    if col in X_test.columns:
        nan_count = X_test[col].isna().sum()
        X_test[col] = X_test[col].fillna("None").astype(str)
        if nan_count > 0:
            print(f"  Fixed: {col} had {nan_count:,} NaN values → replaced with 'None'")

print("  Categorical columns fixed.")

print("\nGenerating risk scores for all patients...")
probabilities = model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------
# SECTION 4: PRECISION AT K VERIFICATION
# ----------------------------------------------------------
print("\nVerifying Precision@K metrics...")
sorted_idx = np.argsort(probabilities)[::-1]
y_sorted = y_test.values[sorted_idx]

for k in [100, 250, 500]:
    precision_k = y_sorted[:k].mean()
    print(f"  Precision@{k:<4}: {precision_k:.1%}  "
          f"({int(y_sorted[:k].sum())} actual readmissions in top {k})")

base_rate = y_test.mean()
decile_n  = len(y_test) // 10
lift      = y_sorted[:decile_n].mean() / base_rate
print(f"  Top-decile lift: {lift:.2f}x  (base rate: {base_rate:.1%})")

# ----------------------------------------------------------
# SECTION 5: SHAP VALUES
# ----------------------------------------------------------
print("\nCalculating SHAP values...")
print("  (This takes 1-5 minutes depending on your computer)")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print(f"  SHAP values calculated. Shape: {shap_values.shape}")

# ----------------------------------------------------------
# SECTION 6: FEATURE IMPORTANCE TABLE
# ----------------------------------------------------------
print("\nBuilding feature importance table...")
feature_importance = pd.DataFrame({
    'feature':         X_test.columns,
    'mean_abs_shap':   np.abs(shap_values).mean(axis=0),
    'mean_shap':       shap_values.mean(axis=0),
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

feature_importance['rank'] = feature_importance.index + 1

output_path = os.path.join(PROJECT_DIR, "shap_feature_importance.csv")
feature_importance.to_csv(output_path, index=False)
print(f"  Saved: shap_feature_importance.csv")
print(f"\n  Top 15 features by SHAP importance:")
print(f"  {'Rank':<5} {'Feature':<35} {'Mean |SHAP|':<15} Direction")
print(f"  {'-'*65}")
for _, row in feature_importance.head(15).iterrows():
    direction = "pushes UP risk" if row['mean_shap'] > 0 else "pushes DOWN risk"
    print(f"  {int(row['rank']):<5} {row['feature']:<35} {row['mean_abs_shap']:.5f}        {direction}")

# ----------------------------------------------------------
# SECTION 7: SHAP BEESWARM PLOT (summary)
# ----------------------------------------------------------
print("\nGenerating SHAP beeswarm plot...")
fig, ax = plt.subplots(figsize=(11, 8))
shap.summary_plot(
    shap_values,
    X_test,
    max_display=15,
    show=False,
    plot_size=None,
)
plt.title(
    "SHAP Summary — Feature Impact on 30-Day Readmission Risk\n"
    "Each dot = one patient | Color: red = high feature value, blue = low",
    fontsize=12,
    pad=12
)
plt.tight_layout()
out = os.path.join(PROJECT_DIR, "shap_beeswarm.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: shap_beeswarm.png")

# ----------------------------------------------------------
# SECTION 8: SHAP BAR CHART (for slides and Power BI)
# ----------------------------------------------------------
print("Generating SHAP bar chart...")
top_n   = feature_importance.head(12)
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(
    top_n['feature'][::-1],
    top_n['mean_abs_shap'][::-1],
    color='#378ADD',
    edgecolor='none',
    height=0.55,
)
ax.set_xlabel("Mean absolute SHAP value  (higher = more influence on prediction)",
              fontsize=10)
ax.set_title("Top 12 Model Drivers — 30-Day Readmission Risk\n"
             "CatBoost model trained on UCI Diabetes dataset",
             fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)
plt.tight_layout()
out = os.path.join(PROJECT_DIR, "shap_global_importance.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: shap_global_importance.png")

# ----------------------------------------------------------
# SECTION 9: INDIVIDUAL PATIENT EXPLANATION
# ----------------------------------------------------------
print("Generating individual patient explanation...")

# Highest risk patient
top_idx  = np.argmax(probabilities)
top_prob = probabilities[top_idx]
actual   = y_test.iloc[top_idx]

print(f"\n  Highest-risk patient (index {top_idx}):")
print(f"  Predicted probability: {top_prob:.1%}")
print(f"  Actual outcome:        {'READMITTED (correct flag)' if actual == 1 else 'Not readmitted (false positive)'}")

# Get this patient's SHAP values sorted by magnitude
patient_shap = pd.DataFrame({
    'feature': X_test.columns,
    'shap_value': shap_values[top_idx],
    'feature_value': X_test.iloc[top_idx].values,
}).sort_values('shap_value', key=abs, ascending=False)

print(f"\n  Why is this patient flagged? Top 10 factors:")
print(f"  {'Feature':<35} {'Value':<15} {'SHAP (impact)':<15} Effect")
print(f"  {'-'*75}")
for _, r in patient_shap.head(10).iterrows():
    effect = "raises risk" if r['shap_value'] > 0 else "lowers risk"
    print(f"  {r['feature']:<35} {str(r['feature_value']):<15} {r['shap_value']:+.4f}          {effect}")

# Save force plot as matplotlib figure
fig, ax = plt.subplots(figsize=(12, 3))
shap.waterfall_plot(
    shap.Explanation(
        values      = shap_values[top_idx],
        base_values = explainer.expected_value,
        data        = X_test.iloc[top_idx].values,
        feature_names = list(X_test.columns),
    ),
    max_display = 12,
    show        = False,
)
plt.title(f"Patient #{top_idx} — Readmission Risk: {top_prob:.1%}  |  "
          f"Actual: {'Readmitted' if actual == 1 else 'Not readmitted'}",
          fontsize=10)
plt.tight_layout()
out = os.path.join(PROJECT_DIR, "shap_individual_patient.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: shap_individual_patient.png")

# ----------------------------------------------------------
# SECTION 10: RISK SCORE TABLE FOR POWER BI
# ----------------------------------------------------------
print("\nBuilding risk score table for Power BI...")

# Define risk segments by probability thresholds
# High = top 10% of scores, Medium = middle 65%, Low = bottom 25%
high_threshold = np.percentile(probabilities, 90)
med_threshold  = np.percentile(probabilities, 25)

def assign_segment(p):
    if p >= high_threshold:
        return 'High'
    elif p >= med_threshold:
        return 'Medium'
    else:
        return 'Low'

risk_table = X_test.copy()
risk_table['risk_score']      = probabilities.round(4)
risk_table['risk_score_pct']  = (probabilities * 100).round(1)
risk_table['risk_segment']    = [assign_segment(p) for p in probabilities]
risk_table['actual_readmit']  = y_test.values

# Add top 3 SHAP driver columns for each patient
top3_features = feature_importance['feature'].head(3).tolist()
for feat in top3_features:
    idx = list(X_test.columns).index(feat)
    risk_table[f"shap_{feat}"] = shap_values[:, idx].round(4)

risk_table = risk_table.sort_values('risk_score', ascending=False).reset_index(drop=True)

out = os.path.join(PROJECT_DIR, "fact_patient_risk.csv")
risk_table.to_csv(out, index=False)
print(f"  Saved: fact_patient_risk.csv ({len(risk_table):,} patients)")
print(f"\n  Risk segment breakdown:")
for seg in ['High', 'Medium', 'Low']:
    sub  = risk_table[risk_table['risk_segment'] == seg]
    rate = sub['actual_readmit'].mean()
    print(f"    {seg:<8}: {len(sub):>6,} patients  |  readmission rate: {rate:.1%}")

print(f"\n  Threshold used: High >= {high_threshold:.4f}, Medium >= {med_threshold:.4f}")

# ----------------------------------------------------------
# SECTION 11: FINAL SUMMARY
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 60)
print("\nFiles saved to your project folder:")
outputs = [
    ("shap_feature_importance.csv",  "Import into Power BI Page 3"),
    ("shap_beeswarm.png",            "Add to GitHub README + slide deck"),
    ("shap_global_importance.png",   "Add to GitHub README + Power BI"),
    ("shap_individual_patient.png",  "Use in hospital demo slide"),
    ("fact_patient_risk.csv",        "Import into Power BI Page 3 patient table"),
]
for fname, usage in outputs:
    fpath = os.path.join(PROJECT_DIR, fname)
    size  = os.path.getsize(fpath) / 1024 if os.path.exists(fpath) else 0
    print(f"  {fname:<40} {size:>6.0f} KB  |  {usage}")

print(f"\nCopy these numbers into your MODEL_CARD.md:")
print(f"  Base readmission rate: {base_rate:.2%}")
print(f"  Precision@100:         {y_sorted[:100].mean():.2%}")
print(f"  Precision@250:         {y_sorted[:250].mean():.2%}")
print(f"  Precision@500:         {y_sorted[:500].mean():.2%}")
print(f"  Top-decile lift:       {lift:.2f}x")
print(f"  Top feature (SHAP):    {feature_importance.iloc[0]['feature']}")
print(f"\nStep 1 complete. Run next: py 05_calibration_analysis.py")