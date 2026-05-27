# 05_calibration_analysis.py
# Uses the SAVED model (catboost_readmission_model.cbm) and saved test set.
# Run with: py 05_calibration_analysis.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)

print("=" * 60)
print("CALIBRATION ANALYSIS")
print("Using saved model: catboost_readmission_model.cbm")
print("=" * 60)

PROJECT_DIR = r"C:\Users\LENOVO\Desktop\readmission_ai_free"
MODEL_PATH  = os.path.join(PROJECT_DIR, "catboost_readmission_model.cbm")
XTEST_PATH  = os.path.join(PROJECT_DIR, "X_test.csv")
YTEST_PATH  = os.path.join(PROJECT_DIR, "y_test.csv")

# ── verify files ──────────────────────────────────────────
print("\nChecking files...")
for label, path in [("Model", MODEL_PATH), ("X_test", XTEST_PATH), ("y_test", YTEST_PATH)]:
    if not os.path.exists(path):
        print(f"  MISSING: {label} -> {path}")
        print("  Fix: run py 03_ml_rebuild_real.py first.")
        sys.exit(1)
    print(f"  FOUND: {label}")

# ── load ──────────────────────────────────────────────────
model  = CatBoostClassifier()
model.load_model(MODEL_PATH)
X_test = pd.read_csv(XTEST_PATH)
y_test = pd.read_csv(YTEST_PATH).squeeze()
if not isinstance(y_test, pd.Series):
    y_test = y_test.iloc[:, 0]
y_test = y_test.astype(int)

print(f"\nLoaded: {len(X_test):,} patients  |  positive class: {y_test.mean():.2%}")

# ── fix categoricals ──────────────────────────────────────
cat_names = [model.feature_names_[i] for i in model.get_cat_feature_indices()]
for col in cat_names:
    if col in X_test.columns:
        X_test[col] = X_test[col].fillna("None").astype(str)

# ── predict ───────────────────────────────────────────────
probs = model.predict_proba(X_test)[:, 1]
print(f"Predictions generated. Score range: {probs.min():.4f} – {probs.max():.4f}")

# ── core metrics ──────────────────────────────────────────
auc   = roc_auc_score(y_test, probs)
ap    = average_precision_score(y_test, probs)
brier = brier_score_loss(y_test, probs)
gini  = 2 * auc - 1

# Null model benchmark (always predict base rate)
base_rate   = y_test.mean()
brier_null  = brier_score_loss(y_test, np.full(len(y_test), base_rate))
brier_skill = 1 - (brier / brier_null)   # > 0 means better than null

print("\n===== CALIBRATION METRICS (VERIFIED — SAVED MODEL) =====")
print(f"  Test patients:        {len(y_test):,}")
print(f"  Base readmission rate:{base_rate:.2%}")
print(f"  AUC-ROC:              {auc:.4f}   (random=0.500, perfect=1.000)")
print(f"  Gini coefficient:     {gini:.4f}   (= 2×AUC - 1)")
print(f"  Average Precision:    {ap:.4f}   (null={base_rate:.3f})")
print(f"  Brier Score:          {brier:.4f}   (null={brier_null:.4f})")
print(f"  Brier Skill Score:    {brier_skill:.4f}   (>0 = better than null, 1.0 = perfect)")
print("=" * 55)

if brier_skill < 0:
    print("  WARNING: Brier Skill Score < 0 — model is WORSE than")
    print("  always predicting the base rate. Check calibration plot.")
elif brier_skill < 0.1:
    print("  NOTE: Low Brier Skill Score — model ranks patients well")
    print("  (good AUC) but absolute probability estimates are imprecise.")
else:
    print("  Brier Skill Score positive — model is better calibrated than null.")

# ── calibration curve ─────────────────────────────────────
fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy='quantile')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: calibration curve
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration', alpha=0.6)
ax.plot(mean_pred, fraction_pos, 'o-', color='#378ADD', linewidth=2,
        markersize=7, label=f'CatBoost model (Brier={brier:.3f})')
ax.fill_between(mean_pred, fraction_pos,
                np.interp(mean_pred, [0,1], [0,1]),
                alpha=0.08, color='#378ADD')
ax.set_xlabel("Mean predicted probability", fontsize=11)
ax.set_ylabel("Fraction actually readmitted", fontsize=11)
ax.set_title("Calibration curve\n(how well probabilities match reality)", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: prediction histogram
ax2 = axes[1]
ax2.hist(probs[y_test == 0], bins=40, alpha=0.6, color='#378ADD',
         label=f'Not readmitted (n={int((y_test==0).sum()):,})', density=True)
ax2.hist(probs[y_test == 1], bins=40, alpha=0.6, color='#E24B4A',
         label=f'Readmitted (n={int((y_test==1).sum()):,})', density=True)
ax2.set_xlabel("Predicted readmission probability", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Score distribution by actual outcome\n(good model = two separated humps)", fontsize=11)
ax2.legend(fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle(
    f"Calibration Analysis — 30-Day Readmission Model\n"
    f"AUC-ROC: {auc:.4f}  |  Avg Precision: {ap:.4f}  |  "
    f"Brier Skill: {brier_skill:.4f}  |  n={len(y_test):,}",
    fontsize=10, y=1.01
)
plt.tight_layout()
out = os.path.join(PROJECT_DIR, "calibration_curve.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: calibration_curve.png")

# ── calibration bins table ─────────────────────────────────
print("\nCalibration bins (predicted vs actual, 10 equal-frequency bins):")
print(f"  {'Bin':<5} {'Predicted prob':<18} {'Actual rate':<15} {'Gap':<10} {'n patients'}")
print(f"  {'-'*60}")
gap_list = []
for i, (pred, act) in enumerate(zip(mean_pred, fraction_pos), 1):
    gap = pred - act
    gap_list.append(abs(gap))
    flag = " <-- over-predicts" if gap > 0.05 else (" <-- under-predicts" if gap < -0.05 else "")
    print(f"  {i:<5} {pred:.4f}            {act:.4f}         {gap:+.4f}    {flag}")
print(f"\n  Mean absolute calibration error: {np.mean(gap_list):.4f}")
print(f"  Max calibration error:           {np.max(gap_list):.4f}")

# ── ROC curve ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, probs)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#378ADD', linewidth=2,
        label=f'CatBoost (AUC = {auc:.4f})')
ax.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.500)')
ax.fill_between(fpr, tpr, alpha=0.08, color='#378ADD')
ax.set_xlabel("False positive rate", fontsize=11)
ax.set_ylabel("True positive rate", fontsize=11)
ax.set_title("ROC Curve — 30-Day Readmission Model", fontsize=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
out_roc = os.path.join(PROJECT_DIR, "roc_curve.png")
plt.savefig(out_roc, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: roc_curve.png")

# ── summary for MODEL_CARD ─────────────────────────────────
print("\n" + "=" * 60)
print("COPY THESE INTO MODEL_CARD.md (VERIFIED NUMBERS)")
print("=" * 60)
print(f"  AUC-ROC:             {auc:.4f}")
print(f"  Gini coefficient:    {gini:.4f}")
print(f"  Average Precision:   {ap:.4f}")
print(f"  Brier Score:         {brier:.4f}")
print(f"  Brier Skill Score:   {brier_skill:.4f}")
print(f"  Null Brier (ref):    {brier_null:.4f}")
print(f"\nCalibration note:")
print(f"  Mean absolute calibration error: {np.mean(gap_list):.4f}")
if np.mean(gap_list) < 0.05:
    print("  Assessment: Well calibrated")
elif np.mean(gap_list) < 0.10:
    print("  Assessment: Moderately calibrated — acceptable for ranking purposes")
else:
    print("  Assessment: Poorly calibrated — probabilities should not be taken literally")
print(f"\nStep 2 complete. Run next: py 08_fairness_audit.py")