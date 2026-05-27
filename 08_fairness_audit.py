# 08_fairness_audit.py
# Evaluates whether model performance differs across demographic groups.
# Uses saved model + calibrated probabilities.
# Run with: py 08_fairness_audit.py

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

print("=" * 60)
print("FAIRNESS AUDIT")
print("30-Day Readmission Risk Model")
print("=" * 60)

PROJECT_DIR = r"C:\Users\LENOVO\Desktop\readmission_ai_free"
MODEL_PATH  = os.path.join(PROJECT_DIR, "catboost_readmission_model.cbm")
XTEST_PATH  = os.path.join(PROJECT_DIR, "X_test.csv")
YTEST_PATH  = os.path.join(PROJECT_DIR, "y_test.csv")
CAL_PATH    = os.path.join(PROJECT_DIR, "isotonic_calibrator.pkl")

# ── verify files ──────────────────────────────────────────
print("\nChecking files...")
for label, path in [
    ("Model",       MODEL_PATH),
    ("X_test",      XTEST_PATH),
    ("y_test",      YTEST_PATH),
    ("Calibrator",  CAL_PATH),
]:
    if not os.path.exists(path):
        print(f"  MISSING: {label} -> {path}")
        sys.exit(1)
    print(f"  FOUND: {label}")

# ── load ──────────────────────────────────────────────────
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

X_test = pd.read_csv(XTEST_PATH)
y_test = pd.read_csv(YTEST_PATH).squeeze()
if not isinstance(y_test, pd.Series):
    y_test = y_test.iloc[:, 0]
y_test = y_test.astype(int)

with open(CAL_PATH, 'rb') as f:
    calibrator = pickle.load(f)

# fix categoricals
cat_names = [model.feature_names_[i] for i in model.get_cat_feature_indices()]
for col in cat_names:
    if col in X_test.columns:
        X_test[col] = X_test[col].fillna("None").astype(str)

raw_probs = model.predict_proba(X_test)[:, 1]
cal_probs = calibrator.predict(raw_probs)

print(f"\nLoaded: {len(X_test):,} patients  |  base rate: {y_test.mean():.2%}")
print(f"Calibrated score range: {cal_probs.min():.4f} – {cal_probs.max():.4f}")

# ── fairness metric function ───────────────────────────────
def fairness_metrics(y_true, y_score, group_name, threshold=0.1479):
    """
    threshold = High Risk cutoff (top-10% calibrated score).
    Computes AUC, Precision@K, FNR, FPR for one demographic group.
    """
    n = len(y_true)
    if n < 30:
        return None

    # AUC (needs both classes present)
    if y_true.sum() == 0 or y_true.sum() == n:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_score)

    # Precision@10% (top decile within this group)
    k = max(1, int(n * 0.10))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    prec_at_10pct = y_true.values[top_k_idx].mean()

    # At high-risk threshold
    flagged      = y_score >= threshold
    n_flagged    = flagged.sum()
    tp           = (flagged & (y_true == 1)).sum()
    fp           = (flagged & (y_true == 0)).sum()
    fn           = (~flagged & (y_true == 1)).sum()
    tn           = (~flagged & (y_true == 0)).sum()

    tpr          = tp / y_true.sum()    if y_true.sum() > 0  else np.nan  # sensitivity
    fnr          = fn / y_true.sum()    if y_true.sum() > 0  else np.nan  # miss rate
    fpr          = fp / (y_true==0).sum() if (y_true==0).sum() > 0 else np.nan
    flag_rate    = flagged.mean()

    return {
        'n':            n,
        'n_positive':   int(y_true.sum()),
        'base_rate':    y_true.mean(),
        'auc':          auc,
        'prec_top10pct': prec_at_10pct,
        'flag_rate':    flag_rate,
        'tpr':          tpr,   # true positive rate (higher = better)
        'fnr':          fnr,   # false negative rate (LOWER = better, missing fewer sick patients)
        'fpr':          fpr,   # false positive rate (lower = fewer unnecessary contacts)
    }

# ── audit by RACE ─────────────────────────────────────────
print("\n" + "=" * 60)
print("FAIRNESS AUDIT — BY RACE")
print("=" * 60)
print("Key metric: FNR (False Negative Rate) — lower = fewer missed readmissions")
print("Disparity threshold: flag if FNR differs by > 5 percentage points from overall\n")

overall_fnr = fairness_metrics(y_test, cal_probs, "Overall")['fnr']
overall_auc = fairness_metrics(y_test, cal_probs, "Overall")['auc']

race_results = []
for race_val in sorted(X_test['race'].unique()):
    mask   = X_test['race'] == race_val
    result = fairness_metrics(y_test[mask], cal_probs[mask], race_val)
    if result:
        result['group'] = str(race_val)
        race_results.append(result)

race_df = pd.DataFrame(race_results).set_index('group')

print(f"{'Race':<22} {'n':>7} {'Base rate':>10} {'AUC':>8} "
      f"{'FNR':>8} {'FPR':>8} {'Flag rate':>10} {'FNR vs overall':>14}")
print("-" * 90)
for group, row in race_df.iterrows():
    fnr_diff = row['fnr'] - overall_fnr
    flag = " <-- DISPARITY" if abs(fnr_diff) > 0.05 else ""
    print(f"{group:<22} {row['n']:>7,} {row['base_rate']:>10.2%} "
          f"{row['auc']:>8.4f} {row['fnr']:>8.2%} {row['fpr']:>8.2%} "
          f"{row['flag_rate']:>10.2%} {fnr_diff:>+13.2%}{flag}")
print(f"\n{'Overall':<22} {len(y_test):>7,} {y_test.mean():>10.2%} "
      f"{overall_auc:>8.4f} {overall_fnr:>8.2%}")

# ── audit by GENDER ────────────────────────────────────────
print("\n" + "=" * 60)
print("FAIRNESS AUDIT — BY GENDER")
print("=" * 60)

gender_results = []
for g in sorted(X_test['gender'].unique()):
    mask   = X_test['gender'] == g
    result = fairness_metrics(y_test[mask], cal_probs[mask], g)
    if result:
        result['group'] = str(g)
        gender_results.append(result)

gender_df = pd.DataFrame(gender_results).set_index('group')
print(f"{'Gender':<22} {'n':>7} {'Base rate':>10} {'AUC':>8} "
      f"{'FNR':>8} {'FPR':>8} {'Flag rate':>10}")
print("-" * 75)
for group, row in gender_df.iterrows():
    print(f"{group:<22} {row['n']:>7,} {row['base_rate']:>10.2%} "
          f"{row['auc']:>8.4f} {row['fnr']:>8.2%} {row['fpr']:>8.2%} "
          f"{row['flag_rate']:>10.2%}")

# ── audit by AGE ───────────────────────────────────────────
print("\n" + "=" * 60)
print("FAIRNESS AUDIT — BY AGE GROUP")
print("=" * 60)

age_results = []
for age_val in sorted(X_test['age'].unique()):
    mask   = X_test['age'] == age_val
    result = fairness_metrics(y_test[mask], cal_probs[mask], age_val)
    if result:
        result['group'] = str(age_val)
        age_results.append(result)

age_df = pd.DataFrame(age_results).set_index('group')
print(f"{'Age group':<22} {'n':>7} {'Base rate':>10} {'AUC':>8} "
      f"{'FNR':>8} {'FPR':>8} {'Flag rate':>10}")
print("-" * 75)
for group, row in age_df.iterrows():
    print(f"{group:<22} {row['n']:>7,} {row['base_rate']:>10.2%} "
          f"{row['auc']:>8.4f} {row['fnr']:>8.2%} {row['fpr']:>8.2%} "
          f"{row['flag_rate']:>10.2%}")

# ── disparity summary ──────────────────────────────────────
print("\n" + "=" * 60)
print("DISPARITY SUMMARY")
print("=" * 60)

def disparity_check(df, dimension):
    if df.empty or 'fnr' not in df.columns:
        return
    valid = df.dropna(subset=['fnr','auc'])
    if len(valid) < 2:
        return
    fnr_range = valid['fnr'].max() - valid['fnr'].min()
    auc_range = valid['auc'].max() - valid['auc'].min()
    worst_fnr = valid['fnr'].idxmax()
    best_fnr  = valid['fnr'].idxmin()
    print(f"\n  {dimension}:")
    print(f"    FNR range:  {fnr_range:.2%}  "
          f"(highest: {worst_fnr} at {valid.loc[worst_fnr,'fnr']:.2%}, "
          f"lowest: {best_fnr} at {valid.loc[best_fnr,'fnr']:.2%})")
    print(f"    AUC range:  {auc_range:.4f}")
    if fnr_range > 0.05:
        print(f"    STATUS: DISPARITY DETECTED — FNR spread exceeds 5pp threshold")
        print(f"    ACTION: Document in FAIRNESS_REMEDIATION.md. Consider re-weighting.")
    elif fnr_range > 0.03:
        print(f"    STATUS: MONITOR — FNR spread between 3–5pp. Watch in production.")
    else:
        print(f"    STATUS: ACCEPTABLE — FNR spread below 3pp threshold.")

disparity_check(race_df,   "Race")
disparity_check(gender_df, "Gender")
disparity_check(age_df,    "Age group")

# ── save results ───────────────────────────────────────────
all_results = []
for dim, df in [("Race", race_df), ("Gender", gender_df), ("Age", age_df)]:
    tmp = df.copy()
    tmp['dimension'] = dim
    all_results.append(tmp)

fairness_df = pd.concat(all_results).reset_index()
fairness_df.columns = [str(c) for c in fairness_df.columns]
out_csv = os.path.join(PROJECT_DIR, "fairness_audit_results.csv")
fairness_df.to_csv(out_csv, index=False)
print(f"\nSaved: fairness_audit_results.csv")

# ── chart: FNR by race ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (df, title, dim) in zip(axes, [
    (race_df,   "FNR by race\n(missed readmissions)", "Race"),
    (gender_df, "FNR by gender\n(missed readmissions)", "Gender"),
    (age_df,    "FNR by age group\n(missed readmissions)", "Age"),
]):
    valid = df.dropna(subset=['fnr'])
    colors = ['#E24B4A' if v > overall_fnr + 0.05 else
              '#EF9F27' if v > overall_fnr + 0.02 else
              '#1D9E75' for v in valid['fnr']]
    bars = ax.barh(valid.index, valid['fnr'], color=colors,
                   edgecolor='none', height=0.55)
    ax.axvline(overall_fnr, color='black', linewidth=1.2,
               linestyle='--', label=f'Overall: {overall_fnr:.1%}')
    ax.set_xlabel("False negative rate", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, valid['fnr']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=8)

plt.suptitle(
    "Fairness Audit — False Negative Rate by Demographic Group\n"
    "Red = disparity detected (>5pp above overall)  |  "
    "Orange = monitor (2–5pp)  |  Green = acceptable",
    fontsize=10
)
plt.tight_layout()
out_png = os.path.join(PROJECT_DIR, "fairness_audit.png")
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fairness_audit.png")

# ── final guidance ─────────────────────────────────────────
print("\n" + "=" * 60)
print("FAIRNESS AUDIT COMPLETE")
print("=" * 60)
print("""
What to add to FAIRNESS_REMEDIATION.md:
  - Copy the DISPARITY SUMMARY section above
  - For any group with STATUS: DISPARITY DETECTED:
      Proposed remediation: Apply group-specific probability
      thresholds so each group achieves equal FNR. This ensures
      the model does not systematically miss readmissions in any
      demographic group.
  - If no disparities found: document that analysis was conducted
      and passed, with the FNR ranges as evidence.

What to add to MODEL_CARD.md:
  - Reference fairness_audit_results.csv
  - State the disparity thresholds used (5pp FNR)
  - State which groups were tested: race, gender, age

Step 3 complete. Run next: py 10_temporal_validation.py
""")