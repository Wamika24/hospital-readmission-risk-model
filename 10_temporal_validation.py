# 10_temporal_validation.py
# Tests whether model generalises across time using encounter_id as proxy.
# Lower encounter_ids = earlier encounters = training era.
# Higher encounter_ids = later encounters = deployment era.
# Run with: py 10_temporal_validation.py

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
print("TEMPORAL VALIDATION")
print("Simulates train-on-past, test-on-future deployment")
print("=" * 60)

PROJECT_DIR = r"C:\Users\LENOVO\Desktop\readmission_ai_free"
MODEL_PATH  = os.path.join(PROJECT_DIR, "catboost_readmission_model.cbm")
XTEST_PATH  = os.path.join(PROJECT_DIR, "X_test.csv")
YTEST_PATH  = os.path.join(PROJECT_DIR, "y_test.csv")
CAL_PATH    = os.path.join(PROJECT_DIR, "isotonic_calibrator.pkl")

for label, path in [("Model", MODEL_PATH), ("X_test", XTEST_PATH),
                     ("y_test", YTEST_PATH), ("Calibrator", CAL_PATH)]:
    if not os.path.exists(path):
        print(f"MISSING: {label} -> {path}")
        sys.exit(1)

model = CatBoostClassifier()
model.load_model(MODEL_PATH)
X_test = pd.read_csv(XTEST_PATH)
y_test = pd.read_csv(YTEST_PATH).squeeze()
if not isinstance(y_test, pd.Series):
    y_test = y_test.iloc[:, 0]
y_test = y_test.astype(int)

with open(CAL_PATH, 'rb') as f:
    calibrator = pickle.load(f)

cat_names = [model.feature_names_[i] for i in model.get_cat_feature_indices()]
for col in cat_names:
    if col in X_test.columns:
        X_test[col] = X_test[col].fillna("None").astype(str)

raw_probs = model.predict_proba(X_test)[:, 1]
cal_probs = calibrator.predict(raw_probs)

print(f"Loaded: {len(X_test):,} patients  |  base rate: {y_test.mean():.2%}")

# ── temporal split strategy ────────────────────────────────
# The UCI dataset has encounter IDs that are roughly chronological.
# We split the test set into thirds by row order as a temporal proxy.
# Early rows = earlier encounters. Late rows = later encounters.
# This simulates whether model performance degrades on more recent data.

n = len(X_test)
split1 = n // 3
split2 = 2 * n // 3

slices = {
    "Early (first third)":  (0,       split1),
    "Middle (second third)": (split1,  split2),
    "Late (final third)":   (split2,  n),
}

print("\nTemporal split (row order as encounter time proxy):")
print(f"  {'Period':<25} {'n':>7} {'Readmissions':>14} {'Base rate':>10}")
print(f"  {'-'*58}")
for label, (s, e) in slices.items():
    y_sl = y_test.iloc[s:e]
    print(f"  {label:<25} {len(y_sl):>7,} {int(y_sl.sum()):>14,} {y_sl.mean():>10.2%}")

# ── metrics per slice ──────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL PERFORMANCE ACROSS TIME PERIODS")
print("=" * 60)
print(f"{'Period':<25} {'AUC':>8} {'Avg Prec':>10} {'P@10%':>8} {'Base rate':>10}")
print("-" * 65)

results = []
for label, (s, e) in slices.items():
    y_sl   = y_test.iloc[s:e]
    p_sl   = cal_probs[s:e]

    if y_sl.sum() < 10:
        print(f"  {label:<25} -- skipped (fewer than 10 positive cases)")
        continue

    auc    = roc_auc_score(y_sl, p_sl)
    ap     = average_precision_score(y_sl, p_sl)
    k      = max(1, int(len(y_sl) * 0.10))
    top_k  = np.argsort(p_sl)[::-1][:k]
    p10    = y_sl.values[top_k].mean()

    results.append({'period': label, 'auc': auc, 'ap': ap,
                    'p10': p10, 'base_rate': y_sl.mean(), 'n': len(y_sl)})
    print(f"  {label:<25} {auc:>8.4f} {ap:>10.4f} {p10:>8.2%} {y_sl.mean():>10.2%}")

results_df = pd.DataFrame(results)

# ── stability assessment ───────────────────────────────────
print(f"\n{'=' * 60}")
print("TEMPORAL STABILITY ASSESSMENT")
print("=" * 60)

if len(results_df) >= 2:
    auc_range = results_df['auc'].max() - results_df['auc'].min()
    ap_range  = results_df['ap'].max()  - results_df['ap'].min()
    p10_range = results_df['p10'].max() - results_df['p10'].min()

    print(f"\n  AUC range across periods:           {auc_range:.4f}")
    print(f"  Average Precision range:            {ap_range:.4f}")
    print(f"  Precision@10% range:                {p10_range:.2%}")

    if auc_range < 0.03:
        auc_status = "STABLE — AUC varies less than 3pp across time"
    elif auc_range < 0.07:
        auc_status = "MODERATE — AUC varies 3–7pp, acceptable for research prototype"
    else:
        auc_status = "UNSTABLE — AUC varies more than 7pp, investigate distribution shift"

    print(f"\n  Stability verdict: {auc_status}")

    early  = results_df.iloc[0]
    late   = results_df.iloc[-1]
    drift  = late['auc'] - early['auc']
    print(f"\n  AUC drift (early → late): {drift:+.4f}")
    if drift < -0.05:
        print(f"  WARNING: Performance degrades on more recent data.")
        print(f"  This suggests distribution shift — model may need retraining")
        print(f"  on more recent data before production deployment.")
    elif drift > 0.05:
        print(f"  NOTE: Performance improves on more recent data.")
        print(f"  Possible explanation: recent data has cleaner coding patterns.")
    else:
        print(f"  No significant drift detected. Model is temporally stable.")

# ── chart ──────────────────────────────────────────────────
if len(results_df) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x = range(len(results_df))
    ax.plot(x, results_df['auc'],   'o-', color='#378ADD',
            linewidth=2, markersize=8, label='AUC-ROC')
    ax.plot(x, results_df['base_rate'], 's--', color='#888780',
            linewidth=1, markersize=6, label='Base rate (readmission %)')
    ax.set_xticks(x)
    ax.set_xticklabels([r['period'].split('(')[0].strip()
                        for r in results_df.to_dict('records')],
                       fontsize=9)
    ax.set_ylabel("AUC-ROC", fontsize=10)
    ax.set_title("AUC stability across time periods\n(flat line = stable model)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0.45, 0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax2 = axes[1]
    ax2.plot(x, results_df['p10'],  'o-', color='#1D9E75',
             linewidth=2, markersize=8, label='Precision@10%')
    ax2.axhline(results_df['base_rate'].mean(), color='#888780',
                linestyle='--', linewidth=1, label='Mean base rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r['period'].split('(')[0].strip()
                         for r in results_df.to_dict('records')],
                        fontsize=9)
    ax2.set_ylabel("Precision at top 10%", fontsize=10)
    ax2.set_title("Top-decile precision across time periods\n(flat = consistent ranking quality)",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle(
        "Temporal Validation — 30-Day Readmission Model\n"
        "Row order used as temporal proxy (UCI dataset lacks explicit timestamps)",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out = os.path.join(PROJECT_DIR, "temporal_validation.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: temporal_validation.png")

# ── save results ───────────────────────────────────────────
results_df.to_csv(os.path.join(PROJECT_DIR, "temporal_validation_results.csv"), index=False)
print(f"Saved: temporal_validation_results.csv")

# ── model card text ────────────────────────────────────────
print("\n" + "=" * 60)
print("COPY INTO MODEL_CARD.md — TEMPORAL VALIDATION")
print("=" * 60)
print("""
## Temporal Validation

**Method:** Test set split into three chronological thirds by row order.
Row order serves as a temporal proxy (UCI dataset lacks explicit timestamps).
Performance evaluated on each third independently.

**Limitation note:** True temporal validation requires splitting the full
training dataset by year (train on 1999–2006, test on 2007–2008). This
analysis uses the test set only as an approximation. A full temporal
retraining study is recommended before production deployment.

**Stability threshold:** AUC variance < 3pp across periods = stable.
""")
if len(results_df) >= 2:
    for _, row in results_df.iterrows():
        print(f"  {row['period']:<28} AUC: {row['auc']:.4f}  P@10%: {row['p10']:.2%}")
    print(f"\n  AUC range: {results_df['auc'].max()-results_df['auc'].min():.4f}  |  "
          f"Verdict: {auc_status}")

print(f"\nStep 4 complete. Run next: py 09_intervention_simulation.py")