# 09_intervention_simulation.py
# Simulates the operational impact of deploying the risk model.
# Answers: "If we outreach the top K patients, how many readmissions
# do we prevent, and what does that cost vs save?"
# Uses saved model + calibrated probabilities. No retraining.
# Run with: py 09_intervention_simulation.py

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

print("=" * 60)
print("INTERVENTION SIMULATION")
print("Operational impact of risk-stratified outreach")
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

n_total   = len(y_test)
n_pos     = int(y_test.sum())
base_rate = y_test.mean()

print(f"\nTest set: {n_total:,} patients  |  "
      f"Readmissions: {n_pos:,}  |  Base rate: {base_rate:.2%}")

# Sort patients by risk score descending (highest risk first)
sorted_idx = np.argsort(cal_probs)[::-1]
y_sorted   = y_test.values[sorted_idx]
p_sorted   = cal_probs[sorted_idx]

# ── assumptions ────────────────────────────────────────────
# Sourced from AHRQ and Coleman Care Transitions Program
# See ASSUMPTIONS_REGISTER.md for full citations

COST_PER_READMISSION  = 15_000   # USD — AHRQ benchmark
OUTREACH_COSTS        = {        # USD per patient outreached
    'Conservative ($200)': 200,
    'Base ($350)':         350,
    'Optimistic ($500)':   500,
}
EFFECTIVENESS_RATES   = {        # proportion of flagged true positives averted
    'Conservative (15%)': 0.15,
    'Base (20%)':         0.20,
    'Optimistic (30%)':   0.30,
}

# Monthly multiplier: test set = 14,303 patients
# A typical 300-bed hospital sees ~2,000 discharges/month
# We scale to per-100-outreached for generalisability
MONTHS = 12

# ── precision at K curve ───────────────────────────────────
print("\n" + "=" * 60)
print("PRECISION AT K — HOW ACCURATE IS THE OUTREACH LIST?")
print("=" * 60)
print(f"\n  {'K (outreached)':<18} {'True positives':>16} "
      f"{'Precision':>12} {'Lift':>8} {'Cumulative %':>14}")
print(f"  {'-'*72}")

k_values  = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000,
             1430, 2000, 3000, 4000, 5000]
prec_rows = []

for k in k_values:
    k = min(k, n_total)
    tp        = int(y_sorted[:k].sum())
    precision = tp / k
    lift      = precision / base_rate
    pct_caught= tp / n_pos * 100
    prec_rows.append({'k': k, 'tp': tp, 'precision': precision,
                      'lift': lift, 'pct_caught': pct_caught})
    print(f"  {k:<18,} {tp:>16,} {precision:>12.1%} "
          f"{lift:>8.2f}x {pct_caught:>13.1f}%")

prec_df = pd.DataFrame(prec_rows)

# ── base ROI: random vs model outreach ────────────────────
print("\n" + "=" * 60)
print("RANDOM OUTREACH vs MODEL-GUIDED OUTREACH")
print("Scenario: outreach 250 patients/month × 12 months = 3,000/year")
print("=" * 60)

K_MONTHLY   = 250
K_ANNUAL    = K_MONTHLY * MONTHS
k_row       = prec_df[prec_df['k'] == 250].iloc[0]
model_prec  = k_row['precision']
random_prec = base_rate

model_tp_annual  = int(round(K_ANNUAL * model_prec))
random_tp_annual = int(round(K_ANNUAL * random_prec))

print(f"\n  Annual outreach volume:      {K_ANNUAL:,} patients")
print(f"  Model precision at K=250:    {model_prec:.1%}")
print(f"  Random precision (baseline): {random_prec:.1%} (= base rate)")
print(f"\n  True positives found per year:")
print(f"    With model:   {model_tp_annual:,} patients")
print(f"    Without model:{random_tp_annual:,} patients")
print(f"    Advantage:    {model_tp_annual - random_tp_annual:,} additional true positives")
print(f"    Lift:         {model_prec/random_prec:.2f}x")

# ── full scenario matrix ───────────────────────────────────
print("\n" + "=" * 60)
print("ROI SCENARIO MATRIX  (annual, 12 months)")
print("=" * 60)

scenarios = [
    ('Conservative', 100,  0.15, 200),
    ('Base',         250,  0.20, 350),
    ('Optimistic',   500,  0.30, 500),
]

scenario_results = []
print(f"\n  {'Scenario':<14} {'K/mo':>6} {'Eff':>5} {'$/pt':>6} "
      f"{'TP/yr':>7} {'Avoided':>8} {'Gross $':>12} "
      f"{'Outreach $':>12} {'Net $':>12} {'ROI':>8}")
print(f"  {'-'*100}")

for name, k_mo, eff, outreach_cost in scenarios:
    k_ann     = k_mo * MONTHS
    # precision at this K — use nearest in prec_df
    nearest_k = prec_df.iloc[(prec_df['k'] - k_mo).abs().argsort()[:1]]['k'].values[0]
    prec_k    = prec_df[prec_df['k'] == nearest_k]['precision'].values[0]

    tp_ann    = k_ann * prec_k
    avoided   = tp_ann * eff
    gross     = avoided * COST_PER_READMISSION
    cost      = k_ann  * outreach_cost
    net       = gross  - cost
    roi       = (net / cost * 100) if cost > 0 else 0

    scenario_results.append({
        'scenario': name, 'k_monthly': k_mo, 'k_annual': k_ann,
        'precision': prec_k, 'tp_annual': tp_ann, 'avoided': avoided,
        'gross_savings': gross, 'outreach_cost': cost,
        'net_savings': net, 'roi_pct': roi
    })

    print(f"  {name:<14} {k_mo:>6,} {eff:>5.0%} {outreach_cost:>6,} "
          f"{tp_ann:>7.0f} {avoided:>8.0f} "
          f"${gross:>10,.0f} ${cost:>10,.0f} "
          f"${net:>10,.0f} {roi:>7.0f}%")

scen_df = pd.DataFrame(scenario_results)

# ── break-even analysis ────────────────────────────────────
print("\n" + "=" * 60)
print("BREAK-EVEN ANALYSIS")
print("=" * 60)
for _, row in scen_df.iterrows():
    be = row['outreach_cost'] / COST_PER_READMISSION
    print(f"  {row['scenario']:<14}: need {be:.1f} avoided readmissions "
          f"to break even  (you achieve {row['avoided']:.0f})")

# ── value of the model vs random ──────────────────────────
print("\n" + "=" * 60)
print("VALUE OF THE MODEL vs RANDOM OUTREACH (Base scenario)")
print("=" * 60)
base_row    = scen_df[scen_df['scenario'] == 'Base'].iloc[0]
random_tp   = base_row['k_annual'] * base_rate
random_avoid= random_tp * 0.20
random_gross= random_avoid * COST_PER_READMISSION
random_net  = random_gross - base_row['outreach_cost']

print(f"\n  Annual outreach: {base_row['k_annual']:,} patients, "
      f"outreach cost: ${base_row['outreach_cost']:,.0f}")
print(f"\n                       {'Model-guided':>16} {'Random':>14} {'Advantage':>14}")
print(f"  {'True positives':<22} {base_row['tp_annual']:>16.0f} "
      f"{random_tp:>14.0f} {base_row['tp_annual']-random_tp:>+14.0f}")
print(f"  {'Avoided readmissions':<22} {base_row['avoided']:>16.0f} "
      f"{random_avoid:>14.0f} {base_row['avoided']-random_avoid:>+14.0f}")
print(f"  {'Gross savings':<22} ${base_row['gross_savings']:>15,.0f} "
      f"${random_gross:>13,.0f} ${base_row['gross_savings']-random_gross:>+13,.0f}")
print(f"  {'Net savings':<22} ${base_row['net_savings']:>15,.0f} "
      f"${random_net:>13,.0f} ${base_row['net_savings']-random_net:>+13,.0f}")

# ── charts ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Chart 1: Precision at K curve
ax = axes[0]
ax.plot(prec_df['k'], prec_df['precision'] * 100,
        'o-', color='#378ADD', linewidth=2, markersize=5)
ax.axhline(base_rate * 100, color='#888780', linestyle='--',
           linewidth=1.2, label=f'Random outreach ({base_rate:.1%})')
for _, k_mo, _, _ in scenarios:
    nearest = prec_df.iloc[(prec_df['k']-k_mo).abs().argsort()[:1]]
    ax.axvline(k_mo, color='#E24B4A', linewidth=0.8, alpha=0.4)
ax.set_xlabel("Patients outreached (K)", fontsize=10)
ax.set_ylabel("Precision — % who are actual readmissions", fontsize=10)
ax.set_title("Precision at K\n(model vs random outreach)", fontsize=11)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Chart 2: Net savings by scenario
ax2 = axes[1]
colors = ['#EF9F27', '#1D9E75', '#7F77DD']
bars   = ax2.bar(scen_df['scenario'],
                 scen_df['net_savings'] / 1e6,
                 color=colors, edgecolor='none', width=0.5)
ax2.set_ylabel("Net annual savings ($M)", fontsize=10)
ax2.set_title("Net savings by scenario\n(annual, after outreach costs)", fontsize=11)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for bar, val in zip(bars, scen_df['net_savings'] / 1e6):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'${val:.2f}M', ha='center', fontsize=10, fontweight='500')

# Chart 3: Gross vs outreach cost
ax3 = axes[2]
x    = np.arange(len(scen_df))
w    = 0.35
ax3.bar(x - w/2, scen_df['gross_savings'] / 1e6,
        width=w, color='#1D9E75', edgecolor='none', label='Gross savings')
ax3.bar(x + w/2, scen_df['outreach_cost'] / 1e6,
        width=w, color='#E24B4A', edgecolor='none', label='Outreach cost')
ax3.set_xticks(x)
ax3.set_xticklabels(scen_df['scenario'], fontsize=10)
ax3.set_ylabel("USD millions", fontsize=10)
ax3.set_title("Gross savings vs outreach cost\n(net = green minus red)",
              fontsize=11)
ax3.legend(fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.suptitle(
    "Intervention Simulation — 30-Day Readmission Risk Model\n"
    f"Base rate: {base_rate:.2%}  |  "
    f"Cost per readmission: ${COST_PER_READMISSION:,}  |  "
    f"Source: AHRQ benchmark",
    fontsize=10, y=1.01
)
plt.tight_layout()
out = os.path.join(PROJECT_DIR, "intervention_simulation.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: intervention_simulation.png")

# ── save tables ────────────────────────────────────────────
prec_df.to_csv(os.path.join(PROJECT_DIR, "precision_at_k_curve.csv"),  index=False)
scen_df.to_csv(os.path.join(PROJECT_DIR, "intervention_scenarios.csv"), index=False)
print(f"Saved: precision_at_k_curve.csv")
print(f"Saved: intervention_scenarios.csv")

# ── final summary ──────────────────────────────────────────
print("\n" + "=" * 60)
print("COPY INTO MODEL_CARD.md — INTERVENTION SIMULATION")
print("=" * 60)
base = scen_df[scen_df['scenario']=='Base'].iloc[0]
cons = scen_df[scen_df['scenario']=='Conservative'].iloc[0]
opti = scen_df[scen_df['scenario']=='Optimistic'].iloc[0]

print(f"""
## Intervention Simulation

Assumptions: $15,000/readmission (AHRQ), effectiveness 15–30%
(Coleman Care Transitions Program), outreach $200–500/patient.

| Scenario     | Monthly outreach | Avoided/year | Net savings/year | ROI  |
|---|---|---|---|---|
| Conservative | 100 patients     | {cons['avoided']:.0f}           | ${cons['net_savings']/1e6:.2f}M            | {cons['roi_pct']:.0f}%  |
| Base         | 250 patients     | {base['avoided']:.0f}           | ${base['net_savings']/1e6:.2f}M            | {base['roi_pct']:.0f}%  |
| Optimistic   | 500 patients     | {opti['avoided']:.0f}           | ${opti['net_savings']/1e6:.2f}M            | {opti['roi_pct']:.0f}%  |

Model-guided vs random outreach (Base scenario):
  Additional true positives identified/year: +{base['tp_annual']-random_tp:.0f}
  Additional net savings vs random:          +${base['net_savings']-random_net:,.0f}
""")
print(f"Step 5 complete. Run next: py 13_business_case_sensitivity.py")