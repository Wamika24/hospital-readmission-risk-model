import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    auc
)

# ======================================================
# DATABASE CONFIG
# ======================================================

DB_CONFIG = {
    "host": "localhost",
    "port": "5433",
    "database": "readmission_db",
    "user": "postgres",
    "password": "1234"
}

# ======================================================
# DIAG GROUPING
# ======================================================

def bucket_diag(code):

    s = str(code).strip()

    if s.lower() in {"nan", "none", "", "?"}:
        return "Unknown"

    if s.startswith("250"):
        return "Diabetes"

    if s.startswith("V") or s.startswith("E"):
        return "External/Other"

    try:
        val = float(s)
    except:
        return "Other"

    if (390 <= val < 460) or val == 785:
        return "Circulatory"

    elif (460 <= val < 520) or val == 786:
        return "Respiratory"

    elif (520 <= val < 580) or val == 787:
        return "Digestive"

    elif (580 <= val < 630) or val == 788:
        return "Genitourinary"

    elif 710 <= val < 740:
        return "Musculoskeletal"

    elif 140 <= val < 240:
        return "Neoplasms"

    elif 800 <= val < 1000:
        return "Injury"

    else:
        return "Other"

# ======================================================
# LOAD DATA
# ======================================================

conn = psycopg2.connect(**DB_CONFIG)

query = """
SELECT
    d.patient_id,
    d.days_in_hospital,
    d.lab_procedures,
    d.procedures,
    d.medications,
    d.outpatient_visits,
    d.emergency_visits,
    d.inpatient_visits,
    d.num_diagnoses,
    d.age,
    d.gender,
    d.race,
    d.admission_type,
    d.discharge_type,
    d.admission_source,
    d.diabetesmed,
    d.medical_specialty,
    d.diag_1,
    d.a1cresult,
    d.insulin,
    p.readmit_30day_flag
FROM diabetes_clean d
JOIN patient_risk_analysis p
    ON d.patient_id = p.patient_id
WHERE p.readmit_30day_flag IS NOT NULL
"""

df = pd.read_sql(query, conn)

conn.close()

print(f"\nRows loaded: {len(df)}")

# ======================================================
# FEATURE ENGINEERING
# ======================================================

df["diag_group"] = df["diag_1"].apply(bucket_diag)

feature_cols = [
    "days_in_hospital",
    "lab_procedures",
    "procedures",
    "medications",
    "outpatient_visits",
    "emergency_visits",
    "inpatient_visits",
    "num_diagnoses",
    "age",
    "gender",
    "race",
    "admission_type",
    "discharge_type",
    "admission_source",
    "diabetesmed",
    "medical_specialty",
    "a1cresult",
    "insulin",
    "diag_group"
]

cat_cols = [
    "age",
    "gender",
    "race",
    "admission_type",
    "discharge_type",
    "admission_source",
    "diabetesmed",
    "medical_specialty",
    "a1cresult",
    "insulin",
    "diag_group"
]

for col in cat_cols:
    df[col] = df[col].fillna("Unknown").astype(str)

TARGET = "readmit_30day_flag"

X = df[feature_cols].copy()
y = df[TARGET].astype(int)

# ======================================================
# SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

cat_idx = [feature_cols.index(c) for c in cat_cols]

train_pool = Pool(
    X_train,
    y_train,
    cat_features=cat_idx
)

test_pool = Pool(
    X_test,
    y_test,
    cat_features=cat_idx
)

# ======================================================
# TRAIN MODEL
# ======================================================

model = CatBoostClassifier(
    iterations=250,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=5,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=False
)

model.fit(train_pool)

print("\nModel trained successfully")

# ======================================================
# PREDICTIONS
# ======================================================

pred_probs = model.predict_proba(test_pool)[:,1]

# ======================================================
# PR CURVE
# ======================================================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    pred_probs
)

ap_score = average_precision_score(
    y_test,
    pred_probs
)

pr_auc = auc(recall, precision)

print("\n===== PR METRICS =====")

print(f"Average Precision : {ap_score:.4f}")

print(f"PR AUC            : {pr_auc:.4f}")

# ======================================================
# PR CURVE PLOT
# ======================================================

plt.figure(figsize=(8,8))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f'PR Curve (AP={ap_score:.3f})'
)

baseline = y_test.mean()

plt.axhline(
    baseline,
    linestyle='--',
    color='red',
    label=f'Random Baseline ({baseline:.3f})'
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "precision_recall_curve.png",
    dpi=300
)

plt.close()

print("\nSaved: precision_recall_curve.png")

# ======================================================
# GAINS / LIFT ANALYSIS
# ======================================================

results_df = pd.DataFrame({
    "actual": y_test.values,
    "probability": pred_probs
})

results_df = results_df.sort_values(
    "probability",
    ascending=False
).reset_index(drop=True)

results_df["cumulative_positives"] = results_df["actual"].cumsum()

total_positives = results_df["actual"].sum()

results_df["gain"] = (
    results_df["cumulative_positives"] / total_positives
)

results_df["population_pct"] = (
    np.arange(1, len(results_df)+1) / len(results_df)
)

results_df["lift"] = (
    results_df["gain"] / results_df["population_pct"]
)

# ======================================================
# GAINS CHART
# ======================================================

plt.figure(figsize=(8,8))

plt.plot(
    results_df["population_pct"],
    results_df["gain"],
    linewidth=2,
    label="Model Gains"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--',
    label="Random Selection"
)

plt.xlabel("Population Contacted")

plt.ylabel("Cumulative Recall")

plt.title("Cumulative Gains Chart")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "cumulative_gains_chart.png",
    dpi=300
)

plt.close()

print("Saved: cumulative_gains_chart.png")

# ======================================================
# LIFT CHART
# ======================================================

plt.figure(figsize=(8,8))

plt.plot(
    results_df["population_pct"],
    results_df["lift"],
    linewidth=2,
    label="Lift"
)

plt.axhline(
    1.0,
    linestyle='--',
    color='red',
    label='Random Baseline'
)

plt.xlabel("Population Contacted")

plt.ylabel("Lift")

plt.title("Lift Chart")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "lift_chart.png",
    dpi=300
)

plt.close()

print("Saved: lift_chart.png")

# ======================================================
# TOP DECILE ANALYSIS
# ======================================================

top_10_pct = int(len(results_df) * 0.10)

top_decile = results_df.head(top_10_pct)

top_decile_precision = top_decile["actual"].mean()

top_decile_recall = (
    top_decile["actual"].sum() / total_positives
)

lift_top_decile = (
    top_decile_precision / y_test.mean()
)

print("\n===== TOP DECILE PERFORMANCE =====")

print(f"Top 10% Precision : {top_decile_precision:.4f}")

print(f"Top 10% Recall    : {top_decile_recall:.4f}")

print(f"Top 10% Lift      : {lift_top_decile:.2f}x")

# ======================================================
# SAVE TO POSTGRESQL
# ======================================================

save_df = results_df[[
    "population_pct",
    "gain",
    "lift"
]].copy()

conn = psycopg2.connect(**DB_CONFIG)

cur = conn.cursor()

cur.execute("""
DROP TABLE IF EXISTS gains_lift_results;
""")

cur.execute("""
CREATE TABLE gains_lift_results (
    population_pct FLOAT,
    gain FLOAT,
    lift FLOAT
);
""")

conn.commit()

cur.executemany(
    """
    INSERT INTO gains_lift_results
    VALUES (%s,%s,%s)
    """,
    list(save_df.itertuples(index=False, name=None))
)

conn.commit()

cur.close()

conn.close()

print("\nSaved table: gains_lift_results")

print("\nPR + Gains + Lift analysis complete")