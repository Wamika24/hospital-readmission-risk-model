import psycopg2
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score


DB_CONFIG = {
    "host": "localhost",
    "port": "5433",
    "database": "readmission_db",
    "user": "postgres",
    "password": "1234"
}


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
    except ValueError:
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


def precision_at_n(y_true, probs, n):
    data = pd.DataFrame({
        "actual": y_true,
        "probability": probs
    }).sort_values("probability", ascending=False)

    top_n = data.head(n)
    return top_n["actual"].mean()


def bootstrap_ci(y_true, probs, metric_name, n_bootstrap=500, random_state=42):
    rng = np.random.default_rng(random_state)
    scores = []

    y_true = np.array(y_true)
    probs = np.array(probs)

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)

        y_sample = y_true[idx]
        p_sample = probs[idx]

        if len(np.unique(y_sample)) < 2:
            continue

        if metric_name == "auc":
            score = roc_auc_score(y_sample, p_sample)

        elif metric_name == "ap":
            score = average_precision_score(y_sample, p_sample)

        elif metric_name == "p100":
            score = precision_at_n(y_sample, p_sample, 100)

        elif metric_name == "p250":
            score = precision_at_n(y_sample, p_sample, 250)

        elif metric_name == "p500":
            score = precision_at_n(y_sample, p_sample, 500)

        else:
            raise ValueError("Unknown metric")

        scores.append(score)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    mean_score = np.mean(scores)

    return mean_score, lower, upper


# =========================
# LOAD DATA
# =========================

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

print("Rows loaded:", len(df))

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

cat_idx = [feature_cols.index(c) for c in cat_cols]

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool = Pool(X_test, y_test, cat_features=cat_idx)

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

print("Model trained successfully")

probs = model.predict_proba(test_pool)[:, 1]
y_test_arr = y_test.values

# =========================
# BASE METRICS
# =========================

auc = roc_auc_score(y_test_arr, probs)
ap = average_precision_score(y_test_arr, probs)
p100 = precision_at_n(y_test_arr, probs, 100)
p250 = precision_at_n(y_test_arr, probs, 250)
p500 = precision_at_n(y_test_arr, probs, 500)

print("\n===== BASE TEST METRICS =====")
print(f"AUC  : {auc:.4f}")
print(f"AP   : {ap:.4f}")
print(f"P@100: {p100:.4f}")
print(f"P@250: {p250:.4f}")
print(f"P@500: {p500:.4f}")

# =========================
# BOOTSTRAP CI
# =========================

metrics = [
    ("AUC", "auc"),
    ("Average Precision", "ap"),
    ("Precision@100", "p100"),
    ("Precision@250", "p250"),
    ("Precision@500", "p500")
]

rows = []

print("\n===== BOOTSTRAP 95% CONFIDENCE INTERVALS =====")

for display_name, metric_key in metrics:
    mean_score, lower, upper = bootstrap_ci(
        y_test_arr,
        probs,
        metric_key,
        n_bootstrap=500
    )

    print(f"{display_name}: {mean_score:.4f} | 95% CI [{lower:.4f}, {upper:.4f}]")

    rows.append((
        display_name,
        float(mean_score),
        float(lower),
        float(upper)
    ))

# =========================
# SAVE TO POSTGRESQL
# =========================

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS bootstrap_confidence_intervals;")

cur.execute("""
CREATE TABLE bootstrap_confidence_intervals (
    metric VARCHAR(100),
    mean_score FLOAT,
    ci_lower FLOAT,
    ci_upper FLOAT
);
""")

conn.commit()

cur.executemany("""
INSERT INTO bootstrap_confidence_intervals
VALUES (%s, %s, %s, %s)
""", rows)

conn.commit()

cur.close()
conn.close()

print("\nSaved table: bootstrap_confidence_intervals")
print("\nBootstrap confidence interval analysis complete")