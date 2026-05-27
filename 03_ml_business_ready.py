import psycopg2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# =========================
# DB CONNECTION
# =========================
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)

# =========================
# LOAD STRUCTURED + TEXT DATA
# =========================
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
    d.admission_type,
    d.discharge_type,
    d.admission_source,
    d.diabetesmed,
    p.risk_score,
    p.risk_segment,
    p.readmit_30day_flag,
    COALESCE(n.sentiment_score, 0) AS sentiment_score,
    COALESCE(n.keyword_hits, 0) AS keyword_hits,
    COALESCE(n.urgency_score, 0) AS urgency_score,
    COALESCE(n.sentiment_label, 'Unknown') AS sentiment_label
FROM diabetes_clean d
JOIN patient_risk_analysis p
    ON d.patient_id = p.patient_id
LEFT JOIN discharge_notes_v2 n
    ON d.patient_id = n.patient_id
WHERE p.readmit_30day_flag IS NOT NULL
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"Rows loaded: {len(df)}")

# =========================
# FEATURE ENGINEERING
# =========================
df["num_diagnoses_safe"] = df["num_diagnoses"].replace(0, 1)
df["days_safe"] = df["days_in_hospital"].replace(0, 1)

df["total_visits"] = (
    df["outpatient_visits"] +
    df["emergency_visits"] +
    df["inpatient_visits"]
)

df["visit_burden"] = (
    df["outpatient_visits"] +
    2 * df["emergency_visits"] +
    3 * df["inpatient_visits"]
)

df["care_load"] = (
    df["lab_procedures"] +
    df["procedures"] +
    df["medications"]
)

df["acute_burden"] = (
    df["emergency_visits"] +
    df["inpatient_visits"]
)

df["meds_per_day"] = df["medications"] / df["days_safe"]
df["labs_per_day"] = df["lab_procedures"] / df["days_safe"]
df["procedures_per_day"] = df["procedures"] / df["days_safe"]
df["meds_per_diagnosis"] = df["medications"] / df["num_diagnoses_safe"]

df["complexity_score"] = (
    df["num_diagnoses"] * 1.5 +
    df["medications"] * 0.5 +
    df["inpatient_visits"] * 2
)

df["long_stay_flag"] = (df["days_in_hospital"] >= 7).astype(int)
df["polypharmacy_flag"] = (df["medications"] >= 15).astype(int)
df["high_diagnosis_flag"] = (df["num_diagnoses"] >= 6).astype(int)
df["repeat_acute_flag"] = (df["acute_burden"] >= 2).astype(int)
df["high_text_risk_flag"] = ((df["keyword_hits"] >= 2) | (df["urgency_score"] >= 5)).astype(int)

# =========================
# ENCODE CATEGORICALS
# =========================
categorical_cols = [
    "age",
    "admission_type",
    "discharge_type",
    "admission_source",
    "diabetesmed",
    "risk_segment",
    "sentiment_label"
]

for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# =========================
# FEATURES / TARGET
# =========================
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
    "admission_type",
    "discharge_type",
    "admission_source",
    "diabetesmed",
    "risk_score",
    "risk_segment",
    "sentiment_score",
    "keyword_hits",
    "urgency_score",
    "sentiment_label",
    "total_visits",
    "visit_burden",
    "care_load",
    "acute_burden",
    "meds_per_day",
    "labs_per_day",
    "procedures_per_day",
    "meds_per_diagnosis",
    "complexity_score",
    "long_stay_flag",
    "polypharmacy_flag",
    "high_diagnosis_flag",
    "repeat_acute_flag",
    "high_text_risk_flag"
]

X = df[feature_cols].copy()
y = df["readmit_30day_flag"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# MODELS
# =========================
models = {
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.03,
        max_iter=400,
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=4000,
        class_weight="balanced"
    )
}

results = []
trained_models = {}

print("\n===== BUSINESS-READY MODEL COMPARISON =====")

for name, model in models.items():
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    else:
        raw = model.decision_function(X_test)
        prob = 1 / (1 + np.exp(-raw))

    pred = (prob >= 0.30).astype(int)

    auc = roc_auc_score(y_test, prob)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    brier = brier_score_loss(y_test, prob)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    results.append({
        "model": name,
        "auc": auc,
        "precision_at_030": precision,
        "recall_at_030": recall,
        "f1_at_030": f1,
        "brier": brier,
        "cv_auc_mean": cv_auc.mean(),
        "cv_auc_std": cv_auc.std()
    })

    trained_models[name] = model

    print(f"\n{name}")
    print(f"AUC ROC         : {auc:.4f}")
    print(f"Precision@0.30  : {precision:.4f}")
    print(f"Recall@0.30     : {recall:.4f}")
    print(f"F1@0.30         : {f1:.4f}")
    print(f"Brier Score     : {brier:.4f}")
    print(f"CV AUC Mean     : {cv_auc.mean():.4f}")
    print(f"CV AUC Std      : {cv_auc.std():.4f}")

results_df = pd.DataFrame(results).sort_values(
    ["auc", "cv_auc_mean", "f1_at_030"],
    ascending=False
).reset_index(drop=True)

print("\n===== FINAL MODEL RANKING =====")
print(results_df)

best_model_name = results_df.loc[0, "model"]
best_model = trained_models[best_model_name]

print(f"\nBest model selected: {best_model_name}")

# =========================
# TEST SET BUSINESS RANKING METRICS
# =========================
if hasattr(best_model, "predict_proba"):
    test_prob = best_model.predict_proba(X_test)[:, 1]
else:
    raw = best_model.decision_function(X_test)
    test_prob = 1 / (1 + np.exp(-raw))

test_ranked = pd.DataFrame({
    "y_true": y_test.values,
    "prob": test_prob
}).sort_values("prob", ascending=False).reset_index(drop=True)

print("\n===== PRECISION @ TOP N =====")
for n in [100, 250, 500, 1000]:
    if len(test_ranked) >= n:
        p_at_n = test_ranked.head(n)["y_true"].mean()
        print(f"Precision@{n}: {p_at_n:.4f}")

# =========================
# CAPACITY-BASED TARGETING
# =========================
if hasattr(best_model, "predict_proba"):
    full_prob = best_model.predict_proba(X)[:, 1]
else:
    raw = best_model.decision_function(X)
    full_prob = 1 / (1 + np.exp(-raw))

scored_df = df[[
    "patient_id",
    "risk_score",
    "risk_segment",
    "days_in_hospital",
    "medications",
    "num_diagnoses",
    "outpatient_visits",
    "emergency_visits",
    "inpatient_visits",
    "sentiment_score",
    "keyword_hits",
    "urgency_score"
]].copy()

scored_df["ml_risk_probability_v3"] = full_prob
scored_df = scored_df.sort_values("ml_risk_probability_v3", ascending=False).reset_index(drop=True)

# Business-ready tiering based on intervention capacity
scored_df["intervention_tier"] = "Tier 4 - Monitor"
scored_df.loc[:99, "intervention_tier"] = "Tier 1 - Immediate Action"
scored_df.loc[100:599, "intervention_tier"] = "Tier 2 - Care Manager Queue"
scored_df.loc[600:2599, "intervention_tier"] = "Tier 3 - Targeted Outreach"

def get_action(tier):
    if tier == "Tier 1 - Immediate Action":
        return "Call within 24 hrs + physician review + medication reconciliation"
    elif tier == "Tier 2 - Care Manager Queue":
        return "Nurse follow-up in 3 days + discharge counselling reinforcement"
    elif tier == "Tier 3 - Targeted Outreach":
        return "Phone outreach in 7 days + adherence reminder"
    else:
        return "Standard discharge workflow + passive monitoring"

scored_df["recommended_action_v3"] = scored_df["intervention_tier"].apply(get_action)

# Cost assumption can be swapped later with cited benchmark
cost_per_readmission = 15200.0
scored_df["estimated_cost_exposure_v3"] = scored_df["ml_risk_probability_v3"] * cost_per_readmission

# top-N class flags for Power BI
scored_df["top100_flag"] = 0
scored_df["top500_flag"] = 0
scored_df["top2000_flag"] = 0
scored_df.loc[:99, "top100_flag"] = 1
scored_df.loc[:499, "top500_flag"] = 1
scored_df.loc[:1999, "top2000_flag"] = 1

print("\n===== INTERVENTION TIER DISTRIBUTION =====")
print(scored_df["intervention_tier"].value_counts())

# =========================
# FEATURE IMPORTANCE
# =========================
if hasattr(best_model, "feature_importances_"):
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n===== TOP FEATURE IMPORTANCE =====")
    print(fi.head(20))
else:
    fi = pd.DataFrame(columns=["feature", "importance"])

# =========================
# SAVE TO POSTGRESQL
# =========================
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS ml_predictions_v3;")
cur.execute("""
CREATE TABLE ml_predictions_v3 (
    patient_id BIGINT,
    risk_score FLOAT,
    risk_segment VARCHAR(20),
    days_in_hospital INT,
    medications INT,
    num_diagnoses INT,
    outpatient_visits INT,
    emergency_visits INT,
    inpatient_visits INT,
    sentiment_score FLOAT,
    keyword_hits INT,
    urgency_score INT,
    ml_risk_probability_v3 FLOAT,
    intervention_tier VARCHAR(50),
    recommended_action_v3 TEXT,
    estimated_cost_exposure_v3 FLOAT,
    top100_flag INT,
    top500_flag INT,
    top2000_flag INT
);
""")
conn.commit()

cur.executemany("""
INSERT INTO ml_predictions_v3 (
    patient_id, risk_score, risk_segment, days_in_hospital, medications,
    num_diagnoses, outpatient_visits, emergency_visits, inpatient_visits,
    sentiment_score, keyword_hits, urgency_score, ml_risk_probability_v3,
    intervention_tier, recommended_action_v3, estimated_cost_exposure_v3,
    top100_flag, top500_flag, top2000_flag
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", list(scored_df.itertuples(index=False, name=None)))
conn.commit()

cur.execute("DROP TABLE IF EXISTS ml_model_metrics_v3;")
cur.execute("""
CREATE TABLE ml_model_metrics_v3 (
    model VARCHAR(50),
    auc FLOAT,
    precision_at_030 FLOAT,
    recall_at_030 FLOAT,
    f1_at_030 FLOAT,
    brier FLOAT,
    cv_auc_mean FLOAT,
    cv_auc_std FLOAT
);
""")
conn.commit()

cur.executemany("""
INSERT INTO ml_model_metrics_v3 (
    model, auc, precision_at_030, recall_at_030, f1_at_030, brier, cv_auc_mean, cv_auc_std
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
""", list(results_df.itertuples(index=False, name=None)))
conn.commit()

cur.close()
conn.close()

print("\nSaved tables:")
print("- ml_predictions_v3")
print("- ml_model_metrics_v3")
print(f"\nBest model: {best_model_name}")