import psycopg2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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
# LOAD STRUCTURED + TEXT FEATURES
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
# avoid divide-by-zero
df["num_diagnoses_safe"] = df["num_diagnoses"].replace(0, 1)

df["total_visits"] = (
    df["outpatient_visits"] +
    df["emergency_visits"] +
    df["inpatient_visits"]
)

df["care_load"] = (
    df["lab_procedures"] +
    df["procedures"] +
    df["medications"]
)

df["meds_per_diagnosis"] = df["medications"] / df["num_diagnoses_safe"]
df["procedures_per_day"] = df["procedures"] / df["days_in_hospital"].replace(0, 1)
df["labs_per_day"] = df["lab_procedures"] / df["days_in_hospital"].replace(0, 1)

df["acute_ratio"] = df["emergency_visits"] / df["total_visits"].replace(0, 1)
df["inpatient_ratio"] = df["inpatient_visits"] / df["total_visits"].replace(0, 1)

df["long_stay_flag"] = (df["days_in_hospital"] >= 7).astype(int)
df["polypharmacy_flag"] = (df["medications"] >= 15).astype(int)
df["high_diagnosis_flag"] = (df["num_diagnoses"] >= 6).astype(int)
df["repeat_visit_flag"] = (df["total_visits"] >= 3).astype(int)

# =========================
# ENCODE CATEGORICALS
# =========================
categorical_cols = [
    "age", "admission_type", "discharge_type",
    "admission_source", "diabetesmed",
    "risk_segment", "sentiment_label"
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
    "care_load",
    "meds_per_diagnosis",
    "procedures_per_day",
    "labs_per_day",
    "acute_ratio",
    "inpatient_ratio",
    "long_stay_flag",
    "polypharmacy_flag",
    "high_diagnosis_flag",
    "repeat_visit_flag"
]

X = df[feature_cols]
y = df["readmit_30day_flag"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# MODEL DEFINITIONS
# =========================
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    )
}

results = []
trained_models = {}

print("\n===== MODEL COMPARISON =====")

for name, model in models.items():
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    # convert decision_function to 0-1-ish only if needed
    if name == "HistGradientBoosting":
        # HGB has predict_proba available in recent sklearn, but this keeps it safe
        try:
            prob = model.predict_proba(X_test)[:, 1]
        except:
            prob = 1 / (1 + np.exp(-prob))

    pred = (prob >= 0.5).astype(int)

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
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "cv_auc_mean": cv_auc.mean(),
        "cv_auc_std": cv_auc.std()
    })

    trained_models[name] = model

    print(f"\n{name}")
    print(f"AUC ROC     : {auc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"Brier Score : {brier:.4f}")
    print(f"CV AUC Mean : {cv_auc.mean():.4f}")
    print(f"CV AUC Std  : {cv_auc.std():.4f}")

results_df = pd.DataFrame(results).sort_values(
    ["auc", "f1", "cv_auc_mean"],
    ascending=False
).reset_index(drop=True)

print("\n===== FINAL MODEL RANKING =====")
print(results_df)

best_model_name = results_df.loc[0, "model"]
best_model = trained_models[best_model_name]

print(f"\nBest model selected: {best_model_name}")

# =========================
# THRESHOLD TUNING
# =========================
best_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)

if best_model_name == "HistGradientBoosting":
    try:
        best_prob = best_model.predict_proba(X_test)[:, 1]
    except:
        best_prob = 1 / (1 + np.exp(-best_prob))

thresholds = np.arange(0.20, 0.71, 0.05)
threshold_rows = []

for t in thresholds:
    pred_t = (best_prob >= t).astype(int)
    threshold_rows.append({
        "threshold": round(float(t), 2),
        "precision": precision_score(y_test, pred_t, zero_division=0),
        "recall": recall_score(y_test, pred_t, zero_division=0),
        "f1": f1_score(y_test, pred_t, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df = threshold_df.sort_values(["f1", "recall"], ascending=False).reset_index(drop=True)

best_threshold = float(threshold_df.loc[0, "threshold"])
print("\n===== THRESHOLD TUNING =====")
print(threshold_df)
print(f"\nSelected threshold: {best_threshold:.2f}")

final_pred = (best_prob >= best_threshold).astype(int)

print("\n===== FINAL TEST METRICS USING SELECTED THRESHOLD =====")
print(f"Precision : {precision_score(y_test, final_pred, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, final_pred, zero_division=0):.4f}")
print(f"F1 Score  : {f1_score(y_test, final_pred, zero_division=0):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, final_pred))

# =========================
# PRECISION @ TOP K
# =========================
test_scores = pd.DataFrame({
    "y_true": y_test.values,
    "prob": best_prob
}).sort_values("prob", ascending=False)

for k in [100, 250, 500, 1000]:
    if len(test_scores) >= k:
        p_at_k = test_scores.head(k)["y_true"].mean()
        print(f"Precision@{k}: {p_at_k:.4f}")

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
# PRODUCTION SCORING ON FULL DATA
# =========================
full_prob = best_model.predict_proba(X)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X)

if best_model_name == "HistGradientBoosting":
    try:
        full_prob = best_model.predict_proba(X)[:, 1]
    except:
        full_prob = 1 / (1 + np.exp(-full_prob))

full_pred = (full_prob >= best_threshold).astype(int)

scored_df = pd.DataFrame({
    "patient_id": df["patient_id"],
    "ml_risk_probability_v2": full_prob,
    "ml_predicted_class_v2": full_pred,
    "selected_model": best_model_name,
    "selected_threshold": best_threshold
})

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

cur.execute("DROP TABLE IF EXISTS ml_predictions_v2;")
cur.execute("""
CREATE TABLE ml_predictions_v2 (
    patient_id BIGINT,
    ml_risk_probability_v2 FLOAT,
    ml_predicted_class_v2 INT,
    selected_model VARCHAR(50),
    selected_threshold FLOAT
);
""")
conn.commit()

cur.executemany("""
INSERT INTO ml_predictions_v2 (
    patient_id,
    ml_risk_probability_v2,
    ml_predicted_class_v2,
    selected_model,
    selected_threshold
)
VALUES (%s, %s, %s, %s, %s)
""", list(scored_df.itertuples(index=False, name=None)))
conn.commit()

cur.execute("DROP TABLE IF EXISTS ml_model_metrics_v2;")
cur.execute("""
CREATE TABLE ml_model_metrics_v2 (
    model VARCHAR(50),
    auc FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1 FLOAT,
    brier FLOAT,
    cv_auc_mean FLOAT,
    cv_auc_std FLOAT
);
""")
conn.commit()

cur.executemany("""
INSERT INTO ml_model_metrics_v2 (
    model, auc, precision, recall, f1, brier, cv_auc_mean, cv_auc_std
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
""", list(results_df[[
    "model", "auc", "precision", "recall", "f1", "brier", "cv_auc_mean", "cv_auc_std"
]].itertuples(index=False, name=None)))
conn.commit()

cur.execute("DROP TABLE IF EXISTS ml_threshold_metrics_v2;")
cur.execute("""
CREATE TABLE ml_threshold_metrics_v2 (
    threshold FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1 FLOAT
);
""")
conn.commit()

cur.executemany("""
INSERT INTO ml_threshold_metrics_v2 (
    threshold, precision, recall, f1
)
VALUES (%s, %s, %s, %s)
""", list(threshold_df.itertuples(index=False, name=None)))
conn.commit()

cur.close()
conn.close()

print("\nSaved tables:")
print("- ml_predictions_v2")
print("- ml_model_metrics_v2")
print("- ml_threshold_metrics_v2")
print(f"\nBest model: {best_model_name}")
print(f"Best threshold: {best_threshold:.2f}")