import psycopg2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ======================
# DB CONNECTION
# ======================
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)

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
    p.readmit_30day_flag
FROM diabetes_clean d
JOIN patient_risk_analysis p
    ON d.patient_id = p.patient_id
"""

df = pd.read_sql(query, conn)
conn.close()

print("Rows loaded:", len(df))

# ======================
# CLEAN / ENCODE
# ======================
categorical_cols = ["age", "admission_type", "discharge_type", "admission_source", "diabetesmed", "risk_segment"]

for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ======================
# FEATURES / TARGET
# ======================
X = df.drop(columns=["patient_id", "readmit_30day_flag"])
y = df["readmit_30day_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ======================
# MODEL 1 RANDOM FOREST
# ======================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print("\n===== RANDOM FOREST =====")
print("Accuracy :", round(accuracy_score(y_test, rf_pred), 4))
print("Precision:", round(precision_score(y_test, rf_pred), 4))
print("Recall   :", round(recall_score(y_test, rf_pred), 4))
print("F1 Score :", round(f1_score(y_test, rf_pred), 4))
print("AUC ROC  :", round(roc_auc_score(y_test, rf_prob), 4))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

# ======================
# CROSS VALIDATION
# ======================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    rf, X, y,
    cv=cv,
    scoring="roc_auc"
)

print("\nCV AUC Scores:", np.round(cv_scores, 4))
print("CV Mean AUC :", round(cv_scores.mean(), 4))
print("CV Std Dev  :", round(cv_scores.std(), 4))

# ======================
# MODEL 2 LOGISTIC REGRESSION
# ======================
lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

lr.fit(X_train, y_train)

lr_prob = lr.predict_proba(X_test)[:, 1]
lr_pred = lr.predict(X_test)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy :", round(accuracy_score(y_test, lr_pred), 4))
print("Precision:", round(precision_score(y_test, lr_pred), 4))
print("Recall   :", round(recall_score(y_test, lr_pred), 4))
print("F1 Score :", round(f1_score(y_test, lr_pred), 4))
print("AUC ROC  :", round(roc_auc_score(y_test, lr_prob), 4))

# ======================
# FEATURE IMPORTANCE
# ======================
fi = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\n===== TOP FEATURE IMPORTANCE =====")
print(fi)