import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# CONNECT TO POSTGRESQL
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)

print("Loading data from diabetes_clean...")

query = """
SELECT
    patient_id,
    age,
    gender,
    days_in_hospital,
    lab_procedures,
    procedures,
    medications,
    outpatient_visits,
    emergency_visits,
    inpatient_visits,
    num_diagnoses,
    admission_type,
    discharge_type,
    admission_source,
    diabetesmed,
    readmit_30day_flag
FROM diabetes_clean
WHERE readmit_30day_flag IS NOT NULL;
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"Loaded {len(df)} rows")

# ENCODE CATEGORICAL COLUMNS
categorical_cols = ["age", "gender", "admission_type", "discharge_type", "admission_source", "diabetesmed"]

for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# FEATURES AND TARGET
feature_cols = [
    "age",
    "gender",
    "days_in_hospital",
    "lab_procedures",
    "procedures",
    "medications",
    "outpatient_visits",
    "emergency_visits",
    "inpatient_visits",
    "num_diagnoses",
    "admission_type",
    "discharge_type",
    "admission_source",
    "diabetesmed"
]

X = df[feature_cols]
y = df["readmit_30day_flag"]

print("Splitting train and test data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# EVALUATION
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print("=" * 50)
print(f"AUC Score: {auc:.3f}")
print(classification_report(y_test, y_pred))

# FEATURE IMPORTANCE
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTOP 10 FEATURES")
print("=" * 50)
print(importance_df.head(10))

# PREDICT FOR ALL PATIENTS
print("\nGenerating predictions for all patients...")

all_prob = model.predict_proba(X)[:, 1]
all_pred = (all_prob >= 0.5).astype(int)

result_df = pd.DataFrame({
    "patient_id": df["patient_id"],
    "ml_risk_probability": all_prob,
    "ml_predicted_class": all_pred
})

# SAVE BACK TO POSTGRESQL
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)

cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS ml_predictions;

CREATE TABLE ml_predictions (
    patient_id BIGINT,
    ml_risk_probability FLOAT,
    ml_predicted_class INT
);
""")
conn.commit()

rows = list(result_df.itertuples(index=False, name=None))

cursor.executemany("""
INSERT INTO ml_predictions (
    patient_id,
    ml_risk_probability,
    ml_predicted_class
)
VALUES (%s, %s, %s)
""", rows)

conn.commit()
conn.close()

print("\nML predictions saved to table: ml_predictions")
print(result_df.head())