from fastapi import FastAPI
from pydantic import BaseModel

import psycopg2
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Hospital Readmission Risk API",
    description="Healthcare Risk Stratification System",
    version="1.0"
)

# =====================================================
# DATABASE CONFIG
# =====================================================

DB_CONFIG = {
    "host": "localhost",
    "port": "5433",
    "database": "readmission_db",
    "user": "postgres",
    "password": "1234"
}

# =====================================================
# DIAGNOSIS GROUPING
# =====================================================

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

# =====================================================
# LOAD TRAINING DATA
# =====================================================

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

# =====================================================
# FEATURE ENGINEERING
# =====================================================

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

X = df[feature_cols]
y = df[TARGET].astype(int)

# =====================================================
# TRAIN MODEL
# =====================================================

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

# =====================================================
# API INPUT SCHEMA
# =====================================================

class PatientData(BaseModel):

    days_in_hospital: int
    lab_procedures: int
    procedures: int
    medications: int
    outpatient_visits: int
    emergency_visits: int
    inpatient_visits: int
    num_diagnoses: int

    age: str
    gender: str
    race: str

    admission_type: str
    discharge_type: str
    admission_source: str

    diabetesmed: str
    medical_specialty: str
    a1cresult: str
    insulin: str

    diag_1: str

# =====================================================
# ROOT ENDPOINT
# =====================================================

@app.get("/")

def home():

    return {
        "message": "Hospital Readmission Risk API Running"
    }

# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")

def predict(patient: PatientData):

    diag_group = bucket_diag(patient.diag_1)

    input_data = pd.DataFrame([{
        "days_in_hospital": patient.days_in_hospital,
        "lab_procedures": patient.lab_procedures,
        "procedures": patient.procedures,
        "medications": patient.medications,
        "outpatient_visits": patient.outpatient_visits,
        "emergency_visits": patient.emergency_visits,
        "inpatient_visits": patient.inpatient_visits,
        "num_diagnoses": patient.num_diagnoses,
        "age": patient.age,
        "gender": patient.gender,
        "race": patient.race,
        "admission_type": patient.admission_type,
        "discharge_type": patient.discharge_type,
        "admission_source": patient.admission_source,
        "diabetesmed": patient.diabetesmed,
        "medical_specialty": patient.medical_specialty,
        "a1cresult": patient.a1cresult,
        "insulin": patient.insulin,
        "diag_group": diag_group
    }])

    pred_prob = model.predict_proba(input_data)[0][1]

    # =================================================
    # RISK TIERS
    # =================================================

    if pred_prob >= 0.70:
        risk_tier = "HIGH"

    elif pred_prob >= 0.40:
        risk_tier = "MEDIUM"

    else:
        risk_tier = "LOW"

    # =================================================
    # SIMPLE EXPLANATION LOGIC
    # =================================================

    risk_factors = []

    if patient.inpatient_visits >= 2:
        risk_factors.append("High inpatient utilization")

    if patient.emergency_visits >= 1:
        risk_factors.append("Emergency visit history")

    if patient.medications >= 20:
        risk_factors.append("High medication burden")

    if patient.num_diagnoses >= 8:
        risk_factors.append("Multiple diagnoses")

    if patient.insulin.lower() == "yes":
        risk_factors.append("Insulin usage")

    if len(risk_factors) == 0:
        risk_factors.append("General clinical risk pattern")

    return {
        "risk_probability": round(float(pred_prob), 4),
        "risk_tier": risk_tier,
        "top_risk_factors": risk_factors
    }