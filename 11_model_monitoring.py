import psycopg2
import pandas as pd
import numpy as np

from scipy.stats import ks_2samp

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
# CONNECT TO DATABASE
# =====================================================

conn = psycopg2.connect(**DB_CONFIG)

# =====================================================
# LOAD DATA
# =====================================================

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
    p.readmit_30day_flag
FROM diabetes_clean d
JOIN patient_risk_analysis p
    ON d.patient_id = p.patient_id
WHERE p.readmit_30day_flag IS NOT NULL
"""

df = pd.read_sql(query, conn)

print(f"\nRows loaded: {len(df)}")

# =====================================================
# CREATE TRAIN VS PRODUCTION SPLIT
# =====================================================

df = df.sort_values("patient_id").reset_index(drop=True)

train_df = df.iloc[:50000]
prod_df = df.iloc[50000:]

print("\n===== MONITORING DATA SPLIT =====")
print(f"Training population rows   : {len(train_df)}")
print(f"Production population rows : {len(prod_df)}")

# =====================================================
# NUMERIC FEATURES
# =====================================================

numeric_features = [
    "days_in_hospital",
    "lab_procedures",
    "procedures",
    "medications",
    "outpatient_visits",
    "emergency_visits",
    "inpatient_visits",
    "num_diagnoses"
]

# =====================================================
# DRIFT ANALYSIS
# =====================================================

results = []

print("\n===== FEATURE DRIFT ANALYSIS =====")

for feature in numeric_features:

    train_vals = train_df[feature].dropna()
    prod_vals = prod_df[feature].dropna()

    train_mean = train_vals.mean()
    prod_mean = prod_vals.mean()

    pct_shift = (
        ((prod_mean - train_mean) / train_mean) * 100
        if train_mean != 0 else 0
    )

    ks_stat, p_value = ks_2samp(train_vals, prod_vals)

    # ============================================
    # DRIFT STATUS
    # ============================================

    if abs(pct_shift) >= 20:
        drift_level = "HIGH"

    elif abs(pct_shift) >= 10:
        drift_level = "MODERATE"

    else:
        drift_level = "LOW"

    # ============================================

    results.append({
        "feature_name": feature,
        "train_mean": round(train_mean, 4),
        "production_mean": round(prod_mean, 4),
        "percent_shift": round(pct_shift, 2),
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 6),
        "drift_level": drift_level
    })

# =====================================================
# RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(results)

print(results_df)

# =====================================================
# DRIFT SUMMARY
# =====================================================

high_drift = len(results_df[results_df["drift_level"] == "HIGH"])
moderate_drift = len(results_df[results_df["drift_level"] == "MODERATE"])

print("\n===== DRIFT SUMMARY =====")

print(f"High drift features     : {high_drift}")
print(f"Moderate drift features : {moderate_drift}")

# =====================================================
# RETRAINING RECOMMENDATION
# =====================================================

print("\n===== RETRAINING DECISION =====")

if high_drift >= 3:

    recommendation = "RETRAIN IMMEDIATELY"

elif moderate_drift >= 3:

    recommendation = "MONITOR CLOSELY"

else:

    recommendation = "MODEL STABLE"

print(f"Recommendation: {recommendation}")

# =====================================================
# SAVE TO POSTGRESQL
# =====================================================

cur = conn.cursor()

cur.execute("""
DROP TABLE IF EXISTS model_monitoring_results;
""")

cur.execute("""
CREATE TABLE model_monitoring_results (
    feature_name VARCHAR(100),
    train_mean FLOAT,
    production_mean FLOAT,
    percent_shift FLOAT,
    ks_statistic FLOAT,
    p_value FLOAT,
    drift_level VARCHAR(20)
);
""")

conn.commit()

insert_rows = [
    (
        row["feature_name"],
        row["train_mean"],
        row["production_mean"],
        row["percent_shift"],
        row["ks_statistic"],
        row["p_value"],
        row["drift_level"]
    )
    for _, row in results_df.iterrows()
]

cur.executemany("""
INSERT INTO model_monitoring_results
VALUES (%s,%s,%s,%s,%s,%s,%s)
""", insert_rows)

conn.commit()

cur.close()
conn.close()

print("\nSaved table: model_monitoring_results")

print("\nModel monitoring analysis complete")