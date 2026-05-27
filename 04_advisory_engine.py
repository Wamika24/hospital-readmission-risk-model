import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)

print("Loading ML predictions...")

query = """
SELECT
    m.patient_id,
    m.ml_risk_probability,
    m.ml_predicted_class,
    d.days_in_hospital,
    d.medications,
    d.num_diagnoses,
    d.emergency_visits,
    d.inpatient_visits
FROM ml_predictions m
JOIN diabetes_clean d
ON m.patient_id = d.patient_id;
"""

df = pd.read_sql(query, conn)
conn.close()

def get_action(prob):
    if prob >= 0.70:
        return "High Priority", "Call within 24 hrs + doctor follow-up in 3 days"
    elif prob >= 0.40:
        return "Medium Priority", "Nurse follow-up in 7 days + medication counselling"
    else:
        return "Low Priority", "Standard discharge workflow"

df[["priority_level", "recommended_action"]] = df["ml_risk_probability"].apply(
    lambda x: pd.Series(get_action(x))
)

# ESTIMATED SAVINGS LOGIC
cost_per_readmission = 15000

df["estimated_cost_exposure"] = df["ml_risk_probability"] * cost_per_readmission

conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)

cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS advisory_actions;

CREATE TABLE advisory_actions (
    patient_id BIGINT,
    ml_risk_probability FLOAT,
    ml_predicted_class INT,
    priority_level VARCHAR(30),
    recommended_action TEXT,
    estimated_cost_exposure FLOAT
);
""")
conn.commit()

rows = list(df[[
    "patient_id",
    "ml_risk_probability",
    "ml_predicted_class",
    "priority_level",
    "recommended_action",
    "estimated_cost_exposure"
]].itertuples(index=False, name=None))

cursor.executemany("""
INSERT INTO advisory_actions (
    patient_id,
    ml_risk_probability,
    ml_predicted_class,
    priority_level,
    recommended_action,
    estimated_cost_exposure
)
VALUES (%s, %s, %s, %s, %s, %s)
""", rows)

conn.commit()
conn.close()

print("Advisory actions saved to advisory_actions table")
print(df[["priority_level"]].value_counts())