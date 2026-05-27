import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)

cursor = conn.cursor()

print("Creating discharge_notes table...")

cursor.execute("""
DROP TABLE IF EXISTS discharge_notes;

CREATE TABLE discharge_notes (
    patient_id BIGINT,
    encounter_id BIGINT,
    risk_segment VARCHAR(20),
    num_medications INT,
    number_diagnoses INT,
    days_in_hospital INT,
    discharge_note TEXT,
    sentiment VARCHAR(20),
    compliance_risk INT,
    social_risk_factors TEXT,
    readmission_signal VARCHAR(10),
    ai_risk_label VARCHAR(20)
);
""")

conn.commit()

cursor.execute("""
SELECT
    patient_id,
    medications,
    num_diagnoses,
    days_in_hospital,
    emergency_visits,
    inpatient_visits,
    readmit_30day_flag,
    CASE
        WHEN risk_score >= 8 THEN 'High'
        WHEN risk_score >= 5 THEN 'Medium'
        ELSE 'Low'
    END AS risk_segment
FROM diabetes_clean
LIMIT 100;
""")

patients = cursor.fetchall()

rows = []

for i, p in enumerate(patients, start=1):
    patient_id, meds, diag, days, emer, inp, flag, risk = p

    note = f"Patient discharged after {days} days. Medication count {meds}. Diagnoses count {diag}. Follow-up advised."

    score = 0
    if meds >= 20:
        score += 1
    if emer >= 2:
        score += 1
    if inp >= 2:
        score += 1
    if flag == 1:
        score += 1

    if score >= 3:
        sentiment = "Negative"
        signal = "Yes"
        label = "High"
    elif score == 2:
        sentiment = "Neutral"
        signal = "No"
        label = "Medium"
    else:
        sentiment = "Positive"
        signal = "No"
        label = "Low"

    compliance = min(10, 2 + meds // 5)

    social = "Poor follow-up risk" if emer >= 2 and inp >= 2 else "None"

    rows.append((
        patient_id,
        patient_id,
        risk,
        meds,
        diag,
        days,
        note,
        sentiment,
        compliance,
        social,
        signal,
        label
    ))

    print(f"{i}/100 done")

cursor.executemany("""
INSERT INTO discharge_notes VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
""", rows)

conn.commit()
conn.close()

print("All notes saved!")

df = pd.DataFrame(rows)
print(df[7].value_counts())
print(df[11].value_counts())