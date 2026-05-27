import random
import psycopg2
import pandas as pd
from textblob import TextBlob

random.seed(42)

# =========================
# DATABASE CONNECTION
# =========================
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)

print("Loading existing discharge_notes table...")

query = """
SELECT
    patient_id,
    encounter_id,
    risk_segment,
    num_medications,
    number_diagnoses,
    days_in_hospital
FROM discharge_notes
"""

df = pd.read_sql(query, conn)

print(f"Loaded {len(df)} rows from discharge_notes")

# =========================
# VARIED CLINICAL PHRASES
# =========================
high_risk_phrases = [
    "poor glycaemic control was noted during admission",
    "patient appears non-compliant with medication instructions",
    "recurrent emergency admission pattern remains concerning",
    "chronic complications increase post-discharge risk",
    "missed follow-up risk may elevate readmission probability",
    "clinical status remains relatively unstable at discharge"
]

medium_risk_phrases = [
    "patient requires close monitoring after discharge",
    "medication adjustment and regular review were advised",
    "follow-up planning remains important for recovery",
    "moderate recurrence risk cannot be ruled out",
    "continued outpatient evaluation is recommended"
]

low_risk_phrases = [
    "patient was stable for discharge",
    "condition improved with treatment during admission",
    "routine follow-up was scheduled appropriately",
    "patient responded well to inpatient treatment",
    "no acute complications were observed at discharge"
]

closing_phrases = {
    "High": [
        "Immediate post-discharge outreach is strongly recommended.",
        "Care manager follow-up within 48 hours is advised.",
        "Early physician review should be prioritized."
    ],
    "Medium": [
        "Nurse follow-up within 7 days is recommended.",
        "Structured post-discharge monitoring is advised.",
        "Medication counselling follow-up should be completed."
    ],
    "Low": [
        "Standard discharge workflow appears sufficient.",
        "Routine outpatient follow-up should be adequate.",
        "No enhanced escalation appears necessary currently."
    ]
}

risk_keywords = [
    "non-compliant",
    "missed follow-up",
    "recurrent",
    "poor glycaemic control",
    "chronic",
    "unstable",
    "emergency",
    "concerning"
]

# =========================
# NOTE GENERATOR
# =========================
def generate_note(risk_segment, meds, diagnoses, days):
    lines = []

    lines.append(f"Patient discharged after {days} days of hospitalization.")
    lines.append(f"Medication burden at discharge was {meds}.")
    lines.append(f"Clinical complexity included {diagnoses} documented diagnoses.")

    if risk_segment == "High":
        lines.append(random.choice(high_risk_phrases).capitalize() + ".")
        lines.append(random.choice(high_risk_phrases).capitalize() + ".")
    elif risk_segment == "Medium":
        lines.append(random.choice(medium_risk_phrases).capitalize() + ".")
        # Medium can sometimes carry one riskier phrase
        if meds >= 15 or diagnoses >= 6:
            lines.append(random.choice(high_risk_phrases).capitalize() + ".")
        else:
            lines.append(random.choice(medium_risk_phrases).capitalize() + ".")
    else:
        lines.append(random.choice(low_risk_phrases).capitalize() + ".")
        if meds >= 12 or diagnoses >= 5:
            lines.append(random.choice(medium_risk_phrases).capitalize() + ".")
        else:
            lines.append(random.choice(low_risk_phrases).capitalize() + ".")

    lines.append(random.choice(closing_phrases.get(risk_segment, closing_phrases["Low"])))

    return " ".join(lines)

# =========================
# TEXT FEATURE EXTRACTION
# =========================
def extract_features(note):
    note_lower = note.lower()

    keyword_list = [kw for kw in risk_keywords if kw in note_lower]
    keyword_hits = len(keyword_list)

    sentiment_score = float(TextBlob(note).sentiment.polarity)

    if sentiment_score < -0.10:
        sentiment_label = "Negative"
    elif sentiment_score < 0.15:
        sentiment_label = "Neutral"
    else:
        sentiment_label = "Positive"

    urgency_score = min(10, keyword_hits * 2 + (1 if "strongly recommended" in note_lower else 0))

    return (
        sentiment_score,
        sentiment_label,
        keyword_hits,
        ", ".join(keyword_list) if keyword_list else "None",
        urgency_score
    )

# =========================
# GENERATE NEW TEXT LAYER
# =========================
notes = []
sentiment_scores = []
sentiment_labels = []
keyword_hits_list = []
keywords_found_list = []
urgency_scores = []

for i, row in df.iterrows():
    note = generate_note(
        row["risk_segment"],
        row["num_medications"],
        row["number_diagnoses"],
        row["days_in_hospital"]
    )

    sentiment_score, sentiment_label, keyword_hits, keywords_found, urgency_score = extract_features(note)

    notes.append(note)
    sentiment_scores.append(sentiment_score)
    sentiment_labels.append(sentiment_label)
    keyword_hits_list.append(keyword_hits)
    keywords_found_list.append(keywords_found)
    urgency_scores.append(urgency_score)

    if (i + 1) % 20 == 0 or (i + 1) == len(df):
        print(f"{i + 1}/{len(df)} done")

df["enhanced_discharge_note"] = notes
df["sentiment_score"] = sentiment_scores
df["sentiment_label"] = sentiment_labels
df["keyword_hits"] = keyword_hits_list
df["risk_keywords_found"] = keywords_found_list
df["urgency_score"] = urgency_scores

# =========================
# WRITE TO NEW TABLE
# =========================
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS discharge_notes_v2;")

cursor.execute("""
CREATE TABLE discharge_notes_v2 (
    patient_id BIGINT,
    encounter_id BIGINT,
    risk_segment VARCHAR(20),
    num_medications INT,
    number_diagnoses INT,
    days_in_hospital INT,
    enhanced_discharge_note TEXT,
    sentiment_score FLOAT,
    sentiment_label VARCHAR(20),
    keyword_hits INT,
    risk_keywords_found TEXT,
    urgency_score INT
);
""")
conn.commit()

insert_rows = list(df[[
    "patient_id",
    "encounter_id",
    "risk_segment",
    "num_medications",
    "number_diagnoses",
    "days_in_hospital",
    "enhanced_discharge_note",
    "sentiment_score",
    "sentiment_label",
    "keyword_hits",
    "risk_keywords_found",
    "urgency_score"
]].itertuples(index=False, name=None))

cursor.executemany("""
INSERT INTO discharge_notes_v2 (
    patient_id,
    encounter_id,
    risk_segment,
    num_medications,
    number_diagnoses,
    days_in_hospital,
    enhanced_discharge_note,
    sentiment_score,
    sentiment_label,
    keyword_hits,
    risk_keywords_found,
    urgency_score
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", insert_rows)

conn.commit()
cursor.close()
conn.close()

print("\ndischarge_notes_v2 created successfully!")

print("\nSentiment distribution:")
print(df["sentiment_label"].value_counts())

print("\nKeyword hit distribution:")
print(df["keyword_hits"].value_counts().sort_index())

print("\nSample output:")
print(df[[
    "patient_id",
    "risk_segment",
    "sentiment_score",
    "sentiment_label",
    "keyword_hits",
    "urgency_score",
    "enhanced_discharge_note"
]].head(5).to_string(index=False))