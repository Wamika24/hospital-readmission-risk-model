import psycopg2
import pandas as pd
import numpy as np
print("SCRIPT STARTED")

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix
)

# =========================================
# DATABASE CONNECTION
# =========================================
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)

# =========================================
# CHECK TEXT COVERAGE
# We will NOT use text features unless coverage is near-full.
# =========================================
text_cov_query = """
SELECT COUNT(DISTINCT patient_id) AS covered_patients
FROM discharge_notes_v2;
"""
text_cov = pd.read_sql(text_cov_query, conn)["covered_patients"].iloc[0]

total_patients_query = """
SELECT COUNT(DISTINCT patient_id) AS total_patients
FROM patient_risk_analysis;
"""
total_patients = pd.read_sql(total_patients_query, conn)["total_patients"].iloc[0]

text_coverage_ratio = text_cov / total_patients if total_patients else 0.0
use_text_features = text_coverage_ratio >= 0.95

print(f"Text coverage: {text_cov}/{total_patients} = {text_coverage_ratio:.2%}")
print(f"Use text features in model: {use_text_features}")

# =========================================
# LOAD FULL FEATURE SET
# =========================================
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
    p.risk_score,
    p.risk_segment,
    p.readmit_30day_flag,
    n.sentiment_score,
    n.keyword_hits,
    n.urgency_score,
    n.sentiment_label
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

# =========================================
# DIAGNOSIS BUCKETING
# =========================================
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

df["diag_group"] = df["diag_1"].apply(bucket_diag)

# =========================================
# FEATURE ENGINEERING
# =========================================
df["days_safe"] = df["days_in_hospital"].replace(0, 1)
df["diag_safe"] = df["num_diagnoses"].replace(0, 1)

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

df["acute_burden"] = df["emergency_visits"] + df["inpatient_visits"]

df["care_load"] = (
    df["lab_procedures"] +
    df["procedures"] +
    df["medications"]
)

df["meds_per_day"] = df["medications"] / df["days_safe"]
df["labs_per_day"] = df["lab_procedures"] / df["days_safe"]
df["procedures_per_day"] = df["procedures"] / df["days_safe"]
df["meds_per_diagnosis"] = df["medications"] / df["diag_safe"]

df["complexity_score"] = (
    1.5 * df["num_diagnoses"] +
    0.5 * df["medications"] +
    2.0 * df["inpatient_visits"]
)

df["long_stay_flag"] = (df["days_in_hospital"] >= 7).astype(int)
df["polypharmacy_flag"] = (df["medications"] >= 15).astype(int)
df["high_diagnosis_flag"] = (df["num_diagnoses"] >= 6).astype(int)
df["repeat_acute_flag"] = (df["acute_burden"] >= 2).astype(int)

# =========================================
# HANDLE MISSING VALUES / TEXT FIELDS
# =========================================
for col in [
    "age", "gender", "race", "admission_type", "discharge_type",
    "admission_source", "diabetesmed", "medical_specialty",
    "a1cresult", "insulin", "risk_segment", "diag_group",
    "sentiment_label"
]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str)

for col in [
    "sentiment_score", "keyword_hits", "urgency_score"
]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# =========================================
# DEFINE FEATURE SETS
# Model A = Raw Clinical Model
# Model B = Augmented Ops Model
# =========================================
raw_numeric = [
    "days_in_hospital",
    "lab_procedures",
    "procedures",
    "medications",
    "outpatient_visits",
    "emergency_visits",
    "inpatient_visits",
    "num_diagnoses",
    "total_visits",
    "visit_burden",
    "acute_burden",
    "care_load",
    "meds_per_day",
    "labs_per_day",
    "procedures_per_day",
    "meds_per_diagnosis",
    "complexity_score",
    "long_stay_flag",
    "polypharmacy_flag",
    "high_diagnosis_flag",
    "repeat_acute_flag"
]

raw_categorical = [
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

raw_features = raw_numeric + raw_categorical

aug_numeric = raw_numeric + ["risk_score"]
aug_categorical = raw_categorical + ["risk_segment"]

if use_text_features:
    aug_numeric = aug_numeric + ["sentiment_score", "keyword_hits", "urgency_score"]
    aug_categorical = aug_categorical + ["sentiment_label"]

aug_features = aug_numeric + aug_categorical

target_col = "readmit_30day_flag"

# =========================================
# TRAIN / TEST SPLIT
# =========================================
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df[target_col]
)

# =========================================
# EVALUATION HELPERS
# =========================================
def precision_recall_at_n(y_true, prob, n):
    ranked = pd.DataFrame({"y_true": y_true, "prob": prob}).sort_values("prob", ascending=False)
    ranked = ranked.head(n)
    tp = ranked["y_true"].sum()
    precision_n = tp / n if n > 0 else 0.0
    recall_n = tp / y_true.sum() if y_true.sum() > 0 else 0.0
    return precision_n, recall_n

def evaluate_catboost(feature_cols, cat_cols, model_name):
    X_train = df.loc[train_idx, feature_cols].copy()
    X_test = df.loc[test_idx, feature_cols].copy()
    y_train = df.loc[train_idx, target_col].astype(int)
    y_test = df.loc[test_idx, target_col].astype(int)
    print("SAVING TEST FILES")
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    print("Test data saved successfully")

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    model.save_model("catboost_readmission_model.cbm")
    print("MODEL SAVED: catboost_readmission_model.cbm")

    prob = model.predict_proba(test_pool)[:, 1]
    auc = roc_auc_score(y_test, prob)
    ap = average_precision_score(y_test, prob)
    brier = brier_score_loss(y_test, prob)

    # business-friendly threshold snapshot
    pred_030 = (prob >= 0.30).astype(int)
    precision_030 = precision_score(y_test, pred_030, zero_division=0)
    recall_030 = recall_score(y_test, pred_030, zero_division=0)
    f1_030 = f1_score(y_test, pred_030, zero_division=0)

    # Top-N metrics
    p100, r100 = precision_recall_at_n(y_test.values, prob, min(100, len(y_test)))
    p250, r250 = precision_recall_at_n(y_test.values, prob, min(250, len(y_test)))
    p500, r500 = precision_recall_at_n(y_test.values, prob, min(500, len(y_test)))
    p1000, r1000 = precision_recall_at_n(y_test.values, prob, min(1000, len(y_test)))

    cm = confusion_matrix(y_test, pred_030)

    # cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = []
    cv_ap = []

    for fold_train_idx, fold_val_idx in skf.split(df[feature_cols], df[target_col]):
        X_fold_train = df.iloc[fold_train_idx][feature_cols].copy()
        y_fold_train = df.iloc[fold_train_idx][target_col].astype(int)
        X_fold_val = df.iloc[fold_val_idx][feature_cols].copy()
        y_fold_val = df.iloc[fold_val_idx][target_col].astype(int)

        fold_train_pool = Pool(X_fold_train, y_fold_train, cat_features=cat_cols)
        fold_val_pool = Pool(X_fold_val, y_fold_val, cat_features=cat_cols)

        fold_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False
        )

        fold_model.fit(fold_train_pool, eval_set=fold_val_pool, use_best_model=True)
        fold_prob = fold_model.predict_proba(fold_val_pool)[:, 1]

        cv_auc.append(roc_auc_score(y_fold_val, fold_prob))
        cv_ap.append(average_precision_score(y_fold_val, fold_prob))

    result = {
        "model_name": model_name,
        "auc": auc,
        "average_precision": ap,
        "brier": brier,
        "precision_at_030": precision_030,
        "recall_at_030": recall_030,
        "f1_at_030": f1_030,
        "precision_at_100": p100,
        "recall_at_100": r100,
        "precision_at_250": p250,
        "recall_at_250": r250,
        "precision_at_500": p500,
        "recall_at_500": r500,
        "precision_at_1000": p1000,
        "recall_at_1000": r1000,
        "cv_auc_mean": float(np.mean(cv_auc)),
        "cv_auc_std": float(np.std(cv_auc)),
        "cv_ap_mean": float(np.mean(cv_ap)),
        "cv_ap_std": float(np.std(cv_ap)),
        "confusion_matrix": cm,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "model_obj": model
    }

    return result

# =========================================
# RUN BOTH MODELS
# =========================================
raw_result = evaluate_catboost(raw_features, raw_categorical, "Raw Clinical CatBoost")
aug_result = evaluate_catboost(aug_features, aug_categorical, "Augmented Ops CatBoost")

results = [raw_result, aug_result]

print("\n===== MODEL RESULTS =====")
for r in results:
    print(f"\n{r['model_name']}")
    print(f"AUC ROC            : {r['auc']:.4f}")
    print(f"Average Precision  : {r['average_precision']:.4f}")
    print(f"Brier Score        : {r['brier']:.4f}")
    print(f"Precision@0.30     : {r['precision_at_030']:.4f}")
    print(f"Recall@0.30        : {r['recall_at_030']:.4f}")
    print(f"F1@0.30            : {r['f1_at_030']:.4f}")
    print(f"Precision@100      : {r['precision_at_100']:.4f}")
    print(f"Recall@100         : {r['recall_at_100']:.4f}")
    print(f"Precision@250      : {r['precision_at_250']:.4f}")
    print(f"Recall@250         : {r['recall_at_250']:.4f}")
    print(f"Precision@500      : {r['precision_at_500']:.4f}")
    print(f"Recall@500         : {r['recall_at_500']:.4f}")
    print(f"CV AUC Mean        : {r['cv_auc_mean']:.4f}")
    print(f"CV AUC Std         : {r['cv_auc_std']:.4f}")
    print(f"CV AP Mean         : {r['cv_ap_mean']:.4f}")
    print(f"CV AP Std          : {r['cv_ap_std']:.4f}")
    print("Confusion Matrix @0.30:")
    print(r["confusion_matrix"])

# =========================================
# SELECT BEST MODEL
# Priority:
# 1. Average Precision
# 2. AUC
# 3. Precision@100
# =========================================
ranking_df = pd.DataFrame([{
    "model_name": r["model_name"],
    "auc": r["auc"],
    "average_precision": r["average_precision"],
    "precision_at_100": r["precision_at_100"],
    "precision_at_250": r["precision_at_250"],
    "precision_at_500": r["precision_at_500"],
    "cv_auc_mean": r["cv_auc_mean"],
    "cv_ap_mean": r["cv_ap_mean"]
} for r in results]).sort_values(
    ["average_precision", "auc", "precision_at_100"],
    ascending=False
).reset_index(drop=True)

print("\n===== FINAL MODEL RANKING =====")
print(ranking_df)

best_name = ranking_df.loc[0, "model_name"]
best_result = raw_result if raw_result["model_name"] == best_name else aug_result
best_model = best_result["model_obj"]
best_feature_cols = best_result["feature_cols"]
best_cat_cols = best_result["cat_cols"]

print(f"\nBest model selected: {best_name}")

# =========================================
# FIT BEST MODEL ON FULL DATA
# =========================================
full_pool = Pool(df[best_feature_cols], df[target_col].astype(int), cat_features=best_cat_cols)
best_model.fit(full_pool, use_best_model=False)

full_prob = best_model.predict_proba(full_pool)[:, 1]

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
    "age",
    "gender",
    "race",
    "medical_specialty",
    "diag_group"
]].copy()

if use_text_features:
    scored_df["sentiment_score"] = df["sentiment_score"]
    scored_df["keyword_hits"] = df["keyword_hits"]
    scored_df["urgency_score"] = df["urgency_score"]

scored_df["ml_risk_probability_rebuild"] = full_prob
scored_df = scored_df.sort_values("ml_risk_probability_rebuild", ascending=False).reset_index(drop=True)

# business capacity tiers
scored_df["intervention_tier_rebuild"] = "Tier 4 - Monitor"
scored_df.loc[:99, "intervention_tier_rebuild"] = "Tier 1 - Immediate Action"
scored_df.loc[100:599, "intervention_tier_rebuild"] = "Tier 2 - Care Manager Queue"
scored_df.loc[600:2599, "intervention_tier_rebuild"] = "Tier 3 - Targeted Outreach"

def action_map(tier):
    if tier == "Tier 1 - Immediate Action":
        return "Call within 24 hrs + physician review + medication reconciliation"
    elif tier == "Tier 2 - Care Manager Queue":
        return "Nurse follow-up in 3 days + discharge counselling reinforcement"
    elif tier == "Tier 3 - Targeted Outreach":
        return "Phone outreach in 7 days + adherence reminder"
    return "Standard discharge workflow + passive monitoring"

scored_df["recommended_action_rebuild"] = scored_df["intervention_tier_rebuild"].apply(action_map)

# =========================================
# SAVE TO POSTGRESQL
# =========================================
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="readmission_db",
    user="postgres",
    password="1234"
)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS ml_rebuild_predictions;")
cur.execute("""
CREATE TABLE ml_rebuild_predictions (
    patient_id BIGINT,
    risk_score FLOAT,
    risk_segment VARCHAR(20),
    days_in_hospital INT,
    medications INT,
    num_diagnoses INT,
    outpatient_visits INT,
    emergency_visits INT,
    inpatient_visits INT,
    age VARCHAR(50),
    gender VARCHAR(50),
    race VARCHAR(100),
    medical_specialty VARCHAR(255),
    diag_group VARCHAR(100),
    sentiment_score FLOAT,
    keyword_hits INT,
    urgency_score INT,
    ml_risk_probability_rebuild FLOAT,
    intervention_tier_rebuild VARCHAR(50),
    recommended_action_rebuild TEXT
);
""")
conn.commit()

# make sure optional columns exist
if "sentiment_score" not in scored_df.columns:
    scored_df["sentiment_score"] = 0.0
if "keyword_hits" not in scored_df.columns:
    scored_df["keyword_hits"] = 0
if "urgency_score" not in scored_df.columns:
    scored_df["urgency_score"] = 0

insert_cols = [
    "patient_id", "risk_score", "risk_segment", "days_in_hospital", "medications",
    "num_diagnoses", "outpatient_visits", "emergency_visits", "inpatient_visits",
    "age", "gender", "race", "medical_specialty", "diag_group",
    "sentiment_score", "keyword_hits", "urgency_score",
    "ml_risk_probability_rebuild", "intervention_tier_rebuild", "recommended_action_rebuild"
]

cur.executemany("""
INSERT INTO ml_rebuild_predictions VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", list(scored_df[insert_cols].itertuples(index=False, name=None)))
conn.commit()

cur.execute("DROP TABLE IF EXISTS ml_rebuild_metrics;")
cur.execute("""
CREATE TABLE ml_rebuild_metrics (
    model_name VARCHAR(100),
    auc FLOAT,
    average_precision FLOAT,
    precision_at_100 FLOAT,
    precision_at_250 FLOAT,
    precision_at_500 FLOAT,
    cv_auc_mean FLOAT,
    cv_auc_std FLOAT,
    cv_ap_mean FLOAT,
    cv_ap_std FLOAT
);
""")
conn.commit()

metrics_insert = ranking_df.copy()
metrics_insert["cv_auc_std"] = [
    raw_result["cv_auc_std"] if name == raw_result["model_name"] else aug_result["cv_auc_std"]
    for name in metrics_insert["model_name"]
]
metrics_insert["cv_ap_std"] = [
    raw_result["cv_ap_std"] if name == raw_result["model_name"] else aug_result["cv_ap_std"]
    for name in metrics_insert["model_name"]
]

cur.executemany("""
INSERT INTO ml_rebuild_metrics VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", list(metrics_insert[[
    "model_name", "auc", "average_precision",
    "precision_at_100", "precision_at_250", "precision_at_500",
    "cv_auc_mean", "cv_auc_std", "cv_ap_mean", "cv_ap_std"
]].itertuples(index=False, name=None)))
conn.commit()

cur.close()
conn.close()

print("\nSaved tables:")
print("- ml_rebuild_predictions")
print("- ml_rebuild_metrics")
print(f"\nBest model: {best_name}")