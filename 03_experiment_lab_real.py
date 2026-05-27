import json
import psycopg2
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix
)

DB_CONFIG = {
    "host": "localhost",
    "port": "5433",
    "database": "readmission_db",
    "user": "postgres",
    "password": "1234"
}


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


def top_n_metrics(y_true, prob, n):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob)

    n = min(n, len(y_true))
    order = np.argsort(-prob)[:n]
    tp = y_true[order].sum()

    precision_n = tp / n if n > 0 else 0.0
    recall_n = tp / y_true.sum() if y_true.sum() > 0 else 0.0
    return float(precision_n), float(recall_n)


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
    p.risk_score,
    p.risk_segment,
    p.readmit_30day_flag
FROM diabetes_clean d
JOIN patient_risk_analysis p
    ON d.patient_id = p.patient_id
WHERE p.readmit_30day_flag IS NOT NULL
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"Rows loaded: {len(df)}")

df["diag_group"] = df["diag_1"].apply(bucket_diag)

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

for col in [
    "age", "gender", "race", "admission_type", "discharge_type",
    "admission_source", "diabetesmed", "medical_specialty",
    "a1cresult", "insulin", "diag_group", "risk_segment"
]:
    df[col] = df[col].fillna("Unknown").astype(str)

TARGET = "readmit_30day_flag"

train_val_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df[TARGET]
)

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.25,
    random_state=42,
    stratify=df.loc[train_val_idx, TARGET]
)

y_test = df.loc[test_idx, TARGET].astype(int).values
test_base_rate = float(np.mean(y_test))

print("\n===== FROZEN TEST SET =====")
print(f"Test rows            : {len(test_idx)}")
print(f"Test positives       : {int(y_test.sum())}")
print(f"Test base rate       : {test_base_rate:.4f}")
print(f"Random P@100 approx  : {test_base_rate:.4f}")
print(f"Random P@250 approx  : {test_base_rate:.4f}")
print(f"Random P@500 approx  : {test_base_rate:.4f}")

basic_numeric = [
    "days_in_hospital",
    "lab_procedures",
    "procedures",
    "medications",
    "outpatient_visits",
    "emergency_visits",
    "inpatient_visits",
    "num_diagnoses"
]

clinical_cats = [
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

engineered_numeric = [
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

experiments = {
    "E1_Basic_Utilization": {
        "features": basic_numeric,
        "cat_cols": []
    },
    "E2_BasicPlusClinicalCats": {
        "features": basic_numeric + clinical_cats,
        "cat_cols": clinical_cats
    },
    "E3_EngineeredClinical": {
        "features": basic_numeric + engineered_numeric + clinical_cats,
        "cat_cols": clinical_cats
    },
    "E4_AugmentedOps": {
        "features": basic_numeric + engineered_numeric + clinical_cats + ["risk_score", "risk_segment"],
        "cat_cols": clinical_cats + ["risk_segment"]
    }
}

param_grid = [
    {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5},
    {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 7}
]

results = []
trained_final_models = {}


for exp_name, exp in experiments.items():
    print(f"\n{'='*60}")
    print(f"RUNNING {exp_name}")
    print(f"{'='*60}")

    feature_cols = exp["features"]
    cat_cols = exp["cat_cols"]
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    X_train = df.loc[train_idx, feature_cols].copy()
    y_train = df.loc[train_idx, TARGET].astype(int)

    X_val = df.loc[val_idx, feature_cols].copy()
    y_val = df.loc[val_idx, TARGET].astype(int)

    X_trainval = df.loc[train_val_idx, feature_cols].copy()
    y_trainval = df.loc[train_val_idx, TARGET].astype(int)

    X_test = df.loc[test_idx, feature_cols].copy()
    y_test_series = df.loc[test_idx, TARGET].astype(int)

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    best_val_record = None
    best_val_params = None

    for i, p in enumerate(param_grid, start=1):
        print(f"  Tuning config {i}/{len(param_grid)}: {p}")

        model = CatBoostClassifier(
            iterations=250,
            depth=p["depth"],
            learning_rate=p["learning_rate"],
            l2_leaf_reg=p["l2_leaf_reg"],
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            od_type="Iter",
            od_wait=40,
            verbose=False
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_prob = model.predict_proba(val_pool)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        val_ap = average_precision_score(y_val, val_prob)
        val_p100, _ = top_n_metrics(y_val.values, val_prob, 100)

        record = {
            "val_auc": float(val_auc),
            "val_ap": float(val_ap),
            "val_p100": float(val_p100),
            "best_iteration": int(model.get_best_iteration())
        }

        print(
            f"    val_auc={val_auc:.4f} | "
            f"val_ap={val_ap:.4f} | "
            f"val_p100={val_p100:.4f} | "
            f"best_iter={record['best_iteration']}"
        )

        if (best_val_record is None) or (
            (record["val_ap"], record["val_p100"], record["val_auc"])
            >
            (best_val_record["val_ap"], best_val_record["val_p100"], best_val_record["val_auc"])
        ):
            best_val_record = record
            best_val_params = p

    final_iterations = max(120, int(best_val_record["best_iteration"] * 1.15) + 20)

    trainval_pool = Pool(X_trainval, y_trainval, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test_series, cat_features=cat_idx)

    final_model = CatBoostClassifier(
        iterations=final_iterations,
        depth=best_val_params["depth"],
        learning_rate=best_val_params["learning_rate"],
        l2_leaf_reg=best_val_params["l2_leaf_reg"],
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False
    )

    final_model.fit(trainval_pool)

    test_prob = final_model.predict_proba(test_pool)[:, 1]

    test_auc = roc_auc_score(y_test_series, test_prob)
    test_ap = average_precision_score(y_test_series, test_prob)
    test_brier = brier_score_loss(y_test_series, test_prob)

    pred_030 = (test_prob >= 0.30).astype(int)
    test_precision_030 = precision_score(y_test_series, pred_030, zero_division=0)
    test_recall_030 = recall_score(y_test_series, pred_030, zero_division=0)
    test_f1_030 = f1_score(y_test_series, pred_030, zero_division=0)
    test_cm = confusion_matrix(y_test_series, pred_030)

    p100, r100 = top_n_metrics(y_test_series.values, test_prob, 100)
    p250, r250 = top_n_metrics(y_test_series.values, test_prob, 250)
    p500, r500 = top_n_metrics(y_test_series.values, test_prob, 500)
    p1000, r1000 = top_n_metrics(y_test_series.values, test_prob, 1000)

    print(f"\n  CHOSEN PARAMS: {best_val_params}")
    print(f"  VALIDATION WINNER -> AP={best_val_record['val_ap']:.4f}, P@100={best_val_record['val_p100']:.4f}, AUC={best_val_record['val_auc']:.4f}")
    print(f"  TEST -> AUC={test_auc:.4f}, AP={test_ap:.4f}, P@100={p100:.4f}, P@250={p250:.4f}, P@500={p500:.4f}")

    results.append({
        "experiment_name": exp_name,
        "selected_params_json": json.dumps(best_val_params),
        "selected_iterations": final_iterations,
        "val_auc": best_val_record["val_auc"],
        "val_ap": best_val_record["val_ap"],
        "val_p100": best_val_record["val_p100"],
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "test_brier": float(test_brier),
        "test_precision_at_030": float(test_precision_030),
        "test_recall_at_030": float(test_recall_030),
        "test_f1_at_030": float(test_f1_030),
        "test_precision_at_100": float(p100),
        "test_recall_at_100": float(r100),
        "test_precision_at_250": float(p250),
        "test_recall_at_250": float(r250),
        "test_precision_at_500": float(p500),
        "test_recall_at_500": float(r500),
        "test_precision_at_1000": float(p1000),
        "test_recall_at_1000": float(r1000),
        "confusion_matrix_json": json.dumps(test_cm.tolist())
    })

    trained_final_models[exp_name] = {
        "model": final_model,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols
    }

results_df = pd.DataFrame(results).sort_values(
    ["val_ap", "val_p100", "val_auc"],
    ascending=False
).reset_index(drop=True)

print(f"\n{'='*60}")
print("FINAL EXPERIMENT RANKING (SELECTED BY VALIDATION ONLY)")
print(f"{'='*60}")
print(results_df[[
    "experiment_name",
    "val_ap",
    "val_p100",
    "val_auc",
    "test_auc",
    "test_ap",
    "test_precision_at_100",
    "test_precision_at_250",
    "test_precision_at_500"
]])

best_experiment = results_df.loc[0, "experiment_name"]
print(f"\nBest experiment selected honestly: {best_experiment}")

best_model_info = trained_final_models[best_experiment]
best_model = best_model_info["model"]
best_feature_cols = best_model_info["feature_cols"]
best_cat_cols = best_model_info["cat_cols"]

full_X = df[best_feature_cols].copy()
full_y = df[TARGET].astype(int)
full_cat_idx = [best_feature_cols.index(c) for c in best_cat_cols]
full_pool = Pool(full_X, full_y, cat_features=full_cat_idx)

best_model.fit(full_pool, verbose=False)
full_prob = best_model.predict_proba(full_pool)[:, 1]

scored_df = df[[
    "patient_id",
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
    "diag_group",
    "risk_score",
    "risk_segment"
]].copy()

scored_df["final_experiment_name"] = best_experiment
scored_df["ml_risk_probability_final"] = full_prob
scored_df = scored_df.sort_values("ml_risk_probability_final", ascending=False).reset_index(drop=True)

scored_df["intervention_tier_final"] = "Tier 4 - Monitor"
scored_df.loc[:99, "intervention_tier_final"] = "Tier 1 - Immediate Action"
scored_df.loc[100:599, "intervention_tier_final"] = "Tier 2 - Care Manager Queue"
scored_df.loc[600:2599, "intervention_tier_final"] = "Tier 3 - Targeted Outreach"

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS ml_experiment_results_real;")
cur.execute("""
CREATE TABLE ml_experiment_results_real (
    experiment_name VARCHAR(100),
    selected_params_json TEXT,
    selected_iterations INT,
    val_auc FLOAT,
    val_ap FLOAT,
    val_p100 FLOAT,
    test_auc FLOAT,
    test_ap FLOAT,
    test_brier FLOAT,
    test_precision_at_030 FLOAT,
    test_recall_at_030 FLOAT,
    test_f1_at_030 FLOAT,
    test_precision_at_100 FLOAT,
    test_recall_at_100 FLOAT,
    test_precision_at_250 FLOAT,
    test_recall_at_250 FLOAT,
    test_precision_at_500 FLOAT,
    test_recall_at_500 FLOAT,
    test_precision_at_1000 FLOAT,
    test_recall_at_1000 FLOAT,
    confusion_matrix_json TEXT
);
""")
conn.commit()

cur.executemany("""
INSERT INTO ml_experiment_results_real VALUES (
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
)
""", list(results_df[[
    "experiment_name",
    "selected_params_json",
    "selected_iterations",
    "val_auc",
    "val_ap",
    "val_p100",
    "test_auc",
    "test_ap",
    "test_brier",
    "test_precision_at_030",
    "test_recall_at_030",
    "test_f1_at_030",
    "test_precision_at_100",
    "test_recall_at_100",
    "test_precision_at_250",
    "test_recall_at_250",
    "test_precision_at_500",
    "test_recall_at_500",
    "test_precision_at_1000",
    "test_recall_at_1000",
    "confusion_matrix_json"
]].itertuples(index=False, name=None)))
conn.commit()

cur.execute("DROP TABLE IF EXISTS ml_final_predictions_real;")
cur.execute("""
CREATE TABLE ml_final_predictions_real (
    patient_id BIGINT,
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
    risk_score FLOAT,
    risk_segment VARCHAR(20),
    final_experiment_name VARCHAR(100),
    ml_risk_probability_final FLOAT,
    intervention_tier_final VARCHAR(50)
);
""")
conn.commit()

cur.executemany("""
INSERT INTO ml_final_predictions_real VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
""", list(scored_df[[
    "patient_id",
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
    "diag_group",
    "risk_score",
    "risk_segment",
    "final_experiment_name",
    "ml_risk_probability_final",
    "intervention_tier_final"
]].itertuples(index=False, name=None)))
conn.commit()

cur.close()
conn.close()

print(f"\n{'='*60}")
print("SAVED TABLES")
print(f"{'='*60}")
print("- ml_experiment_results_real")
print("- ml_final_predictions_real")
print(f"\nFINAL HONEST WINNER: {best_experiment}")