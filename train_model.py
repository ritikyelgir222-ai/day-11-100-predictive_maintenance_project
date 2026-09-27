"""
Phase 8: Model Development & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
THE CENTRAL METHODOLOGICAL QUESTION FOR "SENSOR ANOMALY DETECTION,"
ADDRESSED DIRECTLY: this dataset happens to include historical failure
labels, which makes SUPERVISED classification the stronger choice once
those labels exist — but a brand-new plant with no failure history yet
cannot train a supervised model at all. This script builds and honestly
compares BOTH approaches:

  1. An UNSUPERVISED anomaly detector (Isolation Forest) trained WITHOUT
     ever seeing the failure label — representing the cold-start
     scenario a new production line would actually face.
  2. SUPERVISED classifiers (logistic regression, Random Forest, XGBoost)
     trained WITH the failure label — representing the mature scenario
     once enough failure history has accumulated.

The comparison itself is the point: it shows concretely how much
predictive power is left on the table by not having labels yet, which
is a genuine, quantifiable business case for why a plant should
invest in careful failure logging from day one.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data

CAT_COLS = ["Type"]
NUM_COLS = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]", "temp_diff_K", "power_W", "torque_wear_product"]


def precision_recall_at_k(y_true, y_score, k):
    """Same 'maintenance inspection queue' logic as Day 2's fraud review
    queue: with only ~51 failures in the whole test set, a percentage-
    of-population threshold isn't realistic — an absolute, capacity-
    based queue size is the right framing."""
    y_true = np.asarray(y_true)
    top_idx = np.argsort(y_score)[-k:]
    flagged_true = y_true[top_idx]
    precision = flagged_true.sum() / k
    recall = flagged_true.sum() / y_true.sum() if y_true.sum() > 0 else 0.0
    return precision, recall


def build_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])


def train_and_evaluate():
    raw = load_raw_data()
    engineered = engineer_features(clean_data(raw))
    feature_cols = get_feature_columns(engineered)

    X = engineered[feature_cols]
    y = engineered["Machine failure"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = build_preprocessor()

    # =====================================================================
    # APPROACH 1: UNSUPERVISED — Isolation Forest, no failure label used
    # =====================================================================
    # WHY contamination=0.034: Isolation Forest needs an estimate of what
    # fraction of the data is anomalous. In a genuine cold-start scenario
    # this would be a rough guess from domain expertise (e.g. "we expect
    # maybe 3-5% of readings to be anomalous based on industry norms");
    # here we use the TRAINING SET's known failure rate as a stand-in for
    # that expert guess, since a real new plant wouldn't have this exact
    # number either — this is disclosed as a simplification, not treated
    # as free information the model "shouldn't" have.
    iso_preprocessor = build_preprocessor()
    X_train_unsup = iso_preprocessor.fit_transform(X_train)
    X_test_unsup = iso_preprocessor.transform(X_test)

    contamination_estimate = y_train.mean()
    iso_forest = IsolationForest(contamination=contamination_estimate, random_state=42, n_estimators=200)
    iso_forest.fit(X_train_unsup)
    # decision_function: higher = more normal, lower = more anomalous.
    # Flip sign so higher = more anomalous, consistent with the
    # supervised models' "higher score = more likely to fail" convention.
    iso_test_scores = -iso_forest.decision_function(X_test_unsup)
    iso_avg_precision = average_precision_score(y_test, iso_test_scores)
    iso_roc_auc = roc_auc_score(y_test, iso_test_scores)

    print("=== Approach 1: Unsupervised Isolation Forest (no label used in training) ===")
    print(f"Test ROC-AUC: {iso_roc_auc:.4f}")
    print(f"Test average precision: {iso_avg_precision:.4f}")

    # =====================================================================
    # APPROACH 2: SUPERVISED — logistic regression, random forest, XGBoost
    # =====================================================================
    logreg_pipe = Pipeline([("prep", preprocessor),
                             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    logreg_pipe.fit(X_train, y_train)
    logreg_val_scores = logreg_pipe.predict_proba(X_val)[:, 1]

    rf_pipe = Pipeline([("prep", build_preprocessor()),
                         ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                                          class_weight="balanced", random_state=42))])
    rf_pipe.fit(X_train, y_train)
    rf_val_scores = rf_pipe.predict_proba(X_val)[:, 1]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pipe = Pipeline([("prep", build_preprocessor()),
                          ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  scale_pos_weight=scale_pos_weight,
                                                  eval_metric="aucpr", random_state=42))])
    xgb_pipe.fit(X_train, y_train)
    xgb_val_scores = xgb_pipe.predict_proba(X_val)[:, 1]

    experiment_log = []
    for name, scores in [("logistic_regression (baseline)", logreg_val_scores),
                           ("random_forest", rf_val_scores),
                           ("xgboost", xgb_val_scores)]:
        experiment_log.append({
            "model": name,
            "val_roc_auc": round(roc_auc_score(y_val, scores), 4),
            "val_avg_precision": round(average_precision_score(y_val, scores), 4),
        })

    log_df = pd.DataFrame(experiment_log)
    print("\n=== Approach 2: Supervised models (validation set) ===")
    print(log_df.to_string(index=False))
    log_df.to_csv("outputs/experiment_log.csv", index=False)

    best_model_name = log_df.loc[log_df["val_avg_precision"].idxmax(), "model"]
    print(f"\nBest supervised model by average precision: {best_model_name}")
    final_pipe = {"logistic_regression (baseline)": logreg_pipe, "random_forest": rf_pipe, "xgboost": xgb_pipe}[best_model_name]

    # -----------------------------------------------------------------
    # Phase 9: final evaluation + head-to-head comparison
    # -----------------------------------------------------------------
    test_scores = final_pipe.predict_proba(X_test)[:, 1]
    test_avg_precision = average_precision_score(y_test, test_scores)
    test_roc_auc = roc_auc_score(y_test, test_scores)

    MAINTENANCE_QUEUE_SIZE = 50  # a plausible number of machines a maintenance team can inspect per period
    sup_precision, sup_recall = precision_recall_at_k(y_test.values, test_scores, MAINTENANCE_QUEUE_SIZE)
    unsup_precision, unsup_recall = precision_recall_at_k(y_test.values, iso_test_scores, MAINTENANCE_QUEUE_SIZE)

    comparison = {
        "maintenance_queue_size": MAINTENANCE_QUEUE_SIZE,
        "n_test_machines": len(y_test),
        "n_actual_failures_in_test": int(y_test.sum()),
        "unsupervised_isolation_forest": {
            "test_roc_auc": round(iso_roc_auc, 4),
            "test_avg_precision": round(iso_avg_precision, 4),
            "precision_at_queue": round(unsup_precision, 4),
            "recall_at_queue": round(unsup_recall, 4),
        },
        "supervised_best_model": {
            "model": best_model_name,
            "test_roc_auc": round(test_roc_auc, 4),
            "test_avg_precision": round(test_avg_precision, 4),
            "precision_at_queue": round(sup_precision, 4),
            "recall_at_queue": round(sup_recall, 4),
        },
        "note": (
            "The unsupervised model never saw the failure label during "
            "training (contamination rate estimated from the training "
            "set's own failure rate, standing in for an expert guess a "
            "real cold-start plant would have to make). The gap between "
            "the two approaches quantifies the real business value of "
            "accumulating labeled failure history before switching to a "
            "supervised model."
        ),
    }

    print("\n=== Business Validation: Unsupervised vs. Supervised head-to-head (Phase 9) ===")
    print(json.dumps(comparison, indent=2))
    with open("outputs/business_validation.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # -----------------------------------------------------------------
    # Explainability (Phase 9) — supervised model only (Isolation Forest
    # doesn't have comparable feature importances in the same sense)
    # -----------------------------------------------------------------
    ohe_names = final_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CAT_COLS)
    all_names = list(ohe_names) + NUM_COLS
    clf = final_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=all_names).sort_values(ascending=False)
        print("\n=== Feature Importance (supervised model, top 8) ===")
        print(importances.head(8).round(4).to_string())
        importances.to_csv("outputs/feature_importance.csv", header=["importance"])
    else:
        coefs = pd.Series(clf.coef_[0], index=all_names).sort_values(key=abs, ascending=False)
        print("\n=== Logistic Regression Coefficients (top 8 by magnitude) ===")
        print(coefs.head(8).round(4).to_string())
        coefs.to_csv("outputs/feature_importance.csv", header=["coefficient"])

    # -----------------------------------------------------------------
    # Save artifacts (Phase 10 prep) — the SUPERVISED model is what gets
    # deployed, since this project has failure labels available (the
    # unsupervised model is retained conceptually/in this script as the
    # honest cold-start comparison, not as the production artifact)
    # -----------------------------------------------------------------
    joblib.dump(final_pipe, "outputs/maintenance_model.joblib")
    joblib.dump({"cat_cols": CAT_COLS, "num_cols": NUM_COLS}, "outputs/model_config.joblib")

    with open("outputs/model_card.json", "w") as f:
        json.dump({
            "model_type": best_model_name,
            "data_source": "AI4I 2020 Predictive Maintenance Dataset (UCI: archive.ics.uci.edu/dataset/601)",
            "n_features": len(CAT_COLS) + len(NUM_COLS),
            "training_rows": len(X_train),
            "test_avg_precision": round(test_avg_precision, 4),
            "intended_use": "Score machine sensor readings to prioritize a maintenance inspection queue.",
            "known_limitations": (
                "TWF/HDF/PWF/OSF/RNF sub-labels were correctly excluded as "
                "leakage, but this also means the model cannot distinguish "
                "WHICH failure mode it's predicting, only that failure risk "
                "is elevated — a real information loss relative to a system "
                "that could also flag the likely failure type. RNF (random "
                "failures, ~0.1% of all cases, independent of any sensor "
                "reading) sets a hard ceiling on achievable recall no model "
                "can exceed, since some failures are irreducibly random."
            ),
        }, f, indent=2)

    print("\nSaved model -> outputs/maintenance_model.joblib")
    print("Saved model card -> outputs/model_card.json")


if __name__ == "__main__":
    train_and_evaluate()
