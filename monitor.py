"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY THE RIGHT CADENCE HERE IS NEAR-CONTINUOUS, LIKE DAY 2's FRAUD MODEL
(not monthly like Day 1, not once-per-term like Day 9): sensors emit
readings continuously, and unlike Day 7's loan model (where a default
outcome isn't known for months), a machine failure outcome is usually
known almost immediately — the machine either fails shortly after a
reading or it doesn't. This means genuinely fast feedback is available,
making frequent monitoring both possible AND valuable, much like fraud's
adversarial cadence, but for a different underlying reason (rapid ground-
truth availability rather than adversarial adaptation).
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import average_precision_score

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data

AVG_PRECISION_DROP_THRESHOLD = 0.15  # retrain if average precision drops more than this


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/maintenance_model.joblib")

    raw = load_raw_data()
    engineered = engineer_features(clean_data(raw))
    feature_cols = get_feature_columns(engineered)
    X = engineered[feature_cols]
    y = engineered["Machine failure"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("=== Feature Drift (PSI): key sensor readings, train vs. test ===")
    for feat in ["power_W", "torque_wear_product", "Tool wear [min]"]:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")
    # WHY THESE THREE SPECIFICALLY: they're the physics-grounded
    # engineered features most directly tied to the known failure modes
    # (PWF, OSF) — drift here is the most actionable signal that
    # something about the actual production process has changed (a new
    # tool supplier, a recalibrated sensor, a process change), as
    # opposed to drift on a raw individual sensor reading which is
    # harder to interpret on its own.

    scores = model.predict_proba(X_test)[:, 1]
    avg_precision = average_precision_score(y_test, scores)

    print(f"\n=== Performance on test batch ===")
    print(f"Average precision: {avg_precision:.4f}")

    with open("outputs/business_validation.json") as f:
        baseline = json.load(f)
    baseline_avg_precision = baseline["supervised_best_model"]["test_avg_precision"]
    drop = baseline_avg_precision - avg_precision

    if drop > AVG_PRECISION_DROP_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: average precision dropped by {drop:.3f} (threshold: {AVG_PRECISION_DROP_THRESHOLD})")
    else:
        print(f"\n✅ No retrain needed (change: {drop:+.3f}, threshold: {AVG_PRECISION_DROP_THRESHOLD})")

    print("\n=== Recommended monitoring cadence ===")
    print("Near-continuous / shift-based, similar cadence category to Day 2's")
    print("fraud monitoring — but for a different underlying reason. Fraud")
    print("needs fast monitoring because fraudsters ADAPT (adversarial drift).")
    print("This system needs fast monitoring because machine failure ground")
    print("truth becomes available almost immediately after a reading (unlike")
    print("Day 7's loan defaults, which take months to mature) — fast feedback")
    print("makes fast monitoring both POSSIBLE and worth doing, not just useful.")


if __name__ == "__main__":
    run_monitoring_check()
