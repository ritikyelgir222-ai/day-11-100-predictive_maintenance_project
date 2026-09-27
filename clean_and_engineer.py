"""
Phases 5 (data quality) & 7 (feature engineering)
------------------------------------------------------
THE MOST IMPORTANT DECISION IN THIS PROJECT, EVEN MORE CLEAR-CUT THAN
DAY 9's G2 EXCLUSION: TWF, HDF, PWF, OSF, and RNF are not just
correlated with the target — they are the exact five deterministic
sub-conditions used to CONSTRUCT the "Machine failure" label in the
first place ("if at least one of the above failure modes is true, the
process fails"). Using any of them as a model input would not be subtle
leakage requiring careful judgment (like Day 9's G2) — it would be
almost literally handing the model the answer key. All five are
excluded from the feature set entirely, with zero exceptions.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Columns dropped, and why
# ---------------------------------------------------------------------------
# UDI: a row index (1 to 10000) — no physical meaning, would only let a
#      model memorize row order rather than learn generalizable sensor
#      relationships.
# Product ID: a per-unit serial number (e.g. "M14860") — unique per row,
#      same leakage/overfitting risk as Day 1's customerID.
COLUMNS_TO_DROP = ["UDI", "Product ID"]

# TWF, HDF, PWF, OSF, RNF: THE five deterministic sub-labels that define
# "Machine failure" by construction. See module docstring above.
FAILURE_SUBLABELS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.drop(columns=COLUMNS_TO_DROP)
    # WHY WE VERIFY NO MISSING VALUES: this dataset is documented as
    # complete/synthetic-clean, but we check rather than assume.
    assert df.isnull().sum().sum() == 0, "Unexpected missing values found"
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------------------------------------------
    # Engineered feature 1: temp_diff_K
    # PHYSICS-GROUNDED LOGIC (not a guess): the dataset's own
    # documentation defines Heat Dissipation Failure (HDF) as occurring
    # when (process temperature - air temperature) < 8.6 K AND
    # rotational speed < 1380 rpm. We engineer the exact quantity
    # (temp_diff_K) that the first half of this real physical rule
    # depends on, rather than hoping the model rediscovers it from the
    # two raw temperature columns separately.
    # -----------------------------------------------------------------
    df["temp_diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # -----------------------------------------------------------------
    # Engineered feature 2: power_W
    # PHYSICS-GROUNDED LOGIC: Power Failure (PWF) is defined as
    # occurring when torque x rotational speed (converted to rad/s)
    # falls outside 3500-9000 W. Rotational speed is given in rpm, so
    # converting to rad/s (x 2*pi/60) before multiplying by torque
    # reconstructs actual mechanical power in watts — the literal
    # physical quantity the failure rule is defined on.
    # -----------------------------------------------------------------
    rpm_to_rad_per_s = 2 * np.pi / 60
    df["power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * rpm_to_rad_per_s

    # -----------------------------------------------------------------
    # Engineered feature 3: torque_wear_product
    # PHYSICS-GROUNDED LOGIC: Overstrain Failure (OSF) is defined as
    # occurring when (tool wear x torque) exceeds a type-dependent
    # threshold (11,000 for L, 12,000 for M, 13,000 for H). We compute
    # the exact product the rule depends on; the type-dependent
    # threshold itself is left for the model to learn via the one-hot
    # encoded Type columns interacting with this feature, rather than
    # hardcoding the three threshold values directly into the feature
    # (which would risk baking in the documentation's stated thresholds
    # even if this particular data sample doesn't exactly follow them).
    # -----------------------------------------------------------------
    df["torque_wear_product"] = df["Torque [Nm]"] * df["Tool wear [min]"]

    return df


def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    exclude = {"Machine failure"} | set(FAILURE_SUBLABELS)
    return [c for c in engineered_df.columns if c not in exclude]


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    print(f"Raw columns: {len(raw.columns)}  ->  Final feature columns: {len(feature_cols)}")
    print("\nConfirming failure sub-labels are excluded:")
    for label in FAILURE_SUBLABELS:
        print(f"  '{label}' in features: {label in feature_cols}")
    print(f"\nFinal feature columns: {feature_cols}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
