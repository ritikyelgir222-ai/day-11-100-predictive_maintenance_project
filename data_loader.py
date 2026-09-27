"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the "AI4I 2020 Predictive Maintenance Dataset",
published by S. Matzka (2020) to the UCI Machine Learning Repository:
    https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

10,000 data points from a simulated (but physically-modeled, not
arbitrary) milling machine, with five real sensor readings — air
temperature, process temperature, rotational speed, torque, and tool
wear — plus a binary Machine failure label and five underlying failure
MODE sub-labels (TWF, HDF, PWF, OSF, RNF) that were used to construct
that label deterministically (see clean_and_engineer.py for why this
matters enormously for feature selection).

WHY THIS DATASET FOR DAY 11 (Sensor Anomaly Detection, Manufacturing)
--------------------------------------------------------------------
- It's built around genuine physical failure mechanisms (heat
  dissipation limits, power thresholds, tool overstrain) rather than an
  arbitrary label, so the feature engineering in this project can be
  grounded in real physics, not guesswork.
- It's EXTREMELY imbalanced (3.39% failure rate) — a similar order of
  magnitude to Day 2's fraud problem, but for a completely different
  reason (rare mechanical failure vs. rare adversarial fraud), which
  matters for how monitoring cadence and business framing differ later.
- It comes with an explicit historical failure label, which makes this
  project a good place to directly compare TWO fundamentally different
  approaches to "anomaly detection": training a SUPERVISED classifier
  using the known failure history (what most real predictive-maintenance
  systems eventually do), versus a genuinely UNSUPERVISED anomaly
  detector that uses no failure labels at all (what a brand-new plant
  with no failure history yet would have to start with). Both are built
  and compared in train_model.py.

If you're following along on UCI/Kaggle: download "ai4i2020.csv" from
the link above and place it at `data/ai4i2020.csv` — the schema is
identical to the file already included in this project.
"""

import pandas as pd

RAW_DATA_PATH = "data/ai4i2020.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")  # source file has a BOM on the first column
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} machine readings, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"Failure rate: {df['Machine failure'].mean():.2%}")
    print(f"Failure mode breakdown: TWF={df['TWF'].sum()}, HDF={df['HDF'].sum()}, "
          f"PWF={df['PWF'].sum()}, OSF={df['OSF'].sum()}, RNF={df['RNF'].sum()}")
