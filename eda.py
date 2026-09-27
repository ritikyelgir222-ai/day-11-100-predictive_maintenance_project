"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: this EDA exists specifically to check whether
the physics-grounded engineered features actually behave the way the
dataset's own documented failure rules say they should — a direct
validation of Phase 7's reasoning using real data, not just theory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features


def run_eda(out_dir: str = "outputs"):
    raw = load_raw_data()
    df = engineer_features(clean_data(raw))

    print("=== Overall failure rate ===")
    print(f"{df['Machine failure'].mean():.2%} ({df['Machine failure'].sum()} of {len(df)})")
    # WHY WE STATE THIS LOUDLY: at 3.39%, this is Day 2-level imbalance —
    # accuracy is meaningless here for the same reason it was there.

    print("\n=== Failure rate by product Type (L/M/H quality variant) ===")
    print(df.groupby("Type")["Machine failure"].mean().round(4))
    # WHY: the dataset's own OSF rule has a DIFFERENT overstrain
    # threshold per Type (11,000/12,000/13,000 for L/M/H) — checking
    # whether failure rate actually varies by Type validates that this
    # isn't just a documentation detail with no real effect on the data.

    print("\n=== temp_diff_K: failure vs. no failure ===")
    print(df.groupby("Machine failure")["temp_diff_K"].describe()[["mean", "50%", "min"]])
    # WHY: HDF's rule requires temp_diff_K < 8.6 — checking whether
    # failures cluster below that value validates the engineered
    # feature actually carries the signal the physics rule predicts.

    print("\n=== power_W: failure vs. no failure ===")
    print(df.groupby("Machine failure")["power_W"].describe()[["mean", "50%", "min", "max"]])
    # WHY: PWF's rule requires power outside 3500-9000 W — checking
    # whether failures show a wider or shifted power_W range validates
    # this feature the same way.

    print("\n=== torque_wear_product: failure vs. no failure ===")
    print(df.groupby("Machine failure")["torque_wear_product"].describe()[["mean", "50%", "max"]])

    # ---- Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    df.boxplot(column="temp_diff_K", by="Machine failure", ax=axes[0])
    axes[0].set_title("temp_diff_K by failure status")
    axes[0].axhline(8.6, color="red", linestyle="--", linewidth=1)  # the documented HDF threshold

    df.boxplot(column="power_W", by="Machine failure", ax=axes[1])
    axes[1].set_title("power_W by failure status")
    axes[1].axhline(3500, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(9000, color="red", linestyle="--", linewidth=1)

    df.boxplot(column="torque_wear_product", by="Machine failure", ax=axes[2])
    axes[2].set_title("torque_wear_product by failure status")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    run_eda()
