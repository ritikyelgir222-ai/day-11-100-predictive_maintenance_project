# Sensor Anomaly Detection / Predictive Maintenance — Day 11 (Manufacturing, ML)

Full project code for Day 11 of the 100-day series. Uses the real
**AI4I 2020 Predictive Maintenance Dataset** (UCI: archive.ics.uci.edu/dataset/601,
Matzka 2020): 10,000 real-physics-modeled milling machine readings, 339
failures (3.39% — Day 2-level imbalance).

## The central question this project answers directly

"Sensor Anomaly Detection" could mean two very different things:
1. **No failure history exists yet** (a new production line) → genuinely
   unsupervised anomaly detection
2. **Failure history exists** (this dataset's case) → supervised
   classification, which is almost always stronger once available

This project builds **both** and compares them head-to-head:

| Approach | Test Average Precision | Precision @ top-50 queue | Recall @ top-50 queue |
|---|---|---|---|
| Isolation Forest (unsupervised, no label used) | 0.2298 | 0.24 | 0.24 |
| **Random Forest (supervised, uses failure history)** | **0.9442** | **0.90** | **0.88** |

That gap is the concrete, quantified business case for why a plant
should invest in careful failure logging from day one — going from
unsupervised to supervised roughly **quadruples** the average precision.

## The most clear-cut leakage exclusion in the series so far

This dataset includes TWF/HDF/PWF/OSF/RNF — five sub-labels that are the
literal deterministic conditions used to construct the `Machine failure`
target ("if at least one of these is true, the process fails"). Unlike
Day 9's more subtle G2-timing judgment call, there's no nuance here:
using any of these as a feature would be almost literally handing the
model the answer key. All five are excluded, no exceptions.

## Physics-grounded feature engineering (not guesswork)

Three engineered features are built directly from the dataset's own
documented failure-mode formulas:

- `temp_diff_K` — the exact quantity the Heat Dissipation Failure rule depends on
- `power_W` — mechanical power (torque × angular velocity), the exact quantity the Power Failure rule depends on
- `torque_wear_product` — the exact quantity the Overstrain Failure rule depends on

EDA confirms all three behave exactly as the physical rules predict —
see `eda.py`'s output, especially `power_W`'s failures clustering in
**both tails** (below 3500W and above 9000W), matching the documented
two-sided rule precisely.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py          # Phase 5 — loads real data, failure-mode breakdown
python eda.py                   # Phase 6 — validates the physics-grounded features against real failures
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — the critical sub-label exclusion, feature engineering
python train_model.py          # Phase 8-9 — Isolation Forest vs. supervised models, head-to-head
python monitor.py              # Phase 13 — near-continuous drift/performance monitoring
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/** for a test form.

## File map

| File | SDLC Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Loads real data, documents the failure-mode sub-labels |
| `eda.py` | 6 | Validates physics-grounded features against real failure data |
| `clean_and_engineer.py` | 5 (fixes) + 7 | The critical sub-label exclusion, physics-based feature engineering |
| `split.py` | 7 | Stratified split |
| `train_model.py` | 8–9 | Unsupervised vs. supervised head-to-head comparison |
| `app.py` | 10 | FastAPI real-time scoring service |
| `monitor.py` | 13 | Near-continuous monitoring (fast ground-truth availability) |

## Known limitations (stated honestly)

- Synthetic-but-physics-modeled data, not real factory floor sensor readings — a real deployment needs revalidation on actual production data.
- RNF (random failures, ~0.1% of cases, independent of any sensor reading) sets a hard ceiling on achievable recall no model can exceed.
- The unsupervised Isolation Forest's `contamination` parameter was estimated from the training set's own known failure rate — a real cold-start plant wouldn't have this exact number either, so its reported performance is somewhat optimistic relative to a true blind cold start.
- The model can flag elevated failure risk but not which specific failure mode (TWF/HDF/PWF/OSF) is likely, since those sub-labels were correctly excluded as leakage — a real information trade-off, not a free win.
