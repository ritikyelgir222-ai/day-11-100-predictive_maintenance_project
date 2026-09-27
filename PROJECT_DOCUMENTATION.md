# Project Documentation: Sensor Anomaly Detection / Predictive Maintenance
### Technical Documentation & Handover — Phase 12 of the SDLC

Companion to `SDLC_DOCUMENTATION.md` and `README.md`.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Prioritize a maintenance inspection queue using sensor readings, before failures happen |
| **Client context** | Manufacturing (milling machine operations) |
| **Data source** | AI4I 2020 Predictive Maintenance Dataset — [UCI: archive.ics.uci.edu/dataset/601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| **Dataset size** | 10,000 readings, 339 failures (3.39%) |
| **Final model** | Random Forest (supervised), compared against an Isolation Forest (unsupervised) baseline |
| **Test average precision** | 0.9442 (supervised) vs. 0.2298 (unsupervised) |
| **Business framing** | Rank all machines by failure risk, inspect the top N per period |

---

## 2. Architecture

```
data/ai4i2020.csv (10,000 readings, 5 sensors + Type + failure labels)
        │
        ▼
data_loader.py ──────► loads raw data, documents the 5 failure sub-labels
        │
        ▼
clean_and_engineer.py ─► DROPS UDI/Product ID, EXCLUDES TWF/HDF/PWF/OSF/RNF
        │                (near-perfect leakage), engineers 3 physics-
        │                grounded features (temp_diff_K, power_W,
        │                torque_wear_product)
        ▼
   split.py ───────────► stratified train/val/test split
        │
        ▼
train_model.py ───────► TWO PARALLEL APPROACHES:
        │                (1) Isolation Forest — unsupervised, no label
        │                (2) LogReg/RF/XGBoost — supervised, uses label
        │                Both evaluated on the same test set for a
        │                direct, honest comparison
        ▼
outputs/maintenance_model.joblib (the SUPERVISED model — what gets deployed)
        │
        ├──────────────► app.py ─── FastAPI real-time scoring service
        │
        └──────────────► monitor.py ─ near-continuous drift/performance check
```

---

## 3. Data Dictionary

| Column | Type | Description | Treatment |
|---|---|---|---|
| `UDI` | int | Row index | **Dropped** — no physical meaning |
| `Product ID` | string | Per-unit serial number | **Dropped** — unique identifier, leakage risk |
| `Type` | categorical (L/M/H) | Product quality variant | One-hot encoded |
| `Air temperature [K]` | float | Ambient temperature | Kept as-is |
| `Process temperature [K]` | float | Machine process temperature | Kept as-is |
| `Rotational speed [rpm]` | int | Motor speed | Kept as-is |
| `Torque [Nm]` | float | Applied torque | Kept as-is |
| `Tool wear [min]` | int | Cumulative tool usage | Kept as-is |
| `Machine failure` | int (0/1) | **Target** | — |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | int (0/1) | Failure-mode sub-labels | **EXCLUDED — near-perfect leakage** (these literally define the target) |

**Engineered features (all physics-grounded, not guessed):**

| Feature | Formula | Physical rule it targets |
|---|---|---|
| `temp_diff_K` | Process temp − Air temp | Heat Dissipation Failure (fails if < 8.6 K, combined with low rpm) |
| `power_W` | Torque × Rotational speed (converted to rad/s) | Power Failure (fails if outside 3500–9000 W) |
| `torque_wear_product` | Torque × Tool wear | Overstrain Failure (fails if exceeds a Type-dependent threshold) |

---

## 4. Key EDA Findings

| Finding | Detail |
|---|---|
| Overall failure rate | 3.39% (339 of 10,000) — Day 2-level imbalance |
| Failure rate by Type | L: 3.92% (highest), M: 2.77%, H: 2.09% (lowest) — consistent with L having the LOWEST overstrain threshold (easiest to trigger OSF) |
| `power_W` failure vs. non-failure | Failures cluster in **both tails**: min 1148W (below the 3500W floor) and max 10470W (above the 9000W ceiling) — a textbook confirmation of the documented two-sided PWF rule |
| `torque_wear_product` failure vs. non-failure | Mean 7188 (failures) vs. 4214 (non-failures) — strong separation, consistent with the OSF rule |
| `temp_diff_K` | Only mildly lower for failures (9.40 vs. 10.02) — weaker single-variable signal than expected, since the real HDF rule requires BOTH low temp_diff AND low rotational speed jointly; a univariate EDA view understates a two-condition rule |

---

## 5. Modeling Results

### Approach 1: Unsupervised (Isolation Forest, no label used in training)

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.8345 |
| Test average precision | 0.2298 |
| Precision @ top-50 queue | 0.24 |
| Recall @ top-50 queue | 0.24 |

### Approach 2: Supervised (validation set)

| Model | Val ROC-AUC | Val Average Precision |
|---|---|---|
| Logistic Regression (baseline) | 0.8847 | 0.3529 |
| **Random Forest** | 0.9547 | **0.8440** |
| XGBoost | 0.9505 | 0.8267 |

**A different pattern from Days 1, 7, and 9:** here Random Forest clearly
wins, and by a wide margin over logistic regression (0.844 vs. 0.353) —
the opposite of the "simple model wins" pattern seen repeatedly earlier
in the series. This makes sense given the domain: the real failure rules
(power outside a two-sided range, torque × wear exceeding a threshold)
are inherently non-linear and involve interactions a linear model
structurally cannot represent well, while tree ensembles handle them
naturally.

### Test set: head-to-head comparison (the core Day 11 result)

| | Unsupervised (Isolation Forest) | Supervised (Random Forest) |
|---|---|---|
| Test ROC-AUC | 0.8345 | **0.9881** |
| Test average precision | 0.2298 | **0.9442** |
| Precision @ top-50 queue | 0.24 | **0.90** |
| Recall @ top-50 queue | 0.24 | **0.88** |

Inspecting the same 50 machines per period: the unsupervised approach
catches about 1 in 4 actual failures; the supervised approach catches
about 9 in 10. This quantifies, in concrete operational terms, the value
of accumulating labeled failure history.

### What drives the supervised model

Top feature importances: `Rotational speed [rpm]` (0.1997), `Torque [Nm]`
(0.1953), `power_W` (0.1737), `Tool wear [min]` (0.1317),
`torque_wear_product` (0.1213) — the two engineered physics features
rank 3rd and 5th, confirming they add real, independent signal beyond
the raw sensor readings they're derived from.

---

## 6. API Reference

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser test form |
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | Health check |
| `/score` | POST | Score a single sensor reading |

**Response:**
```json
{
  "failure_probability": 0.9472,
  "flagged_for_inspection": true,
  "top_factors": ["Rotational speed [rpm]", "Torque [Nm]", "power_W"]
}
```

Verified with two constructed readings: a high-overstrain-risk reading
(Type L, high torque, high tool wear) scored 0.9472; a normal reading
(Type H, moderate values) scored 0.0017 — correctly separated.

**Note on the fixed 0.5 threshold:** unlike Day 7's tuned cost-optimal
threshold, this project's business framing (Phase 9) is queue-based —
rank ALL machines and inspect the top N per period — so a single API
call scoring one machine in isolation has no queue to rank within. The
0.5 cutoff here is a simple secondary flag; the real operational
decision happens by ranking many scores together.

---

## 7. Monitoring & Maintenance Plan

`monitor.py` checks PSI drift on the three physics-grounded features
(chosen because drift there is directly interpretable — e.g. "power
readings have shifted" is actionable in a way that drift on an
anonymized feature wouldn't be) and average-precision decay.

**Recommended cadence: near-continuous**, similar category to Day 2's
fraud monitoring but for a different reason — Day 2 needed speed because
fraud is adversarial; this system can and should move fast because
machine failure outcomes are usually known almost immediately after a
reading, unlike Day 7's loans (months-long outcome lag).

---

## 8. Known Limitations

1. **Synthetic-but-physics-modeled data**, not real factory sensor readings — needs revalidation before real deployment.
2. **RNF (random failures, ~0.1% of all cases) sets a hard ceiling on achievable recall** — some failures are, by the dataset's own construction, unpredictable from any sensor reading. No model, however good, can exceed this ceiling.
3. **The Isolation Forest's `contamination` parameter used the training set's own known failure rate** — a real cold-start plant wouldn't have this exact number, so the unsupervised approach's reported performance may be mildly optimistic relative to a genuinely blind start.
4. **The model flags elevated risk but not which failure mode is likely** — TWF/HDF/PWF/OSF were correctly excluded as leakage, which also means that diagnostic detail is unavailable to the deployed model. This is a real trade-off, not a free win.
5. **Single static snapshot** — no genuine time-based validation was possible or attempted, similar to Day 1/7/9.

---

## 9. File Map

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, document the failure sub-labels |
| `eda.py` | 6 | Validate physics-grounded features against real failures |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Critical leakage exclusion, physics-based feature engineering |
| `split.py` | 7 | Stratified split |
| `train_model.py` | 8–9 | Unsupervised vs. supervised head-to-head comparison |
| `app.py` | 10 | FastAPI real-time scoring service |
| `monitor.py` | 13 | Near-continuous drift/performance monitoring |
| `README.md` | 12 | Setup instructions, unsupervised-vs-supervised business case |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation |
| `SDLC_DOCUMENTATION.md` | 1–14 | Full 14-phase narrative |
