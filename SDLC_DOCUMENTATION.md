# SDLC Documentation: Sensor Anomaly Detection / Predictive Maintenance
### Manufacturing Industry | ML | Day 11 of the 100-Day Series
### Filled against the master 14-phase SDLC template, using the real project we built

Every number below comes from the actual pipeline run on the real UCI
AI4I 2020 Predictive Maintenance dataset.

---

## Phase 1: Discovery & Stakeholder Requirement Gathering
**Owner:** Business Analyst / Data Scientist | **Output:** Meeting notes, stakeholder map

- **Stakeholders identified:** plant maintenance manager (problem owner), operations director (budget approver), maintenance technicians (day-to-day users)
- **Discovery findings:**
  - Current process (assumed for this exercise): reactive maintenance — machines are inspected after a failure, or on a fixed calendar schedule regardless of actual condition
  - Decision this project informs: which machines get pulled for inspection THIS period, out of a fixed inspection capacity
  - Critical early framing question: does the plant already have failure history to learn from, or is this a new line with no labels yet? This dataset has labels, but the project deliberately answers BOTH versions of the question (see Phase 8)
- **Deliverable — Problem Statement:** "Prioritize a capacity-limited maintenance inspection queue using sensor readings, and quantify how much better this works once failure history exists versus a cold start with none."

---

## Phase 2: Business Requirement Document (BRD)
**Owner:** Business Analyst | **Output:** BRD

- **Business objective:** maximize failures caught within a fixed inspection capacity (50 machines per period)
- **Scope:** in-scope: single milling-machine process, all five documented failure modes combined into one overall risk score; out-of-scope: per-failure-mode diagnosis (see Phase 9's stated trade-off)
- **Success metrics / KPIs:** precision and recall at the fixed queue size, for both the supervised and unsupervised approaches, compared directly
- **Assumptions & constraints:** synthetic-but-physics-modeled data, not live factory data; RNF's ~0.1% random-failure rate sets a hard recall ceiling no model can beat
- **Sign-off:** N/A — self-directed portfolio project

---

## Phase 3: Functional & Technical Requirement Document (FRD/TRD)
**Owner:** Data Scientist / Tech Lead | **Output:** FRD/TRD

- **Functional requirements:** score sensor readings in real time as they arrive (unlike Day 1's nightly batch, closer to Day 2's real-time pattern); rank machines for a periodic inspection queue
- **Non-functional requirements:** sub-second scoring (confirmed in testing)
- **Data sources:** a single static CSV in this exercise; a real system would connect to a live sensor telemetry stream
- **Integration points:** none in this version — a production version would integrate with a plant's maintenance scheduling system

---

## Phase 4: Project Planning
**Owner:** Project/Delivery Manager | **Output:** Project plan, risk register

- **Build sequence used:** data loading → EDA (validating physics-grounded features) → feature engineering (the sub-label exclusion) → stratified split → parallel unsupervised/supervised modeling → API → monitoring → documentation
- **Real risks encountered during the build:**
  - Risk: the raw CSV file had a byte-order-mark (BOM) on its first column header, which would have silently broken column-name lookups (`UDI` would have loaded as `\ufeffUDI`) — caught and fixed immediately in `data_loader.py` via explicit `encoding="utf-8-sig"`, rather than discovered later as a confusing KeyError downstream
  - Risk: initial temptation to treat TWF/HDF/PWF/OSF/RNF as "just more features to try" — resolved by reading the dataset's own documentation carefully BEFORE writing any modeling code, which revealed they are the literal deterministic basis for the target

---

## Phase 5: Data Collection & Data Understanding
**Owner:** Data Engineer / Data Scientist | **Output:** Data dictionary, data quality report

- **Data inventory:** 10,000 readings, 14 raw columns, real-physics-modeled milling machine data (Matzka, 2020)
- **Data quality report:** no missing values (verified via assertion, not assumed); a BOM character on the first column header (see Phase 4) was the only real data-loading issue, and it was a formatting artifact, not a data quality problem
- **The central data-understanding finding:** TWF, HDF, PWF, OSF, RNF are not independent features correlated with the target — they are literally the five conditions whose logical OR constructs the target. This is a more clear-cut leakage case than Day 9's G2 timing judgment call: there is no nuance or trade-off here, only a bright line
- **Access & governance:** public, synthetic dataset; no governance concerns for this exercise, though real factory sensor data would typically involve proprietary process information requiring careful handling

---

## Phase 6: Exploratory Data Analysis (EDA)
**Owner:** Data Scientist | **Output:** `eda.py`, `outputs/eda_summary.png`

**Actual findings, each directly validating a specific physical failure rule:**

- Overall failure rate: 3.39% — comparable imbalance severity to Day 2's fraud problem
- Failure rate by Type: L 3.92%, M 2.77%, H 2.09% — consistent with L having the lowest (easiest-to-trigger) overstrain threshold
- `power_W`: failures cluster in BOTH tails (min 1148W, max 10470W) — a clean, direct confirmation of the documented two-sided Power Failure rule (outside 3500-9000W)
- `torque_wear_product`: failures average 7188 vs. 4214 for non-failures — confirms the Overstrain Failure rule's dependence on this exact product
- `temp_diff_K`: only mildly different between groups (9.40 vs. 10.02) — a genuinely weaker single-variable signal than the other two, honestly reported rather than overstated, since the real Heat Dissipation rule requires this AND low rotational speed jointly

**Why this EDA differs in KIND from every prior day's EDA:** Days 1, 2, 5, 7, and 9 all looked for patterns in data with unknown or partially-known underlying mechanisms. This EDA instead VALIDATES known, documented physical rules against the actual data — a confirmatory rather than exploratory exercise, and a genuinely different use of Phase 6 within this series.

---

## Phase 7: Data Preprocessing & Feature Engineering
**Owner:** Data Scientist / ML Engineer | **Output:** `clean_and_engineer.py`, `split.py`

- **Cleaning:** dropped `UDI` (row index) and `Product ID` (unique serial number) — same category of reasoning as Day 1's customerID
- **THE critical exclusion:** TWF, HDF, PWF, OSF, RNF removed entirely from the feature set — this is a brighter, more clear-cut line than any prior day's leakage decision, since these columns are the literal deterministic components of the target itself, not merely correlated with it
- **Feature engineering — genuinely physics-grounded, not guessed:** `temp_diff_K`, `power_W` (with proper rpm-to-rad/s unit conversion), and `torque_wear_product` were built by directly reading the dataset's own documented failure-mode formulas, not by trial-and-error feature construction
- **Split strategy:** stratified random (same category as Day 1, 7, 9) — no time axis, no aggregation needed

---

## Phase 8: Model Development
**Owner:** Data Scientist / ML Engineer | **Output:** `train_model.py`, `outputs/experiment_log.csv`

**The defining methodological choice of this project:** building TWO
fundamentally different approaches side by side, not just multiple
supervised models:

1. **Unsupervised** — Isolation Forest, trained with the failure label completely withheld, representing a cold-start plant with no failure history
2. **Supervised** — logistic regression, Random Forest, XGBoost, all using the failure label, representing a plant with accumulated history

**Actual supervised experiment log (validation set):**

| Model | Val ROC-AUC | Val Average Precision |
|---|---|---|
| Logistic Regression (baseline) | 0.8847 | 0.3529 |
| **Random Forest** | 0.9547 | **0.8440** |
| XGBoost | 0.9505 | 0.8267 |

**A notably different pattern from Days 1, 7, and 9:** Random Forest wins
clearly and by a wide margin over logistic regression here — the
opposite of the "simple model wins" pattern that had repeated three
times earlier in the series. This is explained, not just noted: the
real failure rules involve non-linear thresholds and multiplicative
interactions (torque × wear, a two-sided power range) that a linear
model cannot represent well, while tree ensembles handle naturally.

---

## Phase 9: Model Evaluation & Business Validation
**Owner:** Data Scientist + Business Stakeholder | **Output:** `outputs/business_validation.json`

**The core Day 11 result — a direct, quantified head-to-head:**

| | Unsupervised (Isolation Forest) | Supervised (Random Forest) |
|---|---|---|
| Test average precision | 0.2298 | **0.9442** |
| Precision @ top-50 queue | 0.24 | **0.90** |
| Recall @ top-50 queue | 0.24 | **0.88** |

Inspecting the same 50 machines per period, the unsupervised approach
catches roughly 1 in 4 real failures; the supervised approach catches
roughly 9 in 10. This is a concrete, quantified answer to "what's the
business value of collecting failure labels before deploying a
supervised model" — a question every prior day's project assumed the
answer to (having labels) rather than demonstrating the alternative.

- **Bias/fairness check:** not applicable in the demographic sense — no protected-attribute-equivalent features exist for a manufacturing sensor dataset. The closest analogue considered was whether the model treats different product Types (L/M/H) fairly, which the EDA in Phase 6 addressed by confirming failure rate differences by Type match documented physical reasons (different overstrain thresholds), not an arbitrary model bias.
- **Explainability:** feature importances (`Rotational speed`, `Torque`, `power_W` top three) confirm the engineered physics features add real value beyond the raw sensor readings alone.
- **A stated trade-off, not a free win:** the deployed model can flag elevated risk but cannot say WHICH failure mode (TWF/HDF/PWF/OSF) is likely, precisely because those sub-labels were correctly excluded as leakage. A more sophisticated future version might train a separate multi-label model for failure-mode diagnosis, kept strictly separate from the overall risk score used for the inspection queue.

---

## Phase 10: MLOps & Deployment
**Owner:** ML Engineer | **Output:** `app.py`

- **Packaging:** FastAPI service (`/score`, `/health`, browser test form)
- **Real-time pattern:** modeled as per-reading scoring, similar to Day 2's real-time fraud pattern and unlike Day 1's nightly batch — sensor data streams continuously, so a machine's risk should update with each new reading, not once a day
- **Deployed artifact is the SUPERVISED model**, not the Isolation Forest — the unsupervised approach exists in this project specifically as the honest cold-start comparison (Phase 8/9), not as a competing production candidate, since this dataset does have failure labels available
- **Verified with 2 constructed readings:** a high-overstrain-risk reading (Type L, high torque, high tool wear) scored 0.9472; a normal reading scored 0.0017

---

## Phase 11: Testing
**Owner:** QA / Data Scientist | **Output:** manual test log

- **What was actually tested:** full pipeline run end-to-end; API tested via FastAPI's `TestClient` with two constructed sensor readings spanning the risk spectrum, both correctly scored
- **What was NOT done:** no formal unit test suite; no UAT with an actual maintenance team; no test of the Isolation Forest's sensitivity to different `contamination` parameter guesses (a real cold-start deployment would need to test several plausible values, not just the training set's own known rate, which isn't information a genuine cold start would have)

---

## Phase 12: Documentation & Handover
**Owner:** Data Scientist | **Output:** `README.md`, `PROJECT_DOCUMENTATION.md`, this file

- `README.md` leads with the unsupervised-vs-supervised comparison table, since that's the single most important, distinctive result of this entire project
- The BOM-character data-loading bug (Phase 4) and the physics-grounded feature engineering rationale are both documented explicitly, continuing the series' practice of showing real engineering judgment, not just final results

---

## Phase 13: Monitoring & Maintenance
**Owner:** MLOps / Data Scientist | **Output:** `monitor.py`

- **What's monitored:** PSI drift on the three physics-grounded features (chosen specifically because drift there is interpretable to a plant engineer, unlike drift on a raw, less-actionable individual sensor reading) and average-precision decay
- **Recommended cadence: near-continuous** — a similar cadence CATEGORY to Day 2's fraud monitoring, but justified by a completely different mechanism: fraud needs speed because of adversarial adaptation; this system can and should move fast because machine failure outcomes are usually known almost immediately, unlike Day 7's months-long loan-default maturation lag
- **A fourth distinct monitoring cadence now established across the series:** monthly (Day 1, churn), near-real-time/adversarial (Day 2, fraud), quarterly (Day 5, segmentation), once-per-term (Day 9, education), and now near-continuous/fast-feedback (Day 11, manufacturing) — each justified by a different real-world mechanism, not a default template applied everywhere

---

## Phase 14: Project Closure & Delivery
**Owner:** N/A (self-directed project) | **Output:** this document, `PROJECT_DOCUMENTATION.md`

- **Closure against original objective:** a supervised model was built and validated (0.94 average precision, 90% precision / 88% recall at a realistic inspection queue size), AND the unsupervised cold-start alternative was quantified directly, giving a concrete answer to a question most predictive-maintenance discussions only address qualitatively
- **Retrospective:**
  - What went well: reading the dataset's own documented failure-mode formulas BEFORE writing feature engineering code produced features that EDA then confirmed behave exactly as physically predicted — a cleaner, more defensible process than iterating on arbitrary feature ideas and checking what happened to stick
  - What to improve next time: test the Isolation Forest's sensitivity to different contamination-rate guesses, since a real cold-start deployment wouldn't have access to the exact rate used here; consider a separate multi-label model for failure-mode diagnosis as a natural next iteration
- **Handover:** packaged as a complete, runnable project (code + real data + trained model + documentation), consistent with Days 1, 2, 5, 7, and 9

---

## Mapping back to the Day 11 LinkedIn post

| LinkedIn section | Pulled from phases | Real number used |
|---|---|---|
| Client & Problem | 1–2 | Manufacturing, reactive-vs-proactive maintenance |
| Requirements & Data | 3, 5 | Real UCI dataset, 10,000 readings, the 5-sub-label leakage trap |
| Approach | 6–8 | Physics-grounded features → Random Forest, plus an honest unsupervised comparison |
| Result & Business Impact | 9 | 0.94 vs. 0.23 average precision — the quantified value of failure labels |
| Path to Production | 10–13 | Real-time scoring API + near-continuous monitoring |
