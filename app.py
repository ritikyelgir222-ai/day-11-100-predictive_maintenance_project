"""
Phase 10: MLOps & Deployment
--------------------------------
WHY THIS IS A REAL-TIME, PER-READING ENDPOINT: sensor data streams
continuously from a running machine — unlike Day 1's nightly batch
churn scoring, a predictive maintenance system needs to score each new
reading (or a rolling window of them) as it arrives, closer to Day 2's
real-time fraud-scoring pattern than Day 1's.

Run with:  uvicorn app:app --reload
"""

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Predictive Maintenance Scoring API", version="1.0")

MODEL = joblib.load("outputs/maintenance_model.joblib")
CONFIG = joblib.load("outputs/model_config.joblib")


class SensorReading(BaseModel):
    type: str  # "L", "M", or "H" — product quality variant
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float


class RiskResponse(BaseModel):
    failure_probability: float
    flagged_for_inspection: bool
    top_factors: list[str]


# WHY A FIXED THRESHOLD OF 0.5 HERE, UNLIKE DAY 7's TUNED ONE: this
# project's business framing (Phase 9) is queue-based — rank ALL machines
# by score and inspect the top N per period — rather than a fixed yes/no
# cutoff applied independently to each reading. A single API call scoring
# one machine in isolation has no "queue" to rank within, so a
# conventional 0.5 midpoint is used here as a simple, secondary flag,
# while the real operational decision (who's in this period's top 50) is
# made by ranking many scores together, not by this per-call threshold.
FLAG_THRESHOLD = 0.5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=RiskResponse)
def score_reading(reading: SensorReading):
    temp_diff_K = reading.process_temperature_k - reading.air_temperature_k
    power_W = reading.torque_nm * reading.rotational_speed_rpm * (2 * np.pi / 60)
    torque_wear_product = reading.torque_nm * reading.tool_wear_min

    row = {
        "Type": reading.type,
        "Air temperature [K]": reading.air_temperature_k,
        "Process temperature [K]": reading.process_temperature_k,
        "Rotational speed [rpm]": reading.rotational_speed_rpm,
        "Torque [Nm]": reading.torque_nm,
        "Tool wear [min]": reading.tool_wear_min,
        "temp_diff_K": temp_diff_K,
        "power_W": power_W,
        "torque_wear_product": torque_wear_product,
    }
    X = pd.DataFrame([row])

    proba = float(MODEL.predict_proba(X)[0, 1])
    flagged = proba >= FLAG_THRESHOLD

    prep = MODEL.named_steps["prep"]
    clf = MODEL.named_steps["clf"]
    ohe_names = prep.named_transformers_["cat"].get_feature_names_out(CONFIG["cat_cols"])
    all_names = list(ohe_names) + CONFIG["num_cols"]
    importances = pd.Series(clf.feature_importances_, index=all_names)
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return RiskResponse(
        failure_probability=round(proba, 4),
        flagged_for_inspection=bool(flagged),
        top_factors=top_factors,
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Predictive Maintenance Scoring</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>Predictive Maintenance — Sensor Risk Scoring</h2>
        <p>Full docs at <a href="/docs">/docs</a>.</p>
        <form id="sensorForm">
            <label>Product type:
                <select name="type"><option>L</option><option>M</option><option>H</option></select>
            </label><br><br>
            <label>Air temperature (K): <input name="air" type="number" step="0.1" value="300"></label><br><br>
            <label>Process temperature (K): <input name="process" type="number" step="0.1" value="309"></label><br><br>
            <label>Rotational speed (rpm): <input name="speed" type="number" value="1400"></label><br><br>
            <label>Torque (Nm): <input name="torque" type="number" step="0.1" value="65"></label><br><br>
            <label>Tool wear (min): <input name="wear" type="number" value="180"></label><br><br>
            <button type="submit">Score this reading</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("sensorForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {
                type: form.get("type"),
                air_temperature_k: parseFloat(form.get("air")),
                process_temperature_k: parseFloat(form.get("process")),
                rotational_speed_rpm: parseFloat(form.get("speed")),
                torque_nm: parseFloat(form.get("torque")),
                tool_wear_min: parseFloat(form.get("wear"))
            };
            const res = await fetch("/score", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById("result").innerText =
                "Failure probability: " + data.failure_probability +
                " | Flagged: " + data.flagged_for_inspection +
                " | Top factors: " + data.top_factors.join(", ");
        });
        </script>
    </body>
    </html>
    """
