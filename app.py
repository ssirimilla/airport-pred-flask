
import numpy as np
from flask import Flask, request, render_template
import pandas as pd
import os
import xgboost as xgb
from datetime import datetime

# ---------- Load model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "analysis", "xgb_model.json")

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

# ---------- Flask app ----------
app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    # ---- Parse inputs ----
    citizenship = float(form["citizenship"])

    date_str = form["date"]      # YYYY-MM-DD
    time_str = form["time"]      # HH:MM

    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

    year = dt.year
    month = dt.month
    day = dt.weekday()  # 0 = Monday

    hour = dt.hour + dt.minute / 60.0

    # ---- Cyclical encoding ----
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)

    sin_day = np.sin(2 * np.pi * day / 7)
    cos_day = np.cos(2 * np.pi * day / 7)

    # ---- Build dataframe ----
    test = pd.DataFrame([{
        "citizenship": citizenship,
        "Year": year,
        "Month": month,
        "Day": day,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_day": sin_day,
        "cos_day": cos_day
    }])

    # Ensure column order is exactly the same as training
    expected_order = [
        "citizenship", "Year", "Month", "Day",
        "sin_hour", "cos_hour", "sin_day", "cos_day"
    ]
    test = test[expected_order]

    dtest = xgb.DMatrix(test)
    prediction_val = float(booster.predict(dtest)[0])

    return render_template(
        "index.html",
        prediction_text=f"The Predicted Wait Time is {prediction_val:.2f}"
    )

if __name__ == "__main__":
    app.run()
