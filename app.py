# import numpy as np
# from flask import Flask, request, render_template
# import pandas as pd
# import joblib
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "analysis", "model.joblib")

# model = joblib.load(MODEL_PATH)



# flask_app = Flask(__name__)

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods=["POST"])
# def predict():
#     form = request.form


#     feature_test = {
#         "citizenship": float(form["citizenship"]),
#         "Year": float(form["Year"]),
#         "Month": float(form["Month"]),
#         "Day": float(form["Day"]),
#         "sin_hour": float(form["sin_hour"]),
#         "cos_hour": float(form["cos_hour"]),
#         "sin_day": float(form["sin_day"]),
#         "cos_day": float(form["cos_day"]),
#     }

#     test = pd.DataFrame([feature_test])
#     prediction = model.predict(test)
#     prediction_val = float(prediction[0])

#     return render_template(
#         "index.html",
#         prediction_text=f"The Predicted Wait Time is {prediction_val}, {dict(form)}"
#     )

# if __name__ == "__main__":
#     flask_app.run(debug=True)

import numpy as np
from flask import Flask, request, render_template
import pandas as pd
import os
import xgboost as xgb

# ---------- Load model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "analysis", "xgb_model.json")

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

# ---------- Flask app ----------
flask_app = Flask(__name__)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    feature_test = {
        "citizenship": float(form["citizenship"]),
        "Year": float(form["Year"]),
        "Month": float(form["Month"]),
        "Day": float(form["Day"]),
        "sin_hour": float(form["sin_hour"]),
        "cos_hour": float(form["cos_hour"]),
        "sin_day": float(form["sin_day"]),
        "cos_day": float(form["cos_day"]),
    }

    test = pd.DataFrame([feature_test]).astype(float)

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
    flask_app.run(debug=True)
