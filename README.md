# ✈️ Airport Wait Time Prediction App

A full-stack machine learning web application that predicts LAX airport's tom bradley international airport wait times using an XGBoost regression model with cyclical time-based features. The model is packaged behind a Flask web app and deployed to the cloud for public inference.

🔗 Live Demo: https://airport-pred-flask.onrender.com/
🔗 GitHub Repo: https://github.com/ssirimilla/airport-pred-flask

---

## 📌 Project Overview

This project focuses on predicting airport wait times using historical data and time-based patterns. The application allows users to input basic travel information (citizenship, date, and time), computes cyclical features internally, and returns a predicted wait time in real time.

- The goal was to build an end-to-end ML system, covering:
- Feature engineering
- Model training & evaluation
- Fairness analysis
- Backend deployment
- User-facing web interface

---

## 🧠 Model & Methodology

- Model: XGBoost Regressor
- Baselines Compared:
    - Mean predictor
    - Linear regression
    - Random forest
- Feature Engineering:
    - Cyclical encoding of time features using sine/cosine transforms:
    - Hour of day
    - Day of year

- Evaluation:
    - Performance compared against multiple baselines
    - Fairness analysis conducted across citizenship groups, identifying persistent error disparities

---

🌐 Web Application

- Backend: Flask
- Frontend: HTML + Bootstrap (Bootswatch theme)
- Deployment: Render, with 10 min cron-job scheduled
- The web app:
    - Accepts user input via a simple form
    - Computes cyclical features server-side
    - Loads a pre-trained XGBoost model
    - Returns predictions instantly

---

🚀 Running Locally

1. Clone the repository
```git clone https://github.com/ssirimilla/airport-pred-flask.git
cd airport-pred-flask
```
2. Create and activate a virtual environment
```python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```pip install -r requirements.txt
```
4. Run the app
```python app.py
```
Visit the link it generates.

---

📦 Dependencies

'requirements.txt' has:

- Flask
- XGBoost
- scikit-learn
- pandas
- NumPy
- gunicorn

---

⚖️ Fairness Considerations

The project includes an analysis of model performance across different citizenship groups. Results indicated persistent disparities in prediction error, highlighting the importance of fairness evaluation even in non-traditional ML domains.

---

👤 Author
Sujal Sirimilla
Data Science @ UC San Diego