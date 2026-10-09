
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_FILE = "student_academic_model.pkl"

FEATURE_COLUMNS = [
    "age",
    "gender",
    "ethnicity",
    "parental_education",
    "family_income",
    "school_type",
    "school_region",
    "study_hours_per_week",
    "attendance_rate",
    "extracurricular_activities",
    "sports_participation",
    "tutoring_sessions",
    "parental_involvement",
    "internet_access",
    "has_laptop",
    "sleep_hours",
    "stress_level",
    "motivation_score",
    "reading_score",
    "writing_score",
    "math_score",
    "science_score",
    "academic_score_average",
    "technical_score",
    "language_score",
    "study_attendance_index",
    "experience_score",
    "overall_skill_score",
    "attendance_risk",
    "high_stress",
]

RAW_FEATURES = FEATURE_COLUMNS[:22]

NUMERIC_FEATURES = [
    "age",
    "study_hours_per_week",
    "attendance_rate",
    "extracurricular_activities",
    "sports_participation",
    "tutoring_sessions",
    "sleep_hours",
    "stress_level",
    "motivation_score",
    "reading_score",
    "writing_score",
    "math_score",
    "science_score",
]


def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"{MODEL_FILE} not found. Train the model before starting the API."
        )
    return joblib.load(MODEL_FILE)


model = load_model()


def engineer_features(data):
    df = data.copy()

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df["academic_score_average"] = df[
        ["reading_score", "writing_score", "math_score", "science_score"]
    ].mean(axis=1)

    df["technical_score"] = (
        df["math_score"] + df["science_score"]
    ) / 2

    df["language_score"] = (
        df["reading_score"] + df["writing_score"]
    ) / 2

    df["study_attendance_index"] = (
        df["attendance_rate"]
        * np.log1p(df["study_hours_per_week"])
    )

    df["experience_score"] = (
        df["extracurricular_activities"]
        + df["tutoring_sessions"]
    )

    df["overall_skill_score"] = (
        0.6 * df["academic_score_average"]
        + 0.4 * df["technical_score"]
    )

    df["attendance_risk"] = np.where(
        df["attendance_rate"] < 75, 1, 0
    )

    df["high_stress"] = np.where(
        df["stress_level"] >= 7, 1, 0
    )

    return df.reindex(columns=FEATURE_COLUMNS)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Academic Performance Prediction API",
        "prediction_endpoint": "/predict",
        "method": "POST",
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Send a valid JSON object containing student details."
        }), 400

    missing_features = [
        feature for feature in RAW_FEATURES
        if feature not in payload
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required input features.",
            "missing_features": missing_features,
        }), 400

    try:
        input_data = pd.DataFrame(
            [{feature: payload[feature] for feature in RAW_FEATURES}]
        )

        input_data = engineer_features(input_data)
        prediction = int(model.predict(input_data)[0])

        result = "PASS" if prediction == 1 else "FAIL"

        return jsonify({
            "prediction": prediction,
            "result": result,
        })

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": "One or more input values are invalid.",
            "details": str(error),
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
