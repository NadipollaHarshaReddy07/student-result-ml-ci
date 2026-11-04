import os
import sys
import time
import uuid
import logging

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, g, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from werkzeug.exceptions import HTTPException

# Send structured application logs to stdout so Render can collect them.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
app = Flask(__name__)
app.logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# Prometheus metrics for API traffic, latency, prediction outcomes, and model readiness.
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction inference attempts",
    ["result"],
)
MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded",
)

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
        MODEL_LOADED.set(0)
    app.logger.error("Model file is missing: %s", MODEL_FILE)
        raise FileNotFoundError(
            f"{MODEL_FILE} not found. Train the model before starting the API."
        )
    loaded_model = joblib.load(MODEL_FILE)
    MODEL_LOADED.set(1)
    app.logger.info("Prediction model loaded successfully: %s", MODEL_FILE)
    return loaded_model


model = load_model()


@app.before_request
def start_request_logging():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    g.request_start_time = time.perf_counter()
    app.logger.info(
        "Request started request_id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path,
    )


@app.after_request
def finish_request_logging(response):
    request_id = getattr(g, "request_id", "unavailable")
    start_time = getattr(g, "request_start_time", time.perf_counter())
    duration_ms = (time.perf_counter() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id

    # Exclude Prometheus scrapes from user-traffic metrics.
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code),
        ).inc()
        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path,
        ).observe(duration_ms / 1000.0)

    log_message = (
        "Request completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f"
    )
    log_args = (
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    if response.status_code >= 500:
        app.logger.error(log_message, *log_args)
    elif response.status_code >= 400:
        app.logger.warning(log_message, *log_args)
    else:
        app.logger.info(log_message, *log_args)

    return response


@app.errorhandler(HTTPException)
def handle_http_error(error):
    # Preserve standard HTTP status codes such as 404 instead of converting
    # them into generic 500 responses.
    response = jsonify({
        "error": error.name,
        "message": error.description,
        "request_id": getattr(g, "request_id", "unavailable"),
    })
    response.status_code = error.code or 500
    return response


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    # Do not log request bodies, student records, credentials, or other personal data.
    app.logger.exception(
        "Unhandled application error request_id=%s path=%s",
        getattr(g, "request_id", "unavailable"),
        request.path,
    )
    return jsonify({
        "error": "An internal server error occurred.",
        "request_id": getattr(g, "request_id", "unavailable"),
    }), 500


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


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose application metrics in Prometheus text exposition format."""
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST,
    )


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
        app.logger.warning(
            "Prediction rejected request_id=%s reason=invalid_json_object",
            getattr(g, "request_id", "unavailable"),
        )
        return jsonify({
            "error": "Send a valid JSON object containing student details."
        }), 400

    missing_features = [
        feature for feature in RAW_FEATURES
        if feature not in payload
    ]

    if missing_features:
        app.logger.warning(
            "Prediction rejected request_id=%s reason=missing_features count=%s",
            getattr(g, "request_id", "unavailable"),
            len(missing_features),
        )
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
        PREDICTIONS.labels(result="success").inc()

        result = "PASS" if prediction == 1 else "FAIL"
        app.logger.info(
            "Prediction completed request_id=%s prediction=%s result=%s",
            getattr(g, "request_id", "unavailable"),
            prediction,
            result,
        )

        return jsonify({
            "prediction": prediction,
            "result": result,
        })

    except (TypeError, ValueError) as error:
        PREDICTIONS.labels(result="error").inc()
        app.logger.warning(
            "Prediction input validation failed request_id=%s error_type=%s",
            getattr(g, "request_id", "unavailable"),
            type(error).__name__,
        )
        return jsonify({
            "error": "One or more input values are invalid.",
            "details": str(error),
        }), 400


if __name__ == "__main__":
    app.logger.info("Starting Student Academic Performance Prediction API")
    app.run(host="0.0.0.0", port=5000, debug=False)
