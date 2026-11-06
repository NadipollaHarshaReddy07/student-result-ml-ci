import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DRIFT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

from train_model import feature_engineering, prepare_data

HISTORICAL_FILE = ROOT / "students.csv"
CURRENT_FILE = DRIFT_DIR / "current_labeled_data.csv"

CANDIDATE_FILE = DRIFT_DIR / "student_result_model_candidate.pkl"
METRICS_FILE = DRIFT_DIR / "retraining_metrics.json"
REPORT_FILE = DRIFT_DIR / "drift_report.json"


def retrain_candidate():

    # Check whether drift was detected
    with open(REPORT_FILE, "r") as file:
        report = json.load(file)

    if not report["overall_drift_detected"]:
        print("No drift detected. Retraining skipped.")
        return

    # Load historical and recent labeled data
    historical = pd.read_csv(HISTORICAL_FILE)
    current = pd.read_csv(CURRENT_FILE)

    if "passed" not in current.columns:
        raise ValueError("Target column 'passed' is missing.")

    combined = pd.concat(
        [historical, current],
        ignore_index=True
    )

    print("Combined records:", len(combined))

    # Feature engineering and preparation
    combined = feature_engineering(combined)

    X, y, numeric_features, categorical_features = prepare_data(
        combined
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Use the same model configuration
    from train_model import train_model
    import train_model as training

    # Reuse the existing model construction by creating
    # the same preprocessing and classifier pipeline.
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42
        ))
    ])

    print("Training candidate model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, CANDIDATE_FILE)

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "combined_records": len(combined),
        "model": "GradientBoostingClassifier",
        "feature_count": len(X.columns)
    }

    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nCandidate training completed.")
    print("Accuracy:", round(accuracy, 4))
    print("Candidate saved at:", CANDIDATE_FILE)
    print("Metrics saved at:", METRICS_FILE)


if __name__ == "__main__":
    retrain_candidate()