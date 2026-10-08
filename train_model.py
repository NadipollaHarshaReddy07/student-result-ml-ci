import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = "students.csv"
MODEL_FILE = "student_academic_model.pkl"
METRICS_FILE = "metrics.json"

TARGET = "passed"


def load_dataset():
    print("Loading Student Academic Performance dataset...")

    data = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("Number of columns:", len(data.columns))

    return data


def feature_engineering(data):
    print("\nPerforming feature engineering...")

    df = data.copy()

    # Academic score average
    score_cols = [
        "reading_score",
        "writing_score",
        "math_score",
        "science_score"
    ]

    df["academic_score_average"] = df[score_cols].mean(axis=1)

    # Technical score
    df["technical_score"] = (
        df["math_score"] + df["science_score"]
    ) / 2

    # Language score
    df["language_score"] = (
        df["reading_score"] + df["writing_score"]
    ) / 2

    # Study-attendance index
    df["study_attendance_index"] = (
        df["attendance_rate"]
        * np.log1p(df["study_hours_per_week"])
    )

    # Experience score
    df["experience_score"] = (
        df["extracurricular_activities"]
        + df["tutoring_sessions"]
    )

    # Overall skill score
    df["overall_skill_score"] = (
        0.6 * df["academic_score_average"]
        + 0.4 * df["technical_score"]
    )

    # Attendance risk
    df["attendance_risk"] = np.where(
        df["attendance_rate"] < 75,
        1,
        0
    )

    # High stress
    df["high_stress"] = np.where(
        df["stress_level"] >= 7,
        1,
        0
    )

    print("Feature engineering completed.")
    print("Total columns after feature engineering:", len(df.columns))

    return df


def prepare_data(data):
    print("\nPreparing data for machine learning...")

    X = data.drop(
        columns=[
            TARGET,
            "student_id",
            "overall_gpa"
        ],
        errors="ignore"
    )

    y = data[TARGET]

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    print("Total model features:", len(X.columns))
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))

    return X, y, numeric_features, categorical_features


def train_model():

    # Step 1: Load dataset
    data = load_dataset()

    # Step 2: Feature engineering
    data = feature_engineering(data)

    # Step 3: Prepare ML data
    X, y, numeric_features, categorical_features = prepare_data(data)

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records :", len(X_test))

    # Numeric preprocessing
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combined preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # Gradient Boosting model
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    random_state=42
                )
            )
        ]
    )

    # Step 5: Train
    print("\nTraining Gradient Boosting model...")

    model.fit(X_train, y_train)

    print("Model training completed.")

    # Step 6: Prediction
    predictions = model.predict(X_test)

    # Step 7: Evaluation
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    # Step 8: Save model
    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "\nModel saved as:",
        MODEL_FILE
    )

    # Step 9: Save metrics and feature information
    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "model": "GradientBoostingClassifier",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.9,
        "feature_count": int(len(X.columns)),
        "feature_columns": X.columns.tolist()
    }

    with open(
        METRICS_FILE,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        "Metrics saved as:",
        METRICS_FILE
    )

    return accuracy


if __name__ == "__main__":
    train_model()
