import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
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


def prepare_data(data):
    print("\nPreparing data...")

    X = data.drop(
        columns=[TARGET, "student_id", "overall_gpa"],
        errors="ignore"
    )

    y = data[TARGET]

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))

    return X, y, numeric_features, categorical_features


def train_model():
    data = load_dataset()

    X, y, numeric_features, categorical_features = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records :", len(X_test))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=42
                )
            )
        ]
    )

    print("\nTraining Gradient Boosting model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, MODEL_FILE)

    print(
        "\nModel saved as:",
        MODEL_FILE
    )

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "model": "GradientBoostingClassifier"
    }

    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as:", METRICS_FILE)

    return accuracy


if __name__ == "__main__":
    train_model()
