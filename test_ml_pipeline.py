import json
import os
import unittest

import joblib
import numpy as np
import pandas as pd


DATA_FILE = "students.csv"
MODEL_FILE = "student_academic_model.pkl"
METRICS_FILE = "metrics.json"


def feature_engineering(data):
    df = data.copy()

    score_cols = [
        "reading_score",
        "writing_score",
        "math_score",
        "science_score"
    ]

    df["academic_score_average"] = df[score_cols].mean(axis=1)

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
        df["attendance_rate"] < 75,
        1,
        0
    )

    df["high_stress"] = np.where(
        df["stress_level"] >= 7,
        1,
        0
    )

    return df


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATA_FILE)
        cls.model = joblib.load(MODEL_FILE)

        with open(METRICS_FILE, "r") as file:
            cls.metrics = json.load(file)

        cls.feature_data = feature_engineering(cls.data)

        cls.X = cls.feature_data.drop(
            columns=[
                "passed",
                "student_id",
                "overall_gpa"
            ],
            errors="ignore"
        )

        cls.y = cls.feature_data["passed"]

    # Test 1
    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists(DATA_FILE)
        )

    # Test 2
    def test_model_exists(self):
        self.assertTrue(
            os.path.exists(MODEL_FILE)
        )

    # Test 3
    def test_metrics_exists(self):
        self.assertTrue(
            os.path.exists(METRICS_FILE)
        )

    # Test 4
    def test_accuracy_is_valid(self):
        accuracy = self.metrics["accuracy"]

        self.assertGreaterEqual(
            accuracy,
            0.0
        )

        self.assertLessEqual(
            accuracy,
            1.0
        )

    # Test 5
    def test_feature_count(self):
        expected_features = self.metrics[
            "feature_count"
        ]

        actual_features = len(
            self.X.columns
        )

        self.assertEqual(
            actual_features,
            expected_features
        )

    # Test 6
    def test_model_prediction(self):
        sample = self.X.iloc[[0]]

        prediction = self.model.predict(sample)

        self.assertEqual(
            len(prediction),
            1
        )

        self.assertIn(
            int(prediction[0]),
            [0, 1]
        )

    # Test 7
    def test_high_performance_student(self):
        high_performance = (
            self.feature_data[
                self.feature_data["passed"] == 1
            ]
            .sort_values(
                by=[
                    "attendance_rate",
                    "academic_score_average",
                    "overall_skill_score"
                ],
                ascending=False
            )
            .iloc[[0]]
        )

        X_high = high_performance.drop(
            columns=[
                "passed",
                "student_id",
                "overall_gpa"
            ],
            errors="ignore"
        )

        prediction = self.model.predict(X_high)

        self.assertEqual(
            int(prediction[0]),
            1
        )

    # Test 8
    def test_low_performance_student(self):
        low_performance = (
            self.feature_data[
                self.feature_data["passed"] == 0
            ]
            .sort_values(
                by=[
                    "attendance_rate",
                    "academic_score_average",
                    "overall_skill_score"
                ],
                ascending=True
            )
            .iloc[[0]]
        )

        X_low = low_performance.drop(
            columns=[
                "passed",
                "student_id",
                "overall_gpa"
            ],
            errors="ignore"
        )

        prediction = self.model.predict(X_low)

        self.assertEqual(
            int(prediction[0]),
            0
        )


if __name__ == "__main__":
    unittest.main()
