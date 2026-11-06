import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

DRIFT_DIR = Path(__file__).resolve().parent

REFERENCE_FILE = DRIFT_DIR / "reference_data.csv"
CURRENT_FILE = DRIFT_DIR / "current_data.csv"
REPORT_FILE = DRIFT_DIR / "drift_report.json"

P_VALUE_THRESHOLD = 0.05
MIN_DRIFTED_FEATURES = 2


def detect_drift():

    reference = pd.read_csv(REFERENCE_FILE)
    current = pd.read_csv(CURRENT_FILE)

    numeric_features = reference.select_dtypes(
        include=["number"]
    ).columns.tolist()

    results = []
    drifted_features = []

    print("Starting data drift detection...\n")

    for feature in numeric_features:

        statistic, p_value = ks_2samp(
            reference[feature].dropna(),
            current[feature].dropna()
        )

        is_drifted = p_value < P_VALUE_THRESHOLD

        if is_drifted:
            drifted_features.append(feature)

        results.append({
            "feature": feature,
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(is_drifted)
        })

        print(
            f"{feature}: p-value={p_value:.6f}, "
            f"Drift={is_drifted}"
        )

    overall_drift = (
        len(drifted_features) >= MIN_DRIFTED_FEATURES
    )

    report = {
        "p_value_threshold": P_VALUE_THRESHOLD,
        "minimum_drifted_features": MIN_DRIFTED_FEATURES,
        "total_numeric_features": len(numeric_features),
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features,
        "overall_drift_detected": overall_drift,
        "feature_results": results
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print("\nOverall drift detected:", overall_drift)
    print("Drifted features:", drifted_features)
    print("Report saved at:", REPORT_FILE)


if __name__ == "__main__":
    detect_drift()