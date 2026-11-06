import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DRIFT_DIR = Path(__file__).resolve().parent
REFERENCE_FILE = DRIFT_DIR / "reference_data.csv"


def simulate_data(mode):

    reference = pd.read_csv(REFERENCE_FILE)
    current = reference.sample(
        n=2000,
        replace=True,
        random_state=42
    ).copy()

    if mode == "stable":
        current = current.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

    elif mode == "drift":
        rng = np.random.default_rng(42)

        # Shift numeric feature distributions
        current["attendance_rate"] = np.clip(
            current["attendance_rate"] - 20,
            0,
            100
        )

        current["study_hours_per_week"] = (
            current["study_hours_per_week"] + 8
        )

        current["stress_level"] = np.clip(
            current["stress_level"] + 3,
            1,
            10
        )

        # Change categorical distribution
        current["school_type"] = "public"

    output_file = DRIFT_DIR / "current_data.csv"
    current.to_csv(output_file, index=False)

    print("Current dataset generated.")
    print("Mode:", mode)
    print("Shape:", current.shape)
    print("Saved at:", output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["stable", "drift"],
        required=True
    )

    args = parser.parse_args()
    simulate_data(args.mode)