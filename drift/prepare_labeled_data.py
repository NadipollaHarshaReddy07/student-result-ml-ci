from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DRIFT_DIR = Path(__file__).resolve().parent

SOURCE_FILE = ROOT / "students.csv"
OUTPUT_FILE = DRIFT_DIR / "current_labeled_data.csv"


def prepare_labeled_data():

    data = pd.read_csv(SOURCE_FILE)

    # Create a simulated recent labeled batch
    current = data.sample(
        n=2000,
        random_state=100
    ).copy()

    current.to_csv(OUTPUT_FILE, index=False)

    print("Simulated labeled current data created.")
    print("Shape:", current.shape)
    print("Target column:", "passed")
    print("Saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_labeled_data()