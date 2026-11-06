import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Access the main project directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_model import load_dataset, feature_engineering, prepare_data

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "reference_data.csv"


def prepare_reference():

    print("Preparing reference dataset...")

    # Load original dataset
    data = load_dataset()

    # Apply existing feature engineering
    data = feature_engineering(data)

    # Select the exact model input features
    X, y, numeric_features, categorical_features = prepare_data(data)

    # Save reference feature distribution
    X.to_csv(OUTPUT_FILE, index=False)

    print("\nReference dataset created successfully.")
    print("Shape:", X.shape)
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print("Saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_reference()