import json
import shutil
from pathlib import Path

DRIFT_DIR = Path(__file__).resolve().parent
ROOT = DRIFT_DIR.parent

METRICS_FILE = DRIFT_DIR / "retraining_metrics.json"
CANDIDATE_FILE = DRIFT_DIR / "student_result_model_candidate.pkl"

APPROVED_MODEL = ROOT / "student_academic_model.pkl"

MIN_ACCURACY = 0.85


def approve_candidate():

    # Check whether candidate metrics exist
    if not METRICS_FILE.exists():
        print("ERROR: Retraining metrics not found.")
        return

    if not CANDIDATE_FILE.exists():
        print("ERROR: Candidate model not found.")
        return

    with open(METRICS_FILE, "r") as file:
        metrics = json.load(file)

    accuracy = metrics["accuracy"]

    print("Candidate Model Quality Gate")
    print("----------------------------")
    print("Candidate accuracy:", round(accuracy, 4))
    print("Minimum accuracy:", MIN_ACCURACY)

    if accuracy >= MIN_ACCURACY:

        shutil.copy2(CANDIDATE_FILE, APPROVED_MODEL)

        print("\nQUALITY GATE: PASSED")
        print("Candidate promoted successfully.")
        print("Approved model:", APPROVED_MODEL)

    else:

        print("\nQUALITY GATE: FAILED")
        print("Existing approved model remains unchanged.")
        print("Candidate was not promoted.")


if __name__ == "__main__":
    approve_candidate()