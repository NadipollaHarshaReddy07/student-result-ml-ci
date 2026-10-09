# student-result-ml-ci
Student Academic Performance ML model with GitHub Actions CI
## Flask Prediction API

This project provides a Flask REST API for predicting student academic results using a trained Gradient Boosting classifier.

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Displays API information |
| GET | `/health` | Checks API health |
| POST | `/predict` | Predicts PASS or FAIL |

### Running the API Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train the model to generate the model artifact:

```bash
python train_model.py
```

Start the API:

```bash
python app.py
```

The API will be available at `http://127.0.0.1:5000`.

### Prediction Request

Send a POST request to `/predict` with a JSON object containing the 22 original student input features required by the model.

The API calculates the engineered features automatically and returns the prediction as `PASS` or `FAIL`.

### Continuous Integration

GitHub Actions automatically runs the ML pipeline, evaluates the model quality gate, and executes the Flask API tests.
