# NSL-KDD Network Intrusion Detection API

End-to-end machine learning project for binary network intrusion detection. It trains an XGBoost model on NSL-KDD and exposes predictions through a validated FastAPI service.

## What this version fixes

The original exploration remains available in `Projet.ipynb`. The reusable pipeline in `src/` corrects the main evaluation and reproducibility issues:

- preprocessing is fitted only on the training split;
- categorical values are one-hot encoded with unknown-category support;
- feature engineering, scaling, variance filtering and feature selection are bundled with the model;
- the decision threshold is selected on a validation split, never on the test set;
- the NSL-KDD test set is not used by this training script to fit the model or select the threshold;
- the saved model bundle contains its schema, version, threshold and metrics.

## Architecture

```text
data/KDDTrain+.txt ─┐
                    ├─> training pipeline ─> models/nsl_kdd_xgb.joblib
data/KDDTest+.txt  ─┘                              │
                                                    v
JSON request ─> Pydantic validation ─> FastAPI ─> preprocessing + XGBoost ─> prediction
```

## Project structure

```text
app/
  main.py          # HTTP endpoints and application lifecycle
  schemas.py       # input/output validation for all 41 NSL-KDD features
  service.py       # model loading and inference
src/
  data.py          # dataset schema and loading
  pipeline.py      # feature engineering and sklearn/XGBoost pipeline
  train.py         # validation, threshold selection, evaluation and export
tests/
  test_api.py
  test_pipeline.py
models/
  metrics.json     # reproducible evaluation report
Dockerfile          # trains a model and packages the API
compose.yaml        # local container startup
Projet.ipynb       # exploratory analysis and model experiments
.github/workflows/ci.yml  # tests, Docker smoke test and image delivery
```

## Local setup

Python 3.11 or newer is required. The pinned `xgboost-cpu` package is sufficient for this CPU-based project and avoids downloading the GPU dependencies of the full XGBoost package.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ubuntu/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If the project is on `/mnt/c/` under WSL, create the virtual environment inside the Linux home directory instead, for example `python3 -m venv ~/kddclean-venv` and `source ~/kddclean-venv/bin/activate`.

## Train the model

```bash
python -m src.train
```

This command reads `data/KDDTrain+.txt` and `data/KDDTest+.txt`, then creates:

- `models/nsl_kdd_xgb.joblib`: local model bundle, ignored by Git;
- `models/metrics.json`: validation and test metrics tracked by Git.

## Run the API

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API from the generated Swagger interface.

Available endpoints:

- `GET /health`: service and model readiness;
- `GET /model-info`: model version, threshold and evaluation metrics;
- `POST /predict`: prediction for one connection;
- `POST /predict-batch`: predictions for up to 1,000 connections.

If the model is missing, `/health` and prediction endpoints return HTTP 503. Invalid request bodies return HTTP 422.

## Run the tests

The tests create a tiny temporary model. They do not require training or the NSL-KDD files:

```bash
python -m pytest -q tests
```

## Run with Docker

On Windows with an Ubuntu WSL terminal, install and start Docker Desktop **on Windows**, then enable its WSL 2 backend and the Ubuntu integration in Docker Desktop settings. There is no need to install a separate Docker Engine inside Ubuntu. Run the commands below from the Ubuntu terminal in this project directory.

The image builds its own model from the two files in `data/`; the local `models/nsl_kdd_xgb.joblib` is not required. The final image contains the trained model and API but not the training data. The first build installs dependencies and trains the model, so it takes longer than subsequent cached builds.

```bash
docker compose up --build
```

Open `http://127.0.0.1:8000/docs`. Stop with `Ctrl+C`, then run `docker compose down` if needed. The same image can be built without Compose using `docker build -t kdd-api .` and run with `docker run --rm -p 8000:8000 kdd-api`.

## Current honest evaluation

The threshold is selected on a stratified 20% validation split of `KDDTrain+`. The reported test figures come from the untouched `KDDTest+` dataset.

| Split | Accuracy | Attack precision | Attack recall | Attack F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Validation | 98.96% | 99.02% | 98.74% | 98.88% | 99.92% |
| Test | 81.36% | 96.63% | 69.68% | 80.97% | 94.96% |

The validation/test gap is kept visible because `KDDTest+` contains a harder distribution and attack patterns not represented in the same way in training. This is more credible than selecting a threshold directly on the test set.

## CI and container delivery

The workflow in `.github/workflows/ci.yml` runs after a push, on pull requests, or manually from GitHub Actions. Its first job installs the pinned dependencies and runs the eight self-contained Python tests. Only if those tests pass does the second job build the Docker image (which retrains the model), start a container, and check `/health`.

On a successful push to `main` only, the workflow logs in to GitHub Container Registry using GitHub's temporary `GITHUB_TOKEN` and publishes that same tested image as `ghcr.io/ousyas/nsl-kdd-intrusion-api:latest` and `ghcr.io/ousyas/nsl-kdd-intrusion-api:sha-<full-commit-sha>`. The immutable commit-specific tag makes a release traceable; `latest` is a convenient moving pointer. Pull requests and manual runs never publish. The token's package-write permission is scoped to the Docker job. The image is not deployed to a server, so this step does not create a public API URL. A new GitHub Container Registry package is private by default; its visibility can be changed separately in GitHub package settings.

The Docker build needs both dataset files in `data/` to be present in the repository. The trained `models/*.joblib` artifact is intentionally excluded from Git because the Docker build regenerates it.

## Dataset attribution

NSL-KDD is distributed by the University of New Brunswick's Canadian Institute for Cybersecurity. Its [dataset page](https://www.unb.ca/cic/datasets/nsl.html) permits redistribution provided that both the dataset and its reference paper are cited: M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,” IEEE CISDA, 2009.
