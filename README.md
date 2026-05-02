# Student Performance and At-Risk Prediction

## Overview

This system predicts student academic performance (CGPA) and identifies at-risk students using a machine learning pipeline built in Python. It supports training, evaluation, explainability, batch prediction, and a FastAPI + Streamlit interface for presentation.

## Key Capabilities

- Dual models: CGPA regression and at-risk classification
- Robust evaluation: cross-validation, metrics, plots, and explainability
- Strict risk mode: leakage-safe features and group-based splits
- Risk confidence bands: low / medium / high tiers
- API + UI: FastAPI service and Streamlit dashboard
- Synthetic test data generator for demos

## Dataset Requirements

The dataset must contain the following columns:

- name, matric_no, department, level
- attendance, study_hours, ca1, ca2, assignment_score, num_courses
- exam_score, gpa_sem1, gpa_sem2, cgpa, avg_course_score

Optional:

- at_risk (0 or 1)

If at_risk is missing, the system will generate it using:

- cgpa < 2.0
- avg_course_score < 45
- attendance < 50

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py
uvicorn src.app_api:app --reload
```

Then start the UI:

```bash
streamlit run src/app_streamlit.py
```

## Training

Basic training:

```bash
python src/train.py
```

Tuned training:

```bash
python src/train.py --tune --tune-iter 25
```

Strict risk mode (recommended for final-year reporting):

```bash
python src/train.py --tune --tune-iter 25 --strict-risk
```

## Prediction (CLI)

Predict everything:

```bash
python src/predict.py --input Dataset.csv --output predictions.csv
```

Predict risk only:

```bash
python src/predict.py --input Dataset.csv --output predictions.csv --predict risk
```

Use a single model family for both risk and performance (default is random_forest):

```bash
python src/predict.py --input Dataset.csv --output predictions.csv --model-family random_forest
```

Save only at-risk students (default):

```bash
python src/predict.py --input Dataset.csv --output at_risk_only.csv --predict risk --only-at-risk
```

Output all rows:

```bash
python src/predict.py --input Dataset.csv --output predictions.csv --predict risk --include-all
```

## API Service (FastAPI)

```bash
uvicorn src.app_api:app --reload
```

Endpoints:

- GET /health
- POST /predict

Example payload:

```json
{
  "records": [{"name": "Student A", "department": "Physics", "level": 200, "attendance": 80}],
  "predict": "both"
}
```

## Streamlit App

```bash
streamlit run src/app_streamlit.py
```

The Streamlit app calls the FastAPI service. Start the API first, then run Streamlit.

## Render Deployment (API + UI)

This repo includes render.yaml to deploy both services.

Steps:

1. Push this project to GitHub.
2. Create a new Render account and choose "New" -> "Blueprint".
3. Connect your GitHub repo and select render.yaml.
4. Render will create two services:
  - performance-website-api (FastAPI)
  - performance-website (Streamlit)

Notes:

- The API build command trains the model to generate artifacts.
- The UI uses API_URL from Render to call the API service.

## Synthetic Test Data

```bash
python src/generate_synthetic_data.py --rows 300 --output synthetic_students.csv
```

## Artifacts

Training saves outputs to the artifacts folder:

- risk_model.pkl
- performance_model.pkl
- feature_schema.json
- metrics.json
- risk_cv_results.csv
- performance_cv_results.csv
- risk_tuning_results.csv (when --tune is enabled)
- performance_tuning_results.csv (when --tune is enabled)
- predictions_full.csv
- predictions_test.csv
- plots/ (confusion matrix, ROC curve, regression plots, SHAP, feature importance)

## Risk Tiers

The system assigns risk tiers using risk_probability:

- low: below 0.33
- medium: 0.33 to 0.66
- high: above 0.66

You can adjust tier thresholds in src/config.py.

## Report Template

Use the final-year report template here:

- report/Final_Project_Report.md

## Notes

- Missing values are imputed automatically.
- Risk prediction uses a calibrated threshold saved in artifacts/feature_schema.json.
- Strict risk mode improves real-world validity by reducing leakage.
