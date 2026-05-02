import json
import os
from pathlib import Path
from typing import Literal, List, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .config import ARTIFACT_DIR
    from .inference import predict_dataframe
except ImportError:
    import sys

    SRC_DIR = Path(__file__).resolve().parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from config import ARTIFACT_DIR
    from inference import predict_dataframe


ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_DIR", ARTIFACT_DIR))

app = FastAPI(title="Student Performance Predictor", version="1.0")


class PredictionRequest(BaseModel):
    records: List[dict] = Field(..., description="List of student records as JSON objects")
    predict: Literal["risk", "performance", "both"] = "both"
    model_family: Optional[str] = None
    risk_model: Optional[str] = None
    performance_model: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    df = pd.DataFrame(request.records)
    schema_path = ARTIFACTS_PATH / "feature_schema.json"
    if not schema_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts not found. Run `python src/train.py` "
                "to generate artifacts before calling /predict."
            ),
        )
    family_map = {
        "linear": ("logistic_regression", "ridge"),
        "decision_tree": ("decision_tree", "decision_tree"),
        "random_forest": ("random_forest", "random_forest"),
        "gradient_boosting": ("gradient_boosting", "gradient_boosting"),
        "svm": ("svm", "svr"),
    }

    risk_model_name = request.risk_model
    perf_model_name = request.performance_model

    if request.model_family and request.model_family != "auto":
        if request.model_family not in family_map:
            raise HTTPException(status_code=400, detail="Unknown model_family")
        family_risk, family_perf = family_map[request.model_family]
        if risk_model_name is None:
            risk_model_name = family_risk
        if perf_model_name is None:
            perf_model_name = family_perf

    try:
        output = predict_dataframe(
            df,
            ARTIFACTS_PATH,
            predict=request.predict,
            risk_model_name=risk_model_name,
            perf_model_name=perf_model_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    sanitized_df = output.replace([np.inf, -np.inf], np.nan)
    records = json.loads(sanitized_df.to_json(orient="records"))
    return {"predictions": records}
