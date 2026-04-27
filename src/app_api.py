import json
import os
from pathlib import Path
from typing import Literal, List

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
    output = predict_dataframe(df, ARTIFACTS_PATH, predict=request.predict)
    sanitized_df = output.replace([np.inf, -np.inf], np.nan)
    records = json.loads(sanitized_df.to_json(orient="records"))
    return {"predictions": records}
