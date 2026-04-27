import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .data_utils import align_features, coerce_categoricals, coerce_numerics
except ImportError:
    import sys
    from pathlib import Path

    SRC_DIR = Path(__file__).resolve().parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from data_utils import align_features, coerce_categoricals, coerce_numerics


def load_schema(artifacts_dir):
    schema_path = Path(artifacts_dir) / "feature_schema.json"
    return json.loads(schema_path.read_text())


def load_models(artifacts_dir):
    artifacts_dir = Path(artifacts_dir)
    risk_model = joblib.load(artifacts_dir / "risk_model.pkl")
    perf_model = joblib.load(artifacts_dir / "performance_model.pkl")
    return risk_model, perf_model


def prepare_features(df, feature_cols, cat_cols, num_cols):
    X = align_features(df.copy(), feature_cols)
    X = coerce_categoricals(X, cat_cols)
    X = coerce_numerics(X, num_cols)
    return X


def build_risk_tiers(scores, thresholds, labels):
    if scores is None or len(scores) == 0:
        return None
    low_t, med_t = thresholds
    return np.where(scores < low_t, labels[0], np.where(scores < med_t, labels[1], labels[2]))


def predict_dataframe(df, artifacts_dir, predict="both"):
    schema = load_schema(artifacts_dir)
    output = df.copy()
    risk_model = None
    perf_model = None

    if predict in {"risk", "performance", "both"}:
        risk_model, perf_model = load_models(artifacts_dir)

    if predict in {"risk", "both"}:
        risk_features = schema["risk_features"]
        risk_cat_cols = schema["risk_categorical"]
        risk_num_cols = schema["risk_numeric"]
        risk_threshold = float(schema.get("risk_threshold", 0.5))
        risk_tier_thresholds = schema.get("risk_tier_thresholds", [0.33, 0.66])
        risk_tier_labels = schema.get("risk_tier_labels", ["low", "medium", "high"])
        X_risk = prepare_features(output, risk_features, risk_cat_cols, risk_num_cols)
        if hasattr(risk_model, "predict_proba"):
            output["risk_probability"] = risk_model.predict_proba(X_risk)[:, 1]
            output["predicted_at_risk"] = (output["risk_probability"] >= risk_threshold).astype(int)
            output["risk_tier"] = build_risk_tiers(
                output["risk_probability"].to_numpy(),
                risk_tier_thresholds,
                risk_tier_labels,
            )
        else:
            output["predicted_at_risk"] = risk_model.predict(X_risk)
            output["risk_tier"] = pd.Series(
                output["predicted_at_risk"].map({
                    1: risk_tier_labels[2],
                    0: risk_tier_labels[0],
                })
            )

    if predict in {"performance", "both"}:
        perf_features = schema["performance_features"]
        perf_cat_cols = schema["performance_categorical"]
        perf_num_cols = schema["performance_numeric"]
        X_perf = prepare_features(output, perf_features, perf_cat_cols, perf_num_cols)
        output["predicted_cgpa"] = perf_model.predict(X_perf)

    return output
