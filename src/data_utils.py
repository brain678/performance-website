from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    AT_RISK_ATTENDANCE_THRESHOLD,
    AT_RISK_AVG_SCORE_THRESHOLD,
    AT_RISK_CGPA_THRESHOLD,
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_RISK,
)


def load_dataset(path):
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = [col.strip() for col in df.columns]

    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) != len(set(df.columns)):
        raise ValueError("Dataset has duplicate column names.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Dataset is missing required columns: {missing_list}")

    if TARGET_RISK not in df.columns:
        cgpa = pd.to_numeric(df["cgpa"], errors="coerce")
        avg_score = pd.to_numeric(df["avg_course_score"], errors="coerce")
        attendance = pd.to_numeric(df["attendance"], errors="coerce")
        df[TARGET_RISK] = (
            (cgpa < AT_RISK_CGPA_THRESHOLD)
            | (avg_score < AT_RISK_AVG_SCORE_THRESHOLD)
            | (attendance < AT_RISK_ATTENDANCE_THRESHOLD)
        ).astype(int)
    else:
        df[TARGET_RISK] = pd.to_numeric(df[TARGET_RISK], errors="coerce")

    invalid = [value for value in df[TARGET_RISK].dropna().unique() if value not in (0, 1)]
    if invalid:
        raise ValueError(f"Invalid {TARGET_RISK} values found: {sorted(set(invalid))}")

    return df


def build_feature_frame(df, target, extra_drop):
    drop_cols = set(ID_COLUMNS + [target] + extra_drop)
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    return X, feature_cols


def infer_columns(X):
    cat_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
    object_cols = [col for col in X.columns if X[col].dtype == "object" and col not in cat_cols]
    cat_cols.extend(object_cols)
    num_cols = [col for col in X.columns if col not in cat_cols]
    return cat_cols, num_cols


def coerce_categoricals(X, cat_cols):
    for col in cat_cols:
        X[col] = X[col].astype("string")
    return X


def coerce_numerics(X, num_cols):
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def build_preprocessor(cat_cols, num_cols):
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipeline, cat_cols),
            ("numeric", num_pipeline, num_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def align_features(df, feature_cols):
    for col in feature_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[feature_cols].copy()


def get_feature_names(preprocessor):
    try:
        feature_names = preprocessor.get_feature_names_out()
        return [name.replace("categorical__", "").replace("numeric__", "") for name in feature_names]
    except Exception:
        names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                onehot = transformer.named_steps["onehot"]
                names.extend(onehot.get_feature_names_out(cols))
            else:
                if isinstance(cols, (list, tuple)):
                    names.extend(list(cols))
                else:
                    names.append(cols)
        return names
