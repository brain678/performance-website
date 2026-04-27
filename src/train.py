import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from config import (
    ARTIFACT_DIR,
    CV_FOLDS,
    DATA_PATH,
    EXPLAIN_MAX_SAMPLES,
    PERF_DROP_COLUMNS,
    PLOTS_DIR_NAME,
    RANDOM_STATE,
    RISK_DROP_COLUMNS,
    RISK_TIER_LABELS,
    RISK_TIER_THRESHOLDS,
    STRICT_GROUP_COLUMN,
    STRICT_RISK_DROP_COLUMNS,
    TARGET_PERF,
    TARGET_RISK,
    TEST_SIZE,
    TOP_FEATURES,
    TUNE_ITER,
    RISK_THRESHOLD_DEFAULT,
    RISK_VAL_SIZE,
)
from data_utils import (
    build_feature_frame,
    build_preprocessor,
    coerce_categoricals,
    get_feature_names,
    infer_columns,
    load_dataset,
)


def build_pipeline(cat_cols, num_cols, model):
    preprocessor = build_preprocessor(cat_cols, num_cols)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_candidates(task):
    if task == "classification":
        return {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        }

    return {
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def get_param_distributions(model):
    if isinstance(model, RandomForestClassifier):
        return {
            "model__n_estimators": [200, 300, 500, 700],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 4, 6],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.7, 1.0],
            "model__bootstrap": [True, False],
        }
    if isinstance(model, RandomForestRegressor):
        return {
            "model__n_estimators": [200, 300, 500, 700],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 4, 6],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.7, 1.0],
            "model__bootstrap": [True, False],
        }
    if isinstance(model, GradientBoostingClassifier):
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.9, 1.0],
        }
    if isinstance(model, GradientBoostingRegressor):
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.9, 1.0],
        }
    if isinstance(model, LogisticRegression):
        return {
            "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"],
        }
    if isinstance(model, Ridge):
        return {
            "model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
        }
    return {}


def tune_pipeline(X, y, cat_cols, num_cols, model, task, cv_folds, n_iter):
    param_dist = get_param_distributions(model)
    if not param_dist:
        return build_pipeline(cat_cols, num_cols, model), {"enabled": False}, None

    if task == "classification":
        scoring = "f1"
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    else:
        scoring = "neg_mean_squared_error"
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    pipeline = build_pipeline(cat_cols, num_cols, model)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=max(1, n_iter),
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X, y)

    best_score = float(search.best_score_)
    if task == "regression":
        best_score = float(np.sqrt(-best_score))
        score_name = "rmse"
    else:
        score_name = "f1"

    tuning_info = {
        "enabled": True,
        "best_params": search.best_params_,
        "best_score": best_score,
        "score_name": score_name,
    }
    return search.best_estimator_, tuning_info, search.cv_results_


def save_tuning_results(cv_results, task, output_path):
    if not cv_results:
        return
    results_df = pd.DataFrame(cv_results)
    if task == "regression" and "mean_test_score" in results_df:
        results_df["rmse_mean"] = np.sqrt(-results_df["mean_test_score"])
    results_df.to_csv(output_path, index=False)


def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def select_best_threshold(y_true, scores, default_threshold):
    if y_true is None or scores is None:
        return default_threshold, None
    if len(scores) == 0:
        return default_threshold, None

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return default_threshold, None

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def build_risk_tiers(scores, thresholds, labels):
    if scores is None or len(scores) == 0:
        return None
    low_t, med_t = thresholds
    return np.where(scores < low_t, labels[0], np.where(scores < med_t, labels[1], labels[2]))


def evaluate_candidates(X, y, cat_cols, num_cols, task, cv_folds):
    candidates = build_candidates(task)
    if task == "classification":
        scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        neg_metrics = set()
    else:
        scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_mean_squared_error", "r2": "r2"}
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        neg_metrics = {"mae", "rmse"}

    results = []
    for name, model in candidates.items():
        pipeline = build_pipeline(cat_cols, num_cols, model)
        scores = cross_validate(
            pipeline,
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
        )
        row = {"model": name}
        for metric in scoring.keys():
            values = np.array(scores[f"test_{metric}"])
            if metric in neg_metrics:
                values = -values
            if metric == "rmse":
                values = np.sqrt(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        results.append(row)

    results_df = pd.DataFrame(results)
    if task == "classification":
        results_df = results_df.sort_values("f1_mean", ascending=False)
        best_name = results_df.iloc[0]["model"]
    else:
        results_df = results_df.sort_values("rmse_mean", ascending=True)
        best_name = results_df.iloc[0]["model"]

    return results_df, candidates[best_name], best_name


def plot_metric_bar(results_df, metric, output_path, title):
    if metric not in results_df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results_df, x="model", y=metric, hue="model", palette="viridis", legend=False)
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_feature_importance(pipeline, feature_names, output_dir, prefix, top_n):
    model = pipeline.named_steps["model"]
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if hasattr(coef, "ndim") and coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)

    if importances is None:
        return

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / f"{prefix}_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=importance_df.head(top_n),
        x="importance",
        y="feature",
        hue="feature",
        palette="magma",
        legend=False,
    )
    plt.title(f"Top {top_n} Features: {prefix}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_feature_importance.png", dpi=150)
    plt.close()


def save_shap_summary(pipeline, X_sample, output_path, max_samples, max_display):
    try:
        import shap
    except Exception:
        return False

    if X_sample.empty:
        return False

    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=RANDOM_STATE)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocessor)
    X_transformed = preprocessor.transform(X_sample)
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)
    shap.summary_plot(shap_values, X_transformed, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Train models for performance and at-risk prediction.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to Dataset.csv")
    parser.add_argument("--artifacts", default=str(ARTIFACT_DIR), help="Artifact output directory")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP explainability plots")
    parser.add_argument("--top-features", type=int, default=TOP_FEATURES)
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--tune-iter", type=int, default=TUNE_ITER)
    parser.add_argument(
        "--strict-risk",
        action="store_true",
        help="Use stricter leakage controls and group-based split for risk model",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    df = load_dataset(args.data)
    df = df.dropna(subset=[TARGET_RISK, TARGET_PERF]).reset_index(drop=True)

    y_risk = df[TARGET_RISK].astype(int)
    y_perf = df[TARGET_PERF].astype(float)

    risk_drop = STRICT_RISK_DROP_COLUMNS if args.strict_risk else RISK_DROP_COLUMNS
    X_risk, risk_features = build_feature_frame(df, TARGET_RISK, risk_drop)
    X_perf, perf_features = build_feature_frame(df, TARGET_PERF, PERF_DROP_COLUMNS)

    risk_cat_cols, risk_num_cols = infer_columns(X_risk)
    perf_cat_cols, perf_num_cols = infer_columns(X_perf)

    X_risk = coerce_categoricals(X_risk, risk_cat_cols)
    X_perf = coerce_categoricals(X_perf, perf_cat_cols)

    if args.strict_risk and STRICT_GROUP_COLUMN in df.columns:
        group_values = df[STRICT_GROUP_COLUMN].astype("string")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_idx, test_idx = next(splitter.split(df, y_risk, groups=group_values))
    else:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y_risk,
        )

    X_risk_train = X_risk.loc[train_idx]
    X_risk_test = X_risk.loc[test_idx]
    y_risk_train = y_risk.loc[train_idx]
    y_risk_test = y_risk.loc[test_idx]

    try:
        risk_train_idx, risk_val_idx = train_test_split(
            train_idx,
            test_size=RISK_VAL_SIZE,
            random_state=args.random_state,
            stratify=y_risk.loc[train_idx],
        )
    except ValueError:
        risk_train_idx = train_idx
        risk_val_idx = []

    X_risk_fit = X_risk.loc[risk_train_idx]
    y_risk_fit = y_risk.loc[risk_train_idx]
    X_risk_val = X_risk.loc[risk_val_idx] if len(risk_val_idx) else None
    y_risk_val = y_risk.loc[risk_val_idx] if len(risk_val_idx) else None

    X_perf_train = X_perf.loc[train_idx]
    X_perf_test = X_perf.loc[test_idx]
    y_perf_train = y_perf.loc[train_idx]
    y_perf_test = y_perf.loc[test_idx]

    risk_cv_results, risk_best_model, risk_best_name = evaluate_candidates(
        X_risk_fit,
        y_risk_fit,
        risk_cat_cols,
        risk_num_cols,
        "classification",
        args.cv_folds,
    )
    perf_cv_results, perf_best_model, perf_best_name = evaluate_candidates(
        X_perf_train,
        y_perf_train,
        perf_cat_cols,
        perf_num_cols,
        "regression",
        args.cv_folds,
    )

    risk_tuning = {"enabled": False}
    perf_tuning = {"enabled": False}
    risk_tuning_results = None
    perf_tuning_results = None

    if args.tune:
        risk_model, risk_tuning, risk_tuning_results = tune_pipeline(
            X_risk_fit,
            y_risk_fit,
            risk_cat_cols,
            risk_num_cols,
            risk_best_model,
            "classification",
            args.cv_folds,
            args.tune_iter,
        )
        perf_model, perf_tuning, perf_tuning_results = tune_pipeline(
            X_perf_train,
            y_perf_train,
            perf_cat_cols,
            perf_num_cols,
            perf_best_model,
            "regression",
            args.cv_folds,
            args.tune_iter,
        )
    else:
        risk_model = build_pipeline(risk_cat_cols, risk_num_cols, risk_best_model)
        perf_model = build_pipeline(perf_cat_cols, perf_num_cols, perf_best_model)
        risk_model.fit(X_risk_fit, y_risk_fit)
        perf_model.fit(X_perf_train, y_perf_train)

    risk_proba = None
    risk_proba_val = None
    if hasattr(risk_model, "predict_proba"):
        risk_proba = risk_model.predict_proba(X_risk_test)[:, 1]
        if X_risk_val is not None and y_risk_val is not None:
            risk_proba_val = risk_model.predict_proba(X_risk_val)[:, 1]

    risk_threshold, threshold_f1 = select_best_threshold(
        y_risk_val,
        risk_proba_val,
        RISK_THRESHOLD_DEFAULT,
    )
    if risk_proba is not None:
        risk_preds = (risk_proba >= risk_threshold).astype(int)
    else:
        risk_preds = risk_model.predict(X_risk_test)

    perf_preds = perf_model.predict(X_perf_test)

    risk_metrics = {
        "accuracy": float(accuracy_score(y_risk_test, risk_preds)),
        "precision": float(precision_score(y_risk_test, risk_preds, zero_division=0)),
        "recall": float(recall_score(y_risk_test, risk_preds, zero_division=0)),
        "f1": float(f1_score(y_risk_test, risk_preds, zero_division=0)),
        "best_model": risk_best_name,
        "threshold": float(risk_threshold),
        "threshold_f1": threshold_f1,
        "confusion_matrix": confusion_matrix(y_risk_test, risk_preds).tolist(),
        "classification_report": classification_report(
            y_risk_test,
            risk_preds,
            zero_division=0,
            output_dict=True,
        ),
    }
    if risk_proba is not None:
        try:
            risk_metrics["roc_auc"] = float(roc_auc_score(y_risk_test, risk_proba))
        except ValueError:
            risk_metrics["roc_auc"] = None

    perf_metrics = {
        "mae": float(mean_absolute_error(y_perf_test, perf_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_perf_test, perf_preds))),
        "r2": float(r2_score(y_perf_test, perf_preds)),
        "mape": calculate_mape(y_perf_test, perf_preds),
        "best_model": perf_best_name,
    }

    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = artifacts_dir / PLOTS_DIR_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(risk_model, artifacts_dir / "risk_model.pkl")
    joblib.dump(perf_model, artifacts_dir / "performance_model.pkl")

    risk_encoded_features = get_feature_names(risk_model.named_steps["preprocessor"])
    perf_encoded_features = get_feature_names(perf_model.named_steps["preprocessor"])

    schema = {
        "risk_features": risk_features,
        "performance_features": perf_features,
        "risk_categorical": risk_cat_cols,
        "performance_categorical": perf_cat_cols,
        "risk_numeric": risk_num_cols,
        "performance_numeric": perf_num_cols,
        "risk_encoded_features": risk_encoded_features,
        "performance_encoded_features": perf_encoded_features,
        "risk_threshold": risk_threshold,
        "risk_tier_thresholds": RISK_TIER_THRESHOLDS,
        "risk_tier_labels": RISK_TIER_LABELS,
    }
    (artifacts_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    risk_cv_results.to_csv(artifacts_dir / "risk_cv_results.csv", index=False)
    perf_cv_results.to_csv(artifacts_dir / "performance_cv_results.csv", index=False)

    if risk_tuning_results is not None:
        save_tuning_results(
            risk_tuning_results,
            "classification",
            artifacts_dir / "risk_tuning_results.csv",
        )
    if perf_tuning_results is not None:
        save_tuning_results(
            perf_tuning_results,
            "regression",
            artifacts_dir / "performance_tuning_results.csv",
        )

    metrics = {
        "risk": risk_metrics,
        "performance": perf_metrics,
        "rows": int(len(df)),
        "tuning": {
            "risk": risk_tuning,
            "performance": perf_tuning,
        },
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    full_predictions = df.copy()
    full_predictions["split"] = "train"
    full_predictions.loc[test_idx, "split"] = "test"
    if hasattr(risk_model, "predict_proba"):
        full_predictions["risk_probability"] = risk_model.predict_proba(X_risk)[:, 1]
        full_predictions["predicted_at_risk"] = (
            full_predictions["risk_probability"] >= risk_threshold
        ).astype(int)
        full_predictions["risk_tier"] = build_risk_tiers(
            full_predictions["risk_probability"].to_numpy(),
            RISK_TIER_THRESHOLDS,
            RISK_TIER_LABELS,
        )
    else:
        full_predictions["predicted_at_risk"] = risk_model.predict(X_risk)
        full_predictions["risk_tier"] = np.where(
            full_predictions["predicted_at_risk"] == 1,
            RISK_TIER_LABELS[2],
            RISK_TIER_LABELS[0],
        )
    full_predictions["predicted_cgpa"] = perf_model.predict(X_perf)
    full_predictions.to_csv(artifacts_dir / "predictions_full.csv", index=False)
    full_predictions.loc[test_idx].to_csv(artifacts_dir / "predictions_test.csv", index=False)

    if not args.no_plots:
        plot_metric_bar(
            risk_cv_results,
            "f1_mean",
            plots_dir / "risk_cv_f1.png",
            "Risk Model CV (F1)",
        )
        plot_metric_bar(
            perf_cv_results,
            "rmse_mean",
            plots_dir / "performance_cv_rmse.png",
            "Performance Model CV (RMSE)",
        )

        ConfusionMatrixDisplay.from_predictions(y_risk_test, risk_preds, cmap="Blues")
        plt.title("Risk Confusion Matrix")
        plt.tight_layout()
        plt.savefig(plots_dir / "risk_confusion_matrix.png", dpi=150)
        plt.close()

        if risk_proba is not None:
            RocCurveDisplay.from_predictions(y_risk_test, risk_proba)
            plt.title("Risk ROC Curve")
            plt.tight_layout()
            plt.savefig(plots_dir / "risk_roc_curve.png", dpi=150)
            plt.close()

            PrecisionRecallDisplay.from_predictions(y_risk_test, risk_proba)
            plt.title("Risk Precision-Recall Curve")
            plt.tight_layout()
            plt.savefig(plots_dir / "risk_pr_curve.png", dpi=150)
            plt.close()

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=y_perf_test, y=perf_preds)
        min_val = min(y_perf_test.min(), perf_preds.min())
        max_val = max(y_perf_test.max(), perf_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
        plt.xlabel("Actual CGPA")
        plt.ylabel("Predicted CGPA")
        plt.title("Actual vs Predicted CGPA")
        plt.tight_layout()
        plt.savefig(plots_dir / "performance_actual_vs_pred.png", dpi=150)
        plt.close()

        residuals = y_perf_test - perf_preds
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=20, color="teal")
        plt.title("CGPA Residuals")
        plt.xlabel("Residual")
        plt.tight_layout()
        plt.savefig(plots_dir / "performance_residuals.png", dpi=150)
        plt.close()

        save_feature_importance(
            risk_model,
            risk_encoded_features,
            plots_dir,
            "risk",
            args.top_features,
        )
        save_feature_importance(
            perf_model,
            perf_encoded_features,
            plots_dir,
            "performance",
            args.top_features,
        )

    if not args.no_shap:
        save_shap_summary(
            risk_model,
            X_risk_train,
            plots_dir / "risk_shap_summary.png",
            EXPLAIN_MAX_SAMPLES,
            args.top_features,
        )
        save_shap_summary(
            perf_model,
            X_perf_train,
            plots_dir / "performance_shap_summary.png",
            EXPLAIN_MAX_SAMPLES,
            args.top_features,
        )

    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
