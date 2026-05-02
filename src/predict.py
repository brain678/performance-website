import argparse
from pathlib import Path

import pandas as pd

from config import ARTIFACT_DIR
from inference import predict_dataframe


def main():
    parser = argparse.ArgumentParser(description="Predict at-risk status and CGPA.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument(
        "--artifacts",
        default=str(ARTIFACT_DIR),
        help="Artifact directory with saved models",
    )
    parser.add_argument(
        "--predict",
        choices=["risk", "performance", "both"],
        default="both",
        help="Which predictions to run",
    )
    parser.add_argument(
        "--model-family",
        choices=["linear", "decision_tree", "random_forest", "gradient_boosting", "svm"],
        default="random_forest",
        help="Use the same model family for both risk and performance",
    )
    parser.add_argument(
        "--only-at-risk",
        action="store_true",
        help="Keep only rows predicted as at-risk (default)",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Output all rows (overrides --only-at-risk)",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    df = pd.read_csv(args.input)
    family_map = {
        "linear": ("logistic_regression", "ridge"),
        "decision_tree": ("decision_tree", "decision_tree"),
        "random_forest": ("random_forest", "random_forest"),
        "gradient_boosting": ("gradient_boosting", "gradient_boosting"),
        "svm": ("svm", "svr"),
    }

    family_risk, family_perf = family_map[args.model_family]
    risk_model_name = family_risk
    perf_model_name = family_perf

    output = predict_dataframe(
        df,
        artifacts_dir,
        predict=args.predict,
        risk_model_name=risk_model_name,
        perf_model_name=perf_model_name,
    )
    only_at_risk = True if not args.include_all else False
    if args.only_at_risk:
        only_at_risk = True
    if only_at_risk:
        if "predicted_at_risk" not in output.columns:
            raise ValueError("--only-at-risk requires --predict risk or --predict both.")
        output = output[output["predicted_at_risk"] == 1]
    output.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
