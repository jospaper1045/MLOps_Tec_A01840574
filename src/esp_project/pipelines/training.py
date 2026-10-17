import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data_cleaning import clean_data, load_data
from ..data_cleaning.cleaner import TARGET_COLUMN
from ..features import build_preprocessor
from ..models import build_model, evaluate_model
from ..visualization import save_actual_vs_predicted

DEFAULT_DATA_PATH = Path("data/raw/ESP_Data_wells_Anonymized_Rev0_Sin_Well_Name.csv")
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def run_training(
    data_path: str | Path = DEFAULT_DATA_PATH,
    test_size: float = 0.2,
    impute_strategy: str = "median",
    fit_intercept: bool = True,
    random_state: int = 42,
) -> tuple[object, dict[str, float]]:
    """Run the complete ESP regression workflow and track it with MLflow."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

    data = clean_data(load_data(data_path))
    features = data.drop(columns=TARGET_COLUMN)
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    model = build_model(build_preprocessor(x_train, impute_strategy), fit_intercept)
    project_run_id = os.getenv("MLFLOW_RUN_ID")
    if project_run_id:
        run_context = mlflow.start_run(run_id=project_run_id)
    else:
        mlflow.set_experiment("ESP_Power_Regression")
        run_context = mlflow.start_run(run_name="linear_regression")

    with run_context:
        parameters = {
            "data_path": str(data_path),
            "test_size": test_size,
            "impute_strategy": impute_strategy,
            "fit_intercept": fit_intercept,
            "random_state": random_state,
            "model_type": "LinearRegression",
        }
        if project_run_id:
            existing_parameters = mlflow.get_run(project_run_id).data.params
            parameters = {key: value for key, value in parameters.items() if key not in existing_parameters}
        mlflow.log_params(parameters)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = evaluate_model(y_test, predictions)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pd.DataFrame({"actual": y_test, "predicted": predictions}).to_csv(
                temp_path / "predictions.csv", index=False
            )
            save_actual_vs_predicted(y_test, predictions, temp_path / "actual_vs_predicted.png")
            mlflow.log_artifacts(temp_dir, artifact_path="evaluation")

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=["numpy.dtype"],
        )

    return model, metrics


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected true or false")
    return normalized == "true"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the ESP power model.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--impute-strategy", choices=("mean", "median", "most_frequent"), default="median"
    )
    parser.add_argument("--fit-intercept", type=parse_bool, default=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    _, metrics = run_training(
        data_path=args.data_path,
        test_size=args.test_size,
        impute_strategy=args.impute_strategy,
        fit_intercept=args.fit_intercept,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
