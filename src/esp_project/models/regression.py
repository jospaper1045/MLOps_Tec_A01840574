import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def build_model(preprocessor: ColumnTransformer, fit_intercept: bool = True) -> Pipeline:
    """Combine preprocessing and linear regression in one serializable model."""
    return Pipeline(
        [
            ("preprocessing", preprocessor),
            ("regressor", LinearRegression(fit_intercept=fit_intercept)),
        ]
    )


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
