from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_actual_vs_predicted(
    y_true: pd.Series, y_pred: np.ndarray, output_path: str | Path
) -> Path:
    """Save an actual-versus-predicted diagnostic chart."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(y_true, y_pred, alpha=0.35, s=14)
    axis.plot([lower, upper], [lower, upper], "r--", label="Ideal prediction")
    axis.set(xlabel="Actual kW", ylabel="Predicted kW", title="Actual vs predicted ESP power")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
