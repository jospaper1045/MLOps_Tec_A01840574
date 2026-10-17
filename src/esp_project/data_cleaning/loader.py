from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the semicolon-delimited ESP dataset."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")
    return pd.read_csv(data_path, sep=";")
