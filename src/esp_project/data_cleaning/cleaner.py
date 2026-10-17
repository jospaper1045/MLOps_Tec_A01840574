import pandas as pd

CATEGORICAL_COLUMNS = ("Area", "Field", "Bench")
TARGET_COLUMN = "Average_Kilowatt"


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize categories, coerce measurements to numeric and remove invalid rows."""
    missing_columns = set((*CATEGORICAL_COLUMNS, TARGET_COLUMN)) - set(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Required columns are missing: {names}")

    cleaned = data.copy()
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip().replace("", pd.NA)

    numeric_columns = cleaned.columns.difference(CATEGORICAL_COLUMNS)
    cleaned[numeric_columns] = cleaned[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return cleaned.drop_duplicates().dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
