import pandas as pd
import numpy as np
from pathlib import Path
from diabetes_prediction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def process_data() -> None:
    """
        Processes the raw diabetes dataset:
        - Replaces invalid zero values with NaN in selected columns
        - Fills missing values with the column mean
        - Saves the cleaned dataset to the processed data directory
    """
    path = Path(RAW_DATA_DIR / "diabetes.csv")
    save_file = Path(PROCESSED_DATA_DIR / "diabetes_processed.csv")

    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at: {path}")


    df = pd.read_csv(path)

    # Clean the Cols that are invalidly 0 and fills them with the mean instead
    invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[invalid_cols] = df[invalid_cols].replace(0, np.nan)
    df[invalid_cols] = df[invalid_cols].fillna(df[invalid_cols].mean())

    # Save processed data
    df.to_csv(save_file, index=False)
    print(f"Processed data saved to {save_file}")


if __name__ == "__main__":
    process_data()