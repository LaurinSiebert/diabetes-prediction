import typer
import kagglehub
from pathlib import Path
from loguru import logger
from diabetes_prediction.config import RAW_DATA_DIR


app = typer.Typer()


@app.command()
def download_dataset():
    """Download the Pima Indians Diabetes dataset from Kaggle."""

    logger.info("Downloading Pima Indians Diabetes dataset from Kaggle...")

    # Download the dataset using kagglehub
    dataset_path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

    # Source file path
    source_file = Path(dataset_path) / "diabetes.csv"
    destination_file = RAW_DATA_DIR / "diabetes.csv"

    # Ensure the raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the file to the raw data directory
    if source_file.exists():
        destination_file.write_bytes(source_file.read_bytes())
        logger.success(f"Dataset downloaded to {destination_file}")
    else:
        logger.error("Failed to download the dataset.")



if __name__ == "__main__":
    app()