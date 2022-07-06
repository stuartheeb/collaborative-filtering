import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


PROJECT_ROOT_PATH = Path(__file__).parent
DEFAULT_INPUT = PROJECT_ROOT_PATH.joinpath(Path(f"data/data_train.csv")).resolve()
OUTPUT_TEMPLATE = PROJECT_ROOT_PATH.joinpath(Path(f"data/sampleSubmission.csv")).resolve()
RESULTS_DIR = PROJECT_ROOT_PATH.joinpath(Path(f"results")).resolve()


def read_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    # extract user and item indices from the Id label in the dataframe
    df = df.join(df.Id.str.extract(r"r(?P<User>\d+)_c(?P<Item>\d+)").astype(int) - 1)
    # extract user, item and prediction triplets from dataframe
    users = df.User.values
    items = df.Item.values
    preds = df.Prediction.values
    return users, items, preds


def write_csv(predicted_ratings: np.ndarray, path: Path):
    # extract required user-item indices from sample file
    users, items, _ = read_csv(OUTPUT_TEMPLATE)
    # construct output file using given user-item predictions
    df = pd.DataFrame({
        "Id": [f"r{r + 1}_c{c + 1}" for r, c in zip(users, items)],
        "Prediction": [predicted_ratings[r, c] for r, c in zip(users, items)]
    })
    df.to_csv(path, index=False)
