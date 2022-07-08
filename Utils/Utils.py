import numpy as np
from datetime import datetime
from typing import Tuple


def get_curr_timestamp() -> str:
    return datetime.now().strftime("%y-%m-%d_%H-%M-%S")


def construct_ratings_matrix(
        users: np.ndarray,
        items: np.ndarray,
        predictions: np.ndarray,
        n_users: int,
        n_items: int
) -> Tuple[np.ndarray, np.ndarray]:
    ratings = np.full((n_users, n_items), fill_value=0, dtype=float)
    observed = np.full((n_users, n_items), fill_value=False)
    for r, c, v in zip(users, items, predictions):
        observed[r][c] = True
        ratings[r][c] = v

    return ratings, observed


def extract_ratings_from_matrix(
        predicted_ratings: np.ndarray,
        users: np.ndarray,
        items: np.ndarray
) -> np.ndarray:
    """
    Extracts the entries of the
    :param predicted_ratings:
    :param users:
    :param items:
    :return:
    """
    # returns predictions for the users-movies combinations specified based on a full m \times n matrix
    assert (len(users) == len(items)), "users-movies combinations specified should have equal length"
    return np.array([predicted_ratings[r, c] for r, c in zip(users, items)])
