import numpy as np
from typing import Tuple


def pre_process(
        ratings: np.ndarray,
        observed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # turn non-observed entries to NaN, so they are excluded from the computations
    nan_ratings = ratings.copy()
    nan_ratings[~observed] = np.nan
    # compute mean and std and normalized the matrix with NaN values
    mean = np.nanmean(nan_ratings, axis=0, keepdims=True)
    std = np.nanstd(nan_ratings, axis=0, keepdims=True)
    norm_ratings = (nan_ratings - mean) / std
    # return the non-observed entries to 0
    norm_ratings[~observed] = 0
    return norm_ratings, mean, std


def post_process(
        ratings: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray
) -> np.ndarray:
    return (ratings * std) + mean
