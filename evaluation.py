import numpy as np


def RMSE(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.average((x - y) ** 2))


def evaluate(
        ratings: np.ndarray,
        users: np.ndarray,
        items: np.ndarray,
        predictions: np.ndarray,
        model_name: str
) -> float:
    test_predictions = ratings[users, items]
    rmse_score = RMSE(test_predictions, predictions)
    print(f"RMSE using model {model_name} is: {rmse_score:.5f}")
    return rmse_score
