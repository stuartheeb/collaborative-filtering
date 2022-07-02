import numpy as np
import argparse
from pathlib import Path
from Parser import read_csv, write_csv, DEFAULT_INPUT, RESULTS_DIR
from Utils import get_curr_timestamp, construct_ratings_matrix
from NormalizeRatingsPerItem import pre_process, post_process
from evaluation import evaluate
from typing import Optional


def SVD(A: np.ndarray, k: int, n_users: int, n_items: int) -> np.ndarray:
    """
    TODO please add description

    :param A: matrix to decompose
    :param k: number of singular values
    :param n_users: number of users
    :param n_items: number of items
    :return: the decomposition of A, as a 3-tuple.
    """
    assert(k <= min(n_users, n_items)), "choose correct number of singular values"
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    Sk = np.diag(S[:k])
    return U[:, :k] @ Sk @ VT[:k, :]


def SVD_model(input_path: Path, n_users: int, n_items: int, n_factors: int, output_path: Optional[Path] = None):
    """
    TODO please add description

    :param input_path: path to training data .csv file
    :param n_users: number of users
    :param n_items: number of items
    :param n_factors: number of latent factors
    :param output_path: path to .csv file to write the predictions (optional)
    """
    model_str = f"SVD_model_{n_factors}"
    print(model_str)
    print("--Creating Matrix--")
    users, items, preds = read_csv(input_path)
    ratings, observed = construct_ratings_matrix(users, items, preds, n_users, n_items)
    print("--Data Preprocessing--")
    A, mean_ratings, std_ratings = pre_process(ratings, observed)
    print("--Perform SVD--")
    A = SVD(A, n_factors, n_users, n_items)
    print("--Post Process Data--")
    predict_matrix = post_process(A, mean_ratings, std_ratings)
    predict_matrix = np.round(predict_matrix).astype(int)
    print("--Evaluate--")
    evaluate(predict_matrix, users, items, preds, model_str)
    if output_path is None:
        output_path = RESULTS_DIR.joinpath(f"{model_str}_{get_curr_timestamp()}.csv")
    write_csv(predict_matrix, output_path)
    print(f"Results were saved in file {output_path}.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the SVD model")
    parser.add_argument("--num-users", dest="n_users", required=True, type=int, help="number of users")
    parser.add_argument("--num-items", dest="n_items", required=True, type=int, help="number of items")
    parser.add_argument("--num-factors", dest="n_factors", required=True, type=int, help="number of latent factors")
    parser.add_argument("-i", "--input", dest="input_path", type=Path, default=DEFAULT_INPUT,
                        help=f"path to .csv file containing the model's training data (default: {DEFAULT_INPUT})")
    parser.add_argument("-o", "--output", dest="output_path", type=Path, default=None,
                        help=f"path to .csv file containing the model's prediction (by default will be written to "
                             f"directory {RESULTS_DIR})")
    return parser.parse_args()


if __name__ == "__main__":
    params = parse_arguments()
    SVD_model(params.input_path, params.n_users, params.n_items, params.n_factors)
