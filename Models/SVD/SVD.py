import numpy as np
from Utils.CreateMatrix import Create_Matrix
from Utils.Preprocess import preprocess
from Utils.PostProcess import post_process
from Evaluation.evaluation import evaluate
from Utils.submission import submission


def SVD(A,k,number_of_users,number_of_movies):
    k_singular_values = k
    number_of_singular_values = min(number_of_users, number_of_movies)
    assert(k_singular_values <= number_of_singular_values), "choose correct number of singular values"
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.zeros((number_of_movies, number_of_movies))
    S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])
    return U.dot(S).dot(Vt), U, Vt

def SVD_model(number_of_users,number_of_movies,n_factors_svd,n_iterations_svd):
    model_str = "SVD_model" + "_" + str(n_factors_svd) + "_" + str(n_iterations_svd)
    print(model_str)
    print("--Creating Matrix--")
    data, mask_data, users, movies, predictions = Create_Matrix(number_of_users, number_of_movies)
    print("--Data Preprocessing--")
    A, mean_rating, std_rating = preprocess(data, number_of_users, number_of_movies)
    print("--Perform SVD--")
    A, U, Vt = SVD(A, n_factors_svd, number_of_users, number_of_movies)
    for i in range(n_iterations_svd - 1):
        A, U, Vt = SVD(A, n_factors_svd, number_of_users, number_of_movies)
    print("--Post Process Data--")
    predict_matrix = post_process(A, mean_rating, std_rating, number_of_users, number_of_movies)
    print("--Evaluate--")
    evaluate(predict_matrix, users, movies, predictions,model_str)
    submission(predict_matrix, number_of_users, number_of_movies,model_str)