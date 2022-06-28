import numpy as np
from CreateMatrix import Create_Matrix
from Preprocess import surprise_preprocess
from PostProcess import post_process
from evaluation import evaluate
from submission import submission
from surprise import SVD
from surprise import SVDpp
from ALS import runALS

def surpriseSVDALSmodel(number_of_users,number_of_movies,n_factors_svd,n_iterations_svd,lr_all,reg_all,n_factors_als,n_iterations_als, lambda_):
    model_str = "Surprise_SVD_ALS_model" + "_" + str(n_factors_svd) + "_" + str(n_iterations_svd)
    print(model_str)
    print("--Creating Matrix--")
    data, mask_data, users, movies, predictions = Create_Matrix(number_of_users, number_of_movies)
    print("--Data Preprocessing--")
    trainset, A, mean_rating, std_rating = surprise_preprocess(data,number_of_users,number_of_movies)
    print("--Perform SVD--")
    algo = SVD(n_factors=n_factors_svd, n_epochs=n_iterations_svd, lr_all=lr_all, reg_all=reg_all)
    algo.fit(trainset)
    U = algo.pu
    Vt = algo.qi.T
    print("--Apply ALS--")
    predict_matrix = runALS(A, mask_data, n_factors_als, n_iterations_als, lambda_, U, Vt)
    print("--Post Process--")
    predict_matrix = post_process(predict_matrix, mean_rating, std_rating, number_of_users, number_of_movies)
    print("--Evaluate--")
    evaluate(predict_matrix, users, movies, predictions, model_str)
    submission(predict_matrix, number_of_users, number_of_movies, model_str)


def surpriseSVDppALSmodel(number_of_users,number_of_movies,n_factors_svd,n_iterations_svd,lr_all,reg_all,n_factors_als,n_iterations_als, lambda_):
    model_str = "Surprise_SVD_ALS_model" + "_" + str(n_factors_svd) + "_" + str(n_iterations_svd)
    print(model_str)
    print("--Creating Matrix--")
    data, mask_data, users, movies, predictions = Create_Matrix(number_of_users, number_of_movies)
    print("--Data Preprocessing--")
    trainset, A, mean_rating, std_rating = surprise_preprocess(data,number_of_users,number_of_movies)
    print("--Perform SVD--")
    algo = SVDpp(n_factors=n_factors_svd, n_epochs=n_iterations_svd, lr_all=lr_all, reg_all=reg_all)
    algo.fit(trainset)
    U = algo.pu
    Vt = algo.qi.T
    print("--Apply ALS--")
    predict_matrix = runALS(A, mask_data, n_factors_als, n_iterations_als, lambda_, U, Vt)
    print("--Post Process--")
    predict_matrix = post_process(predict_matrix, mean_rating, std_rating, number_of_users, number_of_movies)
    print("--Evaluate--")
    evaluate(predict_matrix, users, movies, predictions, model_str)
    submission(predict_matrix, number_of_users, number_of_movies, model_str)