import numpy as np
from Utils.CreateMatrix import Create_Matrix
from Utils.Preprocess import surprise_preprocess
from Utils.PostProcess import surprise_post_process
from Evaluation.evaluation import evaluate
from Utils.submission import submission
from surprise import SVD
from surprise import SVDpp

def surpriseSVDmodel(number_of_users,number_of_movies,n_factors_svd,n_iterations_svd,lr_all,reg_all):
    model_str = "Surprise_SVD_model" + "_" + str(n_factors_svd) + "_" + str(n_iterations_svd)
    print(model_str)
    print("--Creating Matrix--")
    data, mask_data, users, movies, predictions = Create_Matrix(number_of_users, number_of_movies)
    print("--Data Preprocessing--")
    trainset,A, mean_rating, std_rating = surprise_preprocess(data,number_of_users,number_of_movies)
    print("--Perform SVD--")
    algo = SVD(n_factors=n_factors_svd, n_epochs=n_iterations_svd, lr_all=lr_all, reg_all=reg_all)
    algo.fit(trainset)
    print("--Post Process Data--")
    predict_matrix = surprise_post_process(algo, mean_rating, std_rating, number_of_users, number_of_movies)
    print("--Evaluate--")
    evaluate(predict_matrix, users, movies, predictions, model_str)
    submission(predict_matrix, number_of_users, number_of_movies, model_str)

def surpriseSVDppmodel(number_of_users,number_of_movies,n_factors_svd,n_iterations_svd,lr_all,reg_all):
    model_str = "Surprise_SVDpp_model" + "_" + str(n_factors_svd) + "_" + str(n_iterations_svd)
    print(model_str)
    print("--Creating Matrix--")
    data, mask_data, users, movies, predictions = Create_Matrix(number_of_users, number_of_movies)
    print("--Data Preprocessing--")
    trainset,A, mean_rating, std_rating = surprise_preprocess(data,number_of_users,number_of_movies)
    print("--Perform SVD--")
    algo = SVDpp(n_factors=n_factors_svd, n_epochs=n_iterations_svd, lr_all=lr_all, reg_all=reg_all)
    algo.fit(trainset)
    print("--Post Process Data--")
    predict_matrix = surprise_post_process(algo, mean_rating, std_rating, number_of_users, number_of_movies)
    print("--Evaluate--")
    evaluate(predict_matrix, users, movies, predictions, model_str)
    submission(predict_matrix, number_of_users, number_of_movies, model_str)