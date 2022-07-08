import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Utils.Preprocess import preprocess
from Models.ALS import runALS
from Models.SVD.SVD import SVD
from Utils.PostProcess import post_process
from Evaluation.evaluation import evaluate



def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def Create_Matrix(number_of_users,number_of_movies):
    data_pd = pd.read_csv('data/data_train.csv')
    train, val = train_test_split(data_pd, test_size=0.1)
    train_users, train_movies, train_predictions = extract_users_items_predictions(train)
    val_users, val_movies, val_predictions = extract_users_items_predictions(val)
    train_data = np.full((number_of_users, number_of_movies), -1)
    for user, movie, pred in zip(train_users, train_movies, train_predictions):
        train_data[user][movie] = pred

    total_ratings = 0
    mask_data = np.zeros((number_of_users, number_of_movies))
    for i in range(number_of_users):
        for j in range(number_of_movies):
            if train_data[i][j] != -1:
                total_ratings += 1
                mask_data[i][j] = 1

    rating_percentage = total_ratings / (number_of_users * number_of_movies)
    print('There are a total of ' + str(total_ratings) + ' ratings, which means that ' + str(rating_percentage) + ' of all entries exist')

    return train_data,mask_data,train_users,train_movies,train_predictions,val_users,val_movies,val_predictions

def validation(number_of_users,number_of_movies,n_factors_svd,n_factors_als,n_iterations_svd,n_iterations_als,lambda_):
    model_str = "validation"
    print(model_str)
    print("--Create Matrix--")
    data,mask_data,users,movies,predictions,val_users,val_movies,val_predictions = Create_Matrix(number_of_users,number_of_movies)
    print("--Pre process--")
    A, mean_rating, std_rating = preprocess(data, number_of_users, number_of_movies)
    print("--Apply SVD--")
    B, U, Vt = SVD(A, n_factors_svd, number_of_users, number_of_movies)
    for i in range(n_iterations_svd - 1):
        B, U, Vt = SVD(B, n_factors_svd, number_of_users, number_of_movies)
    print("--Apply ALS--")
    predict_matrix = runALS(A, mask_data, n_factors_als, n_iterations_als, lambda_, U, Vt)
    print("--Post Process--")
    predict_matrix = post_process(predict_matrix,mean_rating,std_rating,number_of_users,number_of_movies)
    print("--Evaluate--")
    evaluate(predict_matrix,users, movies, predictions, "train")
    score = evaluate(predict_matrix,val_users,val_movies,val_predictions,"test")
    evaluate(predict_matrix, np.append(users, val_users), np.append(movies, val_movies),np.append(predictions, val_predictions), "total")
    return score

'''def hopt():
    n_factors_svd = np.arange(3,10)
    n_factors_als = np.arange(3,10)
    n_iterations_svd = np.arange(1,20)
    lambda_ = np.arange(0.05,0.2,0.05)
    best_score = 2
    best_params = (0,0,0,0)
    for k1 in n_factors_svd:
        for k2 in n_factors_als:
            for it in n_iterations_svd:
                for l in lambda_:
                    score = validation(10000, 1000, k1, k2, it, 20, l)
                    if score < best_score:
                        best_score = score
                        best_params = (k1,k2,it,l)
    print(best_params)
    print(best_score)




hopt()'''