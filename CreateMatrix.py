#Loads the data in csv format and converts to USER-Item interaction Matrix
import pandas as pd
import numpy as np


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def Create_Matrix(number_of_users,number_of_movies):
    data_pd = pd.read_csv('data_train.csv')
    users, movies, predictions = extract_users_items_predictions(data_pd)
    data = np.full((number_of_users, number_of_movies), -1)
    for user, movie, pred in zip(users, movies, predictions):
        data[user][movie] = pred

    mask_data = np.zeros((number_of_users, number_of_movies))
    for i in range(number_of_users):
        for j in range(number_of_movies):
            if data[i][j] != -1:
                mask_data[i][j] = 1

    return data,mask_data,users,movies,predictions


