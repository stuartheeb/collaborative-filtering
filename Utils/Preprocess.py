import numpy as np
from surprise import Dataset
from surprise import Reader
import pandas as pd

def preprocess(data,number_of_users,number_of_movies):
    mean_rating = []
    std_rating = []
    norm_data = np.zeros((number_of_users, number_of_movies))
    for i in range(number_of_movies):
        total_movie_rating = []
        for j in range(number_of_users):
            if data[j][i] != -1:
                total_movie_rating.append(data[j][i])
        mean_movie_rating = 0
        std_movie_rating = 0

        if len(total_movie_rating) != 0:
            mean_movie_rating = np.mean(total_movie_rating)
            std_movie_rating = np.std(total_movie_rating)
        mean_rating.append(mean_movie_rating)
        std_rating.append(std_movie_rating)
        for j in range(number_of_users):
            if data[j][i] != -1:
                norm_data[j][i] = (float(data[j][i] - mean_movie_rating)) / std_movie_rating

    return norm_data,mean_rating,std_rating

def surprise_preprocess(data,number_of_users,number_of_movies):
    norm_data, mean_rating, std_rating = preprocess(data, number_of_users, number_of_movies)
    userID = []
    itemID = []
    rating = []
    for i in range(number_of_users):
        for j in range(number_of_movies):
            userID.append(i)
            itemID.append(j)
            rating.append(norm_data[i][j])

    ratings_dict = {'itemID': itemID,
                    'userID': userID,
                    'rating': rating}
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    reader = Reader(rating_scale=(-7, 3))
    surprise_data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader=reader)
    trainset = surprise_data.build_full_trainset()
    return trainset,norm_data,mean_rating,std_rating


def kg_preprocess(data,number_of_users,number_of_movies):
    userID = []
    itemID = []
    rating = []
    for i in range(number_of_users):
        for j in range(number_of_movies):
            if data[i][j] != -1:
                userID.append(str(i))
                itemID.append(str(j))
                rating.append(str(data[i][j]))

    ratings_dict = {'userID': userID,'rating': rating,'itemID': itemID}
    df = pd.DataFrame(ratings_dict)
    return df


