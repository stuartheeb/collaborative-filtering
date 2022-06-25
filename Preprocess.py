import numpy as np

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
