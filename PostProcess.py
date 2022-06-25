import numpy as np

def post_process(predict_matrix,mean_rating,std_rating,number_of_users,number_of_movies):
    A = np.zeros((number_of_users, number_of_movies))
    for i in range(number_of_movies):
        for j in range(number_of_users):
            A[j][i] = predict_matrix[j][i] * std_rating[i] + mean_rating[i]
    return A