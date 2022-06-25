import pandas as pd
import numpy as np



sub_pd = pd.read_csv('data/sampleSubmission.csv',index_col='Id')

def extract_users_items_predictions(data_pd):
    users, movies = [np.squeeze(arr) for arr in np.split(data_pd.index.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

def submission(predict_matrix,number_of_users,number_of_movies,model):
    users, movies, subpred = extract_users_items_predictions(sub_pd)
    subdata = np.full((number_of_users, number_of_movies), 0)
    for user, movie, pred in zip(users, movies, subpred):
        subdata[user][movie] = pred
    Id = []
    pred = []
    for j in range(number_of_movies):
        for i in range(number_of_users):
            if subdata[i][j] != 0:
                Id.append("r"+str(i+1)+"_c"+str(j+1))
                pred.append(predict_matrix[i][j])
    sub_pd['Prediction'] = pred
    sub_pd.to_csv(model+".csv")
