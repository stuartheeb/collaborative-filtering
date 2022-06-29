from sklearn.metrics import mean_squared_error
import math
import numpy as np

rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))


# test our predictions with the true values
def get_score(predictions, target_values):
    return rmse(predictions, target_values)

def extract_prediction_from_full_matrix(reconstructed_matrix, users, movies):
    # returns predictions for the users-movies combinations specified based on a full m \times n matrix
    assert(len(users) == len(movies)), "users-movies combinations specified should have equal length"
    predictions = np.zeros(len(users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]

    return predictions

def evaluate(predict_matrix,users,movies,predictions,model):
    test_predictions = extract_prediction_from_full_matrix(predict_matrix,users,movies)
    print("RMSE using"+" " + model + " is: {:.4f}".format(get_score(test_predictions,predictions)))
    return get_score(test_predictions,predictions)
