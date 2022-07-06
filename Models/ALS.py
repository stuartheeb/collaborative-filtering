import numpy as np
from Utils.CreateMatrix import Create_Matrix
from Utils.Preprocess import preprocess
from Utils.PostProcess import post_process
from Evaluation.evaluation import evaluate
from Utils.submission import submission


def runALS(A, R, n_factors, n_iterations, lambda_, U=None, Vt=None):
    print ("Initiating ")
    n, m = A.shape
    if U.any() and Vt.any():
        Users = U[:,:n_factors]
        Items = Vt[:n_factors,:]
    else:
        Users = 5 * np.random.rand(n, n_factors)
        Items = 5 * np.random.rand(n_factors, m)

    def get_error(A, Users, Items, R):
        # This calculates the MSE of nonzero elements
        return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)

    MSE_List = []

    print ("Starting Iterations")
    for iter in range(n_iterations):
        for i, Ri in enumerate(R):
            Users[i] = np.linalg.solve(np.dot(Items, np.dot(np.diag(Ri), Items.T)) + lambda_ * np.eye(n_factors),
                                       np.dot(Items, np.dot(np.diag(Ri), A[i].T))).T
        print ("Error after solving for User Matrix: " + str(get_error(A, Users, Items, R)))

        for j, Rj in enumerate(R.T):
            Items[:,j] = np.linalg.solve(np.dot(Users.T, np.dot(np.diag(Rj), Users)) + lambda_ * np.eye(n_factors),
                                     np.dot(Users.T, np.dot(np.diag(Rj), A[:, j])))
        print ("Error after solving for Item Matrix: " + str(get_error(A, Users, Items, R)))

        MSE_List.append(get_error(A, Users, Items, R))
        print ('%sth iteration is complete...' + str(iter))

    print (MSE_List)
    return np.dot(Users,Items)

def ALS_model(number_of_users,number_of_movies,n_factors_als,n_iterations_als,lambda_):
    model_str = "ALS_model" + "_" + str(n_factors_als) + "_" + str(n_iterations_als) + "_" + str(lambda_)
    print(model_str)
    print("--Create Matrix")
    data, mask_data, users, movies, predictions = Create_Matrix(number_of_users, number_of_movies)
    print("--Pre process--")
    A, mean_rating, std_rating = preprocess(data, number_of_users, number_of_movies)
    print("--ALS--")
    predict_matrix = runALS(A, mask_data, n_factors_als, n_iterations_als, lambda_)
    print("--Post Process--")
    predict_matrix = post_process(predict_matrix, mean_rating, std_rating, number_of_users, number_of_movies)
    print("--Evaluate--")
    evaluate(predict_matrix, users, movies, predictions,model_str)
    submission(predict_matrix, number_of_users, number_of_movies,model_str)
