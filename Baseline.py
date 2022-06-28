from CreateMatrix import Create_Matrix
from Preprocess import preprocess
from SVD import SVD
from ALS import runALS
from PostProcess import post_process
from evaluation import evaluate
from submission import submission

def baseline(number_of_users,number_of_movies,n_factors_svd,n_factors_als,n_iterations_svd,n_iterations_als,lambda_):
    model_str = "baseline"
    print(model_str)
    print("--Create Matrix--")
    data,mask_data,users,movies,predictions = Create_Matrix(number_of_users,number_of_movies)
    print("--Pre process--")
    A,mean_rating,std_rating = preprocess(data,number_of_users,number_of_movies)
    print("--Apply SVD--")
    B, U, Vt = SVD(A, n_factors_svd, number_of_users, number_of_movies)
    for i in range(n_iterations_svd-1):
        B, U, Vt = SVD(B, n_factors_svd,number_of_users,number_of_movies)
    print("--Apply ALS--")
    predict_matrix = runALS(A, mask_data, n_factors_als, n_iterations_als, lambda_, U, Vt)
    print("--Post Process--")
    predict_matrix = post_process(predict_matrix,mean_rating,std_rating,number_of_users,number_of_movies)
    print("--Evaluate--")
    evaluate(predict_matrix,users, movies, predictions, model_str)
    submission(predict_matrix,number_of_users,number_of_movies,model_str)












