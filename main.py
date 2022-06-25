import argparse
from Baseline import baseline
from SVD import SVD_model
from ALS import ALS_model

ap = argparse.ArgumentParser()
ap.add_argument("-model","--model",type=str,default='baseline',help="Model/Algo to run")
ap.add_argument("-sk", "--svd_latent_factors", type=int, default=3,
                help="Latent Factors SVD")
ap.add_argument("-si", "--svd_iterations", type=int, default=1,
                help="Number SVD iterations")
ap.add_argument("-alsk", "--als_latent_factors", type=int, default=3,
                help="Latent Factors ALS")
ap.add_argument("-alsi", "--als_iterations", type=int, default=20,
                help="Number ALS iterations")
ap.add_argument("-reg", "--regularization", type=float, default=0.1,
                help="lambda ALS")

args = vars(ap.parse_args())

model = args['model']
SVD_Latent_Factors = args['svd_latent_factors']
SVD_Iterations = args['svd_iterations']
ALS_Latent_Factors = args['als_latent_factors']
ALS_Iterations = args['als_iterations']
ALS_lambda = args['regularization']

print("Running model = " + model)
if model == 'baseline':
    baseline(10000,1000,SVD_Latent_Factors,ALS_Latent_Factors,SVD_Iterations,ALS_Iterations,ALS_lambda)
if model == 'svd':
    SVD_model(10000,1000,SVD_Latent_Factors,SVD_Iterations)
if model == 'als':
    ALS_model(10000,1000,ALS_Latent_Factors,ALS_Iterations,ALS_lambda)


