import argparse
from Baseline import baseline
from SVD import SVD_model
from ALS import ALS_model
from surpriseSVD import surpriseSVDmodel
from surpriseSVD import surpriseSVDppmodel
from SurpriseSVDALS import surpriseSVDALSmodel
from SurpriseSVDALS import surpriseSVDppALSmodel
from validation import validation
from validation_surprise import validation_surprise
from stuart_test import stuart_test

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
ap.add_argument("-lr_all","--lr_all",type=float,default=0.005,help="Surprise lr all param")
ap.add_argument("-reg_all","--reg_all",type=float,default=0.02,help="Surprise reg all param")

args = vars(ap.parse_args())

model = args['model']
SVD_Latent_Factors = args['svd_latent_factors']
SVD_Iterations = args['svd_iterations']
ALS_Latent_Factors = args['als_latent_factors']
ALS_Iterations = args['als_iterations']
ALS_lambda = args['regularization']
Surprise_lr = args['lr_all']
Surprise_reg = args['reg_all']

print("Running model = " + model)
if model == 'baseline':
    baseline(10000,1000,SVD_Latent_Factors,ALS_Latent_Factors,SVD_Iterations,ALS_Iterations,ALS_lambda)
if model == 'svd':
    SVD_model(10000,1000,SVD_Latent_Factors,SVD_Iterations)
if model == 'als':
    ALS_model(10000,1000,ALS_Latent_Factors,ALS_Iterations,ALS_lambda)
if model == 'surprise-svd':
    surpriseSVDmodel(10000,1000,SVD_Latent_Factors,SVD_Iterations,Surprise_lr,Surprise_reg)
if model == 'surprise-svdpp':
    surpriseSVDppmodel(10000,1000,SVD_Latent_Factors,SVD_Iterations,Surprise_lr,Surprise_reg)
if model == 'surprise-svd-als':
    surpriseSVDALSmodel(10000,1000,SVD_Latent_Factors,SVD_Iterations,Surprise_lr,Surprise_reg,ALS_Latent_Factors,ALS_Iterations,ALS_lambda)
if model == 'surprise-svdpp-als':
    surpriseSVDppALSmodel(10000,1000,SVD_Latent_Factors,SVD_Iterations,Surprise_lr,Surprise_reg,ALS_Latent_Factors,ALS_Iterations,ALS_lambda)
if model == 'validation':
    validation(10000,1000,SVD_Latent_Factors,ALS_Latent_Factors,SVD_Iterations,ALS_Iterations,ALS_lambda)
if model == 'validation_surprise':
    validation_surprise(10000,1000,SVD_Latent_Factors,ALS_Latent_Factors,SVD_Iterations,ALS_Iterations,ALS_lambda)
if model == 'stuart_test':
    stuart_test
