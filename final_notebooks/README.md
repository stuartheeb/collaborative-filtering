# CIL 2022 Project Repository

Team: Runtime Terror

## Running the code

The notebooks can be run on Google Colab. For the case of X, it is best if the code is run locally due to RAM requirements.

## Content

<<<<<<< HEAD
The project contains the following methods, runnable as self-contained and independent Jupyter notebooks, for collaborative filtering. [This](Bayesian_Factorization_Machines/Ordered_Probit/Bayesian_SVDpp_flipped_with_Embeddings_Ordered_Probit.ipynb) is the notebook that produced the best score.
=======
The project contains the following methods, runnable as self-contained and independent Jupyter notebooks, for collaborative filtering:

### Matrix Factorization

* The baseline solution - SVD + ALS on a normalized dataset ([notebook](Matrix_Factorization/baseline.ipynb)).
* The baseline solution with a k-NN initialization approach ([notebook](Matrix_Factorization/baseline_knn.ipynb)).
* The baseline solution with a Gaussian processes approach ([notebook](Matrix_Factorization/baseline_gp.ipynb)).
* Improved SVD ([notebook](Matrix_Factorization/improved_svd.ipynb)).
* The baseline solution with Improved SVD ([notebook](Matrix_Factorization/baseline_improved_svd.ipynb)).

### Neural Networks

* Neural Net for Collaborative Filtering ([notebook](Neural_Networks/neural_nets.ipynb)).
* Sparse FC ([notebook](Neural_Networks/sparseFC.ipynb)).
* Knowledge Graphs ([notebook](Neural_Networks/Knowledge_Graphs.ipynb)).

### Factorization Machines

* Bayesian Factorization Machine ([notebook](Bayesian_Factorization_Machines/BFM_Regressors/Bayesian_Factorization_Machine.ipynb))
* Bayesian SVD++ ([notebook](Bayesian_Factorization_Machines/BFM_Regressors/Bayesian_SVDpp.ipynb))
* Bayesian SVD++ flipped ([notebook](Bayesian_Factorization_Machines/BFM_Regressors/Bayesian_SVDpp_flipped.ipynb))
* Weighted Bayesian SVD++ flipped ([notebook](Bayesian_Factorization_Machines/BFM_Regressors/Weighted_Bayesian_SVDpp_flipped.ipynb))
* Bayesian SVD++ flipped with rating ([notebook](Bayesian_Factorization_Machines/BFM_Regressors/Bayesian_SVDpp_flipped_rating.ipynb))
* Bayesian SVD++ flipped with embeddings ([notebook](Bayesian_Factorization_Machines/BFM_Regressors/Bayesian_SVDpp_flipped_with_Embeddings.ipynb))
* Bayesian SVD++ flipped Ordered Probit ([notebook](Bayesian_Factorization_Machines/Ordered_Probit/Bayesian_SVDpp_flipped_Ordered_Probit.ipynb))
* Bayesian SVD++ flipped Ordered Probit with rating ([notebook](Bayesian_Factorization_Machines/Ordered_Probit/Bayesian_SVDpp_flipped_rating_Order_Probit.ipynb))
* Bayesian SVD++ flipped Ordered Probit with embeddings ([notebook](Bayesian_Factorization_Machines/Ordered_Probit/Bayesian_SVDpp_flipped_with_Embeddings_Ordered_Probit.ipynb))


>>>>>>> 96c6fd468ee27f754fd042f387fd00ee882c5ed1

## Data Preprocessing

For data input processing and preparation, we used some code found in the notebook for the CIL 2021 course, available [here](https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_1.ipynb).
