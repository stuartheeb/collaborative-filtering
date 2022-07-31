# CIL 2022 Project Repository

Team: Runtime Terror

## Running the code

The notebooks can be run on Google Colab. For the case of X, it is best if the code is run locally due to RAM requirements.

## Content

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

* ...


## Data Preprocessing

For data input processing and preparation, we used some code found in the notebook for the CIL 2021 course, available [here](https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_1.ipynb).