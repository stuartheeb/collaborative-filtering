# CIL 2022 Project Repository

Team: Runtime Terror

## Overview


## Content

The project contains the following methods, runnable as self-contained and independent Jupyter notebooks,  for collaborative filtering:

### Matrix Factorization

* The baseline solution - SVD + ALS on a normalized dataset ([notebook](notebooks/Baseline.ipynb)).

### Neural Networks

*

### Factorization Machines

*

## Setup

1. Start by creating a virtual environment. Using conda,
   ```bash
   conda create --name cil_runtime_terror python=3.6.13
   ```
2. Activate the virtual environment. Using conda,
   ```bash
   conda activate cil_runtime_terror
   ```
3. Go to the project root and install the dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Links and Data

* [Kaggle competition](https://www.kaggle.com/competitions/cil-collaborative-filtering-2022/data)
* [data_train.csv](data/data_train.csv)
* [sampleSubmission.csv](data/sampleSubmission.csv)
