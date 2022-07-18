* [Kaggle competition](https://www.kaggle.com/competitions/cil-collaborative-filtering-2022/data)
* [data_train.csv](data/data_train.csv)
* [sampleSubmission.csv](data/sampleSubmission.csv)

Separate python notebooks for the different methods and models.


DELETE BELOW




* python main.py --model svd --svd_latent_factors 3 --svd_iterations 10 

## Setup

1. Use an existing virtual environment or create a new one for this project.
2. From the project root, install the required python packages by running:
```bash
pip install -r requirements.txt
```

## Models

### SVD

The SVD model requires:

1. A .csv file containing the observed user ratings of the items (defaults to the training data given in Kaggle).
2. The total number of users.
3. The total number of items.
4. The number of latent factors.

Use the following command line to understand how to run the SVD model:

```bash
python yuvalSVD.py --help
```
