import numpy as np
from pykeen.models.predict import get_tail_prediction_df, get_head_prediction_df, get_relation_prediction_df
from pykeen.models import predict
def post_process(predict_matrix,mean_rating,std_rating,number_of_users,number_of_movies):
    A = np.zeros((number_of_users, number_of_movies))
    for i in range(number_of_movies):
        for j in range(number_of_users):
            A[j][i] = predict_matrix[j][i] * std_rating[i] + mean_rating[i]
    return A


def surprise_post_process(algo,mean_rating,std_rating,number_of_users,number_of_movies):
    predict_matrix = np.zeros((number_of_users, number_of_movies))
    for i in range(number_of_movies):
        for j in range(number_of_users):
            pred = algo.predict(j, i, verbose=False).est
            predict_matrix[j][i] = pred * std_rating[i] + mean_rating[i]
    return predict_matrix

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

'''def kg_postprocess(result,number_of_users,number_of_movies):
    predict_matrix = np.zeros((number_of_users, number_of_movies))
    for i in range(number_of_movies):
        print(i)
        for j in range(number_of_users):
            df = get_relation_prediction_df(result.model, str(j), str(i), triples_factory=result.training)
            predict_matrix[j][i] = df.head(1)['relation_label'].values.astype(float)[0]
            df['relation_label'] = df['relation_label'].astype(float)
            df['score'] = softmax(df['score'])
            predict_matrix[j][i] = sum(df['relation_label'] * df['score'])
    return predict_matrix'''

def kg_postprocess(result,number_of_users,number_of_movies):

    predict_matrix = np.zeros((5,number_of_users, number_of_movies))
    predict_matrix2 = np.zeros((number_of_users, number_of_movies))
    triples = []
    for i in range(number_of_users):
        for j in range(number_of_movies):
            for k in range(5):
                triples.append((str(i),str(k+1),str(j)))
    score_df = predict.predict_triples_df(model=result.model, triples=triples, triples_factory=result.training,batch_size=1024, )

    for index, row in score_df.iterrows():
        if index % 10000 == 0:
            print(index)
        predict_matrix[int(row['relation_label'])-1][int(row['head_label'])][int(row['tail_label'])] = float(row['score'])

    with open('test.npy', 'wb') as f:
        np.save(f, predict_matrix)

    for i in range(number_of_users):
        for j in range(number_of_movies):
            predict_matrix2[i, j] = np.sum(softmax(predict_matrix[:, i, :][:, j]) * (np.arange(5) + 1))

    return predict_matrix2


