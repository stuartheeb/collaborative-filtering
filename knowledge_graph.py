from CreateMatrix import Create_Matrix
from Preprocess import kg_preprocess
from PostProcess import kg_postprocess
from evaluation import evaluate
from submission import submission
from pykeen.pipeline import pipeline
from pykeen.models import TransE, TuckER, DistMult, MuRE
from pykeen.triples import TriplesFactory
import pandas as pd
from pykeen.models.predict import get_tail_prediction_df, get_head_prediction_df, get_relation_prediction_df

def knowledge_graph(number_of_users,number_of_movies):
    model_str = "KG_model"
    print(model_str)
    print("--Creating Matrix--")
    data, mask_data, users, movies, predictions = Create_Matrix(number_of_users, number_of_movies)
    print("--Data Preprocessing--")
    kg_df = kg_preprocess(data, number_of_users, number_of_movies)
    kg_df.columns = ['head', 'relation', 'tail']
    print("--Apply KG Model--")
    training_path = TriplesFactory.from_labeled_triples(kg_df.values,create_inverse_triples=False,entity_to_id=None,relation_to_id=None,compact_id=None,filter_out_candidate_inverse_relations=True,metadata=None,)
    testing_path = TriplesFactory.from_labeled_triples(kg_df.head(5).values, create_inverse_triples=False, entity_to_id=None,
                                                        relation_to_id=None, compact_id=None,
                                                        filter_out_candidate_inverse_relations=True, metadata=None, )
    result = pipeline(training=training_path,testing=testing_path,model='MuRE',model_kwargs=dict(embedding_dim=200),training_kwargs=dict(num_epochs=100, batch_size=256),)
    print("Post Process")
    predict_matrix = kg_postprocess(result,number_of_users,number_of_movies)
    print("--Evaluate--")
    evaluate(predict_matrix, users, movies, predictions, model_str)
    submission(predict_matrix, number_of_users, number_of_movies, model_str)


knowledge_graph(10000,1000)