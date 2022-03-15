from pymongo import MongoClient
import pandas as pd
from numpy import round
import os
from env import set_env


set_env()


def map_pretext_name(x):
    if x == '-':
        return ''
    if x == ['RotationPretextTrainer']:
        return 'rotation'
    elif x == ['JigsawPretextTrainer']:
        return 'jigsaw'
    else:
        return 'rotation + jigsaw'


# replace with real credentials during evaluation
mongo_uri = os.environ['MONGO_URI']
db = MongoClient(mongo_uri).iss.results


def create_table(query):

    projection = {
        'downstream_epochs': 1,
        'pretext_epochs': 1,
        'total_test_images': 1,
        'epsilon':1,
        'fooled_times': 1,
        'successfully_predicted': 1,
        'pretext_trainers': 1
    }
    cursor = db.find(query, projection)
    df = pd.DataFrame(list(cursor))
    df['pretext_trainers'] = df['pretext_trainers'].map(map_pretext_name)
    df['epsilon'] = df['epsilon'].round(3)
    df['correct_classification'] = (df['successfully_predicted'] / df['total_test_images'] * 100).round(3)
    df['miss_classification'] = (df['fooled_times'] / df['successfully_predicted'] * 100).round(3)
    df.to_csv('tables/basic_nn.csv', columns=['downstream_epochs', 'pretext_trainers', 'pretext_epochs', 'correct_classification', 'miss_classification', 'epsilon'])
    df.to_latex('tables/basic_nn.tex', columns=['downstream_epochs', 'pretext_trainers', 'pretext_epochs', 'correct_classification', 'miss_classification', 'epsilon'])


if __name__ == '__main__':
    query = {'model_type': 'basic_convolutional_network', 'dataset': 'tf_flowers'}
    create_table(query)
