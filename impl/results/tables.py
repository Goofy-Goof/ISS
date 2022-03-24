from pymongo import MongoClient
import pandas as pd
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


# replace with real credentials during results
mongo_uri = os.environ['MONGO_URI']
db = MongoClient(mongo_uri).iss.results


def create_table(nn):
    query = {'model_type': nn, 'dataset': 'tf_flowers'}
    projection = {
        'downstream_epochs': 1,
        'pretext_epochs': 1,
        'total_test_images': 1,
        'epsilon': 1,
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
    df.to_csv(f'tables/{nn}.csv',
              columns=['downstream_epochs', 'pretext_trainers', 'pretext_epochs', 'correct_classification',
                       'miss_classification', 'epsilon'])
    df.to_latex(f'tables/{nn}.tex',
                columns=['downstream_epochs', 'pretext_trainers', 'pretext_epochs', 'correct_classification',
                         'miss_classification', 'epsilon'])


if __name__ == '__main__':
    create_table('basic_convolutional_network')
    create_table('efficientnetb0')
