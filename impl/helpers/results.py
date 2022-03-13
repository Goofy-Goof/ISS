import pymongo
from pymongo import MongoClient
import pandas as pd
from numpy import round
from datetime import datetime


def map_pretext_name(x):
    if len(x['pretext_trainers']) == 0:
        return '-'
    if x['pretext_trainers'] == ['RotationPretextTrainer']:
        return 'rotation'
    elif x['pretext_trainers'] == ['JigsawPretextTrainer']:
        return 'jigsaw'
    else:
        return 'rotation + jigsaw'


def map_pretext_epochs(x):
    return x['pretext_epochs'] if 'pretext_epochs' in x else '-'


# replace with real credentials during evaluation
mongo_uri = os.environ['MONGO_URI']
model = 'xxx'
dataset = 'xxx'

if __name__ == '__main__':
    client = MongoClient(mongo_uri)
    db = client.iss.results
    query = {'model_type': 'efficientnetb0', 'from': {'$gte': datetime(2022, 1, 18, 00, 00)}}
    result = db.find(query).sort('downstream_epochs', pymongo.ASCENDING)
    result = [x for x in result]
    column_order = ['downstream epochs', 'correctly classified', 'pretext tasks', 'pretext epochs',
                    'missclassified', 'epsilon mean']
    df = pd.DataFrame(columns=column_order)
    df['downstream epochs'] = list(map(lambda x: x['downstream_epochs'], result))
    successfully_predicted = round(list(map(
        lambda x: x['successfully_predicted'] / x['total_test_images'] * 100., result)), 3)
    df['correctly classified'] = list(map(lambda x: str(x) + '%', successfully_predicted))
    df['pretext tasks'] = list(map(map_pretext_name, result))
    df['pretext epochs'] = list(map(map_pretext_epochs, result))
    fooled = round(list(map(lambda x: x['fooled_times'] / x['successfully_predicted'] * 100., result)), 3)
    df['missclassified'] = list(map(lambda x: str(x) + '%', fooled))
    df['epsilon mean'] = round(list(map(lambda x: x['epsilon'], result)), 3)

    df.to_csv('results.csv', columns=column_order)
    df.to_latex('results.txt', columns=column_order)
