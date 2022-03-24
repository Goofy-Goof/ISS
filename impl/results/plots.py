import numpy as np
import matplotlib.pyplot as plt
from tables import db

taks = ['-', ['RotationPretextTrainer'], ['JigsawPretextTrainer'], ['RotationPretextTrainer', 'JigsawPretextTrainer']]
marker_size = 50


def plot_epsilon(nn):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    projection = {'downstream_epochs': 1, 'pretext_epochs': 1, 'epsilon': 1}
    for t in taks:
        query = {
            'model_type': nn,
            'dataset': 'tf_flowers',
            'pretext_trainers': t
        }
        res = list(db.find(query, projection).sort('downstream_epochs'))
        y = [i['downstream_epochs'] for i in res]
        z = [i['epsilon'] for i in res]
        if t == '-':
            x = np.zeros(len(z), dtype=float)
        else:
            x = [i['pretext_epochs'] for i in res]
        ax.scatter(x, y, z, s=marker_size)
    plt.xlabel('# pretext epochs')
    plt.ylabel('# downstream epochs')
    ax.set_zlabel('$\overline{\epsilon}$')
    plt.legend(['no pretext', 'rotation', 'jigsaw', 'rotation + jigsaw'])
    plt.savefig(f'plots/{nn}_eps.png')
    plt.show()


def plot_accuracy(nn):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    projection = {'downstream_epochs': 1, 'pretext_epochs': 1, 'successfully_predicted': 1, 'total_test_images': 1}
    for t in taks:
        query = {
            'model_type': nn,
            'dataset': 'tf_flowers',
            'pretext_trainers': t
        }
        res = list(db.find(query, projection).sort('downstream_epochs'))
        y = [i['downstream_epochs'] for i in res]
        total_images = np.array([i['total_test_images'] for i in res])
        guessed = np.array([i['successfully_predicted'] for i in res])
        z = guessed / total_images * 100
        if t == '-':
            x = np.zeros(len(z), dtype=float)
        else:
            x = [i['pretext_epochs'] for i in res]
        ax.scatter(x, y, z, s=marker_size)
    plt.xlabel('# pretext epochs')
    plt.ylabel('# downstream epochs')
    ax.set_zlabel('Accuracy %')
    plt.legend(['no pretext', 'rotation', 'jigsaw', 'rotation + jigsaw'])
    plt.savefig(f'plots/{nn}_acc.png')
    plt.show()


def plot_missclassified(nn):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    projection = {'downstream_epochs': 1, 'pretext_epochs': 1, 'successfully_predicted': 1, 'fooled_times': 1}
    for t in taks:
        query = {
            'model_type': nn,
            'dataset': 'tf_flowers',
            'pretext_trainers': t
        }
        res = list(db.find(query, projection).sort('downstream_epochs'))
        y = [i['downstream_epochs'] for i in res]
        guessed = np.array([i['successfully_predicted'] for i in res])
        fooled = np.array([i['fooled_times'] for i in res])
        z = fooled / guessed * 100
        if t == '-':
            x = np.zeros(len(z), dtype=float)
        else:
            x = [i['pretext_epochs'] for i in res]
        ax.scatter(x, y, z, s=marker_size)
    plt.xlabel('# pretext epochs')
    plt.ylabel('# downstream epochs')
    ax.set_zlabel('Miss classification %')
    plt.legend(['no pretext', 'rotation', 'jigsaw', 'rotation + jigsaw'])
    plt.savefig(f'plots/{nn}_miss.png')
    plt.show()


if __name__ == '__main__':
    plot_epsilon('basic_convolutional_network')
    plot_epsilon('efficientnetb0')

    plot_accuracy('basic_convolutional_network')
    plot_accuracy('efficientnetb0')

    plot_missclassified('basic_convolutional_network')
    plot_missclassified('efficientnetb0')
