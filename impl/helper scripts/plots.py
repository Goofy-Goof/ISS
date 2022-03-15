import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tables import db, map_pretext_name


def plot_no_pretext():
    projection = {'epsilon': 1, 'downstream_epochs': 1}
    epochs = []
    eps = []
    for i in [10, 20, 30, 50]:
        query = {'model_type': 'basic_convolutional_network', 'dataset': 'tf_flowers', 'pretext_trainers': '-', 'downstream_epochs': i}
        res = list(db.find(query, projection).sort('until'))
        epochs.append([i['downstream_epochs'] for i in res])
        eps.append([i['epsilon'] for i in res])
    epochs = np.array(epochs).T
    eps = np.array(eps).T
    for i, j in zip(epochs, eps):
        plt.plot(i, j, 'o-')
    plt.ylabel('$\overline{\epsilon}$')
    plt.xlabel('# downstream epochs')
    plt.title('$\overline{\epsilon}$ over 5 evalution rounds without pretext training')
    plt.savefig('plots/basic_nn_no_pretext.png')
    plt.show()


def plot_pretext(task):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    query = {
            'model_type': 'basic_convolutional_network',
            'dataset': 'tf_flowers',
            'pretext_trainers': task
        }
    projection = {'downstream_epochs':1, 'pretext_epochs':1, 'epsilon':1}
    res = list(db.find(query, projection).sort('downstream_epochs'))
    x = [i['pretext_epochs'] for i in res]
    y = [i['downstream_epochs'] for i in res]
    z = [i['epsilon'] for i in res]
    ax.scatter(x, y, z, s=50)
    plt.xlabel('# pretext epochs')
    plt.ylabel('# downstream epochs')
    ax.set_zlabel('$\overline{\epsilon}$', rotation=270)
    plt.title('$\overline{\epsilon}$ for ' + map_pretext_name(task) + ' pretext task')
    plt.savefig(f'plots/basic_nn_{map_pretext_name(task)}.png')
    plt.show()


if __name__ == '__main__':
    plot_no_pretext()
    #plot_pretext(['RotationPretextTrainer'])
