import numpy as np
from tables import db
import matplotlib.pyplot as plt
import pandas as pd


query = {
        'model_type': 'eff_net_frozen',
        'dataset': 'tf_flowers',
        'pretext_trainers': '-'
    }


def plot_eff_net_transfer_metrics():
    cursor = list(db.find(query).sort('downstream_epochs'))
    epochs = [i['downstream_epochs'] for i in cursor]
    epsilon = [i['epsilon'] for i in cursor]

    total_images = np.array([i['total_test_images'] for i in cursor])
    guessed = np.array([i['successfully_predicted'] for i in cursor])
    accuracy = guessed / total_images * 100

    fooled = np.array([i['fooled_times'] for i in cursor])
    miss = fooled / guessed * 100

    plt.plot(epochs, epsilon, 'bo-')
    plt.xlabel('# downstream epochs')
    plt.ylabel('$\overline{\epsilon}$')
    plt.savefig('plots/effnet_transfer_eps.png')
    plt.show()

    plt.plot(epochs, accuracy, 'ro-')
    plt.xlabel('# downstream epochs')
    plt.ylabel('accuracy %')
    plt.savefig('plots/effnet_transfer_acc.png')
    plt.show()

    plt.plot(epochs, miss, 'go-')
    plt.xlabel('# downstream epochs')
    plt.ylabel('missclassification %')
    plt.savefig('plots/effnet_transfer_miss.png')
    plt.show()


def create_table():
    cursor = db.find(query)
    df = pd.DataFrame(list(cursor))
    df['epsilon'] = df['epsilon'].round(3)
    df['correct_classification'] = (df['successfully_predicted'] / df['total_test_images'] * 100).round(3)
    df['miss_classification'] = (df['fooled_times'] / df['successfully_predicted'] * 100).round(3)
    df.to_csv(f'tables/effnet_transfer.csv', columns=['downstream_epochs', 'correct_classification', 'miss_classification', 'epsilon'])
    df.to_latex(f'tables/effnet_transfer.tex', columns=['downstream_epochs', 'correct_classification', 'miss_classification', 'epsilon'])


if __name__ == '__main__':
    #plot_eff_net_transfer_metrics()
    create_table()
