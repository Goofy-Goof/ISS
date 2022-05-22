import numpy as np
import matplotlib.pyplot as plt


def _query_results(res_db, task, pr_epochs):
    res = [i for i in res_db.find({
        'model_type': 'efficientnetb0',
        'pretext_task': task,
        'pretext_epochs': {'$in': pr_epochs}
    }).sort('pretext_epochs')]
    return res


def get_acc_and_miss(res_db, task, pr_epochs):
    res = _query_results(res_db, task, pr_epochs)
    predicted = np.array([
        np.mean(
            [*map(lambda xx: xx['predicted'],
                  filter(lambda x: x['pretext_epochs'] == pr_ep, res))]
        ) for pr_ep in pr_epochs])
    miss = np.array([
        np.mean(
            [*map(lambda xx: xx['miss_classified'],
                  filter(lambda x: x['pretext_epochs'] == pr_ep, res))]
        ) for pr_ep in pr_epochs])
    total_images = np.array([
        np.mean(
            [*map(lambda xx: xx['total_test_images'],
                  filter(lambda x: x['pretext_epochs'] == pr_ep, res))]
        ) for pr_ep in pr_epochs])
    accuracy = np.round(predicted / total_images * 100, 3)
    miss_rate = np.round(miss / predicted * 100)
    return accuracy, miss_rate


def plot(jigsaw, rotation, transfer, baseline, pr_epochs, y_label, file_name):
    fig, ax = plt.subplots()

    plt.plot(pr_epochs, jigsaw, marker='o', label='jigsaw', color='blue')
    plt.plot(pr_epochs, rotation, marker='o', label='rotation', color='red')
    plt.plot(pr_epochs, transfer, marker='o', label='transfer learning', color='green')
    plt.axhline(baseline, color='magenta', linestyle='-', label='no pretext')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_xlabel('# pretext epochs')
    ax.set_xticks(pr_epochs)
    ax.set_xticklabels(pr_epochs)
    ax.legend()
    fig.tight_layout()
    return plt.savefig('../paper/images/' + file_name)


def get_baseline(res_db):
    no_pr_res = [i for i in res_db.find({'model_type': 'efficientnetb0', 'pretext_task': {'$exists': False}})]

    no_pr_predicted = np.array([i['predicted'] for i in no_pr_res]).mean()
    no_pr_miss = np.array([i['miss_classified'] for i in no_pr_res]).mean()
    no_pr_total_images = np.array([i['total_test_images'] for i in no_pr_res]).mean()

    no_pr_accuracy = no_pr_predicted / no_pr_total_images * 100
    no_pr_miss_rate = no_pr_miss / no_pr_predicted * 100
    return no_pr_accuracy, no_pr_miss_rate
