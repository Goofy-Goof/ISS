import numpy as np
import matplotlib.pyplot as plt

epsilon_mean = [0.010, 0.013, 0.012, 0.013, 0.028, 0.012, 0.010, 0.011, 0.010, 0.011, 0.011, 0.011,
                0.012, 0.010, 0.011, 0.010]

if __name__ == '__main__':
    model = 'effnet_b0'
    tasks = ['-', 'rotation', 'jigsaw', 'rotation + jigsaw']

    plt.bar(tasks, epsilon_mean[0:4], width=1, edgecolor='white', linewidth=0.7)
    plt.bar(tasks, epsilon_mean[4:8], width=1, edgecolor='white', linewidth=0.7)
    plt.bar(tasks, epsilon_mean[8:12], width=1, edgecolor='white', linewidth=0.7)
    plt.bar(tasks, epsilon_mean[12:], width=1, edgecolor='white', linewidth=0.7)
    plt.legend(['(10, 30)', '(20, 50)', '(30, 10)', '(50, 20)'])
    txt = "1st number in legend represent downstream epochs, 2nd pretext epochs"
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(f'{model}_tasks.png')
    plt.show()

    epochs = ['10', '20', '30', '50']
    plt.bar(epochs, np.take(epsilon_mean, [0, 4, 8, 12]), width=1, edgecolor='white', linewidth=0.7)
    plt.bar(epochs, np.take(epsilon_mean, [1, 5, 9, 13]), width=1, edgecolor='white', linewidth=0.7)
    plt.bar(epochs, np.take(epsilon_mean, [2, 6, 10, 14]), width=1, edgecolor='white', linewidth=0.7)
    plt.bar(epochs, np.take(epsilon_mean, [3, 7, 11, 15]), width=1, edgecolor='white', linewidth=0.7)
    plt.legend(tasks)
    plt.savefig(f'{model}_ep.png')
    plt.show()

    plt.bar(epochs, np.take(epsilon_mean, [9, 13, 1, 5]), width=1, edgecolor='white', linewidth=0.7)
    plt.bar(epochs, np.take(epsilon_mean, [10, 14, 2, 6]), width=1, edgecolor='white', linewidth=0.7)
    plt.bar(epochs, np.take(epsilon_mean, [11, 15, 3, 7]), width=1, edgecolor='white', linewidth=0.7)
    plt.legend(tasks[1:])
    plt.savefig(f'{model}_pret_ep.png')
    plt.show()
