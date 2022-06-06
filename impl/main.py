from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from __future__ import annotations

import argparse
import tensorflow as tf # noqa

from util import *


def main(task, device, iterations, pretext_epochs, downstream_epochs):
    if task not in ['rotation', 'jigsaw', 'none', 'transfer']:
        raise ValueError(f'Unknown task {task}')
    print(f'Evaluating {task}')
    print('-' * 50)
    if task == 'rotation':
        eval_rotation(strategy=device, pretext_epochs=pretext_epochs, downstream_epochs=downstream_epochs,
                      iterations=iterations)
        return
    if task == 'jigsaw':
        eval_jigsaw(strategy=device, pretext_epochs=pretext_epochs, downstream_epochs=downstream_epochs,
                    iterations=iterations)
        return
    if task == 'none':
        eval_no_pretext(strategy=device, downstream_epochs=downstream_epochs, iterations=iterations)
        return
    if task == 'transfer':
        eval_transfer_learning(strategy=device, downstream_epochs=downstream_epochs, iterations=iterations,
                               pretext_epochs=pretext_epochs)
        return


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation hyper parameters')
    parser.add_argument(
        'task',
        type=str,
        help='Evaluation task type'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of iterations to eval',
        default=1
    )
    parser.add_argument(
        '--pretext-epochs',
        type=int,
        nargs='*',
        help='Pretext epochs number',
        default=PRETEXT_EPOCHS
    )
    parser.add_argument(
        '--downstream-epochs',
        type=int,
        help='Downstream epochs number',
        default=DOWNSTREAM_EPOCHS
    )
    return parser.parse_args()


'''
use _device = tf.distribute.OneDeviceStrategy() if running locally in CPU
In case of computational GPU clusters, use one from https://www.tensorflow.org/guide/distributed_training
TPUs are the best though :)
'''

if __name__ == '__main__':
    args = parse_args()
    _device = init_tpu()
    # _device = tf.distribute.OneDeviceStrategy()
    main(
        task=args.task,
        device=_device,
        iterations=args.iterations,
        pretext_epochs=args.pretext_epochs,
        downstream_epochs=args.downstream_epochs
    )
