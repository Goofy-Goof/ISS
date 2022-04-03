import argparse
from argparse import BooleanOptionalAction
from util.utils import init_tpu
from util.evaluation import eval_no_pretext, eval_jigsaw, eval_rotation, eval_transfer_learning
from util.config import PRETEXT_EPOCHS, DOWNSTREAM_EPOCHS
import tensorflow as tf


def main(task, device, iterations, pretext_epochs, downstream_epochs):
    if task not in ['rotation', 'jigsaw', 'none', 'transfer']:
        raise ValueError(f'Unknown task {task}')
    print(f'Evaluating {task}')
    print('-' * 50)
    if task == 'rotation':
        eval_rotation(strategy=device, pretext_epochs=pretext_epochs, downstream_epochs=downstream_epochs, iterations=iterations)
        return
    if task == 'jigsaw':
        eval_jigsaw(strategy=device, pretext_epochs=pretext_epochs, downstream_epochs=downstream_epochs, iterations=iterations)
        return
    if task == 'none':
        eval_no_pretext(strategy=device, downstream_epochs=downstream_epochs, iterations=iterations)
        return
    if task == 'transfer':
        eval_transfer_learning(strategy=device, downstream_epochs=downstream_epochs, iterations=iterations, pretext_epochs=pretext_epochs)
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
    parser.add_argument(
        '--use-cpu',
        type=bool,
        action=BooleanOptionalAction,
        default=False
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.use_cpu:
        _device = tf.distribute.OneDeviceStrategy('/cpu:0')
    else:
        _device = init_tpu()
    main(
        task=args.task,
        device=_device,
        iterations=args.iterations,
        pretext_epochs=args.pretext_epochs,
        downstream_epochs=args.downstream_epochs
    )
