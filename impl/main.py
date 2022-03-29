import argparse
from util.utils import init_tpu
from util.evaluation import eval_no_pretext, eval_jigsaw, eval_rotation, eval_eff_net_pre_trained
from util.config import PRETEXT_EPOCHS, DOWNSTREAM_EPOCHS


def main(task, iterations, pretext_epochs):
    if task not in ['rotation', 'jigsaw', 'none', 'transfer']:
        raise Exception(f'Unknown task {task}')

    tpu = init_tpu()
    print(f'Evaluating {task}')
    print('-' * 50)
    if task == 'rotation':
        eval_rotation(strategy=tpu, pretext_epochs=pretext_epochs, downstream_epochs=DOWNSTREAM_EPOCHS,
                      iterations=iterations)
        return
    if task == 'jigsaw':
        eval_jigsaw(strategy=tpu, pretext_epochs=pretext_epochs, downstream_epochs=DOWNSTREAM_EPOCHS,
                    iterations=iterations)
        return
    if task == 'none':
        eval_no_pretext(strategy=tpu, downstream_epochs=DOWNSTREAM_EPOCHS, iterations=iterations)
        return
    if task == 'transfer':
        eval_eff_net_pre_trained(strategy=tpu, downstream_epochs=DOWNSTREAM_EPOCHS, iterations=iterations)
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
        help='Iterations to eval',
        default=1
    )
    parser.add_argument(
        '--pretext-epochs',
        type=int,
        nargs='*',
        help='Pretext epochs',
        default=PRETEXT_EPOCHS
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        task=args.task,
        iterations=args.iterations,
        pretext_epochs=args.pretext_epochs,
    )
