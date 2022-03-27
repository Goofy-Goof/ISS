import argparse
from util.tpu import init_tpu
from util.evaluation import eval_no_pretext, eval_jigsaw, eval_rotation, eval_eff_net_pre_trained, find_opt_down_epochs
from util.config import EVALUATION_ITERATIONS, PRETEXT_EPOCHS, DOWNSTREAM_EPOCHS, OPTIMAL_DOWNSTREAM_EPOCHS


def main(task, iterations, pretext_epochs, downstream_epochs, opt_downstream_ep):
    if task == 'rotation':
        tpu = init_tpu()
        print('Evaluating rotation')
        print('-' * 50)
        for i in range(iterations):
            print(f'Iteration -> {i}')
            print('-' * 50)
            eval_rotation(strategy=tpu, pretext_epochs=pretext_epochs, downstream_epochs=opt_downstream_ep)
        return
    if task == 'jigsaw':
        print('Evaluating jigsaw')
        print('-' * 50)
        tpu = init_tpu()
        for i in range(iterations):
            print(f'Iteration -> {i}')
            print('-' * 50)
            eval_jigsaw(strategy=tpu, pretext_epochs=pretext_epochs, downstream_epochs=opt_downstream_ep)
        return
    if task == 'None':
        print('Evaluating no pretext')
        print('-' * 50)
        tpu = init_tpu()
        for i in range(iterations):
            print(f'Iteration -> {i}')
            print('-' * 50)
            eval_no_pretext(strategy=tpu, downstream_epochs=opt_downstream_ep)
        return
    if task == 'transfer':
        print('Evaluating pre-trained EffNet')
        print('-' * 50)
        tpu = init_tpu()
        for i in range(iterations):
            print(f'Iteration -> {i}')
            print('-' * 50)
            eval_eff_net_pre_trained(strategy=tpu, downstream_epochs=opt_downstream_ep)
        return
    if task == 'epochs':
        print('Evaluating optimal downstream epochs number')
        print('-' * 50)
        tpu = init_tpu()
        for i in range(iterations):
            print(f'Iteration -> {i}')
            print('-' * 50)
            find_opt_down_epochs(strategy=tpu, down_epochs=downstream_epochs)
        return
    raise Exception(f'Unknown task {task}')


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
        default=EVALUATION_ITERATIONS
    )
    parser.add_argument(
        '--pretext-epochs',
        type=int,
        nargs='*',
        help='Pretext epochs',
        default=PRETEXT_EPOCHS
    )
    parser.add_argument(
        '--opt-downstream-epochs',
        type=int,
        help='Number of optimal downstream epochs after pretext training',
        default=OPTIMAL_DOWNSTREAM_EPOCHS
    )
    parser.add_argument(
        '--downstream-epochs',
        type=int,
        nargs='*',
        help='Number of downstream epochs (while investigating which to use after pretext)',
        default=DOWNSTREAM_EPOCHS
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(
        task=args.task,
        iterations=args.iterations,
        pretext_epochs=args.pretext_epochs,
        downstream_epochs=args.downstream_epochs,
        opt_downstream_ep=args.opt_downstream_epochs
    )
