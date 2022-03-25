import argparse
from util.tpu import init_tpu
from util.evaluation import eval_no_pretext, eval_jigsaw, eval_rotation, eval_eff_net_pre_trained, find_opt_down_epochs
from util.config import EVALUATION_ITERATIONS


def main(task):
    if task == 'rotation':
        tpu = init_tpu()
        print('Evaluating rotation')
        print('-' * 50)
        for i in range(EVALUATION_ITERATIONS):
            print(f'Iteration -> {i}')
            print('-' * 50)
            eval_rotation(strategy=tpu)
        return
    if task == 'jigsaw':
        print('Evaluating jigsaw')
        print('-' * 50)
        tpu = init_tpu()
        for i in range(EVALUATION_ITERATIONS):
            print(f'Iteration -> {i}')
            print('-' * 50)
            eval_jigsaw(strategy=tpu)
        return
    if task == 'None':
        print('Evaluating no pretext')
        print('-' * 50)
        tpu = init_tpu()
        for i in range(EVALUATION_ITERATIONS):
            print(f'Iteration -> {i}')
            print('-' * 50)
            eval_no_pretext(strategy=tpu)
        return
    if task == 'transfer':
        print('Evaluating pre-trained EffNet')
        print('-' * 50)
        tpu = init_tpu()
        for i in range(EVALUATION_ITERATIONS):
            print(f'Iteration -> {i}')
            print('-' * 50)
            eval_eff_net_pre_trained(strategy=tpu)
        return
    if task == 'epochs':
        print('Evaluating optimal downstream epochs number')
        print('-' * 50)
        tpu = init_tpu()
        for i in range(EVALUATION_ITERATIONS):
            print(f'Iteration -> {i}')
            print('-' * 50)
            find_opt_down_epochs(tpu)
    raise Exception(f'Unknown task {task}')


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # Switch
    parser.add_argument('task', type=str, help='Evaluation task type')
    args = parser.parse_args()
    main(args.task)
