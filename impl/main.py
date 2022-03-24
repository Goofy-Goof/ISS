import argparse
from util.tpu import init_tpu
from util.evaluation import eval_no_pretext, eval_jigsaw, eval_rotation, eval_eff_net_pre_trained, find_opt_down_epochs

OPTIMAL_DOWNSTREAM_EPOCHS = 0
RETRIES = 5


def main(task):
    if task == 'rotation':
        tpu = init_tpu()
        for i in range(RETRIES):
            eval_rotation(downstream_epochs=OPTIMAL_DOWNSTREAM_EPOCHS, strategy=tpu)
        return
    if task == 'jigsaw':
        tpu = init_tpu()
        for i in range(RETRIES):
            eval_jigsaw(downstream_epochs=OPTIMAL_DOWNSTREAM_EPOCHS, strategy=tpu)
        return
    if task == 'None':
        tpu = init_tpu()
        for i in range(RETRIES):
            eval_no_pretext(downstream_epochs=OPTIMAL_DOWNSTREAM_EPOCHS, strategy=tpu)
        return
    if task == 'transfer':
        tpu = init_tpu()
        for i in range(RETRIES):
            eval_eff_net_pre_trained(downstream_epochs=OPTIMAL_DOWNSTREAM_EPOCHS, strategy=tpu)
        return
    if task == 'epochs':
        tpu = init_tpu()
        for i in range(RETRIES):
            find_opt_down_epochs(tpu)
    raise Exception(f'Unknown task {task}')


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # Switch
    parser.add_argument('task', type=str, help='Evaluation task type')
    args = parser.parse_args()
    print(f'task = {args.task}')
    main(args.task)
