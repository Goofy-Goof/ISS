from util.dataset import create_flowers_ds
from util.models import create_eff_net_trainable
from util.tpu import init_tpu
from util.evaluation import def_callbacks
from util.utils import prediction_round, persist_result
from util.adversarial import adversarial_round
from datetime import datetime


def find_opt_down_epochs():
    down_epochs = [10, 20, 30, 40, 50]
    strategy = init_tpu()
    for ep in down_epochs:
        start = datetime.now()
        ds = create_flowers_ds()
        NN = create_eff_net_trainable(ds.num_classes, strategy)
        NN.fit(ds.train, validation_data=ds.val, epochs=ep, callbacks=def_callbacks)
        print('Started prediction round')
        pred_img, pred_label = prediction_round(ds, NN)
        print('Started adversarial round')
        epsilons = adversarial_round(pred_img, pred_label, NN)
        end = datetime.now()
        persist_result(model_name=NN.name, dataset=ds, start=start, end=end, downstream_epochs=ep,
                       pretext_epochs=None, pr_trainer=None, predicted_num=len(pred_label), epsilons=epsilons)


if __name__ == '__main__':
    for i in range(10):
        find_opt_down_epochs()
