from typing import Optional
import tensorflow as tf
from .dataset import create_flowers_ds
from .models import create_eff_net_trainable, create_eff_net_pre_trained
from pymongo import MongoClient
from .pretext import PretextTrainer, RotationPretextTrainer, JigsawPretextTrainer
from .utils import prediction_round
from .adversarial import adversarial_round
from datetime import datetime
from .config import BATCH_SIZE, MONGO_URI
import numpy as np

def_callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.01)
]


def freeze_conv_layers(model):
    if model.name == 'efficient_net_frozen':
        return model
    # Freezing the Convolutional Layers while keeping Dense layers as Trainable
    for layer in model.layers:
        if str(layer.name).find('conv') == -1:
            layer.trainable = True
        else:
            layer.trainable = False
    return model


def find_opt_down_epochs(strategy, down_epochs):
    for ep in down_epochs:
        start = datetime.now()
        ds = create_flowers_ds()
        NN = create_eff_net_trainable(ds.num_classes, strategy)
        NN.fit(ds.train, validation_data=ds.val, epochs=ep, callbacks=def_callbacks)
        print('Started prediction round')
        pred_img, pred_label = prediction_round(ds, NN)
        end = datetime.now()
        persist_downstream_epochs_result(model_name=NN.name, dataset=ds, start=start, end=end,
                                         downstream_epochs=ep, predicted_num=len(pred_label))


def eval_no_pretext(strategy, downstream_epochs):
    _eval_round(model_constr=create_eff_net_trainable, strategy=strategy, pr_task=None, pr_epochs=None,
                downstream_epochs=downstream_epochs)


def eval_eff_net_pre_trained(strategy, downstream_epochs):
    _eval_round(model_constr=create_eff_net_pre_trained, strategy=strategy, pr_task=None,
                pr_epochs=None, downstream_epochs=downstream_epochs)


def eval_rotation(strategy, pretext_epochs, downstream_epochs):
    for i in pretext_epochs:
        _eval_round(model_constr=create_eff_net_trainable, strategy=strategy, pr_task=RotationPretextTrainer(),
                    pr_epochs=i, downstream_epochs=downstream_epochs)


def eval_jigsaw(strategy, pretext_epochs, downstream_epochs):
    for i in pretext_epochs:
        _eval_round(model_constr=create_eff_net_trainable, strategy=strategy, pr_task=JigsawPretextTrainer(),
                    pr_epochs=i, downstream_epochs=downstream_epochs)


def _eval_round(model_constr, strategy, pr_task: Optional[PretextTrainer], pr_epochs, downstream_epochs):
    ds = create_flowers_ds()
    NN = model_constr(ds.num_classes, strategy)
    print(f'Evaluating {NN.name}')
    print(f'Pretext task = {pr_task}')
    start = datetime.now()
    if pr_task is not None:
        print(f'Training pretext for {pr_epochs} epochs')
        pr_task.train_pretrext_task(ds, NN, strategy, pr_epochs, def_callbacks)
        NN = freeze_conv_layers(NN)
    print(f'Training downstream for {downstream_epochs} epochs')
    NN.fit(ds.train, validation_data=ds.val, epochs=downstream_epochs, callbacks=def_callbacks)
    print('Started prediction round')
    pred_img, pred_label = prediction_round(ds, NN)
    print('Started adversarial round')
    epsilons = adversarial_round(pred_img, pred_label, NN)
    end = datetime.now()
    persist_result(model_name=NN.name, dataset=ds, start=start, end=end, downstream_epochs=downstream_epochs,
                   pretext_epochs=pr_epochs, pr_trainer=pr_task, predicted_num=len(pred_label), epsilons=epsilons)


def persist_result(model_name, dataset, start, end, downstream_epochs, pretext_epochs, pr_trainer,
                   predicted_num, epsilons):
    json = {
        'model_type': model_name,
        'dataset': dataset.name,
        'from': start,
        'until': end,
        'downstream_epochs': downstream_epochs,
        'total_test_images': len(dataset.test) * BATCH_SIZE,
        'predicted': predicted_num,
        'miss_classified': len(epsilons),
        'epsilon_mean': np.mean(epsilons)
    }
    if pr_trainer is not None:
        json['pretext_epochs'] = pretext_epochs
        json['pretext_task'] = pr_trainer.name
    print('Test results: {}'.format(json))
    client = MongoClient(MONGO_URI)
    db = client.iss
    inserted_id = db.results3.insert_one(json).inserted_id
    print('inserted_id = {}'.format(inserted_id))


def persist_downstream_epochs_result(model_name, dataset, start, end, downstream_epochs, predicted_num):
    json = {
        'model_type': model_name,
        'dataset': dataset.name,
        'from': start,
        'until': end,
        'downstream_epochs': downstream_epochs,
        'total_test_images': len(dataset.test) * BATCH_SIZE,
        'predicted': predicted_num,
    }
    print('Test results: {}'.format(json))
    client = MongoClient(MONGO_URI)
    db = client.iss
    inserted_id = db.results3.insert_one(json).inserted_id
    print('inserted_id = {}'.format(inserted_id))
