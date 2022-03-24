from adversarial import adversarial_round
import logging
from datetime import datetime
from dataset import create_flowers_ds
from pretext import PretextTrainer
from utils import prediction_round
from pymongo import MongoClient
from config import mongo_connection_uri, batch_size, logs_dir
import tensorflow as tf

_log = logging.getLogger(__name__)

def_callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
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


def eval_round(model_constr, pretext_trainers: [PretextTrainer], device_strategy, pretext_epochs):
    # model_constr is a.e create_eff_net_frozen(...)
    ds = create_flowers_ds()
    NN = model_constr(ds.num_classes, device_strategy)
    _log.info(f'Evaluating {NN.name}')
    _log.info(f'Pretext trainers = {[i.name for i in pretext_trainers]}')
    start = datetime.now()
    if len(pretext_trainers) != 0:
        _log.info(f'Training pretext for {pretext_epochs} epochs and downstream for {downstream_epochs}')
        for trainer in pretext_trainers:
            trainer.train_pretrext_task(dataset, NN, device_strategy, pretext_epochs)
        freeze_conv_layers(NN)
    else:
        _log.info(f'Training downstream for {downstream_epochs} epochs')
    NN.fit(dataset.train, validation_data=dataset.val, epochs=downstream_epochs, callbacks=callbacks())
    ev = NN.evaluate(dataset.val)
    _log.info('Started prediction round')
    pred_img, pred_label = prediction_round(dataset, NN)
    _log.info('Started adversarial round')
    epsilons = adversarial_round(pred_img, pred_label, NN)
    end = datetime.now()
    persist_result(NN, dataset, start, end, downstream_epochs, pretext_epochs, pretext_trainers,
                   len(pred_label), epsilons, ev)

