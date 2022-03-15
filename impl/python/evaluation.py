from .adversarial import adversarial_round
import logging
from datetime import datetime
from .dataset_wrapper import Dataset
from .pretext import PretextTrainer
from .utils import prediction_round
from pymongo import MongoClient
from .config import mongo_connection_uri, batch_size, logs_dir
import tensorflow as tf

_log = logging.getLogger(__name__)


def callbacks():
    #_log_dir = f'{logs_dir}_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    #_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=_log_dir, histogram_freq=1)
    return [
        #_tensorboard_callback,
        tf.keras.callbacks.TerminateOnNaN()
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


def eval_round(model_constr, dataset: Dataset, pretext_trainers: [PretextTrainer], device_strategy,
               downstream_epochs, pretext_epochs):
    # model_constr is a.e create_eff_net_frozen(...)
    NN = model_constr(dataset.num_classes, device_strategy)
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


def persist_result(model, dataset, start, end, downstream_epochs, pretext_epochs,
                   trainer: [PretextTrainer], predicted_num, epsilons, eval):
    pr_tasks = '-' if len(trainer) == 0 else [i.name for i in trainer]
    json = {
        'model_type': model.name,
        'dataset': dataset.name,
        'pretext_trainers': pr_tasks,
        'from': start,
        'until': end,
        'downstream_epochs': downstream_epochs,
        'total_test_images': len(dataset.test) * batch_size,
        'successfully_predicted': predicted_num,
        'fooled_times': len(epsilons),
        'epsilon': epsilons.mean(),
        'loss': eval[0],
        'accuracy': eval[1]
    }
    if len(trainer) != 0:
        json['pretext_epochs'] = pretext_epochs
    _log.info('Test results: {}'.format(json))
    client = MongoClient(mongo_connection_uri)
    db = client.iss
    inserted_id = db.results.insert_one(json).inserted_id
    _log.info('inserted_id = {}'.format(inserted_id))
