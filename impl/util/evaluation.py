from typing import Optional
import tensorflow as tf
from .dataset import create_flowers_ds
from pymongo import MongoClient
from .pretext import PretextTrainer, RotationPretextTrainer, JigsawPretextTrainer, freeze_conv_layers, TransferLearningPretextTrainer
from .utils import prediction_round, create_eff_net_trainable
from .adversarial import adversarial_round
from datetime import datetime
from .config import BATCH_SIZE, MONGO_URI

def_callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)
    # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, cooldown=5, verbose=1, min_lr=1e-5)
]


def eval_no_pretext(strategy, downstream_epochs, iterations):
    for i in range(iterations):
        print(f'Iteration -> {i + 1}')
        print('-' * 50)
        _eval_round(model_constr=create_eff_net_trainable, strategy=strategy, pr_task=None, pr_epochs=None,
                    downstream_epochs=downstream_epochs)


def eval_transfer_learning(strategy, pretext_epochs, downstream_epochs, iterations):
    for i in range(iterations):
        print(f'Iteration -> {i + 1}')
        print('-' * 50)
        for j in pretext_epochs:
            _eval_round(model_constr=create_eff_net_trainable, strategy=strategy, pr_task=TransferLearningPretextTrainer(),
                        pr_epochs=j, downstream_epochs=downstream_epochs)


def eval_rotation(strategy, pretext_epochs, downstream_epochs, iterations):
    for i in range(iterations):
        print(f'Iteration -> {i + 1}')
        print('-' * 50)
        for j in pretext_epochs:
            _eval_round(model_constr=create_eff_net_trainable, strategy=strategy, pr_task=RotationPretextTrainer(),
                        pr_epochs=j, downstream_epochs=downstream_epochs)


def eval_jigsaw(strategy, pretext_epochs, downstream_epochs, iterations):
    for i in range(iterations):
        print(f'Iteration -> {i + 1}')
        print('-' * 50)
        for j in pretext_epochs:
            _eval_round(model_constr=create_eff_net_trainable, strategy=strategy, pr_task=JigsawPretextTrainer(),
                        pr_epochs=j, downstream_epochs=downstream_epochs)


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
    miss_num = adversarial_round(pred_img, pred_label, NN)
    end = datetime.now()
    persist_result(model_name=NN.name, dataset=ds, start=start, end=end, downstream_epochs=downstream_epochs,
                   pretext_epochs=pr_epochs, pr_trainer=pr_task, predicted_num=len(pred_label), miss_num=miss_num)


def persist_result(model_name, dataset, start, end, downstream_epochs, pretext_epochs, pr_trainer, predicted_num: int, miss_num: int):
    json = {
        'model_type': model_name,
        'dataset': dataset.name,
        'from': start,
        'until': end,
        'downstream_epochs': downstream_epochs,
        'total_test_images': len(dataset.test) * BATCH_SIZE,
        'predicted': predicted_num,
        'miss_classified': miss_num,
    }
    if pr_trainer is not None:
        json['pretext_epochs'] = pretext_epochs
        json['pretext_task'] = pr_trainer.name
    print('Test results: {}'.format(json))
    client = MongoClient(MONGO_URI)
    db = client.iss
    inserted_id = db.results5.insert_one(json).inserted_id
    print('inserted_id = {}'.format(inserted_id))
