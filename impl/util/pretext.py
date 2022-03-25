from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers
from .config import IMG_WIDTH, IMG_HEIGHT
from functools import reduce
from itertools import permutations
from .dataset import load_dataset, resize_ds, configure_ds, Dataset
from .models import def_metrics, def_loss, def_optimizer
import numpy as np


class PretextTrainer(ABC):

    def __init__(self, pretext_label_num):
        self.pretext_label_num = pretext_label_num

    @staticmethod
    def _replace_output_layer(model, num):
        old_layers = model.layers
        # remove last dense from list
        old_layers = old_layers[:-1]
        predictions = layers.Dense(num, activation='softmax', name='prediction')(old_layers[-1].output)
        new_model = tf.keras.Model(inputs=model.inputs, outputs=predictions, name=model.name)
        new_model.compile(optimizer=def_optimizer, loss=def_loss, metrics=def_metrics)
        return new_model

    def train_pretrext_task(self, dataset: Dataset, model, device_strategy, epochs, callbacks_list):
        print(f'Training pretext with {self.name}')
        ds_train, ds_val = self._create_pretext_dataset(dataset.name)
        with device_strategy.scope():
            p_model = self._replace_output_layer(model, self.pretext_label_num)
        p_model.fit(ds_train, epochs=epochs, validation_data=ds_val, callbacks=callbacks_list)
        with device_strategy.scope():
            og_model = self._replace_output_layer(model, dataset.num_classes)
        return og_model

    def _create_pretext_dataset(self, name):
        print(f'Preparing pretext dataset for {self.name}')
        (train, val), _ = load_dataset(name, ['train[:80%]', 'train[80%:]'])
        train = resize_ds(train)
        val = resize_ds(val)
        new_train, new_val = self._map_to_pretext_dataset(train, val)
        new_train = configure_ds(new_train)
        new_val = configure_ds(new_val)
        return new_train, new_val

    @abstractmethod
    def _map_to_pretext_dataset(self, train, val):
        # The only not generic part for all pretext tasks
        pass

    @staticmethod
    def _concetenate_ds(train, val):
        train_full = reduce(lambda a1, a2: a1.concatenate(a2), train)
        val_full = reduce(lambda a1, a2: a1.concatenate(a2), val)
        return train_full, val_full


@tf.function
def _rotate(x, y, k):
    return x, y * 0 + k


class RotationPretextTrainer(PretextTrainer):
    name = 'rotation'

    def __init__(self):
        super().__init__(4)

    def _map_to_pretext_dataset(self, train, val):
        train_full = []
        val_full = []
        for i in range(4):
            train_full.append(train.map(lambda xt, yt: _rotate(xt, yt, i)))
            val_full.append(val.map(lambda xv, yv: _rotate(xv, yv, i)))
        pr_train, pr_val = self._concetenate_ds(train_full, val_full)
        return pr_train, pr_val


@tf.function
def _make_puzzle(batch, og_label, perm, perm_label):
    """
    back in the days, only god and I knew how this function worked, now god is dead and I have have no idea...
    :param batch: images
    :param og_label: label in originak batch
    :param perm: index of permutation
    :param perm_label: label of genrated permuted batch
    :return:
    """
    tile_size = IMG_HEIGHT // 2, IMG_WIDTH // 2
    image_shape = tf.shape(batch)
    tile_rows = tf.reshape(batch, [image_shape[0], image_shape[1], -1, tile_size[0], image_shape[3]])
    serial_tiles = tf.transpose(tile_rows, [0, 2, 1, 3, 4])
    img_parts = tf.reshape(serial_tiles, [image_shape[0], -1, tile_size[0], tile_size[1], image_shape[3]])
    puzzle = tf.convert_to_tensor([img_parts[:, i] for i in perm])
    puzzle = tf.transpose(puzzle, [1, 0, 2, 3, 4])
    tile_width = puzzle.shape[2]
    serialized_tiles = tf.reshape(puzzle, [image_shape[0], -1, image_shape[1], tile_width, image_shape[3]])
    rowwise_tiles = tf.transpose(serialized_tiles, [0, 2, 1, 3, 4])
    puzzle = tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2], image_shape[3]])
    return puzzle, og_label * 0 + perm_label


class JigsawPretextTrainer(PretextTrainer):
    name = 'jigsaw'

    def __init__(self):
        super().__init__(24)

    def _map_to_pretext_dataset(self, train, val):
        train_full = []
        val_full = []
        possible_perm = list(permutations([0, 1, 2, 3]))
        chosen_perm_index = np.random.choice(range(24), 4)
        chosen_perm = [possible_perm[i] for i in chosen_perm_index]
        for i, perm in enumerate(chosen_perm):
            train_full.append(train.map(lambda xt, yt: _make_puzzle(xt, yt, perm, i)))
            val_full.append(val.map(lambda xv, yv: _make_puzzle(xv, yv, perm, i)))
        pr_train, pr_val = self._concetenate_ds(train_full, val_full)
        return pr_train, pr_val
