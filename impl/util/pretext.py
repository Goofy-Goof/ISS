from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers
from .config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
from functools import reduce
from itertools import permutations
from .dataset import load_dataset, resize_ds, configure_ds, Dataset
from .model import DEF_OPTIMIZER, DEF_LOSS, DEF_METRIC
import numpy as np


class PretextTrainer(ABC):
    name: str

    def __init__(self, pretext_label_num):
        self.pretext_label_num = pretext_label_num

    @staticmethod
    def _replace_output_layer(model, num):
        old_layers = model.layers
        # remove last dense from list
        old_layers = old_layers[:-1]
        predictions = layers.Dense(num, activation='softmax', name='prediction')(old_layers[-1].output)
        new_model = tf.keras.Model(inputs=model.inputs, outputs=predictions, name=model.name)
        new_model.compile(optimizer=DEF_OPTIMIZER, loss=DEF_LOSS, metrics=DEF_METRIC)
        return new_model

    def train_pretrext_task(self, dataset: Dataset, model: tf.keras.Model, device_strategy, epochs, callbacks_list):
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
def rotate(x, y, k):
    """
    :param x: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
    :param y: original label of shape `[batch]` or just int
    :param k: A scalar integer tensor. The number of times the image(s) are rotated by 90 degrees.
    :return: Tensor of rotated images, and tensor of new pseudo-labels
    """
    return tf.image.rot90(image=x, k=k), (y * 0 + k)


class RotationPretextTrainer(PretextTrainer):
    name = 'rotation'

    def __init__(self):
        super().__init__(pretext_label_num=4)

    def _map_to_pretext_dataset(self, train, val):
        train_full = []
        val_full = []
        for i in range(4):
            train_full.append(train.map(lambda xt, yt: rotate(xt, yt, i)))
            val_full.append(val.map(lambda xv, yv: rotate(xv, yv, i)))
        pr_train, pr_val = self._concetenate_ds(train_full, val_full)
        return pr_train, pr_val


@tf.function
def make_puzzle(x, y, perm, perm_label):
    """
    :param x: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
    :param y: original label of shape `[batch]` or just int
    :param perm: permutation list a.e [2, 3, 4, 1] will create from ['a', 'b', 'c', 'd'] a puzzle ['b', 'c', 'd', 'a']
    :param perm_label: pseudo label for puzzle
    :return: Tensor of created puzzles, tensor of pseudo labels
    """
    tile_size = IMG_HEIGHT // 2, IMG_WIDTH // 2
    image_shape = tf.shape(x)
    tile_rows = tf.reshape(x, [image_shape[0], image_shape[1], -1, tile_size[0], image_shape[3]])
    serial_tiles = tf.transpose(tile_rows, [0, 2, 1, 3, 4])
    img_parts = tf.reshape(serial_tiles, [image_shape[0], -1, tile_size[0], tile_size[1], image_shape[3]])
    puzzle = tf.convert_to_tensor([img_parts[:, i] for i in perm])
    puzzle = tf.transpose(puzzle, [1, 0, 2, 3, 4])
    tile_width = puzzle.shape[2]
    serialized_tiles = tf.reshape(puzzle, [image_shape[0], -1, image_shape[1], tile_width, image_shape[3]])
    rowwise_tiles = tf.transpose(serialized_tiles, [0, 2, 1, 3, 4])
    puzzle = tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2], image_shape[3]])
    return puzzle, (y * 0 + perm_label)


class JigsawPretextTrainer(PretextTrainer):
    name = 'jigsaw'
    possible_perm = list(permutations([0, 1, 2, 3]))

    def __init__(self):
        super().__init__(pretext_label_num=24)

    def _map_to_pretext_dataset(self, train, val):
        train_full = []
        val_full = []
        chosen_perm_index = np.random.choice(range(24), 4)
        chosen_perm = [self.possible_perm[i] for i in chosen_perm_index]
        for i, perm in enumerate(chosen_perm):
            train_full.append(train.map(lambda xt, yt: make_puzzle(xt, yt, perm, i)))
            val_full.append(val.map(lambda xv, yv: make_puzzle(xv, yv, perm, i)))
        pr_train, pr_val = self._concetenate_ds(train_full, val_full)
        return pr_train, pr_val


def freeze_conv_layers(model):
    # Freezing the Convolutional Layers while keeping Dense layers as Trainable
    for layer in model.layers:
        if str(layer.name).find('conv') == -1:
            layer.trainable = True
        else:
            layer.trainable = False
    return model


class TransferLearningPretextTrainer(PretextTrainer):
    name = 'transfer-learning'
    transfer_ds_name = 'imagenette'

    def __init__(self):
        super(TransferLearningPretextTrainer, self).__init__(pretext_label_num=10)

    def _map_to_pretext_dataset(self, train, val):
        # Nothing to do here
        pass

    def _create_pretext_dataset(self, name):
        # We just load other ds for image classification
        print(f'Loading dataset {self.transfer_ds_name}')
        (train, val), metadata = load_dataset(name=self.transfer_ds_name, split=['train[:80%]', 'train[80%:]'])
        class_names = metadata.features['label'].names
        num_classes = metadata.features['label'].num_classes
        print(
            f'Found {(len(train) + len(val)) * BATCH_SIZE}'
            f'datapoints belonging to {num_classes} classes:\n'
            f'{class_names}\n '
            f'Using {len(train) * BATCH_SIZE} for training, {len(val) * BATCH_SIZE} '
        )
        train = configure_ds(resize_ds(train))
        val = configure_ds(resize_ds(val))
        return train, val
