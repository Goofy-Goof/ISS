import tensorflow_datasets as tfds
from config import data_dir, img_width, img_height, batch_size
import logging
import tensorflow as tf

_log = logging.getLogger(__name__)


def load_dataset(name, split):
    return tfds.load(name=name, split=split, try_gcs=True, as_supervised=True, with_info=True,
                     batch_size=batch_size, data_dir=data_dir)


def resize_ds(ds):
    return ds.map(
        lambda x, y: (tf.image.resize(x, (img_height, img_width)), y)
    )


def configure_ds(ds):
    return ds.cache().shuffle(100).cache().prefetch(tf.data.experimental.AUTOTUNE)


class Dataset:
    def __init__(self, name):
        self.name = name
        _log.info(f'Loading dataset: {name}')
        self._load_dataset()

    @staticmethod
    def _prepare_set(ds):
        return configure_ds(resize_ds(ds))

    def _load_dataset(self):
        (train, val, test), metadata = load_dataset(self.name, ['train[:75%]', 'train[75%:95%]', 'train[95%:]'])
        self.train = self._prepare_set(train)
        self.val = self._prepare_set(val)
        self.test = self._prepare_set(test)
        self.class_names = metadata.features['label'].names
        self.num_classes = metadata.features['label'].num_classes
        _log.info(
            f'Found {(len(self.train) + len(self.val) + len(self.test)) * batch_size} '
            f'datapoints belonging to {self.num_classes} classes:\n'
            f'{self.class_names}\n '
            f'Using {len(self.train) * batch_size} for training, {len(self.val) * batch_size} '
            f'for validation, and reserved {len(self.test) * batch_size} for testing'
        )


ds_on_gcs_names = [
    'tf_flowers',
    'cats_vs_dogs',
    'horses_or_humans',
    'stanford_dogs',
    'fashion_mnist',
    'rock_paper_scissors',
    'colorectal_histology'
]


def create_flowers_ds():
    return Dataset(ds_on_gcs_names[0])
