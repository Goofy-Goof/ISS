import tensorflow_datasets as tfds
from .config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
import tensorflow as tf


def load_dataset(name, split):
    return tfds.load(name=name, split=split, try_gcs=True, as_supervised=True,
                     with_info=True, batch_size=BATCH_SIZE)


def resize_ds(ds):
    return ds.map(
        lambda x, y: (tf.image.resize(x, (IMG_HEIGHT, IMG_WIDTH)), y)
    )


def configure_ds(ds):
    return ds.cache().shuffle(100).cache().prefetch(tf.data.experimental.AUTOTUNE)


class Dataset:
    def __init__(self, name):
        self.name = name
        print(f'Loading dataset: {name}')
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
        print(
            f'Found {(len(self.train) + len(self.val) + len(self.test)) * BATCH_SIZE} '
            f'datapoints belonging to {self.num_classes} classes:\n'
            f'{self.class_names}\n '
            f'Using {len(self.train) * BATCH_SIZE} for training, {len(self.val) * BATCH_SIZE} '
            f'for validation, and reserved {len(self.test) * BATCH_SIZE} for testing'
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
    print('Creating dataset TF Flowers')
    return Dataset(ds_on_gcs_names[0])
