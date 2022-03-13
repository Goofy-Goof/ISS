from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
import logging
import tensorflow as tf
from .config import img_width, img_height

_log = logging.getLogger(__name__)


def_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
def_metrics = ['accuracy']
basic_model_optimizer = 'adam'
eff_net_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)


def create_basic_model(num_classes, tpu_strategy):
    _log.info('Creating basic model')
    with tpu_strategy.scope():
        model = Sequential([
            layers.Resizing(img_height, img_width),
            layers.Rescaling(1. / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax', name='prediction')
        ], name='basic_convolutional_network')
        model.compile(optimizer=basic_model_optimizer, loss=def_loss, metrics=def_metrics)
    return model


def create_eff_net_frozen(num_classes, tpu_strategy):
    _log.info('Creating EfficientNetB0 frozen')
    # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    with tpu_strategy.scope():
        inputs = layers.Input(shape=(img_height, img_width, 3))
        conv_base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        conv_base.trainable = False
        # Rebuild top
        x = layers.GlobalAveragePooling2D(name='avg_pool')(conv_base.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2, name='top_dropout')(x)
        outputs = layers.Dense(num_classes, activation='softmax', name='prediction')(x)
        # Compile
        func_model = tf.keras.Model(inputs, outputs, name='eff_net_frozen')
        func_model.compile(optimizer=eff_net_optimizer, loss=def_loss, metrics=def_metrics)
    return func_model


def create_eff_net_trainable(num_classes, tpu_strategy):
    with tpu_strategy.scope():
        eff_net = EfficientNetB0(include_top=True, weights=None, classes=num_classes)
        eff_net.compile(optimizer=eff_net_optimizer, loss=def_loss, metrics=def_metrics)
    return eff_net
