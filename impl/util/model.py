import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from .config import IMG_WIDTH, IMG_HEIGHT

DEF_LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
DEF_METRIC = ['accuracy']
DEF_OPTIMIZER = 'adam'


def create_eff_net_pre_trained(num_classes, device_strategy):
    print('Creating EfficientNetB0 pre-trained')
    with device_strategy.scope():
        inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        conv_base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        conv_base.trainable = False
        # Rebuild top
        x = layers.GlobalAveragePooling2D(name='avg_pool')(conv_base.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2, name='top_dropout')(x)
        outputs = layers.Dense(num_classes, activation='softmax', name='prediction')(x)
        # Compile
        func_model = tf.keras.Model(inputs, outputs, name='eff_net_frozen')
        func_model.compile(optimizer=DEF_OPTIMIZER, loss=DEF_LOSS, metrics=DEF_METRIC)
    return func_model


def create_eff_net_trainable(num_classes, device_strategy):
    print('Creating EfficientNetB0')
    with device_strategy.scope():
        eff_net = EfficientNetB0(include_top=True, weights=None, classes=num_classes)
        eff_net.compile(optimizer=DEF_OPTIMIZER, loss=DEF_LOSS, metrics=DEF_METRIC)
    return eff_net
