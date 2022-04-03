import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

DEF_LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
DEF_METRIC = ['accuracy']
DEF_OPTIMIZER = 'adam'


def create_eff_net_trainable(num_classes, device_strategy):
    print('Creating EfficientNetB0')
    with device_strategy.scope():
        eff_net = EfficientNetB0(include_top=True, weights=None, classes=num_classes)
        eff_net.compile(optimizer=DEF_OPTIMIZER, loss=DEF_LOSS, metrics=DEF_METRIC)
    return eff_net
