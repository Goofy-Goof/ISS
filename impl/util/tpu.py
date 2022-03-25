import tensorflow as tf
from .config import TPU_NAME


class TpuStub:

    @staticmethod
    def scope():
        return tf.device('/CPU:0')


def init_tpu():
    print(f'Connecting to {TPU_NAME} TPU')
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_NAME)  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    return strategy

