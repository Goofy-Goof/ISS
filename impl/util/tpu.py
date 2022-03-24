import tensorflow as tf


class TpuStub:

    @staticmethod
    def scope():
        return tf.device('/CPU:0')


def init_tpu():
    tf.config.set_soft_device_placement(True)
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException(
            'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.TPUStrategy(tpu)
    return tpu_strategy
