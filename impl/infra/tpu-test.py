import tensorflow as tf


@tf.function
def add_fn(x, y):
    z = x + y
    return z


if __name__ == '__main__':
    print("Tensorflow version " + tf.__version__)

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='your-tpu-name')  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    x = tf.constant(1.)
    y = tf.constant(1.)
    z = strategy.run(add_fn, args=(x, y))
    print(z)
