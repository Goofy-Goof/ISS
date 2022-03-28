import matplotlib.pyplot as plt
import numpy as np
from .dataset import Dataset
from .config import BATCH_SIZE
import tensorflow as tf


def cpu_strategy():
    return tf.distribute.OneDeviceStrategy(device="/cpu:0")


def init_tpu():
    print('Connecting to TPU')
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    return strategy


def show_image(img: np.ndarray):
    """
    :param img: single image as np array
    :return:
    """
    plt.imshow(img.astype('uint8'))
    plt.axis("off")
    plt.show()


def save_image(img, name):
    plt.imshow(img.astype('uint8'))
    plt.axis("off")
    plt.savefig(name)


def predict_batch(img: np.ndarray, model) -> np.ndarray:
    """
    :param img: batch of images
    :param model: model which should predict
    :param class_names: names of classes
    :return: batch of predicted labes
    """
    return predict_batch_with_probs(img, model)[0]


def predict_batch_with_probs(img: np.ndarray, model):
    predictions_batch = model.predict(img)
    score_batch = tf.nn.softmax(predictions_batch)
    label_batch = np.argmax(score_batch, axis=1)
    probs = np.max(score_batch, axis=1)
    return label_batch, probs


def add_to_arr(target_arr: np.ndarray, elements: np.ndarray, axis=0):
    if target_arr.size == 0:
        target_arr = np.array(elements)
    else:
        target_arr = np.append(target_arr, elements, axis)
    return target_arr


def prediction_round(dataset: Dataset, model) -> (np.ndarray, np.ndarray):
    correct_predicted_images = np.array([])
    correct_predicted_labels = np.array([])
    data = [(x, y) for x, y in [batch for batch in dataset.test]]
    for img_batch, label_batch in data:
        predicted_labels = predict_batch(img_batch, model)
        correct_prediction_indexes = np.argwhere(predicted_labels == label_batch).flatten()
        correct_predicted_images = add_to_arr(correct_predicted_images, img_batch.numpy()[correct_prediction_indexes])
        correct_predicted_labels = add_to_arr(correct_predicted_labels, label_batch.numpy()[correct_prediction_indexes])
    print(f'{len(correct_predicted_labels)} out of {len(data) * BATCH_SIZE} have been correctly classified')
    return correct_predicted_images, correct_predicted_labels


def decode_label(ds: Dataset, label: int):
    return ds.class_names[label]
