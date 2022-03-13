import matplotlib.pyplot as plt
import numpy as np
from .dataset_wrapper import Dataset
import logging
from .config import batch_size
import tensorflow as tf

_log = logging.getLogger(__name__)


def show_image(img: np.ndarray):
    """
    :param img: single image as np array
    :return:
    """
    plt.imshow(img.astype('uint8'))
    plt.axis("off")
    plt.show()


def predict_batch(img: np.ndarray, model) -> np.ndarray:
    """
    :param img: batch of images
    :param model: model which should predict
    :param class_names: names of classes
    :return: batch of predicted labes
    """
    predictions_batch = model.predict(img)
    score_batch = tf.nn.softmax(predictions_batch)
    label_batch = np.argmax(score_batch, axis=1)
    return label_batch


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
    _log.info(f'{len(correct_predicted_labels)} out of {len(data) * batch_size} have been correctly classified')
    return correct_predicted_images, correct_predicted_labels
