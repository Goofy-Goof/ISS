import tensorflow as tf
from .utils import predict_batch, add_to_arr
import numpy as np
import logging

epsilons = [0.01, 0.1, 0.15]
_log = logging.getLogger(__name__)


def _create_adversarial_pattern(image: np.ndarray, label: np.ndarray, model):
    img_tensor = tf.convert_to_tensor(image)
    label_tensor = tf.convert_to_tensor(label)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)
        loss = model.loss(label_tensor, prediction)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, img_tensor)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def adversarial_round(images: np.ndarray, labels: np.ndarray, model) -> np.ndarray:
    img_under_test = images
    labels_under_test = labels
    missclassified_epsilons = np.array([])
    for eps in epsilons:
        perturbations_arr = _create_adversarial_pattern(img_under_test, labels_under_test, model)
        adv_img_arr = tf.clip_by_value(img_under_test + (eps * 255. * perturbations_arr), clip_value_min=0.,
                                       clip_value_max=255.)
        predicted_labels = predict_batch(adv_img_arr, model)
        wrong_prediction_indexes = np.argwhere(labels_under_test != predicted_labels).flatten()
        missclassified_epsilons = add_to_arr(missclassified_epsilons,
                                             np.full(fill_value=eps, shape=wrong_prediction_indexes.shape))
        img_under_test = np.delete(img_under_test, wrong_prediction_indexes, axis=0)
        labels_under_test = np.delete(labels_under_test, wrong_prediction_indexes)
        if img_under_test.size == 0:
            break
    _log.info(f'We have managed to fool out network {len(missclassified_epsilons)} times out of {len(labels)} '
              f'with average epsilon {np.mean(missclassified_epsilons)}')
    return missclassified_epsilons
