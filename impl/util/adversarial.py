import tensorflow as tf
from .utils import predict_batch, add_to_arr
import numpy as np
from .config import EPSILONS


def eval_signed_grad_batch(image: np.ndarray, label: np.ndarray, model):
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


def create_adv_pattern_batch(img_under_test, labels_under_test, model):
    perturbations_arr = eval_signed_grad_batch(img_under_test, labels_under_test, model)
    return 255. * perturbations_arr


def create_adversarial_image_batch(img_under_test, adv_pattern_batch, eps) -> np.ndarray:
    adv_img_arr = img_under_test + eps * adv_pattern_batch
    adv_img_arr_clipped = tf.clip_by_value(adv_img_arr, clip_value_min=0., clip_value_max=255.)
    return adv_img_arr_clipped


def adversarial_round(images: np.ndarray, labels: np.ndarray, model) -> np.ndarray:
    img_under_test = images
    labels_under_test = labels
    missclassified_epsilons = np.array([])
    adv_pattern = create_adv_pattern_batch(img_under_test, labels_under_test, model)
    for eps in EPSILONS:
        adv_img_arr = create_adversarial_image_batch(img_under_test, adv_pattern, eps)
        predicted_labels = predict_batch(adv_img_arr, model)
        wrong_prediction_indexes = np.argwhere(labels_under_test != predicted_labels).flatten()
        missclassified_epsilons = add_to_arr(missclassified_epsilons,
                                             np.full(fill_value=eps, shape=wrong_prediction_indexes.shape))
        img_under_test = np.delete(img_under_test, wrong_prediction_indexes, axis=0)
        labels_under_test = np.delete(labels_under_test, wrong_prediction_indexes)
        adv_pattern = np.delete(adv_pattern, wrong_prediction_indexes, axis=0)
        if img_under_test.size == 0:
            break
    print(f'We have managed to fool out network {len(missclassified_epsilons)} times out of {len(labels)} '
          f'with average epsilon {np.mean(missclassified_epsilons)}')
    return missclassified_epsilons
