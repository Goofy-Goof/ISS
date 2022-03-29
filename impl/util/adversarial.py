import tensorflow as tf
from .utils import predict_batch
import numpy as np
from .config import EPSILON


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


def adversarial_round(images: np.ndarray, labels: np.ndarray, model) -> int:
    adv_pattern = create_adv_pattern_batch(images, labels, model)
    adv_img_arr = create_adversarial_image_batch(images, adv_pattern, EPSILON)
    predicted_labels = predict_batch(adv_img_arr, model)
    wrong_prediction_indexes = np.argwhere(labels != predicted_labels).flatten()
    print(f'We have managed to fool out network {len(wrong_prediction_indexes)} times out of {len(labels)} ')
    return len(wrong_prediction_indexes)
