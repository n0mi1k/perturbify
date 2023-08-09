import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0  # Normalize the image
    image = image[None, ...]
    return image

# Helper function to extract the predicted label and confidence
def get_predicted_label(probs):
    predicted_label = tf.argmax(probs, axis=-1)
    confidence = tf.reduce_max(probs)
    return predicted_label[0], confidence


def create_adversarial_pattern(loss_object, input_image, input_label, trained_model):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = trained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t. to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def load_image(image_path):
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    return image


def apply_mask(image, mask):
    masked_image = np.multiply(image, mask)
    return masked_image


def save_images(image_path, image, index):
    plt.imshow(image[0])
    plt.axis('off')
    outputFormat = f"{os.path.splitext(image_path)[0]}_fgsm{index}.jpg" 
    plt.savefig(outputFormat, bbox_inches='tight', pad_inches=0)
    print(f"[+] Adverserial Image {outputFormat} Created!")


def create_mask(image, threshold):
    mask = np.ones_like(image)
    mask = mask[0]
    mask[np.abs(image[0]) < threshold] = 0  # Adjust the threshold as per your preference
    mask = np.expand_dims(mask, axis=0)
    return mask
