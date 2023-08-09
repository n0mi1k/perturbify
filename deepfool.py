import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys


def deepfool_attack(image, model, num_classes, max_iter, pertMulti):
    # Copy the input image
    image = tf.identity(image)

    # Obtain initial prediction
    f_image = model(image[None, ...])[0]

    # Initialize variables
    r_tot = tf.zeros_like(image)
    k_i = tf.argmax(f_image)
    loop_i = 0

    while loop_i < max_iter:
        pert = float('inf')
        gradients = []

        # Compute the gradients w.r.t each class
        for k in range(num_classes):
            with tf.GradientTape() as tape:
                tape.watch(image)
                prediction_k = model(image[None, ...])[0, k]
            grad_k = tape.gradient(prediction_k, image)
            gradients.append(grad_k)
        
        original_grad = gradients[k_i]

        for k in range(num_classes):
            if k != k_i:
                w_k = gradients[k] - original_grad
                f_k = f_image[k] - f_image[k_i]
                pert_k = tf.abs(f_k) / tf.norm(tf.reshape(w_k, [-1]))
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

        # Change pert_multiplier: Default 1
        pertMultiplier = pertMulti
        # compute r_i and r_tot
        r_i =  pertMultiplier * w / tf.norm(tf.reshape(w, [-1]))
        r_tot += r_i

        image += r_tot
        f_image = model(image[None, ...])[0]
        k_i = tf.argmax(f_image)

        loop_i += 1

    return image


