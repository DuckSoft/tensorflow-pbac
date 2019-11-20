from typing import Tuple
from math import pi

import numpy as np
import tensorflow as tf


class PBAC:
    def __call__(self):
        return self.rsf() + self.pa()

    def __init__(self, image: tf.Tensor,
                 lambda1: float = 0.5,
                 lambda2: float = 0.5,
                 sigma: int = 2,
                 n: int = 1,
                 epsilon: float = 0.00001):
        # initialize gaussian constants
        assert sigma >= 1
        assert n >= 1
        self.sigma_multiplier = tf.constant(1. / ((2 * pi) ** (float(n) / 2) * sigma ** n), dtype=tf.float32)
        self.sigma_exponent = tf.constant(-1. / (2 * sigma * sigma), dtype=tf.float32)

        # initialize lambda constants
        self.lambda1 = tf.constant(lambda1, dtype=tf.float32)
        self.lambda2 = tf.constant(lambda2, dtype=tf.float32)

        # initialize others
        self.image = image
        self.epsilon = epsilon

    def pa(self):
        gb_images = self.log_gabor(self.image)
        gb_images = np.transpose(gb_images, (2, 0, 1))

        images = tf.convert_to_tensor(gb_images, tf.complex64)
        images_fft = tf.map_fn(lambda x: tf.signal.fft2d(x), images)
        images_real = tf.math.abs(tf.math.real(images_fft))
        images_imag = tf.math.abs(tf.math.imag(images_fft))

        t = tf.random.truncated_normal(shape=images.shape, mean=0, stddev=1.0, dtype=tf.float32)

        pa_upper = tf.reduce_sum(tf.floor(images_real - images_imag - t), axis=0)
        pa_lower = (tf.reduce_sum(tf.math.abs(images_fft), axis=0) + self.epsilon)
        pa = pa_upper / pa_lower

        return tf.math.exp(-tf.reduce_mean(pa))

    def rsf(self):
        shape_x, shape_y = self.image.shape
        range_x_x, range_x_y = tf.range(shape_x), tf.range(shape_y)

        return tf.reduce_sum(
            tf.map_fn(lambda x_x:
                      tf.reduce_sum(
                          tf.map_fn(lambda x_y:
                                    tf.reduce_sum(
                                        tf.map_fn(lambda y_row:
                                                  tf.reduce_sum(
                                                      tf.map_fn(lambda y:
                                                                (self.gaussian_kernel(self.image[x_x, x_y], y)
                                                                 * tf.square(y - self.image[x_x, x_y]))
                                                                , y_row, dtype=tf.float32
                                                                )),
                                                  self.window_slice(self.image, (x_x, x_y), 1), dtype=tf.float32
                                                  )) * tf.cond(
                                        tf.greater(self.image[x_x, x_y], 0),
                                        lambda: self.lambda1,
                                        lambda: self.lambda2,
                                    ),
                                    range_x_y, dtype=tf.float32)
                      ), range_x_x, dtype=tf.float32))

    @staticmethod
    def window_slice(img, point: Tuple[int, int], radius: int):
        x1, x2 = point[0] - radius, point[0] + radius
        y1, y2 = point[1] - radius, point[1] + radius
        return img[x1:x2 + 1, y1:y2 + 1]

    def gaussian_kernel(self, x_value, y_value):
        return self.sigma_multiplier * tf.exp(tf.square(x_value - y_value) * self.sigma_exponent)

    @staticmethod
    def log_gabor(img, n_scale=3, n_orient=4, min_wave_length=5, mult=1.6, sigma_on_f=0.75, d_theta_on_sigma=1.5):
        """
        :author: Ruibo Liu
        :comments: adapted from existing matlab code
        :license: unlicensed
        """
        image = np.asarray(img)
        rows, cols, = image.shape

        image_fft = np.fft.fft2(image)

        [x, y] = np.meshgrid(
            np.array([i for i in range(-cols // 2, cols // 2)]) / cols,
            np.array([j for j in range(-rows // 2, rows // 2)]) / rows
        )

        radius = np.sqrt(np.power(x, 2) + np.power(y, 2))
        radius[rows // 2, cols // 2] = 1

        theta = np.arctan2(-y, x)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        theta_sigma = pi / n_orient / d_theta_on_sigma
        wavelength = min_wave_length

        eo = np.zeros((rows, cols, 12), np.float)
        for n in range(n_scale):
            fo = 1. / wavelength
            result = np.exp(-np.power(np.log(radius / fo), 2) / (2 * np.power(np.log(sigma_on_f), 2)))
            result[rows // 2, cols // 2] = 0

            for o in range(n_orient):
                angle = o * pi / n_orient
                ds = sin_theta * np.cos(angle) - cos_theta * np.sin(angle)
                dc = cos_theta * np.cos(angle) - sin_theta * np.sin(angle)
                dtheta = np.abs(np.arctan2(ds, dc))
                spread = np.exp(-np.power(dtheta, 2) / (2 * np.power(theta_sigma, 2)))
                filters = np.fft.fftshift(result * spread)
                loggabout = np.abs(np.fft.ifft2(image_fft * filters))
                eo[:, :, n] = loggabout

            wavelength = wavelength * mult
        return eo
